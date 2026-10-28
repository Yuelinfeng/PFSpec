"""Ranker+MLP decoding with a small exact Feature candidate tree.

The Target tree runtime supplies logits for all rows at once and compacts only
one accepted path. This decoder owns proposal, rejection, recovery, and EOS
logic; those Python decisions remain outside the captured Target forward.
"""
from __future__ import annotations

import time
import math
import random
import json
from contextlib import nullcontext


def torch_acceptance_probability(residual, proposal, token):
    import torch

    if not torch.is_tensor(residual) or not torch.is_tensor(proposal):
        raise TypeError("residual and proposal must be PyTorch Tensors")
    token = int(token)
    if residual.ndim != 1 or proposal.shape != residual.shape:
        raise ValueError("residual and proposal must have the same one-dimensional shape")
    q_value = float(proposal[token].item())
    if not math.isfinite(q_value) or q_value <= 0.0:
        raise ValueError("sampled token has zero proposal probability")
    r_value = float(residual[token].item())
    if not math.isfinite(r_value) or r_value < 0.0:
        raise ValueError("residual probability is invalid")
    return min(1.0, r_value / q_value)


def torch_positive_residual(residual, proposal):
    import torch

    if not torch.is_tensor(residual) or not torch.is_tensor(proposal):
        raise TypeError("residual and proposal must be PyTorch Tensors")
    if residual.ndim != 1 or proposal.shape != residual.shape:
        raise ValueError("residual and proposal must have the same one-dimensional shape")
    raw = torch.clamp(residual.float() - proposal.float(), min=0.0)
    mass = float(raw.sum().item())
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("rejection has no positive target residual")
    return raw / mass


def _draw_categorical(probabilities, rng):
    value = rng.random()
    cumulative = 0.0
    last_positive = None
    for index, probability in enumerate(probabilities):
        if probability > 0.0:
            last_positive = index
        cumulative += probability
        if value < cumulative:
            return index
    if last_positive is None:
        raise ValueError("categorical distribution has no positive mass")
    return last_positive


def _enumerate_position_law(target, proposal):
    """Enumerate the output law of one sampled proposal and exact residual."""
    target = _normalize_sequence(target)
    proposal = _normalize_sequence(proposal)
    if len(target) != len(proposal):
        raise ValueError("target and proposal dimensions differ")
    residual_raw = [max(p - q, 0.0) for p, q in zip(target, proposal)]
    residual_mass = math.fsum(residual_raw)
    output = [0.0] * len(target)
    for candidate, q_value in enumerate(proposal):
        if q_value <= 0.0:
            continue
        accepted_mass = min(target[candidate], q_value)
        output[candidate] += accepted_mass
        rejected_mass = q_value - accepted_mass
        if rejected_mass > 0.0:
            if residual_mass <= 0.0:
                raise ValueError("rejection has no target residual mass")
            for token, raw in enumerate(residual_raw):
                output[token] += rejected_mass * raw / residual_mass
    return tuple(output)


def _normalize_sequence(values):
    values = tuple(float(value) for value in values)
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("distribution must be finite, nonnegative, and nonempty")
    mass = math.fsum(values)
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("distribution must have positive finite mass")
    return tuple(value / mass for value in values)


def _run_position_law_tests():
    cases = {
        "partial_proposal_support": ((0.45, 0.30, 0.20, 0.05), (0.0, 0.7, 0.3, 0.0)),
        "eos_recovered_from_residual": ((0.30, 0.25, 0.15, 0.30), (0.6, 0.4, 0.0, 0.0)),
        "singleton_point_proposal": ((0.20, 0.50, 0.20, 0.10), (0.0, 1.0, 0.0, 0.0)),
        "proposal_matches_target": ((0.20, 0.50, 0.20, 0.10), (0.20, 0.50, 0.20, 0.10)),
    }
    results = {}
    for name, (target, proposal) in cases.items():
        got = _enumerate_position_law(target, proposal)
        expected = _normalize_sequence(target)
        max_abs = max(abs(a - b) for a, b in zip(got, expected))
        if max_abs > 1e-12:
            raise AssertionError(f"{name}: output law differs from target by {max_abs}")
        results[name] = {"max_abs": max_abs, "output": list(got)}
    if cases["eos_recovered_from_residual"][0][-1] <= 0.0:
        raise AssertionError("EOS case must carry positive Target mass")
    return {"passed": True, "cases": results, "tolerance": 1e-12}


def _decode_impl(
    target,
    adapter,
    ranker,
    ranks,
    prompt,
    budget,
    seed,
    *,
    tree_runtime,
    width=1,
    cache_factory=None,
    position_mode="prob",
):
    """Decode one sequence and return ``(stats, generated_suffix)``.

    ``tree_runtime.verify`` evaluates ``[pendingF, P1, F_a, P2_a, ...]`` and
    leaves the logical cache position unchanged. ``commit`` keeps the chosen
    root-to-node path. A single accepted Feature branch uses its own Target P2
    row and pre-finalnorm hidden state.
    """
    import torch

    from draft_a.pf_feature_unified import EOS, F_VOCAB, P_VOCAB, _shared
    from bench_gaussiangpt_pf_block2_cow import (
        CaptureLastBlock,
        decide,
        fork_position_state,
        residual_point,
        truncate,
    )
    from gpt_position_spec_decode_smoke import choose_position
    from tree_sampling import torch_proposals_without_replacement

    shared = _shared()
    require = shared._require
    ratio = shared._ratio
    width = int(width)
    budget = int(budget)
    if budget < 2 or budget % 2:
        raise ValueError("budget must be a positive even number of at least two")
    if width != 1:
        raise ValueError("prob_position_decoder only supports the W1 tree")
    if width > int(getattr(tree_runtime, "max_width", width)):
        raise ValueError("width exceeds the tree runtime capacity")
    if width > F_VOCAB:
        raise ValueError("width exceeds the Feature vocabulary")
    if position_mode not in {"point", "prob"}:
        raise ValueError("position_mode must be 'point' or 'prob'")

    prompt = [int(token) for token in prompt]
    require(bool(prompt), "Prompt must contain BOS")
    shared.validate_sequence(prompt, [], ranks)
    require((len(prompt) - 1) % 2 == 0, "Prompt must end at a complete pair")
    require(EOS not in prompt[1:], "Prompt must not contain terminal EOS")

    device = next(target.parameters()).device
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))
    position_rng = random.Random(int(seed) ^ 0x5A17D3E1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    stats = {
        "target_calls": 0,
        "target_input_tokens": 0,
        "draft_calls": 0,
        "draft_input_tokens": 0,
        "ranker_calls": 0,
        "rounds": 0,
        "proposed": 0,
        "verified": 0,
        "accepted": 0,
        "accepted_lengths": [],
        "round_generated_lengths": [],
        "position_fallbacks": 0,
        "k1_steps": 0,
        "tree_verify_calls": 0,
        "candidate_tree_nodes": 0,
        "target_tree_nodes": 0,
        "feature_attempts": 0,
        "feature_branch_hits": 0,
        "feature_all_rejected": 0,
        "feature_branch_index_counts": {},
        "recovery_calls": 0,
    }
    for kind in shared.KINDS:
        for field in ("proposed", "verified", "accepted"):
            stats[kind + "_" + field] = 0

    def count(kind, field, amount=1):
        amount = int(amount)
        stats[field] += amount
        stats[kind + "_" + field] += amount

    def policy(logits):
        return shared.policy(logits)

    def draw(logits):
        return int(shared.draw(policy(logits), rng))

    def propose_position(state, previous):
        if position_mode == "point":
            return choose_position(ranker, state, previous), None
        proposed = state.propose(previous)
        if proposed is None:
            return None, None
        ids, features, _histogram_guess, invalid = proposed
        ids = [int(token) for token in ids]
        require(bool(ids), "Position candidate set cannot be empty")
        require(len(ids) == len(set(ids)), "Position candidate ids must be unique")
        for token in ids:
            require(0 <= token < P_VOCAB, "Ranker returned a non-Position candidate")
            if previous is not None:
                require(int(ranks[token]) > int(ranks[previous]),
                        "Ranker returned a non-increasing Position")
        if len(ids) == 1:
            probabilities = [1.0]
            selected = 0
        else:
            scores_array = ranker.predict_proba(features)[:, 1]
            scores = [float(value) for value in scores_array]
            require(len(scores) == len(ids), "Ranker score count differs from candidate ids")
            require(all(math.isfinite(value) and value >= 0.0 for value in scores),
                    "Ranker positive-class scores must be finite and nonnegative")
            mass = math.fsum(scores)
            require(mass > 0.0, "Ranker positive-class scores have no mass")
            probabilities = [value / mass for value in scores]
            selected = _draw_categorical(probabilities, position_rng)
        proposal = {
            "ids": ids,
            "probabilities": probabilities,
            "selected_probability": probabilities[selected],
        }
        return (ids[selected], len(ids), invalid), proposal

    def accept_position(target_distribution, candidate, proposal):
        if position_mode == "point":
            return bool(decide(float(target_distribution[candidate].item()), rng, device))
        target_probability = float(target_distribution[candidate].item())
        proposal_probability = float(proposal["selected_probability"])
        alpha = min(1.0, target_probability / proposal_probability)
        return bool(decide(alpha, rng, device))

    def sample_position_residual(target_distribution, proposal):
        ids = torch.tensor(proposal["ids"], dtype=torch.long, device=target_distribution.device)
        q = torch.tensor(proposal["probabilities"], dtype=torch.float32,
                         device=target_distribution.device)
        residual = target_distribution.float().clone()
        residual[ids] = (residual[ids] - q).clamp_min(0.0)
        mass = float(residual.sum().item())
        require(math.isfinite(mass) and mass > 0.0,
                "Rejected Position proposal has no positive Target residual")
        return int(shared.draw(residual / mass, rng))

    def finish_round(begin, accepted):
        stats["accepted_lengths"].append(int(accepted))
        stats["round_generated_lengths"].append(len(prefix) - begin)

    def validate_position(previous, token):
        require(0 <= int(token) < P_VOCAB, "Target emitted an invalid Position id")
        if previous is not None:
            require(int(ranks[token]) > int(ranks[previous]),
                    "Target emitted a non-increasing Position")

    cache = None
    capture = None
    prefix = list(prompt)
    adapter._profile = None
    capacity = len(prompt) + budget + 6
    make_cache = cache_factory or shared.cache_for
    try:
        cache = make_cache(target, capacity)
        capture = CaptureLastBlock(target)
        adapter.prepare(prompt, capacity)

        def target_forward(tokens, *, recovery=False):
            tokens = [int(token) for token in tokens]
            if not tokens:
                raise ValueError("Target forward requires at least one token")
            start = int(cache.get_pos())
            kinds = shared.typed_kinds(start, len(tokens))
            logits, states = capture.run(target, cache, tokens, kinds, device)
            if int(cache.get_pos()) != start + len(tokens):
                raise RuntimeError("ordinary Target call advanced the wrong number of tokens")
            stats["target_calls"] += 1
            stats["target_input_tokens"] += len(tokens)
            if recovery:
                stats["recovery_calls"] += 1
            return logits, states

        def verify_tree(pending_f, p1, features, p2s):
            before = int(cache.get_pos())
            logits, states = tree_runtime.verify(cache, pending_f, p1, features, p2s)
            if int(cache.get_pos()) != before:
                raise RuntimeError("tree verify advanced the logical cache position")
            expected_nodes = 2 + 2 * len(features)
            if logits.ndim == 3:
                logits = logits[0]
            if states.ndim == 3:
                states = states[0]
            if logits.ndim != 2 or states.ndim != 2:
                raise RuntimeError("tree runtime must return rank-2 logits and hidden states")
            if logits.size(0) != expected_nodes or states.size(0) != expected_nodes:
                raise RuntimeError("tree runtime returned a different node count")
            stats["target_calls"] += 1
            stats["target_input_tokens"] += expected_nodes
            stats["tree_verify_calls"] += 1
            stats["candidate_tree_nodes"] += len(features)
            stats["target_tree_nodes"] += expected_nodes
            return logits, states

        def commit_path(before, path):
            after = tree_runtime.commit(cache, path)
            expected = int(before) + len(path)
            if int(after) != expected or int(cache.get_pos()) != expected:
                raise RuntimeError("tree commit left the Target cache at an unexpected position")
            return after

        # A complete prompt leaves its final Feature pending. The Target cache
        # and h_prev stop at the preceding Position, matching the MLP adapter.
        online = None
        h_prev = None
        if len(prompt) > 1:
            _, states = target_forward(prompt[:-1])
            h_prev = states[-1].detach().clone()
            online = shared.position_state(prefix, ranks)

        # BOS-only prompts use the same Target P/F bootstrap as the reference.
        while len(prefix) - len(prompt) < budget:
            if online is None:
                logits, _ = target_forward(prefix[-1:])
                p0 = draw(logits[-1])
                prefix.append(p0)
                if p0 == EOS:
                    break
                validate_position(None, p0)
                logits, states = target_forward([p0])
                f0 = draw(logits[-1])
                require(P_VOCAB <= f0 < EOS, "Target emitted an invalid Feature id")
                prefix.append(f0)
                h_prev = states[-1].detach().clone()
                online = shared.position_state(prefix, ranks)
                continue

            if int(cache.get_pos()) != len(prefix) - 1:
                raise RuntimeError("Target cache does not end immediately before pending Feature")
            stats["rounds"] += 1
            begin = len(prefix)
            before = int(cache.get_pos())
            previous, pending_f = prefix[-2], prefix[-1]
            remaining = budget - (len(prefix) - len(prompt))

            stats["ranker_calls"] += 1
            first, q1 = propose_position(online, previous)
            second = shadow = None
            can_tree = first is not None and remaining >= 4
            if can_tree:
                shadow = fork_position_state(online)
                shadow.finalize(first[0])
                stats["ranker_calls"] += 1
                second, q2 = propose_position(shadow, first[0])
                can_tree = second is not None
            else:
                q2 = None

            if not can_tree:
                stats["k1_steps"] += 1
                if first is None:
                    stats["position_fallbacks"] += 1
                    logits, _ = target_forward([pending_f])
                    actual_p1 = draw(logits[-1])
                    prefix.append(actual_p1)
                    if actual_p1 == EOS:
                        online.drop_eos()
                        finish_round(begin, 0)
                        break
                    validate_position(previous, actual_p1)
                    logits, states = target_forward([actual_p1])
                    actual_f1 = draw(logits[-1])
                    require(P_VOCAB <= actual_f1 < EOS,
                            "Target emitted an invalid Feature id")
                    prefix.append(actual_f1)
                    h_prev = states[-1].detach().clone()
                    online.finalize(actual_p1)
                    finish_round(begin, 0)
                    continue

                p1 = int(first[0])
                count("position", "proposed")
                logits, states = target_forward([pending_f, p1])
                p1_distribution = policy(logits[0])
                count("position", "verified")
                if accept_position(p1_distribution, p1, q1):
                    count("position", "accepted")
                    actual_p1 = p1
                    h_prev = states[1].detach().clone()
                    actual_f1 = draw(logits[1])
                    accepted = 1
                else:
                    actual_p1 = (int(residual_point(p1_distribution, p1, rng))
                                 if position_mode == "point"
                                 else sample_position_residual(p1_distribution, q1))
                    truncate(cache, before + 1, before + 2)
                    if int(cache.get_pos()) != before + 1:
                        raise RuntimeError("K1 Position recovery left a bad cache length")
                    if actual_p1 == EOS:
                        prefix.append(actual_p1)
                        online.drop_eos()
                        finish_round(begin, 0)
                        break
                    validate_position(previous, actual_p1)
                    logits, states = target_forward([actual_p1], recovery=True)
                    actual_f1 = draw(logits[-1])
                    h_prev = states[-1].detach().clone()
                    accepted = 0
                require(P_VOCAB <= actual_f1 < EOS,
                        "Target emitted an invalid Feature id")
                prefix.extend([actual_p1, actual_f1])
                online.finalize(actual_p1)
                finish_round(begin, accepted)
                continue

            p1 = int(first[0])
            p2 = int(second[0])
            count("position", "proposed", 2)
            count("feature", "proposed", width)
            q = adapter.propose(prefix, p1, h_prev, rng)
            require(torch.is_tensor(q) and q.dtype == torch.float32 and q.ndim == 1
                    and q.numel() == F_VOCAB,
                    "Feature proposal must be a full FP32 distribution of size 4096")
            q_mass = float(q.sum().item())
            require(abs(q_mass - 1.0) <= 1e-4,
                    "Feature proposal must be normalized after temperature/top-p")
            candidate_ids, candidate_q = torch_proposals_without_replacement(q, width, rng)
            features = [int(token) + P_VOCAB for token in candidate_ids]
            p2s = [p2] * width

            logits, states = verify_tree(pending_f, p1, features, p2s)
            if int(cache.get_pos()) != before:
                raise RuntimeError("tree verification moved the cache before acceptance")

            count("position", "verified")
            p1_distribution = policy(logits[0])
            if not accept_position(p1_distribution, p1, q1):
                actual_p1 = (int(residual_point(p1_distribution, p1, rng))
                             if position_mode == "point"
                             else sample_position_residual(p1_distribution, q1))
                commit_path(before, [0])
                prefix.append(actual_p1)
                if actual_p1 == EOS:
                    online.drop_eos()
                    finish_round(begin, 0)
                    break
                validate_position(previous, actual_p1)
                online.finalize(actual_p1)
                logits_recovery, states_recovery = target_forward([actual_p1], recovery=True)
                actual_f1 = draw(logits_recovery[-1])
                require(P_VOCAB <= actual_f1 < EOS,
                        "Target emitted an invalid Feature id")
                prefix.append(actual_f1)
                h_prev = states_recovery[-1].detach().clone()
                finish_round(begin, 0)
                continue

            count("position", "accepted")
            online = shadow
            h_p1 = states[1].detach().clone()
            target_feature = policy(logits[1])[P_VOCAB:EOS].float().clone()
            residual = target_feature
            selected_branch = None
            for branch, (candidate_local, proposal_q) in enumerate(
                zip(candidate_ids, candidate_q)
            ):
                candidate_local = int(candidate_local)
                stats["feature_attempts"] += 1
                count("feature", "verified")
                alpha = torch_acceptance_probability(residual, proposal_q,
                                                     candidate_local)
                if bool(decide(alpha, rng, device)):
                    selected_branch = branch
                    count("feature", "accepted")
                    stats["feature_branch_hits"] += 1
                    branch_counts = stats["feature_branch_index_counts"]
                    key = str(branch)
                    branch_counts[key] = int(branch_counts.get(key, 0)) + 1
                    break
                residual = torch_positive_residual(residual, proposal_q)

            if selected_branch is None:
                actual_f1 = int(shared.draw(residual, rng)) + P_VOCAB
                require(P_VOCAB <= actual_f1 < EOS,
                        "Feature residual emitted an invalid Feature id")
                commit_path(before, [0, 1])
                prefix.extend([p1, actual_f1])
                online.drop_eos()
                h_prev = h_p1
                stats["feature_all_rejected"] += 1
                finish_round(begin, 1)
                continue

            branch = int(selected_branch)
            feature_node = 2 + 2 * branch
            p2_node = feature_node + 1
            actual_f1 = features[branch]
            prefix.extend([p1, actual_f1])
            count("position", "verified")
            p2_distribution = policy(logits[feature_node])
            if accept_position(p2_distribution, p2, q2):
                count("position", "accepted")
                h_prev = states[p2_node].detach().clone()
                bonus_f2 = draw(logits[p2_node])
                require(P_VOCAB <= bonus_f2 < EOS,
                        "Target bonus emitted an invalid Feature id")
                commit_path(before, [0, 1, feature_node, p2_node])
                prefix.extend([p2, bonus_f2])
                online.finalize(p2)
                finish_round(begin, 3)
                continue

            actual_p2 = (int(residual_point(p2_distribution, p2, rng))
                         if position_mode == "point"
                         else sample_position_residual(p2_distribution, q2))
            commit_path(before, [0, 1, feature_node])
            prefix.append(actual_p2)
            if actual_p2 == EOS:
                online.drop_eos()
                finish_round(begin, 2)
                break
            validate_position(p1, actual_p2)
            online.finalize(actual_p2)
            logits_recovery, states_recovery = target_forward([actual_p2], recovery=True)
            actual_f2 = draw(logits_recovery[-1])
            require(P_VOCAB <= actual_f2 < EOS,
                    "Target recovery emitted an invalid Feature id")
            prefix.append(actual_f2)
            h_prev = states_recovery[-1].detach().clone()
            finish_round(begin, 2)

    finally:
        stats["draft_calls"] = int(getattr(adapter, "forward_calls", 0))
        stats["draft_input_tokens"] = int(getattr(adapter, "input_tokens", 0))
        try:
            adapter.close()
        finally:
            if capture is not None:
                capture.close()
            cache = None

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    suffix = prefix[len(prompt):]
    shared.validate_sequence(prompt, suffix, ranks)
    require(len(suffix) <= budget, "Output exceeds the requested token budget")
    require(len(stats["accepted_lengths"]) == stats["rounds"]
            == len(stats["round_generated_lengths"]), "Round totals differ")
    for field in ("proposed", "verified", "accepted"):
        require(stats[field] == sum(stats[kind + "_" + field]
                                    for kind in shared.KINDS),
                "Typed counters differ")

    stats.update(
        generated_tokens=len(suffix),
        seconds=elapsed,
        tokens_per_second=ratio(len(suffix), elapsed),
        eos=bool(suffix and suffix[-1] == EOS),
        k=2,
        proposal_width=2 + width,
        tree_width=width,
        seed=int(seed),
        prompt_tokens=len(prompt) - 1,
        requested_tokens=budget,
        temperature=0.9,
        target_top_p=0.9,
        draft_top_p=0.9,
        variant=f"ranker_mlp_feature_tree_w{width}",
        mean_accepted_length=ratio(sum(stats["accepted_lengths"]), stats["rounds"]),
        mean_generated_per_round=ratio(sum(stats["round_generated_lengths"]),
                                       stats["rounds"]),
    )
    if position_mode == "prob":
        stats["position_proposal_mode"] = "normalized_ranker_positive_scores"
        stats["variant"] = f"ranker_prob_position_feature_tree_w{width}"
    if hasattr(tree_runtime, "stats"):
        stats["tree_runtime_stats"] = tree_runtime.stats()
    return shared._add_rates(stats), suffix


def decode(
    target,
    adapter,
    ranker,
    ranks,
    prompt,
    budget,
    seed,
    *,
    tree_runtime,
    width=1,
    cache_factory=None,
    position_mode="prob",
):
    """Run the decoder in inference mode with autocast disabled."""
    import torch

    device = next(target.parameters()).device
    autocast = (torch.autocast(device_type=device.type, enabled=False)
                if device.type == "cuda" else nullcontext())
    with torch.inference_mode(), autocast:
        return _decode_impl(
            target, adapter, ranker, ranks, prompt, budget, seed,
            tree_runtime=tree_runtime,
            width=width,
            cache_factory=cache_factory,
            position_mode=position_mode,
        )


def self_test():
    return _run_position_law_tests()


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2, sort_keys=True))
