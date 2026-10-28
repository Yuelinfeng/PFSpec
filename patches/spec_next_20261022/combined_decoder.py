"""Combined W1 decoding with a Position backup and prior Feature policy.

This decoder combines the five-node Position-rescue path with the same
previous-Target-Feature mixture used by ``feature_reuse_fast_decoder``. The
Target runtime compacts only the chosen path; acceptance, recovery, and EOS
decisions stay in Python.
"""
from __future__ import annotations

import math
import json
import time
from contextlib import nullcontext

try:
    from .prob_position_decoder import (
        _enumerate_position_law,
        _normalize_sequence,
        torch_acceptance_probability,
        torch_positive_residual,
    )
    from .position_rescue_decoder import _run_backup_law_tests
except ImportError:
    from prob_position_decoder import (
        _enumerate_position_law,
        _normalize_sequence,
        torch_acceptance_probability,
        torch_positive_residual,
    )
    from position_rescue_decoder import _run_backup_law_tests


def _best_two_indices(scores, *, include_backup=True):
    scores = [float(value) for value in scores]
    if not scores or any(not math.isfinite(value) for value in scores):
        raise ValueError("Ranker scores must be a nonempty finite sequence")
    best = 0
    for index in range(1, len(scores)):
        if scores[index] > scores[best]:
            best = index
    if not include_backup or len(scores) == 1:
        return best, None
    backup = None
    for index in range(len(scores)):
        if index != best and (backup is None or scores[index] > scores[backup]):
            backup = index
    return best, backup


def _backup_hit_probabilities(target, point, backup):
    target = _normalize_sequence(target)
    point = int(point)
    backup = int(backup)
    if len(target) < 2 or not 0 <= point < len(target) or not 0 <= backup < len(target):
        raise ValueError("point and backup must index the target distribution")
    if point == backup:
        raise ValueError("backup must differ from the point proposal")
    reject_probability = 1.0 - target[point]
    conditional_hit = target[backup] / reject_probability if reject_probability > 0.0 else 0.0
    return reject_probability * conditional_hit, conditional_hit


def _run_backup_law_tests():
    scores = (0.9, 0.9, 0.4)
    if _best_two_indices(scores) != (0, 1):
        raise AssertionError("top-score ties must preserve the original first-index choice")
    if _best_two_indices((0.3,)) != (0, None):
        raise AssertionError("singleton proposal must have no backup")
    target = (0.4, 0.2, 0.1, 0.3)  # final id represents EOS
    point, backup = 0, 1
    output = _enumerate_position_law(target, (1.0, 0.0, 0.0, 0.0))
    if max(abs(left - right) for left, right in zip(output, target)) > 1e-12:
        raise AssertionError("point correction changed the Target output law")
    unconditional, conditional = _backup_hit_probabilities(target, point, backup)
    if abs(unconditional - target[backup]) > 1e-12:
        raise AssertionError("backup hit probability must equal p(backup)")
    if abs(conditional - target[backup] / (1.0 - target[point])) > 1e-12:
        raise AssertionError("conditional backup hit probability is incorrect")
    return {
        "passed": True,
        "point_residual_law_max_abs": max(abs(left - right) for left, right in zip(output, target)),
        "backup_hit_unconditional": unconditional,
        "backup_hit_given_rejection": conditional,
        "eos_mass": target[-1],
    }


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
    rescue_runtime,
    width=1,
    cache_factory=None,
    reuse_weight=0.25,
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

    shared = _shared()
    require = shared._require
    ratio = shared._ratio
    width = int(width)
    budget = int(budget)
    reuse_weight = float(reuse_weight)
    if not math.isfinite(reuse_weight) or not 0.0 <= reuse_weight <= 1.0:
        raise ValueError("reuse_weight must be finite and in [0, 1]")
    if budget < 2 or budget % 2:
        raise ValueError("budget must be a positive even number of at least two")
    if width != 1:
        raise ValueError("position_rescue_decoder only supports the W1 tree")
    if width > int(getattr(tree_runtime, "max_width", width)):
        raise ValueError("width exceeds the tree runtime capacity")
    if width > F_VOCAB:
        raise ValueError("width exceeds the Feature vocabulary")

    prompt = [int(token) for token in prompt]
    require(bool(prompt), "Prompt must contain BOS")
    shared.validate_sequence(prompt, [], ranks)
    require((len(prompt) - 1) % 2 == 0, "Prompt must end at a complete pair")
    require(EOS not in prompt[1:], "Prompt must not contain terminal EOS")

    device = next(target.parameters()).device
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))
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
        "position_backup_nodes": 0,
        "position_backup_rejections": 0,
        "position_rescue_hits": 0,
        "feature_reuse_policy_updates": 0,
        "feature_reuse_mixed_proposals": 0,
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

    last_target_feature_policy = None

    def remember_feature_policy(feature_policy):
        nonlocal last_target_feature_policy
        if (feature_policy.ndim != 1 or feature_policy.numel() != F_VOCAB
                or feature_policy.dtype != torch.float32):
            raise RuntimeError("cached Feature policy must be a 4096-value FP32 vector")
        last_target_feature_policy = feature_policy
        stats["feature_reuse_policy_updates"] += 1

    def draw_target_feature(logits):
        if reuse_weight <= 0.0:
            return draw(logits)
        full_policy = policy(logits)
        remember_feature_policy(full_policy[P_VOCAB:EOS])
        return int(shared.draw(full_policy, rng))

    def propose_top_two(state, previous, *, include_backup=True):
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
            return (ids[0], 1, invalid), None
        score_array = ranker.predict_proba(features)[:, 1]
        scores = [float(value) for value in score_array]
        require(len(scores) == len(ids), "Ranker score count differs from candidate ids")
        require(all(math.isfinite(value) and value >= 0.0 for value in scores),
                "Ranker positive-class scores must be finite and nonnegative")
        best, backup = _best_two_indices(scores, include_backup=include_backup)
        return (ids[best], len(ids), invalid), (
            ids[backup] if backup is not None else None
        )

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

        def verify_tree(pending_f, p1, features, p2s, backup_position=None):
            before = int(cache.get_pos())
            active_tree = tree_runtime
            if backup_position is not None:
                if rescue_runtime is None:
                    raise ValueError("rescue_runtime is required when a backup Position is available")
                active_tree = rescue_runtime
                logits, states = active_tree.verify(
                    cache, pending_f, p1, features, p2s,
                    backup_position=int(backup_position),
                )
            else:
                logits, states = active_tree.verify(cache, pending_f, p1, features, p2s)
            if int(cache.get_pos()) != before:
                raise RuntimeError("tree verify advanced the logical cache position")
            expected_nodes = 2 + 2 * len(features) + int(backup_position is not None)
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
            if backup_position is not None:
                stats["position_backup_nodes"] += 1
            return logits, states, active_tree

        def commit_path(before, path, active_tree):
            after = active_tree.commit(cache, path)
            expected = int(before) + len(path)
            if int(after) != expected or int(cache.get_pos()) != expected:
                raise RuntimeError("tree commit left the Target cache at an unexpected position")
            return after

        # A complete prompt leaves its final Feature pending. The Target cache
        # and h_prev stop at the preceding Position, matching the MLP adapter.
        online = None
        h_prev = None
        if len(prompt) > 1:
            logits, states = target_forward(prompt[:-1])
            h_prev = states[-1].detach().clone()
            if reuse_weight > 0.0:
                full_policy = policy(logits[-1])
                remember_feature_policy(full_policy[P_VOCAB:EOS])
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
                f0 = draw_target_feature(logits[-1])
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
            first, backup_position = propose_top_two(
                online, previous, include_backup=(remaining >= 4)
            )
            second = shadow = None
            can_tree = first is not None and remaining >= 4
            if can_tree:
                shadow = fork_position_state(online)
                shadow.finalize(first[0])
                stats["ranker_calls"] += 1
                second, _unused_p2_backup = propose_top_two(
                    shadow, first[0], include_backup=False
                )
                can_tree = second is not None
            if not can_tree:
                backup_position = None

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
                    actual_f1 = draw_target_feature(logits[-1])
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
                if bool(decide(float(p1_distribution[p1].item()), rng, device)):
                    count("position", "accepted")
                    actual_p1 = p1
                    h_prev = states[1].detach().clone()
                    actual_f1 = draw_target_feature(logits[1])
                    accepted = 1
                else:
                    actual_p1 = int(residual_point(p1_distribution, p1, rng))
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
                    actual_f1 = draw_target_feature(logits[-1])
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
            if reuse_weight > 0.0:
                if last_target_feature_policy is None:
                    raise RuntimeError("no previous Target Feature policy is available")
                q = torch.lerp(q, last_target_feature_policy, reuse_weight)
                stats["feature_reuse_mixed_proposals"] += 1
            q_mass = float(q.sum().item())
            require(abs(q_mass - 1.0) <= 1e-4,
                    "Feature proposal must be normalized after temperature/top-p")
            # Width one uses exactly the same categorical draw as the W1 path.
            candidate_local = int(torch.multinomial(q, 1, generator=rng).item())
            candidate_ids, candidate_q = [candidate_local], [q]
            features = [int(token) + P_VOCAB for token in candidate_ids]
            p2s = [p2] * width

            logits, states, active_tree = verify_tree(
                pending_f, p1, features, p2s,
                backup_position=backup_position,
            )
            if int(cache.get_pos()) != before:
                raise RuntimeError("tree verification moved the cache before acceptance")

            count("position", "verified")
            p1_distribution = policy(logits[0])
            if not bool(decide(float(p1_distribution[p1].item()), rng, device)):
                actual_p1 = int(residual_point(p1_distribution, p1, rng))
                if backup_position is not None:
                    stats["position_backup_rejections"] += 1
                if backup_position is not None and actual_p1 == int(backup_position):
                    stats["position_rescue_hits"] += 1
                    commit_path(before, [0, 4], active_tree)
                    validate_position(previous, actual_p1)
                    online.finalize(actual_p1)
                    actual_f1 = draw_target_feature(logits[4])
                    require(P_VOCAB <= actual_f1 < EOS,
                            "Target backup emitted an invalid Feature id")
                    prefix.extend([actual_p1, actual_f1])
                    h_prev = states[4].detach().clone()
                    finish_round(begin, 0)
                    continue
                commit_path(before, [0], active_tree)
                prefix.append(actual_p1)
                if actual_p1 == EOS:
                    online.drop_eos()
                    finish_round(begin, 0)
                    break
                validate_position(previous, actual_p1)
                online.finalize(actual_p1)
                logits_recovery, states_recovery = target_forward([actual_p1], recovery=True)
                actual_f1 = draw_target_feature(logits_recovery[-1])
                require(P_VOCAB <= actual_f1 < EOS,
                        "Target emitted an invalid Feature id")
                prefix.append(actual_f1)
                h_prev = states_recovery[-1].detach().clone()
                finish_round(begin, 0)
                continue

            count("position", "accepted")
            online = shadow
            h_p1 = states[1].detach().clone()
            full_feature_policy = policy(logits[1])
            target_feature = full_feature_policy[P_VOCAB:EOS].float().clone()
            if reuse_weight > 0.0:
                remember_feature_policy(target_feature)
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
                commit_path(before, [0, 1], active_tree)
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
            if bool(decide(float(p2_distribution[p2].item()), rng, device)):
                count("position", "accepted")
                h_prev = states[p2_node].detach().clone()
                bonus_f2 = draw_target_feature(logits[p2_node])
                require(P_VOCAB <= bonus_f2 < EOS,
                        "Target bonus emitted an invalid Feature id")
                commit_path(before, [0, 1, feature_node, p2_node], active_tree)
                prefix.extend([p2, bonus_f2])
                online.finalize(p2)
                finish_round(begin, 3)
                continue

            actual_p2 = int(residual_point(p2_distribution, p2, rng))
            commit_path(before, [0, 1, feature_node], active_tree)
            prefix.append(actual_p2)
            if actual_p2 == EOS:
                online.drop_eos()
                finish_round(begin, 2)
                break
            validate_position(p1, actual_p2)
            online.finalize(actual_p2)
            logits_recovery, states_recovery = target_forward([actual_p2], recovery=True)
            actual_f2 = draw_target_feature(logits_recovery[-1])
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
        feature_reuse_weight=reuse_weight,
        feature_proposal_mix=("lerp(MLP_q,previous_target_pF,w)_no_post_top_p"
                              if reuse_weight > 0.0 else "MLP_q"),
        variant=(f"ranker_mlp_feature_tree_w{width}_position_backup_"
                 f"reuse{reuse_weight:g}_combined"),
        mean_accepted_length=ratio(sum(stats["accepted_lengths"]), stats["rounds"]),
        mean_generated_per_round=ratio(sum(stats["round_generated_lengths"]),
                                       stats["rounds"]),
    )
    if hasattr(tree_runtime, "stats"):
        stats["tree_runtime_stats"] = tree_runtime.stats()
    if rescue_runtime is not None and hasattr(rescue_runtime, "stats"):
        stats["rescue_runtime_stats"] = rescue_runtime.stats()
    stats["position_rescue_rate_per_backup_node"] = (
        stats["position_rescue_hits"] / stats["position_backup_nodes"]
        if stats["position_backup_nodes"] else None
    )
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
    rescue_runtime,
    width=1,
    cache_factory=None,
    reuse_weight=0.25,
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
            rescue_runtime=rescue_runtime,
            width=width,
            cache_factory=cache_factory,
            reuse_weight=reuse_weight,
        )


def _run_mixture_law_tests():
    target = (0.39, 0.28, 0.21, 0.12)  # final id represents EOS
    mlp_q = (0.0, 0.75, 0.25, 0.0)
    previous_target_feature = (0.35, 0.0, 0.15, 0.50)
    cases = {}
    for weight in (0.0, 0.25, 1.0):
        mixed = _normalize_sequence(
            (1.0 - weight) * q + weight * p
            for q, p in zip(mlp_q, previous_target_feature)
        )
        observed = _enumerate_position_law(target, mixed)
        max_abs = max(abs(left - right) for left, right in zip(observed, target))
        if max_abs > 1e-12:
            raise AssertionError(f"mixture weight {weight}: output law differs by {max_abs}")
        cases[str(weight)] = {
            "proposal": list(mixed),
            "output": list(observed),
            "max_abs": max_abs,
        }
    if target[-1] <= 0.0:
        raise AssertionError("Feature-law test must include EOS mass")
    return {"passed": True, "cases": cases, "tolerance": 1e-12}


def self_test():
    backup = _run_backup_law_tests()
    mixture = _run_mixture_law_tests()
    return {"passed": bool(backup["passed"] and mixture["passed"]),
            "backup": backup, "mixture": mixture}


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2, sort_keys=True))
