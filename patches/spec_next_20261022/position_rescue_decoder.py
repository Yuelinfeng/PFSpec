"""Ranker+MLP decoding with a small exact Feature candidate tree.

The Target tree runtime supplies logits for all rows at once and compacts only
one accepted path. This decoder owns proposal, rejection, recovery, and EOS
logic; those Python decisions remain outside the captured Target forward.
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
except ImportError:
    from prob_position_decoder import (
        _enumerate_position_law,
        _normalize_sequence,
        torch_acceptance_probability,
        torch_positive_residual,
    )


def _ranked_candidate_indices(scores, backup_count):
    scores = [float(value) for value in scores]
    if not scores or any(not math.isfinite(value) for value in scores):
        raise ValueError("Ranker scores must be a nonempty finite sequence")
    backup_count = int(backup_count)
    if not 0 <= backup_count <= 4:
        raise ValueError("backup_count must be between zero and four")
    ranked = sorted(range(len(scores)), key=lambda index: -scores[index])
    return ranked[0], ranked[1:1 + backup_count]


def _best_two_indices(scores, *, include_backup=True):
    best, backups = _ranked_candidate_indices(scores, int(include_backup))
    return best, backups[0] if backups else None


def _validate_backup_count(value):
    try:
        backup_count = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("backup_count must be an integer from one to four") from error
    if isinstance(value, bool) or backup_count != value or not 1 <= backup_count <= 4:
        raise ValueError("backup_count must be an integer from one to four")
    return backup_count


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
    if _ranked_candidate_indices((0.9, 0.9, 0.4), 4) != (0, [1, 2]):
        raise AssertionError("backup count must shrink to the available candidates")
    if _ranked_candidate_indices((0.2, 0.9, 0.9, 0.9), 3) != (1, [2, 3, 0]):
        raise AssertionError("equal scores must retain their original candidate order")
    if _ranked_candidate_indices((0.3,), 4) != (0, []):
        raise AssertionError("a singleton candidate set must remain without backups")
    for invalid_count in (0, 5, 1.5, True):
        try:
            _validate_backup_count(invalid_count)
        except ValueError:
            pass
        else:
            raise AssertionError("an unsupported backup count was accepted")
    target = (0.4, 0.2, 0.1, 0.3)  # final id represents EOS
    point, backups = 0, (1, 2)
    output = _enumerate_position_law(target, (1.0, 0.0, 0.0, 0.0))
    if max(abs(left - right) for left, right in zip(output, target)) > 1e-12:
        raise AssertionError("point correction changed the Target output law")
    cumulative = 0.0
    conditional_cumulative = []
    for backup in backups:
        unconditional, conditional = _backup_hit_probabilities(target, point, backup)
        if abs(unconditional - target[backup]) > 1e-12:
            raise AssertionError("backup hit probability must equal p(backup)")
        if abs(conditional - target[backup] / (1.0 - target[point])) > 1e-12:
            raise AssertionError("conditional backup hit probability is incorrect")
        cumulative += conditional
        conditional_cumulative.append(cumulative)
    expected_cumulative = (1.0 / 3.0, 0.5)
    if max(abs(left - right) for left, right in zip(
            conditional_cumulative, expected_cumulative)) > 1e-12:
        raise AssertionError("conditional cumulative backup mass is incorrect")
    return {
        "passed": True,
        "point_residual_law_max_abs": max(abs(left - right) for left, right in zip(output, target)),
        "backup_hit_unconditional": target[backups[0]],
        "backup_hit_given_rejection": target[backups[0]] / (1.0 - target[point]),
        "backup_hit_unconditional_by_rank": [target[index] for index in backups],
        "backup_hit_given_rejection_by_rank": [
            target[index] / (1.0 - target[point]) for index in backups
        ],
        "backup_conditional_cumulative_mass": list(conditional_cumulative),
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
    backup_count=1,
    backup_diagnostics=False,
    cache_factory=None,
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
    backup_count = _validate_backup_count(backup_count)
    backup_diagnostics = bool(backup_diagnostics)
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
        "position_backup_applicable_rounds": 0,
        "position_backup_extra_nodes": 0,
        "position_backup_applicable_rounds_by_rank": {
            str(rank): 0 for rank in range(1, backup_count + 1)
        },
        "position_backup_extra_nodes_by_rank": {
            str(rank): 0 for rank in range(1, backup_count + 1)
        },
        "position_rescue_hits_by_rank": {
            str(rank): 0 for rank in range(1, backup_count + 1)
        },
    }
    backup_diagnostic_rows = [] if backup_diagnostics else None
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

    def propose_positions(state, previous, *, requested_backups):
        proposed = state.propose(previous)
        if proposed is None:
            return None, []
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
            return (ids[0], 1, invalid), []
        score_array = ranker.predict_proba(features)[:, 1]
        scores = [float(value) for value in score_array]
        require(len(scores) == len(ids), "Ranker score count differs from candidate ids")
        require(all(math.isfinite(value) and value >= 0.0 for value in scores),
                "Ranker positive-class scores must be finite and nonnegative")
        best, backups = _ranked_candidate_indices(scores, requested_backups)
        return (ids[best], len(ids), invalid), [ids[index] for index in backups]

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
    capacity = len(prompt) + budget + 8
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

        def verify_tree(pending_f, p1, features, p2s, backup_positions=()):
            before = int(cache.get_pos())
            active_tree = tree_runtime
            backup_positions = tuple(int(token) for token in backup_positions)
            if backup_positions:
                if rescue_runtime is None:
                    raise ValueError("rescue_runtime is required when a backup Position is available")
                active_tree = rescue_runtime
                if backup_count == 1:
                    logits, states = active_tree.verify(
                        cache, pending_f, p1, features, p2s,
                        backup_position=backup_positions[0],
                    )
                else:
                    logits, states = active_tree.verify(
                        cache, pending_f, p1, features, p2s,
                        backup_positions=list(backup_positions),
                    )
            else:
                logits, states = active_tree.verify(cache, pending_f, p1, features, p2s)
            if int(cache.get_pos()) != before:
                raise RuntimeError("tree verify advanced the logical cache position")
            expected_nodes = 2 + 2 * len(features) + len(backup_positions)
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
            if backup_positions:
                stats["position_backup_applicable_rounds"] += 1
                stats["position_backup_nodes"] += len(backup_positions)
                stats["position_backup_extra_nodes"] += len(backup_positions)
                for rank in range(1, len(backup_positions) + 1):
                    key = str(rank)
                    applicable = stats["position_backup_applicable_rounds_by_rank"]
                    nodes = stats["position_backup_extra_nodes_by_rank"]
                    applicable[key] = int(applicable.get(key, 0)) + 1
                    nodes[key] = int(nodes.get(key, 0)) + 1
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
            first, backup_positions = propose_positions(
                online, previous,
                requested_backups=(backup_count if remaining >= 4 else 0),
            )
            second = shadow = None
            can_tree = first is not None and remaining >= 4
            if can_tree:
                shadow = fork_position_state(online)
                shadow.finalize(first[0])
                stats["ranker_calls"] += 1
                second, _unused_p2_backups = propose_positions(
                    shadow, first[0], requested_backups=0
                )
                can_tree = second is not None
            if not can_tree:
                backup_positions = []

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
                if bool(decide(float(p1_distribution[p1].item()), rng, device)):
                    count("position", "accepted")
                    actual_p1 = p1
                    h_prev = states[1].detach().clone()
                    actual_f1 = draw(logits[1])
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
            # Width one uses exactly the same categorical draw as the W1 path.
            candidate_local = int(torch.multinomial(q, 1, generator=rng).item())
            candidate_ids, candidate_q = [candidate_local], [q]
            features = [int(token) + P_VOCAB for token in candidate_ids]
            p2s = [p2] * width

            logits, states, active_tree = verify_tree(
                pending_f, p1, features, p2s,
                backup_positions=backup_positions,
            )
            if int(cache.get_pos()) != before:
                raise RuntimeError("tree verification moved the cache before acceptance")

            count("position", "verified")
            p1_distribution = policy(logits[0])
            if backup_diagnostic_rows is not None and backup_positions:
                candidate_ids = [p1, *backup_positions]
                target_probabilities = [
                    float(p1_distribution[token].item()) for token in candidate_ids
                ]
                rejection_probability = max(0.0, 1.0 - target_probabilities[0])
                cumulative_mass = 0.0
                conditional_cumulative_mass = []
                for probability in target_probabilities[1:]:
                    if rejection_probability > 0.0:
                        cumulative_mass += probability / rejection_probability
                        conditional_cumulative_mass.append(min(1.0, cumulative_mass))
                    else:
                        conditional_cumulative_mass.append(None)
                backup_diagnostic_rows.append({
                    "round": stats["rounds"],
                    "candidate_ids": candidate_ids,
                    "target_probabilities": target_probabilities,
                    "p1_rejection_probability": rejection_probability,
                    "backup_conditional_cumulative_mass": conditional_cumulative_mass,
                })
            if not bool(decide(float(p1_distribution[p1].item()), rng, device)):
                actual_p1 = int(residual_point(p1_distribution, p1, rng))
                if backup_positions:
                    stats["position_backup_rejections"] += 1
                if actual_p1 in backup_positions:
                    backup_index = backup_positions.index(actual_p1)
                    backup_rank = backup_index + 1
                    node_index = 4 + backup_index
                    stats["position_rescue_hits"] += 1
                    rank_hits = stats["position_rescue_hits_by_rank"]
                    rank_key = str(backup_rank)
                    rank_hits[rank_key] = int(rank_hits.get(rank_key, 0)) + 1
                    commit_path(before, [0, node_index], active_tree)
                    validate_position(previous, actual_p1)
                    online.finalize(actual_p1)
                    actual_f1 = draw(logits[node_index])
                    require(P_VOCAB <= actual_f1 < EOS,
                            "Target backup emitted an invalid Feature id")
                    prefix.extend([actual_p1, actual_f1])
                    h_prev = states[node_index].detach().clone()
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
                bonus_f2 = draw(logits[p2_node])
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
        backup_count=backup_count,
        proposal_width=2 + width,
        tree_width=width,
        seed=int(seed),
        prompt_tokens=len(prompt) - 1,
        requested_tokens=budget,
        temperature=0.9,
        target_top_p=0.9,
        draft_top_p=0.9,
        variant=f"ranker_mlp_feature_tree_w{width}_position_backup",
        mean_accepted_length=ratio(sum(stats["accepted_lengths"]), stats["rounds"]),
        mean_generated_per_round=ratio(sum(stats["round_generated_lengths"]),
                                       stats["rounds"]),
    )
    if backup_diagnostic_rows is not None:
        stats["position_backup_diagnostics"] = backup_diagnostic_rows
    if hasattr(tree_runtime, "stats"):
        stats["tree_runtime_stats"] = tree_runtime.stats()
    if rescue_runtime is not None and hasattr(rescue_runtime, "stats"):
        stats["rescue_runtime_stats"] = rescue_runtime.stats()
    stats["position_rescue_rate_per_backup_node"] = (
        stats["position_rescue_hits"] / stats["position_backup_nodes"]
        if stats["position_backup_nodes"] else None
    )
    stats["position_rescue_rate_by_rank"] = {
        rank: (
            hits / stats["position_backup_applicable_rounds_by_rank"][rank]
            if stats["position_backup_applicable_rounds_by_rank"][rank] else None
        )
        for rank, hits in stats["position_rescue_hits_by_rank"].items()
    }
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
    backup_count=1,
    backup_diagnostics=False,
    cache_factory=None,
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
            backup_count=backup_count,
            backup_diagnostics=backup_diagnostics,
            cache_factory=cache_factory,
        )


def self_test():
    return _run_backup_law_tests()


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2, sort_keys=True))
