"""Ranker+MLP decoding with optional pre-verification Position backups.

The proposal, acceptance, residual draw, and RNG path follow the existing W1
decoder. An optional selector sees only pre-verification inputs; an optional
observer receives post-draw measurements for eligible tree rounds.
"""
from __future__ import annotations

import math
import json
import copy
import statistics
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
    try:
        from patches.spec_next_20261022.prob_position_decoder import (
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
        raise ValueError("backup_count must be an integer from zero to four") from error
    if isinstance(value, bool) or backup_count != value or not 0 <= backup_count <= 4:
        raise ValueError("backup_count must be an integer from zero to four")
    return backup_count


def _select_backup_ids(selection, candidate_backup_ids, maximum):
    """Validate the selector's ordered subset of the legal Ranker candidates."""
    if not isinstance(selection, (tuple, list)):
        raise TypeError("backup_selector must return a list or tuple of Position ids")
    if any(isinstance(token, bool) for token in selection):
        raise ValueError("backup_selector returned a boolean instead of a Position id")
    selected = [int(token) for token in selection]
    candidates = tuple(int(token) for token in candidate_backup_ids)
    if len(selected) > int(maximum):
        raise ValueError("selector returned more backups than the configured maximum")
    if len(selected) != len(set(selected)):
        raise ValueError("selector returned duplicate backup Positions")
    if any(token not in candidates for token in selected):
        raise ValueError("selector may only choose from the supplied Ranker backup pool")
    return selected


def _read_bucket(required, capacity):
    required = int(required)
    capacity = int(capacity)
    if required < 1 or required > capacity:
        raise ValueError("tree shape exceeds Target cache capacity")
    bucket = max(8, 1 << (required - 1).bit_length())
    if bucket > capacity:
        bucket = capacity
    if bucket < required:
        raise ValueError("no read bucket can hold the requested tree shape")
    return bucket


def _runtime_graph_map(runtime, backup_count=0):
    if runtime is None or not bool(getattr(runtime, "graph_enabled", False)):
        return {}
    by_count = getattr(runtime, "_graphs_by_count", None)
    if by_count is not None:
        return by_count.get(int(backup_count), {})
    return getattr(runtime, "_graphs", {}) if int(backup_count) == 0 else {}


def _runtime_graph_count(runtime):
    if runtime is None:
        return 0
    by_count = getattr(runtime, "_graphs_by_count", None)
    if by_count is not None:
        return sum(len(group) for group in by_count.values())
    return len(getattr(runtime, "_graphs", {}))


def _runtime_graph_events(runtime):
    return getattr(runtime, "_graph_events", ()) if runtime is not None else ()


def _graph_shapes(tree_runtime, rescue_runtime, cache, before, max_backups,
                  capacity):
    shapes = {}
    for count in range(int(max_backups) + 1):
        node_count = 4 + count
        bucket = _read_bucket(int(before) + node_count, capacity)
        runtime = tree_runtime if count == 0 else rescue_runtime
        graph_map = _runtime_graph_map(
            runtime, backup_count=count if count else 0
        )
        key = (id(cache), 1, bucket)
        shapes[str(count)] = dict(
            node_count=node_count,
            read_bucket=bucket,
            graph_exists=key in graph_map,
            graph_enabled=bool(getattr(runtime, "graph_enabled", False))
            if runtime is not None else False,
        )
    return shapes


def _copy_graph_snapshot(tree_runtime, rescue_runtime):
    output = {}
    for name, runtime in (("tree", tree_runtime), ("rescue", rescue_runtime)):
        if runtime is None:
            output[name] = dict(graph_count=0, event_count=0)
            continue
        events = _runtime_graph_events(runtime)
        output[name] = dict(
            graph_count=_runtime_graph_count(runtime),
            event_count=len(events),
        )
    return output


def _observer_runtime_snapshot(observer, tree_runtime, rescue_runtime):
    if observer is None:
        return None
    return _copy_graph_snapshot(tree_runtime, rescue_runtime)


def _light_selector_context(
    *, prefix_len, remaining_budget, cache_pos, previous_position_rank,
    main_position, main_position_rank, main_score, candidates,
    position_vocab_size, eos_id, graph_state,
):
    """Build only fields consumed by selection.py for a live decision."""
    graph_fields = ("read_bucket", "graph_exists", "graph_enabled", "candidate_available")
    copied_graph_state = {
        str(key): {name: value[name] for name in graph_fields if name in value}
        for key, value in graph_state.items()
        if isinstance(value, dict)
    }
    base = copied_graph_state.get("0", {})
    return {
        "prefix_len": int(prefix_len),
        "remaining_budget": int(remaining_budget),
        "cache_pos": int(cache_pos),
        "read_bucket": int(base.get("read_bucket", 0)),
        "graph_exists": bool(base.get("graph_exists", False)),
        "graph_cold": bool(base.get("graph_enabled", False)) and not bool(
            base.get("graph_exists", False)
        ),
        "graph_state": copied_graph_state,
        "previous_position_rank": int(previous_position_rank),
        "main_position": int(main_position),
        "main_position_rank": None if main_position_rank is None else int(main_position_rank),
        "main_score": 0.0 if main_score is None else float(main_score),
        "position_vocab_size": int(position_vocab_size),
        "eos_id": int(eos_id),
        "candidates": [dict(row) for row in candidates],
    }


def _mark_candidate_availability(graph_state, candidate_count, rescue_available):
    for count_key, graph_item in graph_state.items():
        count = int(count_key)
        graph_item["candidate_available"] = (
            count <= int(candidate_count) and (count == 0 or bool(rescue_available))
        )
    return graph_state


def _cost_probe_feature_context(source_context, candidates, warmed_graph_state):
    """Build shape-cost inputs from graph state observed after probe warmup."""
    return _light_selector_context(
        prefix_len=source_context["prefix_len"],
        remaining_budget=source_context["remaining_budget"],
        cache_pos=(source_context["cache_pos"] if "cache_pos" in source_context
                   else source_context["logical_cache_pos"]),
        previous_position_rank=source_context["previous_position_rank"],
        main_position=source_context["main_position"],
        main_position_rank=source_context.get("main_position_rank"),
        main_score=source_context.get("main_score"),
        candidates=candidates,
        position_vocab_size=source_context["position_vocab_size"],
        eos_id=source_context["eos_id"],
        graph_state=warmed_graph_state,
    )


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
    if _validate_backup_count(0) != 0 or _validate_backup_count(1) != 1:
        raise AssertionError("fixed zero/one backup actions were not accepted")
    for invalid_count in (-1, 5, 1.5, True):
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
    if _select_backup_ids([], (12, 13), 4) != []:
        raise AssertionError("an empty adaptive action must select no backup")
    if _select_backup_ids([13], (12, 13), 4) != [13]:
        raise AssertionError("an adaptive action must preserve its selected candidate")
    for invalid_selection in ([14], [12, 12], [12, 13, 14]):
        try:
            _select_backup_ids(invalid_selection, (12, 13), 2)
        except ValueError:
            pass
        else:
            raise AssertionError("an invalid adaptive candidate set was accepted")
    if _read_bucket(9, 32) != 16:
        raise AssertionError("read bucket calculation differs from Target graph buckets")
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
    backup_selector=None,
    round_observer=None,
    observer_sync=False,
    cost_probe_stride=0,
    cost_probe_repeats=3,
    max_cost_probes=12,
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
    observer_sync = bool(observer_sync)
    cost_probe_stride = int(cost_probe_stride)
    cost_probe_repeats = int(cost_probe_repeats)
    max_cost_probes = int(max_cost_probes)
    if cost_probe_stride < 0 or cost_probe_repeats < 1 or max_cost_probes < 0:
        raise ValueError("cost probe settings must be nonnegative with positive repeats")
    if cost_probe_stride and round_observer is None:
        raise ValueError("cost probes require round_observer for diagnostic output")
    if backup_selector is not None and rescue_runtime is not None:
        runtime_max = int(getattr(rescue_runtime, "backup_count", 0))
        if runtime_max < backup_count:
            raise ValueError("rescue_runtime capacity is below backup_count")
    if cost_probe_stride and rescue_runtime is None:
        raise ValueError("cost probes require rescue_runtime for backup shapes")
    if backup_selector is not None and backup_count == 0:
        raise ValueError("dynamic backup selection requires a positive backup_count maximum")
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
        "backup_selector_calls": 0,
        "backup_selector_eligible_rounds": 0,
        "backup_selector_empty_candidate_skips": 0,
        "backup_selector_zero_actions": 0,
        "backup_selector_nodes_selected": 0,
        "backup_selector_wall_ms": 0.0,
        "round_observer_events": 0,
        "diagnostic_cost_probe_rounds": 0,
        "diagnostic_cost_probe_target_forwards": 0,
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

    current_round = None
    cost_probe_rounds = 0

    def propose_positions(state, previous, *, requested_backups, capture=None):
        proposed = state.propose(previous)
        if proposed is None:
            if capture is not None:
                capture["candidate_pool"] = []
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
            if capture is not None:
                capture["candidate_pool"] = [dict(
                    token_id=ids[0], ranker_rank=0, ranker_score=None,
                    position_rank=int(ranks[ids[0]]),
                    rank_gap=(int(ranks[ids[0]]) - int(ranks[previous])
                              if previous is not None else None),
                )]
            return (ids[0], 1, invalid), []
        score_array = ranker.predict_proba(features)[:, 1]
        scores = [float(value) for value in score_array]
        require(len(scores) == len(ids), "Ranker score count differs from candidate ids")
        require(all(math.isfinite(value) and value >= 0.0 for value in scores),
                "Ranker positive-class scores must be finite and nonnegative")
        best, backups = _ranked_candidate_indices(scores, requested_backups)
        if capture is not None:
            ranked_indices = sorted(range(len(scores)), key=lambda index: -scores[index])
            capture["candidate_pool"] = [
                dict(
                    token_id=ids[index],
                    ranker_rank=rank,
                    ranker_score=scores[index],
                    position_rank=int(ranks[ids[index]]),
                    rank_gap=(int(ranks[ids[index]]) - int(ranks[previous])
                              if previous is not None else None),
                )
                for rank, index in enumerate(ranked_indices[:1 + requested_backups])
            ]
        return (ids[best], len(ids), invalid), [ids[index] for index in backups]

    def begin_round_observation(before, previous, pending_f, remaining):
        nonlocal current_round
        if round_observer is None:
            return
        if observer_sync and device.type == "cuda":
            torch.cuda.synchronize(device)
        started_at = time.perf_counter()
        runtime_before = _observer_runtime_snapshot(
            round_observer, tree_runtime, rescue_runtime
        )
        graph_state = _graph_shapes(
            tree_runtime, rescue_runtime, cache, before, backup_count,
            int(cache.max_seq_len),
        )
        current_round = dict(
            context=dict(
                round_index=int(stats["rounds"]),
                prefix_tokens=tuple(int(token) for token in prefix),
                initial_prompt_tokens=tuple(int(token) for token in prompt),
                prefix_len=len(prefix),
                remaining_budget=int(remaining),
                logical_cache_pos=int(before),
                cache_pos=int(before),
                previous_position=int(previous),
                previous_position_rank=int(ranks[previous]),
                pending_feature=int(pending_f),
                device=str(device),
                graph_state=graph_state,
                read_bucket=int(graph_state["0"]["read_bucket"]),
                graph_exists=bool(graph_state["0"]["graph_exists"]),
                graph_cold=(
                    bool(graph_state["0"]["graph_enabled"]) and
                    not bool(graph_state["0"]["graph_exists"])
                ),
                position_vocab_size=int(P_VOCAB),
                eos_id=int(EOS),
                candidates=[],
                main_score=None,
                main_position_rank=None,
                graph_runtime_counts={
                    name: int(values["graph_count"])
                    for name, values in runtime_before.items()
                },
                max_backup_count=int(backup_count),
            ),
            candidate_pool=[],
            selected_backup_ids=[],
            labels={},
            runtime_before=runtime_before,
            runtime_after={},
            started_at=started_at,
            start_target_calls=int(stats["target_calls"]),
            start_target_input_tokens=int(stats["target_input_tokens"]),
            start_recovery_calls=int(stats["recovery_calls"]),
            start_tree_verify_calls=int(stats["tree_verify_calls"]),
            selector_wall_ms=0.0,
        )

    def capture_position_policy(distribution, candidate_rows, *, main=None,
                                actual=None, accepted=None):
        if round_observer is None or current_round is None:
            return
        ids = [int(row["token_id"]) for row in candidate_rows]
        for token in (main, actual, EOS):
            if token is not None and int(token) not in ids:
                ids.append(int(token))
        if ids:
            indices = torch.tensor(ids, dtype=torch.long, device=distribution.device)
            values = distribution.index_select(0, indices).detach().cpu().tolist()
            probabilities = {token: float(value)
                             for token, value in zip(ids, values)}
        else:
            probabilities = {}
        label = current_round["labels"]
        label["target_mass"] = [
            dict(token_id=int(row["token_id"]),
                 probability=probabilities.get(int(row["token_id"])))
            for row in candidate_rows
        ]
        label["target_mass_by_token"] = {
            str(int(row["token_id"])): probabilities.get(int(row["token_id"]))
            for row in candidate_rows
        }
        if main is not None:
            p_main = probabilities.get(int(main))
            p_eos = probabilities.get(int(EOS))
            label["p_main"] = p_main
            label["p_eos"] = p_eos
            label["recovery_observation_probability"] = (
                max(0.0, 1.0 - p_main - p_eos)
                if p_main is not None and p_eos is not None else None
            )
            label["residual_mass_by_token"] = {
                str(int(row["token_id"])): probabilities.get(int(row["token_id"]))
                for row in candidate_rows if int(row["token_id"]) != int(main)
            }
            label["residual_conditional_by_token"] = {
                str(int(row["token_id"])): (
                    probabilities.get(int(row["token_id"])) / (1.0 - p_main)
                    if probabilities.get(int(row["token_id"])) is not None and
                    p_main is not None and p_main < 1.0 else 0.0
                )
                for row in candidate_rows if int(row["token_id"]) != int(main)
            }
        if actual is not None:
            label["actual_position"] = int(actual)
            if accepted is False:
                label["actual_residual_position"] = int(actual)
                p_actual = probabilities.get(int(actual))
                label["residual_unconditional_mass"] = p_actual
                p_main = probabilities.get(int(main)) if main is not None else None
                label["residual_conditional_mass"] = (
                    p_actual / (1.0 - p_main)
                    if p_actual is not None and p_main is not None and p_main < 1.0
                    else None
                )
            label["actual_position_source"] = "proposal" if accepted else "residual"
            label["main_accepted"] = bool(accepted)

    def run_shape_cost_probe(pending_f, p1, features, p2s, candidate_backups):
        nonlocal cost_probe_rounds
        if (round_observer is None or cost_probe_stride <= 0 or
                cost_probe_rounds >= max_cost_probes or
                int(stats["rounds"]) % cost_probe_stride != 0):
            return None
        candidate_backups = tuple(int(token) for token in candidate_backups)
        maximum = min(backup_count, len(candidate_backups))
        if maximum < 1:
            return None
        before = int(cache.get_pos())
        if current_round is not None:
            current_round["diagnostic_only"] = True
        measurements = {count: [] for count in range(maximum + 1)}
        graph_state_before_warmup = {
            str(key): dict(value)
            for key, value in current_round["context"]["graph_state"].items()
        } if current_round is not None else {}
        warmup_event_snapshot = _copy_graph_snapshot(tree_runtime, rescue_runtime)
        warmup_event_counts = {
            "tree": warmup_event_snapshot["tree"]["event_count"],
            "rescue": warmup_event_snapshot["rescue"]["event_count"],
        }

        def verify_and_clear(count):
            start_pos = int(cache.get_pos())
            if start_pos != before:
                raise RuntimeError("cache position moved before a shape-cost probe")
            if count == 0:
                active_runtime = tree_runtime
                logits_probe, states_probe = active_runtime.verify(
                    cache, pending_f, p1, features, p2s
                )
            else:
                active_runtime = rescue_runtime
                logits_probe, states_probe = active_runtime.verify(
                    cache, pending_f, p1, features, p2s,
                    backup_positions=list(candidate_backups[:count]),
                )
            if int(cache.get_pos()) != before:
                raise RuntimeError("shape-cost verification changed logical cache position")
            del logits_probe, states_probe
            active_runtime.commit(cache, [])
            if int(cache.get_pos()) != before:
                raise RuntimeError("empty commit changed logical cache position")

        def timed_verify(count):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            verify_and_clear(count)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            return (time.perf_counter() - start) * 1000.0

        for count in range(maximum + 1):
            timed_verify(count)
        graph_state_after_warmup = _graph_shapes(
            tree_runtime, rescue_runtime, cache, before, maximum,
            int(cache.max_seq_len),
        )
        candidate_count = len(candidate_backups)
        _mark_candidate_availability(
            graph_state_after_warmup, candidate_count,
            rescue_runtime is not None,
        )
        cost_probe_feature_context = None
        if current_round is not None:
            cost_probe_feature_context = _cost_probe_feature_context(
                current_round["context"], current_round["candidate_pool"],
                graph_state_after_warmup,
            )
        for count in range(maximum + 1):
            for _ in range(cost_probe_repeats):
                measurements[count].append(timed_verify(count))
        if int(cache.get_pos()) != before:
            raise RuntimeError("shape-cost probes did not restore logical cache position")
        cost_probe_rounds += 1
        stats["diagnostic_cost_probe_rounds"] += 1
        stats["diagnostic_cost_probe_target_forwards"] += sum(
            len(values) + 1 for values in measurements.values()
        )
        medians = {
            str(count): float(statistics.median(values))
            for count, values in measurements.items()
        }
        control_median = medians["0"]
        deltas = {
            str(count): float(medians[str(count)] - control_median)
            for count in range(1, maximum + 1)
        }
        warmup_events = {}
        for name, runtime in (("tree", tree_runtime), ("rescue", rescue_runtime)):
            events = _runtime_graph_events(runtime)
            warmup_events[name] = [dict(event) for event in
                                   events[warmup_event_counts[name]:]]
        return dict(
            shape_cost_ms_by_k=medians,
            extra_shape_cost_ms_by_k=deltas,
            cost_probe_samples_ms_by_k={
                str(count): list(values) for count, values in measurements.items()
            },
            cost_probe_warmup_graph_events=warmup_events,
            cost_probe_graph_state_before_warmup=graph_state_before_warmup,
            cost_probe_graph_state_after_warmup=graph_state_after_warmup,
            cost_probe_feature_context=cost_probe_feature_context,
            cost_probe_repeats=int(cost_probe_repeats),
            cost_probe_scope="diagnostic_only_same_cache_empty_commit",
        )

    def finish_round(begin, accepted):
        nonlocal current_round
        stats["accepted_lengths"].append(int(accepted))
        stats["round_generated_lengths"].append(len(prefix) - begin)
        if (round_observer is not None and current_round is not None and
                current_round.get("eligible_for_value_observation", False)):
            if observer_sync and device.type == "cuda":
                torch.cuda.synchronize(device)
            finished_at = time.perf_counter()
            current_round["labels"].update(
                accepted_tokens=int(accepted),
                generated_tokens=len(prefix) - begin,
                eos=bool(prefix and prefix[-1] == EOS),
                target_calls=int(stats["target_calls"] - current_round["start_target_calls"]),
                target_input_tokens=int(
                    stats["target_input_tokens"] - current_round["start_target_input_tokens"]
                ),
                recovery_calls=int(
                    stats["recovery_calls"] - current_round["start_recovery_calls"]
                ),
                tree_verify_calls=int(
                    stats["tree_verify_calls"] - current_round["start_tree_verify_calls"]
                ),
            )
            current_round["labels"].setdefault("recovery_call_occurred", False)
            current_round["labels"].setdefault("recovery_stage", None)
            current_round["labels"].setdefault("recovery_ms", None)
            current_round["labels"].setdefault("actual_residual_position", None)
            current_round["labels"].setdefault("backup_hit", False)
            current_round["labels"].setdefault("backup_hit_rank", None)
            current_round["labels"].setdefault("p_main", None)
            current_round["labels"].setdefault("p_eos", None)
            current_round["labels"].setdefault("recovery_observation_probability", None)
            for name, runtime in (("tree", tree_runtime), ("rescue", rescue_runtime)):
                before_snapshot = current_round["runtime_before"][name]
                events = _runtime_graph_events(runtime)
                new_events = [dict(event) for event in events[before_snapshot["event_count"]:]]
                after_count = _runtime_graph_count(runtime)
                current_round["runtime_after"][name] = dict(
                    graph_count=after_count,
                    event_count=len(events),
                    new_graph_events=new_events,
                    graph_count_delta=after_count - before_snapshot["graph_count"],
                )
            graph_count_delta = sum(
                current_round["runtime_after"][name]["graph_count_delta"]
                for name in ("tree", "rescue")
            )
            current_round["labels"]["new_graph_count"] = graph_count_delta
            selected_k = len(current_round["selected_backup_ids"])
            selected_graph = current_round["context"]["graph_state"].get(str(selected_k))
            current_round["labels"]["selected_backup_count"] = selected_k
            current_round["labels"]["graph_cold_before_round"] = (
                None if selected_graph is None else not selected_graph["graph_exists"]
            )
            current_round["labels"]["selected_read_bucket"] = (
                None if selected_graph is None else selected_graph["read_bucket"]
            )
            timing_mode = "cuda_synchronized" if observer_sync and device.type == "cuda" else "host_interval"
            current_round["timing"] = dict(
                mode=timing_mode,
                round_wall_ms=(finished_at - current_round["started_at"]) * 1000.0,
                selector_wall_ms=current_round.get("selector_wall_ms", 0.0),
                recovery_forward_timing=current_round.get("recovery_timing_mode"),
            )
            if current_round.get("shape_cost_ms_by_k") is not None:
                current_round["labels"]["shape_cost_ms_by_k"] = current_round["shape_cost_ms_by_k"]
                current_round["labels"]["extra_shape_cost_ms_by_k"] = current_round["extra_shape_cost_ms_by_k"]
                current_round["labels"]["cost_probe_samples_ms_by_k"] = current_round["cost_probe_samples_ms_by_k"]
                current_round["labels"]["cost_probe_warmup_graph_events"] = current_round["cost_probe_warmup_graph_events"]
                current_round["labels"]["cost_probe_graph_state_before_warmup"] = current_round["cost_probe_graph_state_before_warmup"]
                current_round["labels"]["cost_probe_graph_state_after_warmup"] = current_round["cost_probe_graph_state_after_warmup"]
                current_round["labels"]["cost_probe_feature_context"] = current_round["cost_probe_feature_context"]
            recovery_ms = current_round["labels"].get("recovery_ms")
            recovery_called = bool(current_round["labels"].get("recovery_call_occurred"))
            residual_position = current_round["labels"].get("actual_residual_position")
            current_round["labels"]["recovery_value_label_valid_for_w1"] = (
                len(current_round["selected_backup_ids"]) == 0
            )
            current_round["labels"]["recovery_value_observed_ms_by_candidate"] = [
                dict(
                    token_id=int(row["token_id"]),
                    ranker_rank=int(row["ranker_rank"]),
                    actual_residual_match=(
                        residual_position is not None and
                        int(residual_position) == int(row["token_id"])
                    ),
                    recovery_call_occurred=recovery_called,
                    observed_value_ms=(
                        float(recovery_ms)
                        if recovery_called and residual_position is not None and
                        int(residual_position) == int(row["token_id"]) and
                        recovery_ms is not None else 0.0
                    ),
                )
                for row in current_round["candidate_pool"][1:]
            ]
            event = dict(
                schema="recovery_value_round_v1",
                scope="single_round",
                features=current_round["context"],
                round_context=current_round["context"],
                candidates=current_round["candidate_pool"],
                selected_backup_ids=list(current_round["selected_backup_ids"]),
                labels=current_round["labels"],
                runtime_before=current_round["runtime_before"],
                runtime_after=current_round["runtime_after"],
                timing=current_round["timing"],
                diagnostic_only=bool(current_round.get("diagnostic_only", False)),
            )
            round_observer(event)
            stats["round_observer_events"] += 1
        current_round = None

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

        def target_forward(tokens, *, recovery=False, recovery_stage=None):
            tokens = [int(token) for token in tokens]
            if not tokens:
                raise ValueError("Target forward requires at least one token")
            start = int(cache.get_pos())
            kinds = shared.typed_kinds(start, len(tokens))
            measure_recovery = bool(
                recovery and round_observer is not None and current_round is not None
            )
            # Recovery labels are conditional-cost observations, so synchronize
            # the measured Target call even when the broader round timer is host-only.
            sync_recovery = bool(measure_recovery and device.type == "cuda")
            if sync_recovery:
                torch.cuda.synchronize(device)
            recovery_started = time.perf_counter() if measure_recovery else None
            logits, states = capture.run(target, cache, tokens, kinds, device)
            if sync_recovery:
                torch.cuda.synchronize(device)
            if int(cache.get_pos()) != start + len(tokens):
                raise RuntimeError("ordinary Target call advanced the wrong number of tokens")
            stats["target_calls"] += 1
            stats["target_input_tokens"] += len(tokens)
            if recovery:
                stats["recovery_calls"] += 1
                if measure_recovery:
                    current_round["labels"]["recovery_call_occurred"] = True
                    current_round["labels"]["recovery_stage"] = recovery_stage
                    current_round["labels"]["recovery_ms"] = (
                        time.perf_counter() - recovery_started
                    ) * 1000.0
                    current_round["recovery_timing_mode"] = (
                        "cuda_synchronized" if sync_recovery else "host_interval"
                    )
            return logits, states

        def verify_tree(pending_f, p1, features, p2s, backup_positions=()):
            before = int(cache.get_pos())
            active_tree = tree_runtime
            backup_positions = tuple(int(token) for token in backup_positions)
            if backup_positions:
                if rescue_runtime is None:
                    raise ValueError("rescue_runtime is required when a backup Position is available")
                active_tree = rescue_runtime
                if len(backup_positions) == 1:
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
            begin_round_observation(before, previous, pending_f, remaining)

            stats["ranker_calls"] += 1
            proposal_info = {}
            first, backup_positions = propose_positions(
                online, previous,
                requested_backups=(backup_count if remaining >= 4 else 0),
                capture=(proposal_info if backup_selector is not None or
                         round_observer is not None else None),
            )
            if current_round is not None:
                proposal_pool = proposal_info.get("candidate_pool", [])
                current_round["candidate_pool"] = [dict(row) for row in proposal_pool]
                current_round["context"].update(
                    main_position=(int(first[0]) if first is not None else None),
                    ranker_candidate_count=(int(first[1]) if first is not None else 0),
                    candidate_pool=[dict(row) for row in proposal_pool],
                    candidates=[dict(row) for row in proposal_pool],
                    main_score=(proposal_pool[0]["ranker_score"]
                                if proposal_pool else None),
                    main_position_rank=(proposal_pool[0]["position_rank"]
                                        if proposal_pool else None),
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
            if current_round is not None:
                current_round["eligible_for_value_observation"] = bool(can_tree)
            if not can_tree:
                backup_positions = []
            elif backup_selector is not None:
                candidate_pool = proposal_info.get("candidate_pool", [])
                candidate_backup_pool = [
                    int(row["token_id"]) for row in candidate_pool[1:]
                ]
                if current_round is not None:
                    current_round["candidate_pool"] = [dict(row) for row in candidate_pool]
                    _mark_candidate_availability(
                        current_round["context"]["graph_state"],
                        len(candidate_backup_pool), rescue_runtime is not None,
                    )
                    current_round["context"].update(
                        main_position=int(first[0]),
                        ranker_candidate_count=int(first[1]),
                        candidate_pool=[dict(row) for row in candidate_pool],
                        candidates=[dict(row) for row in candidate_pool],
                        main_score=(candidate_pool[0]["ranker_score"]
                                    if candidate_pool else None),
                        main_position_rank=(candidate_pool[0]["position_rank"]
                                            if candidate_pool else None),
                    )
                    decision_context = copy.deepcopy(current_round["context"])
                elif candidate_backup_pool:
                    graph_state = _graph_shapes(
                        tree_runtime, rescue_runtime, cache, before,
                        backup_count, int(cache.max_seq_len),
                    )
                    _mark_candidate_availability(
                        graph_state, len(candidate_backup_pool),
                        rescue_runtime is not None,
                    )
                    decision_context = _light_selector_context(
                        prefix_len=len(prefix),
                        remaining_budget=remaining,
                        cache_pos=before,
                        previous_position_rank=ranks[previous],
                        main_position=first[0],
                        main_position_rank=(candidate_pool[0]["position_rank"]
                                            if candidate_pool else None),
                        main_score=(candidate_pool[0]["ranker_score"]
                                    if candidate_pool else None),
                        candidates=candidate_pool,
                        position_vocab_size=P_VOCAB,
                        eos_id=EOS,
                        graph_state=graph_state,
                    )
                else:
                    decision_context = None
                stats["backup_selector_eligible_rounds"] += 1
                selector_wall_ms = 0.0
                if candidate_backup_pool:
                    selector_started = time.perf_counter()
                    selection = backup_selector(decision_context)
                    selector_wall_ms = (time.perf_counter() - selector_started) * 1000.0
                    stats["backup_selector_calls"] += 1
                    stats["backup_selector_wall_ms"] += selector_wall_ms
                    backup_positions = _select_backup_ids(
                        selection, candidate_backup_pool, backup_count
                    )
                else:
                    backup_positions = []
                    stats["backup_selector_empty_candidate_skips"] += 1
                if backup_positions and rescue_runtime is None:
                    raise ValueError("selector chose backups without rescue_runtime")
                stats["backup_selector_nodes_selected"] += len(backup_positions)
                if candidate_backup_pool and not backup_positions:
                    stats["backup_selector_zero_actions"] += 1
                if current_round is not None:
                    current_round["selected_backup_ids"] = list(backup_positions)
                    current_round["selector_wall_ms"] = selector_wall_ms
            else:
                if current_round is not None:
                    available_backup_count = len(proposal_info.get("candidate_pool", [])) - 1
                    _mark_candidate_availability(
                        current_round["context"]["graph_state"],
                        max(0, available_backup_count), rescue_runtime is not None,
                    )
                    current_round["candidate_pool"] = [
                        dict(row) for row in proposal_info.get("candidate_pool", [])
                    ]
                    current_round["selected_backup_ids"] = list(backup_positions)

            if not can_tree:
                if current_round is not None:
                    current_round["candidate_pool"] = [
                        dict(row) for row in proposal_info.get("candidate_pool", [])
                    ]
                    current_round["selected_backup_ids"] = []
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
                main_accepted = bool(decide(float(p1_distribution[p1].item()), rng, device))
                observed_candidates = (
                    current_round["candidate_pool"] if current_round is not None else
                    proposal_info.get("candidate_pool", [])
                )
                if main_accepted:
                    count("position", "accepted")
                    actual_p1 = p1
                    h_prev = states[1].detach().clone()
                    actual_f1 = draw(logits[1])
                    accepted = 1
                else:
                    actual_p1 = int(residual_point(p1_distribution, p1, rng))
                    capture_position_policy(
                        p1_distribution, observed_candidates, main=p1,
                        actual=actual_p1, accepted=False,
                    )
                    truncate(cache, before + 1, before + 2)
                    if int(cache.get_pos()) != before + 1:
                        raise RuntimeError("K1 Position recovery left a bad cache length")
                    if actual_p1 == EOS:
                        prefix.append(actual_p1)
                        online.drop_eos()
                        finish_round(begin, 0)
                        break
                    validate_position(previous, actual_p1)
                    logits, states = target_forward(
                        [actual_p1], recovery=True, recovery_stage="p1"
                    )
                    actual_f1 = draw(logits[-1])
                    h_prev = states[-1].detach().clone()
                    accepted = 0
                if main_accepted:
                    capture_position_policy(
                        p1_distribution, observed_candidates, main=p1,
                        actual=actual_p1, accepted=True,
                    )
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

            if cost_probe_stride and round_observer is not None:
                probe_candidates = [
                    int(row["token_id"])
                    for row in proposal_info.get("candidate_pool", [])[1:]
                ]
                probe_result = run_shape_cost_probe(
                    pending_f, p1, features, p2s, probe_candidates
                )
                if probe_result is not None and current_round is not None:
                    current_round.update(probe_result)

            logits, states, active_tree = verify_tree(
                pending_f, p1, features, p2s,
                backup_positions=backup_positions,
            )
            if int(cache.get_pos()) != before:
                raise RuntimeError("tree verification moved the cache before acceptance")

            count("position", "verified")
            p1_distribution = policy(logits[0])
            observed_candidates = (
                current_round["candidate_pool"] if current_round is not None else
                proposal_info.get("candidate_pool", [])
            )
            if backup_diagnostic_rows is not None and backup_positions:
                position_candidate_ids = [p1, *backup_positions]
                target_probabilities = [
                    float(p1_distribution[token].item()) for token in position_candidate_ids
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
                    "candidate_ids": position_candidate_ids,
                    "target_probabilities": target_probabilities,
                    "p1_rejection_probability": rejection_probability,
                    "backup_conditional_cumulative_mass": conditional_cumulative_mass,
                })
            main_accepted = bool(decide(float(p1_distribution[p1].item()), rng, device))
            if not main_accepted:
                actual_p1 = int(residual_point(p1_distribution, p1, rng))
                capture_position_policy(
                    p1_distribution, observed_candidates, main=p1,
                    actual=actual_p1, accepted=False,
                )
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
                    if current_round is not None:
                        current_round["labels"].update(
                            backup_hit=True,
                            backup_hit_rank=backup_rank,
                        )
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
                logits_recovery, states_recovery = target_forward(
                    [actual_p1], recovery=True, recovery_stage="p1"
                )
                actual_f1 = draw(logits_recovery[-1])
                require(P_VOCAB <= actual_f1 < EOS,
                        "Target emitted an invalid Feature id")
                prefix.append(actual_f1)
                h_prev = states_recovery[-1].detach().clone()
                finish_round(begin, 0)
                continue

            capture_position_policy(
                p1_distribution, observed_candidates, main=p1,
                actual=p1, accepted=True,
            )
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
            logits_recovery, states_recovery = target_forward(
                [actual_p2], recovery=True, recovery_stage="p2"
            )
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
        backup_action=("dynamic" if backup_selector is not None else "fixed"),
        proposal_width=2 + width,
        tree_width=width,
        seed=int(seed),
        prompt_tokens=len(prompt) - 1,
        requested_tokens=budget,
        temperature=0.9,
        target_top_p=0.9,
        draft_top_p=0.9,
        variant=f"ranker_mlp_feature_tree_w{width}_position_backup",
        backup_selector_enabled=backup_selector is not None,
        round_observer_enabled=round_observer is not None,
        observer_sync=observer_sync,
        measurement_mode=("observer_diagnostic" if round_observer is not None else "decode"),
        diagnostic_only=bool(stats["diagnostic_cost_probe_rounds"]),
        cost_probe_stride=cost_probe_stride,
        cost_probe_repeats=cost_probe_repeats,
        max_cost_probes=max_cost_probes,
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
    selector_stats = getattr(backup_selector, "stats", None)
    if callable(selector_stats):
        stats["backup_selector_stats"] = selector_stats()
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
    backup_selector=None,
    round_observer=None,
    observer_sync=False,
    cost_probe_stride=0,
    cost_probe_repeats=3,
    max_cost_probes=12,
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
            backup_selector=backup_selector,
            round_observer=round_observer,
            observer_sync=observer_sync,
            cost_probe_stride=cost_probe_stride,
            cost_probe_repeats=cost_probe_repeats,
            max_cost_probes=max_cost_probes,
        )


def self_test():
    return _run_backup_law_tests()


decode_rescue = decode


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2, sort_keys=True))
