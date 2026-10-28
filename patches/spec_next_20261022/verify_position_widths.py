#!/usr/bin/env python3
"""Run a short numeric and cost probe for one to four Position backups.

The probe reuses the existing Target capture and tensor comparisons.  It does
not invoke a decoder.  Ranker candidates are kept separate from any explicitly
constructed legal positions used only to exercise a tree path.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
from pathlib import Path
import sys
import time
from contextlib import nullcontext


POLICY_TV_LIMIT = 1e-5
RELATIVE_L2_LIMIT = 2e-6
MAX_BACKUP_COUNT = 4
MIN_RANKER_CANDIDATES = MAX_BACKUP_COUNT + 2


def _jsonable(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        return _jsonable(value.detach().cpu().tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(value), ensure_ascii=False,
                                allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _parse_int_list(value, *, name, minimum=None, maximum=None):
    result = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        number = int(item)
        if minimum is not None and number < minimum:
            raise ValueError(f"{name} contains {number}, below {minimum}")
        if maximum is not None and number > maximum:
            raise ValueError(f"{name} contains {number}, above {maximum}")
        result.append(number)
    if not result:
        raise ValueError(f"{name} is empty")
    return result


def _tensor_metadata(tensor):
    return dict(
        shape=[int(value) for value in tensor.shape],
        ndim=int(tensor.ndim),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        layout=str(tensor.layout),
        numel=int(tensor.numel()),
        element_size=int(tensor.element_size()),
    )


def _path_tensor_metadata(path):
    return dict(
        logits=_tensor_metadata(path["logits"]),
        hidden=_tensor_metadata(path["hidden"]),
        kv_k=_tensor_metadata(path["kv_k"]),
        kv_v=_tensor_metadata(path["kv_v"]),
        tokens=_tensor_metadata(path["tokens"]),
    )


def _support_set(shared, logits, feature_start, feature_end):
    policy = shared.policy(logits).float()
    feature_policy = policy[int(feature_start):int(feature_end)]
    finite = bool(torch_isfinite(feature_policy).all().item())
    if finite:
        local = torch_nonzero(feature_policy > 0)
        local_indices = [int(value) for value in local.detach().cpu().flatten().tolist()]
        global_indices = [int(feature_start) + value for value in local_indices]
        mass = float(feature_policy.sum().item())
    else:
        local_indices = []
        global_indices = []
        mass = None
    return dict(
        finite=finite,
        local_indices=local_indices,
        global_indices=global_indices,
        size=len(local_indices),
        mass=mass,
    )


def torch_isfinite(value):
    import torch

    return torch.isfinite(value)


def torch_nonzero(value):
    import torch

    return torch.nonzero(value, as_tuple=False)


def _feature_argmax(shared, logits, feature_start, feature_end):
    import torch

    policy = shared.policy(logits).float()
    values = policy[int(feature_start):int(feature_end)]
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("Target Feature policy contains a non-finite value")
    return int(feature_start) + int(torch.argmax(values).item())


def _best_ranker_candidates(shared, ranker, ranks, prompt, *, minimum,
                            allow_constructed=False):
    from bench_gaussiangpt_pf_block2_cow import fork_position_state

    state = shared.position_state(prompt, ranks)
    previous = int(prompt[-2])
    proposed = state.propose(previous)
    if proposed is None:
        return None
    position_ids, features, _guess, _invalid = proposed
    position_ids = [int(value) for value in position_ids]
    if len(position_ids) < int(minimum) and not allow_constructed:
        return None
    scores = ranker.predict_proba(features)[:, 1]
    if len(position_ids) != len(scores):
        raise ValueError("Ranker score count differs from Position candidates")
    ranked = sorted(
        range(len(position_ids)),
        key=lambda index: (-float(scores[index]), index),
    )
    candidates = []
    for order, index in enumerate(ranked[:int(minimum)], start=1):
        token = int(position_ids[index])
        if int(ranks[token]) <= int(ranks[previous]):
            raise ValueError("Ranker returned a non-increasing Position")
        candidates.append(dict(
            ranker_order=order,
            position=token,
            ranker_score=float(scores[index]),
            proposal_index=int(index),
            constructed=False,
        ))
    if allow_constructed and len(candidates) < int(minimum):
        used = {item["position"] for item in candidates}
        legal = sorted(range(len(ranks)), key=lambda token: int(ranks[token]))
        for token in legal:
            if token in used or int(ranks[token]) <= int(ranks[previous]):
                continue
            candidates.append(dict(ranker_order=len(candidates) + 1,
                                   position=int(token), ranker_score=None,
                                   proposal_index=None, constructed=True))
            used.add(token)
            if len(candidates) >= int(minimum):
                break
        if len(candidates) < int(minimum):
            return None
    p1 = int(candidates[0]["position"])
    shadow = fork_position_state(state)
    shadow.finalize(p1)
    second = shadow.propose(p1)
    if second is None:
        return None
    p2_ids, p2_features, _p2_guess, _p2_invalid = second
    p2_ids = [int(value) for value in p2_ids]
    if not p2_ids:
        return None
    p2_scores = ranker.predict_proba(p2_features)[:, 1]
    p2_index = max(
        range(len(p2_scores)), key=lambda index: (float(p2_scores[index]), -index)
    )
    p2 = int(p2_ids[p2_index])
    if int(ranks[p2]) <= int(ranks[p1]):
        raise ValueError("Ranker returned a non-increasing second Position")
    return dict(
        previous=previous,
        pending_f=int(prompt[-1]),
        p1=p1,
        feature=int(prompt[-1]),
        p2=p2,
        candidates=candidates,
        candidate_pool_count=len(position_ids),
    )


def _quality_curve(shared, logits, candidates, backup_count, *, p_vocab):
    policy = shared.policy(logits).float()
    p1_probability = float(policy[int(candidates[0]["position"])].item())
    reject_mass = max(0.0, 1.0 - p1_probability)
    ranker_total = sum(float(item["ranker_score"]) for item in candidates
                       if not item.get("constructed", False))
    rows = []
    cumulative_probability = 0.0
    cumulative_ranker_score = 0.0
    for item in candidates[:1 + int(backup_count)]:
        if item.get("constructed", False):
            continue
        probability = float(policy[int(item["position"])].item())
        if int(item["ranker_order"]) > 1:
            cumulative_probability += probability
            cumulative_ranker_score += float(item["ranker_score"])
        rows.append(dict(
            ranker_order=int(item["ranker_order"]),
            position=int(item["position"]),
            ranker_score=float(item["ranker_score"]),
            ranker_score_fraction=(
                float(item["ranker_score"]) / ranker_total
                if ranker_total > 0.0 else None
            ),
            target_probability=probability,
            conditional_probability_after_p1_reject=(
                probability / reject_mass if reject_mass > 0.0 else None
            ),
            cumulative_backup_target_probability=(
                cumulative_probability if int(item["ranker_order"]) > 1 else 0.0
            ),
            cumulative_backup_conditional_probability=(
                cumulative_probability / reject_mass if reject_mass > 0.0 else None
            ),
            cumulative_backup_ranker_score=(
                cumulative_ranker_score if int(item["ranker_order"]) > 1 else 0.0
            ),
        ))
    return dict(
        p1_target_probability=p1_probability,
        p1_reject_mass=reject_mass,
        backup_count=int(backup_count),
        cumulative_backup_target_probability=cumulative_probability,
        cumulative_backup_conditional_probability=(
            cumulative_probability / reject_mass if reject_mass > 0.0 else None
        ),
        cumulative_backup_ranker_score=cumulative_ranker_score,
        candidate_rows=rows,
        p_vocab=int(p_vocab),
    )


def _sync_time(torch, function):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def _warm_tree_costs(torch, runtime, cache, case, backups, repeats):
    measurements = []
    for _ in range(repeats):
        call = lambda: runtime.verify(
            cache, case["nodes"]["pending_f"], case["nodes"]["p1"],
            [case["nodes"]["feature"]], [case["nodes"]["p2"]],
            **({"backup_positions": backups} if backups else {}),
        )
        _, seconds = _sync_time(torch, call)
        measurements.append(seconds * 1000.0)
        runtime.commit(cache, [])
    return dict(samples_ms=measurements, mean_ms=statistics.fmean(measurements),
                median_ms=statistics.median(measurements), repeats=repeats)


def _warm_recovery_costs(torch, target, cache, case, shared, device,
                         CaptureLastBlock, repeats):
    capture = CaptureLastBlock(target)
    start = int(cache.get_pos())
    measurements = []
    try:
        for _ in range(repeats):
            _, seconds = _sync_time(torch, lambda: _capture_forward(
                target, capture, cache, [case["fallback_position"]], shared, device))
            measurements.append(seconds * 1000.0)
            cache.cache_seqlens.fill_(start)
            cache.token_cache[:, start:].fill_(-1)
    finally:
        capture.close()
    return dict(samples_ms=measurements, mean_ms=statistics.fmean(measurements),
                median_ms=statistics.median(measurements), repeats=repeats)


def _prefill(target, cache, prompt, shared, CaptureLastBlock, fp32_context, device):
    cache.reset()
    capture = CaptureLastBlock(target)
    try:
        with fp32_context():
            _capture_forward(target, capture, cache, prompt[:-1], shared, device)
    finally:
        capture.close()


def _capture_forward(target, capture, cache, tokens, shared, device):
    start = int(cache.get_pos())
    kinds = shared.typed_kinds(start, len(tokens))
    logits, hidden = capture.run(
        target, cache, [int(token) for token in tokens], kinds, device
    )
    if int(cache.get_pos()) != start + len(tokens):
        raise RuntimeError("ordinary Target call advanced an unexpected number of tokens")
    if logits.ndim == 3:
        logits = logits[0]
    if hidden.ndim == 3:
        hidden = hidden[0]
    return logits, hidden, start


def _ordinary_path(target, cache, prompt, continuation, shared, device,
                   CaptureLastBlock, fp32_context, TargetTreeRuntime,
                   rank_mask_runtime, rank_mask_p1, rank_mask_width, ranks):
    cache.reset()
    capture = CaptureLastBlock(target)
    try:
        with fp32_context():
            _capture_forward(target, capture, cache, prompt[:-1], shared, device)
            before = int(cache.get_pos())
            logits, hidden, start = _capture_forward(
                target, capture, cache, continuation, shared, device
            )
            if start != before:
                raise RuntimeError("ordinary path position changed before its forward")
            rank_map = target.sparse_position_rank_map.to(logits.device)
            p_vocab = int(target.sparse_position_vocab_size)
            for index in range(0, logits.shape[0], 2):
                previous = int(prompt[-2]) if index == 0 else int(continuation[index - 1])
                if 0 <= previous < p_vocab:
                    logits[index, :p_vocab].masked_fill_(rank_map[:p_vocab] <= rank_map[previous], float("-inf"))
            return dict(
                before=before,
                after=int(cache.get_pos()),
                logits=logits.detach().clone(),
                hidden=hidden.detach().clone(),
                kv_k=cache.k_cache[:, :, :cache.get_pos()].detach().clone(),
                kv_v=cache.v_cache[:, :, :cache.get_pos()].detach().clone(),
                tokens=cache.token_cache[:, :cache.get_pos()].detach().clone(),
            )
    finally:
        capture.close()


def _compare_node(torch, shared, expected_logits, expected_hidden,
                  actual_logits, actual_hidden, *, expected_node, actual_node,
                  atol, rtol):
    from verify_candidates import _node_comparison

    return _node_comparison(
        torch, shared, expected_logits, expected_hidden,
        actual_logits, actual_hidden, atol=atol, rtol=rtol,
        expected_node=expected_node, actual_node=actual_node,
    )


def _graph_node_rows(torch, shared, graph_logits, graph_hidden, primary,
                     references, candidates, *, atol, rtol):
    primary_rows = []
    for index in range(4):
        primary_rows.append(_compare_node(
            torch, shared, primary["logits"][index], primary["hidden"][index],
            graph_logits[index], graph_hidden[index], expected_node=index,
            actual_node=index, atol=atol, rtol=rtol,
        ))
    backup_rows = []
    for index, candidate in enumerate(candidates[1:graph_logits.shape[0] - 3], start=4):
        reference = references[int(candidate["position"])]
        backup_rows.append(_compare_node(
            torch, shared, reference["logits"][1], reference["hidden"][1],
            graph_logits[index], graph_hidden[index],
            expected_node=1, actual_node=index, atol=atol, rtol=rtol,
        ))
    return primary_rows, backup_rows


def _feature_policy_record(shared, graph_logits, reference_logits,
                           feature_start, feature_end):
    graph_support = _support_set(shared, graph_logits, feature_start, feature_end)
    target_support = _support_set(shared, reference_logits, feature_start, feature_end)
    return dict(
        graph=graph_support,
        ordinary_target=target_support,
        support_equal=(
            graph_support["finite"] and target_support["finite"]
            and graph_support["global_indices"] == target_support["global_indices"]
        ),
    )


def _kv_record(torch, graph_cache, reference, end, *, atol, rtol):
    from verify_candidates import _strict_tensor_metrics

    end = int(end)
    token_actual = graph_cache.token_cache[:, :end]
    token_reference = reference["tokens"][:, :end]
    token_equal = bool(torch.equal(token_actual, token_reference))
    k_metrics = _strict_tensor_metrics(
        torch, reference["kv_k"][:, :, :end], graph_cache.k_cache[:, :, :end],
        atol=atol, rtol=rtol,
    )
    v_metrics = _strict_tensor_metrics(
        torch, reference["kv_v"][:, :, :end], graph_cache.v_cache[:, :, :end],
        atol=atol, rtol=rtol,
    )
    passed = bool(
        token_equal
        and k_metrics["all_finite"] and v_metrics["all_finite"]
        and k_metrics["finite_mask_equal"] and v_metrics["finite_mask_equal"]
        and k_metrics["relative_l2"] is not None
        and v_metrics["relative_l2"] is not None
        and k_metrics["relative_l2"] <= RELATIVE_L2_LIMIT
        and v_metrics["relative_l2"] <= RELATIVE_L2_LIMIT
        and k_metrics["max_abs_over_reference_max_abs"] is not None
        and v_metrics["max_abs_over_reference_max_abs"] is not None
        and k_metrics["max_abs_over_reference_max_abs"] <= RELATIVE_L2_LIMIT
        and v_metrics["max_abs_over_reference_max_abs"] <= RELATIVE_L2_LIMIT
    )
    return dict(
        end_position=end,
        token_equal=token_equal,
        k_metrics=k_metrics,
        v_metrics=v_metrics,
        tensor_metadata=dict(
            actual_k=_tensor_metadata(graph_cache.k_cache[:, :, :end]),
            actual_v=_tensor_metadata(graph_cache.v_cache[:, :, :end]),
            actual_tokens=_tensor_metadata(token_actual),
            ordinary_k=_tensor_metadata(reference["kv_k"][:, :, :end]),
            ordinary_v=_tensor_metadata(reference["kv_v"][:, :, :end]),
            ordinary_tokens=_tensor_metadata(token_reference),
        ),
        passed=passed,
    )


def _run_forced_backup(*, torch, target, runtime, graph_cache, case, backups,
                       backup_index, primary, references, shared, device,
                       CaptureLastBlock, TargetTreeRuntime, fp32_context,
                       feature_start, feature_end, atol, rtol):
    from verify_candidates import _strict_tensor_metrics

    _prefill(target, graph_cache, case["prompt"], shared, CaptureLastBlock,
             fp32_context, device)
    before = int(graph_cache.get_pos())
    (graph_output, tree_verify_seconds) = _sync_time(
        torch,
        lambda: runtime.verify(
            graph_cache, case["nodes"]["pending_f"], case["nodes"]["p1"],
            [case["nodes"]["feature"]], [case["nodes"]["p2"]],
            backup_positions=backups,
        ),
    )
    graph_logits, graph_hidden = graph_output
    graph_logits = graph_logits.detach().clone()
    graph_hidden = graph_hidden.detach().clone()
    if graph_logits.ndim == 3:
        graph_logits = graph_logits[0]
    if graph_hidden.ndim == 3:
        graph_hidden = graph_hidden[0]
    node = 4 + int(backup_index)
    position = int(backups[backup_index])
    reference = references[position]
    node_comparison = _compare_node(
        torch, shared, reference["logits"][1], reference["hidden"][1],
        graph_logits[node], graph_hidden[node], expected_node=1,
        actual_node=node, atol=atol, rtol=rtol,
    )
    feature_policy = _feature_policy_record(
        shared, graph_logits[node], reference["logits"][1],
        feature_start, feature_end,
    )
    feature = _feature_argmax(shared, reference["logits"][1],
                              feature_start, feature_end)
    continuation = references[position]["continuations"][feature]
    commit_path = [0, node]
    (commit_result, commit_seconds) = _sync_time(
        torch, lambda: runtime.commit(graph_cache, commit_path)
    )
    commit_kv = _kv_record(
        torch, graph_cache, reference, before + len(commit_path),
        atol=atol, rtol=rtol,
    )
    capture = CaptureLastBlock(target)
    try:
        def run_continuation():
            with fp32_context():
                logits, hidden, start = _capture_forward(
                    target, capture, graph_cache, [feature], shared, device
                )
            return logits.detach().clone(), hidden.detach().clone(), start

        (continuation_output, continuation_seconds) = _sync_time(
            torch, run_continuation
        )
    finally:
        capture.close()
    continuation_logits, continuation_hidden, continuation_start = continuation_output
    if int(continuation_start) != before + len(commit_path):
        raise RuntimeError("backup continuation began at an unexpected cache position")
    continuation_node = _compare_node(
        torch, shared, continuation["logits"][2], continuation["hidden"][2],
        continuation_logits[-1], continuation_hidden[-1], expected_node=2,
        actual_node=0, atol=atol, rtol=rtol,
    )
    continuation_kv = _kv_record(
        torch, graph_cache, continuation, before + len(commit_path) + 1,
        atol=atol, rtol=rtol,
    )
    return dict(
        ranker_order=int(case["candidate_by_position"][position]["ranker_order"]),
        position=position,
        tree_node=node,
        forced=True,
        feature_token=int(feature),
        tree_verify_seconds=tree_verify_seconds,
        commit_seconds=commit_seconds,
        continuation_seconds=continuation_seconds,
        total_path_seconds=(tree_verify_seconds + commit_seconds + continuation_seconds),
        tree_node_comparison=node_comparison,
        feature_policy=feature_policy,
        commit_path=commit_path,
        commit_result_position=int(commit_result),
        commit_kv=commit_kv,
        continuation_node_comparison=continuation_node,
        continuation_kv=continuation_kv,
        tensor_metadata=dict(
            graph_logits=_tensor_metadata(graph_logits[node]),
            graph_hidden=_tensor_metadata(graph_hidden[node]),
            ordinary_logits=_tensor_metadata(reference["logits"][1]),
            ordinary_hidden=_tensor_metadata(reference["hidden"][1]),
            continuation_graph_logits=_tensor_metadata(continuation_logits[-1]),
            continuation_graph_hidden=_tensor_metadata(continuation_hidden[-1]),
            continuation_ordinary_logits=_tensor_metadata(continuation["logits"][2]),
            continuation_ordinary_hidden=_tensor_metadata(continuation["hidden"][2]),
        ),
        passed=bool(
            node_comparison["passed"]
            and feature_policy["support_equal"]
            and feature_policy["graph"]["finite"]
            and commit_kv["passed"]
            and continuation_node["passed"]
            and continuation_kv["passed"]
        ),
    )


def _run_ordinary_recovery(*, torch, target, cache, case, reference, shared,
                           device, CaptureLastBlock, fp32_context, atol, rtol):
    from verify_candidates import _strict_tensor_metrics

    cache.reset()
    capture = CaptureLastBlock(target)
    try:
        with fp32_context():
            _capture_forward(target, capture, cache, case["prompt"][:-1], shared, device)
            before = int(cache.get_pos())
            _capture_forward(target, capture, cache, [case["nodes"]["pending_f"]], shared, device)
        recovery_position = int(case["fallback_position"])

        def run_recovery():
            with fp32_context():
                logits, hidden, start = _capture_forward(
                    target, capture, cache, [recovery_position], shared, device
                )
            return logits.detach().clone(), hidden.detach().clone(), start

        (recovery_output, recovery_seconds) = _sync_time(torch, run_recovery)
        recovery_logits, recovery_hidden, recovery_start = recovery_output
        if int(recovery_start) != before + 1:
            raise RuntimeError("ordinary T1 recovery began at an unexpected position")
        recovery_node = _compare_node(
            torch, shared, reference["logits"][1], reference["hidden"][1],
            recovery_logits[-1], recovery_hidden[-1], expected_node=1,
            actual_node=0, atol=atol, rtol=rtol,
        )
        feature_start = int(case["feature_start"])
        feature_end = int(case["feature_end"])
        feature = _feature_argmax(shared, reference["logits"][1], feature_start, feature_end)
        continuation = reference["continuations"][feature]

        def run_continuation():
            with fp32_context():
                logits, hidden, start = _capture_forward(
                    target, capture, cache, [feature], shared, device
                )
            return logits.detach().clone(), hidden.detach().clone(), start

        (continuation_output, continuation_seconds) = _sync_time(torch, run_continuation)
        continuation_logits, continuation_hidden, continuation_start = continuation_output
        continuation_node = _compare_node(
            torch, shared, continuation["logits"][2], continuation["hidden"][2],
            continuation_logits[-1], continuation_hidden[-1], expected_node=2,
            actual_node=0, atol=atol, rtol=rtol,
        )
        continuation_kv = _kv_record(
            torch, cache, continuation, before + 3, atol=atol, rtol=rtol
        )
        return dict(
            position=recovery_position,
            ranker_order=int(case["candidate_by_position"][recovery_position]["ranker_order"]),
            path=[0, 1],
            feature_token=int(feature),
            recovery_seconds=recovery_seconds,
            continuation_seconds=continuation_seconds,
            recovery_start=int(recovery_start),
            continuation_start=int(continuation_start),
            recovery_node_comparison=recovery_node,
            continuation_node_comparison=continuation_node,
            continuation_kv=continuation_kv,
            tensor_metadata=dict(
                recovery_logits=_tensor_metadata(recovery_logits[-1]),
                recovery_hidden=_tensor_metadata(recovery_hidden[-1]),
                ordinary_logits=_tensor_metadata(reference["logits"][1]),
                ordinary_hidden=_tensor_metadata(reference["hidden"][1]),
                continuation_logits=_tensor_metadata(continuation_logits[-1]),
                continuation_hidden=_tensor_metadata(continuation_hidden[-1]),
            ),
            passed=bool(
                recovery_node["passed"] and continuation_node["passed"]
                and continuation_kv["passed"]
            ),
        )
    finally:
        capture.close()


def _build_references(*, torch, target, case, shared, device, capacity,
                      target_cache_factory, CaptureLastBlock, TargetTreeRuntime,
                      fp32_context, atol, rtol):
    from verify_candidates import _strict_tensor_metrics

    reference_cache = target_cache_factory(target, capacity)
    nodes = case["nodes"]
    primary = _ordinary_path(
        target, reference_cache, case["prompt"],
        [nodes["pending_f"], nodes["p1"], nodes["feature"], nodes["p2"]],
        shared, device, CaptureLastBlock, fp32_context, TargetTreeRuntime,
        case["rank_mask_runtime"], nodes["p1"], 1, case["ranks"],
    )
    references = {}
    for candidate in case["nodes"]["candidates"][1:MIN_RANKER_CANDIDATES]:
        position = int(candidate["position"])
        path = _ordinary_path(
            target, reference_cache, case["prompt"],
            [nodes["pending_f"], position], shared, device,
            CaptureLastBlock, fp32_context, TargetTreeRuntime,
            case["rank_mask_runtime"], position, 0, case["ranks"],
        )
        feature = _feature_argmax(
            shared, path["logits"][1], case["feature_start"], case["feature_end"]
        )
        continuation = _ordinary_path(
            target, reference_cache, case["prompt"],
            [nodes["pending_f"], position, feature], shared, device,
            CaptureLastBlock, fp32_context, TargetTreeRuntime,
            case["rank_mask_runtime"], position, 0, case["ranks"],
        )
        path["continuations"] = {int(feature): continuation}
        references[position] = path
    primary["tensor_metadata"] = _path_tensor_metadata(primary)
    for path in references.values():
        path["tensor_metadata"] = _path_tensor_metadata(path)
        path["continuation_tensor_metadata"] = {
            str(feature): _path_tensor_metadata(continuation)
            for feature, continuation in path["continuations"].items()
        }
    return primary, references


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--project", type=Path)
    parser.add_argument(
        "--source", type=Path,
        default=Path(__file__).resolve().parents[1] / "feature_tree_20261021",
    )
    parser.add_argument("--crops", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sources", type=int, default=1)
    parser.add_argument(
        "--cases", type=int, default=1,
        help="Number of selected source/prefix contexts; one keeps the short probe small",
    )
    parser.add_argument("--prefixes", default="256")
    parser.add_argument("--backup-counts", default="1,2,3,4")
    parser.add_argument("--optimized-recovery", action="store_true",
                        help="Use the existing Target graph runtime for ordinary short recovery calls")
    parser.add_argument("--cost-repeats", type=int, default=0,
                        help="Additional synchronized warm repetitions after path comparisons")
    parser.add_argument("--seed-start", type=int, default=58101)
    parser.add_argument("--cache-capacity", type=int, default=512)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--allow-constructed-fallback", action="store_true",
        help="Allow a legal non-Ranker fallback Position only for the path exercise",
    )
    return parser.parse_args()


def run_cpu_self_test():
    assert _parse_int_list("1,2,4", name="backup_counts", minimum=1, maximum=4) == [1, 2, 4]
    try:
        _parse_int_list("5", name="backup_counts", minimum=1, maximum=4)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range backup count was accepted")
    candidates = [
        dict(ranker_order=index, position=index, ranker_score=1.0 / index)
        for index in range(1, 7)
    ]
    assert candidates[4]["position"] == 5
    return dict(passed=True, backup_counts=[1, 2, 3, 4], forced_orders=[2, 3, 4, 5])


def main():
    args = _parse_args()
    if args.self_test:
        print(json.dumps(run_cpu_self_test(), sort_keys=True))
        return
    if args.project is None or args.crops is None or args.out is None:
        raise SystemExit("--project, --crops, and --out are required")
    try:
        backup_counts = _parse_int_list(
            args.backup_counts, name="backup_counts", minimum=1, maximum=MAX_BACKUP_COUNT
        )
        prefixes = _parse_int_list(args.prefixes, name="prefixes", minimum=2)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if len(set(backup_counts)) != len(backup_counts):
        raise SystemExit("backup_counts must be unique")
    if args.sources < 1 or args.cases < 1 or args.cases > args.sources:
        raise SystemExit("sources and cases must be positive, with cases <= sources")
    if args.cache_capacity < max(prefixes) + 32:
        raise SystemExit("cache_capacity is too small for the selected prefix")
    if args.atol < 0 or args.rtol < 0 or args.cost_repeats < 0:
        raise SystemExit("atol and rtol must be nonnegative")

    project = args.project.resolve()
    source = args.source.resolve()
    crops = args.crops.resolve()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit("--out must name a new directory")
    if not project.is_dir() or not source.is_dir() or not crops.is_file():
        raise SystemExit("project, source, or crops path does not exist")
    out.mkdir(parents=True)
    started = time.time()
    pipeline = dict(
        status="starting", command=sys.argv, started_at=started,
        model_load_seconds=None, case_done=0, width_done=0, error=None,
    )
    _write_json(out / "pipeline.json", pipeline)
    try:
        script_dir = Path(__file__).resolve().parent
        sys.path[:0] = [
            str(source), str(source / "reference"),
            str(source / "reference" / "runtime"), str(project), str(script_dir),
        ]
        os.chdir(project)
        import torch
        from threadpoolctl import threadpool_limits
        from method_adapters import build_context
        from dynamic_target_graph import DynamicTargetGraph
        from tree_runtime import TargetTreeRuntime
        from position_rescue_runtime import PositionRescueRuntime
        from bench_gaussiangpt_pf_block2_cow import CaptureLastBlock
        from draft_a.corpus_v2 import validate
        from draft_a.pf_ranker_eval import cache_for as target_cache_factory
        from draft_a.pf_feature_unified import _shared
        from draft_a.learning_eval import _fp32_runtime
        from verify_candidates import _select_crops
        from verify_candidates import _strict_tensor_metrics

        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the Position width probe")

        raw = torch.load(crops, map_location="cpu", weights_only=True)
        for record in raw:
            validate(record)
        selected = _select_crops(raw, max(prefixes), args.sources)
        if len(selected) < args.sources:
            raise ValueError(
                f"only {len(selected)} sources have crops for prefix {max(prefixes)}"
            )

        load_started = time.perf_counter()
        context = build_context(project, device="cuda")
        target = context["target"]
        load_seconds = time.perf_counter() - load_started
        pipeline["model_load_seconds"] = load_seconds
        shared = _shared()
        feature_start = int(context["p_vocab"])
        feature_end = feature_start + int(context["feature_vocab"])
        capacity = int(args.cache_capacity)
        plan = dict(
            project=str(project), source_snapshot=str(source), crops=str(crops),
            sources=int(args.sources), cases=int(args.cases), prefixes=prefixes,
            backup_counts=backup_counts, max_backup_count=MAX_BACKUP_COUNT,
            candidate_requirement=dict(
                ranker_candidates_required=MIN_RANKER_CANDIDATES,
                ranker_orders_used=[2, 3, 4, 5],
                constructed_fallback_allowed=bool(args.allow_constructed_fallback),
            ),
            interface=dict(
                runtime_constructor="PositionRescueRuntime(..., backup_count=k)",
                runtime_verify="verify(..., backup_positions=[...])",
                commit_paths="[0, 4 + backup_index]",
            ),
            tolerances=dict(
                policy_tv_max=POLICY_TV_LIMIT,
                hidden_and_kv_relative_l2_max=RELATIVE_L2_LIMIT,
                diagnostic_atol=args.atol, diagnostic_rtol=args.rtol,
            ),
            timing_scope=(
                "Model load is separate. Prefix fill is reported separately. "
                "Tree verify, commit, continuation, and ordinary T1 recovery use synchronized CUDA timing."
            ),
            optimized_recovery=bool(args.optimized_recovery),
            cost_repeats=args.cost_repeats,
            tensor_scope=(
                "Raw large K/V arrays are not written; shape, dtype, device, layout, "
                "element count, scalar differences, and policy support indices are retained."
            ),
        )
        _write_json(out / "plan.json", plan)
        _write_json(out / "case_selection.json", dict(selected_sources=[name for name, _ in selected]))
        pipeline["status"] = "running"
        _write_json(out / "pipeline.json", pipeline)

        numeric_cases = []
        skipped = []
        for source_index, (source_id, record) in enumerate(selected):
            for prefix in prefixes:
                if len(numeric_cases) >= int(args.cases):
                    break
                prompt = [int(context["eos"]), *map(int, record["tokens"][:prefix])]
                shared.validate_sequence(prompt, [], context["ranks"])
                if (len(prompt) - 1) % 2 or len(prompt) < 3:
                    skipped.append(dict(source=source_id, prefix=prefix, reason="incomplete prefix"))
                    continue
                nodes = _best_ranker_candidates(
                    shared, context["ranker"], context["ranks"], prompt,
                    minimum=MIN_RANKER_CANDIDATES,
                    allow_constructed=args.allow_constructed_fallback,
                )
                if nodes is None:
                    skipped.append(dict(
                        source=source_id, prefix=prefix,
                        reason="Ranker supplied fewer than six legal candidates",
                    ))
                    continue
                by_position = {
                    int(item["position"]): item for item in nodes["candidates"]
                }
                fallback_item = nodes["candidates"][1 + max(backup_counts)]
                nodes["candidate_by_position"] = by_position
                case = dict(
                    pair_id=f"width-{source_index:02d}-p{prefix:04d}",
                    source=source_id, prefix=prefix, prompt=prompt,
                    seed=int(args.seed_start), ranks=context["ranks"], nodes=nodes,
                    fallback_position=int(fallback_item["position"]),
                    feature_start=feature_start, feature_end=feature_end,
                    candidate_by_position=by_position,
                )
                numeric_cases.append(case)
            if len(numeric_cases) >= int(args.cases):
                break
        if not numeric_cases:
            raise ValueError("no context has the six legal Ranker candidates required by this probe")
        _write_json(out / "numeric_selection.json", dict(
            selected=[dict(
                pair_id=case["pair_id"], source=case["source"], prefix=case["prefix"],
                prompt=case["prompt"], nodes={
                    key: value for key, value in case["nodes"].items()
                    if key != "candidate_by_position"
                }, fallback_position=case["fallback_position"],
            ) for case in numeric_cases], skipped=skipped,
        ))

        linear_runtime = DynamicTargetGraph(
            target, cache_capacity_buckets=(capacity,), record_outputs=False
        )
        graph_cache = target_cache_factory(target, capacity)
        reference_cache = target_cache_factory(target, capacity)
        rank_mask_runtime = PositionRescueRuntime(
            target, linear_runtime, graph=True, backup_count=1
        )
        for case in numeric_cases:
            case["rank_mask_runtime"] = rank_mask_runtime

        records = []
        with (threadpool_limits(limits=1), torch.inference_mode(),
              _fp32_runtime(torch), torch.autocast("cuda", enabled=False),
              linear_runtime.attention_context() if args.optimized_recovery else nullcontext()):
            for case_index, case in enumerate(numeric_cases):
                primary, references = _build_references(
                    torch=torch, target=target, case=case, shared=shared,
                    device=context["device"], capacity=capacity,
                    target_cache_factory=target_cache_factory,
                    CaptureLastBlock=CaptureLastBlock,
                    TargetTreeRuntime=TargetTreeRuntime,
                    fp32_context=lambda: _fp32_runtime(torch),
                    atol=args.atol, rtol=args.rtol,
                )
                case["references"] = references
                width_rows = []
                for backup_count in backup_counts:
                    backups = [
                        int(item["position"])
                        for item in case["nodes"]["candidates"][1:1 + backup_count]
                    ]
                    runtime = PositionRescueRuntime(
                        target, linear_runtime, graph=True, backup_count=backup_count
                    )
                    before_graph_count = int(runtime.stats().get("graph_count", 0))
                    _prefill(
                        target, graph_cache, case["prompt"], shared,
                        CaptureLastBlock, lambda: _fp32_runtime(torch), context["device"],
                    )
                    before = int(graph_cache.get_pos())
                    (cold_output, cold_seconds) = _sync_time(
                        torch,
                        lambda: runtime.verify(
                            graph_cache, case["nodes"]["pending_f"], case["nodes"]["p1"],
                            [case["nodes"]["feature"]], [case["nodes"]["p2"]],
                            backup_positions=backups,
                        ),
                    )
                    cold_logits, cold_hidden = cold_output
                    cold_logits = cold_logits.detach().clone()
                    cold_hidden = cold_hidden.detach().clone()
                    if cold_logits.ndim == 3:
                        cold_logits = cold_logits[0]
                    if cold_hidden.ndim == 3:
                        cold_hidden = cold_hidden[0]
                    graph_count_after = int(runtime.stats().get("graph_count", 0))
                    cold_events = runtime.stats().get("graph_events", [])[before_graph_count:]
                    primary_cold, backup_cold = _graph_node_rows(
                        torch, shared, cold_logits, cold_hidden, primary,
                        references, case["nodes"]["candidates"],
                        atol=args.atol, rtol=args.rtol,
                    )
                    runtime.commit(graph_cache, [0, 4])

                    quality = _quality_curve(
                        shared, cold_logits[0], case["nodes"]["candidates"],
                        backup_count, p_vocab=context["p_vocab"],
                    )
                    fallback_reference = references[int(case["fallback_position"])]
                    ordinary_recovery = _run_ordinary_recovery(
                        torch=torch, target=target, cache=reference_cache,
                        case=case, reference=fallback_reference, shared=shared,
                        device=context["device"], CaptureLastBlock=CaptureLastBlock,
                        fp32_context=lambda: _fp32_runtime(torch),
                        atol=args.atol, rtol=args.rtol,
                    )
                    forced_rows = []
                    for backup_index in range(backup_count):
                        forced_rows.append(_run_forced_backup(
                            torch=torch, target=target, runtime=runtime,
                            graph_cache=graph_cache, case=case, backups=backups,
                            backup_index=backup_index, primary=primary,
                            references=references, shared=shared,
                            device=context["device"], CaptureLastBlock=CaptureLastBlock,
                            TargetTreeRuntime=TargetTreeRuntime,
                            fp32_context=lambda: _fp32_runtime(torch),
                            feature_start=feature_start, feature_end=feature_end,
                            atol=args.atol, rtol=args.rtol,
                        ))
                    passed = bool(
                        all(item["passed"] for item in primary_cold + backup_cold)
                        and all(item["passed"] for item in forced_rows)
                        and ordinary_recovery["passed"]
                    )
                    row = dict(
                        pair_id=case["pair_id"], source=case["source"],
                        prefix=case["prefix"], prompt=case["prompt"],
                        backup_count=int(backup_count),
                        backup_positions=backups,
                        backup_nodes=[4 + index for index in range(backup_count)],
                        fallback_position=int(case["fallback_position"]),
                        ranker_candidates=case["nodes"]["candidates"],
                        quality=quality,
                        cold_graph=dict(
                            seconds=cold_seconds,
                            graph_count_before=before_graph_count,
                            graph_count_after=graph_count_after,
                            new_graph_events=cold_events,
                            primary_nodes=primary_cold,
                            backup_nodes=backup_cold,
                        ),
                        warm_forced_backups=forced_rows,
                        ordinary_t1_recovery=ordinary_recovery,
                        tensor_metadata=dict(
                            primary_reference=primary["tensor_metadata"],
                            backup_references={
                                str(position): reference["tensor_metadata"]
                                for position, reference in references.items()
                            },
                            cold_graph_logits=_tensor_metadata(cold_logits),
                            cold_graph_hidden=_tensor_metadata(cold_hidden),
                        ),
                        passed=passed,
                    )
                    if args.cost_repeats:
                        _prefill(target, graph_cache, case["prompt"], shared,
                                 CaptureLastBlock, lambda: _fp32_runtime(torch), context["device"])
                        w1_cost_runtime = TargetTreeRuntime(target, linear_runtime, graph=True, max_width=1)
                        _warm_tree_costs(torch, w1_cost_runtime, graph_cache, case, [], 1)
                        row["w1_warm_cost"] = _warm_tree_costs(torch, w1_cost_runtime, graph_cache, case, [], args.cost_repeats)
                        row["backup_warm_cost"] = _warm_tree_costs(torch, runtime, graph_cache, case, backups, args.cost_repeats)
                        _prefill(target, reference_cache, case["prompt"], shared,
                                 CaptureLastBlock, lambda: _fp32_runtime(torch), context["device"])
                        pending_capture = CaptureLastBlock(target)
                        try:
                            _capture_forward(target, pending_capture, reference_cache,
                                             [case["nodes"]["pending_f"]], shared, context["device"])
                        finally:
                            pending_capture.close()
                        _warm_recovery_costs(torch, target, reference_cache, case, shared,
                                             context["device"], CaptureLastBlock, 1)
                        row["recovery_warm_cost"] = _warm_recovery_costs(
                            torch, target, reference_cache, case, shared, context["device"],
                            CaptureLastBlock, args.cost_repeats)
                    _append_jsonl(out / "width_records.jsonl", row)
                    records.append(row)
                    pipeline["width_done"] = len(records)
                    _write_json(out / "pipeline.json", pipeline)
                    if not passed:
                        raise AssertionError(
                            f"Position width comparison failed for {case['pair_id']} W{backup_count}"
                        )
                    del runtime
                    gc.collect()
                pipeline["case_done"] = case_index + 1
                _write_json(out / "pipeline.json", pipeline)

        summary = dict(
            status="passed" if all(row["passed"] for row in records) else "failed",
            model_load_seconds=load_seconds,
            case_count=len(numeric_cases), width_record_count=len(records),
            backup_counts=backup_counts, selected_sources=sorted({row["source"] for row in records}),
            max_policy_tv=max((
                item["tree_node_comparison"]["logits"]["policy_tv"]
                for row in records
                for item in row["warm_forced_backups"]
                if item["tree_node_comparison"]["logits"]["policy_tv"] is not None
            ), default=None),
            max_hidden_relative_l2=max((
                item["tree_node_comparison"]["hidden"]["relative_l2"]
                for row in records for item in row["warm_forced_backups"]
                if item["tree_node_comparison"]["hidden"]["relative_l2"] is not None
            ), default=None),
            all_passed=all(row["passed"] for row in records),
            linear_runtime_stats=linear_runtime.stats(),
            files=dict(width_records="width_records.jsonl", plan="plan.json"),
        )
        _write_json(out / "summary.json", summary)
        pipeline.update(status="complete", finished_at=time.time(),
                        elapsed_seconds=time.time() - started)
        _write_json(out / "pipeline.json", pipeline)
        print(json.dumps(dict(
            status="passed", output=str(out), cases=len(numeric_cases),
            width_records=len(records), model_load_seconds=load_seconds,
            elapsed_seconds=time.time() - started,
        ), ensure_ascii=False), flush=True)
    except Exception as error:
        pipeline.update(status="failed", error=f"{type(error).__name__}: {error}",
                        finished_at=time.time(), elapsed_seconds=time.time() - started)
        _write_json(out / "pipeline.json", pipeline)
        raise


if __name__ == "__main__":
    main()
