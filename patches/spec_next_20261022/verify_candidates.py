#!/usr/bin/env python3
"""Verify Position-rescue tree outputs and repeated Feature-policy reuse."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


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


def _source_id(record):
    value = record.get("scene") or record.get("source") or record.get("source_id")
    if value:
        return str(value)
    crop_path = Path(str(record.get("path", "")))
    return crop_path.parents[1].name if len(crop_path.parts) >= 3 else "unknown"


def _select_crops(records, max_prefix: int, source_limit: int):
    by_source = {}
    for record in records:
        by_source.setdefault(_source_id(record), []).append(record)
    selected = []
    for source in sorted(by_source):
        record = next((item for item in by_source[source]
                       if len(item["tokens"]) >= int(max_prefix) + 4), None)
        if record is not None:
            selected.append((source, record))
        if len(selected) >= int(source_limit):
            break
    return selected


def _best_two(scores):
    scores = [float(value) for value in scores]
    if len(scores) < 2 or any(not math.isfinite(value) for value in scores):
        raise ValueError("at least two finite Ranker scores are required")
    ordered = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return ordered[0], ordered[1]


def _semantic_counts(stats):
    keys = (
        "rounds", "proposed", "verified", "accepted", "accepted_lengths",
        "round_generated_lengths", "generated_tokens", "eos",
        "position_fallbacks", "k1_steps", "position_proposed",
        "position_verified", "position_accepted", "feature_proposed",
        "feature_verified", "feature_accepted", "feature_attempts",
        "feature_branch_hits", "feature_all_rejected",
        "feature_branch_index_counts", "eos_proposed", "eos_verified",
        "eos_accepted",
    )
    return {key: stats[key] for key in keys if key in stats}


def _count_differences(left, right):
    keys = sorted(set(left) | set(right))
    missing = [key for key in keys if key not in left or key not in right]
    differing = [key for key in keys
                 if key in left and key in right and left[key] != right[key]]
    return dict(equal=not missing and not differing, missing=missing,
                differing=differing)


def _max_finite_abs(torch, left, right):
    finite = torch.isfinite(left) & torch.isfinite(right)
    if not bool(finite.any().item()):
        return 0.0
    return float((left[finite] - right[finite]).abs().max().item())


def _strict_tensor_metrics(torch, reference, candidate, *, atol, rtol):
    same_shape = tuple(reference.shape) == tuple(candidate.shape)
    if not same_shape:
        return dict(
            shape_equal=False,
            finite_mask_equal=False,
            all_finite=False,
            relative_l2=None,
            max_abs=None,
            reference_max_abs=None,
            max_abs_over_reference_max_abs=None,
            allclose=False,
        )
    reference_finite = torch.isfinite(reference)
    candidate_finite = torch.isfinite(candidate)
    finite_mask_equal = bool(torch.equal(reference_finite, candidate_finite))
    all_finite = bool(reference_finite.all().item() and candidate_finite.all().item())
    allclose = bool(torch.allclose(reference, candidate, atol=float(atol),
                                   rtol=float(rtol), equal_nan=False))
    if all_finite:
        difference = candidate - reference
        max_abs = float(difference.abs().max().item())
        reference_max_abs = float(reference.abs().max().item())
        reference_l2 = float(torch.linalg.vector_norm(reference).item())
        difference_l2 = float(torch.linalg.vector_norm(difference).item())
        relative_l2 = (difference_l2 / reference_l2
                       if reference_l2 > 0.0
                       else (0.0 if difference_l2 == 0.0 else float("inf")))
        max_abs_relative = (max_abs / reference_max_abs
                            if reference_max_abs > 0.0
                            else (0.0 if max_abs == 0.0 else float("inf")))
    else:
        max_abs = _max_finite_abs(torch, reference, candidate)
        reference_finite_values = reference[reference_finite]
        reference_max_abs = (float(reference_finite_values.abs().max().item())
                             if reference_finite_values.numel() else None)
        relative_l2 = None
        max_abs_relative = None
    return dict(
        shape_equal=True,
        finite_mask_equal=finite_mask_equal,
        all_finite=all_finite,
        relative_l2=relative_l2,
        max_abs=max_abs,
        reference_max_abs=reference_max_abs,
        max_abs_over_reference_max_abs=max_abs_relative,
        allclose=allclose,
    )


def _tensor_pair_metrics(torch, shared, left, right, *, atol, rtol):
    if tuple(left.shape) != tuple(right.shape):
        return dict(shape_equal=False, shape_left=list(left.shape),
                    shape_right=list(right.shape), finite_mask_equal=False,
                    allclose=False, max_abs=None, max_row_l2_relative=None,
                    policy_tv=None, policy_support_equal=False)
    left_finite = torch.isfinite(left)
    right_finite = torch.isfinite(right)
    finite_equal = bool(torch.equal(left_finite, right_finite))
    all_finite = bool(left_finite.all().item() and right_finite.all().item())
    allclose = bool(torch.allclose(left, right, atol=float(atol), rtol=float(rtol),
                                   equal_nan=False))
    max_abs = _max_finite_abs(torch, left, right)
    if left.ndim == 1:
        left_rows, right_rows = left.unsqueeze(0), right.unsqueeze(0)
    else:
        left_rows, right_rows = left.reshape(-1, left.shape[-1]), right.reshape(-1, right.shape[-1])
    if all_finite:
        numerator = torch.linalg.vector_norm(left_rows - right_rows, dim=-1)
        denominator = torch.linalg.vector_norm(left_rows, dim=-1).clamp_min(1e-12)
        max_row_l2_relative = float((numerator / denominator).max().item())
    else:
        max_row_l2_relative = None
    tv_values = []
    support_equal = True
    try:
        for left_row, right_row in zip(left_rows, right_rows):
            left_policy = shared.policy(left_row).float()
            right_policy = shared.policy(right_row).float()
            if not bool(torch.isfinite(left_policy).all().item() and
                        torch.isfinite(right_policy).all().item()):
                tv_values.append(float("inf"))
                support_equal = False
                continue
            support_equal = support_equal and bool(torch.equal(
                left_policy > 0, right_policy > 0
            ))
            tv_values.append(float((left_policy - right_policy).abs().sum().item() * 0.5))
        policy_tv = max(tv_values, default=0.0)
    except Exception:
        policy_tv = None
        support_equal = False
    return dict(
        shape_equal=True,
        finite_mask_equal=finite_equal,
        all_finite=all_finite,
        allclose=allclose,
        max_abs=max_abs,
        max_row_l2_relative=max_row_l2_relative,
        policy_tv=policy_tv,
        policy_support_equal=support_equal,
    )


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--source", type=Path,
                        default=Path(__file__).resolve().parents[1] / "feature_tree_20261021",
                        help="Existing feature_tree_20261021 source_run04")
    parser.add_argument("--crops", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sources", type=int, default=10,
                        help="Number of sources for full decoding comparisons")
    parser.add_argument("--numeric-sources", type=int, default=3,
                        help="At least this many sources are used for tree-vs-linear checks")
    parser.add_argument("--prefixes", default="64,256")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=48101)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--cache-capacity", type=int, default=512)
    parser.add_argument("--reuse-weight", type=float, default=0.25)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser


def _choose_tree_tokens(shared, ranker, ranks, prompt):
    from bench_gaussiangpt_pf_block2_cow import fork_position_state

    state = shared.position_state(prompt, ranks)
    previous = int(prompt[-2])
    proposed = state.propose(previous)
    if proposed is None:
        return None
    position_ids, features, _guess, _invalid = proposed
    position_ids = [int(token) for token in position_ids]
    if len(position_ids) < 2:
        return None
    scores = ranker.predict_proba(features)[:, 1]
    if len(position_ids) != len(scores):
        raise ValueError("Ranker score count differs from the Position candidates")
    best_index, backup_index = _best_two(scores)
    p1 = position_ids[best_index]
    p1b = position_ids[backup_index]
    if p1 == p1b or int(ranks[p1]) <= int(ranks[previous]) or int(ranks[p1b]) <= int(ranks[previous]):
        raise ValueError("Ranker returned an invalid backup Position")
    shadow = fork_position_state(state)
    shadow.finalize(p1)
    second = shadow.propose(p1)
    if second is None:
        return None
    p2_ids, p2_features, _p2_guess, _p2_invalid = second
    p2_ids = [int(token) for token in p2_ids]
    if not p2_ids:
        return None
    p2_scores = ranker.predict_proba(p2_features)[:, 1]
    if len(p2_ids) != len(p2_scores):
        raise ValueError("Ranker score count differs from second Position candidates")
    p2_index = max(range(len(p2_scores)), key=lambda index: float(p2_scores[index]))
    p2 = p2_ids[p2_index]
    if int(ranks[p2]) <= int(ranks[p1]):
        raise ValueError("Ranker returned a non-increasing second Position")
    return dict(
        previous=previous,
        pending_f=int(prompt[-1]),
        p1=p1,
        p1b=p1b,
        feature=int(prompt[-1]),
        p2=p2,
        candidate_count=len(position_ids),
    )


def _capture_forward(target, capture, cache, tokens, shared, device):
    start = int(cache.get_pos())
    kinds = shared.typed_kinds(start, len(tokens))
    logits, hidden = capture.run(target, cache, [int(token) for token in tokens],
                                kinds, device)
    if int(cache.get_pos()) != start + len(tokens):
        raise RuntimeError("ordinary Target call advanced an unexpected number of tokens")
    if logits.ndim == 3:
        logits = logits[0]
    if hidden.ndim == 3:
        hidden = hidden[0]
    return logits, hidden, start


def _ordinary_path(target, cache, prompt, continuation, shared, device,
                   CaptureLastBlock, fp32_context, *, rank_mask_runtime,
                   rank_mask_p1, rank_mask_width, ranks, TargetTreeRuntime):
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
            TargetTreeRuntime._rank_mask(
                rank_mask_runtime, logits, cache, before,
                int(rank_mask_p1), int(rank_mask_width),
            )
            return dict(
                before=before,
                after=int(cache.get_pos()),
                logits=logits.detach().clone(),
                hidden=hidden.detach().clone(),
                kv_k=cache.k_cache[:, :, :cache.get_pos()].detach().clone(),
                kv_v=cache.v_cache[:, :, :cache.get_pos()].detach().clone(),
                tokens=cache.token_cache[:, :cache.get_pos()].detach().clone(),
                ranks_validated=bool(len(ranks) > 0),
            )
    finally:
        capture.close()


def _node_comparison(torch, shared, expected_logits, expected_hidden,
                     actual_logits, actual_hidden, *, atol, rtol,
                     expected_node, actual_node):
    logit_metrics = _tensor_pair_metrics(
        torch, shared, expected_logits, actual_logits, atol=atol, rtol=rtol
    )
    hidden_metrics = _strict_tensor_metrics(
        torch, expected_hidden, actual_hidden, atol=atol, rtol=rtol
    )
    passed = (
        logit_metrics["shape_equal"]
        and logit_metrics["finite_mask_equal"]
        and logit_metrics["policy_support_equal"]
        and logit_metrics["policy_tv"] is not None
        and logit_metrics["policy_tv"] <= 1e-5
        and hidden_metrics["shape_equal"]
        and hidden_metrics["finite_mask_equal"]
        and hidden_metrics["all_finite"]
        and hidden_metrics["relative_l2"] is not None
        and hidden_metrics["relative_l2"] <= 2e-6
        and hidden_metrics["max_abs_over_reference_max_abs"] is not None
        and hidden_metrics["max_abs_over_reference_max_abs"] <= 2e-6
    )
    return dict(
        expected_node=int(expected_node), actual_node=int(actual_node),
        logits=logit_metrics,
        hidden=dict(**hidden_metrics,
                    row_l2_relative=hidden_metrics["relative_l2"]),
        passed=bool(passed),
    )


def _run_numeric_context(*, row, tokens, target, shared, rescue_runtime,
                         graph_cache, primary_cache, backup_cache,
                         target_cache_factory, CaptureLastBlock,
                         TargetTreeRuntime, fp32_context, torch, device, atol, rtol):
    graph_cache.reset()
    primary = _ordinary_path(
        target, primary_cache, row["prompt"],
        [tokens["pending_f"], tokens["p1"], tokens["feature"], tokens["p2"]],
        shared, device, CaptureLastBlock, fp32_context,
        rank_mask_runtime=rescue_runtime, rank_mask_p1=tokens["p1"],
        rank_mask_width=1, ranks=row["ranks"], TargetTreeRuntime=TargetTreeRuntime,
    )
    alternate = _ordinary_path(
        target, backup_cache, row["prompt"],
        [tokens["pending_f"], tokens["p1b"]],
        shared, device, CaptureLastBlock, fp32_context,
        rank_mask_runtime=rescue_runtime, rank_mask_p1=tokens["p1b"],
        rank_mask_width=0, ranks=row["ranks"], TargetTreeRuntime=TargetTreeRuntime,
    )
    before = int(graph_cache.get_pos())
    capture = CaptureLastBlock(target)
    try:
        with fp32_context(), torch.inference_mode():
            _capture_forward(target, capture, graph_cache, row["prompt"][:-1],
                             shared, device)
            before = int(graph_cache.get_pos())
            if before != primary["before"] or before != alternate["before"]:
                raise RuntimeError("independent caches reached different prefix positions")
            graph_logits, graph_hidden = rescue_runtime.verify(
                graph_cache,
                tokens["pending_f"], tokens["p1"],
                [tokens["feature"]], [tokens["p2"]],
                backup_position=tokens["p1b"],
            )
            graph_logits = graph_logits.detach().clone()
            graph_hidden = graph_hidden.detach().clone()
            if graph_logits.ndim == 3:
                graph_logits = graph_logits[0]
            if graph_hidden.ndim == 3:
                graph_hidden = graph_hidden[0]
            if graph_logits.size(0) != 5 or graph_hidden.size(0) != 5:
                raise RuntimeError("rescue graph did not return all five nodes")
            comparisons = []
            for index in range(4):
                comparisons.append(_node_comparison(
                    torch, shared, primary["logits"][index], primary["hidden"][index],
                    graph_logits[index], graph_hidden[index], atol=atol, rtol=rtol,
                    expected_node=index, actual_node=index,
                ))
            comparisons.append(_node_comparison(
                torch, shared, alternate["logits"][1], alternate["hidden"][1],
                graph_logits[4], graph_hidden[4], atol=atol, rtol=rtol,
                expected_node=1, actual_node=4,
            ))
            root_vs_alternate = _node_comparison(
                torch, shared, alternate["logits"][0], alternate["hidden"][0],
                graph_logits[0], graph_hidden[0], atol=atol, rtol=rtol,
                expected_node=0, actual_node=0,
            )
            cache_after = rescue_runtime.commit(graph_cache, [0, 4])
            expected_end = before + 2
            if cache_after != expected_end or int(graph_cache.get_pos()) != expected_end:
                raise RuntimeError("commit [0,4] did not retain the two-token alternate path")
            kv_k = graph_cache.k_cache[:, :, :expected_end]
            kv_v = graph_cache.v_cache[:, :, :expected_end]
            token_ids = graph_cache.token_cache[:, :expected_end]
            reference_tokens = alternate["tokens"]
            token_equal = bool(torch.equal(token_ids, reference_tokens))
            k_metrics = _strict_tensor_metrics(
                torch, alternate["kv_k"], kv_k, atol=atol, rtol=rtol
            )
            v_metrics = _strict_tensor_metrics(
                torch, alternate["kv_v"], kv_v, atol=atol, rtol=rtol
            )
            k_allclose = k_metrics["allclose"]
            v_allclose = v_metrics["allclose"]
            kv = dict(
                logical_position=int(graph_cache.get_pos()),
                expected_position=int(expected_end),
                token_cache_equal=token_equal,
                k_allclose=k_allclose,
                v_allclose=v_allclose,
                k_max_abs=k_metrics["max_abs"],
                v_max_abs=v_metrics["max_abs"],
                k_metrics=k_metrics,
                v_metrics=v_metrics,
            )
            passed = (all(item["passed"] for item in comparisons)
                      and root_vs_alternate["passed"] and token_equal
                      and k_metrics["all_finite"] and v_metrics["all_finite"]
                      and k_metrics["finite_mask_equal"] and v_metrics["finite_mask_equal"]
                      and k_metrics["relative_l2"] is not None
                      and k_metrics["relative_l2"] <= 2e-6
                      and v_metrics["relative_l2"] is not None
                      and v_metrics["relative_l2"] <= 2e-6
                      and k_metrics["max_abs_over_reference_max_abs"] is not None
                      and k_metrics["max_abs_over_reference_max_abs"] <= 2e-6
                      and v_metrics["max_abs_over_reference_max_abs"] is not None
                      and v_metrics["max_abs_over_reference_max_abs"] <= 2e-6)
            policy_tv_values = [
                item["logits"]["policy_tv"] for item in comparisons
            ] + [root_vs_alternate["logits"]["policy_tv"]]
            finite_policy_values = [value for value in policy_tv_values
                                    if value is not None]
            return dict(
                pair_id=row["pair_id"], source=row["source"],
                prefix=row["prefix"], prompt=row["prompt"], seed=row["seed"],
                tree_tokens=tokens, before=before, node_comparisons=comparisons,
                root_vs_alternate=root_vs_alternate, commit_0_4=kv,
                max_policy_tv=max(finite_policy_values, default=None),
                all_node_hidden_allclose=all(item["hidden"]["allclose"]
                                             for item in comparisons),
                passed=bool(passed),
            )
    finally:
        capture.close()


def _build_decode_arm(name, context, adapter, *, DynamicTargetGraph,
                      TargetTreeRuntime, PositionRescueRuntime, capacity):
    runtime = DynamicTargetGraph(
        context["target"], cache_capacity_buckets=(int(capacity),)
    )
    from proposal_adapters import build_proposal
    proposal = build_proposal(adapter, verify=False)
    tree = TargetTreeRuntime(context["target"], runtime, graph=True, max_width=1)
    arm = dict(name=name, runtime=runtime, proposal=proposal, tree=tree)
    if name == "rescue":
        arm["rescue"] = PositionRescueRuntime(context["target"], runtime, graph=True)
    return arm


def _run_decode_arm(arm, prompt, seed, budget, context, *, torch,
                    target_cache_factory, validate_sequence,
                    decode_w1, decode_rescue, decode_reuse, reuse_weight=None):
    runtime = arm["runtime"]
    adapter = arm["adapter"]
    factory = lambda core, capacity: runtime.cache_for(
        core, capacity, factory=target_cache_factory
    )
    arm["tree"].stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with arm["proposal"].context(), runtime.attention_context():
        if arm["name"] == "w1":
            stats, suffix = decode_w1(
                context["target"], adapter.model, context["ranker"],
                context["ranks"], prompt, budget, seed,
                tree_runtime=arm["tree"], width=1, cache_factory=factory,
            )
        elif arm["name"] == "rescue":
            stats, suffix = decode_rescue(
                context["target"], adapter.model, context["ranker"],
                context["ranks"], prompt, budget, seed,
                tree_runtime=arm["tree"], rescue_runtime=arm["rescue"],
                width=1, cache_factory=factory,
            )
        elif arm["name"] == "reuse0":
            stats, suffix = decode_reuse(
                context["target"], adapter.model, context["ranker"],
                context["ranks"], prompt, budget, seed,
                tree_runtime=arm["tree"], width=1, cache_factory=factory,
                reuse_weight=0.0,
            )
        elif arm["name"] == "reuse25":
            stats, suffix = decode_reuse(
                context["target"], adapter.model, context["ranker"],
                context["ranks"], prompt, budget, seed,
                tree_runtime=arm["tree"], width=1, cache_factory=factory,
                reuse_weight=float(reuse_weight),
            )
        else:
            raise ValueError(f"unknown decode arm {arm['name']}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    suffix = [int(token) for token in suffix]
    validate_sequence(prompt, suffix, context["ranks"])
    if len(suffix) > int(budget):
        raise RuntimeError("decoder returned more tokens than its budget")
    counters = _semantic_counts(stats)
    row = dict(
        method=arm["name"], output_tokens=suffix,
        generated_tokens=len(suffix), eos=bool(suffix and suffix[-1] == int(context["eos"])),
        counters=counters,
        recovery_calls=int(stats.get("recovery_calls", 0)),
        target_calls=int(stats.get("target_calls", 0)),
        target_input_tokens=int(stats.get("target_input_tokens", 0)),
        target_tree_nodes=int(stats.get("target_tree_nodes", 0)),
        position_rescue_hits=int(stats.get("position_rescue_hits", 0)),
        position_backup_rejections=int(stats.get("position_backup_rejections", 0)),
        elapsed_seconds=elapsed,
        sequence_validated=True,
        runtime=runtime.stats(),
        tree_runtime=arm["tree"].stats(),
        rescue_runtime=(arm["rescue"].stats() if "rescue" in arm else None),
        proposal_runtime=arm["proposal"].stats(),
    )
    return row


def run_cpu_self_test():
    best, backup = _best_two([0.9, 0.9, 0.4])
    assert (best, backup) == (0, 1)
    left = {"position_accepted": 2, "feature_accepted": 1}
    right = {"position_accepted": 2, "feature_accepted": 1}
    assert _count_differences(left, right)["equal"]
    assert not _count_differences(left, {"position_accepted": 2})["equal"]
    records = [
        {"scene": "a", "tokens": list(range(80))},
        {"scene": "b", "tokens": list(range(40))},
    ]
    selected = _select_crops(records, 64, 2)
    assert [name for name, _record in selected] == ["a"]
    return dict(passed=True, top_two=[best, backup], selected_sources=[x[0] for x in selected])


def main():
    parser = _parser()
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(run_cpu_self_test(), sort_keys=True))
        return
    if args.project is None or args.crops is None or args.out is None:
        parser.error("--project, --crops, and --out are required")
    prefixes = [int(item.strip()) for item in args.prefixes.split(",") if item.strip()]
    if (args.sources < 1 or args.numeric_sources < 3 or args.seeds < 1
            or args.budget < 4 or args.budget % 2 or not prefixes
            or any(item < 2 or item % 2 for item in prefixes)
            or not math.isfinite(args.reuse_weight)
            or not 0.0 <= args.reuse_weight <= 1.0
            or args.atol < 0 or args.rtol < 0
            or args.cache_capacity < max(prefixes) + args.budget + 8):
        parser.error("invalid source, prefix, budget, capacity, or tolerance values")

    project, source, crops, out = (
        args.project.resolve(), args.source.resolve(), args.crops.resolve(), args.out.resolve()
    )
    if out.exists():
        parser.error("--out must name a new directory")
    if not project.is_dir() or not source.is_dir() or not crops.is_file():
        parser.error("project, source, or crops path does not exist")
    out.mkdir(parents=True)
    pipeline = dict(status="starting", command=sys.argv, started_at=time.time(),
                    numeric_done=0, decode_done=0, error=None)
    _write_json(out / "pipeline.json", pipeline)
    try:
        script_dir = Path(__file__).resolve().parent
        sys.path[:0] = [str(source), str(source / "reference"),
                        str(source / "reference" / "runtime"), str(project),
                        str(script_dir)]
        os.chdir(project)
        import torch
        from threadpoolctl import threadpool_limits
        from method_adapters import build_context, load_method
        from dynamic_target_graph import DynamicTargetGraph
        from tree_runtime import TargetTreeRuntime
        from tree_decoder import decode as decode_w1
        from position_rescue_decoder import decode as decode_rescue
        from feature_reuse_decoder import decode as decode_reuse
        from position_rescue_runtime import PositionRescueRuntime
        from bench_gaussiangpt_pf_block2_cow import CaptureLastBlock
        from draft_a.corpus_v2 import validate
        from draft_a.pf_ranker_eval import cache_for as target_cache_factory
        from draft_a.pf_ranker_eval import validate_sequence
        from draft_a.pf_feature_unified import _shared
        from draft_a.learning_eval import _fp32_runtime

        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Target graph verification")
        raw = torch.load(crops, map_location="cpu", weights_only=True)
        for record in raw:
            validate(record)
        max_prefix = max(prefixes)
        selected = _select_crops(raw, max_prefix, args.sources)
        if len(selected) < args.sources:
            raise ValueError(f"only {len(selected)} sources have crops for prefix {max_prefix}")
        context = build_context(project, device="cuda")
        target = context["target"]
        shared = _shared()
        adapter = load_method("pf_mlp", context)
        capacity = int(args.cache_capacity)
        plan = dict(
            project=str(project), source_snapshot=str(source), crops=str(crops),
            selected_sources=[name for name, _record in selected],
            prefixes=prefixes, numeric_sources=args.numeric_sources,
            full_decode_sources=args.sources, seeds=args.seeds,
            seed_start=args.seed_start, budget=args.budget,
            cache_capacity=capacity, reuse_weight=args.reuse_weight,
            tolerances=dict(
                logits_policy_tv_max=1e-5,
                hidden_and_kv=dict(
                    all_finite=True,
                    overall_relative_l2_max=2e-6,
                    max_abs_over_reference_max_abs_max=2e-6,
                    elementwise_allclose="diagnostic only",
                    diagnostic_atol=args.atol,
                    diagnostic_rtol=args.rtol,
                ),
            ),
            numeric_scope=("five-node graph rows 0..3 versus an ordinary four-token Target call; row 4 versus an ordinary two-token sibling call; commit [0,4] K/V and token state comparison"),
            numeric_artifact_scope=("only scalar comparison metrics are retained for large K/V arrays; full K/V tensors are not saved, so this run does not support an independent raw-tensor recomputation"),
            decode_scope=("W1, Position rescue, reuse weight 0, and the requested reuse weight twice; all outputs are sequence-validated"),
        )
        _write_json(out / "plan.json", plan)
        _write_json(out / "status.json", dict(status="running", plan=plan))
        torch_device = context["device"]
        linear_runtime = DynamicTargetGraph(
            target, cache_capacity_buckets=(capacity,), record_outputs=False
        )
        rescue_runtime = PositionRescueRuntime(target, linear_runtime, graph=True)
        cache_slots = {}

        def get_numeric_cache(name):
            cache = cache_slots.get(name)
            if cache is None:
                cache = target_cache_factory(target, capacity)
                cache_slots[name] = cache
            else:
                cache.reset()
            return cache

        numeric_rows = []
        seed = int(args.seed_start)
        numeric_cases = []
        numeric_skips = []
        for source_index, (source_id, record) in enumerate(selected):
            for prefix in prefixes:
                prompt = [int(context["eos"]), *map(int, record["tokens"][:prefix])]
                shared.validate_sequence(prompt, [], context["ranks"])
                if (len(prompt) - 1) % 2 or len(prompt) < 3:
                    raise ValueError(f"prefix {prefix} is not a complete P/F context")
                nodes = _choose_tree_tokens(shared, context["ranker"],
                                            context["ranks"], prompt)
                if nodes is None:
                    numeric_skips.append(dict(source=source_id, prefix=prefix,
                                              reason="ranker returned fewer than two Position candidates or no P2"))
                    continue
                numeric_cases.append(dict(
                    pair_id=f"tree-{source_index:02d}-p{prefix:04d}",
                    source=source_id, prefix=prefix, prompt=prompt,
                    seed=seed, ranks=context["ranks"], nodes=nodes,
                ))
        numeric_source_ids = sorted({row["source"] for row in numeric_cases})
        prefix_coverage = sorted({int(row["prefix"]) for row in numeric_cases})
        _write_json(out / "numeric_selection.json", dict(
            selected=[dict(pair_id=row["pair_id"], source=row["source"],
                           prefix=row["prefix"], tree_tokens=row["nodes"])
                      for row in numeric_cases],
            skipped=numeric_skips,
            source_count=len(numeric_source_ids),
            prefix_coverage=prefix_coverage,
        ))
        if (len(numeric_source_ids) < int(args.numeric_sources)
                or any(prefix not in prefix_coverage for prefix in prefixes)):
            raise ValueError(
                f"only {len(numeric_source_ids)} sources and prefixes {prefix_coverage} "
                f"have backup candidates; need {args.numeric_sources} sources and prefixes {prefixes}"
            )
        with threadpool_limits(limits=1), torch.inference_mode(), _fp32_runtime(torch), torch.autocast("cuda", enabled=False):
            for row in numeric_cases:
                torch.cuda.synchronize()
                result = _run_numeric_context(
                    row=row, tokens=row["nodes"], target=target, shared=shared,
                    rescue_runtime=rescue_runtime,
                    graph_cache=get_numeric_cache("graph"),
                    primary_cache=get_numeric_cache("primary"),
                    backup_cache=get_numeric_cache("backup"),
                    target_cache_factory=target_cache_factory,
                    CaptureLastBlock=CaptureLastBlock,
                    TargetTreeRuntime=TargetTreeRuntime,
                    fp32_context=lambda: _fp32_runtime(torch),
                    torch=torch, device=torch_device, atol=args.atol, rtol=args.rtol,
                )
                _append_jsonl(out / "numeric_records.jsonl", result)
                numeric_rows.append(result)
                pipeline["numeric_done"] = len(numeric_rows)
                _write_json(out / "pipeline.json", pipeline)
                if not result["passed"]:
                    raise AssertionError(
                        f"tree numerical comparison failed for {row['pair_id']}; details saved"
                    )
        numeric_report = dict(
            status="passed" if all(row["passed"] for row in numeric_rows) else "failed",
            context_count=len(numeric_rows),
            minimum_source_count=int(args.numeric_sources),
            selected_source_count=len(numeric_source_ids),
            prefix_coverage=prefix_coverage,
            skipped_contexts=numeric_skips,
            graph_count=rescue_runtime.stats()["graph_count"],
            graph_calls=rescue_runtime.stats()["calls"],
            max_policy_tv=max((row["max_policy_tv"] for row in numeric_rows), default=None),
            max_hidden_relative_l2=max((
                item["hidden"]["relative_l2"]
                for row in numeric_rows for item in row["node_comparisons"]
                if item["hidden"]["relative_l2"] is not None
            ), default=None),
            max_hidden_row_l2_relative=max((
                item["hidden"]["row_l2_relative"]
                for row in numeric_rows for item in row["node_comparisons"]
                if item["hidden"]["row_l2_relative"] is not None
            ), default=None),
            max_hidden_normalized_max_abs=max((
                item["hidden"]["max_abs_over_reference_max_abs"]
                for row in numeric_rows for item in row["node_comparisons"]
                if item["hidden"]["max_abs_over_reference_max_abs"] is not None
            ), default=None),
            commit_0_4_within_scalar_limits=all(
                row["commit_0_4"]["k_metrics"]["all_finite"]
                and row["commit_0_4"]["v_metrics"]["all_finite"]
                and row["commit_0_4"]["k_metrics"]["relative_l2"] <= 2e-6
                and row["commit_0_4"]["v_metrics"]["relative_l2"] <= 2e-6
                and row["commit_0_4"]["k_metrics"]["max_abs_over_reference_max_abs"] <= 2e-6
                and row["commit_0_4"]["v_metrics"]["max_abs_over_reference_max_abs"] <= 2e-6
                and row["commit_0_4"]["token_cache_equal"] for row in numeric_rows
            ),
            commit_0_4_allclose_diagnostic=all(
                row["commit_0_4"]["k_allclose"] and row["commit_0_4"]["v_allclose"]
                and row["commit_0_4"]["token_cache_equal"] for row in numeric_rows
            ),
            commit_0_4_allclose=all(
                row["commit_0_4"]["k_allclose"] and row["commit_0_4"]["v_allclose"]
                and row["commit_0_4"]["token_cache_equal"] for row in numeric_rows
            ),
            rows=numeric_rows,
        )
        _write_json(out / "numeric_report.json", numeric_report)
        pipeline["numeric_done"] = len(numeric_rows)
        pipeline["numeric_status"] = numeric_report["status"]
        _write_json(out / "pipeline.json", pipeline)

        arms = {}
        for name in ("w1", "rescue", "reuse0", "reuse25"):
            arm = _build_decode_arm(
                name, context, adapter, DynamicTargetGraph=DynamicTargetGraph,
                TargetTreeRuntime=TargetTreeRuntime,
                PositionRescueRuntime=PositionRescueRuntime,
                capacity=capacity,
            )
            arm["adapter"] = adapter
            arms[name] = arm

        decode_rows = []
        pair_checks = []
        decode_prompts = []
        for source_index, (source_id, record) in enumerate(selected):
            for prefix in prefixes:
                for seed_index in range(int(args.seeds)):
                    seed = int(args.seed_start) + seed_index
                    prompt = [int(context["eos"]), *map(int, record["tokens"][:prefix])]
                    pair_id = f"decode-{source_index:02d}-p{prefix:04d}-s{seed_index:02d}"
                    shared.validate_sequence(prompt, [], context["ranks"])
                    decode_prompts.append(dict(pair_id=pair_id, source=source_id,
                                               crop_path=record.get("path"),
                                               prefix=prefix, seed=seed, prompt=prompt))
        _write_json(out / "decode_prompts.json", dict(prompts=decode_prompts))
        expected_matrix_rows = len(decode_prompts) * 5
        bos_prompt = [int(context["eos"])]
        bos_seed = int(args.seed_start) + 1000003
        bos_specs = [
            ("w1", "bos_w1"), ("reuse0", "bos_reuse0"),
            ("reuse25", "bos_reuse25_first"), ("reuse25", "bos_reuse25_repeat"),
        ]
        with threadpool_limits(limits=1), torch.inference_mode(), _fp32_runtime(torch), torch.autocast("cuda", enabled=False):
            for prompt_index, prompt_row in enumerate(decode_prompts):
                specs = [
                    ("w1", "w1"), ("rescue", "rescue"), ("reuse0", "reuse0"),
                    ("reuse25", "reuse25_first"), ("reuse25", "reuse25_repeat"),
                ]
                rotation = prompt_index % len(specs)
                specs = specs[rotation:] + specs[:rotation]
                paired = {}
                for arm_name, label in specs:
                    result = _run_decode_arm(
                        arms[arm_name], prompt_row["prompt"], prompt_row["seed"],
                        args.budget, context, torch=torch,
                        target_cache_factory=target_cache_factory,
                        validate_sequence=validate_sequence,
                        decode_w1=decode_w1, decode_rescue=decode_rescue,
                        decode_reuse=decode_reuse, reuse_weight=args.reuse_weight,
                    )
                    result.update(pair_id=prompt_row["pair_id"], source=prompt_row["source"],
                                  crop_path=prompt_row["crop_path"],
                                  prefix=prompt_row["prefix"], seed=prompt_row["seed"],
                                  budget=args.budget, action=label)
                    _append_jsonl(out / "decode_records.jsonl", result)
                    decode_rows.append(result)
                    paired[label] = result
                    pipeline["decode_done"] = len(decode_rows)
                    _write_json(out / "pipeline.json", pipeline)
                by_label = {item["action"]: item for item in paired.values()}
                w1 = by_label["w1"]
                rescue = by_label["rescue"]
                reuse0 = by_label["reuse0"]
                reuse_a = by_label["reuse25_first"]
                reuse_b = by_label["reuse25_repeat"]
                rescue_delta = int(w1["recovery_calls"]) - int(rescue["recovery_calls"])
                rescue_hits = int(rescue["position_rescue_hits"])
                semantic = dict(
                    w1_rescue_output_equal=w1["output_tokens"] == rescue["output_tokens"],
                    w1_rescue_semantic_counters=_count_differences(w1["counters"], rescue["counters"]),
                    recovery_delta_equals_hits=(rescue_delta == rescue_hits),
                    recovery_delta=rescue_delta,
                    rescue_hits=rescue_hits,
                    w1_reuse0_output_equal=w1["output_tokens"] == reuse0["output_tokens"],
                    w1_reuse0_semantic_counters=_count_differences(w1["counters"], reuse0["counters"]),
                    reuse25_replay_output_equal=reuse_a["output_tokens"] == reuse_b["output_tokens"],
                    reuse25_replay_semantic_counters=_count_differences(reuse_a["counters"], reuse_b["counters"]),
                    all_sequences_validated=all(item["sequence_validated"] for item in paired.values()),
                )
                semantic["passed"] = all((
                    semantic["w1_rescue_output_equal"],
                    semantic["w1_rescue_semantic_counters"]["equal"],
                    semantic["recovery_delta_equals_hits"],
                    semantic["w1_reuse0_output_equal"],
                    semantic["w1_reuse0_semantic_counters"]["equal"],
                    semantic["reuse25_replay_output_equal"],
                    semantic["reuse25_replay_semantic_counters"]["equal"],
                    semantic["all_sequences_validated"],
                ))
                semantic_row = dict(pair_id=prompt_row["pair_id"], source=prompt_row["source"],
                                    prefix=prompt_row["prefix"], seed=prompt_row["seed"],
                                    **semantic)
                _append_jsonl(out / "pair_checks.jsonl", semantic_row)
                pair_checks.append(semantic_row)
                if not semantic_row["passed"]:
                    raise AssertionError(
                        f"full decode comparison failed for {prompt_row['pair_id']}; details saved"
                    )
        bos_results = {}
        with threadpool_limits(limits=1), torch.inference_mode(), _fp32_runtime(torch), torch.autocast("cuda", enabled=False):
            for arm_name, label in bos_specs:
                bos_row = _run_decode_arm(
                    arms[arm_name], bos_prompt, bos_seed, 16, context, torch=torch,
                    target_cache_factory=target_cache_factory,
                    validate_sequence=validate_sequence,
                    decode_w1=decode_w1, decode_rescue=decode_rescue,
                    decode_reuse=decode_reuse, reuse_weight=args.reuse_weight,
                )
                bos_row.update(pair_id="bos-bootstrap", source="bos-only", crop_path=None,
                               prefix=0, seed=bos_seed, budget=16, action=label)
                _append_jsonl(out / "decode_records.jsonl", bos_row)
                decode_rows.append(bos_row)
                bos_results[label] = bos_row
                pipeline["decode_done"] = len(decode_rows)
                _write_json(out / "pipeline.json", pipeline)
        bos_checks = dict(
            w1_reuse0_output_equal=(
                bos_results["bos_w1"]["output_tokens"] ==
                bos_results["bos_reuse0"]["output_tokens"]
            ),
            w1_reuse0_semantic_counters=_count_differences(
                bos_results["bos_w1"]["counters"],
                bos_results["bos_reuse0"]["counters"]
            ),
            reuse25_replay_output_equal=(
                bos_results["bos_reuse25_first"]["output_tokens"] ==
                bos_results["bos_reuse25_repeat"]["output_tokens"]
            ),
            reuse25_replay_semantic_counters=_count_differences(
                bos_results["bos_reuse25_first"]["counters"],
                bos_results["bos_reuse25_repeat"]["counters"]
            ),
            all_sequences_validated=all(item["sequence_validated"]
                                        for item in bos_results.values()),
        )
        bos_checks["passed"] = all((
            bos_checks["w1_reuse0_output_equal"],
            bos_checks["w1_reuse0_semantic_counters"]["equal"],
            bos_checks["reuse25_replay_output_equal"],
            bos_checks["reuse25_replay_semantic_counters"]["equal"],
            bos_checks["all_sequences_validated"],
        ))
        _write_json(out / "bos_bootstrap_check.json", bos_checks)
        if not bos_checks["passed"]:
            raise AssertionError("BOS-only reuse bootstrap comparison failed")
        expected_total_runs = expected_matrix_rows + len(bos_specs)
        if len(decode_rows) != expected_total_runs:
            raise RuntimeError(f"recorded {len(decode_rows)} of {expected_total_runs} decode runs")
        hit_count = sum(int(row["rescue_hits"]) for row in pair_checks)
        if hit_count < 1:
            raise AssertionError("the selected short decode matrix did not exercise a backup hit")
        summary = dict(
            status="passed",
            numeric=numeric_report,
            decode=dict(
                prompt_count=len(decode_prompts), expected_runs=expected_total_runs,
                completed_runs=len(decode_rows), pair_check_count=len(pair_checks),
                total_rescue_hits=hit_count,
                total_backup_rejections=sum(
                    int(row["position_backup_rejections"])
                    for row in decode_rows if row["method"] == "rescue"
                ),
                recovery_delta_equals_hit_count=all(
                    row["recovery_delta_equals_hits"] for row in pair_checks
                ),
                all_checks_passed=(all(row["passed"] for row in pair_checks)
                                   and bos_checks["passed"]),
                bos_bootstrap=bos_checks,
                per_method=dict(
                    methods={name: dict(
                        run_count=sum(1 for row in decode_rows if row["method"] == name),
                        mean_generated_tokens=statistics.fmean(
                            row["generated_tokens"] for row in decode_rows if row["method"] == name
                        ),
                    ) for name in ("w1", "rescue", "reuse0", "reuse25")}
                ),
            ),
            files=dict(numeric_records="numeric_records.jsonl",
                       decode_records="decode_records.jsonl",
                       pair_checks="pair_checks.jsonl"),
        )
        _write_json(out / "summary.json", summary)
        pipeline.update(status="complete", finished_at=time.time(),
                        decode_done=len(decode_rows), total_rescue_hits=hit_count)
        _write_json(out / "pipeline.json", pipeline)
        for arm in arms.values():
            arm["proposal"].close()
        print(json.dumps(dict(status="passed", output=str(out),
                              numeric_contexts=len(numeric_rows),
                              decode_runs=len(decode_rows), rescue_hits=hit_count),
                         ensure_ascii=False), flush=True)
    except Exception as error:
        pipeline.update(status="failed", error=f"{type(error).__name__}: {error}",
                        finished_at=time.time())
        _write_json(out / "pipeline.json", pipeline)
        raise


if __name__ == "__main__":
    main()
