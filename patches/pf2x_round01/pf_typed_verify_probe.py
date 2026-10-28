#!/usr/bin/env python3
"""Compare original and type-grouped Target output paths on saved real contexts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time
from types import MethodType


EOS = 12096
P_VOCAB = 8000
F_VOCAB = 4096
DEFAULT_CONTEXTS = Path(
    "/root/autodl-tmp/gaussiangpt_pf2x_round01_20261020/run01/collection/contexts.jsonl"
)
DEFAULT_TARGET = Path("checkpoints/gpt_vfront.ckpt")
DEFAULT_VQVAE = Path("checkpoints/vqvae_vfront.ckpt")
DEFAULT_CANDIDATE = Path(
    "docs/project/artifacts/pf2x_loop_20261019/round02/source_candidate/"
    "model/gpt/nanochat_gpt.py"
)
DEFAULT_SAMPLE_IDS = (
    "003ac11d-2abc-44f8-9836-4354e7dfa543|64|34702|35790|6",
    "003ac11d-2abc-44f8-9836-4354e7dfa543|256|34702|39054|0",
    "154620a2-e63e-474b-b950-b7cc287b0e43|64|34702|35790|0",
    "154620a2-e63e-474b-b950-b7cc287b0e43|256|34703|39055|0",
    "22f7de3c-4349-4dcc-9495-dde7bf690278|64|34703|35791|0",
    "22f7de3c-4349-4dcc-9495-dde7bf690278|256|34703|39055|0",
    "9a21209b-f836-4d36-b106-04495545481a|64|34703|35791|8",
    "9a21209b-f836-4d36-b106-04495545481a|256|34703|39055|0",
)
TOLERANCE = dict(logits_atol=2e-5, logits_rtol=2e-5,
                 hidden_atol=2e-5, hidden_rtol=2e-5,
                 rope_atol=2e-5, rope_rtol=2e-5,
                 cache_atol=2e-5, cache_rtol=2e-5,
                 policy_atol=5e-5, policy_tv=5e-5)
CALL_CATEGORIES = ("prefill", "reference", "odd_start_prefill", "odd_reference",
                   "warmup", "paired", "odd_candidate")


class StopRequested(Exception):
    pass


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def _eligible(row: dict) -> tuple[bool, str]:
    try:
        prefix_length = int(row["prefix"])
        context = [int(token) for token in row["context_prefix"]]
        inputs = [int(token) for token in row["target_inputs"]]
        p1 = int(row["p1_id"])
        p2 = int(row["p2_id"])
    except (KeyError, TypeError, ValueError):
        return False, "missing_or_invalid_fields"
    if prefix_length not in {64, 256}:
        return False, "prefix_not_64_or_256"
    if row.get("source_split") != "development":
        return False, "outside_development_sources"
    if len(context) < 2 or context[0] != EOS:
        return False, "invalid_context_prefix"
    if len(inputs) < 4:
        return False, "fewer_than_four_saved_inputs"
    if inputs[0] != context[-1]:
        return False, "first_saved_input_differs_from_pending_token"
    if inputs[1] != p1 or inputs[3] != p2:
        return False, "saved_position_inputs_differ_from_candidates"
    if int(row.get("f1a_local_id", -1)) + P_VOCAB != inputs[2]:
        return False, "saved_feature_input_differs_from_candidate"
    if p1 == EOS or p2 == EOS:
        return False, "position_candidate_is_eos"
    if not (0 <= p1 < P_VOCAB and 0 <= p2 < P_VOCAB):
        return False, "position_candidate_outside_position_vocabulary"
    if not (P_VOCAB <= inputs[0] < EOS and P_VOCAB <= inputs[2] < EOS):
        return False, "feature_input_outside_feature_vocabulary"
    if not row.get("scene") or not row.get("sample_id"):
        return False, "missing_source_or_sample_id"
    return True, "eligible"


def _select_planned_contexts(rows: list[dict], sample_ids: list[str],
                             min_sources: int) -> tuple[list[dict], dict]:
    if not 8 <= len(sample_ids) <= 20:
        raise ValueError("sample-id count must be between 8 and 20")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample IDs must be unique")
    row_by_id = {str(row.get("sample_id")): row for row in rows}
    selected = []
    for sample_id in sample_ids:
        if sample_id not in row_by_id:
            raise ValueError(f"planned sample is absent from collection: {sample_id}")
        row = row_by_id[sample_id]
        valid, reason = _eligible(row)
        if not valid:
            raise ValueError(f"planned sample {sample_id} is ineligible: {reason}")
        selected.append(row)
    scenes = {str(row["scene"]) for row in selected}
    prefixes = {int(row["prefix"]) for row in selected}
    if len(scenes) < min_sources or prefixes != {64, 256}:
        raise ValueError(
            f"planned sample coverage is insufficient: sources={len(scenes)}, "
            f"prefixes={sorted(prefixes)}")
    f1_accepted = sum(row.get("f1_accepted_actual") is True for row in selected)
    f1_rejected = sum(row.get("f1_accepted_actual") is False for row in selected)
    p2_reached = [row for row in selected if row.get("p2_accepted_actual") is not None]
    p2_accepted = sum(row.get("p2_accepted_actual") is True for row in p2_reached)
    p2_rejected = sum(row.get("p2_accepted_actual") is False for row in p2_reached)
    coverage = dict(p1_accepted=sum(row.get("p1_accepted_actual") is True
                                    for row in selected),
                    f1_accepted=f1_accepted, f1_rejected=f1_rejected,
                    p2_accepted=p2_accepted, p2_rejected=p2_rejected,
                    p2_not_reached=len(selected) - len(p2_reached))
    if len(selected) == 8 and coverage != dict(p1_accepted=8, f1_accepted=4,
                                               f1_rejected=4, p2_accepted=2,
                                               p2_rejected=2, p2_not_reached=4):
        raise ValueError(f"planned rows no longer match recorded branch coverage: {coverage}")
    return selected, dict(selected_contexts=len(selected), selected_sources=len(scenes),
                          selected_prefix_counts={str(prefix): sum(
                              int(row["prefix"]) == prefix for row in selected)
                              for prefix in (64, 256)},
                          selected_sample_ids=list(sample_ids), branch_coverage=coverage,
                          selection_source="explicit R1 sample IDs")


def _schedule(context_index: int, shape: int, warmups: int,
              pair_repeats: int) -> list[dict]:
    if warmups < 2 or pair_repeats < 12:
        raise ValueError("at least two warmups and twelve pairs are required")
    rows = []
    for warmup_index in range(warmups * 2):
        mode = ("original" if (context_index + shape + warmup_index) % 2 == 0
                else "grouped")
        rows.append(dict(record_type="warmup", warmup_index=warmup_index,
                         pair_id=None, mode=mode, shape=shape, order=warmup_index))
    for pair_index in range(pair_repeats):
        first = "original" if (context_index + shape + pair_index) % 2 == 0 else "grouped"
        order = (first, "grouped" if first == "original" else "original")
        pair_id = f"context-{context_index}:shape-{shape}:pair-{pair_index:02d}"
        for order_index, mode in enumerate(order):
            rows.append(dict(record_type="paired", warmup_index=None,
                             pair_id=pair_id, pair_index=pair_index,
                             mode=mode, shape=shape, order=order_index,
                             first_mode=first))
    return rows


def _planned_counts(context_count: int) -> dict:
    return dict(target_call_counts=dict(
                    prefill=context_count,
                    reference=3 * context_count,
                    odd_start_prefill=context_count,
                    odd_reference=context_count,
                    warmup=12 * context_count,
                    paired=72 * context_count,
                    odd_candidate=2 * context_count,
                    total=92 * context_count),
                base_target_forwards=85 * context_count,
                extra_correctness_forwards=7 * context_count,
                kv_restores=91 * context_count,
                numeric_comparisons=86 * context_count,
                paired_comparisons=36 * context_count)


def _self_test() -> dict:
    rows = []
    for index, sample_id in enumerate(DEFAULT_SAMPLE_IDS):
        source, prefix, *_ = sample_id.split("|")
        p2_reached = index in {1, 2, 4, 6}
        p2_accepted = index in {1, 2}
        f1_accepted = p2_reached
        p1, p2 = 100 + index, 200 + index
        prefix_tokens = [EOS, 10, 8001, 11, 8002]
        rows.append(dict(prefix=int(prefix), scene=source, sample_id=sample_id,
                         source_split="development", context_prefix=prefix_tokens,
                         target_inputs=[8002, p1, 8010, p2], p1_id=p1, p2_id=p2,
                         f1a_local_id=10, p1_accepted_actual=True,
                         f1_accepted_actual=f1_accepted,
                         p2_accepted_actual=(p2_accepted if p2_reached else None)))
    bad = dict(rows[0], sample_id="eos-case", p2_id=EOS,
               target_inputs=[8002, 12, 8010, EOS])
    rows.append(bad)
    selected, report = _select_planned_contexts(rows, list(DEFAULT_SAMPLE_IDS), min_sources=4)
    assert len(selected) == 8 and report["selected_sources"] >= 4
    assert {row["prefix"] for row in selected} == {64, 256}
    assert all(row["sample_id"] != "eos-case" for row in selected)
    for context_index in range(4):
        for shape in (1, 2, 4):
            schedule = _schedule(context_index, shape, warmups=2, pair_repeats=12)
            warmups = [row for row in schedule if row["record_type"] == "warmup"]
            pairs: dict[str, list[dict]] = defaultdict(list)
            for row in schedule:
                if row["record_type"] == "paired":
                    pairs[row["pair_id"]].append(row)
            assert sum(row["mode"] == "original" for row in warmups) == 2
            assert sum(row["mode"] == "grouped" for row in warmups) == 2
            assert len(pairs) == 12
            for pair_index, pair in enumerate(pairs.values()):
                assert len(pair) == 2 and {row["mode"] for row in pair} == {"original", "grouped"}
                assert [row["order"] for row in pair] == [0, 1]
                first = "original" if (context_index + shape + pair_index) % 2 == 0 else "grouped"
                assert pair[0]["mode"] == first
    timing_rows = [
        dict(record_type="paired_comparison", shape=4, scene="scene-a",
             original_host_wall_seconds=0.010, grouped_host_wall_seconds=0.008,
             original_cuda_seconds=0.006, grouped_cuda_seconds=0.005,
             grouped_over_original_host_ratio=0.8,
             grouped_over_original_cuda_ratio=5.0 / 6.0),
        dict(record_type="paired_comparison", shape=4, scene="scene-a",
             original_host_wall_seconds=0.012, grouped_host_wall_seconds=0.010,
             original_cuda_seconds=0.007, grouped_cuda_seconds=0.006,
             grouped_over_original_host_ratio=5.0 / 6.0,
             grouped_over_original_cuda_ratio=6.0 / 7.0),
        dict(record_type="kv_restore", shape=4, scene="scene-a"),
    ]
    timing = _timing_summary_rows(timing_rows)
    shape_summary = timing["by_shape"]["4"]
    assert timing["complete_pair_rows"] == 2
    assert shape_summary["pair_count"] == 2
    assert shape_summary["grouped_over_original_host_ratio_median"] == statistics.median(
        [0.8, 5.0 / 6.0])
    counts = _planned_counts(8)
    assert counts["target_call_counts"]["total"] == 736
    assert counts["base_target_forwards"] == 680
    assert counts["extra_correctness_forwards"] == 56
    assert counts["kv_restores"] == 728
    assert counts["numeric_comparisons"] == 688
    assert counts["paired_comparisons"] == 288
    return dict(passed=True, contexts_selected=len(selected), sources=report["selected_sources"],
                prefixes=report["selected_prefix_counts"],
                eos_performance_context_excluded=True,
                shapes=[1, 2, 4], warmups_per_mode=2, paired_repeats=12,
                schedule_forwards_per_context=3 * (4 + 24),
                planned_default_counts=counts,
                paired_summary_rows_tested=timing["complete_pair_rows"])


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _clone_tree(value, torch):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, tuple):
        return tuple(_clone_tree(item, torch) for item in value)
    if isinstance(value, list):
        return [_clone_tree(item, torch) for item in value]
    if isinstance(value, dict):
        return {key: _clone_tree(item, torch) for key, item in value.items()}
    return value


def _cache_snapshot(cache, torch) -> dict:
    position = int(cache.get_pos())
    return dict(position=position,
                cache_seqlens=cache.cache_seqlens.detach().cpu().clone(),
                token_cache=cache.token_cache[:, :position].detach().cpu().clone(),
                k_cache=cache.k_cache[:, :, :position].detach().cpu().clone(),
                v_cache=cache.v_cache[:, :, :position].detach().cpu().clone())


def _tensor_metrics(left, right, torch, atol: float, rtol: float) -> dict:
    if not torch.is_tensor(left) or not torch.is_tensor(right):
        raise TypeError("tensor metrics received a non-tensor")
    if left.shape != right.shape:
        return dict(shape_match=False, left_shape=list(left.shape),
                    right_shape=list(right.shape), max_abs=None, allclose=False)
    left = left.detach().cpu()
    right = right.detach().cpu()
    if left.is_complex() or right.is_complex():
        difference = (left - right).abs()
    else:
        difference = (left - right).abs()
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    return dict(shape_match=True, max_abs=max_abs,
                allclose=bool(torch.allclose(left, right, atol=atol, rtol=rtol,
                                             equal_nan=False)))


def _nested_metrics(left, right, torch, atol: float, rtol: float):
    if torch.is_tensor(left) and torch.is_tensor(right):
        return _tensor_metrics(left, right, torch, atol, rtol)
    if type(left) is not type(right):
        return dict(type_match=False, left_type=type(left).__name__,
                    right_type=type(right).__name__, allclose=False)
    if isinstance(left, (tuple, list)):
        if len(left) != len(right):
            return dict(length_match=False, allclose=False)
        items = [_nested_metrics(a, b, torch, atol, rtol)
                 for a, b in zip(left, right)]
        return dict(items=items, allclose=all(item.get("allclose", False) for item in items))
    if isinstance(left, dict):
        if set(left) != set(right):
            return dict(keys_match=False, allclose=False)
        items = {key: _nested_metrics(left[key], right[key], torch, atol, rtol)
                 for key in left}
        return dict(items=items, allclose=all(item.get("allclose", False)
                                              for item in items.values()))
    same = left == right
    return dict(value_match=bool(same), allclose=bool(same))


def _logit_metrics(left, right, torch) -> dict:
    if left.shape != right.shape:
        return dict(shape_match=False, finite_mask_mismatches=None,
                    max_abs=None, allclose=False)
    left = left.detach().cpu().float()
    right = right.detach().cpu().float()
    left_finite = torch.isfinite(left)
    right_finite = torch.isfinite(right)
    mask_mismatches = int((left_finite != right_finite).sum().item())
    nan_count = int((torch.isnan(left).sum() + torch.isnan(right).sum()).item())
    if mask_mismatches:
        max_abs = None
        close = False
    else:
        values = left_finite
        difference = (left[values] - right[values]).abs()
        max_abs = float(difference.max().item()) if difference.numel() else 0.0
        close = bool(torch.allclose(left[values], right[values],
                                    atol=TOLERANCE["logits_atol"],
                                    rtol=TOLERANCE["logits_rtol"], equal_nan=False))
    return dict(shape_match=True, finite_mask_mismatches=mask_mismatches,
                nan_count=nan_count, finite_values=int(left_finite.sum().item()),
                max_abs=max_abs, allclose=close and nan_count == 0)


def _policy_metrics(left, right, shared, torch) -> dict:
    results = []
    for row_index in range(int(left.size(1))):
        p = shared.policy(left[0, row_index, :].detach().cpu()).float()
        q = shared.policy(right[0, row_index, :].detach().cpu()).float()
        max_abs = float((p - q).abs().max().item())
        tv = float((p - q).abs().sum().item() / 2.0)
        results.append(dict(row=row_index, max_abs=max_abs, total_variation=tv,
                            allclose=max_abs <= TOLERANCE["policy_atol"] and
                            tv <= TOLERANCE["policy_tv"]))
    return dict(rows=results, max_abs=max(row["max_abs"] for row in results),
                max_total_variation=max(row["total_variation"] for row in results),
                allclose=all(row["allclose"] for row in results))


def _cache_metrics(left, right, torch) -> dict:
    if left["position"] != right["position"]:
        return dict(position_match=False, allclose=False)
    parts = {}
    for key in ("cache_seqlens", "token_cache"):
        parts[key] = dict(equal=bool(torch.equal(left[key], right[key])))
    for key in ("k_cache", "v_cache"):
        parts[key] = _tensor_metrics(left[key], right[key], torch,
                                     TOLERANCE["cache_atol"], TOLERANCE["cache_rtol"])
    passed = all(item.get("equal", item.get("allclose", False))
                 for item in parts.values())
    return dict(position_match=True, position=left["position"], parts=parts, allclose=passed)


def _compare_runs(left: dict, right: dict, shared, torch) -> dict:
    logits = _logit_metrics(left["logits"], right["logits"], torch)
    hidden = _tensor_metrics(left["hidden"], right["hidden"], torch,
                             TOLERANCE["hidden_atol"], TOLERANCE["hidden_rtol"])
    rope = _nested_metrics(left["rope"], right["rope"], torch,
                           TOLERANCE["rope_atol"], TOLERANCE["rope_rtol"])
    policy = _policy_metrics(left["logits"], right["logits"], shared, torch)
    cache = _cache_metrics(left["cache"], right["cache"], torch)
    return dict(logits=logits, pre_final_norm_hidden=hidden, rope=rope,
                policy=policy, valid_cache=cache,
                passed=all(item.get("allclose", False) for item in
                           (logits, hidden, rope, policy, cache)))


def _finite_slot_counts(logits, kinds, core, torch) -> list[dict]:
    finite = torch.isfinite(logits[0])
    rows = []
    for index, kind in enumerate(kinds):
        ids = torch.nonzero(finite[index], as_tuple=False).flatten()
        contains_eos = bool(finite[index, int(core.sparse_eos_token_id)])
        rows.append(dict(row=index, predicted_kind=kind,
                         finite_count=int(ids.numel()), eos_is_finite=contains_eos,
                         position_id_count=int(((ids < core.sparse_position_vocab_size)).sum().item()),
                         feature_id_count=int(((ids >= P_VOCAB) & (ids < EOS)).sum().item()),
                         eos_policy_valid=(contains_eos if kind == "position"
                                           else not contains_eos)))
    return rows


def _timing_summary_rows(rows: list[dict]) -> dict:
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    by_shape: dict[int, list[dict]] = defaultdict(list)
    pair_row_count = 0
    for row in rows:
        if row.get("record_type") != "paired_comparison":
            continue
        pair_row_count += 1
        shape = int(row["shape"])
        grouped[(shape, str(row["scene"]))].append(row)
        by_shape[shape].append(row)

    def summarize(values: list[dict]) -> dict:
        host_ratios = [row["grouped_over_original_host_ratio"] for row in values
                       if row.get("grouped_over_original_host_ratio") is not None]
        cuda_ratios = [row["grouped_over_original_cuda_ratio"] for row in values
                       if row.get("grouped_over_original_cuda_ratio") is not None]
        return dict(pair_count=len(values), host_pair_median_seconds=dict(
                        original=statistics.median([row["original_host_wall_seconds"]
                                                    for row in values]) if values else None,
                        grouped=statistics.median([row["grouped_host_wall_seconds"]
                                                   for row in values]) if values else None),
                    cuda_event_pair_median_seconds=dict(
                        original=statistics.median([row["original_cuda_seconds"]
                                                    for row in values]) if values else None,
                        grouped=statistics.median([row["grouped_cuda_seconds"]
                                                   for row in values]) if values else None),
                    grouped_over_original_host_ratio_median=(
                        statistics.median(host_ratios) if host_ratios else None),
                    grouped_over_original_cuda_ratio_median=(
                        statistics.median(cuda_ratios) if cuda_ratios else None))

    return dict(
        scope="Target microbenchmark only; these ratios are not end-to-end speed claims",
        by_shape={str(shape): summarize(values) for shape, values in sorted(by_shape.items())},
        per_source_shape={f"{shape}:{scene}": summarize(values)
                          for (shape, scene), values in sorted(grouped.items())},
        complete_pair_rows=pair_row_count)


def _timing_summary(path: Path) -> dict:
    return _timing_summary_rows(_read_jsonl(path))


def _precision_state(torch) -> dict:
    try:
        autocast_enabled = bool(torch.is_autocast_enabled("cuda"))
    except TypeError:
        autocast_enabled = bool(torch.is_autocast_enabled())
    return dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
                cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
                cuda_autocast_enabled=autocast_enabled)


def _run(args) -> int:
    started_wall = time.time()
    started_mono = time.monotonic()
    deadline = started_mono + args.deadline_seconds
    project_root = args.project_root.resolve()

    def resolve_input(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (project_root / path).resolve()

    contexts_path = resolve_input(args.contexts)
    target_path = resolve_input(args.target)
    vqvae_path = resolve_input(args.vqvae)
    candidate_path = resolve_input(args.candidate)
    output = resolve_input(args.output)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    row_path = output / "probe_rows.jsonl"
    pair_path = output / "paired_target.jsonl"
    row_path.touch()
    pair_path.touch()
    state = dict(stop_requested=False, signal_number=None)

    def request_stop(signum, _frame):
        state["stop_requested"] = True
        state["signal_number"] = int(signum)

    old_signal = signal.signal(signal.SIGTERM, request_stop)
    old_alarm = signal.signal(signal.SIGALRM, request_stop)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.setitimer(signal.ITIMER_REAL, max(0.0, deadline - time.monotonic()))
    progress = dict(status="running", started_at=started_wall,
                    deadline_at=started_wall + args.deadline_seconds,
                    deadline_seconds=args.deadline_seconds, completed_target_calls=0,
                    attempted_target_calls=0, completed_contexts=0,
                    current_sample_id=None, current_stage="setup")
    zero_categories = {key: 0 for key in CALL_CATEGORIES}
    metadata = dict(status="running", project_root=str(project_root),
                    contexts=str(contexts_path),
                    target_checkpoint=str(target_path),
                    vqvae_checkpoint=str(vqvae_path),
                    candidate_source=str(candidate_path),
                    output_dir=str(output), context_count=len(args.sample_ids),
                    min_sources=args.min_sources,
                    shapes=[1, 2, 4], warmups_per_mode=args.warmups,
                    paired_repeats=args.paired_repeats,
                    policy=dict(target_precision="FP32", temperature=0.9, top_p=0.9),
                    hidden_capture="last Transformer block output before final norm",
                    target_head="actual core.forward with temporary candidate forward replacement",
                    timing_scope="Target forward only; cache restore logged separately",
                    tolerance=TOLERANCE,
                    target_call_attempts=dict(zero_categories, total=0),
                    target_call_counts=dict(zero_categories, total=0),
                    planned_counts=_planned_counts(len(args.sample_ids)),
                    kv_restore_count=0, gpu_calls_are_diagnostic=True,
                    status_reason=None)
    _write_json(output / "metadata.json", metadata)
    _write_json(output / "progress.json", progress)

    torch = None
    core = None
    capture = None
    fp32_runtime = None
    fp32_runtime_entered = False
    thread_runtime = None
    thread_runtime_entered = False
    report = dict(status="failed", reason=None, started_at=started_wall,
                  deadline_at=started_wall + args.deadline_seconds,
                  selection=None, elapsed_seconds=None, metadata=metadata)

    def checkpoint(stage: str, sample_id=None):
        if state["stop_requested"]:
            raise StopRequested(f"received signal {state['signal_number']}")
        if time.monotonic() >= deadline:
            raise StopRequested("shared 900-second deadline reached")
        progress.update(current_stage=stage, current_sample_id=sample_id,
                        elapsed_seconds=time.monotonic() - started_mono,
                        completed_target_calls=metadata["target_call_counts"]["total"],
                        attempted_target_calls=progress["attempted_target_calls"])
        _write_json(output / "progress.json", progress)

    def save_row(row: dict):
        _append_jsonl(row_path, row)
        progress["completed_target_calls"] = metadata["target_call_counts"]["total"]
        progress["elapsed_seconds"] = time.monotonic() - started_mono
        _write_json(output / "progress.json", progress)

    try:
        checkpoint("load_contexts")
        rows = _read_jsonl(contexts_path)
        selected, selection_report = _select_planned_contexts(
            rows, list(args.sample_ids), args.min_sources)
        report["selection"] = selection_report
        metadata["selection"] = selection_report
        metadata["selected_sample_ids"] = [str(row["sample_id"]) for row in selected]
        _write_json(output / "metadata.json", metadata)

        os.chdir(project_root)
        sys.path.insert(0, str(project_root))
        sys.path.insert(0, str(project_root / "patches" / "pf2x_round01"))
        checkpoint("load_project_modules")
        import torch as torch_module
        import bench_gaussiangpt_pf_block2_cow as bench
        from draft_a import pf_ranker_eval as shared
        from draft_a.feature_acceptance_eval import cache_for
        from draft_a.learning_eval import _fp32_runtime
        from measure_target_shapes import _load_target
        from threadpoolctl import threadpool_limits

        torch = torch_module
        if not torch.cuda.is_available():
            raise RuntimeError("the real Target probe requires the existing CUDA environment")
        torch.set_num_threads(6)
        metadata["precision_before"] = _precision_state(torch)
        fp32_runtime = _fp32_runtime(torch)
        fp32_runtime.__enter__()
        fp32_runtime_entered = True
        thread_runtime = threadpool_limits(limits=1, user_api="openmp")
        thread_runtime.__enter__()
        thread_runtime_entered = True
        metadata["precision_during_probe"] = _precision_state(torch)
        metadata["precision_during_probe"]["target_autocast_enabled"] = False
        if metadata["precision_during_probe"]["float32_matmul_precision"] != "highest":
            raise RuntimeError("FP32 runtime did not select highest matmul precision")
        if metadata["precision_during_probe"]["cuda_matmul_allow_tf32"]:
            raise RuntimeError("FP32 runtime left CUDA TF32 enabled")
        if metadata["precision_during_probe"]["cudnn_allow_tf32"]:
            raise RuntimeError("FP32 runtime left cuDNN TF32 enabled")
        checkpoint("load_target")
        load_started = time.perf_counter()
        target_model = _load_target(target_path, vqvae_path)
        core = target_model.transformer
        metadata["target_load_wall_seconds"] = time.perf_counter() - load_started
        metadata["target_device"] = str(next(core.parameters()).device)
        metadata["target_dtype"] = str(next(core.parameters()).dtype)
        if metadata["target_device"] != "cuda:0" and not metadata["target_device"].startswith("cuda"):
            raise RuntimeError(f"expected CUDA Target, got {metadata['target_device']}")

        checkpoint("load_candidate")
        spec = importlib.util.spec_from_file_location(
            "model.gpt.nanochat_gpt_round02_candidate", candidate_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load candidate source: {candidate_path}")
        candidate = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = candidate
        spec.loader.exec_module(candidate)
        candidate_forward = MethodType(candidate.GPT.forward, core)
        helper_marker = object()
        original_grouped_helper = core.__dict__.get("_project_typed_logits_grouped", helper_marker)
        candidate_helper = MethodType(candidate.GPT._project_typed_logits_grouped, core)
        core._project_typed_logits_grouped = candidate_helper
        original_forward = core.forward
        original_forward_attr = core.__dict__.get("forward", helper_marker)

        rope_values = []
        original_get_rope = core._get_rope
        original_get_rope_attr = core.__dict__.get("_get_rope", helper_marker)

        def capture_rope(idx, kv_cache=None):
            rope = original_get_rope(idx, kv_cache=kv_cache)
            rope_values[:] = [rope]
            return rope

        core._get_rope = capture_rope
        capture = bench.CaptureLastBlock(core)
        device = torch.device("cuda")

        def run_target(context, tokens: list[int], mode: str | None,
                       category: str, keep: int, cache,
                       operation: dict | None = None) -> dict:
            checkpoint(category, context.get("sample_id"))
            before = int(cache.get_pos())
            if before != keep:
                raise RuntimeError(f"call starts at KV {before}, expected keep={keep}")
            snapshot_before = _cache_snapshot(cache, torch)
            kinds = shared.typed_kinds(before, len(tokens))
            idx = torch.as_tensor([tokens], device=device, dtype=torch.long)
            rope_values.clear()
            capture.output = None
            if mode is None:
                core.forward = original_forward
            else:
                core.forward = candidate_forward
            progress["attempted_target_calls"] += 1
            metadata["target_call_attempts"][category] += 1
            metadata["target_call_attempts"]["total"] += 1
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record(torch.cuda.current_stream(device))
            wall_started = time.perf_counter()
            try:
                if mode is None:
                    with torch.autocast(device_type="cuda", enabled=False):
                        logits = core(idx, kv_cache=cache, logit_types=kinds)
                else:
                    with torch.autocast(device_type="cuda", enabled=False):
                        logits = core(idx, kv_cache=cache, logit_types=kinds,
                                      typed_projection=mode)
            finally:
                end.record(torch.cuda.current_stream(device))
                torch.cuda.synchronize(device)
            host_seconds = time.perf_counter() - wall_started
            cuda_seconds = begin.elapsed_time(end) / 1000.0
            metadata["target_call_counts"][category] += 1
            metadata["target_call_counts"]["total"] += 1
            progress["completed_target_calls"] = metadata["target_call_counts"]["total"]
            after = int(cache.get_pos())
            if after != keep + len(tokens):
                raise RuntimeError(f"Target advanced KV to {after}, expected {keep + len(tokens)}")
            if capture.output is None or int(capture.output.size(1)) != len(tokens):
                raise RuntimeError("last-block hook did not capture every requested row")
            result = dict(
                logits=logits.detach().cpu().clone(),
                hidden=capture.output.detach().cpu().clone(),
                rope=_clone_tree(rope_values[0], torch) if rope_values else None,
                cache=_cache_snapshot(cache, torch),
                cache_before=snapshot_before,
                cuda_seconds=cuda_seconds,
                host_wall_seconds=host_seconds,
                kv_position_before=before,
                kv_position_after=after,
                mode=mode or "unmodified",
                kinds=list(kinds),
                slot_counts=(_finite_slot_counts(logits.detach(), kinds, core, torch)
                             if len(tokens) <= 4 else []),
            )
            row = dict(record_type=category, sample_id=context.get("sample_id"),
                       scene=context.get("scene"), prefix=int(context.get("prefix", -1)),
                       mode=result["mode"], input_ids=list(tokens), input_tokens=len(tokens),
                       logit_types=list(kinds), kv_position_before=before,
                       kv_position_after=after, cuda_seconds=cuda_seconds,
                       host_wall_seconds=host_seconds,
                       device=str(idx.device), pair_id=(operation or {}).get("pair_id"),
                       pair_index=(operation or {}).get("pair_index"),
                       schedule_order=(operation or {}).get("order"),
                       schedule_first_mode=(operation or {}).get("first_mode"),
                       eos_finite_by_row=result["slot_counts"],
                       additional_target_forward=True)
            if operation is not None:
                row["schedule_record_type"] = operation["record_type"]
            save_row(row)
            if any(not item["eos_policy_valid"] for item in result["slot_counts"]):
                raise AssertionError("EOS availability does not match the predicted P/F row")
            return result

        def restore(context, cache, keep: int, stage: str,
                    operation: dict | None = None) -> dict:
            old = int(cache.get_pos())
            if old < keep:
                raise RuntimeError(f"cannot restore KV {old} to later keep {keep}")
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record(torch.cuda.current_stream(device))
            wall_started = time.perf_counter()
            bench.truncate(cache, keep, old)
            end.record(torch.cuda.current_stream(device))
            torch.cuda.synchronize(device)
            host_seconds = time.perf_counter() - wall_started
            cuda_seconds = begin.elapsed_time(end) / 1000.0
            metadata["kv_restore_count"] += 1
            row = dict(record_type="kv_restore", sample_id=context.get("sample_id"),
                       scene=context.get("scene"), prefix=int(context.get("prefix", -1)),
                       stage=stage, kv_position_before=old,
                       kv_position_after=int(cache.get_pos()),
                       cuda_seconds=cuda_seconds, host_wall_seconds=host_seconds,
                       mode=(operation or {}).get("mode"),
                       pair_id=(operation or {}).get("pair_id"),
                       pair_index=(operation or {}).get("pair_index"),
                       schedule_order=(operation or {}).get("order"),
                       included_in_forward_time=False)
            _append_jsonl(row_path, row)
            _write_json(output / "metadata.json", metadata)
            return dict(cuda_seconds=cuda_seconds, host_wall_seconds=host_seconds,
                        after=_cache_snapshot(cache, torch))

        def validate_restored(restored: dict, baseline: dict, context, stage: str):
            metrics = _cache_metrics(restored, baseline, torch)
            if not metrics["allclose"]:
                raise RuntimeError(f"valid KV prefix changed after {stage}: {metrics}")

        def one_shape(context_index: int, context: dict, cache, keep: int,
                      base_cache: dict, shape: int):
            inputs = [int(token) for token in context["target_inputs"][:shape]]
            if len(inputs) != shape:
                raise ValueError("saved target_inputs does not cover requested shape")
            reference = run_target(context, inputs, None, "reference", keep, cache)
            restore_result = restore(context, cache, keep, "reference")
            validate_restored(restore_result["after"], base_cache, context, "reference")
            slot_counts = reference["slot_counts"]
            ref_metrics = dict(logits_vs_self_finite_mask=True,
                               position_rows_with_eos=sum(
                                   row["predicted_kind"] == "position" and row["eos_is_finite"]
                                   for row in slot_counts),
                               feature_rows_without_eos=all(
                                   row["predicted_kind"] != "feature" or not row["eos_is_finite"]
                                   for row in slot_counts))

            schedule = _schedule(context_index, shape, args.warmups, args.paired_repeats)
            warmup_results = []
            pair_results: dict[str, list[dict]] = defaultdict(list)

            def record_pair(pair_id: str, pair: list[dict]):
                if len(pair) != 2 or {item["mode"] for item in pair} != {"original", "grouped"}:
                    raise RuntimeError(f"incomplete timing pair {pair_id}")
                by_mode = {item["mode"]: item for item in pair}
                old = by_mode["original"]
                new = by_mode["grouped"]
                summary = dict(record_type="paired_comparison", sample_id=context["sample_id"],
                               scene=context["scene"], prefix=int(context["prefix"]),
                               shape=shape, pair_id=pair_id, input_ids=inputs,
                               kv_keep=keep, pair_index=pair[0]["pair_index"],
                               first_mode=pair[0]["mode"],
                               original_host_wall_seconds=old["host_wall_seconds"],
                               grouped_host_wall_seconds=new["host_wall_seconds"],
                               original_over_grouped_host_ratio=(
                                   old["host_wall_seconds"] / new["host_wall_seconds"]
                                   if new["host_wall_seconds"] > 0 else None),
                               grouped_over_original_host_ratio=(
                                   new["host_wall_seconds"] / old["host_wall_seconds"]
                                   if old["host_wall_seconds"] > 0 else None),
                               original_cuda_seconds=old["cuda_seconds"],
                               grouped_cuda_seconds=new["cuda_seconds"],
                               original_over_grouped_cuda_ratio=(
                                   old["cuda_seconds"] / new["cuda_seconds"]
                                   if new["cuda_seconds"] > 0 else None),
                               grouped_over_original_cuda_ratio=(
                                   new["cuda_seconds"] / old["cuda_seconds"]
                                   if old["cuda_seconds"] > 0 else None),
                               original_restore_host_wall_seconds=old[
                                   "restore_host_wall_seconds"],
                               grouped_restore_host_wall_seconds=new[
                                   "restore_host_wall_seconds"],
                               original_restore_cuda_seconds=old["restore_cuda_seconds"],
                               grouped_restore_cuda_seconds=new["restore_cuda_seconds"],
                               original_forward_plus_restore_host_seconds=(
                                   old["host_wall_seconds"] + old["restore_host_wall_seconds"]),
                               grouped_forward_plus_restore_host_seconds=(
                                   new["host_wall_seconds"] + new["restore_host_wall_seconds"]),
                               original_forward_plus_restore_cuda_seconds=(
                                   old["cuda_seconds"] + old["restore_cuda_seconds"]),
                               grouped_forward_plus_restore_cuda_seconds=(
                                   new["cuda_seconds"] + new["restore_cuda_seconds"]),
                               restore_costs_separate=True,
                               mode_order=[item["mode"] for item in pair],
                               original_numeric_match=old["numeric_match"],
                               grouped_numeric_match=new["numeric_match"])
                _append_jsonl(pair_path, summary)
                _write_json(output / "progress.json", progress)

            for operation in schedule:
                label = "warmup" if operation["record_type"] == "warmup" else "paired"
                result = run_target(context, inputs, operation["mode"], label, keep,
                                    cache, operation)
                metrics = _compare_runs(reference, result, shared, torch)
                _append_jsonl(row_path, dict(
                    record_type="numerical_comparison", sample_id=context["sample_id"],
                    scene=context["scene"], prefix=int(context["prefix"]), shape=shape,
                    stage=operation["record_type"], mode=operation["mode"],
                    pair_id=operation.get("pair_id"), pair_index=operation.get("pair_index"),
                    order=operation.get("order"), input_ids=inputs,
                    keep=keep, compared_with="unmodified GPT.forward", metrics=metrics))
                if not metrics["passed"]:
                    raise RuntimeError(
                        f"{operation['record_type']} {operation['mode']} differs from "
                        f"unmodified forward for {context['sample_id']} shape={shape}: {metrics}"
                    )
                restored = restore(context, cache, keep, operation["record_type"], operation)
                validate_restored(restored["after"], base_cache, context,
                                  operation["record_type"])
                observation = dict(operation, cuda_seconds=result["cuda_seconds"],
                                   host_wall_seconds=result["host_wall_seconds"],
                                   restore_cuda_seconds=restored["cuda_seconds"],
                                   restore_host_wall_seconds=restored["host_wall_seconds"],
                                   numeric_match=metrics)
                if operation["record_type"] == "warmup":
                    warmup_results.append(observation)
                else:
                    pair_results[operation["pair_id"]].append(observation)
                    if len(pair_results[operation["pair_id"]]) == 2:
                        record_pair(operation["pair_id"], pair_results[operation["pair_id"]])
            return dict(shape=shape, reference=ref_metrics,
                        warmup_forwards=len(warmup_results), pairs=len(pair_results),
                        pair_summaries=[dict(
                            sample_id=context["sample_id"], shape=shape,
                            pair_id=pair_id,
                            original_host_seconds=next(item["host_wall_seconds"] for item in pair
                                                       if item["mode"] == "original"),
                            grouped_host_seconds=next(item["host_wall_seconds"] for item in pair
                                                      if item["mode"] == "grouped"),
                            original_cuda_seconds=next(item["cuda_seconds"] for item in pair
                                                       if item["mode"] == "original"),
                            grouped_cuda_seconds=next(item["cuda_seconds"] for item in pair
                                                      if item["mode"] == "grouped"))
                            for pair_id, pair in pair_results.items()])

        for context_index, context in enumerate(selected):
            checkpoint("context_setup", context["sample_id"])
            progress.update(current_sample_id=context["sample_id"], current_stage="context_setup")
            _write_json(output / "progress.json", progress)
            prefix = [int(token) for token in context["context_prefix"]]
            keep = len(prefix) - 1
            if keep < 1:
                raise ValueError("saved context has no processed prefix before pending token")
            checkpoint("cache_allocate", context["sample_id"])
            allocation_started = time.perf_counter()
            cache = cache_for(core, len(prefix) + 8)
            allocation_seconds = time.perf_counter() - allocation_started
            _append_jsonl(row_path, dict(record_type="cache_allocate",
                                         sample_id=context["sample_id"], scene=context["scene"],
                                         prefix=int(context["prefix"]),
                                         host_wall_seconds=allocation_seconds,
                                         cuda_seconds=0.0, kv_keep=keep))

            core.forward = original_forward
            prefill = prefix[:-1]
            prefill_result = run_target(context, prefill, None, "prefill", 0, cache)
            if int(cache.get_pos()) != keep:
                raise RuntimeError("prefix replay did not reach saved keep position")
            base_cache = _cache_snapshot(cache, torch)
            metadata["prefill_cuda_seconds_total"] = (
                metadata.get("prefill_cuda_seconds_total", 0.0) +
                prefill_result["cuda_seconds"])
            metadata["prefill_host_seconds_total"] = (
                metadata.get("prefill_host_seconds_total", 0.0) +
                prefill_result["host_wall_seconds"])
            completed_shapes = []
            for shape in (1, 2, 4):
                checkpoint("shape", context["sample_id"])
                completed_shapes.append(one_shape(context_index, context, cache, keep,
                                                  base_cache, shape))

            # Reproduce the odd-start Feature prediction after processing pending F.
            checkpoint("odd_start_prefill", context["sample_id"])
            odd_prefill = run_target(context, [int(context["target_inputs"][0])], None,
                                     "odd_start_prefill", keep, cache)
            odd_keep = keep + 1
            odd_base = _cache_snapshot(cache, torch)
            odd_input = [int(context["target_inputs"][1])]
            odd_reference = run_target(context, odd_input, None, "odd_reference",
                                       odd_keep, cache)
            restored = restore(context, cache, odd_keep, "odd_reference")
            validate_restored(restored["after"], odd_base, context, "odd_reference")
            odd_results = []
            for mode in ("original", "grouped"):
                result = run_target(context, odd_input, mode, "odd_candidate",
                                    odd_keep, cache)
                metrics = _compare_runs(odd_reference, result, shared, torch)
                _append_jsonl(row_path, dict(
                    record_type="numerical_comparison", sample_id=context["sample_id"],
                    scene=context["scene"], prefix=int(context["prefix"]),
                    stage="odd_start", mode=mode, input_ids=odd_input,
                    keep=odd_keep, predicted_kind="feature",
                    compared_with="unmodified GPT.forward", metrics=metrics))
                if not metrics["passed"]:
                    raise RuntimeError(
                        f"odd-start {mode} differs at {context['sample_id']}: {metrics}"
                    )
                restored = restore(context, cache, odd_keep, "odd_candidate")
                validate_restored(restored["after"], odd_base, context, "odd_candidate")
                odd_results.append(dict(mode=mode, numeric_match=metrics))
            restore(context, cache, keep, "odd_start_cleanup")
            validate_restored(_cache_snapshot(cache, torch), base_cache,
                              context, "odd-start cleanup")
            _append_jsonl(row_path, dict(
                record_type="odd_start_correctness", sample_id=context["sample_id"],
                scene=context["scene"], prefix=int(context["prefix"]),
                pending_feature_input=int(context["target_inputs"][0]),
                position_input=int(odd_input[0]), input_ids=odd_input,
                expected_prediction_kind="feature", keep_before=keep,
                odd_start_keep=odd_keep,
                unmodified_host_wall_seconds=odd_reference["host_wall_seconds"],
                candidate_modes=odd_results, prefill_host_wall_seconds=odd_prefill[
                    "host_wall_seconds"], measured_as_performance=False))
            progress["completed_contexts"] = context_index + 1
            metadata["completed_shapes"] = metadata.get("completed_shapes", []) + [
                dict(sample_id=context["sample_id"], shapes=completed_shapes)]
            _write_json(output / "metadata.json", metadata)
            _write_json(output / "progress.json", progress)

        if state["stop_requested"] or time.monotonic() >= deadline:
            raise StopRequested("shared 900-second deadline or stop signal reached")
        planned_counts = _planned_counts(len(selected))
        if metadata["target_call_counts"] != planned_counts["target_call_counts"]:
            raise RuntimeError(
                f"completed Target call counts differ from the planned schedule: "
                f"{metadata['target_call_counts']} != {planned_counts['target_call_counts']}")
        if metadata["kv_restore_count"] != planned_counts["kv_restores"]:
            raise RuntimeError("completed KV restore count differs from the planned schedule")
        output_rows = _read_jsonl(row_path)
        pair_rows = _read_jsonl(pair_path)
        if sum(row.get("record_type") == "numerical_comparison" for row in output_rows) != (
                planned_counts["numeric_comparisons"]):
            raise RuntimeError("completed numeric comparison count differs from the planned schedule")
        if len(pair_rows) != planned_counts["paired_comparisons"]:
            raise RuntimeError("completed paired comparison count differs from the planned schedule")
        metadata["planned_counts"] = planned_counts
        report.update(status="complete", reason="all selected real contexts completed",
                      elapsed_seconds=time.monotonic() - started_mono)
    except StopRequested as error:
        report.update(status="partial", reason=str(error),
                      elapsed_seconds=time.monotonic() - started_mono)
    except BaseException as error:
        report.update(status="failed", reason=f"{type(error).__name__}: {error}",
                      elapsed_seconds=time.monotonic() - started_mono)
        report["traceback"] = __import__("traceback").format_exc()
    finally:
        if core is not None:
            try:
                if "helper_marker" in locals():
                    for name, original_attribute in (
                        ("forward", original_forward_attr),
                        ("_get_rope", original_get_rope_attr),
                        ("_project_typed_logits_grouped", original_grouped_helper),
                    ):
                        if original_attribute is helper_marker:
                            if name in core.__dict__:
                                delattr(core, name)
                        else:
                            setattr(core, name, original_attribute)
                if capture is not None:
                    capture.close()
            except Exception:
                pass
        if thread_runtime_entered:
            try:
                thread_runtime.__exit__(None, None, None)
            except Exception as error:
                metadata["thread_limit_restore_error"] = f"{type(error).__name__}: {error}"
        if fp32_runtime_entered:
            try:
                fp32_runtime.__exit__(None, None, None)
                metadata["precision_after_restore"] = _precision_state(torch)
            except Exception as error:
                metadata["precision_restore_error"] = f"{type(error).__name__}: {error}"
        report["target_call_counts"] = metadata["target_call_counts"]
        report["target_call_attempts"] = metadata["target_call_attempts"]
        report["kv_restore_count"] = metadata["kv_restore_count"]
        report["signal_number"] = state["signal_number"]
        report["deadline_reached"] = time.monotonic() >= deadline
        report["budget_overrun"] = report["deadline_reached"]
        metadata["status"] = report["status"]
        metadata["status_reason"] = report["reason"]
        metadata["elapsed_seconds"] = report["elapsed_seconds"]
        try:
            report["timing_summary"] = _timing_summary(pair_path)
        except Exception as error:
            report["timing_summary_error"] = f"{type(error).__name__}: {error}"
        report["deadline_reached"] = time.monotonic() >= deadline
        report["budget_overrun"] = report["deadline_reached"]
        if report["status"] == "complete" and (
                state["stop_requested"] or time.monotonic() >= deadline):
            report.update(status="partial", reason="shared 900-second deadline or stop signal reached",
                          elapsed_seconds=time.monotonic() - started_mono,
                          deadline_reached=time.monotonic() >= deadline,
                          budget_overrun=time.monotonic() >= deadline)
            metadata["status"] = report["status"]
            metadata["status_reason"] = report["reason"]
            metadata["elapsed_seconds"] = report["elapsed_seconds"]
        progress.update(status=report["status"], reason=report["reason"],
                        elapsed_seconds=report["elapsed_seconds"],
                        completed_target_calls=metadata["target_call_counts"]["total"],
                        signal_number=state["signal_number"])
        _write_json(output / "metadata.json", metadata)
        _write_json(output / "progress.json", progress)
        _write_json(output / "summary.json", report)
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
        signal.signal(signal.SIGTERM, old_signal)
        signal.signal(signal.SIGALRM, old_alarm)
    return 0 if report["status"] == "complete" else (2 if report["status"] == "partial" else 1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/root/workspace/GaussianGPT"))
    parser.add_argument("--contexts", type=Path, default=DEFAULT_CONTEXTS)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--vqvae", type=Path, default=DEFAULT_VQVAE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--sample-ids", nargs="+", default=list(DEFAULT_SAMPLE_IDS))
    parser.add_argument("--min-sources", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--paired-repeats", type=int, default=12)
    parser.add_argument("--deadline-seconds", type=int, default=900)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.self_test:
        result = _self_test()
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if args.output is None:
        parser.error("--output is required for a GPU run")
    if args.deadline_seconds != 900:
        parser.error("the shared microbenchmark budget is fixed at 900 seconds")
    if args.warmups != 2 or args.paired_repeats != 12:
        parser.error("the approved probe uses two warmups and twelve paired repeats")
    if not 8 <= len(args.sample_ids) <= 20:
        parser.error("--sample-ids must contain between 8 and 20 values")
    if not 4 <= args.min_sources <= len(args.sample_ids):
        parser.error("--min-sources must be between 4 and sample count")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
