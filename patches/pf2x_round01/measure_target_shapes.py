#!/usr/bin/env python3
"""Measure legal two- and four-token Target calls from collected W1 contexts."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Sequence

from analyze_cost_probe import conditional_second_q


P_VOCAB = 8000
F_VOCAB = 4096
EOS = P_VOCAB + F_VOCAB
MAX_BRANCHES = 96


def _path(value: Path, root: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


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


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                allow_nan=False) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _balanced(rows: list[dict], limit: int, seed: int,
              q1_vectors, raw_logits, q2_mode: str) -> tuple[list[dict], int]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    unsupported = 0
    for row in rows:
        if row.get("p1_accepted_actual") is True and row.get("f1_accepted_actual") is False:
            if row.get("f1_residual_local_id") is None:
                continue
            vector_index = int(row["vector_index"])
            try:
                conditional_second_q(q1_vectors[vector_index].tolist(),
                                     int(row["f1a_local_id"]), q2_mode,
                                     raw_logits[vector_index].tolist())
            except ValueError:
                unsupported += 1
                continue
            grouped.setdefault((row["scene"], int(row["prefix"])), []).append(row)
    rng = random.Random(seed)
    for values in grouped.values():
        rng.shuffle(values)
    keys = sorted(grouped)
    selected = []
    while len(selected) < limit:
        changed = False
        for key in keys:
            if grouped[key]:
                selected.append(grouped[key].pop())
                changed = True
                if len(selected) == limit:
                    break
        if not changed:
            break
    return selected, unsupported


def _sample(probabilities, rng) -> int:
    total = sum(float(value) for value in probabilities)
    if total <= 0.0:
        raise ValueError("cannot sample a zero-mass distribution")
    point = rng.random() * total
    cumulative = 0.0
    last = 0
    for index, value in enumerate(probabilities):
        last = index
        cumulative += float(value)
        if point < cumulative:
            return index
    return last


def _timed_call(torch, fn, device) -> tuple[object, float, float]:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record(torch.cuda.current_stream(device))
    started = time.perf_counter()
    result = fn()
    end.record(torch.cuda.current_stream(device))
    torch.cuda.synchronize(device)
    return result, begin.elapsed_time(end) / 1000.0, time.perf_counter() - started


def _measure_maintenance(torch, fn, device) -> tuple[float, float]:
    _, cuda_seconds, wall_seconds = _timed_call(torch, fn, device)
    return cuda_seconds, wall_seconds


def _append_shape(path: Path, context: dict, shape: str, inputs: list[int],
                  before: int, cuda_seconds: float, wall_seconds: float,
                  extras: dict | None = None) -> None:
    row = dict(record_type="target_forward", sample_id=context["sample_id"],
               scene=context["scene"], prefix=int(context["prefix"]),
               shape=shape, input_tokens=len(inputs), input_ids=[int(token) for token in inputs],
               kv_position_before=before, cuda_seconds=cuda_seconds,
               wall_seconds=wall_seconds, additional_target_forward=True)
    if extras:
        row.update(extras)
    _append_jsonl(path, row)


def _append_kv(path: Path, context: dict, operation: str,
               cuda_seconds: float, wall_seconds: float, before: int, after: int,
               details: dict | None = None) -> None:
    row = dict(record_type="kv_operation", sample_id=context["sample_id"],
               scene=context["scene"], prefix=int(context["prefix"]),
               operation=operation, cuda_seconds=cuda_seconds,
               wall_seconds=wall_seconds, kv_before=before, kv_after=after,
               additional_diagnostic_work=True)
    if details:
        row.update(details)
    _append_jsonl(path, row)


def _append_non_target(path: Path, context: dict, operation: str,
                       cuda_seconds: float, wall_seconds: float, detail: dict | None = None) -> None:
    row = dict(record_type="non_target_operation", sample_id=context["sample_id"],
               scene=context["scene"], prefix=int(context["prefix"]),
               operation=operation, cuda_seconds=cuda_seconds,
               wall_seconds=wall_seconds, additional_diagnostic_work=True)
    if detail:
        row.update(detail)
    _append_jsonl(path, row)


def _shape_schedule(context_index: int, sample_id: str,
                    pair_repeats: int) -> list[dict]:
    if context_index < 0 or not 1 <= pair_repeats <= 16:
        raise ValueError("paired timing schedule is outside its bounds")
    shapes = ("short_2", "next_4")
    warmup_order = shapes if context_index % 2 == 0 else tuple(reversed(shapes))
    schedule = [dict(record_type="shape_prewarm", sample_id=sample_id,
                     pair_id=f"{sample_id}:warmup", pair_index=None,
                     order=order, pair_order=order, pair_start_shape=warmup_order[0],
                     shape=shape)
                for order, shape in enumerate(warmup_order)]
    for pair_index in range(pair_repeats):
        first = shapes[(context_index + pair_index) % 2]
        pair_order = (first, shapes[1] if first == shapes[0] else shapes[0])
        pair_id = f"{sample_id}:pair:{pair_index:02d}"
        schedule.extend(dict(record_type="shape_pair_forward", sample_id=sample_id,
                             pair_id=pair_id, pair_index=pair_index,
                             order=order, pair_order=order, pair_start_shape=first,
                             shape=shape)
                        for order, shape in enumerate(pair_order))
    return schedule


def _target_timing_category(operation: dict) -> str:
    record_type = operation.get("record_type")
    if record_type == "shape_prewarm":
        return "warmup"
    if record_type == "shape_pair_forward":
        return "paired"
    raise ValueError(f"unexpected shape timing record_type: {record_type!r}")


def _execute_shape_schedule(cache, keep: int, inputs_by_shape: dict[str, list[int]],
                            schedule: Sequence[dict], forward, restore,
                            device_name: str) -> list[dict]:
    records = []
    for operation in schedule:
        shape = operation["shape"]
        inputs = [int(token) for token in inputs_by_shape[shape]]
        before = int(cache.get_pos())
        if before != keep:
            raise RuntimeError(f"timing input starts at KV {before}, expected keep={keep}")
        cuda_seconds, host_wall_seconds = forward(shape, inputs, operation)
        after = int(cache.get_pos())
        expected_after = keep + len(inputs)
        if after != expected_after:
            raise RuntimeError(f"{shape} advanced KV to {after}, expected {expected_after}")
        restore_cuda_seconds, restore_host_wall_seconds = restore(keep, operation)
        after_restore = int(cache.get_pos())
        if after_restore != keep:
            raise RuntimeError(f"restore ended at KV {after_restore}, expected keep={keep}")
        record = dict(operation)
        record.update(input_tokens=len(inputs), input_ids=inputs,
                      kv_position_before=before, kv_position_after=after,
                      kv_position_after_restore=after_restore, device=device_name,
                      cuda_seconds=cuda_seconds, host_wall_seconds=host_wall_seconds,
                      restore_cuda_seconds=restore_cuda_seconds,
                      restore_host_wall_seconds=restore_host_wall_seconds,
                      timing_scope="forward_only_restore_separate",
                      additional_target_forward=True)
        records.append(record)
    return records


def _restore_branch4_to_keep(cache, keep: int, restore) -> dict:
    before = int(cache.get_pos())
    if before != keep + 4:
        raise RuntimeError(f"branch_4 ended at KV {before}, expected keep+4={keep + 4}")
    details = dict(timing_kind="shape_pair_setup")
    cuda_seconds, host_wall_seconds = restore(keep, details)
    after = int(cache.get_pos())
    if after != keep:
        raise RuntimeError(f"branch_4 restore ended at KV {after}, expected keep={keep}")
    return dict(kv_position_before=before, kv_position_after=after,
                restore_cuda_seconds=cuda_seconds,
                restore_host_wall_seconds=host_wall_seconds)


def _cpu_schedule_test() -> dict:
    context_ids = [f"sample-{index}" for index in range(4)]
    inputs = {"short_2": [8101, 8012],
              "next_4": [8101, 8012, 9003, 7004]}
    schedule = []
    for context_index, sample_id in enumerate(context_ids):
        schedule.extend(_shape_schedule(context_index, sample_id, 8))
    warmups = [row for row in schedule if row["record_type"] == "shape_prewarm"]
    pairs: dict[str, list[dict]] = {}
    for row in schedule:
        if row["record_type"] == "shape_pair_forward":
            pairs.setdefault(row["pair_id"], []).append(row)
    if len(warmups) != 8 or len(pairs) != 32:
        raise AssertionError("warmup or paired-forward count differs from the default cap")
    if any({row["shape"] for row in rows} != {"short_2", "next_4"} or len(rows) != 2
           for rows in pairs.values()):
        raise AssertionError("each pair must contain one short and one next forward")
    for context_index, sample_id in enumerate(context_ids):
        context_schedule = [row for row in schedule if row["sample_id"] == sample_id and
                            row["record_type"] == "shape_pair_forward"]
        for pair_index in range(8):
            rows = [row for row in context_schedule if row["pair_index"] == pair_index]
            expected_first = ("short_2" if (context_index + pair_index) % 2 == 0
                              else "next_4")
            if [row["shape"] for row in rows] != [
                    expected_first, "next_4" if expected_first == "short_2" else "short_2"]:
                raise AssertionError("paired shape order did not alternate by context and pair")

    class FakeCache:
        def __init__(self, position: int):
            self.position = position

        def get_pos(self) -> int:
            return self.position

    called_inputs = []
    restore_calls = []
    setup_restores = []
    records = []
    for context_index, sample_id in enumerate(context_ids):
        cache = FakeCache(16)
        classified_calls = {"warmup": 0, "paired": 0}

        def forward(shape: str, token_ids: list[int], operation: dict):
            if token_ids != inputs[shape]:
                raise AssertionError("schedule changed a fixed input sequence")
            classified_calls[_target_timing_category(operation)] += 1
            called_inputs.append((shape, list(token_ids)))
            cache.position += len(token_ids)
            return 0.001, 0.002

        def restore(keep: int, operation: dict):
            restore_calls.append(dict(operation))
            if operation.get("timing_kind") == "shape_pair_setup":
                setup_restores.append((cache.position, keep))
            cache.position = keep
            return 0.0001, 0.0002

        setup = _restore_branch4_to_keep(cache, 12, restore)
        if setup["kv_position_before"] != 16 or setup["kv_position_after"] != 12:
            raise AssertionError("branch_4 setup did not truncate once to keep")
        context_records = _execute_shape_schedule(
            cache, 12, inputs, _shape_schedule(context_index, sample_id, 8),
            forward, restore, "simulated-device")
        if classified_calls != {"warmup": 2, "paired": 16}:
            raise AssertionError(
                f"context {sample_id} classified calls incorrectly: {classified_calls}")
        records.extend(context_records)
    if len(records) != 72 or len(called_inputs) != 72 or len(restore_calls) != 76:
        raise AssertionError("simulation did not count every forward and restore")
    if any(row["kv_position_before"] != 12 or row["kv_position_after_restore"] != 12
           for row in records):
        raise AssertionError("keep position was not restored before the next call")
    if any(row["kv_position_after"] != 12 + len(inputs[row["shape"]])
           for row in records):
        raise AssertionError("forward did not reach the expected KV position")
    return dict(passed=True, contexts=len(context_ids), warmup_calls=len(warmups),
                pair_count=len(pairs), paired_forward_calls=len(pairs) * 2,
                total_timing_forwards=len(schedule), simulated_forwards=len(called_inputs),
                simulated_restores=len(restore_calls),
                per_context_call_categories={"warmup": 2, "paired": 16},
                setup_truncations_for_default_contexts=len(setup_restores),
                maximum_added_truncations=len(restore_calls),
                fixed_inputs=True, branch_sampling_calls=0)


def _position_tail(ranker, shared, choose_position, fork_position_state,
                   ranks: Sequence[int], prefix: list[int],
                   p1: int, f1: int, p2: int):
    import time

    entered = list(prefix) + [int(p1), int(f1)]
    started = time.perf_counter()
    online = shared.position_state(entered, ranks)
    replay_seconds = time.perf_counter() - started
    started = time.perf_counter()
    second = choose_position(ranker, online, int(p1))
    p2_seconds = time.perf_counter() - started
    if second is None or int(second[0]) != int(p2):
        raise ValueError("replayed Position ranker did not reproduce the saved P2")
    started = time.perf_counter()
    shadow = fork_position_state(online)
    shadow.finalize(int(p2))
    fork_seconds = time.perf_counter() - started
    started = time.perf_counter()
    third = choose_position(ranker, shadow, int(p2))
    p3_seconds = time.perf_counter() - started
    return (int(third[0]) if third is not None else None,
            dict(ranker_replay_seconds=replay_seconds,
                 ranker_p2_seconds=p2_seconds,
                 ranker_fork_finalize_seconds=fork_seconds,
                 ranker_p3_seconds=p3_seconds))


def _load_target(target_path: Path, vqvae_path: Path):
    from model.gaussian_gpt import GaussianGPT
    from model.gaussian_vqvae import GaussianVQVAE

    vqvae = GaussianVQVAE.load_from_checkpoint(str(vqvae_path), map_location="cpu").eval()
    wrapper = GaussianGPT.load_from_checkpoint(
        str(target_path), map_location="cpu", vqvae=vqvae).eval()
    gpt = wrapper.gpt._orig_mod if hasattr(wrapper.gpt, "_orig_mod") else wrapper.gpt
    if int(gpt.eos_token_id) != EOS:
        raise RuntimeError("Target EOS id differs from the existing PF decoder")
    if int(wrapper.position_vocab_size) != P_VOCAB or int(wrapper.feature_vocab_size) != F_VOCAB:
        raise RuntimeError("Target Position/Feature vocabulary sizes differ")
    if str(wrapper.model_config.chunk_order) != "xyz":
        raise RuntimeError("Target coordinate order differs from the existing project")
    if not wrapper.model_config.sparse_position_hard_constraints:
        raise RuntimeError("Target sparse Position constraints are disabled")
    return gpt.cuda().float().eval().requires_grad_(False)


def _run(args) -> int:
    project_root = args.project_root.resolve()
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))
    collection = _path(args.collection_dir, project_root)
    target_path = _path(args.target_checkpoint, project_root)
    vqvae_path = _path(args.vqvae_checkpoint, project_root)
    ranker_path = _path(args.ranker, project_root)
    feature_path = _path(args.feature_checkpoint, project_root)
    head_path = _path(args.feature_head, project_root)
    output = _path(args.output_dir, project_root)
    for required in (collection / "metadata.json", collection / "contexts.jsonl",
                     collection / "vectors.npz", target_path, vqvae_path,
                     ranker_path, feature_path, head_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output.exists():
        raise FileExistsError(output)
    if not 1 <= args.max_contexts <= MAX_BRANCHES:
        raise ValueError(f"max_contexts must be in [1, {MAX_BRANCHES}]")
    if not 1 <= args.pair_contexts <= 20:
        raise ValueError("pair_contexts must be in [1, 20]")
    if not 1 <= args.pair_repeats <= 16:
        raise ValueError("pair_repeats must be in [1, 16]")
    if str(json.loads((collection / "metadata.json").read_text()).get("feature_step")) != str(args.feature_step):
        raise ValueError("feature_step differs from collection metadata")
    if str(Path(json.loads((collection / "metadata.json").read_text()).get(
            "feature_checkpoint", "")).resolve()) != str(feature_path):
        raise ValueError("feature checkpoint differs from collection metadata")
    output.mkdir(parents=True, exist_ok=False)

    import numpy as np
    import torch
    from joblib import load
    from threadpoolctl import threadpool_limits
    from bench_fast_hgb import FastHistRanker
    import bench_gaussiangpt_pf_block2_cow as bench
    import collect_feature_teacher_from_s3 as teacher
    import gpt_position_spec_decode_smoke as position
    from draft_a import pf_ranker_eval as shared
    from draft_a.feature_acceptance_eval import cache_for
    from draft_a.learning_eval import _fp32_runtime

    metadata = json.loads((collection / "metadata.json").read_text(encoding="utf-8"))
    context_rows = _read_jsonl(collection / "contexts.jsonl")
    with np.load(collection / "vectors.npz", allow_pickle=False) as vectors:
        q1_vectors = vectors["q1"].astype("float64")
        raw_logits = vectors["raw_logits"].astype("float64")
    selected, unsupported_q2_contexts = _balanced(
        context_rows, args.max_contexts, args.selection_seed,
        q1_vectors, raw_logits, args.q2_mode)
    pair_context_limit = min(args.pair_contexts, len(selected))
    row_by_id = {row["sample_id"]: row for row in context_rows}
    vector_by_id = {row["sample_id"]: int(row["vector_index"]) for row in context_rows}
    metadata_out = dict(
        status="running", project_root=str(project_root), collection_dir=str(collection.resolve()),
        target_checkpoint=str(target_path), vqvae_checkpoint=str(vqvae_path), ranker=str(ranker_path),
        feature_checkpoint=str(feature_path), feature_step=str(args.feature_step),
        q2_mode=args.q2_mode, max_contexts=args.max_contexts,
        paired_context_limit=args.pair_contexts, paired_repeats=args.pair_repeats,
        paired_contexts_available=pair_context_limit,
        maximum_added_pair_forwards=pair_context_limit * (2 + 2 * args.pair_repeats),
        maximum_added_pair_truncations=pair_context_limit * (3 + 2 * args.pair_repeats),
        selected_contexts=len(selected), selection_seed=args.selection_seed,
        unsupported_q2_contexts=unsupported_q2_contexts,
        policy=dict(target_precision="FP32", feature_precision="FP32",
                    temperature=0.9, top_p=0.9),
        additional_target_calls="all calls below are diagnostic and separate from W1 decoding",
        target_call_counts=dict(prefill=0, original_shape=0, warmup=0, paired=0, total=0),
        all_diagnostic_target_calls=0,
        kv_truncation_counts=dict(all=0, pair_setup=0, prewarm_restore=0,
                                  pair_restore=0, other=0),
        tree_attention="not used",
    )
    _write_json(output / "metadata.json", metadata_out)
    for name in ("shape_timings.jsonl", "branch_rows.jsonl"):
        (output / name).touch()
    if not selected:
        metadata_out.update(status="complete", elapsed_seconds=0.0,
                            reason="no accepted P1 / rejected F1 context has positive q2 support",
                            branch_rows=0, shape_rows=0,
                            target_call_counts=dict(prefill=0, original_shape=0,
                                                    warmup=0, paired=0, total=0),
                            all_diagnostic_target_calls=0,
                            kv_truncation_counts=dict(all=0, pair_setup=0,
                                                      prewarm_restore=0, pair_restore=0,
                                                      other=0),
                            resource_cap=dict(maximum_branches=MAX_BRANCHES,
                                              maximum_contexts=args.max_contexts,
                                              paired_context_limit=args.pair_contexts,
                                              paired_repeats=args.pair_repeats))
        _write_json(output / "metadata.json", metadata_out)
        _write_json(output / "progress.json", dict(status="complete", contexts=0,
                                                     branches=0, output=str(output)))
        print(json.dumps(dict(status="complete", contexts=0, branches=0,
                              output=str(output)), ensure_ascii=False))
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Target shape measurements")
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision("high")
    hidden_p1_rows = []
    hidden_p2_short_rows = []
    hidden_p2_branch_rows = []
    branch_rows = []
    outer_rows = []
    target_call_totals = dict(prefill=0, original_shape=0, warmup=0, paired=0)
    truncate_totals = dict(all=0, pair_setup=0, prewarm_restore=0,
                           pair_restore=0, other=0)
    paired_contexts_with_p3 = 0
    paired_contexts_skipped_no_p3 = 0
    started = time.perf_counter()

    with torch.inference_mode(), _fp32_runtime(torch), \
            threadpool_limits(limits=1, user_api="openmp"):
        target = _load_target(target_path, vqvae_path)
        core = target.transformer
        feature, _, _ = bench.load_feature_model(
            feature_path, head_path, core, torch.device("cuda"), P_VOCAB, EOS)
        ranker = FastHistRanker(load(ranker_path))
        ranks = core.sparse_position_rank_map.cpu().tolist()
        device = torch.device("cuda")

        for sample_index, selected_row in enumerate(selected):
            context = row_by_id[selected_row["sample_id"]]
            vector_index = vector_by_id[context["sample_id"]]
            q1 = q1_vectors[vector_index].tolist()
            raw = raw_logits[vector_index].tolist()
            prefix = [int(token) for token in context["context_prefix"]]
            if not prefix or prefix[0] != EOS or prefix[-1] != int(context["target_inputs"][0]):
                raise ValueError("saved source prefix does not end with the pending Feature")
            p1 = int(context["p1_id"])
            f1a = P_VOCAB + int(context["f1a_local_id"])
            p2 = int(context["p2_id"])
            f1_residual = P_VOCAB + int(context["f1_residual_local_id"])
            q2 = conditional_second_q(q1, int(context["f1a_local_id"]),
                                      args.q2_mode, raw)
            branch_rng = random.Random(args.branch_seed + sample_index * 1000003)
            f1_branch_local = _sample(q2, branch_rng)
            if f1_branch_local == int(context["f1a_local_id"]):
                raise AssertionError("second Feature repeated the first candidate")
            f1_branch = P_VOCAB + f1_branch_local
            before_context = len(prefix) - 1
            cache = None
            capture = None
            sample_started = time.perf_counter()
            sample_target_calls = dict(prefill=0, original_shape=0, warmup=0, paired=0)
            sample_truncations = dict(all=0, pair_setup=0, prewarm_restore=0,
                                      pair_restore=0, other=0)
            p3 = None
            next_inputs = None
            try:
                allocation_started = time.perf_counter()
                cache = cache_for(core, len(prefix) + 12)
                torch.cuda.synchronize(device)
                _append_kv(output / "shape_timings.jsonl", context, "kv_allocate",
                           0.0, time.perf_counter() - allocation_started, 0, 0)
                capture = bench.CaptureLastBlock(core)

                def target_call(inputs: list[int], category: str):
                    if category not in sample_target_calls:
                        raise ValueError(f"unknown Target call category: {category}")
                    kinds = shared.typed_kinds(int(cache.get_pos()), len(inputs))
                    result, cuda_seconds, host_wall_seconds = _timed_call(
                        torch, lambda: capture.run(core, cache, inputs, kinds, device), device)
                    sample_target_calls[category] += 1
                    return result, cuda_seconds, host_wall_seconds

                if len(prefix) > 1:
                    prefill_tokens = prefix[:-1]
                    _, prefill_cuda, prefill_wall = target_call(prefill_tokens, "prefill")
                    if int(cache.get_pos()) != before_context:
                        raise ValueError("prefix replay ended at a different KV position")
                    _append_kv(output / "shape_timings.jsonl", context, "prefix_replay",
                               prefill_cuda, prefill_wall, 0, int(cache.get_pos()))

                main_inputs = [prefix[-1], p1, f1a, p2]
                main_result, main_cuda, main_wall = target_call(main_inputs, "original_shape")
                main_logits, main_states = main_result
                if int(cache.get_pos()) != before_context + 4:
                    raise ValueError("four-token main call advanced KV by the wrong amount")
                h_p1 = main_states[1].detach().clone()
                _append_shape(output / "shape_timings.jsonl", context, "main_4",
                              main_inputs, before_context, main_cuda, main_wall)
                hidden_p1_rows.append(h_p1.float().cpu().numpy().copy())

                def rollback(keep: int, operation: str,
                             details: dict | None = None) -> tuple[float, float]:
                    old = int(cache.get_pos())
                    cuda_seconds, wall_seconds = _measure_maintenance(
                        torch, lambda: bench.truncate(cache, keep, old), device)
                    sample_truncations["all"] += 1
                    timing_kind = (details or {}).get("timing_kind")
                    if timing_kind == "shape_pair_setup":
                        sample_truncations["pair_setup"] += 1
                    elif timing_kind == "shape_prewarm":
                        sample_truncations["prewarm_restore"] += 1
                    elif timing_kind == "shape_pair_forward":
                        sample_truncations["pair_restore"] += 1
                    else:
                        sample_truncations["other"] += 1
                    _append_kv(output / "shape_timings.jsonl", context, operation,
                               cuda_seconds, wall_seconds, old, int(cache.get_pos()),
                               details=details)
                    return cuda_seconds, wall_seconds

                rollback(before_context + 2, "restore_after_main4")
                short_inputs = [f1_residual, p2]
                short_before = int(cache.get_pos())
                short_result, short_cuda, short_wall = target_call(short_inputs, "original_shape")
                short_logits, short_states = short_result
                target_p2 = shared.policy(short_logits[0]).float()
                p2_accept_probability = float(target_p2[p2])
                p2_eos_probability = float(target_p2[EOS])
                _append_shape(output / "shape_timings.jsonl", context, "short_2",
                              short_inputs, short_before, short_cuda, short_wall,
                              dict(p2_accept_probability=p2_accept_probability,
                                   p2_eos_probability=p2_eos_probability,
                                   branch_feature_id=int(context["f1_residual_local_id"]),
                                   next_round_generated_tokens=context.get(
                                       "next_round_generated_tokens")))
                hidden_p2_short_rows.append(short_states[1].detach().float().cpu().numpy().copy())
                rollback(before_context + 2, "restore_after_short2")

                p3, rank_times = _position_tail(
                    ranker, shared, position.choose_position,
                    bench.fork_position_state, ranks,
                    prefix, p1, f1_residual, p2)
                for operation, seconds in rank_times.items():
                    _append_non_target(output / "shape_timings.jsonl", context,
                                       operation, 0.0, seconds)

                def feature_candidate(feature_id: int, salt: int):
                    feature_rng = torch.Generator(device=device)
                    feature_rng.manual_seed(args.branch_seed + sample_index * 1000003 + salt)
                    previous_feature = torch.tensor([feature_id], device=device)
                    position_id = torch.tensor([p2], device=device)
                    started_mlp = time.perf_counter()
                    event_begin = torch.cuda.Event(enable_timing=True)
                    event_end = torch.cuda.Event(enable_timing=True)
                    event_begin.record(torch.cuda.current_stream(device))
                    with torch.autocast(device_type="cuda", enabled=False):
                        _, logits = feature(h_p1.unsqueeze(0), previous_feature, position_id)
                    event_end.record(torch.cuda.current_stream(device))
                    torch.cuda.synchronize(device)
                    mlp_cuda = event_begin.elapsed_time(event_end) / 1000.0
                    mlp_wall = time.perf_counter() - started_mlp
                    q = teacher.policy_batch(logits.float(), 0.9, 0.9)[0]
                    sample_started_inner = time.perf_counter()
                    token = int(torch.multinomial(q, 1, generator=feature_rng).item())
                    _append_non_target(output / "shape_timings.jsonl", context,
                                       "feature_forward", mlp_cuda, mlp_wall,
                                       dict(input_position=p2, previous_feature=feature_id))
                    _append_non_target(output / "shape_timings.jsonl", context,
                                       "feature_policy_and_draw", 0.0,
                                       time.perf_counter() - sample_started_inner)
                    return P_VOCAB + token

                if p3 is not None:
                    f2_residual = feature_candidate(int(context["f1_residual_local_id"]), 101)
                    next_inputs = [f1_residual, p2, f2_residual, p3]
                    next_before = int(cache.get_pos())
                    next_result, next_cuda, next_wall = target_call(
                        next_inputs, "original_shape")
                    if int(cache.get_pos()) != next_before + 4:
                        raise ValueError("next four-token call advanced KV by the wrong amount")
                    _append_shape(output / "shape_timings.jsonl", context, "next_4",
                                  next_inputs, next_before, next_cuda, next_wall,
                                  dict(branch_feature_id=int(context["f1_residual_local_id"]),
                                       p3_id=p3))
                    rollback(before_context + 2, "restore_after_next4")
                else:
                    _append_jsonl(output / "shape_timings.jsonl", dict(
                        record_type="shape_unavailable", sample_id=context["sample_id"],
                        scene=context["scene"], prefix=int(context["prefix"]),
                        shape="next_4", reason="ranker returned no legal P3"))

                branch_inputs = [f1_branch, p2]
                branch_before = int(cache.get_pos())
                branch_result, branch_cuda, branch_wall = target_call(
                    branch_inputs, "original_shape")
                if int(cache.get_pos()) != branch_before + 2:
                    raise ValueError("branch two-token call advanced KV by the wrong amount")
                branch_logits, branch_states = branch_result
                p2_distribution = shared.policy(branch_logits[0]).float()
                p2_probability = float(p2_distribution[p2])
                p2_eos_branch = float(p2_distribution[EOS])
                hidden_p2_branch_rows.append(branch_states[1].detach().float().cpu().numpy().copy())
                _append_shape(output / "shape_timings.jsonl", context, "branch_2",
                              branch_inputs, branch_before, branch_cuda, branch_wall,
                              dict(p2_accept_probability=p2_probability,
                                   p2_eos_probability=p2_eos_branch,
                                   branch_feature_id=f1_branch_local))
                branch_rows.append(dict(
                    sample_id=context["sample_id"], scene=context["scene"],
                    prefix=int(context["prefix"]), q2_mode=args.q2_mode,
                    first_reject_local_id=int(context["f1a_local_id"]),
                    second_candidate_local_id=f1_branch_local,
                    q2_candidate_probability=float(q2[f1_branch_local]),
                    p2_accept_probability=p2_probability,
                    p2_eos_probability=p2_eos_branch,
                    conditional_forward="Target call from KV replayed through the accepted P1",
                    additional_target_forward=True,
                ))
                rollback(before_context + 2, "restore_after_branch2")

                if p3 is not None:
                    f2_branch = feature_candidate(f1_branch_local, 211)
                    branch_next_inputs = [f1_branch, p2, f2_branch, p3]
                    branch_next_before = int(cache.get_pos())
                    _, branch_next_cuda, branch_next_wall = target_call(
                        branch_next_inputs, "original_shape")
                    if int(cache.get_pos()) != branch_next_before + 4:
                        raise ValueError("branch four-token call advanced KV by the wrong amount")
                    _append_shape(output / "shape_timings.jsonl", context, "branch_4",
                                  branch_next_inputs, branch_next_before,
                                  branch_next_cuda, branch_next_wall,
                                  dict(branch_feature_id=f1_branch_local,
                                       next_feature_id=f2_branch - P_VOCAB,
                                       p3_id=p3))
                else:
                    _append_jsonl(output / "shape_timings.jsonl", dict(
                        record_type="shape_unavailable", sample_id=context["sample_id"],
                        scene=context["scene"], prefix=int(context["prefix"]),
                        shape="branch_4", reason="ranker returned no legal P3"))

                if sample_index < pair_context_limit:
                    if p3 is None or next_inputs is None:
                        paired_contexts_skipped_no_p3 += 1
                        _append_jsonl(output / "shape_timings.jsonl", dict(
                            record_type="shape_repeat_unavailable",
                            sample_id=context["sample_id"], scene=context["scene"],
                            prefix=int(context["prefix"]),
                            reason="ranker returned no legal P3; no replacement context selected"))
                    else:
                        paired_contexts_with_p3 += 1
                        _restore_branch4_to_keep(
                            cache, before_context + 2,
                            lambda keep, details: rollback(
                                keep, "restore_after_branch4_before_timing_repeats",
                                details=details))
                        fixed_inputs = dict(short_2=list(short_inputs),
                                            next_4=list(next_inputs))
                        schedule = _shape_schedule(
                            sample_index, context["sample_id"], args.pair_repeats)

                        def repeat_forward(shape: str, inputs: list[int], operation: dict):
                            category = _target_timing_category(operation)
                            _, cuda_seconds, host_wall_seconds = target_call(inputs, category)
                            return cuda_seconds, host_wall_seconds

                        def repeat_restore(keep: int, operation: dict):
                            details = dict(
                                timing_kind=operation["record_type"],
                                pair_id=operation["pair_id"],
                                pair_index=operation["pair_index"],
                                pair_order=operation["pair_order"],
                                order=operation["order"],
                                shape=operation["shape"])
                            return rollback(keep, "restore_after_timing_shape", details=details)

                        repeat_records = _execute_shape_schedule(
                            cache, before_context + 2, fixed_inputs, schedule,
                            repeat_forward, repeat_restore, str(device))
                        for repeat_row in repeat_records:
                            repeat_row.update(scene=context["scene"],
                                              prefix=int(context["prefix"]))
                            _append_jsonl(output / "shape_timings.jsonl", repeat_row)
            finally:
                if capture is not None:
                    capture.close()
                cache = None
                torch.cuda.synchronize(device)
            sample_target_calls["total"] = sum(sample_target_calls.values())
            for category in target_call_totals:
                target_call_totals[category] += sample_target_calls[category]
            for category in truncate_totals:
                truncate_totals[category] += sample_truncations[category]
            outer_rows.append(dict(record_type="outerwall", sample_id=context["sample_id"],
                                   scene=context["scene"], prefix=int(context["prefix"]),
                                   diagnostic_sequence_wall_seconds=time.perf_counter() - sample_started,
                                   target_call_counts=sample_target_calls,
                                   kv_truncation_counts=sample_truncations,
                                   original_shape_target_calls=sample_target_calls["original_shape"],
                                   prefill_target_calls=sample_target_calls["prefill"],
                                   warmup_target_calls=sample_target_calls["warmup"],
                                   paired_target_calls=sample_target_calls["paired"],
                                   all_diagnostic_target_calls=sample_target_calls["total"],
                                   additional_target_calls=sample_target_calls["total"],
                                   ordinary_decode_wall=False))
            print(json.dumps(dict(event="shape_sample", sample_id=context["sample_id"],
                                  completed=sample_index + 1, total=len(selected)),
                             ensure_ascii=False), flush=True)

    for row in branch_rows:
        _append_jsonl(output / "branch_rows.jsonl", row)
    for row in outer_rows:
        _append_jsonl(output / "shape_timings.jsonl", row)
    hidden_size = int(target.transformer.lm_head.weight.size(1))
    def stack_or_empty(rows):
        return np.stack(rows).astype("float32") if rows else np.empty((0, hidden_size), dtype="float32")
    np.savez_compressed(
        output / "branch_states.npz",
        target_hidden_p1=stack_or_empty(hidden_p1_rows),
        target_hidden_p2_after_short=stack_or_empty(hidden_p2_short_rows),
        target_hidden_p2_after_branch=stack_or_empty(hidden_p2_branch_rows),
    )
    metadata_out.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                        selected_contexts=len(selected), branch_rows=len(branch_rows),
                        paired_contexts_with_p3=paired_contexts_with_p3,
                        paired_contexts_skipped_no_p3=paired_contexts_skipped_no_p3,
                        target_call_counts=dict(target_call_totals,
                                                total=sum(target_call_totals.values())),
                        all_diagnostic_target_calls=sum(target_call_totals.values()),
                        kv_truncation_counts=truncate_totals,
                        paired_measurement_limits=dict(
                            contexts_requested=args.pair_contexts,
                            contexts_available=pair_context_limit,
                            repeats_per_context=args.pair_repeats,
                            maximum_forward_calls=pair_context_limit *
                            (2 + 2 * args.pair_repeats),
                            maximum_restore_calls=pair_context_limit *
                            (3 + 2 * args.pair_repeats),
                            no_new_q2_or_branch_rows=True),
                        branch_state_path=str((output / "branch_states.npz").resolve()),
                        shape_rows=sum(1 for row in _read_jsonl(output / "shape_timings.jsonl")
                                       if row.get("record_type") == "target_forward"),
                        resource_cap=dict(maximum_branches=MAX_BRANCHES,
                                          maximum_contexts=args.max_contexts))
    _write_json(output / "metadata.json", metadata_out)
    _write_json(output / "progress.json", dict(status="complete", contexts=len(selected),
                                                 branches=len(branch_rows), output=str(output)))
    print(json.dumps(dict(status="complete", contexts=len(selected),
                          branches=len(branch_rows), output=str(output)), ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--vqvae-checkpoint", type=Path, required=True)
    parser.add_argument("--ranker", type=Path, required=True)
    parser.add_argument("--feature-checkpoint", type=Path, required=True)
    parser.add_argument("--feature-head", type=Path, required=True)
    parser.add_argument("--feature-step", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--q2-mode", choices=("narrow", "wide"), required=True)
    parser.add_argument("--max-contexts", type=int, default=MAX_BRANCHES)
    parser.add_argument("--pair-contexts", type=int, default=4)
    parser.add_argument("--pair-repeats", type=int, default=8)
    parser.add_argument("--selection-seed", type=int, default=20261021)
    parser.add_argument("--branch-seed", type=int, default=20261022)
    parser.add_argument("--cpu-test", action="store_true",
                        help="run the standard-library paired timing schedule simulation")
    parser.add_argument("--show-plan", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args == ["--cpu-test"]:
        print(json.dumps(_cpu_schedule_test(), ensure_ascii=False, sort_keys=True))
        return 0
    args = parser.parse_args(raw_args)
    if args.cpu_test:
        parser.error("--cpu-test must be used by itself")
    if not args.execute:
        if not args.show_plan:
            parser.error("pass --show-plan or --execute")
        if not 1 <= args.max_contexts <= MAX_BRANCHES:
            parser.error(f"--max-contexts must be in [1, {MAX_BRANCHES}]")
        if not 1 <= args.pair_contexts <= 20:
            parser.error("--pair-contexts must be in [1, 20]")
        if not 1 <= args.pair_repeats <= 16:
            parser.error("--pair-repeats must be in [1, 16]")
        planned_pair_contexts = min(args.pair_contexts, args.max_contexts)
        print(json.dumps(dict(project_root=str(args.project_root),
                              collection_dir=str(args.collection_dir),
                              feature_checkpoint=str(args.feature_checkpoint),
                              feature_step=str(args.feature_step), q2_mode=args.q2_mode,
                              maximum_contexts=args.max_contexts,
                              maximum_branches=args.max_contexts,
                              paired_contexts=planned_pair_contexts,
                              paired_repeats=args.pair_repeats,
                              maximum_pair_target_forwards=planned_pair_contexts *
                              (2 + 2 * args.pair_repeats),
                              maximum_pair_truncations=planned_pair_contexts *
                              (3 + 2 * args.pair_repeats),
                              additional_target_forwards="diagnostic only"),
                         ensure_ascii=False))
        return 0
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
