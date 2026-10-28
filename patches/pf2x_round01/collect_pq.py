#!/usr/bin/env python3
"""Collect bounded Target/Feature distributions from the existing PF decoder."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Sequence


PREFIXES = (64, 256)
SOURCES = 10
MAX_CONTEXTS = 24
MAX_BUDGET = 256
P_VOCAB = 8000
F_VOCAB = 4096
EOS = P_VOCAB + F_VOCAB
PARITY_STATS = (
    "target_calls", "draft_calls", "ranker_calls", "target_input_tokens",
    "draft_input_tokens", "rounds", "proposed", "verified", "accepted",
    "accepted_lengths", "round_generated_lengths", "position_fallbacks",
    "position_proposed", "position_verified", "position_accepted",
    "feature_proposed", "feature_verified", "feature_accepted",
)
SAVE_RESERVE_SECONDS = 60.0
_STOP_REQUESTED = False


def _path(value: Path, root: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _read_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seed list must be non-empty and unique")
    return seeds


def _append_jsonl(path: Path, rows: Sequence[dict] | dict) -> None:
    if isinstance(rows, dict):
        rows = (rows,)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_jsonl_atomic(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_npz_atomic(np, path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _request_stop(signum, frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _stop_reason(deadline_unix: float | None) -> str | None:
    if _STOP_REQUESTED:
        return "sigterm"
    if deadline_unix is not None and deadline_unix - time.time() <= SAVE_RESERVE_SECONDS:
        return "deadline_write_reserve"
    return None


def _selected_contexts(reservoirs: dict, np) -> tuple[list[dict], object, object, object]:
    selected = []
    for cell in sorted(reservoirs):
        for source_row in reservoirs[cell]:
            row = {key: value for key, value in source_row.items() if key != "vector_values"}
            row["sample_id"] = "{}|{}|{}|{}|{}".format(
                row["scene"], row["prefix"], row["seed_base"], row["seed"], row["round_index"])
            row["vector_index"] = len(selected)
            row["policy"] = dict(temperature=0.9, top_p=0.9,
                                 target="FP32 Target policy", q1="FP32 MLP policy")
            row["source_split"] = "development"
            row["original_w1_path"] = "draft_a.pf_feature_unified.decode"
            selected.append((row, source_row["vector_values"]))
    if selected:
        p_f = np.stack([values["p_f"] for _, values in selected]).astype("float32")
        q1 = np.stack([values["q1"] for _, values in selected]).astype("float32")
        raw_logits = np.stack([values["raw_logits"] for _, values in selected]).astype("float32")
    else:
        p_f = np.empty((0, F_VOCAB), dtype="float32")
        q1 = np.empty((0, F_VOCAB), dtype="float32")
        raw_logits = np.empty((0, F_VOCAB), dtype="float32")
    return [row for row, _ in selected], p_f, q1, raw_logits


def _snapshot(np, output: Path, metadata: dict, records: list[dict], seeds: list[int],
              args, reservoirs: dict, seen: dict, actual_counts: dict,
              run_rows: list[dict], profiles: list[dict], run_count: int,
              started: float, status: str, stop_reason: str | None = None,
              current: dict | None = None) -> None:
    context_rows, p_f, q1, raw_logits = _selected_contexts(reservoirs, np)
    _write_npz_atomic(np, output / "vectors.npz", p_f=p_f, q1=q1, raw_logits=raw_logits)
    _write_jsonl_atomic(output / "contexts.jsonl", context_rows)
    _write_jsonl_atomic(output / "w1_runs.jsonl", run_rows)
    _write_jsonl_atomic(output / "w1_profiles.jsonl", profiles)
    count_rows = []
    for cell in sorted(actual_counts):
        counts = dict(actual_counts[cell])
        proposals = counts["p1_proposals"]
        counts["p1_probability_mean"] = (
            counts["p1_probability_sum"] / proposals if proposals else None)
        counts["p1_acceptance_actual_rate"] = (
            counts["p1_accepted"] / proposals if proposals else None)
        count_rows.append(counts)
    shortfalls = [dict(scene=scene, prefix=prefix,
                       requested=args.max_per_source_prefix,
                       collected=len(reservoirs[(scene, prefix)]),
                       attempted=seen[(scene, prefix)])
                  for scene, prefix in sorted(reservoirs)
                  if len(reservoirs[(scene, prefix)]) < args.max_per_source_prefix]
    metadata.update(
        status=status, elapsed_seconds=time.perf_counter() - started,
        completed_runs=run_count, sample_count=len(context_rows),
        expected_runs=len(records) * len(PREFIXES) * len(seeds),
        sample_shortfalls=shortfalls, by_source_prefix=count_rows)
    if stop_reason is not None:
        metadata["stop_reason"] = stop_reason
    elif "stop_reason" in metadata:
        metadata.pop("stop_reason")
    _write_json(output / "metadata.json", metadata)
    progress = dict(
        status=status, completed_runs=run_count,
        expected_runs=len(records) * len(PREFIXES) * len(seeds),
        collected_contexts=len(context_rows),
        sample_limit=SOURCES * len(PREFIXES) * args.max_per_source_prefix,
        output=str(output))
    if current is not None:
        progress["current"] = current
    if stop_reason is not None:
        progress["stop_reason"] = stop_reason
    _write_json(output / "progress.json", progress)


class _Probe:
    def __init__(self, metadata: dict, torch):
        self.metadata = metadata
        self.torch = torch
        self.events: list[dict] = []
        self.operations: list[dict] = []
        self.pending_mlp = None
        self.next_mlp_context = None
        self.active_event = None
        self.shadow_ids: set[int] = set()
        self.round_index = -1

    def start_round(self) -> None:
        self.round_index += 1
        self.active_event = None

    def capture_mlp(self, logits, q):
        context = self.next_mlp_context
        if context is None:
            return
        raw = logits.detach().float()[0]
        q_row = q.detach().float()[0]
        self.pending_mlp = dict(context=context, raw_logits=raw, q1=q_row,
                                q_data_ptr=q_row.data_ptr(), f1a_local_id=None)
        self.next_mlp_context = None

    def capture_q_draw(self, probabilities, token: int) -> None:
        if (self.pending_mlp is not None and probabilities.numel() == F_VOCAB
                and probabilities.data_ptr() == self.pending_mlp["q_data_ptr"]):
            self.pending_mlp["f1a_local_id"] = int(token)

    def start_target_block(self, tokens, logits):
        self.active_event = None
        pending = self.pending_mlp
        if pending is None or len(tokens) != 4 or pending["f1a_local_id"] is None:
            return
        row = dict(
            context=pending["context"], round_index=self.round_index,
            target_inputs=[int(token) for token in tokens],
            raw_mlp_logits=pending["raw_logits"], q1=pending["q1"],
            target_p1_logits=logits[0].detach().float(),
            target_f1_logits=logits[1].detach().float(),
            p1_id=int(pending["context"]["p1_id"]),
            f1a_local_id=int(pending["f1a_local_id"]),
            p2_id=int(tokens[3]), decisions=[], f1_residual_local_id=None,
        )
        if row["target_inputs"][0] != row["context"]["prefix_tokens"][-1]:
            raise ValueError("captured pending Feature differs from the source prefix")
        if row["target_inputs"][1] != row["p1_id"]:
            raise ValueError("captured P1 differs from the sampled context")
        if row["target_inputs"][2] != P_VOCAB + row["f1a_local_id"]:
            raise ValueError("captured F1 differs from the same-call MLP sample")
        self.events.append(row)
        self.active_event = row
        self.pending_mlp = None

    def materialize(self, shared) -> list[dict]:
        torch = self.torch
        result = []
        for index, event in enumerate(self.events):
            target_p1_logits = event.pop("target_p1_logits").detach().float().cpu()
            target_f1_logits = event.pop("target_f1_logits").detach().float().cpu()
            raw_mlp_logits = event.pop("raw_mlp_logits").detach().float().cpu()
            q1 = event.pop("q1").detach().float().cpu()
            p1_distribution = shared.policy(target_p1_logits).float()
            p_f = shared.policy(target_f1_logits).float()[P_VOCAB:EOS]
            p1_probability = float(p1_distribution[event["p1_id"]])
            p_mass = float(p_f.sum())
            if abs(p_mass - 1.0) > 2e-4:
                raise ValueError(f"Target Feature probability mass is {p_mass:.8f}")
            if abs(float(q1.sum()) - 1.0) > 2e-5:
                raise ValueError("MLP proposal mass differs from one")
            event["sample_id"] = "{}|{}|{}|{}|{}".format(
                event["context"]["scene"], event["context"]["prefix"],
                event["context"]["seed_base"], event["context"]["seed"],
                event["round_index"],
            )
            event["p1_accept_probability"] = p1_probability
            event["p_f_mass"] = p_mass
            event["p1_accepted_actual"] = (event["decisions"][0]["accepted"]
                                             if event["decisions"] else None)
            event["f1_accepted_actual"] = (
                event["decisions"][1]["accepted"]
                if event["p1_accepted_actual"] is True and len(event["decisions"]) > 1
                else None
            )
            event["p2_accepted_actual"] = (
                event["decisions"][2]["accepted"]
                if event["f1_accepted_actual"] is True and len(event["decisions"]) > 2
                else None
            )
            event["context"].pop("p1_id", None)
            event["policy"] = dict(temperature=0.9, top_p=0.9,
                                   target="FP32 Target policy", q1="FP32 MLP policy")
            event["vectors"] = dict(
                p_f=p_f.numpy().astype("float32", copy=True),
                q1=q1.numpy().astype("float32", copy=True),
                raw_logits=raw_mlp_logits.numpy().astype("float32", copy=True),
            )
            event["target_inputs"] = [int(token) for token in event["target_inputs"]]
            event["feature_step"] = self.metadata["feature_step"]
            event["feature_checkpoint"] = self.metadata["feature_checkpoint"]
            result.append(event)
        return result


@contextmanager
def _instrument(probe: _Probe, modules: dict, torch):
    unified = modules["unified"]
    shared = modules["shared"]
    teacher = modules["teacher"]
    bench = modules["bench"]
    position = modules["position"]
    capture_cls = bench.CaptureLastBlock
    old_values = {
        "timer_run": unified._StageProfile.run,
        "propose": unified.MLPAdapter.propose,
        "policy_batch": teacher.policy_batch,
        "capture_run": capture_cls.run,
        "draw": shared.draw,
        "decide": bench.decide,
        "residual_sample": bench.residual_sample,
        "fork": bench.fork_position_state,
        "choose": position.choose_position,
    }

    def timed_run(timer, stage, fn, *args, gpu=False, input_tokens=0,
                  operation="", context_tokens=0, **kwargs):
        begin = end = None
        if gpu:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record(torch.cuda.current_stream(timer.device))
        started = time.perf_counter()
        try:
            return old_values["timer_run"](
                timer, stage, fn, *args, gpu=gpu, input_tokens=input_tokens,
                operation=operation, context_tokens=context_tokens, **kwargs)
        finally:
            host_seconds = time.perf_counter() - started
            if end is not None:
                end.record(torch.cuda.current_stream(timer.device))
            probe.operations.append(dict(stage=stage, operation=operation,
                                        input_tokens=int(input_tokens),
                                        context_tokens=int(context_tokens),
                                        host_seconds=host_seconds,
                                        cuda_event_begin=begin, cuda_event_end=end,
                                        round_index=probe.round_index))

    def propose(adapter, prefix, p1, h_prev, rng):
        probe.next_mlp_context = dict(prefix_tokens=[int(token) for token in prefix],
                                      p1_id=int(p1), scene=probe.metadata["active_scene"],
                                      source_path=probe.metadata["active_source_path"],
                                      crop_seed=probe.metadata["active_crop_seed"],
                                      prefix=int(probe.metadata["active_prefix"]),
                                      seed_base=int(probe.metadata["active_seed_base"]),
                                      seed=int(probe.metadata["active_seed"]),
                                      round_index=probe.round_index)
        return old_values["propose"](adapter, prefix, p1, h_prev, rng)

    def policy_batch(logits, temperature, top_p):
        result = old_values["policy_batch"](logits, temperature, top_p)
        if (probe.next_mlp_context is not None and logits.ndim == 2
                and logits.shape[-1] == F_VOCAB and result.ndim == 2
                and result.shape == logits.shape):
            probe.capture_mlp(logits, result)
        return result

    def capture_run(capture, target, cache, tokens, kinds, device):
        result = old_values["capture_run"](capture, target, cache, tokens, kinds, device)
        logits, _ = result
        probe.start_target_block(tokens, logits)
        return result

    def draw(probabilities, rng):
        token = old_values["draw"](probabilities, rng)
        probe.capture_q_draw(probabilities, token)
        return token

    def decide(alpha, rng, device):
        accepted = old_values["decide"](alpha, rng, device)
        if probe.active_event is not None:
            probe.active_event["decisions"].append(
                dict(alpha=float(alpha), accepted=bool(accepted)))
        return accepted

    def residual_sample(p, q, rng, *, atol=1e-8):
        token = old_values["residual_sample"](p, q, rng, atol=atol)
        if probe.active_event is not None:
            probe.active_event["f1_residual_local_id"] = int(token)
        return token

    def fork(source):
        shadow = old_values["fork"](source)
        probe.shadow_ids.add(id(shadow))
        return shadow

    def choose(ranker, causal, previous):
        if id(causal) in probe.shadow_ids:
            probe.shadow_ids.discard(id(causal))
        else:
            probe.start_round()
        return old_values["choose"](ranker, causal, previous)

    unified._StageProfile.run = timed_run
    unified.MLPAdapter.propose = propose
    teacher.policy_batch = policy_batch
    capture_cls.run = capture_run
    shared.draw = draw
    bench.decide = decide
    bench.residual_sample = residual_sample
    bench.fork_position_state = fork
    position.choose_position = choose
    try:
        yield
    finally:
        unified._StageProfile.run = old_values["timer_run"]
        unified.MLPAdapter.propose = old_values["propose"]
        teacher.policy_batch = old_values["policy_batch"]
        capture_cls.run = old_values["capture_run"]
        shared.draw = old_values["draw"]
        bench.decide = old_values["decide"]
        bench.residual_sample = old_values["residual_sample"]
        bench.fork_position_state = old_values["fork"]
        position.choose_position = old_values["choose"]


def _resolve_operations(probe: _Probe, run_metadata: dict) -> list[dict]:
    resolved = []
    for operation in probe.operations:
        row = {key: value for key, value in operation.items()
               if key not in {"cuda_event_begin", "cuda_event_end"}}
        begin = operation["cuda_event_begin"]
        end = operation["cuda_event_end"]
        row["cuda_seconds"] = (begin.elapsed_time(end) / 1000.0
                                if begin is not None and end is not None else 0.0)
        row.update(run_metadata)
        resolved.append(row)
    return resolved


def _trajectory_parity(unified, adapter, ranker, ranks, prompt, budget, seed,
                       probe_metadata: dict, modules: dict, torch) -> dict:
    plain_stats, plain_tokens = unified.decode(
        modules["target"].transformer, adapter, ranker, ranks,
        prompt, budget, seed, profile=False)
    torch.cuda.synchronize()
    probe = _Probe(probe_metadata, torch)
    with _instrument(probe, modules, torch):
        measured_stats, measured_tokens = unified.decode(
            modules["target"].transformer, adapter, ranker, ranks,
            prompt, budget, seed, profile=False)
    torch.cuda.synchronize()
    plain_counts = {key: plain_stats.get(key) for key in PARITY_STATS}
    measured_counts = {key: measured_stats.get(key) for key in PARITY_STATS}
    matching = plain_tokens == measured_tokens and plain_counts == measured_counts
    row = dict(status="passed" if matching else "failed", seed=int(seed),
               budget=int(budget), output_tokens_plain=[int(token) for token in plain_tokens],
               output_tokens_instrumented=[int(token) for token in measured_tokens],
               stats_plain=plain_counts, stats_instrumented=measured_counts,
               target_forwards_each=int(plain_stats["target_calls"]))
    probe.events.clear()
    probe.operations.clear()
    probe.pending_mlp = None
    probe.active_event = None
    return row


def _reservoir_push(reservoir: list[dict], seen: int, row: dict,
                    limit: int, rng: random.Random) -> int:
    seen += 1
    if len(reservoir) < limit:
        reservoir.append(row)
    else:
        slot = rng.randrange(seen)
        if slot < limit:
            reservoir[slot] = row
    return seen


def _select_sources(torch, crops: Path, validate) -> list[dict]:
    raw = torch.load(crops, map_location="cpu", weights_only=True)
    by_scene = {}
    for record in raw:
        validate(record)
        scene = record.get("scene") or Path(record["path"]).parents[1].name
        if len(record["tokens"]) >= 256 and scene not in by_scene:
            by_scene[scene] = dict(record, scene=scene)
    records = [by_scene[name] for name in sorted(by_scene)]
    if len(records) != SOURCES:
        raise ValueError(f"expected {SOURCES} development sources, got {len(records)}")
    return records


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
    crops = _path(args.crops, project_root)
    target_path = _path(args.target_checkpoint, project_root)
    vqvae_path = _path(args.vqvae_checkpoint, project_root)
    ranker_path = _path(args.ranker, project_root)
    feature_path = _path(args.feature_checkpoint, project_root)
    head_path = _path(args.feature_head, project_root)
    output = _path(args.output_dir, project_root)
    for required in (crops, target_path, vqvae_path, ranker_path, feature_path, head_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output.exists():
        raise FileExistsError(output)
    if not 1 <= args.max_per_source_prefix <= MAX_CONTEXTS:
        raise ValueError(f"max_per_source_prefix must be in [1, {MAX_CONTEXTS}]")
    if args.budget < 4 or args.budget > MAX_BUDGET or args.budget % 2:
        raise ValueError(f"budget must be even and within [4, {MAX_BUDGET}]")
    if args.temperature != 0.9 or args.top_p != 0.9:
        raise ValueError("this probe is fixed to temperature=0.9 and top_p=0.9")
    if args.deadline_unix is not None and not math.isfinite(args.deadline_unix):
        raise ValueError("deadline_unix must be finite")
    seeds = _read_seeds(args.seed_bases)
    output.mkdir(parents=True, exist_ok=False)

    import numpy as np
    import torch
    from joblib import load
    from threadpoolctl import threadpool_limits
    from bench_fast_hgb import FastHistRanker
    import bench_gaussiangpt_pf_block2_cow as bench
    import collect_feature_teacher_from_s3 as teacher
    import gpt_position_spec_decode_smoke as position
    from draft_a import learning_eval
    from draft_a import pf_feature_unified as unified
    from draft_a import pf_ranker_eval as shared
    from draft_a.corpus_v2 import validate

    torch.set_num_threads(6)
    records = _select_sources(torch, crops, validate)
    metadata = dict(
        status="running", project_root=str(project_root), crops=str(crops),
        target_checkpoint=str(target_path), vqvae_checkpoint=str(vqvae_path),
        ranker=str(ranker_path), feature_checkpoint=str(feature_path),
        feature_head=str(head_path), feature_step=str(args.feature_step),
        source_count=len(records), prefixes=list(PREFIXES),
        seed_bases=seeds, budget=args.budget,
        max_per_source_prefix=args.max_per_source_prefix,
        maximum_contexts=SOURCES * len(PREFIXES) * args.max_per_source_prefix,
        policy=dict(target_precision="FP32", feature_precision="FP32",
                    temperature=0.9, top_p=0.9),
        data_scope="existing development sources only",
        additional_target_forwards_per_sampled_w1_decode=0,
        trajectory_parity_replays=2,
        trajectory_parity_budget=min(16, args.budget),
        raw_feature_logits="captured from the same MLP forward used for q1",
        timing_scope="W1 operations are instrumented; outer time is diagnostic only",
        selected_sources=[dict(scene=row["scene"], crop_seed=int(row["crop_seed"]),
                               source_path=str(row["path"])) for row in records],
    )
    reservoirs: dict[tuple[str, int], list[dict]] = {}
    seen: dict[tuple[str, int], int] = {}
    actual_counts: dict[tuple[str, int], dict] = {}
    for record in records:
        for prefix_len in PREFIXES:
            cell = (record["scene"], prefix_len)
            reservoirs[cell] = []
            seen[cell] = 0
            actual_counts[cell] = dict(
                scene=record["scene"], prefix=prefix_len, source_rows=0,
                p1_proposals=0, p1_probability_sum=0.0,
                p1_accepted=0, p1_rejected=0, f1_accepted=0,
                f1_rejected=0, seed_runs=0)
    profiles = []
    run_rows = []
    run_count = 0
    started = time.perf_counter()
    _snapshot(np, output, metadata, records, seeds, args, reservoirs, seen,
              actual_counts, run_rows, profiles, run_count, started, "running")

    def partial_result(reason: str) -> int:
        _snapshot(np, output, metadata, records, seeds, args, reservoirs, seen,
                  actual_counts, run_rows, profiles, run_count, started,
                  "partial", stop_reason=reason)
        print(json.dumps(dict(status="partial", completed_runs=run_count,
                              samples=sum(len(rows) for rows in reservoirs.values()),
                              stop_reason=reason, output=str(output)), ensure_ascii=False),
              flush=True)
        return 2

    reason = _stop_reason(args.deadline_unix)
    if reason is not None:
        return partial_result(reason)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for online p/q collection")
    torch.set_float32_matmul_precision("high")

    with torch.inference_mode(), learning_eval._fp32_runtime(torch), \
            threadpool_limits(limits=1, user_api="openmp"):
        reason = _stop_reason(args.deadline_unix)
        if reason is not None:
            return partial_result(reason)
        target = _load_target(target_path, vqvae_path)
        reason = _stop_reason(args.deadline_unix)
        if reason is not None:
            return partial_result(reason)
        ranker = FastHistRanker(load(ranker_path))
        ranks = target.transformer.sparse_position_rank_map.cpu().tolist()
        reason = _stop_reason(args.deadline_unix)
        if reason is not None:
            return partial_result(reason)
        feature, _, _ = bench.load_feature_model(
            feature_path, head_path, target.transformer, torch.device("cuda"), P_VOCAB, EOS)
        adapter = unified.MLPAdapter(feature, bf16=False)
        modules = dict(unified=unified, shared=shared, teacher=teacher,
                       bench=bench, position=position, target=target)

        reason = _stop_reason(args.deadline_unix)
        if reason is not None:
            return partial_result(reason)
        parity_record = records[0]
        parity_prefix = PREFIXES[0]
        parity_seed_base = seeds[0]
        parity_seed = (parity_seed_base + int(parity_record["crop_seed"]) * 1009 +
                       parity_prefix * 17)
        parity_meta = dict(
            scene=parity_record["scene"], source_path=str(parity_record["path"]),
            crop_seed=int(parity_record["crop_seed"]), prefix=parity_prefix,
            seed_base=parity_seed_base, seed=parity_seed,
            feature_step=str(args.feature_step), feature_checkpoint=str(feature_path),
            active_scene=parity_record["scene"],
            active_source_path=str(parity_record["path"]),
            active_crop_seed=int(parity_record["crop_seed"]),
            active_prefix=parity_prefix, active_seed_base=parity_seed_base,
            active_seed=parity_seed)
        parity_budget = min(16, args.budget)
        parity_prompt = [EOS] + parity_record["tokens"][:parity_prefix].tolist()
        parity_result = _trajectory_parity(
            unified, adapter, ranker, ranks, parity_prompt, parity_budget,
            parity_seed, parity_meta, modules, torch)
        _write_json(output / "trajectory_parity.json", parity_result)
        metadata["trajectory_parity_target_calls_each"] = parity_result["target_forwards_each"]
        _snapshot(np, output, metadata, records, seeds, args, reservoirs, seen,
                  actual_counts, run_rows, profiles, run_count, started, "running")
        if parity_result["status"] != "passed":
            raise RuntimeError("instrumented same-seed W1 replay changed tokens or counts")
        reason = _stop_reason(args.deadline_unix)
        if reason is not None:
            return partial_result(reason)

        stop_reason = None
        for scene_index, record in enumerate(records):
            for prefix_len in PREFIXES:
                cell = (record["scene"], prefix_len)
                reservoir_rng = random.Random(args.sample_seed + scene_index * 100003 + prefix_len)
                for seed_base in seeds:
                    stop_reason = _stop_reason(args.deadline_unix)
                    if stop_reason is not None:
                        break
                    prompt = [EOS] + record["tokens"][:prefix_len].tolist()
                    seed = seed_base + int(record["crop_seed"]) * 1009 + prefix_len * 17
                    run_meta = dict(scene=record["scene"], source_path=str(record["path"]),
                                    crop_seed=int(record["crop_seed"]),
                                    prefix=prefix_len, seed_base=seed_base, seed=seed,
                                    feature_step=str(args.feature_step))
                    probe_meta = dict(run_meta, feature_step=str(args.feature_step),
                                      feature_checkpoint=str(feature_path),
                                      active_scene=record["scene"],
                                      active_source_path=str(record["path"]),
                                      active_crop_seed=int(record["crop_seed"]),
                                      active_prefix=prefix_len,
                                      active_seed_base=seed_base, active_seed=seed)
                    probe = _Probe(probe_meta, torch)
                    run_started = time.perf_counter()
                    with _instrument(probe, modules, torch):
                        stats, output_tokens = unified.decode(
                            target.transformer, adapter, ranker, ranks,
                            prompt, args.budget, seed, profile=False)
                    torch.cuda.synchronize()
                    outer_wall = time.perf_counter() - run_started
                    events = probe.materialize(shared)
                    for event in events:
                        p1_accepted = event["p1_accepted_actual"]
                        f1_accepted = event["f1_accepted_actual"]
                        actual_counts[cell]["p1_proposals"] += 1
                        actual_counts[cell]["p1_probability_sum"] += event["p1_accept_probability"]
                        if p1_accepted is True:
                            actual_counts[cell]["p1_accepted"] += 1
                        elif p1_accepted is False:
                            actual_counts[cell]["p1_rejected"] += 1
                        if f1_accepted is True:
                            actual_counts[cell]["f1_accepted"] += 1
                        elif f1_accepted is False:
                            actual_counts[cell]["f1_rejected"] += 1
                        if p1_accepted is not True:
                            continue
                        round_index = int(event["round_index"])
                        round_lengths = stats["round_generated_lengths"]
                        accepted_lengths = stats["accepted_lengths"]
                        event["round_generated_tokens"] = (
                            int(round_lengths[round_index])
                            if 0 <= round_index < len(round_lengths) else None)
                        event["accepted_draft_count"] = (
                            int(accepted_lengths[round_index])
                            if 0 <= round_index < len(accepted_lengths) else None)
                        event["next_round_generated_tokens"] = (
                            int(round_lengths[round_index + 1])
                            if 0 <= round_index + 1 < len(round_lengths) else None)
                        vector_values = event.pop("vectors")
                        context = event.pop("context")
                        event.update(context)
                        event["context_prefix"] = event.pop("prefix_tokens")
                        event["vector_values"] = vector_values
                        seen[cell] = _reservoir_push(
                            reservoirs[cell], seen[cell], event,
                            args.max_per_source_prefix, reservoir_rng)
                    actual_counts[cell]["source_rows"] += len(events)
                    actual_counts[cell]["seed_runs"] += 1
                    resolved_ops = _resolve_operations(probe, run_meta)
                    for op in resolved_ops:
                        op["rounds"] = int(stats["rounds"])
                        op["generated_tokens"] = int(stats["generated_tokens"])
                        op["outer_seconds"] = outer_wall
                        op["timing_instrumented"] = True
                    profiles.extend(resolved_ops)
                    run_rows.append(dict(
                        **run_meta, requested_budget=int(args.budget),
                        generated_tokens=int(stats["generated_tokens"]),
                        output_tokens=[int(token) for token in output_tokens],
                        stats=stats, outer_seconds=outer_wall,
                        timing_instrumented=True))
                    run_count += 1
                    _snapshot(np, output, metadata, records, seeds, args, reservoirs, seen,
                              actual_counts, run_rows, profiles, run_count, started, "running",
                              current=dict(scene=record["scene"], prefix=prefix_len,
                                           seed_base=seed_base))
                    print(json.dumps(dict(event="w1_collection", scene=record["scene"],
                                          prefix=prefix_len, seed_base=seed_base,
                                          blocks=len(events), seconds=outer_wall),
                                     ensure_ascii=False), flush=True)
                    stop_reason = _stop_reason(args.deadline_unix)
                    if stop_reason is not None:
                        break
                if stop_reason is not None:
                    break
            if stop_reason is not None:
                break
    expected_runs = len(records) * len(PREFIXES) * len(seeds)
    if stop_reason is not None and run_count < expected_runs:
        return partial_result(stop_reason)
    _snapshot(np, output, metadata, records, seeds, args, reservoirs, seen,
              actual_counts, run_rows, profiles, run_count, started, "complete")
    sample_count = sum(len(rows) for rows in reservoirs.values())
    print(json.dumps(dict(status="complete", runs=run_count, samples=sample_count,
                          maximum=SOURCES * len(PREFIXES) * args.max_per_source_prefix,
                          output=str(output)), ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True,
                        help="existing development crop file")
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--vqvae-checkpoint", type=Path, required=True)
    parser.add_argument("--ranker", type=Path, required=True)
    parser.add_argument("--feature-checkpoint", type=Path, required=True)
    parser.add_argument("--feature-head", type=Path, required=True)
    parser.add_argument("--feature-step", required=True,
                        help="step label matching the explicitly supplied Feature checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-bases", default="34701,34702,34703")
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--max-per-source-prefix", type=int, default=MAX_CONTEXTS)
    parser.add_argument("--sample-seed", type=int, default=20261020)
    parser.add_argument("--deadline-unix", type=float,
                        help="absolute Unix deadline shared with the pipeline runner")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--show-plan", action="store_true",
                        help="print limits and selected arguments without loading a model")
    parser.add_argument("--execute", action="store_true",
                        help="run the W1 distribution collection on the selected device")
    parser.add_argument("--cpu-test", action="store_true",
                        help="write one synthetic completed trajectory to an isolated output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cpu_test:
        if args.output_dir.exists():
            parser.error("--cpu-test output directory already exists")
        args.output_dir.mkdir(parents=True)
        sample = dict(sample_id="cpu-source|64|34701|34701|0", vector_index=0,
                      source_split="development", scene="cpu-source", prefix=64)
        run_row = dict(scene="cpu-source", prefix=64, seed_base=34701, seed=34701,
                       output_tokens=[12096, 3, 4], stats=dict(target_calls=2, rounds=1))
        _append_jsonl(args.output_dir / "w1_runs.jsonl", run_row)
        _write_jsonl_atomic(args.output_dir / "contexts.jsonl", [sample])
        _write_json(args.output_dir / "metadata.json",
                    dict(status="partial", completed_runs=1, sample_count=1,
                         stop_reason="cpu_test"))
        _write_json(args.output_dir / "progress.json",
                    dict(status="partial", completed_runs=1, collected_contexts=1,
                         stop_reason="cpu_test"))
        return 2
    if not args.execute:
        if not args.show_plan:
            parser.error("pass --show-plan or --execute")
        if not 1 <= args.max_per_source_prefix <= MAX_CONTEXTS:
            parser.error(f"--max-per-source-prefix must be in [1, {MAX_CONTEXTS}]")
        if args.budget < 4 or args.budget > MAX_BUDGET or args.budget % 2:
            parser.error(f"--budget must be even and within [4, {MAX_BUDGET}]")
        print(json.dumps(dict(project_root=str(args.project_root), crops=str(args.crops),
                              feature_checkpoint=str(args.feature_checkpoint),
                              feature_step=str(args.feature_step), prefixes=list(PREFIXES),
                              sources=SOURCES, max_per_source_prefix=args.max_per_source_prefix,
                              maximum_contexts=SOURCES * len(PREFIXES) * args.max_per_source_prefix,
                              budget=args.budget, seed_bases=_read_seeds(args.seed_bases),
                              policy=dict(temperature=args.temperature, top_p=args.top_p)),
                         ensure_ascii=False))
        return 0
    signal.signal(signal.SIGTERM, _request_stop)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
