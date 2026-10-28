#!/usr/bin/env python3
"""Paired Target/PF runtime comparison under an optional attention implementation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("original_target", "original_pf", "candidate_pf", "candidate_target")
COUNTER_FIELDS = (
    "target_calls", "target_input_tokens", "draft_calls", "draft_input_tokens",
    "ranker_calls", "rounds", "proposed", "verified", "accepted",
    "position_fallbacks", "position_proposed", "position_verified",
    "position_accepted", "feature_proposed", "feature_verified",
    "feature_accepted", "proposal_rejections", "mlp_calls",
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def target_graph_context(runtime, position_module=None, base_cache_for=None,
                        decode_function=None):
    """Route this decode through a graph runtime and its stable Target cache."""
    if runtime is None:
        yield
        return
    patched_namespaces = []
    namespaces = []
    cache_factory = None
    if position_module is not None or decode_function is not None:
        if base_cache_for is None:
            raise ValueError("base_cache_for is required when patching the decoder cache factory")
        cache_factory = lambda core, capacity: runtime.cache_for(
            core, capacity, factory=base_cache_for,
        )
    if position_module is not None:
        namespaces.append(position_module.__dict__)
    if decode_function is not None:
        namespace = decode_function.__globals__
        if not any(namespace is saved_namespace for saved_namespace in namespaces):
            namespaces.append(namespace)
    if any("cache_for" not in namespace for namespace in namespaces):
        raise RuntimeError("A Target decode namespace has no cache_for global")
    for namespace in namespaces:
        patched_namespaces.append((namespace, "cache_for", namespace["cache_for"]))
        namespace["cache_for"] = cache_factory
    try:
        with runtime.attention_context(enabled=True):
            yield
    finally:
        for namespace, name, original in reversed(patched_namespaces):
            namespace[name] = original


def graph_event_delta(before: dict | None, after: dict | None) -> dict | None:
    if after is None:
        return None
    old_graphs = before.get("graph_events", []) if before else []
    old_caches = before.get("cache_events", []) if before else []
    new_graphs = after.get("graph_events", [])[len(old_graphs):]
    new_caches = after.get("cache_events", [])[len(old_caches):]
    return dict(
        graph_count_before=len(old_graphs), graph_count_after=len(after.get("graph_events", [])),
        new_graph_events=new_graphs, new_cache_events=new_caches,
        new_graph_setup_ms=math.fsum(
            float(item.get("prepare_ms", 0.0)) + float(item.get("warmup_ms", 0.0))
            + float(item.get("capture_ms", 0.0)) for item in new_graphs
        ),
        runtime_init_ms=(float(after.get("runtime_init_ms", 0.0))
                         if before is None else 0.0),
        new_cache_allocation_host_enqueue_ms=math.fsum(
            float(item.get("host_enqueue_ms", 0.0)) for item in new_caches
            if item.get("event") == "allocate"
        ),
        new_cache_reset_host_enqueue_ms=math.fsum(
            float(item.get("host_enqueue_ms", 0.0)) for item in new_caches
            if item.get("event") == "reset"
        ),
    )


def dynamic_setup_summary(runtime_stats: dict | None, rows: list[dict]) -> dict | None:
    if runtime_stats is None:
        return None
    graph_setup_ms = math.fsum(
        float(event.get("prepare_ms", 0.0)) + float(event.get("warmup_ms", 0.0))
        + float(event.get("capture_ms", 0.0))
        for event in runtime_stats.get("graph_events", [])
    )
    cache_allocation_enqueue_ms = math.fsum(
        float(event.get("host_enqueue_ms", 0.0))
        for event in runtime_stats.get("cache_events", [])
        if event.get("event") == "allocate"
    )
    runtime_init_ms = float(runtime_stats.get("runtime_init_ms", 0.0))
    cold_setup_breakdown_ms = runtime_init_ms + cache_allocation_enqueue_ms + graph_setup_ms
    candidate_decodes = sum(
        row.get("method", "").startswith("candidate_") for row in rows
    )
    return dict(
        runtime_init_ms=runtime_init_ms,
        cache_allocation_host_enqueue_ms=cache_allocation_enqueue_ms,
        graph_prepare_warmup_capture_ms=graph_setup_ms,
        cold_setup_breakdown_ms=cold_setup_breakdown_ms,
        candidate_decode_count=candidate_decodes,
        amortized_setup_breakdown_ms_per_candidate_decode=(
            cold_setup_breakdown_ms / candidate_decodes if candidate_decodes else None
        ),
        accounting=(
            "Breakdown only: runtime construction, cache allocation and graph setup occur inside "
            "measured candidate outer wall time; do not add these values to the wall-time ratios. "
            "The recorded host enqueue and graph setup components need not equal total device allocation time."
        ),
    )


def jsonable(value):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "tolist"):
        return jsonable(value.detach().cpu().tolist())
    if hasattr(value, "item"):
        return jsonable(value.item())
    raise TypeError("Cannot serialize " + type(value).__name__)


def source_name(record: dict) -> str:
    if record.get("scene") or record.get("source"):
        return str(record.get("scene") or record.get("source"))
    path = Path(record.get("path", record.get("name", "unknown")))
    return str(path.parents[1].name if len(path.parents) > 1 else path.parent.name)


def select_sources(raw, validate, requested: int, minimum_prefix: int) -> list[dict]:
    selected = {}
    for record in raw:
        validate(record)
        name = source_name(record)
        if len(record["tokens"]) >= minimum_prefix and name not in selected:
            selected[name] = dict(record, scene=name)
    names = sorted(selected)
    if len(names) < requested:
        raise ValueError(f"Need {requested} sources with {minimum_prefix} tokens; found {len(names)}")
    return [selected[name] for name in names[:requested]]


def build_target_decode(position_decode):
    """Expose the output already assembled by the existing k=0 decoder."""
    import inspect

    source = inspect.getsource(position_decode)
    if source.count("def decode(") != 1 or source.count("return _add_rates(stats)") != 1:
        raise RuntimeError("Position decoder source layout differs from the saved runner copy")
    source = source.replace("def decode(", "def decode_with_output(", 1)
    source = source.replace("return _add_rates(stats)",
                             "return _add_rates(stats), output", 1)
    namespace = dict(position_decode.__globals__)
    exec(compile(source, position_decode.__code__.co_filename, "exec"), namespace)
    return namespace["decode_with_output"]


def quantile(values: list[float], probability: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    point = (len(ordered) - 1) * probability
    lower = int(point)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = point - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def geometric_summary(source_logs: dict[str, float], seed: int = 381012) -> dict:
    names = sorted(source_logs)
    if not names:
        return dict(source_count=0, source_equal_geometric_mean=None,
                    source_bootstrap_95_ci=None, per_source={})
    mean_log = math.fsum(source_logs.values()) / len(source_logs)
    rng = random.Random(seed)
    samples = [math.exp(math.fsum(source_logs[rng.choice(names)] for _ in names) / len(names))
               for _ in range(10000)] if len(names) > 1 else []
    return dict(
        source_count=len(names), source_equal_geometric_mean=math.exp(mean_log),
        source_bootstrap_seed=seed,
        source_bootstrap_95_ci=([quantile(samples, .025), quantile(samples, .975)]
                                if samples else None),
        per_source={name: math.exp(value) for name, value in source_logs.items()},
    )


def summarize(rows: list[dict], prompts: list[dict], implementation: str) -> dict:
    methods = {}
    for method in METHODS:
        chosen = [row for row in rows if row["method"] == method]
        tokens = sum(int(row["generated_tokens"]) for row in chosen)
        seconds = math.fsum(float(row["outer_wall_seconds"]) for row in chosen)
        methods[method] = dict(
            runs=len(chosen), generated_tokens=tokens, outer_wall_seconds=seconds,
            pooled_tokens_per_second=(tokens / seconds if seconds else None),
            eos_count=sum(bool(row["eos"]) for row in chosen),
            output_lengths={str(length): sum(int(row["generated_tokens"]) == length for row in chosen)
                            for length in sorted({int(row["generated_tokens"]) for row in chosen})},
            counters={key: sum(int(row.get(key, 0)) for row in chosen)
                      for key in COUNTER_FIELDS},
        )

    by_group = {}
    for row in rows:
        by_group.setdefault(row["pair_id"], {})[row["method"]] = row
    contrasts = {
        "original_pf_over_original_target": ("original_pf", "original_target"),
        "candidate_pf_over_candidate_target": ("candidate_pf", "candidate_target"),
        "candidate_pf_over_original_target": ("candidate_pf", "original_target"),
        "candidate_pf_over_original_pf": ("candidate_pf", "original_pf"),
        "candidate_target_over_original_target": ("candidate_target", "original_target"),
    }
    ratios = {}
    source_by_group = {prompt["pair_id"]: prompt["source"] for prompt in prompts}
    for name, (numerator, denominator) in contrasts.items():
        source_values = {}
        full_source_values = {}
        for pair_id, group in by_group.items():
            if set(group) != set(METHODS):
                continue
            up, down = group[numerator], group[denominator]
            if not up["outer_wall_seconds"] or not down["outer_wall_seconds"]:
                continue
            source = source_by_group[pair_id]
            value = math.log(float(up["tokens_per_second"]) /
                             float(down["tokens_per_second"]))
            source_values.setdefault(source, []).append(value)
            requested = int(up.get("requested_new_tokens", up.get("budget", 0)))
            if (requested > 0
                    and int(up["generated_tokens"]) == int(down["generated_tokens"])
                    and int(up["generated_tokens"]) == requested):
                full_source_values.setdefault(source, []).append(
                    math.log(float(down["outer_wall_seconds"]) /
                             float(up["outer_wall_seconds"]))
                )
        means = {source: math.fsum(values) / len(values)
                 for source, values in source_values.items() if values}
        full_means = {source: math.fsum(values) / len(values)
                      for source, values in full_source_values.items() if values}
        ratios[name] = dict(
            all_output_lengths=geometric_summary(means),
            common_full_budget=geometric_summary(full_means),
        )

    parity = {}
    for original, candidate, label in (
        ("original_target", "candidate_target", "target"),
        ("original_pf", "candidate_pf", "pf"),
    ):
        pairs = []
        for pair_id, group in by_group.items():
            if original not in group or candidate not in group:
                continue
            left, right = group[original], group[candidate]
            left_counts = {key: left.get(key, 0) for key in COUNTER_FIELDS}
            right_counts = {key: right.get(key, 0) for key in COUNTER_FIELDS}
            first_difference = next(
                (index for index, (a, b) in enumerate(zip(left["output_tokens"],
                                                          right["output_tokens"]))
                 if a != b),
                None,
            )
            if (first_difference is None
                    and len(left["output_tokens"]) != len(right["output_tokens"])):
                first_difference = min(len(left["output_tokens"]), len(right["output_tokens"]))
            pairs.append(dict(
                pair_id=pair_id,
                output_equal=left["output_tokens"] == right["output_tokens"],
                counters_equal=left_counts == right_counts,
                first_token_difference=first_difference,
                lengths=(len(left["output_tokens"]), len(right["output_tokens"])),
            ))
        parity[label] = dict(
            pairs=pairs,
            output_equal_count=sum(item["output_equal"] for item in pairs),
            counters_equal_count=sum(item["counters_equal"] for item in pairs),
        )

    return dict(
        status="complete" if len(rows) == len(prompts) * len(METHODS) else "partial",
        implementation=implementation, expected_runs=len(prompts) * len(METHODS),
        completed_runs=len(rows), prompts=len(prompts), by_method=methods,
        source_equal_geometric_speed_ratios=ratios, same_seed_parity=parity,
        timing_scope=("Synchronized outer wall per decode, including cache creation, prefix prefill, "
                      "proposal, Target calls, recovery, sampling and adapter preparation/close."),
    )


def verify_attention_parity(torch, target, cache_for, capture_type, typed_kinds, policy,
                            attention_context, implementation, prompts, records,
                            feature, output_dir: Path, device, graph_runtime=None) -> dict:
    result = dict(
        status="running", prefix_body_tokens=256, block_sizes=[1, 2, 4],
        candidate_implementation=implementation,
        semantic_acceptance=dict(
            target_logits_support_equal=True,
            target_policy_support_equal=True,
            target_policy_total_variation_max=1e-5,
            target_pre_finalnorm_hidden_finite=True,
            target_pre_finalnorm_hidden_same_shape=True,
            target_pre_finalnorm_hidden_per_row_relative_l2_max=1e-5,
            feature_mlp_q_support_equal=True,
            feature_mlp_q_total_variation_max=1e-5,
            feature_mlp_h_prev_finite=True,
            feature_mlp_h_prev_same_shape=True,
            feature_mlp_h_prev_relative_l2_max=1e-5,
        ),
        elementwise_hidden_atol=1e-4, elementwise_hidden_rtol=1e-4,
        elementwise_hidden_allclose_role="diagnostic only; it does not independently fail semantic acceptance",
        blocks={}, tensor_artifact=str(output_dir / "attention_parity_tensors.pt"),
        feature_q_artifact=str(output_dir / "feature_q_parity_tensors.pt"),
    )
    tensor_rows = {}
    verification_failures = []
    elementwise_hidden_failures = []
    capture = None
    try:
        prompt = next((row for row in prompts if int(row["prefix"]) == 256), None)
        if prompt is None:
            raise ValueError("Attention parity needs a prefix of 256 body tokens")
        record = next((row for row in records if row["scene"] == prompt["source"]), None)
        if record is None:
            raise ValueError("The selected prefix record is missing")
        body = [int(token) for token in record["tokens"]]
        continuation = body[256:260]
        if len(continuation) < 4:
            raise ValueError("The prefix-256 source needs four real continuation tokens")
        prefix_tokens = [int(token) for token in prompt["prompt"]]
        capture = capture_type(target)
        result.update(pair_id=prompt["pair_id"], source=prompt["source"],
                      prefix_token_count=len(prefix_tokens),
                      continuation_tokens=continuation)

        for block_size in (1, 2, 4):
            real_block = continuation[:block_size]
            batch = torch.tensor([prefix_tokens], device=device, dtype=torch.long)
            gathered = {}
            for name, chosen_implementation in (
                    ("original", "default"), ("candidate", implementation)):
                if graph_runtime is not None and name == "candidate":
                    cache = graph_runtime.cache_for(
                        target, len(prefix_tokens) + block_size + 2, factory=cache_for,
                    )
                else:
                    cache = cache_for(target, len(prefix_tokens) + block_size + 2)
                with attention_context(chosen_implementation), target_graph_context(
                        graph_runtime if name == "candidate" else None), torch.autocast(
                        device_type="cuda", enabled=False):
                    target(batch, kv_cache=cache,
                           logit_types=typed_kinds(0, len(prefix_tokens)))
                    if int(cache.get_pos()) != len(prefix_tokens):
                        raise RuntimeError("Prefix prefill advanced to an unexpected cache position")
                    start = int(cache.get_pos())
                    logits, hidden = capture.run(
                        target, cache, real_block, typed_kinds(start, block_size), device,
                    )
                gathered[name] = (logits.float(), hidden.float())

            reference_logits, reference_hidden = gathered["original"]
            candidate_logits, candidate_hidden = gathered["candidate"]
            logits_shape_equal = tuple(reference_logits.shape) == tuple(candidate_logits.shape)
            hidden_shape_equal = tuple(reference_hidden.shape) == tuple(candidate_hidden.shape)
            if not logits_shape_equal or not hidden_shape_equal:
                result["blocks"][str(block_size)] = dict(
                    logits_shape_equal=logits_shape_equal,
                    original_logits_shape=list(reference_logits.shape),
                    candidate_logits_shape=list(candidate_logits.shape),
                    hidden_shape_equal=hidden_shape_equal,
                    original_hidden_shape=list(reference_hidden.shape),
                    candidate_hidden_shape=list(candidate_hidden.shape),
                )
                tensor_rows[str(block_size)] = dict(
                    real_input_tokens=real_block,
                    original_logits=reference_logits.detach().cpu(),
                    candidate_logits=candidate_logits.detach().cpu(),
                    original_pre_finalnorm_hidden=reference_hidden.detach().cpu(),
                    candidate_pre_finalnorm_hidden=candidate_hidden.detach().cpu(),
                )
                torch.save(tensor_rows, output_dir / "attention_parity_tensors.pt")
                verification_failures.append(f"T={block_size}: logits/hidden tensor shapes differ")
                continue
            reference_support = torch.isfinite(reference_logits)
            candidate_support = torch.isfinite(candidate_logits)
            support_equal = bool(torch.equal(reference_support, candidate_support))
            if not support_equal:
                differing_support = int((reference_support != candidate_support).sum().item())
            else:
                differing_support = 0
            logits_finite_on_support = (
                not bool(torch.isnan(reference_logits).any())
                and not bool(torch.isposinf(reference_logits).any())
                and not bool(torch.isnan(candidate_logits).any())
                and not bool(torch.isposinf(candidate_logits).any())
            )
            max_tv = 0.0
            policy_support_equal = True
            for row_index in range(block_size):
                reference_policy = policy(reference_logits[row_index])
                candidate_policy = policy(candidate_logits[row_index])
                if (not bool(torch.isfinite(reference_policy).all())
                        or not bool(torch.isfinite(candidate_policy).all())):
                    raise RuntimeError(f"Attention policy has nonfinite values for T={block_size}")
                row_tv = float(0.5 * (reference_policy - candidate_policy).abs().sum().item())
                if not math.isfinite(row_tv):
                    raise RuntimeError(f"Attention policy TV is nonfinite for T={block_size}")
                max_tv = max(max_tv, row_tv)
                policy_support_equal &= bool(torch.equal(
                    reference_policy > 0, candidate_policy > 0,
                ))
            hidden_delta = (reference_hidden - candidate_hidden).abs()
            max_hidden_abs = float(hidden_delta.max().item())
            max_hidden_relative = float((hidden_delta / reference_hidden.abs().clamp_min(1e-12)).max().item())
            reference_hidden_rms = reference_hidden.square().mean(dim=-1).sqrt().clamp_min(1e-12)
            hidden_rms_normalized_max_abs = float(
                (hidden_delta / reference_hidden_rms.unsqueeze(-1)).max().item()
            )
            hidden_relative_l2 = float(
                (hidden_delta.norm(dim=-1) / reference_hidden.norm(dim=-1).clamp_min(1e-12)).max().item()
            )
            if not math.isfinite(max_hidden_abs) or not math.isfinite(max_hidden_relative):
                raise RuntimeError(f"Attention hidden values are nonfinite for T={block_size}")
            hidden_close = bool(torch.allclose(
                reference_hidden, candidate_hidden, atol=1e-4, rtol=1e-4,
            ))
            result["blocks"][str(block_size)] = dict(
                logits_shape_equal=logits_shape_equal,
                hidden_shape_equal=hidden_shape_equal,
                logits_finite_on_support=logits_finite_on_support,
                hidden_finite=bool(torch.isfinite(reference_hidden).all()
                                   and torch.isfinite(candidate_hidden).all()),
                support_equal=support_equal, differing_support_values=differing_support,
                policy_support_equal=policy_support_equal, max_policy_tv=max_tv,
                max_pre_finalnorm_hidden_abs=max_hidden_abs,
                max_pre_finalnorm_hidden_relative=max_hidden_relative,
                max_pre_finalnorm_hidden_abs_over_row_rms=hidden_rms_normalized_max_abs,
                max_pre_finalnorm_hidden_relative_l2=hidden_relative_l2,
                pre_finalnorm_hidden_close=hidden_close,
                rows=block_size, real_input_tokens=real_block,
            )
            tensor_rows[str(block_size)] = dict(
                real_input_tokens=real_block,
                original_logits=reference_logits.detach().cpu(),
                candidate_logits=candidate_logits.detach().cpu(),
                original_pre_finalnorm_hidden=reference_hidden.detach().cpu(),
                candidate_pre_finalnorm_hidden=candidate_hidden.detach().cpu(),
            )
            torch.save(tensor_rows, output_dir / "attention_parity_tensors.pt")
            if not support_equal:
                verification_failures.append(f"T={block_size}: logits support differs")
            if not logits_finite_on_support:
                verification_failures.append(f"T={block_size}: supported logits are nonfinite")
            if not result["blocks"][str(block_size)]["hidden_finite"]:
                verification_failures.append(f"T={block_size}: hidden values are nonfinite")
            if max_tv > 1e-5:
                verification_failures.append(f"T={block_size}: policy TV exceeds 1e-5")
            if not policy_support_equal:
                verification_failures.append(f"T={block_size}: policy support differs")
            if hidden_relative_l2 > 1e-5:
                verification_failures.append(f"T={block_size}: per-row hidden L2 relative exceeds 1e-5")
            if not hidden_close:
                elementwise_hidden_failures.append(f"T={block_size}")

        # Check the Feature MLP input that drives PF proposals: hidden at the
        # preceding real Position, the real pending Feature, and the next real
        # Position from this same source. This catches MLP distribution changes
        # that can be hidden by a small raw-hidden difference.
        from collect_feature_teacher_from_s3 import policy_batch

        previous_feature_token = int(continuation[1])
        previous_feature = previous_feature_token - 8000
        proposal_position = int(continuation[2])
        if not (0 <= previous_feature < 4096 and 0 <= proposal_position < 8000):
            raise ValueError("Selected real PF context has invalid Feature/Position inputs")
        verify_block = [int(prefix_tokens[-1]), *[int(token) for token in continuation[:3]]]
        if len(verify_block) != 4:
            raise RuntimeError("Feature MLP q parity requires a real T=4 PF verification block")
        feature_cache_capacity = len(prefix_tokens) - 1 + len(verify_block)
        feature_q = {}
        feature_h = {}
        for name, chosen_implementation in (
                ("original", "default"), ("candidate", implementation)):
            if graph_runtime is not None and name == "candidate":
                cache = graph_runtime.cache_for(
                    target, feature_cache_capacity, factory=cache_for,
                )
            else:
                cache = cache_for(target, feature_cache_capacity)
            capture_for_feature = capture_type(target)
            try:
                with attention_context(chosen_implementation), target_graph_context(
                        graph_runtime if name == "candidate" else None), torch.autocast(
                        device_type="cuda", enabled=False):
                    capture_for_feature.run(
                        target, cache, prefix_tokens[:-1],
                        typed_kinds(0, len(prefix_tokens) - 1), device,
                    )
                    if int(cache.get_pos()) != len(prefix_tokens) - 1:
                        raise RuntimeError("Feature MLP context ended at an unexpected cache position")
                    start = int(cache.get_pos())
                    _, verified_hidden = capture_for_feature.run(
                        target, cache, verify_block,
                        typed_kinds(start, len(verify_block)), device,
                    )
                    h_prev = verified_hidden[1].float()
                    _, mlp_logits = feature(
                        h_prev.unsqueeze(0),
                        torch.tensor([previous_feature], device=device, dtype=torch.long),
                        torch.tensor([proposal_position], device=device, dtype=torch.long),
                    )
                    q = policy_batch(mlp_logits.float(), 0.9, 0.9)[0].float()
                if q.shape != (4096,) or not bool(torch.isfinite(q).all()):
                    raise RuntimeError("Feature MLP returned invalid 4096-way probabilities")
                feature_h[name] = h_prev.detach().cpu()
                feature_q[name] = q.detach().cpu()
            finally:
                capture_for_feature.close()

        original_q = feature_q["original"]
        candidate_q = feature_q["candidate"]
        q_support_equal = bool(torch.equal(original_q > 0, candidate_q > 0))
        q_tv = float(0.5 * (original_q - candidate_q).abs().sum().item())
        if not math.isfinite(q_tv):
            raise RuntimeError("Feature MLP policy TV is nonfinite")
        q_hidden_delta = (feature_h["original"] - feature_h["candidate"]).abs()
        q_hidden_abs = float(q_hidden_delta.max().item())
        q_hidden_relative = float((q_hidden_delta / feature_h["original"].abs().clamp_min(1e-12)).max().item())
        q_hidden_rms = feature_h["original"].square().mean().sqrt().clamp_min(1e-12)
        q_hidden_rms_normalized_max_abs = float((q_hidden_delta / q_hidden_rms).max().item())
        q_hidden_relative_l2 = float(
            q_hidden_delta.norm() / feature_h["original"].norm().clamp_min(1e-12)
        )
        q_hidden_finite = bool(
            torch.isfinite(feature_h["original"]).all()
            and torch.isfinite(feature_h["candidate"]).all()
            and torch.isfinite(original_q).all()
            and torch.isfinite(candidate_q).all()
        )
        result["feature_mlp"] = dict(
            h_prev_shape_equal=(tuple(feature_h["original"].shape)
                                == tuple(feature_h["candidate"].shape)),
            h_prev_finite=q_hidden_finite,
            previous_feature_token=previous_feature_token,
            previous_feature_index=int(previous_feature),
            real_proposal_position=proposal_position,
            real_target_verify_tokens=verify_block,
            h_prev_row_in_verify_block=1,
            q_support_equal=q_support_equal,
            differing_q_support_values=int((original_q > 0).ne(candidate_q > 0).sum().item()),
            max_q_policy_tv=q_tv,
            max_pre_finalnorm_h_prev_abs=q_hidden_abs,
            max_pre_finalnorm_h_prev_relative=q_hidden_relative,
            max_pre_finalnorm_h_prev_abs_over_rms=q_hidden_rms_normalized_max_abs,
            max_pre_finalnorm_h_prev_relative_l2=q_hidden_relative_l2,
            q_temperature=0.9, q_top_p=0.9,
            rows=1, q_vocab=4096,
        )
        torch.save(dict(
            source=prompt["source"], pair_id=prompt["pair_id"],
            prefix_tokens=prefix_tokens, real_target_verify_tokens=verify_block,
            previous_feature_token=previous_feature_token,
            real_proposal_position=proposal_position,
            original_pre_finalnorm_h_prev=feature_h["original"],
            candidate_pre_finalnorm_h_prev=feature_h["candidate"],
            original_q=original_q, candidate_q=candidate_q,
        ), output_dir / "feature_q_parity_tensors.pt")
        if not q_support_equal:
            verification_failures.append("Feature MLP q support differs")
        if q_tv > 1e-5:
            verification_failures.append("Feature MLP q policy TV exceeds 1e-5")
        if not q_hidden_finite:
            verification_failures.append("Feature MLP h_prev/q contains nonfinite values")
        if (tuple(feature_h["original"].shape) != tuple(feature_h["candidate"].shape)
                or q_hidden_relative_l2 > 1e-5):
            verification_failures.append("Feature MLP h_prev shape/L2 relative failed")
        result["validation_failures"] = verification_failures
        result["elementwise_hidden_allclose_failures"] = elementwise_hidden_failures
        result["elementwise_hidden_allclose_is_diagnostic_only"] = True
        if graph_runtime is not None:
            result["dynamic_graph_runtime"] = graph_runtime.stats()
        result["status"] = "failed" if verification_failures else "passed"
        if verification_failures:
            raise RuntimeError("Parity failed after all T=1/2/4 and Feature MLP q diagnostics")
        return result
    except Exception as exc:
        result.update(status="failed", error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        if capture is not None:
            capture.close()
        if graph_runtime is not None and "dynamic_graph_runtime" not in result:
            result["dynamic_graph_runtime"] = graph_runtime.stats()
        if result["status"] == "running":
            result.update(status="failed", error=dict(type="RuntimeError", message="verification did not complete"))
        write_json(output_dir / "verification.json", result)


def warmup_modes(torch, target, feature, ranker, ranks, prompts, budget, implementation,
                 attention_context, target_decode, pf_decode, adapter_type,
                 validate_sequence, device, position_module=None,
                 dynamic_graph_class=None, dynamic_graph_enabled=False,
                 graph_cache_capacity=1024) -> dict:
    warm_prompt = next((row for row in prompts if int(row["prefix"]) == 64), prompts[0])
    warmup_budget = 32
    records = []
    started = time.perf_counter()
    graph_runtime = None
    for method in METHODS:
        selected_implementation = "default" if method.startswith("original_") else implementation
        torch.cuda.synchronize()
        begin = time.perf_counter()
        adapter = None
        runtime_before = None
        runtime_created = False
        try:
            if dynamic_graph_enabled and method.startswith("candidate_"):
                if graph_runtime is None:
                    graph_runtime = dynamic_graph_class(
                        target, cache_capacity_buckets=(graph_cache_capacity,),
                    )
                    runtime_created = True
                runtime_before = graph_runtime.stats()
            adapter = adapter_type(feature) if method.endswith("pf") else None
            with attention_context(selected_implementation), target_graph_context(
                    graph_runtime if method.startswith("candidate_") else None,
                    position_module=position_module,
                    base_cache_for=(position_module.cache_for if position_module is not None else None),
                    decode_function=target_decode,
            ), torch.autocast(
                    device_type="cuda", enabled=False):
                if method.endswith("target"):
                    stats, output = target_decode(
                        target, None, ranker, ranks, warm_prompt["prompt"],
                        warmup_budget, 0, warm_prompt["seed"],
                    )
                else:
                    stats, output = pf_decode(
                        target, adapter, ranker, ranks, warm_prompt["prompt"],
                        warmup_budget, warm_prompt["seed"],
                    )
            torch.cuda.synchronize()
        finally:
            if adapter is not None:
                adapter.close()
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        runtime_after = graph_runtime.stats() if graph_runtime is not None and method.startswith("candidate_") else None
        output = [int(token) for token in output]
        validate_sequence(warm_prompt["prompt"], output, ranks)
        record = dict(
            method=method, attention_implementation=selected_implementation,
            requested_tokens=warmup_budget, generated_tokens=len(output),
            wall_seconds=elapsed, counters={key: int(stats.get(key, 0)) for key in COUNTER_FIELDS},
        )
        if runtime_after is not None:
            runtime_delta = graph_event_delta(runtime_before, runtime_after)
            if runtime_created and runtime_delta is not None:
                runtime_delta["runtime_init_ms"] = float(
                    runtime_after.get("runtime_init_ms", 0.0)
                )
            record.update(dynamic_graph=runtime_delta,
                          dynamic_graph_runtime_created=runtime_created)
        records.append(record)
    torch.cuda.synchronize()
    result = dict(
        total_wall_seconds=time.perf_counter() - started, warmup_budget=warmup_budget,
        pair_id=warm_prompt["pair_id"], records=records,
    )
    if graph_runtime is not None:
        result["dynamic_graph_runtime"] = graph_runtime.stats()
    graph_runtime = None
    gc.collect()
    return result


def parse_int_list(value: str, name: str) -> list[int]:
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not result or any(item <= 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError(f"{name} must contain unique positive integers")
    return result


_ACTIVE_PIPELINE_PATH = None
_ACTIVE_PIPELINE = None


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--crops", type=Path, required=True,
                        help="Existing crop records in torch-save format")
    parser.add_argument("--feature-checkpoint", type=Path, default=Path(
        "draft_a/runs/feature_target_continue_20261014/selected.pt"))
    parser.add_argument("--out", type=Path, required=True, help="New result directory")
    parser.add_argument("--implementation", choices=("default", "math", "explicit_bmm"),
                        default="explicit_bmm")
    parser.add_argument("--dynamic-graph", action="store_true",
                        help="Run candidate PF/Target with the isolated dynamic Target Graph runtime")
    parser.add_argument("--graph-cache-capacity", type=int, default=1024,
                        help="Shared KV capacity bucket for the dynamic graph runtime")
    parser.add_argument("--sources", type=int, default=3)
    parser.add_argument("--source-scope", choices=("development", "independent"),
                        default="development",
                        help="Label the supplied crop set in plan and summary metadata")
    parser.add_argument("--prefixes", default="64,256",
                        help="Comma-separated prompt body lengths")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=38101)
    parser.add_argument("--budget", type=int, default=128)
    args = parser.parse_args()

    prefixes = parse_int_list(args.prefixes, "prefixes")
    if 256 not in prefixes:
        parser.error("prefixes must include 256 for attention parity validation")
    if (args.sources < 1 or args.seeds < 1 or args.budget < 2 or args.budget % 2
            or args.graph_cache_capacity < 1):
        parser.error("sources/seeds must be positive and budget must be an even integer >= 2")
    project = args.project.expanduser().resolve()
    crops = args.crops.expanduser()
    crops = crops.resolve() if crops.is_absolute() else (project / crops).resolve()
    feature_checkpoint = args.feature_checkpoint.expanduser()
    feature_checkpoint = (feature_checkpoint.resolve() if feature_checkpoint.is_absolute()
                         else (project / feature_checkpoint).resolve())
    output_dir = args.out.expanduser()
    output_dir = (output_dir.resolve() if output_dir.is_absolute()
                  else (project / output_dir).resolve())
    if output_dir.exists():
        parser.error("--out must name a new directory: " + str(output_dir))
    if not project.is_dir() or not crops.is_file() or not feature_checkpoint.is_file():
        parser.error("Project, crop records or Feature checkpoint are missing")

    os.chdir(project)
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if args.dynamic_graph:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "runtime"))
    import torch
    from joblib import load
    from threadpoolctl import threadpool_limits
    from attention_override import attention_context
    from bench_fast_hgb import FastHistRanker
    from bench_gaussiangpt_pf_block2_cow import CaptureLastBlock, load_feature_model
    from draft_a import feature_target_v2
    from draft_a import pf_ranker_eval as position_module
    from draft_a.corpus_v2 import EOS, load_target, validate
    from draft_a.feature_acceptance_eval import cache_for, policy
    from draft_a.learning_eval import _fp32_runtime
    from draft_a.pf_feature_unified import MLPAdapter, decode as pf_decode
    from draft_a.pf_ranker_eval import decode as position_decode, typed_kinds, validate_sequence
    if args.dynamic_graph:
        from dynamic_target_graph import DynamicTargetGraph
    else:
        DynamicTargetGraph = None

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this runtime comparison")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    output_dir.mkdir(parents=True)
    started = time.time()
    pipeline = dict(
        status="running", started_at_unix=started, project=str(project), crops=str(crops),
        feature_checkpoint=str(feature_checkpoint), output=str(output_dir),
        implementation=args.implementation, methods=list(METHODS), completed_runs=0,
        dynamic_graph_enabled=bool(args.dynamic_graph),
        graph_cache_capacity=args.graph_cache_capacity,
    )
    global _ACTIVE_PIPELINE_PATH, _ACTIVE_PIPELINE
    _ACTIVE_PIPELINE_PATH = output_dir / "pipeline.json"
    _ACTIVE_PIPELINE = pipeline
    write_json(output_dir / "pipeline.json", pipeline)

    raw_records = torch.load(crops, map_location="cpu", weights_only=True)
    records = select_sources(raw_records, validate, args.sources, max(prefixes) + 4)
    prompts = []
    for source_index, record in enumerate(records):
        body = [int(token) for token in record["tokens"]]
        for prefix_index, prefix in enumerate(prefixes):
            for seed_index in range(args.seeds):
                seed_base = args.seed_start + seed_index
                seed = seed_base + int(record.get("crop_seed", record.get("seed", 0))) * 1009 + prefix * 17
                pair_id = f"{source_index:02d}-{prefix:04d}-{seed_index:02d}"
                prompts.append(dict(
                    pair_id=pair_id, source=record["scene"], source_index=source_index,
                    prefix=prefix, prefix_index=prefix_index, seed_index=seed_index,
                    seed=seed, seed_base=seed_base, crop_seed=int(record.get("crop_seed", record.get("seed", 0))),
                    budget=args.budget, prompt=[int(EOS), *body[:prefix]],
                    input_path=str(record.get("path", record.get("name", ""))),
                ))
    if not prompts:
        raise RuntimeError("No prompts were selected")
    write_json(output_dir / "plan.json", dict(
        project=str(project), crops=str(crops), feature_checkpoint=str(feature_checkpoint),
        source_scope=args.source_scope,
        implementation=args.implementation, methods=list(METHODS), sources=[r["scene"] for r in records],
        dynamic_graph_enabled=bool(args.dynamic_graph),
        graph_cache_capacity=args.graph_cache_capacity,
        prefixes=prefixes, seeds=args.seeds, seed_start=args.seed_start, budget=args.budget,
        prompt_count=len(prompts), planned_runs=len(prompts) * len(METHODS),
        temperature=0.9, top_p=0.9, target_precision="FP32",
        matmul_precision=torch.get_float32_matmul_precision(), tf32_enabled=bool(
            torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32),
        warmup_tokens=32, attention_parity=dict(
            prefix_body_tokens=256, block_sizes=[1, 2, 4],
            target_logits_support_equal=True, target_policy_support_equal=True,
            target_policy_tv_max=1e-5, target_hidden_finite_same_shape=True,
            target_hidden_relative_l2_max=1e-5,
            feature_mlp_q_support_equal=True, feature_mlp_q_tv_max=1e-5,
            feature_mlp_h_prev_finite_same_shape=True,
            feature_mlp_h_prev_relative_l2_max=1e-5,
            elementwise_hidden_atol=1e-4, elementwise_hidden_rtol=1e-4,
            elementwise_hidden_is_diagnostic_only=True,
        ),
        method_order_rule="Methods rotate by context index; every context shares the same prompt and seed.",
    ))
    write_json(output_dir / "prompts.json", dict(contexts=prompts, bos_id=int(EOS)))

    with torch.inference_mode(), _fp32_runtime(torch), threadpool_limits(limits=1, user_api="openmp"):
        if (torch.get_float32_matmul_precision() != "highest"
                or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
            raise RuntimeError("FP32 highest precision with TF32 disabled was not established")
        target_wrapper = load_target().float().eval().requires_grad_(False)
        target = target_wrapper.transformer
        ranks = target.sparse_position_rank_map.detach().cpu().tolist()
        if len(ranks) != 8000:
            raise ValueError("Expected 8000 sparse Position ranks")
        head = Path(feature_target_v2.HEAD)
        head = head.resolve() if head.is_absolute() else (project / head).resolve()
        if not head.is_file():
            raise FileNotFoundError("Frozen Feature head is missing: " + str(head))
        feature, feature_temperature, feature_top_p = load_feature_model(
            feature_checkpoint, head, target, device, 8000, EOS,
        )
        if (feature_temperature, feature_top_p) != (0.9, 0.9):
            raise ValueError("The selected Feature MLP must use temperature=.9 and top_p=.9")
        ranker_path = Path(feature_target_v2.RANKER)
        ranker_path = ranker_path.resolve() if ranker_path.is_absolute() else (project / ranker_path).resolve()
        if not ranker_path.is_file():
            raise FileNotFoundError("Position Ranker is missing: " + str(ranker_path))
        ranker = FastHistRanker(load(ranker_path))
        target_decode = build_target_decode(position_decode)
        verification_runtime = (
            DynamicTargetGraph(target, cache_capacity_buckets=(args.graph_cache_capacity,))
            if args.dynamic_graph else None
        )
        verification = verify_attention_parity(
            torch, target, cache_for, CaptureLastBlock, typed_kinds, policy,
            attention_context, args.implementation, prompts, records, feature,
            output_dir, device, graph_runtime=verification_runtime,
        )
        verification_runtime = None
        gc.collect()
        warmup = warmup_modes(
            torch, target, feature, ranker, ranks, prompts, args.budget,
            args.implementation, attention_context, target_decode, pf_decode,
            MLPAdapter, validate_sequence, device, position_module=position_module,
            dynamic_graph_class=DynamicTargetGraph,
            dynamic_graph_enabled=args.dynamic_graph,
            graph_cache_capacity=args.graph_cache_capacity,
        )
        write_json(output_dir / "warmup.json", warmup)
        pipeline.update(
            hardware_software=dict(
                python=sys.version, torch=torch.__version__, cuda_runtime=torch.version.cuda,
                device=torch.cuda.get_device_name(0), matmul_precision=torch.get_float32_matmul_precision(),
                tf32_enabled=False, target_precision="FP32", temperature=0.9, top_p=0.9,
            ),
            head_path=str(head), ranker_path=str(ranker_path),
            target_core=type(target).__name__,
            attention_verification=verification,
            warmup=dict(total_wall_seconds=warmup["total_wall_seconds"],
                        per_mode=warmup["records"],
                        dynamic_graph_runtime=warmup.get("dynamic_graph_runtime")),
            expected_runs=len(prompts) * len(METHODS),
        )
        write_json(output_dir / "pipeline.json", pipeline)
        rows = []
        measured_graph_runtime = None
        for group_index, group in enumerate(prompts):
            order = list(METHODS[group_index % len(METHODS):] + METHODS[:group_index % len(METHODS)])
            for order_index, method in enumerate(order):
                implementation = ("default" if method.startswith("original_")
                                  else args.implementation)
                torch.cuda.synchronize()
                begin = time.perf_counter()
                adapter = None
                runtime = None
                runtime_before = None
                runtime_created = False
                try:
                    if args.dynamic_graph and method.startswith("candidate_"):
                        if measured_graph_runtime is None:
                            measured_graph_runtime = DynamicTargetGraph(
                                target, cache_capacity_buckets=(args.graph_cache_capacity,),
                            )
                            runtime_created = True
                        runtime = measured_graph_runtime
                        runtime_before = runtime.stats()
                    adapter = MLPAdapter(feature) if method.endswith("pf") else None
                    with attention_context(implementation), target_graph_context(
                            runtime, position_module=position_module,
                            base_cache_for=cache_for, decode_function=target_decode), torch.autocast(
                            device_type="cuda", enabled=False):
                        if method.endswith("target"):
                            stats, output = target_decode(
                                target, None, ranker, ranks, group["prompt"],
                                args.budget, 0, group["seed"],
                            )
                        else:
                            stats, output = pf_decode(
                                target, adapter, ranker, ranks, group["prompt"],
                                args.budget, group["seed"],
                            )
                    torch.cuda.synchronize()
                finally:
                    if adapter is not None:
                        adapter.close()
                    torch.cuda.synchronize()
                outer = time.perf_counter() - begin
                runtime_after = runtime.stats() if runtime is not None else None
                runtime_delta = graph_event_delta(runtime_before, runtime_after)
                if runtime_created and runtime_delta is not None:
                    runtime_delta["runtime_init_ms"] = float(runtime_after.get("runtime_init_ms", 0.0))
                token_ids = [int(token) for token in output]
                validate_sequence(group["prompt"], token_ids, ranks)
                stats = dict(stats)
                counters = {key: int(stats.get(key, 0)) for key in COUNTER_FIELDS}
                counters["proposal_rejections"] = max(
                    0, int(stats.get("verified", 0)) - int(stats.get("accepted", 0)))
                if adapter is not None:
                    counters["mlp_calls"] = int(adapter.forward_calls)
                eos = bool(token_ids and token_ids[-1] == EOS)
                row = dict(
                    pair_id=group["pair_id"], source=group["source"],
                    source_index=group["source_index"], prefix=group["prefix"],
                    prefix_index=group["prefix_index"], seed_index=group["seed_index"],
                    crop_seed=group["crop_seed"], input_path=group["input_path"],
                    seed=group["seed"], budget=args.budget, method=method,
                    attention_implementation=implementation, order_index=order_index,
                    generated_tokens=len(token_ids), eos=eos,
                    full_budget=(len(token_ids) == args.budget),
                    outer_wall_seconds=outer,
                    tokens_per_second=(len(token_ids) / outer if outer else None),
                    output_tokens=token_ids, counters=counters, stats=jsonable(stats),
                    target_calls=counters["target_calls"],
                    target_input_tokens=counters["target_input_tokens"],
                    draft_calls=counters["draft_calls"],
                    draft_input_tokens=counters["draft_input_tokens"],
                    ranker_calls=counters["ranker_calls"], rounds=counters["rounds"],
                    proposed=counters["proposed"], verified=counters["verified"],
                    accepted=counters["accepted"],
                    position_proposed=counters["position_proposed"],
                    position_verified=counters["position_verified"],
                    position_accepted=counters["position_accepted"],
                    feature_proposed=counters["feature_proposed"],
                    feature_verified=counters["feature_verified"],
                    feature_accepted=counters["feature_accepted"],
                    proposal_rejections=counters["proposal_rejections"],
                    mlp_calls=counters["mlp_calls"],
                    dynamic_graph_enabled=bool(runtime is not None),
                    dynamic_graph_runtime_created=runtime_created,
                    dynamic_graph=runtime_delta,
                )
                append_jsonl(output_dir / "records.jsonl", jsonable(row))
                rows.append(row)
                pipeline.update(status="running", completed_runs=len(rows),
                                expected_runs=len(prompts) * len(METHODS),
                                elapsed_seconds=time.time() - started)
                write_json(output_dir / "pipeline.json", pipeline)
                print(json.dumps(dict(
                    pair_id=group["pair_id"], source=group["source"], prefix=group["prefix"],
                    method=method, implementation=implementation,
                    tokens=len(token_ids), wall_seconds=outer, tokens_per_second=row["tokens_per_second"],
                    completed_runs=len(rows),
                ), ensure_ascii=False), flush=True)

    summary = summarize(rows, prompts, args.implementation)
    summary.update(dict(
        started_at_unix=started, elapsed_seconds=time.time() - started,
        source_scope=args.source_scope,
        eos_note="Actual output lengths and EOS are reported per run; full-budget ratios are separate.",
        attention_verification=verification, warmup=warmup,
        dynamic_graph_enabled=bool(args.dynamic_graph),
        dynamic_graph_cache_capacity=args.graph_cache_capacity,
        dynamic_graph_runtime=(measured_graph_runtime.stats()
                               if measured_graph_runtime is not None else None),
        dynamic_graph_setup=dynamic_setup_summary(
            measured_graph_runtime.stats() if measured_graph_runtime is not None else None,
            rows,
        ),
        dynamic_graph_setup_accounting=(
            "Measured candidate runtime is created on first candidate decode and reused across the matrix. "
            "Its constructor, first KV allocation, graph warmup/capture and any later bucket captures are "
            "inside the corresponding synchronized outer decode wall; graph setup_ms per record is a "
            "breakdown of time already included there. The separate 32-token warmup runtime is discarded."
            if args.dynamic_graph else "Dynamic Target Graph was disabled."
        ),
    ))
    write_json(output_dir / "summary.json", jsonable(summary))
    pipeline.update(status=summary["status"], completed_runs=len(rows),
                    elapsed_seconds=summary["elapsed_seconds"])
    write_json(output_dir / "pipeline.json", pipeline)
    print(json.dumps(dict(status=summary["status"], output=str(output_dir),
                          completed_runs=len(rows)), ensure_ascii=False), flush=True)


def main() -> None:
    global _ACTIVE_PIPELINE_PATH, _ACTIVE_PIPELINE
    _ACTIVE_PIPELINE_PATH = None
    _ACTIVE_PIPELINE = None
    try:
        _main()
    except Exception as exc:
        if _ACTIVE_PIPELINE_PATH is not None and _ACTIVE_PIPELINE is not None:
            _ACTIVE_PIPELINE.update(
                status="failed", error=dict(type=type(exc).__name__, message=str(exc)),
                failed_at_unix=time.time(),
            )
            if "started_at_unix" in _ACTIVE_PIPELINE:
                _ACTIVE_PIPELINE["elapsed_seconds"] = time.time() - _ACTIVE_PIPELINE["started_at_unix"]
            try:
                write_json(_ACTIVE_PIPELINE_PATH, _ACTIVE_PIPELINE)
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
