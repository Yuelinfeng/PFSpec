#!/usr/bin/env python3
"""Compare eager and graph EAGLE Draft calls on one real crop prefix."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time


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
    return str(value)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def cpu(tensor):
    return tensor.detach().to(device="cpu").clone()


def cache_copy(draft, source, capacity):
    target = draft.new_cache(int(capacity))
    pos = int(source.get_pos())
    if pos:
        target.k_cache[:, :, :pos].copy_(source.k_cache[:, :, :pos])
        target.v_cache[:, :, :pos].copy_(source.v_cache[:, :, :pos])
        target.token_cache[:, :pos].copy_(source.token_cache[:, :pos])
    target.cache_seqlens.copy_(source.cache_seqlens)
    return target


def cache_snapshot(cache, pos):
    return dict(
        pos=int(pos),
        k=cache.k_cache[:, :, :pos].detach().clone(),
        v=cache.v_cache[:, :, :pos].detach().clone(),
        tokens=cache.token_cache[:, :pos].detach().clone(),
    )


def restore_cache(cache, saved):
    pos = int(saved["pos"])
    if pos:
        cache.k_cache[:, :, :pos].copy_(saved["k"])
        cache.v_cache[:, :, :pos].copy_(saved["v"])
        cache.token_cache[:, :pos].copy_(saved["tokens"])
    cache.cache_seqlens.fill_(pos)


def row_l2_relative(torch, actual, expected):
    if tuple(actual.shape) != tuple(expected.shape):
        return dict(finite=False, shape_equal=False, max_row_l2_relative_error=None)
    left, right = actual.float().reshape(-1, actual.shape[-1]), expected.float().reshape(-1, expected.shape[-1])
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        return dict(finite=False, shape_equal=True, max_row_l2_relative_error=None)
    numerator = torch.linalg.vector_norm(left - right, dim=-1)
    denominator = torch.linalg.vector_norm(right, dim=-1).clamp_min(1e-12)
    value = float((numerator / denominator).max().item()) if numerator.numel() else 0.0
    return dict(finite=True, shape_equal=True, max_row_l2_relative_error=value,
                passed=value <= 1e-5)


def compare_policy(torch, draft, runtime, actual, expected):
    from speculative.pf_engine import policy

    if tuple(actual.shape) != tuple(expected.shape):
        return dict(shape_equal=False, finite=False, max_tv=None,
                    support_equal=False, support_mismatches=None)
    kinds = ("F",) if draft.config.mode == "feature" else ("P", "F")
    actual_rows = actual.reshape(-1, actual.shape[-1])
    expected_rows = expected.reshape(-1, expected.shape[-1])
    values = []
    for kind in kinds:
        row_values = []
        for row in range(actual_rows.shape[0]):
            left_logits = draft.logits(actual_rows[row], kind)
            right_logits = runtime.eager_logits(expected_rows[row], kind)
            left = policy(left_logits.float(), temperature=0.9, top_p=0.9)
            right = policy(right_logits.float(), temperature=0.9, top_p=0.9)
            if not (bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all())):
                return dict(shape_equal=True, finite=False, max_tv=None,
                            support_equal=False, support_mismatches=None)
            tv = float((left - right).abs().sum().mul(0.5).item())
            support_diff = (left > 0) != (right > 0)
            row_values.append(dict(
                tv=tv,
                support_mismatches=int(support_diff.sum().item()),
                graph_policy=cpu(left),
                eager_policy=cpu(right),
            ))
        row_tv = max((item["tv"] for item in row_values), default=0.0)
        row_support = sum(item["support_mismatches"] for item in row_values)
        values.append(dict(kind=kind, rows=row_values, max_tv=row_tv,
                           support_mismatches=row_support,
                           support_equal=row_support == 0))
    max_tv = max(item["max_tv"] for item in values)
    mismatches = sum(item["support_mismatches"] for item in values)
    return dict(shape_equal=True, finite=True, max_tv=max_tv,
                support_equal=mismatches == 0, support_mismatches=mismatches,
                kinds=values,
                passed=max_tv <= 1e-5 and mismatches == 0)


def compare_kv(torch, left, right, start, end):
    left_k = left.k_cache[:, :, start:end]
    right_k = right.k_cache[:, :, start:end]
    left_v = left.v_cache[:, :, start:end]
    right_v = right.v_cache[:, :, start:end]
    if (tuple(left_k.shape) != tuple(right_k.shape)
            or tuple(left_v.shape) != tuple(right_v.shape)):
        return dict(shape_equal=False, finite=False, max_abs_error=None, passed=False)
    finite = bool(torch.isfinite(left_k).all() and torch.isfinite(right_k).all()
                  and torch.isfinite(left_v).all() and torch.isfinite(right_v).all())
    max_abs = max(float((left_k.float() - right_k.float()).abs().max().item()),
                  float((left_v.float() - right_v.float()).abs().max().item()))
    passed = finite and bool(torch.allclose(left_k, right_k, atol=1e-5, rtol=1e-5)
                             and torch.allclose(left_v, right_v, atol=1e-5, rtol=1e-5))
    return dict(shape_equal=True, finite=finite, max_abs_error=max_abs, passed=passed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method", default="eagle_v3_g3",
                        choices=("eagle_v3_g3", "eagle_v3_g5", "eagle_v4_reuse_g3"))
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--prefix-body", type=int, default=126)
    parser.add_argument("--steps", type=int, default=140)
    parser.add_argument("--cache-capacity", type=int, default=1024)
    args = parser.parse_args()
    project = args.project.expanduser().resolve()
    crops = args.crops.expanduser()
    crops = crops.resolve() if crops.is_absolute() else (project / crops).resolve()
    out_dir = args.out_dir.expanduser()
    out_dir = out_dir.resolve() if out_dir.is_absolute() else (project / out_dir).resolve()
    if out_dir.exists():
        parser.error("--out-dir must name a new directory")
    if not project.is_dir() or not crops.is_file():
        parser.error("Project or crop records are missing")
    if (args.record_index < 0 or args.prefix_body < 8 or args.prefix_body % 2
            or args.steps < 4 or args.cache_capacity < args.prefix_body + args.steps + 2):
        parser.error("Invalid record, prefix, step count or cache capacity")
    out_dir.mkdir(parents=True)
    metadata_path = out_dir / "metadata.json"
    metadata = dict(
        status="starting",
        project=str(project), crops=str(crops), output=str(out_dir),
        method=args.method, record_index=args.record_index,
        prefix_body_tokens=args.prefix_body, recurrent_steps=args.steps,
        cache_capacity=args.cache_capacity,
        raw_artifact=str(out_dir / "raw_outputs.pt"),
    )
    write_json(metadata_path, metadata)
    started = time.perf_counter()
    try:
        os.chdir(project)
        sys.path.insert(0, str(project))
        sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import torch
        from draft_a.corpus_v2 import EOS, load_target, validate
        from draft_a.learning_eval import _fp32_runtime
        from speculative.pf_engine import target_cache
        from method_adapters import build_context, load_method
        from eagle_graph import EagleDraftGraph

        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this model verification")
        records = torch.load(crops, map_location="cpu", weights_only=True)
        if not 0 <= args.record_index < len(records):
            raise IndexError("record-index is outside the crop file")
        record = records[args.record_index]
        validate(record)
        body = [int(token) for token in record["tokens"]]
        need_body = args.prefix_body + args.steps + 2
        if len(body) < need_body:
            raise ValueError(f"crop has {len(body)} body tokens; need {need_body}")

        context = build_context(project, device="cuda")
        adapter = load_method(args.method, context)
        draft = adapter.model
        core = context["target"]
        if draft.config.mode != "feature":
            raise ValueError("This verifier expects the feature-only EAGLE Draft")
        full_tokens = [int(EOS), *body[:need_body]]
        ids = torch.tensor([full_tokens], dtype=torch.long, device="cuda")
        coordinates = core._build_sparse_position_indices(ids)
        kinds = ["position" if index % 2 == 0 else "feature"
                 for index in range(len(full_tokens))]
        tgt_cache = target_cache(core, len(full_tokens))
        with torch.inference_mode(), _fp32_runtime(torch), torch.autocast(
            device_type="cuda", enabled=False
        ):
            _, target_hidden = core(
                ids, kv_cache=tgt_cache, logit_types=kinds, return_hidden=True
            )
            prefix_cache = draft.new_cache(args.cache_capacity)
            prefix_hidden = target_hidden[:, :args.prefix_body]
            prefix_tokens = ids[:, 1:args.prefix_body + 1]
            prefix_coordinates = coordinates[:, 1:args.prefix_body + 1]
            draft.forward(prefix_hidden, prefix_tokens, prefix_coordinates,
                          cache=prefix_cache)
            prefix_pos = int(prefix_cache.get_pos())
            if prefix_pos != args.prefix_body:
                raise RuntimeError(f"Draft prefix ended at {prefix_pos}")

            fallback_candidate = cache_copy(draft, prefix_cache, args.cache_capacity)
            fallback_reference = cache_copy(draft, prefix_cache, args.cache_capacity)
            recurrent_candidate = cache_copy(draft, prefix_cache, args.cache_capacity)
            recurrent_reference = cache_copy(draft, prefix_cache, args.cache_capacity)
            runtime = EagleDraftGraph(
                draft,
                max_cache_tokens=args.cache_capacity,
                warmup_calls=2,
                verify=False,
                profile_copies=True,
            )
            rows = []
            raw_rows = []

            def paired_call(name, previous_hidden, actual_tokens, actual_coordinates,
                            candidate_cache, reference_cache):
                before_candidate = int(candidate_cache.get_pos())
                before_reference = int(reference_cache.get_pos())
                if before_candidate != before_reference:
                    raise RuntimeError(f"{name}: branch positions differ before the call")
                candidate_output = draft.forward(
                    previous_hidden, actual_tokens, actual_coordinates,
                    cache=candidate_cache,
                ).detach().clone()
                reference_output = runtime.eager_forward(
                    previous_hidden, actual_tokens, actual_coordinates,
                    cache=reference_cache,
                ).detach().clone()
                end_candidate = int(candidate_cache.get_pos())
                end_reference = int(reference_cache.get_pos())
                hidden_result = row_l2_relative(torch, candidate_output, reference_output)
                policy_result = compare_policy(torch, draft, runtime,
                                               candidate_output, reference_output)
                kv_result = compare_kv(torch, candidate_cache, reference_cache,
                                       before_candidate, end_candidate)
                row = dict(
                    operation=name,
                    start_pos=before_candidate,
                    candidate_end_pos=end_candidate,
                    reference_end_pos=end_reference,
                    query_tokens=int(actual_tokens.shape[1]),
                    input_tokens=actual_tokens.detach().cpu().tolist(),
                    hidden=hidden_result,
                    policy=policy_result,
                    kv=kv_result,
                    position_equal=end_candidate == end_reference,
                )
                rows.append(row)
                raw_rows.append(dict(
                    operation=name,
                    input_previous_hidden=cpu(previous_hidden),
                    input_tokens=cpu(actual_tokens),
                    input_coordinates=cpu(actual_coordinates),
                    graph_hidden=cpu(candidate_output),
                    eager_hidden=cpu(reference_output),
                    graph_feature_policy=[
                        item["graph_policy"]
                        for item in policy_result.get("kinds", [{}])[0].get("rows", [])
                    ],
                    eager_feature_policy=[
                        item["eager_policy"]
                        for item in policy_result.get("kinds", [{}])[0].get("rows", [])
                    ],
                    graph_k=cpu(candidate_cache.k_cache[:, :, before_candidate:end_candidate]),
                    eager_k=cpu(reference_cache.k_cache[:, :, before_reference:end_reference]),
                    graph_v=cpu(candidate_cache.v_cache[:, :, before_candidate:end_candidate]),
                    eager_v=cpu(reference_cache.v_cache[:, :, before_reference:end_reference]),
                ))
                return reference_output

            with runtime.context():
                fallback_hidden = target_hidden[:, args.prefix_body:args.prefix_body + 2]
                fallback_tokens = ids[:, args.prefix_body + 1:args.prefix_body + 3]
                fallback_coordinates = coordinates[:, args.prefix_body + 1:args.prefix_body + 3]
                paired_call("eager_T2_fallback", fallback_hidden, fallback_tokens,
                            fallback_coordinates, fallback_candidate, fallback_reference)

                common_hidden = target_hidden[:, args.prefix_body:args.prefix_body + 1]
                checkpoint_candidate = None
                checkpoint_reference = None
                checkpoint_hidden = None
                checkpoint_next_token = None
                checkpoint_next_coordinates = None
                for step in range(args.steps):
                    body_index = args.prefix_body + step
                    absolute_index = body_index + 1
                    input_token = ids[:, absolute_index:absolute_index + 1]
                    input_coordinates = coordinates[:, absolute_index:absolute_index + 1]
                    common_hidden = paired_call(
                        f"graph_T1_{step:03d}", common_hidden, input_token,
                        input_coordinates, recurrent_candidate, recurrent_reference,
                    )[:, -1:, :]
                    if step == 0:
                        pos = int(recurrent_reference.get_pos())
                        checkpoint_candidate = cache_snapshot(recurrent_candidate, pos)
                        checkpoint_reference = cache_snapshot(recurrent_reference, pos)
                        checkpoint_hidden = common_hidden.detach().clone()
                        next_absolute = pos + 1
                        original = int(ids[0, next_absolute].item())
                        feature_start = int(draft.spec.offset)
                        feature_end = feature_start + int(draft.spec.features)
                        if not feature_start <= original < feature_end:
                            raise RuntimeError("rewind point is not followed by a Feature token")
                        checkpoint_next_token = torch.tensor(
                            [[feature_start + ((original - feature_start + 1)
                                               % int(draft.spec.features))]],
                            dtype=torch.long, device="cuda",
                        )
                        checkpoint_next_coordinates = coordinates[:, next_absolute:next_absolute + 1]

                before_graph_count = runtime.stats()["graph_count"]
                restore_cache(recurrent_candidate, checkpoint_candidate)
                restore_cache(recurrent_reference, checkpoint_reference)
                paired_call("rewind_alternate_feature_T1", checkpoint_hidden,
                            checkpoint_next_token, checkpoint_next_coordinates,
                            recurrent_candidate, recurrent_reference)
                after_graph_count = runtime.stats()["graph_count"]

            runtime_stats = runtime.stats()
            events = runtime_stats["graph_events"]
            graph_buckets = sorted({int(event["read_bucket"]) for event in events
                                    if "read_bucket" in event})
            graph_calls = int(runtime_stats["calls"].get("graph_forward", 0))
            fallback_calls = int(runtime_stats["fallbacks"].get("unsupported_query_tokens", 0))
            read_bucket_crossed = len(graph_buckets) >= 2
            comparisons_passed = all(
                row["hidden"].get("passed", False)
                and row["policy"].get("passed", False)
                and row["kv"].get("passed", False)
                and row["position_equal"]
                for row in rows
            )
            checks = dict(
                graph_calls_at_least_steps=graph_calls >= args.steps + 1,
                eager_fallback_seen=fallback_calls >= 1,
                multiple_read_buckets=read_bucket_crossed,
                rewind_reused_graph=after_graph_count == before_graph_count,
                paired_comparisons_passed=comparisons_passed,
            )
            passed = all(checks.values())
            torch.save(dict(rows=raw_rows), out_dir / "raw_outputs.pt")
            metadata.update(
                status="passed" if passed else "failed",
                source=str(record.get("scene", record.get("source", "unknown"))),
                input_prefix_tokens=len(full_tokens),
                draft_prefix_pos=prefix_pos,
                recurrent_steps=args.steps,
                fallback_query_tokens=2,
                graph_read_buckets=graph_buckets,
                graph_calls=graph_calls,
                graph_count=runtime_stats["graph_count"],
                fallback_counts=runtime_stats["fallbacks"],
                staging_copy_ms=runtime_stats["staging_copy_ms"],
                staging_copy_bytes=runtime_stats["staging_copy_bytes"],
                checks=checks,
                operation_summaries=rows,
                runtime_stats=runtime_stats,
                elapsed_seconds=time.perf_counter() - started,
            )
            runtime.close()
        write_json(metadata_path, metadata)
        if not passed:
            raise RuntimeError("one or more eager/graph Draft comparisons did not pass")
    except Exception as error:
        metadata.update(status="failed", error=dict(type=type(error).__name__, message=str(error)),
                        elapsed_seconds=time.perf_counter() - started)
        write_json(metadata_path, metadata)
        raise


if __name__ == "__main__":
    main()
