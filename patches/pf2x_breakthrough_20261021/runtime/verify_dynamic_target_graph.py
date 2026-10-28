#!/usr/bin/env python3
"""Compare eager SDPA and dynamic graph on one real crop, including rewind."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return str(value)


def compare(torch, left, right, policy, atol: float, rtol: float) -> dict:
    if tuple(left.shape) != tuple(right.shape):
        return dict(passed=False, shape_left=list(left.shape),
                    shape_right=list(right.shape))
    left_finite = torch.isfinite(left)
    right_finite = torch.isfinite(right)
    support_equal = bool(torch.equal(left_finite, right_finite))
    if bool(torch.isnan(left).any().item()) or bool(torch.isnan(right).any().item()):
        return dict(passed=False, support_equal=support_equal,
                    finite=False, nan_values=True)
    if policy is None and not (bool(left_finite.all().item()) and
                               bool(right_finite.all().item())):
        return dict(passed=False, support_equal=support_equal,
                    finite=False, nan_values=False,
                    max_row_l2_relative_error=None)
    finite = left_finite & right_finite
    if bool(finite.any().item()):
        delta = (left[finite] - right[finite]).abs()
        limit = atol + rtol * right[finite].abs()
        max_abs = float(delta.max().item())
        max_tolerance_fraction = float((delta / limit.clamp_min(1e-30)).max().item())
        all_close = bool(torch.all(delta <= limit).item())
    else:
        max_abs = 0.0
        max_tolerance_fraction = 0.0
        all_close = True
    result = dict(
        passed=support_equal,
        support_equal=support_equal,
        differing_support_values=int((left_finite != right_finite).sum().item()),
        max_abs_error=max_abs,
        max_tolerance_fraction=max_tolerance_fraction,
        allclose=all_close,
    )
    if policy is not None:
        tv_values = []
        policy_support_equal = True
        policy_finite = True
        left_rows = left.reshape(-1, left.size(-1))
        right_rows = right.reshape(-1, right.size(-1))
        for row in range(left_rows.size(0)):
            left_policy = policy(left_rows[row])
            right_policy = policy(right_rows[row])
            if not (bool(torch.isfinite(left_policy).all().item()) and
                    bool(torch.isfinite(right_policy).all().item())):
                policy_finite = False
                continue
            policy_support_equal &= bool(torch.equal(
                left_policy > 0, right_policy > 0
            ))
            tv_values.append(float((left_policy - right_policy).abs().sum().item()) * 0.5)
        result["max_policy_tv"] = max(tv_values, default=0.0)
        result["policy_support_equal"] = policy_support_equal
        result["policy_finite"] = policy_finite
        result["passed"] = bool(
            support_equal and policy_support_equal and policy_finite and
            len(tv_values) == left_rows.size(0) and
            result["max_policy_tv"] <= 1e-5
        )
    else:
        left_rows = left.float().reshape(-1, left.size(-1))
        right_rows = right.float().reshape(-1, right.size(-1))
        relative = ((left_rows - right_rows).norm(dim=-1)
                    / left_rows.norm(dim=-1).clamp_min(1e-12))
        result["max_row_l2_relative_error"] = float(relative.max().item())
        result["passed"] = bool(support_equal and result["max_row_l2_relative_error"] <= 1e-5)
    return result


def run_one_path(torch, core, cache, capture, runtime, *, enabled, prefix,
                 continuation, eos, typed_kinds, cache_factory, prefix_body,
                 output_rows, arm_name, raw_path):
    device = next(core.parameters()).device
    capacity = len(prefix) + 16
    if cache is None:
        cache = (runtime.cache_for(core, capacity, factory=cache_factory)
                 if enabled else cache_factory(core, capacity))
    operation_rows = []

    def run_block(name: str, tokens: list[int]) -> dict:
        start_pos = int(cache.get_pos())
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
        kinds = typed_kinds(start_pos, len(tokens))
        capture.output = None
        logits = core(idx, kv_cache=cache, logit_types=kinds)
        if capture.output is None or int(capture.output.size(1)) != len(tokens):
            raise RuntimeError(f"last-block hidden was not captured for {arm_name}/{name}")
        end_pos = int(cache.get_pos())
        if end_pos != start_pos + len(tokens):
            raise RuntimeError(f"cache moved {start_pos} -> {end_pos} for {arm_name}/{name}")
        row = dict(
            arm=arm_name,
            operation=name,
            start_pos=start_pos,
            end_pos=end_pos,
            input_tokens=list(tokens),
            logit_types=list(kinds),
            logits=logits.detach().to(device="cpu").clone(),
            pre_final_norm_hidden=capture.output.detach().to(device="cpu").clone(),
        )
        output_rows.append(row)
        torch.save(dict(rows=output_rows), raw_path)
        operation_rows.append(dict(
            operation=name, start_pos=start_pos, end_pos=end_pos,
            input_tokens=list(tokens), logit_types=list(kinds),
            read_bucket=(
                next((bucket for bucket in runtime.read_buckets
                      if bucket >= end_pos), None) if enabled else None
            ),
        ))
        return dict(logits=logits.detach().clone(),
                    hidden=capture.output.detach().clone(),
                    start_pos=start_pos, end_pos=end_pos, kinds=kinds)

    with torch.inference_mode(), runtime.attention_context(
            enabled=enabled, hidden_capture=capture):
        prefix_idx = torch.tensor([prefix[:-1]], dtype=torch.long, device=device)
        prefix_kinds = typed_kinds(0, len(prefix) - 1)
        core(prefix_idx, kv_cache=cache, logit_types=prefix_kinds)
        if int(cache.get_pos()) != len(prefix) - 1:
            raise RuntimeError(f"long prefill ended at {cache.get_pos()}, expected {len(prefix)-1}")

        first = run_block("T1", [prefix[-1]])
        second = run_block("T2", continuation[:2])
        third = run_block("T4", continuation[2:6])
        committed_pos = first["end_pos"]
        old_pos = int(cache.get_pos())
        from bench_gaussiangpt_pf_block2_cow import truncate
        truncate(cache, committed_pos, old_pos)
        after_truncate = int(cache.get_pos())
        if after_truncate != committed_pos:
            raise RuntimeError("cache did not return to the committed position")

        alternate = list(continuation[:2])
        for index, token in enumerate(alternate):
            feature_index = int(token) - 8000
            if 0 <= feature_index < 4096:
                alternate[index] = 8000 + ((feature_index + 1) % 4096)
                break
        if alternate == continuation[:2]:
            raise RuntimeError("could not form a distinct alternate Feature branch")
        rewind_branch = run_block("rewind_T2_alternate", alternate)

    return cache, dict(
        long_prefill_tokens=len(prefix) - 1,
        prefix_body_tokens=prefix_body,
        initial_cache_pos=len(prefix) - 1,
        operations=operation_rows,
        read_bucket_crossed=any(
            left["read_bucket"] != right["read_bucket"]
            for left, right in zip(operation_rows, operation_rows[1:])
            if left["read_bucket"] is not None and right["read_bucket"] is not None
        ),
        committed_pos=committed_pos,
        truncated_from_pos=old_pos,
        truncated_to_pos=after_truncate,
        alternate_input_tokens=alternate,
        eos_id=eos,
        compared_blocks=dict(T1=first, T2=second, T4=third,
                             rewind_T2_alternate=rewind_branch),
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix-body", type=int, default=510)
    parser.add_argument("--record-index", type=int, default=0)
    args = parser.parse_args()
    project = args.project.expanduser().resolve()
    crops = args.crops.expanduser()
    crops = crops.resolve() if crops.is_absolute() else (project / crops).resolve()
    output_dir = args.out_dir.expanduser()
    output_dir = (output_dir.resolve() if output_dir.is_absolute()
                  else (project / output_dir).resolve())
    if output_dir.exists():
        parser.error("--out-dir must name a new directory: " + str(output_dir))
    if not project.is_dir() or not crops.is_file():
        parser.error("Project or crop records are missing")
    if args.prefix_body < 8 or args.record_index < 0:
        parser.error("prefix-body must be at least 8 and record-index nonnegative")
    output_dir.mkdir(parents=True)
    result_path = output_dir / "metadata.json"
    result = dict(status="starting", project=str(project), crops=str(crops),
                  output=str(output_dir), prefix_body_tokens=args.prefix_body,
                  record_index=args.record_index,
                  tensor_artifact=str(output_dir / "raw_outputs.pt"))
    write_json(result_path, result)
    started = time.perf_counter()
    try:
        sys.path.insert(0, str(project))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import torch
        import model.gpt.flash_attention as flash_module
        from bench_gaussiangpt_pf_block2_cow import CaptureLastBlock
        from draft_a.corpus_v2 import EOS, load_target, validate
        from draft_a.feature_acceptance_eval import cache_for, policy
        from draft_a.pf_ranker_eval import typed_kinds
        from dynamic_target_graph import DynamicTargetGraph

        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this real-model verifier")
        device = torch.device("cuda")
        records = torch.load(crops, map_location="cpu", weights_only=True)
        if not 0 <= args.record_index < len(records):
            raise IndexError("record-index is outside the crop file")
        record = records[args.record_index]
        validate(record)
        body = [int(token) for token in record["tokens"]]
        end = args.prefix_body + 6
        if len(body) < end:
            raise ValueError(f"crop has {len(body)} tokens; need at least {end}")
        prefix = [int(EOS), *body[:args.prefix_body]]
        continuation = body[args.prefix_body:end]

        wrapper = load_target().float().eval().requires_grad_(False)
        core = wrapper.transformer
        if next(core.parameters()).dtype != torch.float32:
            raise RuntimeError("Target weights must be FP32")
        runtime = DynamicTargetGraph(core, record_outputs=False)
        capture = CaptureLastBlock(core)
        original_override = flash_module._override_impl
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        rows = []
        try:
            flash_module._override_impl = "sdpa"
            reference_cache, reference = run_one_path(
                torch, core, None, capture, runtime, enabled=False,
                prefix=prefix, continuation=continuation, eos=int(EOS),
                typed_kinds=typed_kinds, cache_factory=cache_for,
                prefix_body=args.prefix_body, output_rows=rows,
                raw_path=output_dir / "raw_outputs.pt",
                arm_name="reference_sdpa",
            )
            candidate_cache, candidate = run_one_path(
                torch, core, None, capture, runtime, enabled=True,
                prefix=prefix, continuation=continuation, eos=int(EOS),
                typed_kinds=typed_kinds, cache_factory=cache_for,
                prefix_body=args.prefix_body, output_rows=rows,
                raw_path=output_dir / "raw_outputs.pt",
                arm_name="dynamic_graph",
            )
            comparisons = {}
            for name in ("T1", "T2", "T4", "rewind_T2_alternate"):
                left = reference["compared_blocks"][name]
                right = candidate["compared_blocks"][name]
                comparisons[name] = dict(
                    logits=compare(torch, left["logits"], right["logits"],
                                   policy, 1e-4, 1e-4),
                    pre_final_norm_hidden=compare(
                        torch, left["hidden"], right["hidden"], None, 1e-4, 1e-4
                    ),
                    reference_pos=[left["start_pos"], left["end_pos"]],
                    candidate_pos=[right["start_pos"], right["end_pos"]],
                    reference_logit_types=list(left["kinds"]),
                    candidate_logit_types=list(right["kinds"]),
                )
            runtime_stats = runtime.stats()
            graph_events = runtime_stats["graph_events"]
            bucket_sequence = [event["read_bucket"] for event in graph_events]
            graph_shapes = {int(event["query_tokens"]) for event in graph_events}
            graph_replays = int(runtime_stats["calls"].get("graph_forward", 0))
            unexpected_fallbacks = {
                reason: count for reason, count in runtime_stats["fallbacks"].items()
                if reason not in ("unsupported_query_tokens",)
            }
            crosses_bucket = len(set(bucket_sequence)) > 1
            if graph_replays < 4:
                raise RuntimeError(
                    f"expected four graph replays, observed {graph_replays}"
                )
            if runtime_stats["graph_count"] < 3 or not {1, 2, 4}.issubset(graph_shapes):
                raise RuntimeError(
                    "candidate did not create graph events for T1/T2/T4: "
                    f"count={runtime_stats['graph_count']} shapes={sorted(graph_shapes)}"
                )
            if unexpected_fallbacks:
                raise RuntimeError(
                    f"unexpected dynamic-path fallbacks: {unexpected_fallbacks}"
                )
            passed = all(
                item["logits"]["passed"] and
                item["pre_final_norm_hidden"]["passed"] and
                item["logits"].get("max_policy_tv", float("inf")) <= 1e-5 and
                item["pre_final_norm_hidden"].get(
                    "max_row_l2_relative_error", float("inf")
                ) <= 1e-5
                for item in comparisons.values()
            )
            if not crosses_bucket:
                raise RuntimeError("the dynamic path did not create graphs across a read-bucket boundary")
            if not passed:
                raise RuntimeError("one or more real-crop tensor comparisons exceeded tolerance")
            torch.save(dict(
                source=str(record.get("scene", record.get("source", "unknown"))),
                record_index=args.record_index,
                prefix_tokens=prefix,
                continuation_tokens=continuation,
                rows=rows,
            ), output_dir / "raw_outputs.pt")
            result.update(
                status="passed",
                source=str(record.get("scene", record.get("source", "unknown"))),
                prefix_tokens=len(prefix),
                continuation_tokens=continuation,
                comparisons=comparisons,
                reference_operations=reference["operations"],
                candidate_operations=candidate["operations"],
                rewind=dict(
                    committed_pos=candidate["committed_pos"],
                    truncated_from_pos=candidate["truncated_from_pos"],
                    truncated_to_pos=candidate["truncated_to_pos"],
                    alternate_input_tokens=candidate["alternate_input_tokens"],
                ),
                graph_buckets=bucket_sequence,
                runtime_stats=runtime_stats,
                elapsed_seconds=time.perf_counter() - started,
            )
        finally:
            capture.close()
            flash_module._override_impl = original_override
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        write_json(result_path, jsonable(result))
    except Exception as error:
        result.update(status="failed", error=dict(type=type(error).__name__, message=str(error)),
                      elapsed_seconds=time.perf_counter() - started)
        write_json(result_path, jsonable(result))
        raise


if __name__ == "__main__":
    _main()
