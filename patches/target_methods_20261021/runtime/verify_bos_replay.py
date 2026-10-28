#!/usr/bin/env python3
"""Compare LFQ BOS Target outputs and BF16 draft probabilities on one shared trace."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


METHOD_NAME = "lfq_scratch_pf_bos"


def json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return str(value)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def cpu_clone(value):
    return value.detach().to(device="cpu").clone()


def first_saved_mismatch(run_dir: Path, seed: int) -> dict | None:
    rows = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()
            if line.strip()]
    by_arm = {
        row["execution"]: row
        for row in rows
        if row.get("method") == METHOD_NAME and int(row.get("seed", -1)) == seed
    }
    original = by_arm.get("original")
    optimized = by_arm.get("optimized")
    if original is None or optimized is None:
        return None
    left = [int(x) for x in original.get("output_tokens", [])]
    right = [int(x) for x in optimized.get("output_tokens", [])]
    mismatch = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
    return dict(
        original_tokens=len(left), optimized_tokens=len(right),
        first_different_index=mismatch,
        original_token=None if mismatch is None or mismatch >= len(left) else left[mismatch],
        optimized_token=None if mismatch is None or mismatch >= len(right) else right[mismatch],
    )


def pair_source_hidden(torch, draft, reference_calls, candidate_calls):
    """Find the Target call row that produced each exact LFQ h_prev tensor."""
    pairs = []
    for item in draft:
        hidden = item["h_prev"]
        limit = min(int(item["call_count_before_draft"]), len(reference_calls))
        found = None
        for call_id in range(limit - 1, -1, -1):
            ref_hidden = reference_calls[call_id]["hidden"]
            for row in range(ref_hidden.shape[0] - 1, -1, -1):
                if torch.equal(hidden, ref_hidden[row]):
                    found = (call_id, row)
                    break
            if found is not None:
                break
        if found is None:
            pairs.append(dict(draft_id=item["draft_id"], matched=False))
            continue
        call_id, row = found
        ref_call = reference_calls[call_id]
        candidate_call = candidate_calls[call_id]
        pairs.append(dict(
            draft_id=item["draft_id"], matched=True,
            source_call_id=call_id, source_row=row,
            source_absolute_position=int(ref_call["pos_before"]) + row,
            reference_h_prev=hidden,
            candidate_h_prev=candidate_call["hidden"][row],
        ))
    return pairs


def compare_rows(torch, reference_calls, candidate_calls, target_policy, *, temp, top_p):
    summaries = []
    first_policy_change = None
    first_hidden_change = None
    first_policy_over_limit = None
    max_policy_tv = 0.0
    max_hidden_l2 = 0.0
    max_logit_abs = 0.0
    support_equal = True
    policy_support_equal = True

    if len(reference_calls) != len(candidate_calls):
        raise RuntimeError(
            f"Target call count changed: reference={len(reference_calls)} "
            f"graph={len(candidate_calls)}"
        )

    for ref, got in zip(reference_calls, candidate_calls):
        for field in ("pos_before", "pos_after", "tokens", "kinds"):
            if ref[field] != got[field]:
                raise RuntimeError(
                    f"Replay trace differs at call {ref['call_id']} field {field}: "
                    f"{ref[field]!r} != {got[field]!r}"
                )
        left_logits = ref["logits"].to(device=got["logits"].device, dtype=torch.float32)
        right_logits = got["logits"].float()
        left_hidden = ref["hidden"].to(device=got["hidden"].device, dtype=torch.float32)
        right_hidden = got["hidden"].float()
        if tuple(left_logits.shape) != tuple(right_logits.shape):
            raise RuntimeError(f"Logit shape changed at call {ref['call_id']}")
        if tuple(left_hidden.shape) != tuple(right_hidden.shape):
            raise RuntimeError(f"Hidden shape changed at call {ref['call_id']}")

        per_call_tv = []
        per_call_hidden = []
        per_call_abs = []
        for row in range(left_logits.shape[0]):
            left_finite = torch.isfinite(left_logits[row])
            right_finite = torch.isfinite(right_logits[row])
            row_support_equal = bool(torch.equal(left_finite, right_finite))
            support_equal &= row_support_equal
            finite = left_finite & right_finite
            abs_error = float((left_logits[row][finite] - right_logits[row][finite]).abs().max().item()) if bool(finite.any().item()) else 0.0
            max_logit_abs = max(max_logit_abs, abs_error)
            per_call_abs.append(abs_error)

            p = target_policy(left_logits[row], temp=temp, top_p=top_p)
            q = target_policy(right_logits[row], temp=temp, top_p=top_p)
            row_tv = float((p - q).abs().sum().item()) * 0.5
            row_policy_support_equal = bool(torch.equal(p > 0, q > 0))
            policy_support_equal &= row_policy_support_equal
            max_policy_tv = max(max_policy_tv, row_tv)
            per_call_tv.append(row_tv)
            absolute_position = int(ref["pos_before"]) + row
            if first_policy_change is None and (row_tv > 0.0 or not row_policy_support_equal):
                first_policy_change = dict(
                    call_id=ref["call_id"], row=row,
                    absolute_position=absolute_position, input_kind=ref["kinds"][row],
                    policy_tv=row_tv, support_equal=row_policy_support_equal,
                    max_logit_abs=abs_error,
                )
            if first_policy_over_limit is None and row_tv > 1e-5:
                first_policy_over_limit = dict(
                    call_id=ref["call_id"], row=row,
                    absolute_position=absolute_position, input_kind=ref["kinds"][row],
                    policy_tv=row_tv,
                )

            delta = right_hidden[row] - left_hidden[row]
            ref_norm = left_hidden[row].norm().clamp_min(1e-12)
            row_l2 = float((delta.norm() / ref_norm).item())
            max_hidden_l2 = max(max_hidden_l2, row_l2)
            per_call_hidden.append(row_l2)
            if first_hidden_change is None and row_l2 > 0.0:
                first_hidden_change = dict(
                    call_id=ref["call_id"], row=row,
                    absolute_position=absolute_position, input_kind=ref["kinds"][row],
                    row_l2_relative=row_l2,
                )

        summaries.append(dict(
            call_id=ref["call_id"], pos_before=ref["pos_before"],
            query_tokens=len(ref["tokens"]), kinds=ref["kinds"],
            max_policy_tv=max(per_call_tv, default=0.0),
            max_hidden_row_l2_relative=max(per_call_hidden, default=0.0),
            max_logit_abs=max(per_call_abs, default=0.0),
        ))

    return dict(
        target_call_count=len(summaries), max_policy_tv=max_policy_tv,
        max_hidden_row_l2_relative=max_hidden_l2, max_logit_abs=max_logit_abs,
        logit_support_equal=support_equal, policy_support_equal=policy_support_equal,
        first_policy_change=first_policy_change,
        first_policy_tv_above_1e_5=first_policy_over_limit,
        first_hidden_change=first_hidden_change, calls=summaries,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=44101)
    parser.add_argument("--budget", type=int, default=512)
    args = parser.parse_args()
    project = args.project.resolve()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        parser.error(f"Output directory already exists: {out_dir}")
    if not (run_dir / "records.jsonl").is_file():
        parser.error(f"Missing records.jsonl under {run_dir}")
    if args.budget < 2 or args.budget % 2:
        parser.error("--budget must be a positive even number")

    out_dir.mkdir(parents=True)
    started = time.time()
    metadata = dict(
        status="starting", started_at_unix=started, seed=args.seed,
        budget=args.budget, method=METHOD_NAME,
        run_dir=str(run_dir), project=str(project), command=sys.argv,
        scope="Replay the original LFQ Target call/truncate sequence through the graph; no sampling or timing comparison.",
        saved_first_difference=first_saved_mismatch(run_dir, args.seed),
    )
    write_json(out_dir / "metadata.json", metadata)

    try:
        os.chdir(project)
        sys.path.insert(0, str(project))
        runtime_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(runtime_dir.parent))
        sys.path.insert(0, str(runtime_dir))

        import torch
        from method_adapters import build_context, load_method, decode_method
        from dynamic_target_graph import DynamicTargetGraph
        from draft_a.learning_eval import _fp32_runtime

        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")

        context = build_context(project, device="cuda")
        adapter = load_method(METHOD_NAME, context)
        target = context["target"]
        device = context["device"]
        lfq = importlib.import_module("bench_lfq_scratch_pf_cow")
        cache_module = importlib.import_module("profiling.s2.s2_decode_faithful_agreement")
        feature = adapter.model
        saved_forward = feature.forward
        saved_capture_run = lfq.CaptureLastBlock.run
        saved_truncate = lfq.truncate
        saved_policy_batch = lfq.policy_batch

        reference_calls = []
        trace_events = []
        draft_records = []
        recording = {"enabled": True, "cache": None}

        def traced_capture_run(capture, gpt, cache, tokens, kinds, call_device):
            pos_before = int(cache.get_pos())
            logits, hidden = saved_capture_run(capture, gpt, cache, tokens, kinds, call_device)
            pos_after = int(cache.get_pos())
            call_id = len(reference_calls)
            if recording["enabled"]:
                reference_calls.append(dict(
                    call_id=call_id, pos_before=pos_before, pos_after=pos_after,
                    tokens=[int(x) for x in tokens], kinds=[str(x) for x in kinds],
                    logits=cpu_clone(logits), hidden=cpu_clone(hidden),
                ))
                trace_events.append(dict(type="forward", call_id=call_id))
                recording["cache"] = cache
            return logits, hidden

        def traced_truncate(cache, keep_pos, old_pos):
            if recording["enabled"]:
                trace_events.append(dict(
                    type="truncate", keep_pos=int(keep_pos), old_pos=int(old_pos),
                    pos_before=int(cache.get_pos()),
                ))
            return saved_truncate(cache, keep_pos, old_pos)

        def traced_feature_forward(prev_state, prev_feature, position):
            result = saved_forward(prev_state, prev_feature, position)
            if recording["enabled"]:
                draft_records.append(dict(
                    draft_id=len(draft_records),
                    call_count_before_draft=len(reference_calls),
                    cache_pos=(int(recording["cache"].get_pos())
                               if recording["cache"] is not None else None),
                    h_prev=cpu_clone(prev_state.reshape(-1, prev_state.shape[-1])[0]),
                    prev_feature=int(prev_feature.reshape(-1)[0].item()),
                    position=int(position.reshape(-1)[0].item()),
                    logits=cpu_clone(result[1]), q=None,
                ))
            return result

        def traced_policy_batch(logits, temperature, top_p):
            q = saved_policy_batch(logits, temperature, top_p)
            if recording["enabled"]:
                if not draft_records:
                    raise RuntimeError("LFQ policy_batch was called without a traced draft")
                draft_records[-1]["draft_temperature"] = float(temperature)
                draft_records[-1]["draft_top_p"] = float(top_p)
                draft_records[-1]["q"] = cpu_clone(q.reshape(-1, q.shape[-1])[0])
            return q

        lfq.CaptureLastBlock.run = traced_capture_run
        lfq.truncate = traced_truncate
        lfq.policy_batch = traced_policy_batch
        feature.forward = traced_feature_forward
        try:
            with _fp32_runtime(torch):
                reference_stats, reference_suffix = decode_method(
                    adapter, target, [context["eos"]], args.budget, args.seed, runtime=None
                )
            recording["enabled"] = False
        finally:
            lfq.CaptureLastBlock.run = saved_capture_run
            lfq.truncate = saved_truncate
            lfq.policy_batch = saved_policy_batch
            feature.forward = saved_forward

        saved_original = None
        for line in (run_dir / "records.jsonl").read_text().splitlines():
            row = json.loads(line)
            if (row.get("method") == METHOD_NAME and row.get("execution") == "original"
                    and int(row.get("seed", -1)) == args.seed):
                saved_original = [int(x) for x in row.get("output_tokens", [])]
                break
        reference_suffix = [int(x) for x in reference_suffix]
        if len(reference_calls) < 2:
            raise RuntimeError("Reference trace recorded too few Target calls")
        reference_matches_saved = saved_original == reference_suffix

        # Persist the trace before graph work so a later runtime failure still leaves evidence.
        import torch as torch_module
        torch_module.save(dict(
            reference_calls=reference_calls, trace_events=trace_events,
            reference_draft_calls=draft_records, reference_suffix=reference_suffix,
        ), out_dir / "reference_trace.pt")
        reference_metadata = dict(metadata)
        reference_metadata.update(
            status="reference_recorded",
            reference_matches_saved_original=reference_matches_saved,
            reference_call_count=len(reference_calls), truncate_count=sum(
                item["type"] == "truncate" for item in trace_events
            ), draft_call_count=len(draft_records),
            reference_stats={key: int(value) for key, value in reference_stats.items()
                             if isinstance(value, (int, bool))},
        )
        write_json(out_dir / "metadata.json", reference_metadata)
        if not reference_matches_saved:
            metadata.update(
                status="reference_mismatch", finished_at_unix=time.time(),
                reference_matches_saved_original=False,
                reference_call_count=len(reference_calls),
                truncate_count=sum(item["type"] == "truncate" for item in trace_events),
                draft_call_count=len(draft_records),
                conclusion="Saved original output differs from this reference rerun; graph replay was not started.",
                duration_seconds=time.time() - started,
            )
            write_json(out_dir / "metadata.json", metadata)
            print(json.dumps({
                "status": metadata["status"], "out_dir": str(out_dir),
                "reference_matches_saved_original": False,
                "reference_calls": len(reference_calls),
            }, ensure_ascii=False))
            return 2

        capacity_request = 1 + args.budget + 8
        runtime = DynamicTargetGraph(
            target, cache_capacity_buckets=(1024,), max_query_tokens=4,
            max_graphs=128, record_outputs=False,
        )
        cache_factory = lambda core, capacity: cache_module.make_kv_cache(
            core, capacity, device
        )
        cache = runtime.cache_for(target, capacity_request, factory=cache_factory)
        graph_capture = lfq.CaptureLastBlock(target)
        candidate_calls = []
        try:
            with torch.inference_mode(), _fp32_runtime(torch), torch.autocast("cuda", enabled=False):
                with runtime.attention_context(enabled=True, hidden_capture=graph_capture):
                    for event in trace_events:
                        if event["type"] == "truncate":
                            if int(cache.get_pos()) != int(event["pos_before"]):
                                raise RuntimeError(
                                    f"Truncate begins at {cache.get_pos()}, expected {event['pos_before']}"
                                )
                            saved_truncate(cache, event["keep_pos"], event["old_pos"])
                            if int(cache.get_pos()) != event["keep_pos"]:
                                raise RuntimeError("Graph cache truncate did not restore the recorded position")
                            continue

                        ref = reference_calls[event["call_id"]]
                        pos_before = int(cache.get_pos())
                        if pos_before != ref["pos_before"]:
                            raise RuntimeError(
                                f"Graph replay call {ref['call_id']} starts at {pos_before}, "
                                f"expected {ref['pos_before']}"
                            )
                        logits, hidden = graph_capture.run(
                            target, cache, ref["tokens"], ref["kinds"], device
                        )
                        pos_after = int(cache.get_pos())
                        if pos_after != ref["pos_after"]:
                            raise RuntimeError(
                                f"Graph replay call {ref['call_id']} ends at {pos_after}, "
                                f"expected {ref['pos_after']}"
                            )
                        candidate_calls.append(dict(
                            call_id=ref["call_id"], pos_before=pos_before,
                            pos_after=pos_after, tokens=list(ref["tokens"]),
                            kinds=list(ref["kinds"]), logits=cpu_clone(logits),
                            hidden=cpu_clone(hidden),
                        ))
        finally:
            graph_capture.close()

        target_policy = importlib.import_module("eval_lean_candidate_to_draft").target_policy
        target_metrics = compare_rows(
            torch, reference_calls, candidate_calls, target_policy, temp=0.9, top_p=0.9
        )
        draft_pairs = pair_source_hidden(torch, draft_records, reference_calls, candidate_calls)
        draft_by_id = {item["draft_id"]: item for item in draft_records}
        q_tv_values = []
        q_support_equal = True
        first_q_change = None
        draft_summaries = []
        for pair in draft_pairs:
            if not pair["matched"]:
                draft_summaries.append(dict(draft_id=pair["draft_id"], matched=False))
                continue
            ref_draft = draft_by_id[pair["draft_id"]]
            candidate_hidden = pair["candidate_h_prev"].to(device=device)
            prev_feature = torch.tensor([ref_draft["prev_feature"]], device=device)
            position = torch.tensor([ref_draft["position"]], device=device)
            with _fp32_runtime(torch), torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True):
                _, candidate_logits = feature(candidate_hidden.unsqueeze(0), prev_feature, position)
            candidate_q = saved_policy_batch(
                candidate_logits.float(), ref_draft["draft_temperature"], ref_draft["draft_top_p"]
            ).reshape(-1, candidate_logits.shape[-1])[0]
            reference_q = ref_draft["q"].to(device=device, dtype=torch.float32)
            row_tv = float((reference_q - candidate_q).abs().sum().item()) * 0.5
            row_support_equal = bool(torch.equal(reference_q > 0, candidate_q > 0))
            q_tv_values.append(row_tv)
            q_support_equal &= row_support_equal
            if first_q_change is None and (row_tv > 0.0 or not row_support_equal):
                first_q_change = dict(
                    draft_id=pair["draft_id"], source_call_id=pair["source_call_id"],
                    source_row=pair["source_row"],
                    source_absolute_position=pair["source_absolute_position"],
                    q_tv=row_tv, support_equal=row_support_equal,
                )
            draft_summaries.append(dict(
                draft_id=pair["draft_id"], matched=True,
                source_call_id=pair["source_call_id"], source_row=pair["source_row"],
                source_absolute_position=pair["source_absolute_position"],
                cache_pos=ref_draft["cache_pos"], q_tv=row_tv,
                support_equal=row_support_equal,
            ))
            pair["reference_q"] = ref_draft["q"]
            pair["candidate_q"] = cpu_clone(candidate_q)
            pair["reference_draft_logits"] = ref_draft["logits"]
            pair["candidate_draft_logits"] = cpu_clone(candidate_logits)
            pair["reference_h_prev"] = ref_draft["h_prev"]
        graph_stats = runtime.stats()
        graph_calls = graph_stats.get("calls", {})
        if int(graph_calls.get("graph_forward", 0)) != len(candidate_calls):
            raise RuntimeError(
                f"Not all replayed Target calls used graphs: graph_forward="
                f"{graph_calls.get('graph_forward', 0)} calls={len(candidate_calls)}; "
                f"fallbacks={graph_stats.get('fallbacks')}"
            )
        metadata.update(
            status="complete", finished_at_unix=time.time(),
            reference_matches_saved_original=reference_matches_saved,
            reference_call_count=len(reference_calls),
            trace_event_count=len(trace_events),
            truncate_count=sum(item["type"] == "truncate" for item in trace_events),
            draft_call_count=len(draft_records),
            target_comparison=target_metrics,
            draft_comparison=dict(
                draft_call_count=len(draft_records), matched_call_count=sum(
                    item.get("matched", False) for item in draft_summaries
                ), max_q_policy_tv=max(q_tv_values, default=0.0),
                q_support_equal=q_support_equal, first_q_change=first_q_change,
                calls=draft_summaries,
            ),
            runtime=dict(
                graph_count=graph_stats.get("graph_count"),
                graph_forward=graph_calls.get("graph_forward", 0),
                graph_forward_by_query={
                    key: value for key, value in graph_calls.items()
                    if key.startswith("graph_forward_T")
                },
                fallbacks=graph_stats.get("fallbacks"),
                disabled_graphs=graph_stats.get("disabled_graphs"),
                graph_events=graph_stats.get("graph_events"),
                cache_events=graph_stats.get("cache_events"),
            ),
            conclusions=dict(
                target_policy_max_tv=target_metrics["max_policy_tv"],
                target_policy_support_equal=target_metrics["policy_support_equal"],
                pre_finalnorm_hidden_max_row_l2_relative=(
                    target_metrics["max_hidden_row_l2_relative"]
                ),
                bf16_draft_policy_max_tv=max(q_tv_values, default=0.0),
                bf16_draft_policy_support_equal=q_support_equal,
                interpretation="Measured on one saved-original token path; no quality or semantic pass threshold is applied.",
            ),
        )

        torch_module.save(dict(
            reference_calls=reference_calls,
            graph_calls=candidate_calls,
            trace_events=trace_events,
            draft_comparison_raw=draft_pairs,
        ), out_dir / "raw_outputs.pt")
        metadata["raw_outputs"] = str(out_dir / "raw_outputs.pt")
        metadata["duration_seconds"] = time.time() - started
        write_json(out_dir / "metadata.json", metadata)
        print(json.dumps({
            "status": metadata["status"], "out_dir": str(out_dir),
            "reference_matches_saved_original": reference_matches_saved,
            "target_calls": len(reference_calls),
            "max_target_policy_tv": target_metrics["max_policy_tv"],
            "max_hidden_row_l2_relative": target_metrics["max_hidden_row_l2_relative"],
            "max_draft_q_policy_tv": max(q_tv_values, default=0.0),
            "graph_count": graph_stats.get("graph_count"),
            "graph_forward": graph_calls.get("graph_forward", 0),
        }, ensure_ascii=False))
        return 0
    except Exception as exc:
        metadata.update(
            status="error", error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(), finished_at_unix=time.time(),
            duration_seconds=time.time() - started,
        )
        write_json(out_dir / "metadata.json", metadata)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
