#!/usr/bin/env python3
"""Summarize CUDA Graph use and synchronized timing from an existing run directory."""

import argparse
import json
import math
from pathlib import Path
import random


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def quantile(values, probability):
    values = sorted(values)
    point = (len(values) - 1) * probability
    low = int(point)
    high = min(low + 1, len(values) - 1)
    if high == low:
        return values[low]
    return values[low] * (high - point) + values[high] * (point - low)


def paired_ratio(rows, numerator, denominator, budget, full_only=False, selected=None,
                 selected_pair=None):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["pair_id"], {})[row["arm"]] = row

    source_logs = {}
    pair_ids = []
    for pair_id, arms in grouped.items():
        if numerator not in arms or denominator not in arms:
            continue
        top, bottom = arms[numerator], arms[denominator]
        if selected is not None and not selected(top):
            continue
        if selected_pair is not None and not selected_pair(top, bottom):
            continue
        if full_only and (top["generated_tokens"] != budget or bottom["generated_tokens"] != budget):
            continue
        if min(top["generated_tokens"], bottom["generated_tokens"]) <= 0:
            continue
        log_ratio = math.log(top["generated_tokens"] / top["outer_wall_seconds"])
        log_ratio -= math.log(bottom["generated_tokens"] / bottom["outer_wall_seconds"])
        source_logs.setdefault(top["source"], []).append(log_ratio)
        pair_ids.append(pair_id)

    per_source_logs = {source: math.fsum(logs) / len(logs)
                       for source, logs in source_logs.items()}
    sources = sorted(per_source_logs)
    result = dict(pair_count=len(pair_ids), source_count=len(sources),
                  source_equal_geometric_mean=None, per_source={},
                  source_bootstrap_95_ci=None, pair_ids=sorted(pair_ids),
                  sample_note="no eligible paired requests" if not pair_ids else None)
    if not sources:
        return result

    result["source_equal_geometric_mean"] = math.exp(
        math.fsum(per_source_logs.values()) / len(sources))
    result["per_source"] = {source: math.exp(per_source_logs[source]) for source in sources}
    if len(sources) > 1:
        rng = random.Random(402101)
        samples = [math.exp(math.fsum(per_source_logs[rng.choice(sources)] for _ in sources)
                            / len(sources)) for _ in range(10000)]
        result["source_bootstrap_95_ci"] = [quantile(samples, .025), quantile(samples, .975)]
    else:
        result["sample_note"] = "one source; source-level bootstrap interval unavailable"
    return result


def graph_shape(event):
    kinds = event.get("logit_types")
    mode = "full" if kinds is None else "typed"
    return f"T{event['query_tokens']}/read{event['read_bucket']}/{mode}"


def stable_graph_row(row):
    if row.get("execution") != "optimized":
        return False
    delta = row.get("runtime_delta") or {}
    return (row.get("runtime_created") is False
            and not delta.get("new_graph_events")
            and delta.get("calls", {}).get("graph_forward", 0) > 0)


def event_totals(events):
    return dict(
        graph_event_count=len(events),
        prepare_ms=math.fsum(item.get("prepare_ms", 0.0) for item in events),
        warmup_ms=math.fsum(item.get("warmup_ms", 0.0) for item in events),
        capture_ms=math.fsum(item.get("capture_ms", 0.0) for item in events),
        allocated_delta_bytes=math.fsum(item.get("memory_allocated_delta", 0) for item in events),
    )


def runtime_method_report(method, runtime, rows):
    events = runtime.get("graph_events", [])
    shape_counts = {}
    for event in events:
        shape = graph_shape(event)
        shape_counts[shape] = shape_counts.get(shape, 0) + 1

    method_rows = [row for row in rows if row.get("method") == method
                   and row.get("execution") == "optimized"]
    first_runtime_events = []
    later_events = []
    newly_built_by_pair = {}
    for row in method_rows:
        delta = row.get("runtime_delta") or {}
        new_events = delta.get("new_graph_events", [])
        if row.get("runtime_created"):
            first_runtime_events.extend(new_events)
        else:
            later_events.extend(new_events)
        if new_events:
            newly_built_by_pair[row["pair_id"]] = [graph_shape(item) for item in new_events]

    calls = runtime.get("calls", {})
    return dict(
        graph_count=runtime.get("graph_count", len(events)),
        graph_event_count=len(events),
        graph_shapes=shape_counts,
        calls=calls,
        graph_forward_calls=calls.get("graph_forward", 0),
        full_read_calls=calls.get("graph_full_forward", 0),
        return_hidden_calls=calls.get("graph_return_hidden", 0),
        original_forward_calls=calls.get("original_forward", 0),
        fallbacks=runtime.get("fallbacks", {}),
        cache_bytes_by_capacity=runtime.get("cache_bytes_by_capacity", {}),
        cache_allocate_events=[row.get("runtime_delta", {}).get("new_cache_events", [])
                               for row in method_rows if row.get("runtime_delta", {}).get("new_cache_events")],
        runtime_init_ms=runtime.get("runtime_init_ms"),
        graph_setup_all_measured_requests=event_totals(events),
        graph_setup_on_runtime_creation=event_totals(first_runtime_events),
        graph_setup_on_later_bucket_builds=event_totals(later_events),
        newly_built_graphs_by_pair=newly_built_by_pair,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_path = args.out.resolve() if args.out else run_dir / "runtime_diagnostics.json"
    if out_path.exists():
        parser.error(f"Output already exists: {out_path}")

    plan = read_json(run_dir / "plan.json")
    summary = read_json(run_dir / "summary.json")
    rows = read_rows(run_dir / "records.jsonl")
    methods = summary.get("methods", plan.get("methods", []))
    executions = summary.get("executions", plan.get("executions", []))
    budget = int(plan["budget"])

    main_ratios = {}
    steady_ratios = {}
    steady_pair_ids = {}
    algorithm_gain_optimized = {}
    algorithm_steady_pair_ids = {}
    runtime_methods = {}
    dynamic_runtimes = summary.get("dynamic_runtimes", {})
    reported_contrasts = summary.get("contrasts", {})
    available_arms = {row.get("arm") for row in rows}
    for method in methods:
        original = f"original:{method}"
        optimized = f"optimized:{method}"
        if original in available_arms and optimized in available_arms:
            main_ratios[method] = dict(
                all_output_lengths=paired_ratio(rows, optimized, original, budget),
                common_full_budget=paired_ratio(rows, optimized, original, budget, full_only=True),
            )

            def stable_graph_request(row, selected_method=method):
                return row.get("method") == selected_method and stable_graph_row(row)

            steady = paired_ratio(rows, optimized, original, budget, selected=stable_graph_request)
            steady_full = paired_ratio(rows, optimized, original, budget, full_only=True,
                                       selected=stable_graph_request)
            steady_pair_ids[method] = steady["pair_ids"]
            steady_ratios[method] = dict(all_output_lengths=steady, common_full_budget=steady_full)

        if method != "target" and optimized in available_arms and "optimized:target" in available_arms:
            summary_key = f"algorithm_gain:optimized:{method}"
            reported = reported_contrasts.get(summary_key)
            if reported is None:
                reported = dict(
                    numerator=optimized,
                    denominator="optimized:target",
                    all_output_lengths=paired_ratio(rows, optimized, "optimized:target", budget),
                    common_full_budget=paired_ratio(rows, optimized, "optimized:target", budget,
                                                    full_only=True),
                )

            def common_stable_graph_pair(top, bottom):
                return stable_graph_row(top) and stable_graph_row(bottom)

            stable_all = paired_ratio(rows, optimized, "optimized:target", budget,
                                      selected_pair=common_stable_graph_pair)
            stable_full = paired_ratio(rows, optimized, "optimized:target", budget,
                                       full_only=True, selected_pair=common_stable_graph_pair)
            algorithm_steady_pair_ids[method] = stable_all["pair_ids"]
            algorithm_gain_optimized[method] = dict(
                all_requests=reported,
                common_stable_no_new_graph_subset=dict(
                    criteria="both optimized rows have runtime_created=false, no new graph events, and at least one graph forward; all wall time remains in each paired measurement.",
                    all_output_lengths=stable_all,
                    common_full_budget=stable_full,
                ),
            )

        if method in dynamic_runtimes:
            runtime_methods[method] = runtime_method_report(method, dynamic_runtimes[method], rows)

    cache_total = 0
    for runtime in dynamic_runtimes.values():
        cache_total += sum(int(value) for value in runtime.get("cache_bytes_by_capacity", {}).values())

    warmups = []
    warmup_path = run_dir / "warmup.json"
    if warmup_path.exists():
        for item in read_json(warmup_path):
            runtime = item.get("runtime") or {}
            events = runtime.get("graph_events", [])
            warmups.append(dict(
                method=item.get("method"), execution=item.get("execution"),
                elapsed_seconds=item.get("seconds"),
                graph_count=runtime.get("graph_count", len(events)),
                graph_forward_calls=runtime.get("calls", {}).get("graph_forward", 0),
                graph_setup=event_totals(events),
                discarded_runtime=True,
            ))

    payload = dict(
        status="complete" if summary.get("status") == "complete" else "partial",
        run_dir=str(run_dir),
        completed_runs=len(rows), expected_runs=summary.get("expected_runs"),
        methods=methods, executions=executions, budget=budget,
        main_timing_scope=summary.get("timing_scope", plan.get("timing_scope")),
        main_paired_wall_time_results=main_ratios,
        steady_no_new_graph_subset=dict(
            criteria="optimized row has runtime_created=false, no new graph events, and at least one graph forward; matching original row remains in the pair. All wall time is retained, including prefill and any expected fallback.",
            pair_ids_by_method=steady_pair_ids,
            paired_results=steady_ratios,
        ),
        algorithm_gain_optimized=algorithm_gain_optimized,
        algorithm_gain_optimized_common_stable_pair_ids=algorithm_steady_pair_ids,
        runtime_by_method=runtime_methods,
        simultaneous_target_kv_cache_bytes=cache_total,
        discarded_warmup_runtime_summary=warmups,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=payload["status"], output=str(out_path),
                          completed_runs=len(rows), methods=len(methods),
                          steady_pairs=sum(len(value) for value in steady_pair_ids.values()),
                          algorithm_steady_pairs=sum(len(value) for value in algorithm_steady_pair_ids.values()),
                          graph_forwards={key:value.get("graph_forward_calls", 0)
                                          for key,value in runtime_methods.items()}),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
