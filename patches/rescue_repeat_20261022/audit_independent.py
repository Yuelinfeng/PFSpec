#!/usr/bin/env python3
"""Independent source-weighted audit for four W1/rescue repeat directories."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import tempfile


EXPECTED_SOURCES = 30
EXPECTED_PREFIXES = (64, 256)
EXPECTED_SEEDS = (52101, 52102)
EXPECTED_RUNS = 4
EXPECTED_ROWS = 240
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 104212
EXECUTION_COUNTERS = {
    "target_calls", "target_input_tokens", "candidate_tree_nodes",
    "target_tree_nodes", "recovery_calls",
}


def _case(row):
    return str(row["source"]), int(row["prefix"]), int(row["seed"])


def _case_name(case):
    return f"{case[0]}|p{case[1]}|s{case[2]}"


def _records_path(value):
    path = Path(value).expanduser().resolve()
    return path if path.is_file() else path / "records.jsonl"


def _load(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_row_number"] = len(rows)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def _semantic_counters(row):
    counters = dict(row.get("counters") or {})
    return {key: value for key, value in counters.items()
            if key not in EXECUTION_COUNTERS}


def _semantic_signature(row):
    return (
        list(row.get("output_tokens", [])),
        int(row.get("generated_tokens", -1)),
        bool(row.get("eos", False)),
        bool(row.get("sequence_validated", False)),
        _semantic_counters(row),
    )


def _has_new_graph(row):
    if bool(row.get("runtime_created")):
        return True
    if float(row.get("graph_setup_ms", 0.0) or 0.0) > 0.0:
        return True
    for key, value in row.items():
        if (key == "runtime_delta" or key.endswith("_delta")) and isinstance(value, dict):
            if value.get("new_graph_events"):
                return True
    return False


def _quantile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return None
    location = (len(ordered) - 1) * probability
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap(log_values, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    sources = sorted(log_values)
    if not sources:
        return dict(draws=draws, seed=seed, geometric_mean=None, ci95=[None, None])
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        sample = [log_values[rng.choice(sources)] for _ in sources]
        means.append(statistics.fmean(sample))
    low = _quantile(means, 0.025)
    high = _quantile(means, 0.975)
    return dict(draws=draws, seed=seed, geometric_mean=math.exp(statistics.fmean(means)),
                ci95=[math.exp(low), math.exp(high)])


def _summarize(case_logs, case_sources, require_repeats=None):
    per_source = defaultdict(list)
    used_cases = 0
    for case, values in case_logs.items():
        if not values or (require_repeats is not None and len(values) != require_repeats):
            continue
        per_source[case_sources[case]].append(statistics.fmean(values))
        used_cases += 1
    source_logs = {source: statistics.fmean(values)
                   for source, values in per_source.items() if values}
    if not source_logs:
        return dict(case_count=0, source_count=0, geometric_mean=None, ci95=[None, None],
                    source_geometric_means={})
    mean_log = statistics.fmean(source_logs.values())
    boot = _bootstrap(source_logs)
    return dict(
        case_count=used_cases,
        source_count=len(source_logs),
        geometric_mean=math.exp(mean_log),
        ci95=boot["ci95"],
        bootstrap_draws=boot["draws"],
        bootstrap_seed=boot["seed"],
        source_geometric_means={source: math.exp(value)
                                 for source, value in sorted(source_logs.items())},
        source_log_means=source_logs,
    )


def _pair_run(rows, label, errors):
    grouped = defaultdict(list)
    for row in rows:
        try:
            grouped[_case(row)].append(row)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{label}: invalid case fields at row {row.get('_line_number')}: {error}")
    pairs = {}
    first_counts = {"w1": 0, "rescue": 0}
    for case, case_rows in grouped.items():
        arms = defaultdict(list)
        for row in case_rows:
            arms[str(row.get("method"))].append(row)
        if set(arms) != {"w1", "rescue"} or any(len(arms[name]) != 1 for name in ("w1", "rescue")):
            errors.append(f"{label}: {_case_name(case)} does not contain one W1/rescue pair")
            continue
        ordered = sorted(case_rows, key=lambda item: item["_row_number"])
        first_method = str(ordered[0]["method"])
        if first_method in first_counts:
            first_counts[first_method] += 1
        w1, rescue = arms["w1"][0], arms["rescue"][0]
        for row in (w1, rescue):
            tokens = row.get("output_tokens")
            counters = row.get("counters") or {}
            wall = row.get("outer_wall_seconds")
            if not isinstance(tokens, list) or int(row.get("generated_tokens", -1)) != len(tokens):
                errors.append(f"{label}: {_case_name(case)} {row.get('method')} output length mismatch")
            if row.get("sequence_validated") is not True:
                errors.append(f"{label}: {_case_name(case)} {row.get('method')} sequence not validated")
            if counters.get("eos") is not None and bool(counters["eos"]) != bool(row.get("eos")):
                errors.append(f"{label}: {_case_name(case)} {row.get('method')} EOS field mismatch")
            if wall is None or not math.isfinite(float(wall)) or float(wall) <= 0.0:
                errors.append(f"{label}: {_case_name(case)} {row.get('method')} invalid wall time")
        if list(w1.get("output_tokens", [])) != list(rescue.get("output_tokens", [])):
            errors.append(f"{label}: {_case_name(case)} W1/rescue token outputs differ")
        if _semantic_signature(w1) != _semantic_signature(rescue):
            errors.append(f"{label}: {_case_name(case)} W1/rescue semantic counters differ")
        w1_stats, rescue_stats = w1.get("stats") or {}, rescue.get("stats") or {}
        hits = int(rescue_stats.get("position_rescue_hits", 0))
        recovery_delta = int((w1.get("counters") or {}).get("recovery_calls", 0)) - int(
            (rescue.get("counters") or {}).get("recovery_calls", 0))
        if recovery_delta != hits:
            errors.append(f"{label}: {_case_name(case)} recovery difference {recovery_delta} != hits {hits}")
        budget = int(w1.get("budget", rescue.get("budget", 512)))
        w1_wall, rescue_wall = float(w1["outer_wall_seconds"]), float(rescue["outer_wall_seconds"])
        pairs[case] = dict(
            source=case[0], prefix=case[1], seed=case[2], first_method=first_method,
            log_ratio=math.log(w1_wall / rescue_wall),
            full_budget=(int(w1.get("generated_tokens", -1)) >= budget and
                         int(rescue.get("generated_tokens", -1)) >= budget),
            new_graph=(_has_new_graph(w1) or _has_new_graph(rescue)),
            hits=hits, recovery_delta=recovery_delta,
            w1=w1, rescue=rescue,
        )
    return pairs, first_counts


def _case_summary(pairs_by_run, case_sources, predicate=None, require_repeats=EXPECTED_RUNS):
    logs = defaultdict(list)
    for run_pairs in pairs_by_run:
        for case, pair in run_pairs.items():
            if predicate is None or predicate(pair):
                logs[case].append(pair["log_ratio"])
    summary = _summarize(logs, case_sources, require_repeats=require_repeats)
    summary["pair_observations"] = sum(len(values) for values in logs.values())
    summary["cases_with_observations"] = len(logs)
    return summary


def audit(run_dirs):
    errors = []
    labels, rows_by_run, pairs_by_run, order_by_run = [], [], [], []
    for run_index, value in enumerate(run_dirs):
        path = _records_path(value)
        label = Path(value).resolve().name
        labels.append(label)
        if not path.is_file():
            errors.append(f"{label}: missing records file {path}")
            rows, pairs, order = [], {}, {"w1": 0, "rescue": 0}
        else:
            rows = _load(path)
            if len(rows) != EXPECTED_ROWS:
                errors.append(f"{label}: expected {EXPECTED_ROWS} rows, found {len(rows)}")
            pairs, order = _pair_run(rows, label, errors)
            if len(pairs) != 120:
                errors.append(f"{label}: expected 120 complete pairs, found {len(pairs)}")
        rows_by_run.append(rows)
        pairs_by_run.append(pairs)
        order_by_run.append(order)

    all_cases = set().union(*(set(run) for run in pairs_by_run)) if pairs_by_run else set()
    case_sources = {case: case[0] for case in all_cases}
    actual_sources = {case[0] for case in all_cases}
    if len(actual_sources) != EXPECTED_SOURCES:
        errors.append(f"expected {EXPECTED_SOURCES} sources, found {len(actual_sources)}")
    expected_case_grid = {
        (source, prefix, seed)
        for source in actual_sources
        for prefix in EXPECTED_PREFIXES
        for seed in EXPECTED_SEEDS
    }
    if len(all_cases) != 120 or (actual_sources and all_cases != expected_case_grid):
        errors.append(f"expected 120 cases on prefix/seed grid, found {len(all_cases)}")
    for run_index, run_pairs in enumerate(pairs_by_run):
        if set(run_pairs) != all_cases:
            errors.append(f"{labels[run_index]}: case set differs from suite union")
        for case, pair in run_pairs.items():
            for arm in ("w1", "rescue"):
                row = pair[arm]
                if (int(row.get("prefix", -1)) != case[1] or
                        int(row.get("seed", -1)) != case[2] or
                        str(row.get("source")) != case[0]):
                    errors.append(f"{labels[run_index]}: case key disagrees with row fields")
    for case in all_cases:
        for arm in ("w1", "rescue"):
            signatures = [_semantic_signature(run[case][arm])
                          for run in pairs_by_run if case in run]
            if signatures and any(item != signatures[0] for item in signatures[1:]):
                errors.append(f"cross-run {arm} output/core semantics differ for {_case_name(case)}")
            outputs = [list(run[case][arm].get("output_tokens", []))
                       for run in pairs_by_run if case in run]
            if outputs and any(item != outputs[0] for item in outputs[1:]):
                errors.append(f"cross-run {arm} tokens differ for {_case_name(case)}")

    overall = _case_summary(pairs_by_run, case_sources)
    per_run = []
    for run_index, run_pairs in enumerate(pairs_by_run):
        logs = {case: [pair["log_ratio"]] for case, pair in run_pairs.items()}
        summary = _summarize(logs, case_sources, require_repeats=1)
        per_run.append(dict(run_index=run_index, label=labels[run_index], **summary))

    per_seed = {}
    for seed in EXPECTED_SEEDS:
        logs = defaultdict(list)
        for run_pairs in pairs_by_run:
            for case, pair in run_pairs.items():
                if case[2] == seed:
                    logs[case].append(pair["log_ratio"])
        per_seed[str(seed)] = _summarize(logs, case_sources, require_repeats=EXPECTED_RUNS)

    full_budget = _case_summary(
        pairs_by_run, case_sources, predicate=lambda pair: pair["full_budget"],
        require_repeats=None)
    no_new_graph = _case_summary(
        pairs_by_run, case_sources, predicate=lambda pair: not pair["new_graph"],
        require_repeats=None)
    first_graph = _case_summary(
        pairs_by_run, case_sources, predicate=lambda pair: pair["new_graph"],
        require_repeats=None)

    order_summaries = {}
    for first_method in ("w1", "rescue"):
        logs = defaultdict(list)
        counts = 0
        for run_pairs in pairs_by_run:
            for case, pair in run_pairs.items():
                if pair["first_method"] == first_method:
                    logs[case].append(pair["log_ratio"])
                    counts += 1
        order_summaries[first_method] = dict(
            pair_observations=counts,
            summary=_summarize(logs, case_sources),
        )

    leave_one = []
    source_logs = overall.get("source_log_means", {})
    if len(source_logs) > 1:
        for omitted in sorted(source_logs):
            remaining = [value for source, value in source_logs.items() if source != omitted]
            leave_one.append((omitted, math.exp(statistics.fmean(remaining))))

    total_hits = sum(pair["hits"] for run in pairs_by_run for pair in run.values())
    total_recovery_difference = sum(pair["recovery_delta"]
                                    for run in pairs_by_run for pair in run.values())
    report = dict(
        status="passed" if not errors else "failed",
        input_run_directories=[str(Path(value).expanduser().resolve()) for value in run_dirs],
        integrity=dict(
            run_count=len(run_dirs), rows_per_run=[len(rows) for rows in rows_by_run],
            complete_pairs_per_run=[len(run) for run in pairs_by_run],
            unique_cases=len(all_cases), source_count=len(actual_sources),
            prefixes=sorted({case[1] for case in all_cases}),
            seeds=sorted({case[2] for case in all_cases}),
            pair_output_and_semantics="checked; execution-node and recovery counters excluded",
            cross_run_outputs_and_core_semantics="checked per arm",
            recovery_difference_equals_rescue_hits=dict(
                checked_pairs=sum(len(run) for run in pairs_by_run),
                recovery_difference=total_recovery_difference, rescue_hits=total_hits,
                equal=(total_recovery_difference == total_hits)),
        ),
        timing=dict(
            definition="case mean of per-run log(W1 wall / rescue wall), then equal cases within source, then equal sources",
            overall=overall,
            per_run=per_run,
            per_seed=per_seed,
            full_budget_cases=full_budget,
            no_new_graph_cases=no_new_graph,
            first_graph_cases=first_graph,
            method_order_from_record_row_order=dict(
                counts_by_run=[dict(run_index=i, label=labels[i], **order_by_run[i])
                               for i in range(len(order_by_run))],
                first_w1=order_summaries["w1"],
                first_rescue=order_summaries["rescue"],
            ),
            leave_one_source_geometric_mean_range=(
                [min(value for _, value in leave_one), max(value for _, value in leave_one)]
                if leave_one else [None, None]),
            leave_one_source_values={source: value for source, value in leave_one},
        ),
        errors=errors,
    )
    return report


def _synthetic_row(source, prefix, seed, method, seconds, graph_ms):
    recovery = 1 if method == "w1" else 0
    counters = dict(
        generated_tokens=512, eos=False, accepted=3, position_accepted=2,
        feature_accepted=1, recovery_calls=recovery, target_calls=10 + recovery,
        target_input_tokens=50 + recovery, candidate_tree_nodes=4 if method == "w1" else 5,
        target_tree_nodes=8 if method == "w1" else 10,
    )
    return dict(
        source=source, prefix=prefix, seed=seed, method=method, budget=512,
        output_tokens=list(range(512)), generated_tokens=512, eos=False,
        sequence_validated=True, outer_wall_seconds=seconds, graph_setup_ms=graph_ms,
        counters=counters, stats={"position_rescue_hits": int(method == "rescue")},
        runtime_delta={"new_graph_events": []},
    )


def self_test():
    assert _has_new_graph({"runtime_created": True})
    for key in ("runtime_delta", "proposal_runtime_delta", "tree_runtime_delta",
                "rescue_runtime_delta"):
        assert _has_new_graph({key: {"new_graph_events": [{"capture_ms": 1.0}]}})
        assert not _has_new_graph({key: {"new_graph_events": []}})
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        runs = []
        for run_index in range(EXPECTED_RUNS):
            run_dir = root / f"repeat_{run_index + 1:02d}"
            run_dir.mkdir()
            records = []
            for source_index in range(EXPECTED_SOURCES):
                source = f"scene-{source_index:02d}"
                for prefix in EXPECTED_PREFIXES:
                    for seed in EXPECTED_SEEDS:
                        case_index = source_index * 4 + (prefix == 256) * 2 + (seed == 52102)
                        methods = ("w1", "rescue") if (run_index + case_index) % 2 == 0 else ("rescue", "w1")
                        graph_ms = 1.0 if (run_index == 0 and case_index == 0) else 0.0
                        for method in methods:
                            seconds = 2.0 if method == "w1" else 1.0
                            records.append(_synthetic_row(source, prefix, seed, method,
                                                          seconds, graph_ms))
            path = run_dir / "records.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
            runs.append(run_dir)
        report = audit(runs)
        assert report["status"] == "passed", report["errors"][:5]
        timing = report["timing"]
        assert abs(timing["overall"]["geometric_mean"] - 2.0) < 1e-12
        assert timing["overall"]["ci95"] == [2.0, 2.0]
        assert all(abs(run["geometric_mean"] - 2.0) < 1e-12 for run in timing["per_run"])
        assert abs(timing["full_budget_cases"]["geometric_mean"] - 2.0) < 1e-12
        assert abs(timing["no_new_graph_cases"]["geometric_mean"] - 2.0) < 1e-12
        assert timing["no_new_graph_cases"]["pair_observations"] == 479
        assert timing["first_graph_cases"]["case_count"] == 1
        assert timing["first_graph_cases"]["pair_observations"] == 1
        assert sum(item["w1"] + item["rescue"] for item in
                   timing["method_order_from_record_row_order"]["counts_by_run"]) == 480
        json.dumps(report, allow_nan=False)
        return dict(status="passed", runs=4, rows_per_run=240, unique_cases=120,
                    known_geometric_mean=2.0, full_budget=2.0, no_new_graph=2.0,
                    equal_outputs=True, record_order_counted=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="*", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False))
        return 0
    if len(args.run_dirs) != EXPECTED_RUNS or args.out is None:
        parser.error("provide four run directories and --out JSON")
    report = audit(args.run_dirs)
    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(dict(status=report["status"], output=str(output),
                          errors=len(report["errors"])), ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
