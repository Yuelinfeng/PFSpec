#!/usr/bin/env python3
"""Analyze repeated W1/rescue runs using only the Python standard library."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_BOOTSTRAP = 10000
DEFAULT_SEED = 402101
EXPECTED_RUNS = 4
EXPECTED_SOURCES = 30
EXPECTED_PREFIXES = [64, 256]
EXPECTED_SEEDS = [52101, 52102]
EXPECTED_METHODS = {"w1", "rescue"}
EXPECTED_BUDGET = 512
EXPECTED_REPEATS = 4
EXPECTED_SOURCE_OFFSETS = [0, 8, 15, 23]

PAIR_COST_KEYS = {
    "target_calls",
    "target_input_tokens",
    "target_tree_nodes",
    "candidate_tree_nodes",
    "recovery_calls",
    "position_backup_nodes",
    "position_backup_rejections",
    "position_rescue_hits",
}
STATS_IGNORED_KEYS = {
    "seconds",
    "tokens_per_second",
    "target_calls",
    "target_input_tokens",
    "target_tree_nodes",
    "candidate_tree_nodes",
    "recovery_calls",
    "position_backup_nodes",
    "position_backup_rejections",
    "position_rescue_hits",
    "position_rescue_rate_per_backup_node",
    "tree_runtime_stats",
    "rescue_runtime_stats",
    "proposal_runtime_stats",
    "runtime_stats",
    "variant",
    "tree_width",
    "proposal_width",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _method(row: dict[str, Any]) -> str | None:
    value = row.get("method")
    if value is not None:
        return str(value)
    arm = row.get("arm")
    if arm in ("w1", "rescue"):
        return str(arm)
    if isinstance(arm, str) and arm.endswith(":w1"):
        return "w1"
    if isinstance(arm, str) and arm.endswith(":rescue"):
        return "rescue"
    return None


def _source(row: dict[str, Any]) -> str | None:
    value = row.get("source", row.get("source_id", row.get("scene")))
    return None if value is None else str(value)


def _case_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        _source(row) or "<missing-source>",
        str(row.get("prefix", "<missing-prefix>")),
        str(row.get("seed", "<missing-seed>")),
        str(row.get("budget", "<missing-budget>")),
    )


def _case_label(key: tuple[str, str, str, str]) -> str:
    return f"source={key[0]} prefix={key[1]} seed={key[2]} budget={key[3]}"


def _generated(row: dict[str, Any]) -> int | None:
    value = row.get("generated_tokens")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    output = row.get("output_tokens", row.get("output"))
    return len(output) if isinstance(output, list) else None


def _output_tokens(row: dict[str, Any]) -> Any:
    return row.get("output_tokens", row.get("output"))


def _tps(row: dict[str, Any]) -> float | None:
    value = row.get("tokens_per_second")
    if value is not None:
        try:
            result = float(value)
            if math.isfinite(result) and result > 0.0:
                return result
        except (TypeError, ValueError):
            pass
    count = _generated(row)
    seconds = row.get("outer_wall_seconds")
    if count is None or seconds is None:
        return None
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds) or seconds <= 0.0:
        return None
    return count / seconds


def _number(row: dict[str, Any], key: str) -> float | None:
    for container in (row.get("counters"), row.get("stats")):
        if isinstance(container, dict) and key in container:
            value = container[key]
            if isinstance(value, bool):
                return float(value)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _has_capture(row: dict[str, Any]) -> bool:
    try:
        if float(row.get("graph_setup_ms", 0.0) or 0.0) > 0.0:
            return True
    except (TypeError, ValueError):
        pass
    for key, value in row.items():
        if key.endswith("_delta") and isinstance(value, dict):
            if value.get("new_graph_events"):
                return True
    return bool(row.get("new_graph_events"))


def _has_new_graph(row: dict[str, Any]) -> bool:
    return bool(row.get("runtime_created")) or _has_capture(row)


def _semantic_view(row: dict[str, Any]) -> dict[str, Any]:
    counters = row.get("counters") or {}
    stats = row.get("stats") or {}
    if not isinstance(counters, dict):
        counters = {}
    if not isinstance(stats, dict):
        stats = {}
    return {
        "counters": {
            key: value for key, value in counters.items()
            if key not in PAIR_COST_KEYS
        },
        "stats": {
            key: value for key, value in stats.items()
            if key not in STATS_IGNORED_KEYS and not key.endswith("_runtime_stats")
        },
    }


def _planned_seeds(plan: dict[str, Any]) -> list[int] | None:
    value = plan.get("seeds")
    if isinstance(value, list):
        try:
            return sorted({int(item) for item in value})
        except (TypeError, ValueError):
            return None
    if isinstance(value, int):
        start = plan.get("seed_start")
        if start is not None:
            try:
                return list(range(int(start), int(start) + value))
            except (TypeError, ValueError):
                return None
    start, count = plan.get("seed_start"), plan.get("seed_count")
    if start is not None and count is not None:
        try:
            return list(range(int(start), int(start) + int(count)))
        except (TypeError, ValueError):
            return None
    return None


def _read_run(path: Path, index: int) -> dict[str, Any]:
    path = path.expanduser().resolve()
    records_path = path / "records.jsonl" if path.is_dir() else path
    root = records_path.parent
    if not records_path.exists():
        raise FileNotFoundError(f"missing records file: {records_path}")
    plan_path = root / "plan.json"
    pipeline_path = root / "pipeline.json"
    schedule_path = root / "repeat_schedule.json"
    suite_path = root.parent / "suite.json"
    plan = read_json(plan_path) if plan_path.exists() else {}
    pipeline = read_json(pipeline_path) if pipeline_path.exists() else {}
    schedule = read_json(schedule_path) if schedule_path.exists() else {}
    suite = read_json(suite_path) if suite_path.exists() else {}
    rows = read_jsonl(records_path)
    prompts_path = root / "prompts.json"
    prompts = read_json(prompts_path) if prompts_path.exists() else []
    if isinstance(prompts, dict):
        prompts = prompts.get("prompts", [])
    run_name = str(plan.get("run_id") or pipeline.get("run_id") or root.name)
    repeat_index = schedule.get("repeat_index", plan.get("repeat_index"))
    suite_entry = None
    if isinstance(suite.get("repeats"), list):
        for entry in suite["repeats"]:
            if entry.get("name") == root.name or (
                repeat_index is not None and entry.get("repeat_index") == repeat_index
            ):
                suite_entry = entry
                break
    process_values = {}
    for key in ("pid", "process_id", "runner_pid", "worker_pid"):
        value = next((source.get(key) for source in (schedule, suite_entry or {}, pipeline, plan)
                      if source.get(key) is not None), None)
        if value is not None:
            process_values[key] = value
    row_process_ids = sorted({str(row["process_id"]) for row in rows if row.get("process_id") is not None})
    if row_process_ids:
        process_values["record_process_id"] = row_process_ids
    schedule_fields = {
        key: value
        for source_name, source_data in (("plan", plan), ("pipeline", pipeline),
                                         ("repeat_schedule", schedule), ("suite_entry", suite_entry or {}))
        for key, value in source_data.items()
        if any(token in key.lower() for token in ("offset", "order", "schedule", "cold"))
    }
    return {
        "index": index,
        "path": str(root),
        "records_path": str(records_path),
        "name": run_name,
        "plan": plan,
        "pipeline": pipeline,
        "repeat_schedule": schedule,
        "suite_entry": suite_entry,
        "suite": suite,
        "repeat_index": repeat_index,
        "rows": rows,
        "prompts": prompts,
        "process_values": process_values,
        "schedule_fields": schedule_fields,
    }


def _pair_rows(run: dict[str, Any]) -> tuple[dict[tuple[str, str, str, str], dict[str, list]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    issues = []
    for row_index, row in enumerate(run["rows"]):
        method = _method(row)
        if method is None:
            issues.append({"run": run["name"], "row_index": row_index, "issue": "unknown_method"})
            continue
        groups[_case_key(row)][method].append((row_index, row))
    return groups, issues


def _expected_order(run: dict[str, Any]) -> list[str] | None:
    methods = run.get("repeat_schedule", {}).get("method_order")
    if not isinstance(methods, list):
        methods = (run.get("suite_entry") or {}).get("method_order")
    if not isinstance(methods, list):
        methods = run["plan"].get("methods")
    if isinstance(methods, list):
        converted = [str(item).lower() for item in methods]
        if set(converted) == EXPECTED_METHODS and len(converted) == 2:
            return converted
    return None


def _expected_pair_order(run: dict[str, Any], key: tuple[str, str, str, str]) -> list[str] | None:
    order = _expected_order(run)
    prompts = run.get("prompts", [])
    if not order or not isinstance(prompts, list):
        return None
    for prompt_index, prompt in enumerate(prompts):
        if (_source(prompt), str(prompt.get("prefix")), str(prompt.get("seed"))) == key[:3]:
            rotation = (prompt_index + int(key[3])) % len(order)
            return order[rotation:] + order[:rotation]
    return None


def _pair_observation(run: dict[str, Any], key: tuple[str, str, str, str], pair: dict[str, list]) -> dict[str, Any] | None:
    if len(pair.get("w1", [])) != 1 or len(pair.get("rescue", [])) != 1:
        return None
    w1_index, w1 = pair["w1"][0]
    rescue_index, rescue = pair["rescue"][0]
    w1_output, rescue_output = _output_tokens(w1), _output_tokens(rescue)
    output_available = isinstance(w1_output, list) and isinstance(rescue_output, list)
    output_equal = output_available and w1_output == rescue_output
    generated_equal = _generated(w1) == _generated(rescue)
    eos_equal = bool(w1.get("eos")) == bool(rescue.get("eos"))
    input_context_equal = w1.get("context_id") == rescue.get("context_id")
    prompt_tokens_equal = w1.get("prompt_tokens") == rescue.get("prompt_tokens")
    output_equivalent = bool(output_equal and generated_equal and eos_equal
                             and input_context_equal and prompt_tokens_equal)
    semantic_equal = _semantic_view(w1) == _semantic_view(rescue)
    w1_tps, rescue_tps = _tps(w1), _tps(rescue)
    tps_speedup = (
        rescue_tps / w1_tps
        if w1_tps is not None and rescue_tps is not None and w1_tps > 0.0
        else None
    )
    w1_wall, rescue_wall = w1.get("outer_wall_seconds"), rescue.get("outer_wall_seconds")
    try:
        w1_wall, rescue_wall = float(w1_wall), float(rescue_wall)
    except (TypeError, ValueError):
        w1_wall = rescue_wall = None
    if w1_wall is not None and rescue_wall is not None and w1_wall > 0.0 and rescue_wall > 0.0:
        wall_ratio = rescue_wall / w1_wall
        wall_speedup = w1_wall / rescue_wall
    else:
        wall_ratio = wall_speedup = None
    w1_recovery = _number(w1, "recovery_calls")
    rescue_recovery = _number(rescue, "recovery_calls")
    rescue_hits = _number(rescue, "position_rescue_hits")
    recovery_saved = (
        w1_recovery - rescue_recovery
        if w1_recovery is not None and rescue_recovery is not None
        else None
    )
    recovery_match = (
        recovery_saved == rescue_hits
        if recovery_saved is not None and rescue_hits is not None
        else None
    )
    run_budget = rescue.get("budget", w1.get("budget"))
    try:
        budget = int(run_budget)
    except (TypeError, ValueError):
        budget = None
    full_budget = (
        budget is not None
        and _generated(w1) == budget
        and _generated(rescue) == budget
    )
    any_new_graph = _has_new_graph(w1) or _has_new_graph(rescue)
    any_capture = _has_capture(w1) or _has_capture(rescue)

    delta_fields = {
        "target_tree_nodes": ("target_tree_nodes", "target_tree_nodes"),
        "target_input_tokens": ("target_input_tokens", "target_input_tokens"),
        "target_calls": ("target_calls", "target_calls"),
        "recovery_calls": ("recovery_calls", "recovery_calls"),
        "position_backup_nodes": (None, "position_backup_nodes"),
        "position_backup_rejections": (None, "position_backup_rejections"),
        "position_rescue_hits": (None, "position_rescue_hits"),
    }
    deltas = {}
    for label, (w1_key, rescue_key) in delta_fields.items():
        left = _number(w1, w1_key) if w1_key else 0.0
        right = _number(rescue, rescue_key)
        deltas[label] = (right - left) if left is not None and right is not None else None
    deltas["recovery_calls_saved"] = recovery_saved

    return {
        "run": run["name"],
        "run_index": run["index"],
        "source": key[0],
        "prefix": key[1],
        "seed": key[2],
        "budget": key[3],
        "case": _case_label(key),
        "pair_id_w1": w1.get("pair_id"),
        "pair_id_rescue": rescue.get("pair_id"),
        "pair_row_order": [name for _, name in sorted(((w1_index, "w1"), (rescue_index, "rescue")))],
        "output_equal": bool(output_equal),
        "generated_count_equal": bool(generated_equal),
        "eos_equal": bool(eos_equal),
        "input_context_equal": bool(input_context_equal),
        "prompt_tokens_equal": bool(prompt_tokens_equal),
        "output_equivalent": output_equivalent,
        "semantic_counters_equal": semantic_equal,
        "tps_speedup_rescue_over_w1": tps_speedup,
        "rescue_over_w1_wall_time_ratio": wall_ratio,
        "wall_speedup_rescue_over_w1": wall_speedup,
        "wall_ratio_equivalent_work": wall_speedup if output_equivalent else None,
        "generated_tokens_w1": _generated(w1),
        "generated_tokens_rescue": _generated(rescue),
        "w1_wall_seconds": w1_wall,
        "rescue_wall_seconds": rescue_wall,
        "full_budget": bool(full_budget),
        "no_new_graphs": not any_new_graph,
        "cold_graph_capture": any_capture,
        "rescue_hits": rescue_hits,
        "recovery_calls_saved": recovery_saved,
        "recovery_delta_equals_hits": recovery_match,
        "deltas": deltas,
        "w1_row_index": w1_index,
        "rescue_row_index": rescue_index,
    }


def _percentile_interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    values = sorted(values)
    lo = max(0, int(math.floor(0.025 * (len(values) - 1))))
    hi = min(len(values) - 1, int(math.ceil(0.975 * (len(values) - 1))))
    return [values[lo], values[hi]]


def _source_metric(observations: list[dict[str, Any]], metric: str, bootstrap: int, seed: int,
                   leave_one_out: bool = False) -> dict[str, Any]:
    by_source_case: dict[str, dict[tuple[str, str, str], list[float]]] = defaultdict(lambda: defaultdict(list))
    missing = 0
    for observation in observations:
        value = observation.get(metric)
        if value is None:
            missing += 1
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            missing += 1
            continue
        if not math.isfinite(value) or value <= 0.0:
            missing += 1
            continue
        case = (str(observation["prefix"]), str(observation["seed"]), str(observation["budget"]))
        by_source_case[str(observation["source"])][case].append(math.log(value))
    case_logs = {
        source: {case: statistics.fmean(values) for case, values in cases.items() if values}
        for source, cases in by_source_case.items()
    }
    source_logs = {
        source: statistics.fmean(values.values())
        for source, values in case_logs.items() if values
    }
    if not source_logs:
        return {
            "observation_count": len(observations),
            "valid_observation_count": 0,
            "missing_or_invalid_count": missing,
            "source_count": 0,
            "source_equal_gm": None,
            "ci95": None,
            "per_source_gm": {},
            "case_count_by_source": {},
            "source_signs": {"positive": [], "negative": [], "equal": []},
            "leave_one_source_out": None,
        }
    names = sorted(source_logs)
    point = math.exp(statistics.fmean(source_logs.values()))
    samples = []
    if bootstrap > 0 and len(names) > 1:
        rng = random.Random(seed)
        for _ in range(bootstrap):
            selected = [source_logs[rng.choice(names)] for _ in names]
            samples.append(math.exp(statistics.fmean(selected)))
    per_source = {name: math.exp(source_logs[name]) for name in names}
    signs = {
        "positive": [name for name in names if per_source[name] > 1.0],
        "negative": [name for name in names if per_source[name] < 1.0],
        "equal": [name for name in names if per_source[name] == 1.0],
    }
    loo = None
    if leave_one_out and len(names) > 1:
        values = {
            name: math.exp(statistics.fmean(value for other, value in source_logs.items() if other != name))
            for name in names
        }
        low_name = min(values, key=values.get)
        high_name = max(values, key=values.get)
        loo = {
            "minimum": values[low_name],
            "minimum_omitted_source": low_name,
            "maximum": values[high_name],
            "maximum_omitted_source": high_name,
            "per_omitted_source": values,
        }
    return {
        "observation_count": len(observations),
        "valid_observation_count": len(observations) - missing,
        "missing_or_invalid_count": missing,
        "source_count": len(source_logs),
        "source_equal_gm": point,
        "ci95": _percentile_interval(samples),
        "bootstrap_samples": len(samples),
        "bootstrap_seed": seed,
        "bootstrap_unit": "source",
        "per_source_gm": per_source,
        "case_count_by_source": {source: len(values) for source, values in case_logs.items()},
        "source_signs": {
            "positive_count": len(signs["positive"]),
            "negative_count": len(signs["negative"]),
            "equal_count": len(signs["equal"]),
            **signs,
        },
        "leave_one_source_out": loo,
    }


def _sum_delta(observations: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [item["deltas"].get(key) for item in observations]
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return {
        "pair_count_with_value": len(numeric),
        "total": sum(numeric) if numeric else None,
        "mean_per_pair": statistics.fmean(numeric) if numeric else None,
    }


def _scope_summary(observations: list[dict[str, Any]], bootstrap: int, seed: int,
                   leave_one_out: bool = False) -> dict[str, Any]:
    equivalent = [item for item in observations if item.get("output_equivalent")]
    recovery_checks = [item.get("recovery_delta_equals_hits") for item in observations]
    return {
        "pair_observation_count": len(observations),
        "source_count": len({item["source"] for item in observations}),
        "output_equivalent_pairs": sum(bool(item.get("output_equivalent")) for item in observations),
        "semantic_counter_equal_pairs": sum(bool(item.get("semantic_counters_equal")) for item in observations),
        "recovery_delta_hit_matches": sum(value is True for value in recovery_checks),
        "recovery_delta_hit_mismatches": sum(value is False for value in recovery_checks),
        "recovery_relation_unavailable": sum(value is None for value in recovery_checks),
        "tps_speedup_rescue_over_w1": _source_metric(
            observations, "tps_speedup_rescue_over_w1", bootstrap, seed, leave_one_out
        ),
        "wall_speedup_equivalent_output_only": _source_metric(
            equivalent, "wall_speedup_rescue_over_w1", bootstrap, seed, leave_one_out
        ),
        "equivalent_wall_ratio_rescue_over_w1": _source_metric(
            equivalent, "rescue_over_w1_wall_time_ratio", bootstrap, seed, leave_one_out
        ),
        "cost_deltas": {
            key: _sum_delta(observations, key)
            for key in (
                "target_tree_nodes", "target_input_tokens", "target_calls",
                "recovery_calls", "recovery_calls_saved", "position_backup_nodes",
                "position_backup_rejections", "position_rescue_hits",
            )
        },
    }


def _scopes(observations: list[dict[str, Any]], bootstrap: int, seed: int,
            leave_one_out: bool = False) -> dict[str, Any]:
    filters = {
        "all_pairs": lambda x: True,
        "both_full_512": lambda x: x["full_budget"],
        "no_new_graphs": lambda x: x["no_new_graphs"],
        "full_512_and_no_new_graphs": lambda x: x["full_budget"] and x["no_new_graphs"],
        "cold_graph_capture": lambda x: x["cold_graph_capture"],
        "cold_graph_capture_and_full_512": lambda x: x["cold_graph_capture"] and x["full_budget"],
    }
    return {
        name: _scope_summary([item for item in observations if predicate(item)], bootstrap, seed,
                             leave_one_out and name == "all_pairs")
        for name, predicate in filters.items()
    }


def _repeat_consistency(runs: list[dict[str, Any]], bootstrap: int, seed: int) -> dict[str, Any]:
    by_case_method: dict[tuple[str, str, str, str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    case_runs: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    duplicate_counts = []
    for run in runs:
        groups, _ = _pair_rows(run)
        for key, methods in groups.items():
            case_runs[key].add(run["name"])
            for method, items in methods.items():
                for _, row in items:
                    by_case_method[(*key, method)].append((run["name"], row))
            for method, items in methods.items():
                if len(items) > 1:
                    duplicate_counts.append({"run": run["name"], "case": _case_label(key), "method": method,
                                             "duplicate_row_count": len(items)})
    method_summary = {}
    mismatch_examples = []
    for method in sorted(EXPECTED_METHODS):
        case_keys = [key for key in by_case_method if key[-1] == method]
        repeat_counts = Counter(len(by_case_method[key]) for key in case_keys)
        output_equal_cases = 0
        semantic_equal_cases = 0
        context_equal_cases = 0
        complete_cases = 0
        for key in case_keys:
            entries = by_case_method[key]
            if len(entries) != EXPECTED_REPEATS:
                continue
            complete_cases += 1
            outputs = [_output_tokens(row) for _, row in entries]
            output_equal = all(value == outputs[0] for value in outputs[1:])
            contexts = [row.get("context_id") for _, row in entries]
            prompt_lengths = [row.get("prompt_tokens") for _, row in entries]
            context_equal = len(set(map(str, contexts))) == 1 and len(set(map(str, prompt_lengths))) == 1
            semantic = [_semantic_view(row) for _, row in entries]
            semantic_equal = all(value == semantic[0] for value in semantic[1:])
            output_equal_cases += int(output_equal)
            semantic_equal_cases += int(semantic_equal)
            context_equal_cases += int(context_equal)
            if (not output_equal or not semantic_equal or not context_equal) and len(mismatch_examples) < 100:
                mismatch_examples.append({
                    "method": method,
                    "case": _case_label(key[:-1]),
                    "repeat_runs": [name for name, _ in entries],
                    "output_equal_across_repeats": output_equal,
                    "semantic_counters_equal_across_repeats": semantic_equal,
                    "input_context_equal_across_repeats": context_equal,
                })
        method_summary[method] = {
            "case_count_with_any_record": len(case_keys),
            "cases_with_expected_repeat_count": complete_cases,
            "repeat_count_distribution": {str(k): v for k, v in sorted(repeat_counts.items())},
            "output_equal_across_repeats": output_equal_cases,
            "semantic_counters_equal_across_repeats": semantic_equal_cases,
            "input_context_equal_across_repeats": context_equal_cases,
        }
    distribution = Counter(len(names) for names in case_runs.values())
    return {
        "unique_case_count": len(case_runs),
        "case_repeat_distribution": {str(k): v for k, v in sorted(distribution.items())},
        "expected_repeats_per_case": EXPECTED_REPEATS,
        "method_consistency": method_summary,
        "mismatch_examples": mismatch_examples,
        "duplicate_rows": duplicate_counts,
    }


def _paired_contrast(left: list[dict[str, Any]], right: list[dict[str, Any]],
                     bootstrap: int, seed: int) -> dict[str, Any]:
    def source_logs(rows):
        grouped = defaultdict(lambda: defaultdict(list))
        for row in rows:
            ratio = row.get("tps_speedup_rescue_over_w1")
            if ratio is not None and ratio > 0.0:
                case = (str(row["prefix"]), str(row["seed"]), str(row["budget"]))
                grouped[row["source"]][case].append(math.log(float(ratio)))
        return {
            name: statistics.fmean(statistics.fmean(case_values) for case_values in cases.values())
            for name, cases in grouped.items() if cases
        }

    left_logs, right_logs = source_logs(left), source_logs(right)
    common = sorted(set(left_logs) & set(right_logs))
    deltas = {name: left_logs[name] - right_logs[name] for name in common}
    if not deltas:
        return {"source_count": 0, "ratio_of_speedups": None, "ci95": None, "per_source_ratio": {}}
    point = math.exp(statistics.fmean(deltas.values()))
    samples = []
    if bootstrap > 0 and len(common) > 1:
        rng = random.Random(seed)
        for _ in range(bootstrap):
            samples.append(math.exp(statistics.fmean(rng.choice(list(deltas.values())) for _ in common)))
    return {
        "source_count": len(common),
        "ratio_of_speedups": point,
        "ci95": _percentile_interval(samples),
        "bootstrap_samples": len(samples),
        "bootstrap_seed": seed,
        "bootstrap_unit": "paired source",
        "per_source_ratio": {name: math.exp(value) for name, value in deltas.items()},
    }


def analyze(runs: list[dict[str, Any]], bootstrap: int, seed: int) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    all_observations = []
    run_observations: dict[str, list[dict[str, Any]]] = {}
    observations_with_case = defaultdict(list)
    row_method_counts = Counter()
    seen_names = Counter(run["name"] for run in runs)
    if any(count > 1 for count in seen_names.values()):
        issues.append({"issue": "run_names_not_unique", "names": dict(seen_names)})
        used_names = set()
        for run in runs:
            original = run["name"]
            candidate = f"{original}__input{run['index'] + 1}"
            if original in used_names or seen_names[original] > 1:
                run["original_name"] = original
                run["name"] = candidate
            used_names.add(run["name"])

    for run in runs:
        groups, group_issues = _pair_rows(run)
        issues.extend(group_issues)
        run_obs = []
        plan_order = _expected_order(run)
        for key, pair in groups.items():
            for method, items in pair.items():
                row_method_counts[f"{run['name']}:{method}"] += len(items)
            if any(len(pair.get(name, [])) != 1 for name in EXPECTED_METHODS):
                issues.append({"run": run["name"], "case": _case_label(key),
                               "issue": "missing_or_duplicate_method_rows",
                               "method_counts": {name: len(pair.get(name, [])) for name in sorted(EXPECTED_METHODS)}})
                continue
            observation = _pair_observation(run, key, pair)
            if observation is None:
                continue
            observation["method_order_in_plan"] = plan_order
            actual_order = observation["pair_row_order"]
            expected_pair_order = _expected_pair_order(run, key)
            observation["expected_pair_order"] = expected_pair_order
            observation["method_order_matches_plan"] = expected_pair_order == actual_order if expected_pair_order else None
            if expected_pair_order is None:
                issues.append({"run": run["name"], "case": _case_label(key),
                               "issue": "prompt_order_not_available"})
            if expected_pair_order and actual_order != expected_pair_order:
                issues.append({"run": run["name"], "case": _case_label(key),
                               "issue": "record_order_differs_from_prompt_rotation",
                               "expected_order": expected_pair_order, "record_order": actual_order})
            run_obs.append(observation)
            observations_with_case[key].append(observation)
            all_observations.append(observation)
        run_observations[run["name"]] = run_obs

    # Completeness expectations are reported as findings; every usable pair remains in the analysis.
    run_reports = []
    all_sources = set()
    all_prefixes = set()
    all_seeds = set()
    all_budgets = set()
    per_run_orders = []
    process_ids = []
    for run in runs:
        rows = run["rows"]
        plan = run["plan"]
        pipeline = run["pipeline"]
        groups, _ = _pair_rows(run)
        sources = sorted({_source(row) for row in rows if _source(row) is not None})
        prefixes = sorted({int(row["prefix"]) for row in rows if row.get("prefix") is not None})
        seeds = sorted({int(row["seed"]) for row in rows if row.get("seed") is not None})
        budgets = sorted({int(row["budget"]) for row in rows if row.get("budget") is not None})
        all_sources.update(sources)
        all_prefixes.update(prefixes)
        all_seeds.update(seeds)
        all_budgets.update(budgets)
        order = _expected_order(run)
        per_run_orders.append(order)
        process_ids.append(run["process_values"])
        run_expectations = {
            "records": int(plan.get("expected_runs", 240)) if isinstance(plan.get("expected_runs", 240), (int, float)) else 240,
            "pairs": int(plan.get("prompt_count", 120)) if isinstance(plan.get("prompt_count", 120), (int, float)) else 120,
            "sources": len(plan.get("sources", [])) if isinstance(plan.get("sources"), list) else EXPECTED_SOURCES,
            "methods": list(plan.get("methods", [])),
            "budgets": plan.get("budgets"),
            "seeds": _planned_seeds(plan),
            "prefixes": plan.get("prefixes"),
        }
        run_issues = []
        if len(rows) != run_expectations["records"]:
            run_issues.append("record_count_differs_from_plan")
        if len(groups) != run_expectations["pairs"]:
            run_issues.append("case_count_differs_from_plan")
        if len(run_observations.get(run["name"], [])) != run_expectations["pairs"]:
            run_issues.append("complete_pair_count_differs_from_plan")
        if set(sources) != set(plan.get("sources", sources)):
            run_issues.append("actual_sources_differ_from_plan")
        if run_expectations["seeds"] is not None and seeds != run_expectations["seeds"]:
            run_issues.append("actual_seeds_differ_from_plan")
        if isinstance(run_expectations["prefixes"], list):
            planned_prefixes = sorted(int(value) for value in run_expectations["prefixes"])
            if prefixes != planned_prefixes:
                run_issues.append("actual_prefixes_differ_from_plan")
        if isinstance(run_expectations["budgets"], list):
            planned_budgets = sorted(int(value) for value in run_expectations["budgets"])
            if budgets != planned_budgets:
                run_issues.append("actual_budgets_differ_from_plan")
        if plan.get("methods") and set(str(x).lower() for x in plan["methods"]) != EXPECTED_METHODS:
            run_issues.append("method_set_differs_from_w1_rescue")
        plan_order = [str(item).lower() for item in plan.get("methods", [])] if isinstance(plan.get("methods"), list) else None
        wrapper_order = run.get("repeat_schedule", {}).get("method_order")
        if not isinstance(wrapper_order, list):
            wrapper_order = (run.get("suite_entry") or {}).get("method_order")
        if plan_order and isinstance(wrapper_order, list) and plan_order != [str(x).lower() for x in wrapper_order]:
            run_issues.append("wrapper_order_differs_from_plan")
        if run.get("repeat_schedule") and run["repeat_schedule"].get("status") != "complete":
            run_issues.append(f"wrapper_status_{run['repeat_schedule'].get('status')}")
        if run.get("suite_entry") and run["suite_entry"].get("status") != "complete":
            run_issues.append(f"suite_status_{run['suite_entry'].get('status')}")
        if run.get("repeat_schedule", {}).get("process_id") is not None and (run.get("suite_entry") or {}).get("process_id") is not None:
            if str(run["repeat_schedule"]["process_id"]) != str(run["suite_entry"]["process_id"]):
                run_issues.append("wrapper_and_suite_process_id_differ")
        if pipeline.get("status") not in (None, "complete", "analysis_complete"):
            run_issues.append(f"pipeline_status_{pipeline.get('status')}")
        completed = pipeline.get("completed_runs")
        if completed is not None and completed != len(rows):
            run_issues.append("pipeline_completed_count_differs_from_records")
        source_offset = next((source.get("source_offset") for source in
                              (run.get("repeat_schedule", {}), run.get("suite_entry") or {})
                              if source.get("source_offset") is not None), None)
        observed_source_order = list(dict.fromkeys(_source(row) for row in rows if _source(row) is not None))
        planned_sources = plan.get("sources")
        source_rotation_matches = None
        if source_offset is not None and isinstance(planned_sources, list) and planned_sources:
            offset = int(source_offset) % len(planned_sources)
            original_sources = sorted(planned_sources)
            expected_source_order = original_sources[offset:] + original_sources[:offset]
            source_rotation_matches = (observed_source_order == planned_sources == expected_source_order)
            if not source_rotation_matches:
                run_issues.append("observed_source_order_differs_from_offset")
        run_reports.append({
            "run": run["name"],
            "path": run["path"],
            "row_count": len(rows),
            "complete_pair_count": len(run_observations.get(run["name"], [])),
            "unique_case_count": len(groups),
            "sources": sources,
            "prefixes": prefixes,
            "seeds": seeds,
            "budgets": budgets,
            "plan_expectations": run_expectations,
            "pipeline_status": pipeline.get("status"),
            "pipeline_completed_runs": completed,
            "method_order": order,
            "repeat_index": run.get("repeat_index"),
            "source_offset": source_offset,
            "source_rotation_matches_offset": source_rotation_matches,
            "process_values": run["process_values"],
            "schedule_fields": run["schedule_fields"],
            "observed_source_order": observed_source_order,
            "issues": run_issues,
        })
        for issue in run_issues:
            issues.append({"run": run["name"], "issue": issue})

    case_repeat_counts = Counter(len(items) for items in observations_with_case.values())
    if len(runs) != EXPECTED_RUNS:
        issues.append({"issue": "run_count_differs_from_expected", "expected": EXPECTED_RUNS, "actual": len(runs)})
    if len(all_observations) != EXPECTED_RUNS * 120:
        issues.append({"issue": "paired_observation_count_differs_from_expected",
                       "expected": EXPECTED_RUNS * 120, "actual": len(all_observations)})
    if len(all_sources) != EXPECTED_SOURCES:
        issues.append({"issue": "source_count_differs_from_expected", "expected": EXPECTED_SOURCES,
                       "actual": len(all_sources)})
    if sorted(all_prefixes) != EXPECTED_PREFIXES:
        issues.append({"issue": "prefix_set_differs_from_expected", "expected": EXPECTED_PREFIXES,
                       "actual": sorted(all_prefixes)})
    if sorted(all_seeds) != EXPECTED_SEEDS:
        issues.append({"issue": "seed_set_differs_from_expected", "expected": EXPECTED_SEEDS,
                       "actual": sorted(all_seeds)})
    if sorted(all_budgets) != [EXPECTED_BUDGET]:
        issues.append({"issue": "budget_set_differs_from_expected", "expected": [EXPECTED_BUDGET],
                       "actual": sorted(all_budgets)})
    expected_case_count = EXPECTED_SOURCES * len(EXPECTED_PREFIXES) * len(EXPECTED_SEEDS)
    if len(observations_with_case) != expected_case_count:
        issues.append({"issue": "unique_case_count_differs_from_expected", "expected": expected_case_count,
                       "actual": len(observations_with_case)})
    if case_repeat_counts != Counter({EXPECTED_REPEATS: expected_case_count}):
        issues.append({"issue": "case_repeat_distribution_differs_from_expected",
                       "expected": {str(EXPECTED_REPEATS): expected_case_count},
                       "actual": {str(k): v for k, v in sorted(case_repeat_counts.items())}})

    if len(per_run_orders) == EXPECTED_RUNS and all(order for order in per_run_orders):
        for index in range(1, len(per_run_orders)):
            if per_run_orders[index] == per_run_orders[index - 1]:
                issues.append({"issue": "method_order_not_alternating", "run_index": index,
                               "previous": per_run_orders[index - 1], "current": per_run_orders[index]})
    actual_offsets = sorted({
        int(report["source_offset"])
        for report in run_reports if report.get("source_offset") is not None
    })
    if actual_offsets and actual_offsets != EXPECTED_SOURCE_OFFSETS:
        issues.append({"issue": "source_offset_set_differs_from_schedule",
                       "expected": EXPECTED_SOURCE_OFFSETS, "actual": actual_offsets})
    distinct_process_values = {}
    for key in ("pid", "process_id", "runner_pid", "worker_pid"):
        values = [entry.get(key) for entry in process_ids if key in entry]
        if values:
            distinct_process_values[key] = {"values": values, "distinct_count": len(set(map(str, values)))}
            if len(values) == EXPECTED_RUNS and len(set(map(str, values))) != EXPECTED_RUNS:
                issues.append({"issue": "process_identifier_reused_across_runs", "field": key,
                               "values": values})
    row_process_values = [entry.get("record_process_id") for entry in process_ids if entry.get("record_process_id")]
    if row_process_values:
        flattened_process_ids = [value for group in row_process_values for value in group]
        distinct_process_values["record_process_id"] = {
            "values_by_run": row_process_values,
            "distinct_count": len(set(flattened_process_ids)),
        }
        if any(len(set(group)) != 1 for group in row_process_values):
            issues.append({"issue": "multiple_record_process_ids_within_run", "values_by_run": row_process_values})
        if len(row_process_values) == EXPECTED_RUNS and len(set(flattened_process_ids)) != EXPECTED_RUNS:
            issues.append({"issue": "record_process_identifier_reused_across_runs", "values_by_run": row_process_values})
    any_process_metadata = any(entry for entry in process_ids)
    if not any_process_metadata:
        issues.append({"issue": "process_identifier_metadata_missing"})
    elif len(process_ids) != EXPECTED_RUNS or any(not entry for entry in process_ids):
        issues.append({"issue": "process_identifier_missing_from_run"})
    suite_states = {}
    for run in runs:
        suite = run.get("suite") or {}
        if suite:
            suite_states[str(Path(run["path"]).parent)] = {
                "status": suite.get("status"),
                "repeat_count": len(suite.get("repeats", [])) if isinstance(suite.get("repeats"), list) else None,
            }
            if suite.get("status") not in (None, "complete"):
                issues.append({"run": run["name"], "issue": f"suite_overall_status_{suite.get('status')}"})

    paired_parity = {
        "pair_count": len(all_observations),
        "output_equivalent_pairs": sum(item["output_equivalent"] for item in all_observations),
        "output_sequence_equal_pairs": sum(item["output_equal"] for item in all_observations),
        "semantic_counter_equal_pairs": sum(item["semantic_counters_equal"] for item in all_observations),
        "recovery_relation_matches": sum(item["recovery_delta_equals_hits"] is True for item in all_observations),
        "recovery_relation_mismatches": sum(item["recovery_delta_equals_hits"] is False for item in all_observations),
        "recovery_relation_unavailable": sum(item["recovery_delta_equals_hits"] is None for item in all_observations),
    }
    mismatches = [
        {
            "run": item["run"], "case": item["case"],
            "output_equal": item["output_equal"],
            "generated_count_equal": item["generated_count_equal"],
            "eos_equal": item["eos_equal"],
            "input_context_equal": item["input_context_equal"],
            "prompt_tokens_equal": item["prompt_tokens_equal"],
            "semantic_counters_equal": item["semantic_counters_equal"],
            "recovery_delta_equals_hits": item["recovery_delta_equals_hits"],
        }
        for item in all_observations
        if not item["output_equivalent"] or not item["semantic_counters_equal"]
        or item["recovery_delta_equals_hits"] is False
    ]

    run_scopes = {
        name: _scopes(items, bootstrap, seed, leave_one_out=True)
        for name, items in run_observations.items()
    }
    seed_scopes = {}
    for seed_value in sorted({item["seed"] for item in all_observations}):
        seed_scopes[seed_value] = _scopes(
            [item for item in all_observations if item["seed"] == seed_value],
            bootstrap, seed, leave_one_out=True
        )
    overall_scopes = _scopes(all_observations, bootstrap, seed, leave_one_out=True)
    actual_order_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    planned_order_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in all_observations:
        actual_order = item.get("pair_row_order")
        if actual_order:
            actual_order_groups["_then_".join(actual_order)].append(item)
        planned_order = item.get("method_order_in_plan")
        if planned_order:
            planned_order_groups["_then_".join(planned_order)].append(item)

    def summarize_order_groups(groups: dict[str, list[dict[str, Any]]]):
        scopes = {
            name: _scopes(items, bootstrap, seed, leave_one_out=True)
            for name, items in sorted(groups.items())
        }
        names = list(sorted(groups))
        contrast = None
        if len(names) >= 2:
            contrast = {
                "left_order": names[0],
                "right_order": names[1],
                "left_over_right_speedup": _paired_contrast(
                    groups[names[0]], groups[names[1]], bootstrap, seed
                ),
            }
        return scopes, contrast

    actual_order_scopes, actual_order_contrast = summarize_order_groups(actual_order_groups)
    planned_order_scopes, planned_order_contrast = summarize_order_groups(planned_order_groups)

    run_point_estimates = {
        name: scopes["all_pairs"]["tps_speedup_rescue_over_w1"]["source_equal_gm"]
        for name, scopes in run_scopes.items()
    }
    valid_run_points = [float(value) for value in run_point_estimates.values()
                        if isinstance(value, (int, float)) and value > 0.0]
    run_level_variation = {
        "run_point_estimates": run_point_estimates,
        "run_count_with_estimate": len(valid_run_points),
        "minimum": min(valid_run_points) if valid_run_points else None,
        "maximum": max(valid_run_points) if valid_run_points else None,
        "median": statistics.median(valid_run_points) if valid_run_points else None,
        "geometric_mean_of_run_points": (
            math.exp(statistics.fmean(math.log(value) for value in valid_run_points))
            if valid_run_points else None
        ),
        "note": "Descriptive variation across four run-level estimates; source bootstrap intervals are reported separately.",
    }

    repeats = _repeat_consistency(runs, bootstrap, seed)
    method_order_counts = Counter("_then_".join(order) for order in per_run_orders if order)
    return {
        "analysis": {
            "run_count": len(runs),
            "row_count": sum(len(run["rows"]) for run in runs),
            "paired_observation_count": len(all_observations),
            "unique_case_count": len(observations_with_case),
            "case_repeat_distribution": {str(k): v for k, v in sorted(case_repeat_counts.items())},
            "source_count": len(all_sources),
            "sources": sorted(all_sources),
            "prefixes": sorted(all_prefixes),
            "seeds": sorted(all_seeds),
            "budgets": sorted(all_budgets),
            "methods": sorted(EXPECTED_METHODS),
            "bootstrap": {"samples": bootstrap, "seed": seed, "unit": "source"},
            "aggregation": "average log ratios across repeats within each source-prefix-seed-budget case; average cases equally within each source; average sources equally",
            "wall_equivalence_rule": "wall-time comparison is treated as equal-work only when W1 and rescue output token sequences, generated counts, and EOS flags all match",
            "no_new_graph_rule": "neither paired row has runtime_created, positive graph_setup_ms, or a new_graph_events entry",
            "cold_graph_rule": "at least one paired row has positive graph_setup_ms or a new_graph_events entry",
        },
        "completeness": {
            "expected": {
                "runs": EXPECTED_RUNS,
                "rows_per_run": 240,
                "pairs_per_run": 120,
                "total_rows": EXPECTED_RUNS * 240,
                "total_paired_observations": EXPECTED_RUNS * 120,
                "sources": EXPECTED_SOURCES,
                "prefixes": EXPECTED_PREFIXES,
                "seeds": EXPECTED_SEEDS,
                "budget": EXPECTED_BUDGET,
                "repeats_per_case": EXPECTED_REPEATS,
                "unique_cases": expected_case_count,
            },
            "actual": {
                "runs": len(runs),
                "rows_per_run": {run["name"]: len(run["rows"]) for run in runs},
                "total_rows": sum(len(run["rows"]) for run in runs),
                "paired_observations": len(all_observations),
                "unique_cases": len(observations_with_case),
                "sources": len(all_sources),
                "prefixes": sorted(all_prefixes),
                "seeds": sorted(all_seeds),
                "budgets": sorted(all_budgets),
                "repeat_distribution": {str(k): v for k, v in sorted(case_repeat_counts.items())},
                "method_rows": dict(row_method_counts),
                "method_order_counts": dict(method_order_counts),
                "process_identifiers": distinct_process_values,
                "suite_files": suite_states,
            },
            "runs": run_reports,
            "issues": issues,
            "issue_count": len(issues),
        },
        "paired_parity": paired_parity,
        "pair_mismatches": mismatches,
        "same_case_repeat_consistency": repeats,
        "overall": overall_scopes,
        "by_run": run_scopes,
        "by_seed": seed_scopes,
        "by_method_order": actual_order_scopes,
        "method_order_effect": actual_order_contrast,
        "by_planned_method_order": planned_order_scopes,
        "planned_method_order_effect": planned_order_contrast,
        "run_level_variation": run_level_variation,
        "paired_observations": all_observations,
    }


def _fmt_metric(metric: dict[str, Any]) -> str:
    value = metric.get("source_equal_gm")
    interval = metric.get("ci95")
    if value is None:
        return "n/a"
    if interval:
        return f"{value:.5f} [{interval[0]:.5f}, {interval[1]:.5f}]"
    return f"{value:.5f}"


def render_markdown(result: dict[str, Any]) -> str:
    analysis = result["analysis"]
    completeness = result["completeness"]
    parity = result["paired_parity"]
    lines = [
        "# W1 / rescue 四批重复运行分析",
        "",
        "## 数据完整度",
        "",
        f"读取 {analysis['run_count']} 个运行目录、{analysis['row_count']} 条生成记录、{analysis['paired_observation_count']} 个 W1/rescue 配对；覆盖 {analysis['source_count']} 个来源、prefix {analysis['prefixes']}、seed {analysis['seeds']}，预算 {analysis['budgets']}。",
        f"唯一 case 数为 {analysis['unique_case_count']}，每个 case 的重复数分布为 `{json.dumps(analysis['case_repeat_distribution'], ensure_ascii=False)}`；预期为 120 个 case 各重复 4 次。数据差异项共 {completeness['issue_count']} 条。",
        "",
        "运行目录和来源调度信息：",
        "",
        "| 运行 | 行数 / 配对 | 计划顺序 | seed | 来源数 | 首次出现的来源片段 |",
        "|---|---:|---|---|---:|---|",
    ]
    for run in completeness["runs"]:
        order = " → ".join(run.get("method_order") or []) or "缺少顺序字段"
        source_head = ", ".join(run.get("observed_source_order", [])[:4])
        lines.append(
            f"| {run['run']} | {run['row_count']} / {run['complete_pair_count']} | {order} | {run['seeds']} | {len(run['sources'])} | {source_head} |"
        )
    lines.extend([
        "",
        "## 速度与等工作量对照",
        "",
        "先在 source×prefix×seed×budget case 内平均重复进程的 log 比，再对来源内 case 等权平均，最后对来源等权汇总；区间按来源重采样 10,000 次。大于 1 表示 rescue 更快。wall 时间等工作量倍率只纳入三项输出条件都相同的配对。",
        "",
        "| 范围 | 配对数 / 来源数 | TPS rescue/W1 | wall W1/rescue，仅同输出 | 输出相同 |",
        "|---|---:|---:|---:|---:|",
    ])
    for label in ("all_pairs", "both_full_512", "no_new_graphs", "full_512_and_no_new_graphs", "cold_graph_capture", "cold_graph_capture_and_full_512"):
        summary = result["overall"][label]
        tps = summary["tps_speedup_rescue_over_w1"]
        wall = summary["wall_speedup_equivalent_output_only"]
        lines.append(
            f"| {label} | {summary['pair_observation_count']} / {summary['source_count']} | {_fmt_metric(tps)} | {_fmt_metric(wall)} | {summary['output_equivalent_pairs']} |"
        )
    lines.extend(["", "### 按实际记录顺序、计划顺序与 seed", "", "| 分组 | 配对数 | TPS rescue/W1 | 同输出 wall W1/rescue |", "|---|---:|---:|---:|"])
    for run, scopes in result["by_run"].items():
        item = scopes["all_pairs"]
        lines.append(f"| run `{run}` | {item['pair_observation_count']} | {_fmt_metric(item['tps_speedup_rescue_over_w1'])} | {_fmt_metric(item['wall_speedup_equivalent_output_only'])} |")
    for seed_value, scopes in result["by_seed"].items():
        item = scopes["all_pairs"]
        lines.append(f"| seed `{seed_value}` | {item['pair_observation_count']} | {_fmt_metric(item['tps_speedup_rescue_over_w1'])} | {_fmt_metric(item['wall_speedup_equivalent_output_only'])} |")
    for order, scopes in result["by_method_order"].items():
        item = scopes["all_pairs"]
        lines.append(f"| 实际 `{order}` | {item['pair_observation_count']} | {_fmt_metric(item['tps_speedup_rescue_over_w1'])} | {_fmt_metric(item['wall_speedup_equivalent_output_only'])} |")
    for order, scopes in result["by_planned_method_order"].items():
        item = scopes["all_pairs"]
        lines.append(f"| 计划 `{order}` | {item['pair_observation_count']} | {_fmt_metric(item['tps_speedup_rescue_over_w1'])} | {_fmt_metric(item['wall_speedup_equivalent_output_only'])} |")
    variation = result["run_level_variation"]
    if variation["run_count_with_estimate"]:
        lines.extend(["", f"四个进程级倍率范围为 {variation['minimum']:.5f}–{variation['maximum']:.5f}，中位数 {variation['median']:.5f}；这是运行时段波动描述，与来源重采样区间分开。"])
    lines.extend(["", "## 输出、计数与计算量", "", f"W1/rescue 输出等工作量配对 {parity['output_equivalent_pairs']}/{parity['pair_count']}；语义计数相同 {parity['semantic_counter_equal_pairs']}/{parity['pair_count']}；恢复差额等于 rescue 命中 {parity['recovery_relation_matches']}/{parity['pair_count']}，差异 {parity['recovery_relation_mismatches']}。跨重复输出和语义计数相同情况见 JSON 的 `same_case_repeat_consistency`。", ""])
    overall_costs = result["overall"]["all_pairs"]["cost_deltas"]
    for key in ("target_tree_nodes", "target_input_tokens", "target_calls", "recovery_calls_saved", "position_backup_nodes", "position_backup_rejections", "position_rescue_hits"):
        value = overall_costs.get(key, {})
        total = value.get("total")
        mean_value = value.get("mean_per_pair")
        lines.append(f"- `{key}`：总差额 {total if total is not None else 'n/a'}；每配对均值 {mean_value if mean_value is not None else 'n/a'}。")
    overall_metric = result["overall"]["all_pairs"]["tps_speedup_rescue_over_w1"]
    signs = overall_metric.get("source_signs", {})
    loo = overall_metric.get("leave_one_source_out")
    lines.extend(["", "## 来源分布与影响范围", ""])
    if loo:
        lines.append(f"全量来源中 rescue/W1 更快 {signs.get('positive_count', 0)} 个、更慢 {signs.get('negative_count', 0)} 个、相同 {signs.get('equal_count', 0)} 个。来源留一重算范围为 {loo['minimum']:.5f}（省略 `{loo['minimum_omitted_source']}`）到 {loo['maximum']:.5f}（省略 `{loo['maximum_omitted_source']}`）。")
    else:
        lines.append("当前范围没有可用于来源留一重算的有效来源倍率。")
    lines.append("")
    contrast = result.get("method_order_effect")
    if contrast:
        effect = contrast["left_over_right_speedup"]
        ci = effect.get("ci95")
        ci_text = f" [{ci[0]:.5f}, {ci[1]:.5f}]" if ci else ""
        ratio = effect.get("ratio_of_speedups")
        ratio_text = f"{ratio:.5f}" if isinstance(ratio, (int, float)) else "n/a"
        lines.append(f"实际记录顺序的来源配对倍率（`{contrast['left_order']}` 相对 `{contrast['right_order']}`）为 {ratio_text}{ci_text}，来源数 {effect['source_count']}。")
    planned_contrast = result.get("planned_method_order_effect")
    if planned_contrast:
        effect = planned_contrast["left_over_right_speedup"]
        ci = effect.get("ci95")
        ci_text = f" [{ci[0]:.5f}, {ci[1]:.5f}]" if ci else ""
        ratio = effect.get("ratio_of_speedups")
        ratio_text = f"{ratio:.5f}" if isinstance(ratio, (int, float)) else "n/a"
        lines.append(f"计划运行顺序的来源配对倍率（`{planned_contrast['left_order']}` 相对 `{planned_contrast['right_order']}`）为 {ratio_text}{ci_text}，来源数 {effect['source_count']}。")
    if completeness["issues"]:
        lines.extend(["", "## 数据差异项", "", "差异项保留在 JSON 的 `completeness.issues`；成对不一致保留在 `pair_mismatches`，没有从逐条记录中删除。"])
    lines.extend([
        "",
        "## 计算口径与限制",
        "",
        "主比较先在每个 source×prefix×seed×budget case 内平均重复进程的 log 速度比，再对来源内 case 等权平均，最后对来源等权。bootstrap 抽样单位是来源，不是 480 个配对。no_new_graphs 按两臂均无运行时新建标记、正 graph_setup_ms 或新图事件定义；cold_graph_capture 仅按实际图准备耗时或新图事件定义。",
        "",
        "wall 时间倍率只在 W1 与 rescue 的 token 序列、生成数、EOS 标记都一致时作为等工作量对照；TPS 比仍对所有可用成对记录保留。输出不同、计数不同、恢复差额不符和重复间变化都会单独列出。",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="four completed run directories or records.jsonl paths")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for JSON and Markdown results")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if args.bootstrap < 0:
        parser.error("--bootstrap must be non-negative")
    runs = [_read_run(path, index) for index, path in enumerate(args.runs)]
    result = analyze(runs, args.bootstrap, args.seed)
    output_dir = args.output_dir.expanduser().resolve()
    json_path = output_dir / "repeat_analysis.json"
    markdown_path = output_dir / "REPEAT_ANALYSIS.md"
    if json_path.exists() or markdown_path.exists():
        parser.error(f"refusing to replace existing output: {json_path} or {markdown_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(json_path)
    print(markdown_path)
    print(json.dumps({
        "runs": result["analysis"]["run_count"],
        "rows": result["analysis"]["row_count"],
        "pairs": result["analysis"]["paired_observation_count"],
        "unique_cases": result["analysis"]["unique_case_count"],
        "issue_count": result["completeness"]["issue_count"],
        "overall_tps_ratio": result["overall"]["all_pairs"]["tps_speedup_rescue_over_w1"]["source_equal_gm"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
