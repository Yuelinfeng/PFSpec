#!/usr/bin/env python3
"""Independently audit and summarize proposal-side graph runs from raw JSON."""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import sys


BOOTSTRAP_SEED = 402101
BOOTSTRAP_DRAWS = 10000
DEFAULT_POSITION_VOCAB = 8000
DEFAULT_FEATURE_OFFSET = 8000
EXPECTED_EXECUTIONS = {"target_only", "proposal_graph"}


class InputDataError(RuntimeError):
    pass


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise InputDataError(f"Missing input file: {path}") from error
    except json.JSONDecodeError as error:
        raise InputDataError(f"Invalid JSON at {path}:{error.lineno}:{error.colno}") from error


def read_records(path: Path):
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise InputDataError(
                        f"Invalid record at {path}:{line_no}:{error.colno}"
                    ) from error
                if not isinstance(row, dict):
                    raise InputDataError(f"Record line {line_no} is not an object")
                rows.append(row)
    except FileNotFoundError as error:
        raise InputDataError(f"Missing input file: {path}") from error
    return rows


def finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def close_number(left, right, *, rel_tol=1e-10, abs_tol=1e-10) -> bool:
    return (
        finite_number(left)
        and finite_number(right)
        and math.isclose(float(left), float(right), rel_tol=rel_tol, abs_tol=abs_tol)
    )


def quantile(values, probability):
    ordered = sorted(values)
    point = (len(ordered) - 1) * probability
    low = int(point)
    high = min(low + 1, len(ordered) - 1)
    if high == low:
        return ordered[low]
    return ordered[low] * (high - point) + ordered[high] * (point - low)


def validate_sequence(prompt, output, *, eos_id, position_vocab, feature_offset, ranks):
    errors = []
    if not isinstance(prompt, list) or not prompt or prompt[0] != eos_id:
        return ["prompt BOS sentinel differs from eos_id"]
    if not isinstance(output, list) or not output:
        return ["output is empty or is not a list"]
    if any(type(token) is not int for token in prompt + output):
        return ["prompt or output contains a non-integer token"]

    prefix = prompt[1:]
    if eos_id in prefix:
        errors.append("prompt contains EOS after its BOS sentinel")
    if len(prefix) % 2:
        errors.append("prompt does not end on a complete P/F pair")
    body = prefix + output
    if eos_id in body:
        if body[-1] != eos_id or body.count(eos_id) != 1:
            errors.append("EOS is duplicated or is not the final generated token")
        else:
            body = body[:-1]
    if len(body) % 2:
        errors.append("P/F body does not contain complete pairs")

    previous_rank = None
    for index, token in enumerate(body):
        if index % 2 == 0:
            if not 0 <= token < position_vocab:
                errors.append(f"Position token out of range at body index {index}: {token}")
            elif ranks is not None:
                rank = ranks[token]
                if previous_rank is not None and rank <= previous_rank:
                    errors.append(f"Position rank is not increasing at body index {index}")
                previous_rank = rank
        elif not feature_offset <= token < eos_id:
            errors.append(f"Feature token out of range at body index {index}: {token}")
    return errors


def planned_arms(methods, executions):
    arms = set()
    for method in methods:
        method_executions = ["target_only"] if method == "target" else executions
        arms.update(f"{execution}:{method}" for execution in method_executions)
    return arms


def expected_run_count(methods, executions, prompt_count):
    per_prompt = sum(1 if method == "target" else len(executions) for method in methods)
    return per_prompt * prompt_count


def summarize_arm(rows):
    tokens = sum(int(row["generated_tokens"]) for row in rows)
    seconds = math.fsum(float(row["outer_wall_seconds"]) for row in rows)
    lengths = Counter(str(int(row["generated_tokens"])) for row in rows)
    counter_totals = Counter()
    for row in rows:
        for key, value in (row.get("counters") or {}).items():
            if isinstance(value, int) and not isinstance(value, bool):
                counter_totals[key] += value
    return {
        "runs": len(rows),
        "generated_tokens": tokens,
        "outer_wall_seconds": seconds,
        "pooled_tokens_per_second": tokens / seconds if seconds else None,
        "eos_count": sum(bool(row.get("eos")) for row in rows),
        "output_lengths": dict(sorted(lengths.items(), key=lambda item: int(item[0]))),
        "integer_counter_totals": dict(sorted(counter_totals.items())),
    }


def ratio_summary(rows_by_pair, numerator, denominator, budget, *, pair_ids=None,
                  full_only=False):
    eligible = None if pair_ids is None else set(pair_ids)
    source_logs = defaultdict(list)
    pair_count = 0
    for pair_id, arms in rows_by_pair.items():
        if eligible is not None and pair_id not in eligible:
            continue
        if numerator not in arms or denominator not in arms:
            continue
        up, down = arms[numerator], arms[denominator]
        if full_only and (
            up["generated_tokens"] != budget or down["generated_tokens"] != budget
        ):
            continue
        if min(up["generated_tokens"], down["generated_tokens"]) <= 0:
            continue
        up_rate = up["generated_tokens"] / up["outer_wall_seconds"]
        down_rate = down["generated_tokens"] / down["outer_wall_seconds"]
        source_logs[str(up["source"])].append(math.log(up_rate) - math.log(down_rate))
        pair_count += 1

    source_means = {
        source: math.fsum(values) / len(values)
        for source, values in source_logs.items()
    }
    source_names = sorted(source_means)
    result = {
        "numerator": numerator,
        "denominator": denominator,
        "pair_count": pair_count,
        "source_count": len(source_names),
        "source_equal_geometric_mean": None,
        "source_bootstrap_95_ci": None,
        "per_source": {},
    }
    if not source_names:
        return result

    result["source_equal_geometric_mean"] = math.exp(
        math.fsum(source_means.values()) / len(source_names)
    )
    result["per_source"] = {
        source: math.exp(source_means[source]) for source in source_names
    }
    result["source_bootstrap_seed"] = BOOTSTRAP_SEED
    result["source_bootstrap_replicates"] = BOOTSTRAP_DRAWS
    if len(source_names) > 1:
        rng = random.Random(BOOTSTRAP_SEED)
        draws = []
        for _ in range(BOOTSTRAP_DRAWS):
            sampled = [source_means[rng.choice(source_names)] for _ in source_names]
            draws.append(math.exp(math.fsum(sampled) / len(source_names)))
        result["source_bootstrap_95_ci"] = [
            quantile(draws, 0.025), quantile(draws, 0.975)
        ]
    return result


def summarize_ratio_pair(rows_by_pair, numerator, denominator, budget, *, pair_ids=None):
    return {
        "all_output_lengths": ratio_summary(
            rows_by_pair, numerator, denominator, budget, pair_ids=pair_ids
        ),
        "common_full_budget": ratio_summary(
            rows_by_pair, numerator, denominator, budget,
            pair_ids=pair_ids, full_only=True
        ),
    }


def graph_call_keys(calls):
    keys = []
    for key, value in calls.items():
        name = str(key).lower()
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
            and ("graph" in name or name in {"replay", "cuda_replay", "capture_replay"})
        ):
            keys.append(str(key))
    return sorted(keys)


def _runtime_has_no_new_graphs(row, delta_key):
    delta = row.get(delta_key)
    if not isinstance(delta, dict):
        return False
    events = delta.get("new_graph_events")
    before = delta.get("graph_count_before")
    after = delta.get("graph_count_after")
    return (
        isinstance(events, list)
        and not events
        and type(before) is int
        and type(after) is int
        and before == after
    )


def stable_runtime_reuse_pair_ids(method, pair_ids, rows_by_pair):
    """Select rows after all three runtimes exist and no Target/proposal graph was added."""
    selected = []
    for pair_id in sorted(pair_ids):
        arms = rows_by_pair.get(pair_id, {})
        proposal = arms.get(f"proposal_graph:{method}")
        method_target = arms.get(f"target_only:{method}")
        common_target = arms.get("target_only:target")
        if proposal is None or method_target is None or common_target is None:
            continue
        if any(row.get("runtime_created") is not False
               for row in (proposal, method_target, common_target)):
            continue
        if not _runtime_has_no_new_graphs(proposal, "proposal_runtime_delta"):
            continue
        if not _runtime_has_no_new_graphs(proposal, "runtime_delta"):
            continue
        if not _runtime_has_no_new_graphs(method_target, "runtime_delta"):
            continue
        if not _runtime_has_no_new_graphs(common_target, "runtime_delta"):
            continue
        proposal_delta = proposal.get("proposal_runtime_delta") or {}
        if not graph_call_keys(proposal_delta.get("calls") or {}):
            continue
        selected.append(pair_id)
    return selected


def summarize_event_cost(events):
    keys = ("prepare_ms", "warmup_ms", "capture_ms")
    totals = {key: 0.0 for key in keys}
    counts = {key: 0 for key in keys}
    for event in events:
        if not isinstance(event, dict):
            continue
        for key in keys:
            value = event.get(key)
            if finite_number(value):
                totals[key] += float(value)
                counts[key] += 1
    return {
        key: {"sum_ms": totals[key], "event_values": counts[key]}
        for key in keys
    }


_VOLATILE_RUNTIME_KEY_PARTS = (
    "time", "duration", "elapsed", "address", "pointer", "_id", "id_",
)
_RUNTIME_OBSERVATION_KEYS = {
    "cache",
    "memory_allocated_before", "memory_allocated_after", "memory_allocated_delta",
    "memory_reserved_before", "memory_reserved_after", "memory_reserved_delta",
}


def normalize_runtime_event(value):
    """Keep Target runtime semantics while dropping timing and process-local observations."""
    if isinstance(value, list):
        return [normalize_runtime_event(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_runtime_event(item) for item in value]
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            name = str(key)
            lower = name.lower()
            if name in _RUNTIME_OBSERVATION_KEYS or lower.endswith("_ms"):
                continue
            if any(part in lower for part in _VOLATILE_RUNTIME_KEY_PARTS):
                continue
            normalized[name] = normalize_runtime_event(item)
        return {key: normalized[key] for key in sorted(normalized)}
    return value


def normalize_disabled_graphs(value):
    """Remove cache identity from DynamicTargetGraph's repr(tuple) graph keys."""
    if not isinstance(value, dict):
        return {"invalid_type": type(value).__name__}
    entries = []
    for key, reason in value.items():
        normalized_key = str(key)
        try:
            parsed = ast.literal_eval(normalized_key)
            if isinstance(parsed, tuple) and len(parsed) >= 2:
                normalized_key = repr(parsed[1:])
        except (SyntaxError, ValueError):
            pass
        entries.append((normalized_key, normalize_runtime_event(reason)))
    return [
        {"graph_shape": key, "reason": reason}
        for key, reason in sorted(entries, key=lambda item: item[0])
    ]


def normalized_counter(value, *, field_name):
    if not isinstance(value, dict):
        raise InputDataError(f"{field_name} is not an object")
    result = {}
    for key, count in value.items():
        if type(count) is not int or count < 0:
            raise InputDataError(f"{field_name}.{key} is not a nonnegative integer")
        if count:
            result[str(key)] = count
    return dict(sorted(result.items()))


def aggregate_target_runtime_rows(arm, rows, final_runtime):
    """Rebuild one dynamic Target runtime from its per-decode row deltas."""
    errors = []
    calls = Counter()
    fallbacks = Counter()
    graph_events = []
    cache_events = []
    previous_graph_count = 0
    runtime_created_rows = []

    for row in rows:
        pair_id = str(row.get("pair_id"))
        if row.get("runtime_created") is True:
            runtime_created_rows.append(pair_id)
        delta = row.get("runtime_delta")
        if not isinstance(delta, dict):
            errors.append(f"{arm}/{pair_id}: Target runtime delta is absent")
            continue
        try:
            row_calls = normalized_counter(delta.get("calls", {}), field_name="calls")
            row_fallbacks = normalized_counter(delta.get("fallbacks", {}), field_name="fallbacks")
        except InputDataError as error:
            errors.append(f"{arm}/{pair_id}: {error}")
            continue
        calls.update(row_calls)
        fallbacks.update(row_fallbacks)

        before = delta.get("graph_count_before")
        after = delta.get("graph_count_after")
        events = delta.get("new_graph_events")
        new_cache_events = delta.get("new_cache_events")
        if type(before) is not int or type(after) is not int or before < 0 or after < before:
            errors.append(f"{arm}/{pair_id}: Target graph counts are invalid")
            continue
        if before != previous_graph_count:
            errors.append(
                f"{arm}/{pair_id}: graph_count_before={before}, expected {previous_graph_count}"
            )
        if not isinstance(events, list):
            errors.append(f"{arm}/{pair_id}: new_graph_events is not a list")
            events = []
        if after - before != len(events):
            errors.append(f"{arm}/{pair_id}: graph-count increase differs from new graph events")
        if not isinstance(new_cache_events, list):
            errors.append(f"{arm}/{pair_id}: new_cache_events is not a list")
            new_cache_events = []
        graph_events.extend(events)
        cache_events.extend(new_cache_events)
        previous_graph_count = after

    if rows and runtime_created_rows != [str(rows[0].get("pair_id"))]:
        errors.append(
            f"{arm}: runtime_created rows are {runtime_created_rows}, expected only first row"
        )

    rebuilt = {
        "calls": dict(sorted(calls.items())),
        "fallbacks": dict(sorted(fallbacks.items())),
        "graph_count": previous_graph_count,
        "graph_events": normalize_runtime_event(graph_events),
        "cache_events": normalize_runtime_event(cache_events),
    }
    if not isinstance(final_runtime, dict):
        errors.append(f"runtime_summary.dynamic_runtimes lacks {arm}")
        return rebuilt, errors

    try:
        final_calls = normalized_counter(final_runtime.get("calls", {}), field_name="calls")
        final_fallbacks = normalized_counter(final_runtime.get("fallbacks", {}), field_name="fallbacks")
    except InputDataError as error:
        errors.append(f"{arm}: {error}")
        final_calls, final_fallbacks = {}, {}
    final_graph_events = final_runtime.get("graph_events")
    final_cache_events = final_runtime.get("cache_events")
    if not isinstance(final_graph_events, list):
        errors.append(f"runtime_summary.dynamic_runtimes.{arm}.graph_events is not a list")
        final_graph_events = []
    if not isinstance(final_cache_events, list):
        errors.append(f"runtime_summary.dynamic_runtimes.{arm}.cache_events is not a list")
        final_cache_events = []
    try:
        final_graph_count = int(final_runtime.get("graph_count"))
    except (TypeError, ValueError):
        final_graph_count = -1
    final = {
        "calls": final_calls,
        "fallbacks": final_fallbacks,
        "graph_count": final_graph_count,
        "graph_events": normalize_runtime_event(final_graph_events),
        "cache_events": normalize_runtime_event(final_cache_events),
    }
    for key in rebuilt:
        if rebuilt[key] != final[key]:
            errors.append(f"{arm}: row deltas differ from runtime_summary.{key}")
    if final_graph_count != len(final_graph_events):
        errors.append(f"{arm}: graph_count differs from runtime_summary graph event count")

    rebuilt["disabled_graphs"] = normalize_disabled_graphs(
        final_runtime.get("disabled_graphs", {})
    )
    rebuilt["cache_bytes_by_capacity"] = normalize_runtime_event(
        final_runtime.get("cache_bytes_by_capacity", {})
    )

    return rebuilt, errors


def target_runtime_delta_shape(delta):
    """Comparable per-decode Target fields; event clocks and allocator snapshots are omitted."""
    if not isinstance(delta, dict):
        return None
    return {
        "calls": normalized_counter(delta.get("calls", {}), field_name="calls"),
        "fallbacks": normalized_counter(delta.get("fallbacks", {}), field_name="fallbacks"),
        "graph_count_before": delta.get("graph_count_before"),
        "graph_count_after": delta.get("graph_count_after"),
        "new_graph_events": normalize_runtime_event(delta.get("new_graph_events", [])),
        "new_cache_events": normalize_runtime_event(delta.get("new_cache_events", [])),
    }


def first_output_difference(old, new):
    left = old.get("output_tokens")
    right = new.get("output_tokens")
    if not isinstance(left, list) or not isinstance(right, list):
        return {"index": None, "old_token": None, "new_token": None}
    for index, (old_token, new_token) in enumerate(zip(left, right)):
        if old_token != new_token:
            return {"index": index, "old_token": old_token, "new_token": new_token}
    if len(left) != len(right):
        index = min(len(left), len(right))
        return {
            "index": index,
            "old_token": left[index] if index < len(left) else None,
            "new_token": right[index] if index < len(right) else None,
        }
    return {"index": None, "old_token": None, "new_token": None}


def counter_difference(old, new):
    left = old.get("counters") or {}
    right = new.get("counters") or {}
    keys = sorted(set(left) | set(right))
    differences = []
    for key in keys:
        if left.get(key) != right.get(key):
            old_value, new_value = left.get(key), right.get(key)
            entry = {"field": key}
            if isinstance(old_value, (int, float, str, bool)) or old_value is None:
                entry["target_only"] = old_value
            else:
                entry["target_only_type"] = type(old_value).__name__
                entry["target_only_length"] = len(old_value) if hasattr(old_value, "__len__") else None
            if isinstance(new_value, (int, float, str, bool)) or new_value is None:
                entry["proposal_graph"] = new_value
            else:
                entry["proposal_graph_type"] = type(new_value).__name__
                entry["proposal_graph_length"] = len(new_value) if hasattr(new_value, "__len__") else None
            differences.append(entry)
    return differences


def load_inputs(run_dir: Path):
    plan = read_json(run_dir / "plan.json")
    prompts_doc = read_json(run_dir / "prompts.json")
    rows = read_records(run_dir / "records.jsonl")
    pipeline = read_json(run_dir / "pipeline.json")
    runtime_summary_path = run_dir / "runtime_summary.json"
    runtime_summary = read_json(runtime_summary_path) if runtime_summary_path.exists() else None
    return plan, prompts_doc, rows, pipeline, runtime_summary


def analyze(run_dir: Path, requested_budget=None):
    plan, prompts_doc, rows, pipeline, runtime_summary = load_inputs(run_dir)
    prompts = prompts_doc.get("contexts") if isinstance(prompts_doc, dict) else prompts_doc
    if not isinstance(prompts, list):
        raise InputDataError("prompts.json lacks a contexts list")
    methods = plan.get("methods")
    executions = plan.get("executions")
    if not isinstance(methods, list) or not methods:
        raise InputDataError("plan.json lacks methods")
    if not isinstance(executions, list) or set(executions) != EXPECTED_EXECUTIONS:
        raise InputDataError("plan.executions must contain target_only and proposal_graph")
    if len(set(methods)) != len(methods) or "target" not in methods:
        raise InputDataError("plan.methods must be unique and include target")
    if pipeline.get("status") != "complete":
        raise InputDataError("pipeline.json status is not complete")
    if runtime_summary is not None and runtime_summary.get("status") != "complete":
        raise InputDataError("runtime_summary.json status is not complete")

    budget = int(plan.get("budget", 0))
    if requested_budget is not None and int(requested_budget) != budget:
        raise InputDataError(
            f"--budget {requested_budget} differs from plan budget {budget}"
        )
    if budget < 2 or budget % 2:
        raise InputDataError("plan.budget must be a positive even integer")
    prompt_count = len(prompts)
    expected_runs = expected_run_count(methods, executions, prompt_count)
    if int(plan.get("prompt_count", -1)) != prompt_count:
        raise InputDataError("plan.prompt_count differs from prompts.json")
    if int(plan.get("expected_runs", -1)) != expected_runs:
        raise InputDataError("plan.expected_runs differs from planned combinations")
    if pipeline.get("completed_runs") != expected_runs or len(rows) != expected_runs:
        raise InputDataError("pipeline or records count differs from planned combinations")
    if runtime_summary is not None and runtime_summary.get("completed_runs", expected_runs) != expected_runs:
        raise InputDataError("runtime_summary.completed_runs differs from planned combinations")

    prompt_eos = prompts_doc.get("eos_id", -1) if isinstance(prompts_doc, dict) else -1
    eos_id = int(plan.get("eos_id", prompt_eos))
    if eos_id < 0:
        raise InputDataError("plan or prompts lacks eos_id")
    if isinstance(prompts_doc, dict) and "eos_id" in prompts_doc:
        if int(prompts_doc["eos_id"]) != eos_id:
            raise InputDataError("plan.eos_id differs from prompts.eos_id")
    position_vocab = int(
        plan.get("position_vocab_size", plan.get("p_vocab", DEFAULT_POSITION_VOCAB))
    )
    feature_offset = int(
        plan.get("feature_token_offset", plan.get("feature_offset", DEFAULT_FEATURE_OFFSET))
    )
    rank_values = plan.get("position_rank_map")
    if not isinstance(rank_values, list) or len(rank_values) < position_vocab:
        raise InputDataError("position_rank_map is absent or too short")
    if any(type(value) is not int for value in rank_values[:position_vocab]):
        raise InputDataError("position_rank_map contains non-integer values")
    ranks = rank_values[:position_vocab]

    prompt_by_id = {}
    prompt_errors = []
    for prompt in prompts:
        if not isinstance(prompt, dict):
            prompt_errors.append("prompt entry is not an object")
            continue
        pair_id = str(prompt.get("pair_id", ""))
        if not pair_id or pair_id in prompt_by_id:
            prompt_errors.append(f"pair_id is empty or repeated: {pair_id!r}")
            continue
        tokens = prompt.get("prompt")
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            prompt_errors.append(f"prompt token list is invalid for {pair_id}")
            continue
        prompt_by_id[pair_id] = prompt
        prompt_errors.extend(
            f"{pair_id}: {message}"
            for message in validate_sequence(
                tokens, [], eos_id=eos_id, position_vocab=position_vocab,
                feature_offset=feature_offset, ranks=ranks
            )
            if message != "output is empty or is not a list"
        )

    expected_sources = {str(value) for value in plan.get("sources", [])}
    prompt_sources = {str(prompt.get("source")) for prompt in prompt_by_id.values()}
    source_match = not expected_sources or expected_sources == prompt_sources
    rows_by_pair = defaultdict(dict)
    duplicate_arms = []
    unplanned_rows = []
    row_errors = []
    for index, row in enumerate(rows):
        pair_id = str(row.get("pair_id", ""))
        method = row.get("method")
        execution = row.get("execution")
        if pair_id not in prompt_by_id or method not in methods:
            unplanned_rows.append({"index": index, "pair_id": pair_id, "method": method})
            continue
        allowed_executions = ["target_only"] if method == "target" else executions
        if execution not in allowed_executions:
            unplanned_rows.append({"index": index, "pair_id": pair_id,
                                   "method": method, "execution": execution})
            continue
        expected_arm = f"{execution}:{method}"
        if row.get("arm") != expected_arm:
            row_errors.append(
                f"record {index}: arm={row.get('arm')!r}, expected {expected_arm!r}"
            )
        if expected_arm in rows_by_pair[pair_id]:
            duplicate_arms.append({"pair_id": pair_id, "arm": expected_arm})
        rows_by_pair[pair_id][expected_arm] = row
        prompt = prompt_by_id[pair_id]
        for key in ("source", "prefix", "seed"):
            if row.get(key) != prompt.get(key):
                row_errors.append(f"record {index}: {key} differs from prompt {pair_id}")

        output = row.get("output_tokens")
        if not isinstance(output, list) or not output or any(type(token) is not int for token in output):
            row_errors.append(f"record {index}: output_tokens is invalid")
            output = [] if not isinstance(output, list) else output
        generated = row.get("generated_tokens")
        seconds = row.get("outer_wall_seconds")
        if type(generated) is not int or generated != len(output):
            row_errors.append(f"record {index}: generated_tokens differs from output length")
        if len(output) > budget:
            row_errors.append(f"record {index}: output exceeds plan budget")
        if not finite_number(seconds) or seconds <= 0:
            row_errors.append(f"record {index}: outer_wall_seconds must be finite and positive")
        if output and finite_number(seconds) and seconds > 0:
            if not close_number(row.get("tokens_per_second"), len(output) / seconds):
                row_errors.append(f"record {index}: tokens_per_second differs from raw length/time")
        output_has_eos = bool(output and output[-1] == eos_id)
        if type(row.get("eos")) is not bool or row.get("eos") != output_has_eos:
            row_errors.append(f"record {index}: eos flag differs from output")
        if row.get("sequence_validated") is not True:
            row_errors.append(f"record {index}: sequence_validated is not true")
        if output:
            row_errors.extend(
                f"record {index}: {message}"
                for message in validate_sequence(
                    prompt["prompt"], output, eos_id=eos_id,
                    position_vocab=position_vocab, feature_offset=feature_offset,
                    ranks=ranks
                )
            )
        if not isinstance(row.get("counters"), dict):
            row_errors.append(f"record {index}: counters is missing or invalid")
        if not isinstance(row.get("runtime_delta"), dict):
            row_errors.append(f"record {index}: Target runtime delta is missing or invalid")
        if execution == "proposal_graph" and not isinstance(row.get("proposal_runtime_delta"), dict):
            row_errors.append(f"record {index}: proposal_runtime_delta is missing or invalid")
        if execution == "target_only" and row.get("proposal_runtime_delta") is not None:
            row_errors.append(f"record {index}: target_only has proposal_runtime_delta")

    expected_pair_ids = set(prompt_by_id)
    expected_arms = planned_arms(methods, executions)
    completeness_errors = []
    for pair_id in expected_pair_ids:
        present = set(rows_by_pair.get(pair_id, {}))
        if present != expected_arms:
            completeness_errors.append(
                f"{pair_id}: missing={sorted(expected_arms-present)} extra={sorted(present-expected_arms)}"
            )

    parity_by_method = {}
    parity_errors = []
    candidate_methods = [method for method in methods if method != "target"]
    for method in candidate_methods:
        pairs = []
        for pair_id in sorted(expected_pair_ids):
            arms = rows_by_pair.get(pair_id, {})
            eager = arms.get(f"target_only:{method}")
            graphed = arms.get(f"proposal_graph:{method}")
            if eager is None or graphed is None:
                continue
            output_equal = eager.get("output_tokens") == graphed.get("output_tokens")
            counters_equal = eager.get("counters") == graphed.get("counters")
            entry = {
                "pair_id": pair_id,
                "source": eager.get("source"),
                "output_equal": output_equal,
                "counters_equal": counters_equal,
                "output_lengths": [
                    eager.get("generated_tokens"), graphed.get("generated_tokens")
                ],
            }
            if not output_equal:
                entry["first_output_difference"] = first_output_difference(eager, graphed)
            if not counters_equal:
                entry["counter_difference_fields"] = counter_difference(eager, graphed)
            pairs.append(entry)
            if not output_equal or not counters_equal:
                parity_errors.append(f"{method}/{pair_id}: output or counters differ")
        parity_by_method[method] = {
            "pair_count": len(pairs),
            "output_equal_count": sum(item["output_equal"] for item in pairs),
            "counters_equal_count": sum(item["counters_equal"] for item in pairs),
            "pairs": pairs,
        }

    rows_by_arm = defaultdict(list)
    for row in rows:
        rows_by_arm[row.get("arm")].append(row)
    arm_summaries = {
        arm: summarize_arm(chosen)
        for arm, chosen in sorted(rows_by_arm.items())
    }

    contrast_summaries = {}
    ratio_errors = []
    for method in candidate_methods:
        comparisons = {
            "target_only_over_common_target": (
                f"target_only:{method}", "target_only:target"
            ),
            "proposal_over_method_target_only": (
                f"proposal_graph:{method}", f"target_only:{method}"
            ),
            "proposal_over_common_target": (
                f"proposal_graph:{method}", "target_only:target"
            ),
        }
        contrast_summaries[method] = {}
        for label, (numerator, denominator) in comparisons.items():
            if not all(any(row.get("arm") == arm for row in rows) for arm in (numerator, denominator)):
                ratio_errors.append(f"{method}: missing arm for {label}")
            value = summarize_ratio_pair(rows_by_pair, numerator, denominator, budget)
            contrast_summaries[method][label] = value
            if value["all_output_lengths"]["pair_count"] == 0:
                ratio_errors.append(f"{method}: {label} has no paired rows")
            if value["all_output_lengths"]["source_equal_geometric_mean"] is None:
                ratio_errors.append(f"{method}: {label} has no ratio result")

    graph_errors = []
    graph_summaries = {}
    timing_scope = str(plan.get("timing_scope", ""))
    timing_lower = timing_scope.lower()
    graph_setup_in_outer_timing = (
        "graph creation" in timing_lower
        and ("outer wall" in timing_lower or "outer time" in timing_lower)
    )
    for method in candidate_methods:
        arm = f"proposal_graph:{method}"
        chosen = rows_by_arm.get(arm, [])
        calls = Counter()
        fallbacks = Counter()
        graph_events = []
        graph_rows = []
        runtime_created_pairs = []
        first_graph_pair = None
        graph_used_pair_ids = []
        new_graph_pair_ids = []
        no_new_graph_pair_ids = []
        for row in chosen:
            if row.get("runtime_created") is True:
                runtime_created_pairs.append(row.get("pair_id"))
            delta = row.get("proposal_runtime_delta")
            if not isinstance(delta, dict):
                graph_errors.append(f"{arm}/{row.get('pair_id')}: proposal runtime delta is absent")
                continue
            row_calls = delta.get("calls") or {}
            row_fallbacks = delta.get("fallbacks") or {}
            if not isinstance(row_calls, dict) or not isinstance(row_fallbacks, dict):
                graph_errors.append(f"{arm}/{row.get('pair_id')}: calls or fallbacks is invalid")
                continue
            for key, value in row_calls.items():
                if type(value) is not int or value < 0:
                    graph_errors.append(f"{arm}/{row.get('pair_id')}: calls.{key} is invalid")
                else:
                    calls[str(key)] += value
            for key, value in row_fallbacks.items():
                if type(value) is not int or value < 0:
                    graph_errors.append(f"{arm}/{row.get('pair_id')}: fallbacks.{key} is invalid")
                elif value:
                    fallbacks[str(key)] += value
            events = delta.get("new_graph_events") or []
            if not isinstance(events, list):
                graph_errors.append(f"{arm}/{row.get('pair_id')}: new_graph_events is invalid")
                events = []
            before = delta.get("graph_count_before")
            after = delta.get("graph_count_after")
            if type(before) is not int or type(after) is not int or after < before:
                graph_errors.append(f"{arm}/{row.get('pair_id')}: graph counts are invalid")
                graph_increased = False
            else:
                graph_increased = after > before
                if after - before != len(events):
                    graph_errors.append(
                        f"{arm}/{row.get('pair_id')}: graph count delta differs from event count"
                    )
            keys = graph_call_keys(row_calls)
            used = bool(keys)
            pair_id = str(row.get("pair_id"))
            if used:
                graph_used_pair_ids.append(pair_id)
            if events or graph_increased:
                new_graph_pair_ids.append(pair_id)
                if first_graph_pair is None:
                    first_graph_pair = pair_id
            elif used:
                no_new_graph_pair_ids.append(pair_id)
            graph_events.extend(events)
            graph_rows.append({
                "pair_id": pair_id,
                "calls": dict(sorted(row_calls.items())),
                "graph_call_keys": keys,
                "fallbacks": dict(sorted(row_fallbacks.items())),
                "graph_count_before": before,
                "graph_count_after": after,
                "new_graph_event_count": len(events),
                "graph_used": used,
                "runtime_created": row.get("runtime_created") is True,
            })
            for event_index, event in enumerate(events):
                if not isinstance(event, dict):
                    graph_errors.append(
                        f"{arm}/{pair_id}: graph event {event_index} is not an object"
                    )
                    continue
                for key in ("prepare_ms", "warmup_ms", "capture_ms"):
                    value = event.get(key)
                    if value is not None and (
                        not finite_number(value) or value < 0
                    ):
                        graph_errors.append(
                            f"{arm}/{pair_id}: graph event {event_index} {key} is invalid"
                        )

        if not graph_used_pair_ids:
            graph_errors.append(f"{arm}: no proposal graph call was recorded")
        if chosen and len(runtime_created_pairs) != 1:
            graph_errors.append(
                f"{arm}: runtime_created count is {len(runtime_created_pairs)}, expected one"
            )
        final_runtime = None
        if runtime_summary is not None:
            final_runtime = runtime_summary.get("proposal_runtimes", {}).get(arm)
            if not isinstance(final_runtime, dict):
                graph_errors.append(f"runtime_summary.proposal_runtimes lacks {arm}")
            else:
                if final_runtime.get("calls", {}) != dict(calls):
                    graph_errors.append(f"{arm}: runtime summary calls differ from row deltas")
                if final_runtime.get("fallbacks", {}) != dict(fallbacks):
                    graph_errors.append(f"{arm}: runtime summary fallbacks differ from row deltas")
                if final_runtime.get("graph_count") != len(graph_events):
                    graph_errors.append(f"{arm}: runtime summary graph count differs from events")
                final_events = final_runtime.get("graph_events")
                if isinstance(final_events, list) and len(final_events) != len(graph_events):
                    graph_errors.append(f"{arm}: runtime summary event list differs from row deltas")

        first_setup_ids = [first_graph_pair] if first_graph_pair is not None else []
        stable_reuse_pair_ids = stable_runtime_reuse_pair_ids(
            method, expected_pair_ids, rows_by_pair
        )
        subset_pairs = {
            "first_graph_creation": first_setup_ids,
            "any_new_graphs": sorted(set(new_graph_pair_ids)),
            "proposal_graph_reuse_without_new_proposal_graphs": sorted(
                set(no_new_graph_pair_ids)
            ),
            "stable_reuse_without_new_target_or_proposal_graphs": stable_reuse_pair_ids,
        }
        subset_definitions = {
            "first_graph_creation": (
                "The pair whose proposal runtime first recorded a new graph event."
            ),
            "any_new_graphs": (
                "Pairs whose proposal runtime recorded at least one new graph event."
            ),
            "proposal_graph_reuse_without_new_proposal_graphs": (
                "Proposal graph was used and this pair added no proposal graph event; "
                "Target graph creation is not constrained."
            ),
            "stable_reuse_without_new_target_or_proposal_graphs": (
                "All three runtimes already existed; the proposal graph was used; "
                "the proposal, same-method Target and common Target added no graph events."
            ),
        }
        subset_comparisons = {}
        for subset_name, pair_ids in subset_pairs.items():
            subset_comparisons[subset_name] = {}
            for label, (numerator, denominator) in {
                "target_only_over_common_target": (
                    f"target_only:{method}", "target_only:target"
                ),
                "proposal_over_method_target_only": (
                    f"proposal_graph:{method}", f"target_only:{method}"
                ),
                "proposal_over_common_target": (
                    f"proposal_graph:{method}", "target_only:target"
                ),
            }.items():
                subset_comparisons[subset_name][label] = summarize_ratio_pair(
                    rows_by_pair, numerator, denominator, budget, pair_ids=pair_ids
                )
        # Keep the existing key as an alias while giving its proposal-only scope a clear name.
        subset_comparisons["graph_reuse_without_new_graphs"] = subset_comparisons[
            "proposal_graph_reuse_without_new_proposal_graphs"
        ]
        subset_definitions["graph_reuse_without_new_graphs"] = subset_definitions[
            "proposal_graph_reuse_without_new_proposal_graphs"
        ]
        graph_summaries[method] = {
            "proposal_graph_runs": len(chosen),
            "runtime_created_pair_ids": runtime_created_pairs,
            "calls": dict(sorted(calls.items())),
            "fallbacks": dict(sorted(fallbacks.items())),
            "graph_call_keys": sorted({
                key for item in graph_rows for key in item["graph_call_keys"]
            }),
            "actual_graph_use_pair_count": len(set(graph_used_pair_ids)),
            "new_graph_event_count": len(graph_events),
            "graph_event_costs": summarize_event_cost(graph_events),
            "first_graph_creation_pair_id": first_graph_pair,
            "pairs_with_new_graph_events": sorted(set(new_graph_pair_ids)),
            "pairs_with_graph_reuse_and_no_new_events": sorted(set(no_new_graph_pair_ids)),
            "pairs_with_proposal_graph_reuse_without_new_proposal_graphs": sorted(
                set(no_new_graph_pair_ids)
            ),
            "pairs_with_stable_reuse_without_new_target_or_proposal_graphs": stable_reuse_pair_ids,
            "row_runtime": graph_rows,
            "subset_definitions": subset_definitions,
            "subset_comparisons": subset_comparisons,
            "cold_setup_in_outer_timing": graph_setup_in_outer_timing,
        }

    target_runtime_errors = []
    target_runtime_arm_details = {}
    target_runtime_rebuilt = {}
    for arm, chosen in sorted(rows_by_arm.items()):
        final_runtime = None
        if runtime_summary is not None:
            final_runtime = runtime_summary.get("dynamic_runtimes", {}).get(arm)
        rebuilt, errors = aggregate_target_runtime_rows(arm, chosen, final_runtime)
        target_runtime_rebuilt[arm] = rebuilt
        target_runtime_errors.extend(errors)
        target_runtime_arm_details[arm] = {
            "row_count": len(chosen),
            "row_runtime_created_pair_ids": [
                row.get("pair_id") for row in chosen if row.get("runtime_created") is True
            ],
            "row_rebuilt": rebuilt,
            "matches_runtime_summary": not errors,
            "errors": errors,
        }

    target_runtime_pair_details = {}
    for method in candidate_methods:
        arm_pair_details = []
        for pair_id in sorted(expected_pair_ids):
            arms = rows_by_pair.get(pair_id, {})
            eager = arms.get(f"target_only:{method}")
            graphed = arms.get(f"proposal_graph:{method}")
            if eager is None or graphed is None:
                continue
            try:
                eager_shape = target_runtime_delta_shape(eager.get("runtime_delta"))
                graph_shape = target_runtime_delta_shape(graphed.get("runtime_delta"))
            except InputDataError as error:
                message = f"{method}/{pair_id}: {error}"
                target_runtime_errors.append(message)
                arm_pair_details.append({"pair_id": pair_id, "error": str(error)})
                continue
            field_equal = {
                field: eager_shape[field] == graph_shape[field]
                for field in eager_shape
            }
            differences = {
                field: {
                    "target_only": eager_shape[field],
                    "proposal_graph": graph_shape[field],
                }
                for field, equal in field_equal.items()
                if not equal
            }
            if differences:
                target_runtime_errors.append(
                    f"{method}/{pair_id}: Target runtime deltas differ in "
                    f"{', '.join(sorted(differences))}"
                )
            arm_pair_details.append({
                "pair_id": pair_id,
                "source": eager.get("source"),
                "output_equal": eager.get("output_tokens") == graphed.get("output_tokens"),
                "counters_equal": eager.get("counters") == graphed.get("counters"),
                "field_equal": field_equal,
                "differences": differences,
            })

        eager_arm = target_runtime_rebuilt.get(f"target_only:{method}")
        graph_arm = target_runtime_rebuilt.get(f"proposal_graph:{method}")
        summary_field_equal = {}
        summary_differences = {}
        if eager_arm is not None and graph_arm is not None:
            for field in eager_arm:
                same = eager_arm[field] == graph_arm[field]
                summary_field_equal[field] = same
                if not same:
                    summary_differences[field] = {
                        "target_only": eager_arm[field],
                        "proposal_graph": graph_arm[field],
                    }
            if summary_differences:
                target_runtime_errors.append(
                    f"{method}: Target runtime arm summaries differ in "
                    f"{', '.join(sorted(summary_differences))}"
                )
        target_runtime_pair_details[method] = {
            "pair_count": len(arm_pair_details),
            "pairs": arm_pair_details,
            "arm_summary_field_equal": summary_field_equal,
            "arm_summary_differences": summary_differences,
        }

    target_runtime_pair_parity = {
        "pair_count": sum(item["pair_count"] for item in target_runtime_pair_details.values()),
        "errors": target_runtime_errors,
        "passed": not target_runtime_errors,
        "arm_summaries": target_runtime_arm_details,
        "same_method_pair_comparisons": target_runtime_pair_details,
        "comparison_fields": [
            "calls", "fallbacks", "graph_count_before", "graph_count_after",
            "new_graph_events", "new_cache_events",
        ],
        "event_comparison": (
            "Preserves query_tokens, read_bucket, logit_types, first_pos, cache/requested "
            "capacities and event kind; removes timing, allocator snapshots and cache identity."
        ),
    }

    result_checks = [
        {
            "name": "complete record count",
            "passed": (
                pipeline.get("status") == "complete"
                and pipeline.get("completed_runs") == expected_runs
                and len(rows) == expected_runs
            ),
            "details": {
                "prompt_count": prompt_count,
                "method_count": len(methods),
                "planned_execution_count": len(executions),
                "expected_records": expected_runs,
                "actual_records": len(rows),
            },
        },
        {
            "name": "unique complete pair arms",
            "passed": not duplicate_arms and not unplanned_rows and not completeness_errors,
            "details": {
                "duplicate_arms": duplicate_arms,
                "unplanned_rows": unplanned_rows,
                "errors": completeness_errors,
            },
        },
        {
            "name": "source prompt correspondence",
            "passed": source_match and not prompt_errors,
            "details": {
                "plan_sources": sorted(expected_sources),
                "prompt_sources": sorted(prompt_sources),
                "errors": prompt_errors,
            },
        },
        {
            "name": "legal output and recorded sequence validation",
            "passed": not row_errors,
            "details": {"error_count": len(row_errors), "errors": row_errors[:100]},
        },
        {
            "name": "proposal graph output and counter replay",
            "passed": not parity_errors,
            "details": parity_by_method | {"errors": parity_errors[:100]},
        },
        {
            "name": "source weighted comparisons",
            "passed": not ratio_errors,
            "details": {
                "comparisons_per_method": list(
                    next(iter(contrast_summaries.values())).keys()
                ) if contrast_summaries else [],
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_DRAWS,
                "errors": ratio_errors,
            },
        },
        {
            "name": "proposal graph calls and event deltas",
            "passed": not graph_errors,
            "details": {"errors": graph_errors[:100]},
        },
        {
            "name": "graph setup included in outer timing",
            "passed": graph_setup_in_outer_timing,
            "details": {"timing_scope": timing_scope},
        },
        {
            "name": "Target runtime row aggregation and same-method arm parity",
            "passed": not target_runtime_errors,
            "details": {
                "pair_count": target_runtime_pair_parity["pair_count"],
                "arm_count": len(target_runtime_arm_details),
                "comparison_fields": target_runtime_pair_parity["comparison_fields"],
                "errors": target_runtime_errors[:100],
                "same_method_pair_comparisons": target_runtime_pair_details,
            },
        },
    ]
    failures = [item["name"] for item in result_checks if not item["passed"]]

    rank_all = []
    rank_full = []
    for method in candidate_methods:
        comparison = contrast_summaries.get(method, {}).get("proposal_over_common_target", {})
        all_value = comparison.get("all_output_lengths", {})
        full_value = comparison.get("common_full_budget", {})
        if all_value.get("source_equal_geometric_mean") is not None:
            rank_all.append((method, all_value["source_equal_geometric_mean"]))
        if full_value.get("source_equal_geometric_mean") is not None:
            rank_full.append((method, full_value["source_equal_geometric_mean"]))

    execution_meaning = plan.get("execution_meaning", {})
    return {
        "status": "passed" if not failures else "failed",
        "run_dir": str(run_dir.resolve()),
        "analysis_scope": (
            "CPU-only recomputation from plan.json, prompts.json and records.jsonl; "
            "no runner or metrics module imported."
        ),
        "budget": budget,
        "execution_meaning": execution_meaning,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_DRAWS,
            "unit": "source",
            "quantile": "linear interpolation",
        },
        "run_shape": {
            "prompt_count": prompt_count,
            "method_count": len(methods),
            "methods": methods,
            "executions": executions,
            "expected_records": expected_runs,
            "actual_records": len(rows),
            "budget": budget,
            "sources": sorted(prompt_sources),
        },
        "arm_summaries": arm_summaries,
        "comparisons": contrast_summaries,
        "same_seed_parity": parity_by_method,
        "proposal_graph_runtime": graph_summaries,
        "target_runtime_pair_parity": target_runtime_pair_parity,
        "rankings_vs_common_target": {
            "all_output_lengths": [
                {"method": method, "ratio": value}
                for method, value in sorted(rank_all, key=lambda item: item[1], reverse=True)
            ],
            "common_full_budget": [
                {"method": method, "ratio": value}
                for method, value in sorted(rank_full, key=lambda item: item[1], reverse=True)
            ],
        },
        "cold_setup_interpretation": {
            "first_graph_creation_is_in_outer_time": graph_setup_in_outer_timing,
            "per_method_first_creation_and_no_new_graph_subsets": True,
            "speed_values_are_end_to_end_outer_wall_rate": True,
        },
        "checks": result_checks,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed proposal graph run directory")
    parser.add_argument("--budget", type=int, help="Expected token budget; defaults to plan.json")
    parser.add_argument(
        "--summary-output", type=Path,
        help="Summary JSON path; defaults to run_dir/summary.json",
    )
    parser.add_argument(
        "--audit-output", type=Path,
        help="Audit JSON path; defaults to run_dir/audit.json",
    )
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        parser.error(f"run_dir is not a directory: {run_dir}")
    summary_path = (
        args.summary_output.resolve()
        if args.summary_output is not None
        else (run_dir / "summary.json").resolve()
    )
    audit_path = (
        args.audit_output.resolve()
        if args.audit_output is not None
        else (run_dir / "audit.json").resolve()
    )
    if summary_path == audit_path:
        parser.error("Summary and audit outputs must use different paths")
    existing = [str(path) for path in (summary_path, audit_path) if path.exists()]
    if existing:
        parser.error(f"Existing outputs were preserved: {', '.join(existing)}")
    try:
        report = analyze(run_dir, requested_budget=args.budget)
    except InputDataError as error:
        parser.error(str(error))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    summary_report = {
        key: report[key]
        for key in (
            "status", "run_dir", "analysis_scope", "budget", "execution_meaning",
            "bootstrap", "run_shape", "arm_summaries", "comparisons",
            "rankings_vs_common_target", "cold_setup_interpretation",
        )
    }
    summary_report["proposal_graph_runtime"] = {
        method: {
            key: value
            for key, value in runtime.items()
            if key != "row_runtime"
        }
        for method, runtime in report["proposal_graph_runtime"].items()
    }
    audit_report = {
        "status": report["status"],
        "run_dir": report["run_dir"],
        "run_shape": report["run_shape"],
        "checks": report["checks"],
        "same_seed_parity": report["same_seed_parity"],
        "target_runtime_pair_parity": report["target_runtime_pair_parity"],
        "proposal_graph_runtime": report["proposal_graph_runtime"],
    }
    try:
        with summary_path.open("x", encoding="utf-8") as stream:
            json.dump(summary_report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        with audit_path.open("x", encoding="utf-8") as stream:
            json.dump(audit_report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except FileExistsError as error:
        parser.error(f"Existing output was preserved: {error.filename}")
    passed = sum(bool(item["passed"]) for item in report["checks"])
    print(json.dumps({
        "status": report["status"],
        "summary": str(summary_path),
        "audit": str(audit_path),
        "passed_checks": passed,
        "check_count": len(report["checks"]),
    }, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
