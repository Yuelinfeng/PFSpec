#!/usr/bin/env python3
"""Independently summarize an original-Target comparison run from raw records."""

from __future__ import annotations

import argparse
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


def quantile(values, probability):
    ordered = sorted(values)
    point = (len(ordered) - 1) * probability
    low = int(point)
    high = min(low + 1, len(ordered) - 1)
    if high == low:
        return ordered[low]
    return ordered[low] * (high - point) + ordered[high] * (point - low)


def validate_prompt(prompt, *, eos_id, position_vocab, feature_offset, ranks):
    errors = []
    if not isinstance(prompt, list) or not prompt or prompt[0] != eos_id:
        return ["prompt BOS sentinel differs from eos_id"]
    if any(type(token) is not int for token in prompt):
        return ["prompt contains a non-integer token"]
    body = prompt[1:]
    if eos_id in body:
        errors.append("prompt contains EOS after its BOS sentinel")
    if len(body) % 2:
        errors.append("prompt does not end on a complete P/F pair")
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


def validate_sequence(prompt, output, *, eos_id, position_vocab, feature_offset, ranks):
    errors = []
    if not isinstance(output, list) or not output:
        return ["output is empty or is not a list"]
    if any(type(token) is not int for token in output):
        return ["output contains a non-integer token"]
    body = prompt[1:] + output
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
    arms = {"original_target:target", "target_only:target"}
    for method in methods:
        if method == "target":
            continue
        for execution in executions:
            arms.add(f"{execution}:{method}")
    return arms


def first_output_difference(left, right):
    if not isinstance(left, list) or not isinstance(right, list):
        return {"index": None, "left_token": None, "right_token": None}
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return {"index": index, "left_token": left_token, "right_token": right_token}
    if len(left) != len(right):
        index = min(len(left), len(right))
        return {
            "index": index,
            "left_token": left[index] if index < len(left) else None,
            "right_token": right[index] if index < len(right) else None,
        }
    return {"index": None, "left_token": None, "right_token": None}


def counter_difference(left, right):
    left_counters = left.get("counters") or {}
    right_counters = right.get("counters") or {}
    differences = []
    for key in sorted(set(left_counters) | set(right_counters)):
        if left_counters.get(key) != right_counters.get(key):
            differences.append({
                "field": key,
                "original_target": left_counters.get(key),
                "target_only": right_counters.get(key),
            })
    return differences


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


def ratio_summary(rows_by_pair, numerator, denominator, budget, *, full_only=False):
    source_logs = defaultdict(list)
    pair_count = 0
    numerator_tokens = 0
    denominator_tokens = 0
    numerator_seconds = 0.0
    denominator_seconds = 0.0
    for arms in rows_by_pair.values():
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
        numerator_tokens += int(up["generated_tokens"])
        denominator_tokens += int(down["generated_tokens"])
        numerator_seconds += float(up["outer_wall_seconds"])
        denominator_seconds += float(down["outer_wall_seconds"])

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
        "numerator_generated_tokens": numerator_tokens,
        "denominator_generated_tokens": denominator_tokens,
        "numerator_outer_wall_seconds": numerator_seconds,
        "denominator_outer_wall_seconds": denominator_seconds,
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


def summarize_ratio(rows_by_pair, numerator, denominator, budget):
    return {
        "all_output_lengths": ratio_summary(
            rows_by_pair, numerator, denominator, budget
        ),
        "common_full_budget": ratio_summary(
            rows_by_pair, numerator, denominator, budget, full_only=True
        ),
    }


def graph_call_keys(calls):
    if not isinstance(calls, dict):
        return []
    return sorted(
        str(key) for key, value in calls.items()
        if type(value) is int and value > 0
        and ("graph" in str(key).lower() or "replay" in str(key).lower())
    )


def analyze_loaded(plan, prompts_doc, rows, pipeline, runtime_summary, requested_budget=None):
    errors = []
    warnings = []
    if not isinstance(plan, dict):
        raise InputDataError("plan.json is not an object")
    prompts = prompts_doc.get("contexts") if isinstance(prompts_doc, dict) else prompts_doc
    if not isinstance(prompts, list):
        raise InputDataError("prompts.json lacks a contexts list")
    methods = plan.get("methods")
    executions = plan.get("executions")
    if not isinstance(methods, list) or not methods or "target" not in methods:
        raise InputDataError("plan.methods must be a nonempty list including target")
    if len(set(methods)) != len(methods):
        raise InputDataError("plan.methods contains duplicates")
    if not isinstance(executions, list) or not executions or any(
        execution != "proposal_graph" for execution in executions
    ):
        raise InputDataError("plan.executions must list proposal_graph for candidate methods")
    if "original_target" not in plan.get("execution_meaning", {}):
        warnings.append("plan.execution_meaning does not describe original_target")
    if not isinstance(pipeline, dict) or pipeline.get("status") != "complete":
        raise InputDataError("pipeline.json status is not complete")
    if not isinstance(runtime_summary, dict) or runtime_summary.get("status") != "complete":
        raise InputDataError("runtime_summary.json status is not complete")

    budget = int(plan.get("budget", 0))
    if requested_budget is not None and int(requested_budget) != budget:
        raise InputDataError(f"--budget {requested_budget} differs from plan budget {budget}")
    if budget < 2 or budget % 2:
        raise InputDataError("plan.budget must be a positive even integer")
    if int(plan.get("prompt_count", -1)) != len(prompts):
        raise InputDataError("plan.prompt_count differs from prompts.json")
    expected_runs = len(prompts) * (2 + (len(methods) - 1) * len(executions))
    if int(plan.get("expected_runs", -1)) != expected_runs:
        raise InputDataError("plan.expected_runs differs from planned arm count")
    if pipeline.get("completed_runs") != expected_runs or len(rows) != expected_runs:
        raise InputDataError("pipeline or records count differs from planned arm count")
    if runtime_summary.get("completed_runs") != expected_runs:
        raise InputDataError("runtime_summary.completed_runs differs from planned arm count")

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
        prompt_by_id[pair_id] = prompt
        prompt_errors.extend(
            f"{pair_id}: {message}"
            for message in validate_prompt(
                prompt.get("prompt"), eos_id=eos_id, position_vocab=position_vocab,
                feature_offset=feature_offset, ranks=ranks
            )
        )
    errors.extend(prompt_errors)

    expected_sources = {str(value) for value in plan.get("sources", [])}
    prompt_sources = {str(prompt.get("source")) for prompt in prompt_by_id.values()}
    if expected_sources and expected_sources != prompt_sources:
        errors.append("plan.sources differs from prompts.json sources")

    rows_by_pair = defaultdict(dict)
    rows_by_arm = defaultdict(list)
    duplicate_arms = []
    unplanned_rows = []
    sequence_validated_rows = 0
    for index, row in enumerate(rows):
        pair_id = str(row.get("pair_id", ""))
        method = row.get("method")
        execution = row.get("execution")
        if pair_id not in prompt_by_id or method not in methods:
            unplanned_rows.append({"index": index, "pair_id": pair_id, "method": method})
            continue
        allowed_execution = (
            {"original_target", "target_only"} if method == "target"
            else set(executions)
        )
        if execution not in allowed_execution:
            unplanned_rows.append({
                "index": index, "pair_id": pair_id,
                "method": method, "execution": execution,
            })
            continue
        expected_arm = f"{execution}:{method}"
        if row.get("arm") != expected_arm:
            errors.append(f"record {index}: arm={row.get('arm')!r}, expected {expected_arm!r}")
        if expected_arm in rows_by_pair[pair_id]:
            duplicate_arms.append({"pair_id": pair_id, "arm": expected_arm})
        rows_by_pair[pair_id][expected_arm] = row
        rows_by_arm[expected_arm].append(row)
        prompt = prompt_by_id[pair_id]
        for key in ("source", "prefix", "seed"):
            if row.get(key) != prompt.get(key):
                errors.append(f"record {index}: {key} differs from prompt {pair_id}")

        output = row.get("output_tokens")
        if not isinstance(output, list) or not output or any(type(token) is not int for token in output):
            errors.append(f"record {index}: output_tokens is invalid")
            output = [] if not isinstance(output, list) else output
        generated = row.get("generated_tokens")
        seconds = row.get("outer_wall_seconds")
        if type(generated) is not int or generated != len(output):
            errors.append(f"record {index}: generated_tokens differs from output length")
        if len(output) > budget:
            errors.append(f"record {index}: output exceeds plan budget")
        if not finite_number(seconds) or seconds <= 0:
            errors.append(f"record {index}: outer_wall_seconds must be finite and positive")
        if output and finite_number(seconds) and seconds > 0:
            expected_rate = len(output) / seconds
            actual_rate = row.get("tokens_per_second")
            if not finite_number(actual_rate) or not math.isclose(
                float(actual_rate), expected_rate, rel_tol=1e-10, abs_tol=1e-10
            ):
                errors.append(f"record {index}: tokens_per_second differs from output length/time")
        output_has_eos = bool(output and output[-1] == eos_id)
        if type(row.get("eos")) is not bool or row.get("eos") != output_has_eos:
            errors.append(f"record {index}: eos flag differs from output")
        if row.get("sequence_validated") is not True:
            errors.append(f"record {index}: sequence_validated is not true")
        prompt_tokens = prompt.get("prompt")
        if output and isinstance(prompt_tokens, list) and all(
            type(token) is int for token in prompt_tokens
        ):
            sequence_errors = validate_sequence(
                prompt_tokens, output, eos_id=eos_id,
                position_vocab=position_vocab, feature_offset=feature_offset,
                ranks=ranks,
            )
            if not sequence_errors:
                sequence_validated_rows += 1
            errors.extend(f"record {index}: {message}" for message in sequence_errors)
        if not isinstance(row.get("counters"), dict):
            errors.append(f"record {index}: counters is missing or invalid")
        if not isinstance(row.get("runtime_delta"), dict):
            errors.append(f"record {index}: runtime_delta is missing or invalid")
        if execution == "proposal_graph" and not isinstance(row.get("proposal_runtime_delta"), dict):
            errors.append(f"record {index}: proposal_runtime_delta is missing or invalid")
        if execution != "proposal_graph" and row.get("proposal_runtime_delta") is not None:
            errors.append(f"record {index}: non-proposal execution has proposal_runtime_delta")

    expected_arms = planned_arms(methods, executions)
    missing_arms = []
    extra_arms = []
    for pair_id in prompt_by_id:
        present = set(rows_by_pair.get(pair_id, {}))
        if expected_arms - present:
            missing_arms.append({"pair_id": pair_id, "arms": sorted(expected_arms - present)})
        if present - expected_arms:
            extra_arms.append({"pair_id": pair_id, "arms": sorted(present - expected_arms)})
    if duplicate_arms:
        errors.append(f"duplicate arm rows: {len(duplicate_arms)}")
    if unplanned_rows:
        errors.append(f"unplanned record rows: {len(unplanned_rows)}")
    if missing_arms:
        errors.append(f"incomplete pairs: {len(missing_arms)}")
    if extra_arms:
        errors.append(f"unexpected arms: {len(extra_arms)}")

    candidate_methods = [method for method in methods if method != "target"]
    parity_pairs = []
    parity_counts = Counter()
    for pair_id in sorted(prompt_by_id):
        arms = rows_by_pair.get(pair_id, {})
        original = arms.get("original_target:target")
        optimized = arms.get("target_only:target")
        if original is None or optimized is None:
            continue
        output_equal = original.get("output_tokens") == optimized.get("output_tokens")
        counter_equal = original.get("counters") == optimized.get("counters")
        entry = {
            "pair_id": pair_id,
            "source": original.get("source"),
            "prefix": original.get("prefix"),
            "output_equal": output_equal,
            "counters_equal": counter_equal,
            "generated_tokens": [
                original.get("generated_tokens"), optimized.get("generated_tokens")
            ],
        }
        if not output_equal:
            entry["first_output_difference"] = first_output_difference(
                original.get("output_tokens"), optimized.get("output_tokens")
            )
            parity_counts["output_different"] += 1
        else:
            parity_counts["output_equal"] += 1
        if not counter_equal:
            entry["counter_differences"] = counter_difference(original, optimized)
            parity_counts["counters_different"] += 1
        else:
            parity_counts["counters_equal"] += 1
        if original.get("generated_tokens") != optimized.get("generated_tokens"):
            parity_counts["length_different"] += 1
        parity_pairs.append(entry)
    if len(parity_pairs) != len(prompt_by_id):
        errors.append("original/optimized Target parity rows do not cover all prompt pairs")

    arm_summaries = {
        arm: summarize_arm(chosen)
        for arm, chosen in sorted(rows_by_arm.items())
    }
    comparisons = {}
    comparison_pairs = {
        "optimized_target_over_original": (
            "target_only:target", "original_target:target"
        ),
    }
    for method in candidate_methods:
        comparison_pairs[f"{method}_over_original_target"] = (
            f"proposal_graph:{method}", "original_target:target"
        )
        comparison_pairs[f"{method}_over_optimized_target"] = (
            f"proposal_graph:{method}", "target_only:target"
        )
    for label, (numerator, denominator) in comparison_pairs.items():
        if numerator not in rows_by_arm or denominator not in rows_by_arm:
            errors.append(f"missing arm for comparison {label}")
            continue
        value = summarize_ratio(rows_by_pair, numerator, denominator, budget)
        comparisons[label] = value
        if value["all_output_lengths"]["source_equal_geometric_mean"] is None:
            errors.append(f"comparison {label} has no complete paired ratios")

    graph_usage = {}
    graph_errors = []
    for method in candidate_methods:
        arm = f"proposal_graph:{method}"
        calls = Counter()
        fallbacks = Counter()
        events = []
        graph_used_pairs = []
        runtime_created_pairs = []
        previous_count = 0
        arm_rows = rows_by_arm.get(arm, [])
        for row in arm_rows:
            pair_id = str(row["pair_id"])
            if row.get("runtime_created") is True:
                runtime_created_pairs.append(pair_id)
            delta = row.get("proposal_runtime_delta") or {}
            row_calls = delta.get("calls") or {}
            row_fallbacks = delta.get("fallbacks") or {}
            if not isinstance(row_calls, dict) or not isinstance(row_fallbacks, dict):
                graph_errors.append(f"{arm}/{pair_id}: calls or fallbacks is invalid")
                continue
            for key, value in row_calls.items():
                if type(value) is not int or value < 0:
                    graph_errors.append(f"{arm}/{pair_id}: calls.{key} is invalid")
                else:
                    calls[str(key)] += value
            for key, value in row_fallbacks.items():
                if type(value) is not int or value < 0:
                    graph_errors.append(f"{arm}/{pair_id}: fallbacks.{key} is invalid")
                elif value:
                    fallbacks[str(key)] += value
            if graph_call_keys(row_calls):
                graph_used_pairs.append(pair_id)
            before, after = delta.get("graph_count_before"), delta.get("graph_count_after")
            new_events = delta.get("new_graph_events")
            if type(before) is not int or type(after) is not int or after < before:
                graph_errors.append(f"{arm}/{pair_id}: graph counts are invalid")
                new_events = new_events if isinstance(new_events, list) else []
            else:
                if before != previous_count:
                    graph_errors.append(
                        f"{arm}/{pair_id}: graph_count_before={before}, expected {previous_count}"
                    )
                if not isinstance(new_events, list):
                    graph_errors.append(f"{arm}/{pair_id}: new_graph_events is invalid")
                    new_events = []
                if after - before != len(new_events):
                    graph_errors.append(f"{arm}/{pair_id}: graph count delta differs from event count")
                previous_count = after
            events.extend(new_events)
        final = runtime_summary.get("proposal_runtimes", {}).get(arm)
        final_matches = False
        if not isinstance(final, dict):
            graph_errors.append(f"runtime_summary.proposal_runtimes lacks {arm}")
        else:
            final_matches = (
                final.get("calls", {}) == dict(calls)
                and final.get("fallbacks", {}) == dict(fallbacks)
                and final.get("graph_count") == len(events)
                and isinstance(final.get("graph_events"), list)
                and len(final.get("graph_events", [])) == len(events)
            )
            if not final_matches:
                graph_errors.append(f"{arm}: row deltas differ from runtime_summary")
        if not graph_used_pairs:
            graph_errors.append(f"{arm}: no proposal graph call was recorded")
        if len(runtime_created_pairs) != 1:
            graph_errors.append(f"{arm}: expected one runtime creation row, found {len(runtime_created_pairs)}")
        graph_usage[method] = {
            "rows": len(arm_rows),
            "graph_used_pair_count": len(set(graph_used_pairs)),
            "runtime_created_pair_ids": runtime_created_pairs,
            "calls": dict(sorted(calls.items())),
            "fallbacks": dict(sorted(fallbacks.items())),
            "new_graph_event_count": len(events),
            "runtime_summary_matches_row_deltas": final_matches,
        }
    errors.extend(graph_errors)

    return {
        "status": "complete" if not errors else "failed",
        "run_dir": None,
        "budget": budget,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_DRAWS,
                       "unit": "source", "aggregation": "equal-source mean of pair log-rate ratios"},
        "plan": {
            "methods": methods,
            "executions": executions,
            "prompt_count": len(prompts),
            "expected_runs": expected_runs,
            "actual_runs": len(rows),
            "source_count": len(prompt_sources),
            "prefixes": sorted({int(prompt["prefix"]) for prompt in prompt_by_id.values()}),
            "seeds": sorted({int(prompt["seed"]) for prompt in prompt_by_id.values()}),
            "timing_scope": plan.get("timing_scope"),
            "execution_meaning": plan.get("execution_meaning"),
            "proposal_verified": plan.get("proposal_verified"),
        },
        "arm_summaries": arm_summaries,
        "comparisons": comparisons,
        "original_vs_optimized_target_parity": {
            "pair_count": len(parity_pairs),
            "output_equal_count": parity_counts["output_equal"],
            "output_different_count": parity_counts["output_different"],
            "counter_equal_count": parity_counts["counters_equal"],
            "counter_different_count": parity_counts["counters_different"],
            "length_different_count": parity_counts["length_different"],
            "pairs": parity_pairs,
        },
        "proposal_graph_usage": graph_usage,
        "audit": {
            "errors": errors,
            "warnings": warnings,
            "duplicate_arms": duplicate_arms,
            "unplanned_rows": unplanned_rows,
            "missing_arms": missing_arms,
            "extra_arms": extra_arms,
            "records_with_valid_sequences": sequence_validated_rows,
        },
        "interpretation": {
            "main_ratio": "proposal_graph method throughput divided by original_target:target throughput",
            "internal_ratios": "optimized Target relative to original Target and proposal_graph method relative to optimized Target",
            "graph_gain_limit": "No same-method eager arm is present, so proposal_graph relative to optimized Target includes method and proposal-graph effects together.",
            "length_policy": "Generated lengths, EOS and wall time are retained as recorded; rates are not normalized to equal output lengths.",
            "ratio_direction": "Values above 1 indicate higher generated-token throughput for the numerator arm.",
        },
    }


def analyze_run(run_dir: Path, requested_budget=None):
    plan = read_json(run_dir / "plan.json")
    prompts = read_json(run_dir / "prompts.json")
    rows = read_records(run_dir / "records.jsonl")
    pipeline = read_json(run_dir / "pipeline.json")
    runtime_summary = read_json(run_dir / "runtime_summary.json")
    summary = analyze_loaded(plan, prompts, rows, pipeline, runtime_summary, requested_budget)
    summary["run_dir"] = str(run_dir.resolve())
    return summary


def fmt_ratio(value):
    estimate = value.get("source_equal_geometric_mean")
    if estimate is None:
        return "无有效配对"
    ci = value.get("source_bootstrap_95_ci")
    if ci is None:
        interval = "单来源"
    else:
        interval = f"95%区间 [{ci[0]:.3f}, {ci[1]:.3f}]"
    return f"{estimate:.3f}×；{interval}；{value['pair_count']} 对 / {value['source_count']} 来源"


def results_markdown(summary):
    plan = summary["plan"]
    lines = [
        "# 同轮原始 Target 对照结果",
        "",
        f"状态：{summary['status']}；样本 {plan['actual_runs']}/{plan['expected_runs']} 条，",
        f"{plan['source_count']} 个来源，前缀 {', '.join(map(str, plan['prefixes']))}，预算 {summary['budget']} 个输出 token。",
        "",
        "主比较以每个提案图方法的实际生成 token/s 对比同一轮原始 Target。每个来源先平均其配对 log 比值，再对来源等权取几何平均；区间以来源为单位重抽 10,000 次。所有输出长度与计时均按记录保留。",
        "",
        "## 主结果",
        "",
        "| 方法 | 相对原始 Target | 双臂均达预算长度 | 提案臂生成 token | 原始 Target 生成 token |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in plan["methods"]:
        if method == "target":
            continue
        key = f"{method}_over_original_target"
        value = summary["comparisons"].get(key)
        if not value:
            continue
        all_lengths = value["all_output_lengths"]
        full = value["common_full_budget"]
        lines.append(
            f"| {method} | {fmt_ratio(all_lengths)} | {fmt_ratio(full)} | "
            f"{all_lengths['numerator_generated_tokens']} | {all_lengths['denominator_generated_tokens']} |"
        )
    lines += [
        "",
        "## 内部拆分",
        "",
        f"优化 Target / 原始 Target：{fmt_ratio(summary['comparisons']['optimized_target_over_original']['all_output_lengths'])}。",
        "下表的提案图方法 / 优化 Target 包含方法本身与图执行的合并影响；本轮没有相同方法的普通执行臂，不能把该差值解释为图单独带来的提升。",
        "",
        "| 方法 | 提案图方法 / 优化 Target | 双臂均达预算长度 |",
        "|---|---:|---:|",
    ]
    for method in plan["methods"]:
        if method == "target":
            continue
        value = summary["comparisons"].get(f"{method}_over_optimized_target")
        if value:
            lines.append(
                f"| {method} | {fmt_ratio(value['all_output_lengths'])} | "
                f"{fmt_ratio(value['common_full_budget'])} |"
            )
    parity = summary["original_vs_optimized_target_parity"]
    lines += [
        "",
        "## 原始与优化 Target 一致性",
        "",
        f"逐行比较 {parity['pair_count']} 对：输出完全相同 {parity['output_equal_count']} 对，"
        f"输出不同 {parity['output_different_count']} 对；计数相同 {parity['counter_equal_count']} 对，"
        f"计数不同 {parity['counter_different_count']} 对；生成长度不同 {parity['length_different_count']} 对。",
        "差异保留在 `summary.json` 的 `original_vs_optimized_target_parity.pairs` 中，含首个 token 分歧、各臂长度与计数字段差异。",
        "",
        "## 图调用与资料完整性",
        "",
    ]
    for method, usage in summary["proposal_graph_usage"].items():
        lines.append(
            f"- {method}：{usage['graph_used_pair_count']}/{usage['rows']} 对记录到图或重放调用，"
            f"新建图事件 {usage['new_graph_event_count']} 个，运行时计数与逐行差量一致："
            f"{usage['runtime_summary_matches_row_deltas']}。"
        )
    lines += [
        "",
        f"数据核算状态：{summary['status']}。",
    ]
    if summary["audit"]["errors"]:
        lines += ["", "输入问题："]
        lines.extend(f"- {error}" for error in summary["audit"]["errors"])
    if summary["audit"]["warnings"]:
        lines += ["", "说明："]
        lines.extend(f"- {warning}" for warning in summary["audit"]["warnings"])
    return "\n".join(lines) + "\n"


def write_outputs(summary, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    results_path = output_dir / "RESULTS.md"
    existing = [str(path) for path in (summary_path, results_path) if path.exists()]
    if existing:
        raise InputDataError("Refusing to overwrite existing output: " + ", ".join(existing))
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    results_text = results_markdown(summary)
    summary_tmp = summary_path.with_suffix(".json.tmp")
    results_tmp = results_path.with_suffix(".md.tmp")
    summary_tmp.write_text(summary_text, encoding="utf-8")
    results_tmp.write_text(results_text, encoding="utf-8")
    summary_tmp.replace(summary_path)
    results_tmp.replace(results_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed evaluate_original_target.py output directory")
    parser.add_argument("--budget", type=int, help="Require this output budget")
    parser.add_argument("--output-dir", type=Path, help="Output location; defaults to run_dir")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir
    try:
        summary = analyze_run(run_dir, args.budget)
        write_outputs(summary, output_dir)
    except InputDataError as error:
        print(f"analysis failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": summary["status"],
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "errors": len(summary["audit"]["errors"]),
        "proposal_methods": len(summary["proposal_graph_usage"]),
        "rows": summary["plan"]["actual_runs"],
    }, ensure_ascii=False))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
