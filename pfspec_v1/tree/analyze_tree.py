#!/usr/bin/env python3
"""Independently summarize Feature-tree decoding records using the Python standard library."""

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
METHODS = ("target", "pf_mlp", "tree_w1", "tree_w2")
ARMS = {
    "original_target:target": ("target", "original_target"),
    "proposal_graph:pf_mlp": ("pf_mlp", "proposal_graph"),
    "tree_graph:tree_w1": ("tree_w1", "tree_graph"),
    "tree_graph:tree_w2": ("tree_w2", "tree_graph"),
}
BRANCH_DELTA_KEY = {
    "proposal_graph:pf_mlp": "proposal_runtime_delta",
    "tree_graph:tree_w1": "tree_runtime_delta",
    "tree_graph:tree_w2": "tree_runtime_delta",
}
METRIC_ALIASES = {
    "candidate_tree_nodes": (
        "candidate_tree_nodes", "tree_node_count", "tree_nodes",
        "tree_nodes_proposed", "tree_nodes_visited", "tree_nodes_expanded",
        "tree_candidate_nodes", "tree_candidate_node_count", "candidate_tree_node_count",
        "nodes_proposed", "candidate_nodes", "visited_nodes", "nodes_expanded",
    ),
    "target_tree_nodes": (
        "target_tree_nodes", "target_tree_node_count", "tree_verify_nodes",
    ),
    "feature_proposals": (
        "feature_proposed", "feature_proposals", "features_proposed",
        "feature_candidates", "tree_feature_proposals", "feature_nodes_proposed",
    ),
    "feature_attempts": (
        "feature_verified", "feature_checked", "feature_attempted",
        "feature_attempts", "feature_attempt_count", "features_attempted",
        "features_checked", "feature_validation_attempts",
    ),
    "feature_accepted": (
        "feature_accepted", "feature_accepts", "features_accepted",
        "accepted_features", "accepted_feature_count", "feature_acceptance_count",
        "feature_nodes_accepted", "tree_feature_accepted",
    ),
    "target_calls": (
        "target_calls", "target_call_count", "target_forward_calls", "target_forwards",
        "target_invocations",
    ),
}
TREE_METADATA_KEYS = {
    "treewidth", "proposalwidth", "variant", "treeruntimestats",
}
TREE_ONLY_COUNTER_KEYS = {
    "k1steps", "treeverifycalls", "candidatetreenodes", "targettreenodes",
    "featureattempts", "featurebranchhits", "featureallrejected",
    "featurebranchindexcounts", "recoverycalls",
}
_NO_REFERENCE = object()


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


def finite_number(value):
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
    if low == high:
        return ordered[low]
    return ordered[low] * (high - point) + ordered[high] * (point - low)


def normalize_key(value):
    return "".join(character for character in str(value).lower() if character.isalnum())


def count_value(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and math.isfinite(value) and value >= 0:
        return value
    if isinstance(value, dict):
        values = [count_value(item) for item in value.values()]
        if values and all(item is not None for item in values):
            return sum(values)
    if isinstance(value, list):
        values = [count_value(item) for item in value]
        if values and all(item is not None for item in values):
            return sum(values)
    return None


def find_metric(value, aliases, path=""):
    wanted = {normalize_key(alias): alias for alias in aliases}
    if isinstance(value, dict):
        for key, item in value.items():
            if normalize_key(key) in wanted:
                total = count_value(item)
                if total is not None:
                    return total, f"{path}.{key}" if path else str(key)
        for key, item in value.items():
            child = find_metric(item, aliases, f"{path}.{key}" if path else str(key))
            if child is not None:
                return child
    return None


def metric_for_row(row, aliases):
    for source in ("counters", "stats", "tree_runtime_delta",
                   "proposal_runtime_delta", "runtime_delta"):
        value = row.get(source)
        if value is None:
            continue
        found = find_metric(value, aliases, source)
        if found is not None:
            return found
    return None, None


def graph_call_keys(delta):
    calls = delta.get("calls") if isinstance(delta, dict) else None
    if not isinstance(calls, dict):
        return []
    return sorted(
        str(key) for key, value in calls.items()
        if type(value) in (int, float) and value > 0
        and any(word in str(key).lower() for word in ("graph", "replay", "tree"))
    )


def event_list(delta):
    if not isinstance(delta, dict):
        return None
    for key in ("new_graph_events", "new_tree_graph_events", "new_tree_events"):
        value = delta.get(key)
        if isinstance(value, list):
            return value
    return None


def event_added(delta):
    events = event_list(delta)
    if events is not None and events:
        return True
    if not isinstance(delta, dict):
        return False
    for before_key, after_key in (
        ("graph_count_before", "graph_count_after"),
        ("tree_graph_count_before", "tree_graph_count_after"),
        ("tree_count_before", "tree_count_after"),
    ):
        before, after = delta.get(before_key), delta.get(after_key)
        if type(before) is int and type(after) is int and after > before:
            return True
    return False


def no_new_events(delta):
    events = event_list(delta)
    if events is None or events:
        return False
    if not isinstance(delta, dict):
        return False
    for before_key, after_key in (
        ("graph_count_before", "graph_count_after"),
        ("tree_graph_count_before", "tree_graph_count_after"),
        ("tree_count_before", "tree_count_after"),
    ):
        before, after = delta.get(before_key), delta.get(after_key)
        if type(before) is int and type(after) is int:
            return before == after
    return True


def output_difference(left, right):
    first = left.get("output_tokens")
    second = right.get("output_tokens")
    if not isinstance(first, list) or not isinstance(second, list):
        return None
    for index, (a, b) in enumerate(zip(first, second)):
        if a != b:
            return {"index": index, "left_token": a, "right_token": b}
    if len(first) != len(second):
        index = min(len(first), len(second))
        return {
            "index": index,
            "left_token": first[index] if index < len(first) else None,
            "right_token": second[index] if index < len(second) else None,
        }
    return None


def strip_tree_specific(value, path="", reference=None):
    ignored = []
    if isinstance(value, dict):
        kept = {}
        has_reference = reference is not None
        reference = reference if isinstance(reference, dict) else {}
        for key, item in value.items():
            lowered = normalize_key(key)
            if lowered in TREE_METADATA_KEYS or (
                lowered in TREE_ONLY_COUNTER_KEYS and has_reference and key not in reference
            ):
                ignored.append(f"{path}.{key}" if path else str(key))
                continue
            child, child_ignored = strip_tree_specific(
                item, f"{path}.{key}" if path else str(key),
                reference.get(key, _NO_REFERENCE) if has_reference else None,
            )
            kept[str(key)] = child
            ignored.extend(child_ignored)
        return kept, ignored
    if isinstance(value, list):
        kept = []
        for index, item in enumerate(value):
            reference_item = (
                reference[index]
                if isinstance(reference, list) and index < len(reference)
                else _NO_REFERENCE if reference is not None else None
            )
            child, child_ignored = strip_tree_specific(
                item, f"{path}[{index}]", reference_item
            )
            kept.append(child)
            ignored.extend(child_ignored)
        return kept, ignored
    return value, ignored


def differing_fields(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                differences.append(child_path)
            else:
                differences.extend(differing_fields(left[key], right[key], child_path))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [path]
        differences = []
        for index, (a, b) in enumerate(zip(left, right)):
            differences.extend(differing_fields(a, b, f"{path}[{index}]"))
        return differences
    return [] if left == right else [path]


def sequence_errors(prompt, output, eos_id, ranks=None,
                    position_vocab=8000, feature_offset=8000):
    errors = []
    if not isinstance(prompt, list) or not prompt or prompt[0] != eos_id:
        return ["prompt BOS sentinel differs from eos_id"]
    if any(type(token) is not int for token in prompt):
        return ["prompt contains a non-integer token"]
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


def pair_rate_summary(rows_by_pair, numerator, denominator, budget, pair_ids=None,
                      full_only=False):
    allowed = None if pair_ids is None else set(pair_ids)
    per_source = defaultdict(list)
    pairs = 0
    numerator_tokens = denominator_tokens = 0
    numerator_seconds = denominator_seconds = 0.0
    for pair_id, arms in rows_by_pair.items():
        if allowed is not None and pair_id not in allowed:
            continue
        if numerator not in arms or denominator not in arms:
            continue
        up, down = arms[numerator], arms[denominator]
        if full_only and (up["generated_tokens"] != budget or down["generated_tokens"] != budget):
            continue
        if up["generated_tokens"] <= 0 or down["generated_tokens"] <= 0:
            continue
        up_rate = up["generated_tokens"] / up["outer_wall_seconds"]
        down_rate = down["generated_tokens"] / down["outer_wall_seconds"]
        per_source[str(up["source"])].append(math.log(up_rate) - math.log(down_rate))
        pairs += 1
        numerator_tokens += up["generated_tokens"]
        denominator_tokens += down["generated_tokens"]
        numerator_seconds += up["outer_wall_seconds"]
        denominator_seconds += down["outer_wall_seconds"]

    source_logs = {
        source: math.fsum(values) / len(values)
        for source, values in per_source.items()
    }
    sources = sorted(source_logs)
    result = {
        "numerator": numerator,
        "denominator": denominator,
        "pair_count": pairs,
        "source_count": len(sources),
        "numerator_generated_tokens": numerator_tokens,
        "denominator_generated_tokens": denominator_tokens,
        "numerator_outer_wall_seconds": numerator_seconds,
        "denominator_outer_wall_seconds": denominator_seconds,
        "source_equal_geometric_mean": None,
        "source_bootstrap_95_ci": None,
        "per_source": {},
    }
    if not sources:
        return result
    result["source_equal_geometric_mean"] = math.exp(
        math.fsum(source_logs.values()) / len(sources)
    )
    result["per_source"] = {source: math.exp(source_logs[source]) for source in sources}
    result["source_bootstrap_seed"] = BOOTSTRAP_SEED
    result["source_bootstrap_replicates"] = BOOTSTRAP_DRAWS
    if len(sources) > 1:
        rng = random.Random(BOOTSTRAP_SEED)
        samples = []
        for _ in range(BOOTSTRAP_DRAWS):
            sampled = [source_logs[rng.choice(sources)] for _ in sources]
            samples.append(math.exp(math.fsum(sampled) / len(sources)))
        result["source_bootstrap_95_ci"] = [
            quantile(samples, 0.025), quantile(samples, 0.975)
        ]
    return result


def ratio_pair(rows_by_pair, numerator, denominator, budget, pair_ids=None):
    return {
        "all_output_lengths": pair_rate_summary(
            rows_by_pair, numerator, denominator, budget, pair_ids=pair_ids
        ),
        "common_full_budget": pair_rate_summary(
            rows_by_pair, numerator, denominator, budget,
            pair_ids=pair_ids, full_only=True,
        ),
    }


def summarize_arm(chosen):
    total_tokens = sum(int(row["generated_tokens"]) for row in chosen)
    total_seconds = math.fsum(float(row["outer_wall_seconds"]) for row in chosen)
    lengths = Counter(str(int(row["generated_tokens"])) for row in chosen)
    integer_counters = Counter()
    for row in chosen:
        for key, value in (row.get("counters") or {}).items():
            if type(value) is int and value >= 0:
                integer_counters[str(key)] += value
    return {
        "runs": len(chosen),
        "generated_tokens": total_tokens,
        "outer_wall_seconds": total_seconds,
        "pooled_tokens_per_second": total_tokens / total_seconds if total_seconds else None,
        "eos_count": sum(bool(row.get("eos")) for row in chosen),
        "output_lengths": dict(sorted(lengths.items(), key=lambda item: int(item[0]))),
        "top_level_integer_counter_totals": dict(sorted(integer_counters.items())),
    }


def summarize_metric(rows, aliases):
    total = 0.0
    observed = 0
    sources = Counter()
    per_run = []
    for row in rows:
        value, source = metric_for_row(row, aliases)
        if value is None:
            per_run.append(None)
            continue
        total += value
        observed += 1
        sources[str(source)] += 1
        per_run.append(value)
    return {
        "total": total if observed else None,
        "observed_rows": observed,
        "rows": len(rows),
        "mean_per_observed_row": total / observed if observed else None,
        "source_fields": dict(sorted(sources.items())),
        "per_run_values": per_run,
    }


def sum_calls(chosen, delta_key):
    totals = Counter()
    for row in chosen:
        delta = row.get(delta_key)
        calls = delta.get("calls") if isinstance(delta, dict) else None
        if isinstance(calls, dict):
            for key, value in calls.items():
                if type(value) in (int, float) and value >= 0:
                    totals[str(key)] += value
    return dict(sorted(totals.items()))


def graph_cost(events):
    totals = {key: 0.0 for key in ("prepare_ms", "warmup_ms", "capture_ms")}
    counts = {key: 0 for key in totals}
    for event in events:
        if not isinstance(event, dict):
            continue
        for key in totals:
            value = event.get(key)
            if finite_number(value) and value >= 0:
                totals[key] += float(value)
                counts[key] += 1
    return {key: {"sum_ms": totals[key], "event_values": counts[key]} for key in totals}


def summarize_graph_runtime(rows_by_arm):
    graph_summary = {}
    reuse_summary = {}
    first_requests = {}
    for arm, chosen in rows_by_arm.items():
        if arm not in ARMS:
            continue
        branch_delta_key = BRANCH_DELTA_KEY.get(arm)
        target_call_pairs = []
        branch_call_pairs = []
        target_events = []
        branch_events = []
        created_rows = []
        for row in chosen:
            if row.get("runtime_created") is True:
                created_rows.append(row)
            target_delta = row.get("runtime_delta")
            if graph_call_keys(target_delta):
                target_call_pairs.append(str(row.get("pair_id")))
            target_events.extend(event_list(target_delta) or [])
            if branch_delta_key is not None:
                branch_delta = row.get(branch_delta_key)
                if graph_call_keys(branch_delta):
                    branch_call_pairs.append(str(row.get("pair_id")))
                branch_events.extend(event_list(branch_delta) or [])
        graph_summary[arm] = {
            "rows": len(chosen),
            "runtime_created_pair_ids": [str(row.get("pair_id")) for row in created_rows],
            "target_graph_call_pair_count": len(set(target_call_pairs)),
            "target_graph_call_keys": sorted({
                key for row in chosen for key in graph_call_keys(row.get("runtime_delta"))
            }),
            "target_new_graph_event_count": len(target_events),
            "target_graph_event_costs": graph_cost(target_events),
            "branch_runtime_delta": branch_delta_key,
            "branch_graph_call_pair_count": len(set(branch_call_pairs)),
            "branch_graph_call_keys": sorted({
                key for row in chosen for key in graph_call_keys(row.get(branch_delta_key))
            }) if branch_delta_key else [],
            "branch_new_graph_event_count": len(branch_events),
            "branch_graph_event_costs": graph_cost(branch_events),
        }
        if branch_delta_key is None:
            continue

        branch_reuse_ids = []
        all_runtime_reuse_ids = []
        branch_first_row = None
        branch_first_events = []
        for row in chosen:
            branch_delta = row.get(branch_delta_key)
            target_delta = row.get("runtime_delta")
            branch_used = bool(graph_call_keys(branch_delta))
            if branch_used and no_new_events(branch_delta):
                branch_reuse_ids.append(str(row.get("pair_id")))
            stable = (
                branch_used
                and row.get("runtime_created") is not True
                and no_new_events(branch_delta)
                and no_new_events(target_delta)
            )
            if stable:
                all_runtime_reuse_ids.append(str(row.get("pair_id")))
            added = event_list(branch_delta) or []
            if branch_first_row is None and (added or event_added(branch_delta)):
                branch_first_row = row
                branch_first_events = added

        if branch_first_row is None and created_rows:
            branch_first_row = created_rows[0]
            branch_first_events = event_list(branch_first_row.get(branch_delta_key)) or []
        if branch_first_row is not None:
            first_target_events = event_list(branch_first_row.get("runtime_delta")) or []
            first_requests[arm] = {
                "pair_id": branch_first_row.get("pair_id"),
                "runtime_created": branch_first_row.get("runtime_created"),
                "outer_wall_seconds": branch_first_row.get("outer_wall_seconds"),
                "generated_tokens": branch_first_row.get("generated_tokens"),
                "target_graph_event_count": len(first_target_events),
                "branch_graph_event_count": len(branch_first_events),
                "target_graph_setup_costs": graph_cost(first_target_events),
                "branch_graph_setup_costs": graph_cost(branch_first_events),
            }

        reuse_summary[arm] = {
            "branch_graph_used_without_new_branch_graph_pair_count": len(set(branch_reuse_ids)),
            "branch_graph_reuse_pair_ids": sorted(set(branch_reuse_ids)),
            "all_runtime_graphs_used_without_new_graph_pair_count": len(set(all_runtime_reuse_ids)),
            "all_runtime_graphs_reuse_pair_ids": sorted(set(all_runtime_reuse_ids)),
        }
    return graph_summary, reuse_summary, first_requests


def analyze_loaded(plan, prompts_doc, rows, pipeline, requested_budget=None):
    errors = []
    warnings = []
    if not isinstance(plan, dict):
        raise InputDataError("plan.json is not an object")
    prompts = prompts_doc.get("contexts") if isinstance(prompts_doc, dict) else prompts_doc
    if not isinstance(prompts, list):
        raise InputDataError("prompts.json lacks a contexts list")
    if not isinstance(pipeline, dict) or pipeline.get("status") != "complete":
        raise InputDataError("pipeline.json status is not complete")
    budget = int(plan.get("budget", 0))
    if budget < 2 or budget % 2:
        raise InputDataError("plan.budget must be a positive even integer")
    if requested_budget is not None and int(requested_budget) != budget:
        raise InputDataError(f"--budget {requested_budget} differs from plan budget {budget}")
    prompt_count = len(prompts)
    expected_runs = int(plan.get("expected_runs", -1))
    if int(plan.get("prompt_count", -1)) != prompt_count:
        errors.append("plan.prompt_count differs from prompts.json")
    if expected_runs != prompt_count * len(ARMS):
        errors.append("plan.expected_runs differs from four planned arms per prompt")
    if pipeline.get("completed_runs") != expected_runs or len(rows) != expected_runs:
        errors.append("pipeline or records count differs from plan.expected_runs")

    eos_from_prompt = prompts_doc.get("eos_id") if isinstance(prompts_doc, dict) else None
    eos_id = int(plan.get("eos_id", eos_from_prompt if eos_from_prompt is not None else -1))
    if eos_id < 0:
        raise InputDataError("plan or prompts lacks eos_id")
    if eos_from_prompt is not None and int(eos_from_prompt) != eos_id:
        errors.append("plan.eos_id differs from prompts.eos_id")
    position_vocab = int(plan.get("position_vocab_size", plan.get("p_vocab", 8000)))
    feature_offset = int(plan.get("feature_token_offset", plan.get("feature_offset", 8000)))
    ranks = plan.get("position_rank_map")
    if not isinstance(ranks, list):
        ranks = None

    prompt_by_id = {}
    for prompt in prompts:
        if not isinstance(prompt, dict):
            errors.append("prompt entry is not an object")
            continue
        pair_id = str(prompt.get("pair_id", ""))
        if not pair_id or pair_id in prompt_by_id:
            errors.append(f"prompt pair_id is empty or repeated: {pair_id!r}")
            continue
        prompt_by_id[pair_id] = prompt
        tokens = prompt.get("prompt")
        if not isinstance(tokens, list) or not tokens or tokens[0] != eos_id:
            errors.append(f"{pair_id}: prompt BOS sentinel differs from eos_id")
        elif any(type(token) is not int for token in tokens):
            errors.append(f"{pair_id}: prompt contains a non-integer token")
        elif (len(tokens) - 1) % 2:
            errors.append(f"{pair_id}: prompt does not end on a complete P/F pair")

    planned_sources = plan.get("sources")
    prompt_sources = {str(prompt.get("source")) for prompt in prompt_by_id.values()}
    if isinstance(planned_sources, list) and planned_sources:
        if {str(value) for value in planned_sources} != prompt_sources:
            errors.append("plan.sources differs from prompt sources")
    elif isinstance(planned_sources, int) and planned_sources != len(prompt_sources):
        errors.append("plan.sources count differs from prompt sources")

    rows_by_pair = defaultdict(dict)
    rows_by_arm = defaultdict(list)
    duplicate_rows = []
    unknown_rows = []
    valid_sequence_rows = 0
    for index, row in enumerate(rows):
        pair_id = str(row.get("pair_id", ""))
        arm = row.get("arm")
        if pair_id not in prompt_by_id or arm not in ARMS:
            unknown_rows.append({"index": index, "pair_id": pair_id, "arm": arm})
            continue
        expected_method, expected_execution = ARMS[arm]
        if row.get("method") != expected_method or row.get("execution") != expected_execution:
            errors.append(f"record {index}: method/execution differs from arm {arm}")
        if arm in rows_by_pair[pair_id]:
            duplicate_rows.append({"pair_id": pair_id, "arm": arm})
        rows_by_pair[pair_id][arm] = row
        rows_by_arm[arm].append(row)
        prompt = prompt_by_id[pair_id]
        for key in ("source", "prefix", "seed"):
            if key in prompt and row.get(key) != prompt.get(key):
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
            rate = row.get("tokens_per_second")
            if not finite_number(rate) or not math.isclose(
                float(rate), len(output) / float(seconds), rel_tol=1e-10, abs_tol=1e-10
            ):
                errors.append(f"record {index}: tokens_per_second differs from raw output length/time")
        output_has_eos = bool(output and output[-1] == eos_id)
        if type(row.get("eos")) is not bool or row.get("eos") != output_has_eos:
            errors.append(f"record {index}: eos flag differs from output")
        if row.get("sequence_validated") is False:
            errors.append(f"record {index}: sequence_validated is false")
        prompt_tokens = prompt.get("prompt")
        if output and isinstance(prompt_tokens, list):
            seq_errors = sequence_errors(
                prompt_tokens, output, eos_id, ranks=ranks,
                position_vocab=position_vocab, feature_offset=feature_offset,
            )
            if not seq_errors:
                valid_sequence_rows += 1
            errors.extend(f"record {index}: {message}" for message in seq_errors)
        if not isinstance(row.get("counters"), dict):
            errors.append(f"record {index}: counters is missing or invalid")
        if not isinstance(row.get("stats"), dict):
            warnings.append(f"record {index}: stats is absent or not an object")
        if not isinstance(row.get("runtime_delta"), dict):
            warnings.append(f"record {index}: runtime_delta is absent or not an object")
        if arm in BRANCH_DELTA_KEY and not isinstance(row.get(BRANCH_DELTA_KEY[arm]), dict):
            errors.append(f"record {index}: {BRANCH_DELTA_KEY[arm]} is missing or invalid")

    expected_pair_ids = set(prompt_by_id)
    expected_arm_set = set(ARMS)
    missing_arms = []
    extra_arms = []
    for pair_id in sorted(expected_pair_ids):
        present = set(rows_by_pair.get(pair_id, {}))
        if expected_arm_set - present:
            missing_arms.append({"pair_id": pair_id, "arms": sorted(expected_arm_set - present)})
        if present - expected_arm_set:
            extra_arms.append({"pair_id": pair_id, "arms": sorted(present - expected_arm_set)})
    if duplicate_rows:
        errors.append(f"duplicate pair/arm rows: {len(duplicate_rows)}")
    if unknown_rows:
        errors.append(f"unknown rows: {len(unknown_rows)}")
    if missing_arms:
        errors.append(f"pairs with missing arms: {len(missing_arms)}")
    if extra_arms:
        errors.append(f"pairs with unplanned arms: {len(extra_arms)}")

    w1_parity_pairs = []
    for pair_id in sorted(expected_pair_ids):
        arms = rows_by_pair.get(pair_id, {})
        mlp = arms.get("proposal_graph:pf_mlp")
        w1 = arms.get("tree_graph:tree_w1")
        if mlp is None or w1 is None:
            continue
        output_equal = mlp.get("output_tokens") == w1.get("output_tokens")
        mlp_counters, mlp_tree_fields = strip_tree_specific(mlp.get("counters") or {})
        w1_counters, w1_tree_fields = strip_tree_specific(
            w1.get("counters") or {}, reference=mlp_counters
        )
        counter_equal = mlp_counters == w1_counters
        item = {
            "pair_id": pair_id,
            "source": mlp.get("source"),
            "seed": mlp.get("seed"),
            "output_equal": output_equal,
            "shared_counters_equal": counter_equal,
            "generated_tokens": [mlp.get("generated_tokens"), w1.get("generated_tokens")],
            "tree_specific_counter_fields_ignored": sorted(set(mlp_tree_fields + w1_tree_fields)),
        }
        if not output_equal:
            item["first_output_difference"] = output_difference(mlp, w1)
        if not counter_equal:
            item["shared_counter_difference_fields"] = differing_fields(mlp_counters, w1_counters)[:100]
            errors.append(f"{pair_id}: W1 output counters differ from PF-MLP shared counters")
        if not output_equal:
            errors.append(f"{pair_id}: W1 output differs from PF-MLP under the same prompt and seed")
        w1_parity_pairs.append(item)
    if len(w1_parity_pairs) != len(prompt_by_id):
        errors.append("W1/PF-MLP parity rows do not cover all prompt pairs")

    arm_summaries = {
        arm: summarize_arm(chosen)
        for arm, chosen in sorted(rows_by_arm.items())
    }
    comparisons = {}
    main_arms = (
        ("proposal_graph:pf_mlp", "PF-MLP / 原始 Target"),
        ("tree_graph:tree_w1", "W1 / 原始 Target"),
        ("tree_graph:tree_w2", "W2 / 原始 Target"),
    )
    for numerator, _ in main_arms:
        comparisons[numerator.replace(":", "_") + "_over_original_target"] = ratio_pair(
            rows_by_pair, numerator, "original_target:target", budget
        )
    comparisons["tree_w2_over_pf_mlp"] = ratio_pair(
        rows_by_pair, "tree_graph:tree_w2", "proposal_graph:pf_mlp", budget
    )
    comparisons["tree_w2_over_tree_w1"] = ratio_pair(
        rows_by_pair, "tree_graph:tree_w2", "tree_graph:tree_w1", budget
    )

    method_metrics = {}
    for arm, chosen in sorted(rows_by_arm.items()):
        if arm == "original_target:target":
            continue
        metrics = {
            name: summarize_metric(chosen, aliases)
            for name, aliases in METRIC_ALIASES.items()
        }
        attempts = metrics["feature_attempts"]["total"]
        accepted = metrics["feature_accepted"]["total"]
        metrics["feature_acceptance_rate"] = (
            accepted / attempts if attempts not in (None, 0) and accepted is not None else None
        )
        metrics["target_runtime_calls"] = sum_calls(chosen, "runtime_delta")
        metrics["branch_runtime_calls"] = sum_calls(
            chosen, BRANCH_DELTA_KEY.get(arm, "tree_runtime_delta")
        )
        method_metrics[arm] = metrics

    graph_summary, reuse_summary, first_requests = summarize_graph_runtime(rows_by_arm)
    reuse_comparisons = {}
    for arm, info in reuse_summary.items():
        label = arm.replace(":", "_")
        reuse_comparisons[label] = {
            "branch_graph_reuse_without_new_branch_events_vs_original": ratio_pair(
                rows_by_pair, arm, "original_target:target", budget,
                pair_ids=info["branch_graph_reuse_pair_ids"],
            ),
            "all_runtime_graphs_reuse_without_new_events_vs_original": ratio_pair(
                rows_by_pair, arm, "original_target:target", budget,
                pair_ids=info["all_runtime_graphs_reuse_pair_ids"],
            ),
        }

    comparison_errors = []
    for key, value in comparisons.items():
        if value["all_output_lengths"]["source_equal_geometric_mean"] is None:
            comparison_errors.append(f"{key}: no paired throughput result")
    errors.extend(comparison_errors)
    return {
        "status": "complete" if not errors else "failed",
        "run_dir": None,
        "budget": budget,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_DRAWS,
            "unit": "source",
            "aggregation": "equal-source average of per-pair log token-rate ratios",
        },
        "plan": {
            "methods": list(METHODS),
            "prompt_count": prompt_count,
            "expected_runs": expected_runs,
            "actual_runs": len(rows),
            "source_count": len(prompt_sources),
            "source_ids": sorted(prompt_sources),
            "prefixes": sorted({prompt.get("prefix") for prompt in prompt_by_id.values()
                                 if isinstance(prompt.get("prefix"), int)}),
            "seeds": sorted({prompt.get("seed") for prompt in prompt_by_id.values()
                              if isinstance(prompt.get("seed"), int)}),
            "timing_scope": plan.get("timing_scope"),
            "execution_meaning": plan.get("execution_meaning"),
        },
        "arm_summaries": arm_summaries,
        "comparisons": comparisons,
        "tree_w1_vs_pf_mlp_parity": {
            "pair_count": len(w1_parity_pairs),
            "output_equal_count": sum(item["output_equal"] for item in w1_parity_pairs),
            "shared_counters_equal_count": sum(
                item["shared_counters_equal"] for item in w1_parity_pairs
            ),
            "pairs": w1_parity_pairs,
        },
        "method_counters": method_metrics,
        "graph_runtime": graph_summary,
        "first_graph_request": first_requests,
        "graph_reuse_subsets": reuse_summary,
        "graph_reuse_comparisons_vs_original_target": reuse_comparisons,
        "integrity": {
            "errors": errors,
            "warnings": warnings,
            "duplicate_rows": duplicate_rows,
            "unknown_rows": unknown_rows,
            "missing_arms": missing_arms,
            "extra_arms": extra_arms,
            "records_with_valid_sequences": valid_sequence_rows,
        },
        "interpretation": {
            "main_ratio": "candidate arm throughput divided by original_target:target throughput",
            "internal_ratios": "tree_graph:tree_w2 divided by proposal_graph:pf_mlp and tree_graph:tree_w1",
            "feature_attempts": "Uses the first available verified/checked/attempted counter; proposals are reported separately.",
            "tree_node_metric": "Uses the first available node-count alias and records its source field.",
            "length_policy": "Generated lengths, EOS and synchronized outer wall are retained as recorded.",
            "ratio_direction": "Values above 1 indicate higher generated-token throughput in the numerator arm.",
        },
    }


def analyze_run(run_dir, requested_budget=None):
    plan = read_json(run_dir / "plan.json")
    prompts = read_json(run_dir / "prompts.json")
    rows = read_records(run_dir / "records.jsonl")
    pipeline = read_json(run_dir / "pipeline.json")
    summary = analyze_loaded(plan, prompts, rows, pipeline, requested_budget)
    summary["run_dir"] = str(run_dir.resolve())
    return summary


def display_ratio(item):
    estimate = item.get("source_equal_geometric_mean")
    if estimate is None:
        return "无配对数据"
    interval = item.get("source_bootstrap_95_ci")
    if interval is None:
        ci_text = "单来源"
    else:
        ci_text = f"95%来源区间 [{interval[0]:.3f}, {interval[1]:.3f}]"
    return f"{estimate:.3f}×；{ci_text}；{item['pair_count']} 对 / {item['source_count']} 来源"


def results_markdown(summary):
    lines = [
        "# Feature 树分支运行结果",
        "",
        f"状态：{summary['status']}；记录 {summary['plan']['actual_runs']}/"
        f"{summary['plan']['expected_runs']} 条，{summary['plan']['source_count']} 个来源，"
        f"预算 {summary['budget']} 个输出 token。",
        "",
        "倍率采用每条记录的实际 token/s。每个来源内先平均配对 log 倍率，再对来源等权取几何平均；区间按来源重抽 10,000 次。",
        "",
        "## 主结果",
        "",
        "| 方法 | 相对原始 Target | 双臂均达预算 | 候选生成 token | 原始 Target 生成 token |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, label in (
        ("proposal_graph:pf_mlp", "PF-MLP"),
        ("tree_graph:tree_w1", "W1"),
        ("tree_graph:tree_w2", "W2"),
    ):
        key = arm.replace(":", "_") + "_over_original_target"
        value = summary["comparisons"].get(key)
        if not value:
            continue
        all_rows = value["all_output_lengths"]
        full_rows = value["common_full_budget"]
        lines.append(
            f"| {label} | {display_ratio(all_rows)} | {display_ratio(full_rows)} | "
            f"{all_rows['numerator_generated_tokens']} | {all_rows['denominator_generated_tokens']} |"
        )
    lines += [
        "",
        "## W2 内部对照",
        "",
        f"W2 / PF-MLP：{display_ratio(summary['comparisons']['tree_w2_over_pf_mlp']['all_output_lengths'])}。",
        f"W2 / W1：{display_ratio(summary['comparisons']['tree_w2_over_tree_w1']['all_output_lengths'])}。",
        "",
    ]
    w1_parity = summary["tree_w1_vs_pf_mlp_parity"]
    lines.append(
        f"W1 与 PF-MLP 同 pair、同 seed 比较 {w1_parity['pair_count']} 对："
        f"输出相同 {w1_parity['output_equal_count']} 对，共有计数相同 "
        f"{w1_parity['shared_counters_equal_count']} 对；树专用字段单列。"
    )
    lines += [
        "",
        "## 节点、接受与 Target 调用",
        "",
        "| 方法 | 候选树节点 | Target 树输入节点 | Feature 提案 | Feature 验证尝试 | Feature 接受 | 接受比例 | Target 调用 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, label in (
        ("proposal_graph:pf_mlp", "PF-MLP"),
        ("tree_graph:tree_w1", "W1"),
        ("tree_graph:tree_w2", "W2"),
    ):
        metrics = summary["method_counters"].get(arm, {})
        values = [metrics.get(name, {}).get("total") for name in (
            "candidate_tree_nodes", "target_tree_nodes", "feature_proposals", "feature_attempts",
            "feature_accepted", "target_calls",
        )]
        rate = metrics.get("feature_acceptance_rate")
        node_text = values[0] if values[0] is not None else "未提供"
        target_nodes_text = values[1] if values[1] is not None else "未提供"
        proposal_text = values[2] if values[2] is not None else "未提供"
        attempt_text = values[3] if values[3] is not None else "未提供"
        accepted_text = values[4] if values[4] is not None else "未提供"
        target_text = values[5] if values[5] is not None else "未提供"
        rate_text = f"{rate:.3f}" if rate is not None else "未提供"
        lines.append(
            f"| {label} | {node_text} | {target_nodes_text} | {proposal_text} | {attempt_text} | "
            f"{accepted_text} | {rate_text} | {target_text} |"
        )
    lines += ["", "计数命中的原始字段路径保存在 `summary.json` 的 `method_counters` 中。", "",
              "## 图运行时与首次成本", ""]
    for arm, info in summary["graph_runtime"].items():
        first = summary["first_graph_request"].get(arm, {})
        lines.append(
            f"- `{arm}`：Target 图调用 {info['target_graph_call_pair_count']}/{info['rows']} 对，"
            f"提案图调用 {info['branch_graph_call_pair_count']}/{info['rows']} 对；"
            f"首次图请求 {first.get('pair_id', '无记录')}，外层时长 "
            f"{first.get('outer_wall_seconds', 0):.3f} 秒，Target 新图事件 "
            f"{first.get('target_graph_event_count', 0)} 个，提案新图事件 "
            f"{first.get('branch_graph_event_count', 0)} 个。"
        )
    lines += ["", "## 无新增图子集", ""]
    for arm, info in summary["graph_reuse_comparisons_vs_original_target"].items():
        branch = info["branch_graph_reuse_without_new_branch_events_vs_original"]["all_output_lengths"]
        stable = info["all_runtime_graphs_reuse_without_new_events_vs_original"]["all_output_lengths"]
        lines.append(f"- `{arm}`：提案图已使用且该行无新增提案图事件：{display_ratio(branch)}；"
                     f"Target 与提案图均无新增事件：{display_ratio(stable)}。")
    lines += ["", f"独立数据状态：{summary['status']}。"]
    if summary["integrity"]["errors"]:
        lines += ["", "数据问题："]
        lines.extend(f"- {item}" for item in summary["integrity"]["errors"])
    if summary["integrity"]["warnings"]:
        lines += ["", "数据说明："]
        lines.extend(f"- {item}" for item in summary["integrity"]["warnings"])
    return "\n".join(lines) + "\n"


def write_outputs(summary, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    markdown_path = output_dir / "RESULTS.md"
    existing = [str(path) for path in (json_path, markdown_path) if path.exists()]
    if existing:
        raise InputDataError("Refusing to overwrite existing output: " + ", ".join(existing))
    json_tmp = json_path.with_suffix(".json.tmp")
    markdown_tmp = markdown_path.with_suffix(".md.tmp")
    json_tmp.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_tmp.write_text(results_markdown(summary), encoding="utf-8")
    json_tmp.replace(json_path)
    markdown_tmp.replace(markdown_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed tree experiment output directory")
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
        "row_count": summary["plan"]["actual_runs"],
        "error_count": len(summary["integrity"]["errors"]),
    }, ensure_ascii=False))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
