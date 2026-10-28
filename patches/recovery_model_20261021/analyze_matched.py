#!/usr/bin/env python3
"""Analyze matched K1/W1/W2 action records with source-held-out CPU models."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ACTIONS = ("K1", "W1", "W2")
ACTION_WIDTH = {"K1": 1, "W1": 1, "W2": 2}
ACTION_NODES = {"K1": 0, "W1": 4, "W2": 6}
BASE_FEATURES = (
    "log_prefix_len", "cache_pos_before", "remaining_budget_before",
    "action_budget_tokens", "log_read_bucket", "node_count",
    "last_position_rank_norm", "prior_position_rank_norm",
    "last_gap_rank_norm", "position_context_missing",
)
NODE_FEATURES = (
    "log_prefix_len", "cache_pos_before", "remaining_budget_before",
    "action_budget_tokens", "log_read_bucket", "node_count",
    "last_position_rank_norm", "prior_position_rank_norm",
    "last_gap_rank_norm", "position_context_missing",
)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "1", "yes"}:
            return True
        if low in {"false", "0", "no", ""}:
            return False
        return None
    return bool(value)


def _first(mapping: dict[str, Any], names: tuple[str, ...], default=None):
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _component_map(row: dict[str, Any], key: str) -> dict[str, float]:
    value = row.get(key)
    if not isinstance(value, dict):
        for container_name in ("round", "probe_round", "stats"):
            container = row.get(container_name)
            if isinstance(container, dict) and isinstance(container.get(key), dict):
                value = container[key]
                break
    if not isinstance(value, dict):
        return {}
    result = {}
    for name, item in value.items():
        number = _number(item)
        if number is not None:
            result[str(name)] = number
    return result


def _graph_event_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    number = _number(value)
    return max(0, int(number or 0))


def normalize_row(row: dict[str, Any], line_number: int,
                  prompt_features: dict[str, float] | None = None) -> dict[str, Any]:
    scope = str(row.get("scope", "single_round"))
    action = str(_first(row, ("action_id", "action"), "")).upper()
    if action not in ACTIONS:
        raise ValueError(f"line {line_number}: unknown action {action!r}")
    source = _first(row, ("source_id", "source", "scene"))
    if source is None:
        raise ValueError(f"line {line_number}: source_id is required")
    prefix_len = _number(_first(row, ("prefix_len", "prompt_tokens")))
    if prefix_len is None:
        raise ValueError(f"line {line_number}: prefix_len is required")
    context = _first(row, ("context_id", "crop_id"))
    if context is None:
        crop_path = _first(row, ("crop_path", "source_path"), "")
        context = f"{source}|{int(prefix_len)}|{crop_path}"
    seed = _first(row, ("seed", "decode_seed"))
    if seed is None:
        raise ValueError(f"line {line_number}: seed is required")
    repeat = int(_number(row.get("repeat_index")) or 0)

    if scope == "single_round":
        progress = _number(_first(row, (
            "R", "generated_tokens", "round_generated_tokens", "actual_tokens_count",
        )))
        cost = _number(row.get("round_wall_ms"))
    elif scope == "full_request":
        progress = _number(_first(row, (
            "R", "generated_tokens", "actual_tokens_count",
        )))
        cost = _number(_first(row, ("full_request_wall_ms", "decode_outer_ms")))
    else:
        raise ValueError(f"line {line_number}: invalid scope {scope!r}")
    if progress is None:
        raise ValueError(f"line {line_number}: R/generated_tokens is required")

    action_budget = _number(row.get("action_budget_tokens"))
    if action_budget is None:
        action_budget = 2.0 if scope == "single_round" and action == "K1" else (
            4.0 if scope == "single_round" else 512.0
        )
    planned_nodes = float(ACTION_NODES[action])
    planned_read_bucket = 0.0
    if planned_nodes > 0:
        planned_read = max(1, int(prefix_len) + int(planned_nodes))
        planned_read_bucket = float(max(8, 1 << (planned_read - 1).bit_length()))
    prompt_features = prompt_features or {}
    feature_dict = {
        "log_prefix_len": math.log1p(max(0.0, prefix_len)),
        "cache_pos_before": float(prefix_len),
        "remaining_budget_before": float(action_budget),
        "action_budget_tokens": float(action_budget),
        "log_read_bucket": math.log1p(planned_read_bucket),
        "node_count": planned_nodes,
        "last_position_rank_norm": float(prompt_features.get("last_position_rank_norm", 0.0)),
        "prior_position_rank_norm": float(prompt_features.get("prior_position_rank_norm", 0.0)),
        "last_gap_rank_norm": float(prompt_features.get("last_gap_rank_norm", 0.0)),
        "position_context_missing": float(prompt_features.get("position_context_missing", 1.0)),
        "action_K1": 1.0 if action == "K1" else 0.0,
        "action_W1": 1.0 if action == "W1" else 0.0,
        "action_W2": 1.0 if action == "W2" else 0.0,
    }
    for name in ACTIONS:
        indicator = 1.0 if name == action else 0.0
        feature_dict[f"{name}_x_prefix"] = indicator * feature_dict["log_prefix_len"]
        feature_dict[f"{name}_x_last_gap"] = indicator * feature_dict["last_gap_rank_norm"]
        feature_dict[f"{name}_x_read_bucket"] = indicator * feature_dict["log_read_bucket"]
        feature_dict[f"{name}_x_node_count"] = indicator * feature_dict["node_count"]
    for name, value in prompt_features.items():
        if name.startswith("known_last_feature_"):
            feature_dict[name] = float(value)

    eos = _as_bool(row.get("eos"))
    accepted = []
    for key in ("accepted_p1", "accepted_feature", "accepted_p2"):
        value = _as_bool(row.get(key))
        if value is not None:
            accepted.append(value)
    if scope == "full_request" and isinstance(row.get("accepted_lengths"), list):
        accepted_count = float(sum(_number(value) or 0.0 for value in row["accepted_lengths"]))
    elif not accepted:
        raw_accepted = _number(row.get("accepted_count"))
        if raw_accepted is None:
            raw_accepted = _number(row.get("accepted"))
        if raw_accepted is not None:
            accepted_count = raw_accepted
        elif "proposed_p1" in row and row.get("proposed_p1") is None:
            accepted_count = 0.0
        else:
            accepted_count = None
    else:
        accepted_count = float(sum(accepted))

    recovery_calls = _number(_first(row, (
        "recovery_target_calls", "recovery_calls",
    )))
    recovery_inputs = _number(_first(row, (
        "recovery_target_input_tokens", "recovery_input_tokens",
    )))
    if recovery_calls is None and isinstance(row.get("round"), dict):
        recovery_calls = _number(_first(row["round"], (
            "recovery_target_calls", "recovery_calls",
        )))
    if recovery_inputs is None and isinstance(row.get("round"), dict):
        recovery_inputs = _number(_first(row["round"], (
            "recovery_target_input_tokens", "recovery_input_tokens",
        )))

    graph_count = _graph_event_count(_first(row, ("new_graph_events",), 0))
    cold_graph_ms = _number(row.get("cold_graph_ms")) or 0.0
    explicit_cold = _as_bool(row.get("cold"))
    if explicit_cold is None and row.get("cold_status") in {"cold", "warm"}:
        explicit_cold = row.get("cold_status") == "cold"
    if explicit_cold is not None:
        thermal = "cold" if explicit_cold else "warm"
    else:
        thermal = "cold" if graph_count > 0 or cold_graph_ms > 0 else "warm"
        if scope == "full_request" and _as_bool(row.get("runtime_created")):
            thermal = "cold"

    cpu_components = _component_map(row, "component_cpu_ms")
    device_components = _component_map(row, "component_device_ms")
    output_tokens = _first(row, ("output_tokens", "suffix_tokens", "actual_tokens"))
    return {
        "line_number": line_number,
        "source": str(source),
        "context": str(context),
        "prefix_len": prefix_len,
        "seed": str(seed),
        "scope": scope,
        "action": action,
        "repeat_index": repeat,
        "thermal": thermal,
        "progress": progress,
        "cost_ms": cost,
        "eos": eos,
        "eos_stage": row.get("eos_stage"),
        "accepted_count": accepted_count,
        "recovery_calls": recovery_calls,
        "recovery_input_tokens": recovery_inputs,
        "target_calls": _number(row.get("target_calls")),
        "target_input_tokens": _number(row.get("target_input_tokens")),
        "draft_calls": _number(row.get("draft_calls")),
        "draft_input_tokens": _number(row.get("draft_input_tokens")),
        "ranker_calls": _number(row.get("ranker_calls")),
        "ranker_p1_candidate_count": _number(row.get("ranker_p1_candidate_count")),
        "features": feature_dict,
        "request_budget_tokens": float(action_budget),
        "actual_node_count": _number(row.get("node_count")),
        "actual_read_bucket": _number(_first(row, ("tree_read_bucket", "read_bucket"))),
        "cpu_components": cpu_components,
        "device_components": device_components,
        "graph_events_new": graph_count,
        "runtime_created": _as_bool(row.get("runtime_created")),
        "setup_ms": _number(row.get("setup_ms")),
        "prefill_ms": _number(row.get("prefill_ms")),
        "decode_outer_ms": _number(row.get("decode_outer_ms")),
        "arm_constructor_ms": _number(row.get("arm_constructor_ms")),
        "outer_wall_including_arm_constructor_ms": _number(row.get("outer_wall_including_arm_constructor_ms")),
        "cold_graph_ms": cold_graph_ms,
        "output_tokens": output_tokens,
    }


def _read_ranks(path: Path | None) -> tuple[list[int] | None, str | None]:
    if path is None:
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload
    if isinstance(payload, dict):
        values = _first(payload, ("position_rank_map", "ranks", "sparse_position_rank_map"))
        if values is None:
            nested = payload.get("context")
            if isinstance(nested, dict):
                values = _first(nested, ("position_rank_map", "ranks", "sparse_position_rank_map"))
    if not isinstance(values, list):
        raise ValueError(f"rank input {path} must contain a rank list or position_rank_map")
    ranks = []
    for value in values:
        number = _number(value)
        if number is None:
            raise ValueError(f"rank input {path} contains a non-integer value")
        ranks.append(int(number))
    return ranks, str(path)


def _prompt_feature_map(paths: list[Path], prompts_path: Path | None,
                        ranks_path: Path | None) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    candidates = [prompts_path] if prompts_path is not None else [path.parent / "prompts.json" for path in paths]
    selected = next((path for path in candidates if path is not None and path.exists()), None)
    if selected is None:
        return {}, {"prompts_file": None, "rank_file": str(ranks_path) if ranks_path else None,
                    "contexts_with_prompts": 0, "position_context_missing": True}
    payload = json.loads(selected.read_text(encoding="utf-8"))
    contexts = payload.get("contexts", []) if isinstance(payload, dict) else payload
    if not isinstance(contexts, list):
        raise ValueError(f"prompt input {selected} must contain a contexts array")
    ranks, resolved_rank_path = _read_ranks(ranks_path)
    rank_scale = max(1, max((value for value in ranks if value >= 0), default=0)) if ranks is not None else 1
    output = {}
    missing_rank = 0
    for context in contexts:
        if not isinstance(context, dict) or context.get("context_id") is None:
            continue
        prompt = context.get("prompt")
        if not isinstance(prompt, list) or len(prompt) < 3:
            continue
        prefix_len = int(context.get("prefix_len", max(0, len(prompt) - 1)))
        # The prompt is [BOS, P0, F0, P1, F1, ...]. Positions and Features
        # are read only from the known prompt, never from generated output.
        position_tokens = [int(token) for token in prompt[1::2]]
        feature_tokens = [int(token) for token in prompt[2::2]]
        item = {
            "last_position_rank_norm": 0.0,
            "prior_position_rank_norm": 0.0,
            "last_gap_rank_norm": 0.0,
            "position_context_missing": 1.0,
        }
        if ranks is not None and len(position_tokens) >= 2:
            last_token, prior_token = position_tokens[-1], position_tokens[-2]
            if 0 <= last_token < len(ranks) and 0 <= prior_token < len(ranks):
                last_rank, prior_rank = ranks[last_token], ranks[prior_token]
                if last_rank >= 0 and prior_rank >= 0:
                    item.update(
                        last_position_rank_norm=last_rank / rank_scale,
                        prior_position_rank_norm=prior_rank / rank_scale,
                        last_gap_rank_norm=(last_rank - prior_rank) / rank_scale,
                        position_context_missing=0.0,
                    )
        if item["position_context_missing"]:
            missing_rank += 1
        if feature_tokens:
            # Keep Feature identity categorical; its integer ID has no numeric order.
            item[f"known_last_feature_{feature_tokens[-1]}"] = 1.0
        output[str(context["context_id"])] = item
    return output, {"prompts_file": str(selected), "rank_file": resolved_rank_path,
                    "contexts_with_prompts": len(output), "position_context_missing": missing_rank,
                    "feature_identity": "one-hot category for the last Feature token in the known prompt"}


def read_jsonl(paths: list[Path], prompts_path: Path | None = None,
               ranks_path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    total_lines = 0
    prompt_features, prompt_info = _prompt_feature_map(paths, prompts_path, ranks_path)
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                total_lines += 1
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
                value["_input_path"] = str(path)
                value["_line_number"] = line_number
                context_key = str(_first(value, ("context_id", "crop_id"), ""))
                rows.append(normalize_row(value, line_number, prompt_features.get(context_key)))
    return rows, {"input_files": [str(path) for path in paths], "rows": len(rows),
                  "nonblank_lines": total_lines, "prompt_features": prompt_info}


def _median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def aggregate_repeats(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["source"], row["context"], row["scope"], row["action"], row["seed"], row["thermal"])
        groups[key].append(row)
    aggregated = []
    warnings = []
    repeat_count_histogram = Counter()
    for key, group in groups.items():
        group = sorted(group, key=lambda r: r["repeat_index"])
        repeat_count_histogram[str(len(group))] += 1
        progresses = [r["progress"] for r in group]
        if max(progresses) - min(progresses) > 1e-9:
            warnings.append({"key": list(key), "kind": "progress_differs_across_timing_repeats", "values": progresses})
        eos_values = [r["eos"] for r in group if r["eos"] is not None]
        if len(set(eos_values)) > 1:
            warnings.append({"key": list(key), "kind": "eos_differs_across_timing_repeats", "values": eos_values})
        tokens = [json.dumps(r["output_tokens"], sort_keys=True) for r in group if r["output_tokens"] is not None]
        if len(set(tokens)) > 1:
            warnings.append({"key": list(key), "kind": "output_differs_across_timing_repeats"})

        def mean_field(field):
            values = [r[field] for r in group if r[field] is not None]
            return statistics.fmean(values) if values else None

        first = group[0]
        feature_names = set().union(*(r["features"].keys() for r in group))
        features = {name: statistics.fmean(r["features"].get(name, 0.0) for r in group)
                    for name in feature_names}
        component_names = set().union(*(r["device_components"].keys() for r in group))
        device_components = {
            name: _median_or_none([r["device_components"].get(name, 0.0) for r in group]) or 0.0
            for name in component_names
        }
        cpu_names = set().union(*(r["cpu_components"].keys() for r in group))
        cpu_components = {
            name: _median_or_none([r["cpu_components"].get(name, 0.0) for r in group]) or 0.0
            for name in cpu_names
        }
        aggregated.append({
            "source": first["source"], "context": first["context"],
            "prefix_len": statistics.fmean(r["prefix_len"] for r in group),
            "request_budget_tokens": statistics.fmean(r["request_budget_tokens"] for r in group),
            "scope": first["scope"], "action": first["action"],
            "seed": first["seed"], "thermal": first["thermal"],
            "progress": statistics.fmean(progresses),
            "cost_ms": _median_or_none([r["cost_ms"] for r in group if r["cost_ms"] is not None]),
            "eos": any(eos_values) if eos_values else None,
            "eos_stage": next((r["eos_stage"] for r in reversed(group) if r["eos_stage"]), None),
            "accepted_count": mean_field("accepted_count"),
            "recovery_calls": mean_field("recovery_calls"),
            "recovery_input_tokens": mean_field("recovery_input_tokens"),
            "target_calls": mean_field("target_calls"),
            "target_input_tokens": mean_field("target_input_tokens"),
            "draft_calls": mean_field("draft_calls"),
            "draft_input_tokens": mean_field("draft_input_tokens"),
            "ranker_calls": mean_field("ranker_calls"),
            "ranker_p1_candidate_count": mean_field("ranker_p1_candidate_count"),
            "features": features,
            "device_components": device_components,
            "cpu_components": cpu_components,
            "repeat_count": len(group),
            "repeat_costs_ms": [r["cost_ms"] for r in group if r["cost_ms"] is not None],
            "graph_events_new": sum(r["graph_events_new"] for r in group),
            "actual_node_count": _median_or_none([r["actual_node_count"] for r in group
                                                   if r["actual_node_count"] is not None]),
            "actual_read_bucket": _median_or_none([r["actual_read_bucket"] for r in group
                                                    if r["actual_read_bucket"] is not None]),
            "setup_ms": _median_or_none([r["setup_ms"] for r in group if r["setup_ms"] is not None]),
            "prefill_ms": _median_or_none([r["prefill_ms"] for r in group if r["prefill_ms"] is not None]),
            "decode_outer_ms": _median_or_none([r["decode_outer_ms"] for r in group if r["decode_outer_ms"] is not None]),
            "arm_constructor_ms": _median_or_none([r["arm_constructor_ms"] for r in group if r["arm_constructor_ms"] is not None]),
            "outer_wall_including_arm_constructor_ms": _median_or_none([
                r["outer_wall_including_arm_constructor_ms"] for r in group
                if r["outer_wall_including_arm_constructor_ms"] is not None
            ]),
            "cold_graph_ms": statistics.fmean(r["cold_graph_ms"] for r in group),
        })
    return aggregated, {"repeat_count_histogram": dict(repeat_count_histogram), "warnings": warnings}


def build_context_samples(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_context: dict[tuple, dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        key = (row["source"], row["context"], row["scope"], row["thermal"])
        by_context[key][row["action"]][row["seed"]] = row
    samples = []
    omitted = Counter()
    for (source, context, scope, thermal), actions in by_context.items():
        if any(action not in actions for action in ACTIONS):
            omitted["missing_action"] += 1
            continue
        common_seeds = set(actions[ACTIONS[0]])
        for action in ACTIONS[1:]:
            common_seeds.intersection_update(actions[action])
        if not common_seeds:
            omitted["no_common_seed"] += 1
            continue
        prefix_values = [actions[action][seed]["prefix_len"]
                         for action in ACTIONS for seed in common_seeds]
        for action in ACTIONS:
            seed_rows = [actions[action][seed] for seed in sorted(common_seeds)]
            if any(r["cost_ms"] is None or r["cost_ms"] <= 0 for r in seed_rows):
                omitted["missing_or_invalid_cost"] += 1
                break
        else:
            pass
        if any(any(r["cost_ms"] is None or r["cost_ms"] <= 0 for r in [actions[a][s]])
               for a in ACTIONS for s in common_seeds):
            continue
        for action in ACTIONS:
            seed_rows = [actions[action][seed] for seed in sorted(common_seeds)]
            feature_names = set().union(*(r["features"].keys() for r in seed_rows))
            features = {name: statistics.fmean(r["features"].get(name, 0.0) for r in seed_rows)
                        for name in feature_names}
            all_components = set().union(*(r["device_components"].keys() for r in seed_rows))
            component_means = {
                name: statistics.fmean(r["device_components"].get(name, 0.0) for r in seed_rows)
                for name in all_components
            }
            component_cpu_means = {}
            cpu_names = set().union(*(r["cpu_components"].keys() for r in seed_rows))
            for name in cpu_names:
                component_cpu_means[name] = statistics.fmean(r["cpu_components"].get(name, 0.0) for r in seed_rows)
            stages = Counter(str(r["eos_stage"]) for r in seed_rows if r["eos"] and r["eos_stage"])
            samples.append({
                "source": source, "context": context, "scope": scope, "thermal": thermal,
                "action": action, "prefix_len": statistics.fmean(r["prefix_len"] for r in seed_rows),
                "request_budget_tokens": statistics.fmean(r["request_budget_tokens"] for r in seed_rows),
                "actual_node_counts": [r["actual_node_count"] for r in seed_rows],
                "actual_read_buckets": [r["actual_read_bucket"] for r in seed_rows],
                "seed_count": len(seed_rows), "seeds": [r["seed"] for r in seed_rows],
                "seed_rows": seed_rows, "features": features,
                "R": statistics.fmean(r["progress"] for r in seed_rows),
                "C": statistics.fmean(r["cost_ms"] for r in seed_rows),
                "ratio": statistics.fmean(r["progress"] for r in seed_rows) /
                         statistics.fmean(r["cost_ms"] for r in seed_rows),
                "eos_count": sum(bool(r["eos"]) for r in seed_rows),
                "eos_rate": statistics.fmean(1.0 if r["eos"] else 0.0 for r in seed_rows),
                "eos_stages": dict(stages),
                "accepted_count": _mean_available(seed_rows, "accepted_count"),
                "recovery_calls": _mean_available(seed_rows, "recovery_calls"),
                "recovery_input_tokens": _mean_available(seed_rows, "recovery_input_tokens"),
                "target_calls": _mean_available(seed_rows, "target_calls"),
                "target_input_tokens": _mean_available(seed_rows, "target_input_tokens"),
                "draft_calls": _mean_available(seed_rows, "draft_calls"),
                "draft_input_tokens": _mean_available(seed_rows, "draft_input_tokens"),
                "ranker_calls": _mean_available(seed_rows, "ranker_calls"),
                "ranker_p1_candidate_count": _mean_available(seed_rows, "ranker_p1_candidate_count"),
                "device_components": component_means,
                "cpu_components": component_cpu_means,
                "seed_R": {r["seed"]: r["progress"] for r in seed_rows},
                "seed_C": {r["seed"]: r["cost_ms"] for r in seed_rows},
                "seed_eos": {r["seed"]: bool(r["eos"]) for r in seed_rows},
                "seed_R_sd": statistics.stdev(r["progress"] for r in seed_rows) if len(seed_rows) > 1 else 0.0,
                "seed_C_sd": statistics.stdev(r["cost_ms"] for r in seed_rows) if len(seed_rows) > 1 else 0.0,
            })
    return samples, {"omitted_contexts": dict(omitted), "complete_action_contexts": len(samples) // 3,
                     "sources": len(set(x["source"] for x in samples)),
                     "contexts": len(set((x["source"], x["context"]) for x in samples))}


def _mean_available(rows, field):
    values = [r[field] for r in rows if r.get(field) is not None]
    return statistics.fmean(values) if values else None


def action_feature(sample: dict[str, Any], action: str, *, full=True) -> dict[str, float]:
    base = dict(sample["features"])
    action_budget = (2.0 if action == "K1" else 4.0) if sample["scope"] == "single_round" else sample["request_budget_tokens"]
    planned_nodes = float(ACTION_NODES[action])
    planned_read_bucket = 0.0
    if planned_nodes > 0:
        planned_read = max(1, int(sample["prefix_len"]) + int(planned_nodes))
        planned_read_bucket = float(max(8, 1 << (planned_read - 1).bit_length()))
    base["cache_pos_before"] = float(sample["prefix_len"])
    base["remaining_budget_before"] = action_budget
    base["action_budget_tokens"] = action_budget
    base["node_count"] = planned_nodes
    base["log_read_bucket"] = math.log1p(planned_read_bucket)
    for name in ACTIONS:
        indicator = 1.0 if name == action else 0.0
        base[f"{name}_x_prefix"] = indicator * base.get("log_prefix_len", 0.0)
        base[f"{name}_x_last_gap"] = indicator * base.get("last_gap_rank_norm", 0.0)
        base[f"{name}_x_read_bucket"] = indicator * base.get("log_read_bucket", 0.0)
        base[f"{name}_x_node_count"] = indicator * base.get("node_count", 0.0)
        base[f"action_{name}"] = indicator
    if not full:
        return {name: base.get(name, 0.0) for name in NODE_FEATURES}
    for name in ACTIONS:
        base[f"action_{name}"] = 1.0 if name == action else 0.0
    return base


def _solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    augmented = [list(matrix[i]) + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            augmented[pivot][col] += 1e-8
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        divisor = augmented[col][col]
        if abs(divisor) < 1e-15:
            continue
        for j in range(col, n + 1):
            augmented[col][j] /= divisor
        for i in range(n):
            if i == col:
                continue
            factor = augmented[i][col]
            if factor == 0:
                continue
            for j in range(col, n + 1):
                augmented[i][j] -= factor * augmented[col][j]
    return [augmented[i][n] for i in range(n)]


class Ridge:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.names: list[str] = []
        self.means: dict[str, float] = {}
        self.scales: dict[str, float] = {}
        self.weights: list[float] = []
        self.y_mean = 0.0
        self.y_scale = 1.0
        self.fitted = False

    def fit(self, rows: list[dict[str, Any]], ys: list[float], sample_weights: list[float], feature_names: list[str]):
        if not rows:
            raise ValueError("cannot fit a model without rows")
        self.names = list(feature_names)
        total_w = sum(sample_weights)
        for name in self.names:
            mean = sum(w * float(row.get(name, 0.0)) for row, w in zip(rows, sample_weights)) / total_w
            variance = sum(w * (float(row.get(name, 0.0)) - mean) ** 2
                           for row, w in zip(rows, sample_weights)) / total_w
            self.means[name] = mean
            self.scales[name] = math.sqrt(variance) if variance > 1e-12 else 1.0
        self.y_mean = sum(w * y for w, y in zip(sample_weights, ys)) / total_w
        y_var = sum(w * (y - self.y_mean) ** 2 for w, y in zip(sample_weights, ys)) / total_w
        self.y_scale = math.sqrt(y_var) if y_var > 1e-12 else 1.0
        width = len(self.names) + 1
        gram = [[0.0] * width for _ in range(width)]
        rhs = [0.0] * width
        for row, y, weight in zip(rows, ys, sample_weights):
            x = [1.0] + [(float(row.get(name, 0.0)) - self.means[name]) / self.scales[name]
                         for name in self.names]
            yz = (y - self.y_mean) / self.y_scale
            for i in range(width):
                rhs[i] += weight * x[i] * yz
                for j in range(width):
                    gram[i][j] += weight * x[i] * x[j]
        for i in range(1, width):
            gram[i][i] += self.alpha
        self.weights = _solve_linear(gram, rhs)
        self.fitted = True
        return self

    def predict(self, row: dict[str, Any]) -> float:
        if not self.fitted:
            raise RuntimeError("model is not fitted")
        value = self.weights[0]
        for i, name in enumerate(self.names, 1):
            value += self.weights[i] * (float(row.get(name, 0.0)) - self.means[name]) / self.scales[name]
        return self.y_mean + self.y_scale * value

    def export(self) -> dict[str, Any]:
        return {"kind": "standardized_ridge", "alpha": self.alpha,
                "features": self.names, "means": self.means, "scales": self.scales,
                "target_mean": self.y_mean, "target_scale": self.y_scale,
                "weights_standardized": self.weights}

    def raw_coefficients(self) -> dict[str, float]:
        if not self.fitted:
            return {}
        coefficients = {
            name: self.weights[index] * self.y_scale / self.scales[name]
            for index, name in enumerate(self.names, 1)
        }
        intercept = self.y_mean + self.weights[0] * self.y_scale
        intercept -= sum(coefficients[name] * self.means[name] for name in self.names)
        return {"intercept": intercept, **coefficients}


def _action_rows(samples: list[dict[str, Any]], action: str) -> list[dict[str, Any]]:
    return [s for s in samples if s["action"] == action]


def _sample_weights(samples: list[dict[str, Any]]) -> list[float]:
    counts = Counter(s["source"] for s in samples)
    return [1.0 / counts[s["source"]] for s in samples]


def _fit_target(samples, target, alpha, feature_names, transform=lambda s: s.get("features", {})):
    usable = [s for s in samples if s.get(target) is not None and math.isfinite(float(s[target]))]
    if not usable:
        return None
    model = Ridge(alpha).fit([transform(s) for s in usable], [float(s[target]) for s in usable],
                             _sample_weights(usable), feature_names)
    return model


def _train_action_means(samples: list[dict[str, Any]], field: str) -> dict[str, float]:
    result = {}
    for action in ACTIONS:
        rows = _action_rows(samples, action)
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        result[action] = statistics.fmean(values) if values else 0.0
    return result


def _recovery_cost_feature(sample: dict[str, Any], action: str, recovery_calls: float) -> dict[str, float]:
    feature = action_feature(sample, action)
    feature["recovery_calls"] = float(recovery_calls)
    log_prefix = feature.get("log_prefix_len", 0.0)
    for name in ACTIONS:
        indicator = 1.0 if name == action else 0.0
        value = float(recovery_calls) * indicator
        feature[f"recovery_x_action_{name}"] = value
        feature[f"recovery_x_action_{name}_x_prefix"] = value * log_prefix
    return feature


def _recovery_cost_feature_names(base_names: list[str]) -> list[str]:
    names = list(base_names) + ["recovery_calls"]
    for action in ACTIONS:
        names.extend((f"recovery_x_action_{action}",
                      f"recovery_x_action_{action}_x_prefix"))
    return list(dict.fromkeys(names))


def _choose(scores: dict[str, float]) -> str:
    return max(ACTIONS, key=lambda action: (scores.get(action, -math.inf), -ACTIONS.index(action)))


def _predict_for_context(models, sample_by_action):
    preds = {}
    elapsed = {}
    for name, spec in models.items():
        started = time.perf_counter_ns()
        scores = {}
        invalid_costs = 0
        for action in ACTIONS:
            sample = sample_by_action[action]
            if spec["kind"] == "fixed":
                scores[action] = float(spec["scores"][action])
            elif spec["kind"] == "progress":
                scores[action] = spec["r_model"].predict(action_feature(sample, action))
            elif spec["kind"] == "node_cost":
                expected_progress = spec["action_r"][action]
                cost = spec["c_model"].predict(action_feature(sample, action, full=False))
                if cost <= 0:
                    invalid_costs += 1
                    scores[action] = -1e100
                else:
                    scores[action] = max(0.0, expected_progress) / cost
            elif spec["kind"] == "tree_proxy":
                if action == "K1":
                    scores[action] = spec["action_r"][action] / max(spec["action_c"][action], 1e-9)
                else:
                    cost = spec["tree_model"].predict(action_feature(sample, action, full=False))
                    if cost <= 0:
                        invalid_costs += 1
                        scores[action] = -1e100
                    else:
                        scores[action] = max(0.0, spec["action_r"][action]) / cost
            elif spec["kind"] == "ratio":
                scores[action] = spec["ratio_model"].predict(action_feature(sample, action))
            elif spec["kind"] == "separate":
                expected_progress = spec["r_model"].predict(action_feature(sample, action))
                cost = spec["c_model"].predict(action_feature(sample, action))
                if cost <= 0:
                    invalid_costs += 1
                    scores[action] = -1e100
                else:
                    scores[action] = max(0.0, expected_progress) / cost
            elif spec["kind"] == "recovery_cost":
                features = action_feature(sample, action)
                expected_calls = spec["recovery_model"].predict(features)
                max_calls = (1.0 if sample["scope"] == "single_round"
                             else max(1.0, float(sample["request_budget_tokens"]) / 2.0))
                expected_calls = min(max_calls, max(0.0, expected_calls))
                cost = spec["c_model"].predict(
                    _recovery_cost_feature(sample, action, max(0.0, expected_calls))
                )
                expected_progress = spec["r_model"].predict(features)
                if cost <= 0:
                    invalid_costs += 1
                    scores[action] = -1e100
                else:
                    scores[action] = max(0.0, expected_progress) / cost
            elif spec["kind"] == "accepted":
                expected_accepts = spec["a_model"].predict(action_feature(sample, action))
                cost = spec["c_model"].predict(action_feature(sample, action))
                if cost <= 0:
                    invalid_costs += 1
                    scores[action] = -1e100
                else:
                    scores[action] = max(0.0, expected_accepts) / cost
            elif spec["kind"] == "component_proxy":
                expected_progress = spec["r_model"].predict(action_feature(sample, action))
                cost = sum(max(0.0, model.predict(action_feature(sample, action)))
                           for model in spec["component_models"].values())
                if spec["recovery_call_model"] is not None:
                    expected_calls = max(0.0, spec["recovery_call_model"].predict(
                        action_feature(sample, action)
                    ))
                    cost += expected_calls * spec["recovery_component_ms_per_call"]
                if cost <= 0:
                    invalid_costs += 1
                    scores[action] = -1e100
                else:
                    scores[action] = max(0.0, expected_progress) / cost
            else:
                raise ValueError(f"unknown policy model {spec['kind']}")
        elapsed[name] = (time.perf_counter_ns() - started) / 1000.0
        preds[name] = {"scores": scores, "action": _choose(scores), "invalid_cost_predictions": invalid_costs}
    return preds, elapsed


def _fit_policy_models(train: list[dict[str, Any]], alpha: float) -> dict[str, dict[str, Any]]:
    full_names = list(BASE_FEATURES) + [f"action_{a}" for a in ACTIONS]
    full_names.extend(sorted(set().union(*(s["features"].keys() for s in train)) - set(full_names)))
    for action in ACTIONS:
        full_names.extend((f"{action}_x_prefix", f"{action}_x_last_gap",
                           f"{action}_x_read_bucket", f"{action}_x_node_count"))
    full_names = list(dict.fromkeys(full_names))
    node_names = list(NODE_FEATURES) + [f"action_{a}" for a in ACTIONS]
    transform_full = lambda s: action_feature(s, s["action"])
    transform_node = lambda s: action_feature(s, s["action"], full=False)
    r_model = _fit_target(train, "R", alpha, full_names, transform_full)
    c_model = _fit_target(train, "C", alpha, full_names, transform_full)
    ratio_model = _fit_target(train, "ratio", alpha, full_names, transform_full)
    node_c_model = _fit_target(train, "C", alpha, node_names, transform_node)
    accepted_model = _fit_target(train, "accepted_count", alpha, full_names, transform_full)
    recovery_model = _fit_target(train, "recovery_calls", alpha, full_names, transform_full)
    rec_cost_names = _recovery_cost_feature_names(full_names)
    rec_cost_transform = lambda s: _recovery_cost_feature(
        s, s["action"], float(s.get("recovery_calls") or 0.0)
    )
    recovery_cost_model = _fit_target(train, "C", alpha, rec_cost_names, rec_cost_transform)
    action_r = _train_action_means(train, "R")
    action_score = {a: _train_action_means(train, "ratio")[a] for a in ACTIONS}
    models = {
        "fixed_w1": {"kind": "fixed", "scores": {a: 1.0 if a == "W1" else 0.0 for a in ACTIONS}},
        "training_action_mean": {"kind": "fixed", "scores": action_score},
    }
    if r_model:
        models["progress_only"] = {"kind": "progress", "r_model": r_model}
    if c_model:
        models["full_separate_R_C"] = {"kind": "separate", "r_model": r_model, "c_model": c_model}
    if recovery_model and recovery_cost_model and r_model:
        models["recovery_event_cost"] = {
            "kind": "recovery_cost", "r_model": r_model,
            "recovery_model": recovery_model, "c_model": recovery_cost_model,
        }
    if ratio_model:
        models["direct_R_over_C"] = {"kind": "ratio", "ratio_model": ratio_model}
    if node_c_model and r_model:
        models["node_context_cost"] = {"kind": "node_cost", "action_r": action_r, "c_model": node_c_model}
    tree_rows = []
    for sample in train:
        value = dict(sample)
        value["tree_verify_ms"] = sample.get("device_components", {}).get("tree_verify")
        if value["tree_verify_ms"] is not None:
            tree_rows.append(value)
    if tree_rows and len(tree_rows) >= 6:
        tree_model = _fit_target(tree_rows, "tree_verify_ms", alpha, node_names, transform_node)
        action_c = _train_action_means(train, "C")
        if tree_model:
            models["tree_verify_time_proxy"] = {
                "kind": "tree_proxy", "action_r": action_r, "action_c": action_c,
                "tree_model": tree_model,
            }
    if accepted_model and c_model:
        models["accepted_count_over_cost"] = {"kind": "accepted", "a_model": accepted_model, "c_model": c_model}

    component_names = sorted(set().union(*(s.get("device_components", {}).keys() for s in train)))
    recovery_component_names = [name for name in component_names
                                if name.startswith("target_recovery") or name.startswith("recovery")]
    ordinary_component_names = [name for name in component_names
                                if name not in recovery_component_names]
    component_models = {}
    for name in ordinary_component_names:
        target_name = f"component:{name}"
        copied = []
        for sample in train:
            value = dict(sample)
            value[target_name] = float(sample.get("device_components", {}).get(name, 0.0))
            copied.append(value)
        model = _fit_target(copied, target_name, alpha, full_names, transform_full)
        if model:
            component_models[name] = model
    recovery_call_model = _fit_target(train, "recovery_calls", alpha, full_names, transform_full)
    recovery_cost_numerator = 0.0
    recovery_cost_denominator = 0.0
    source_context_counts = Counter(s["source"] for s in train)
    for sample in train:
        calls = float(sample.get("recovery_calls") or 0.0)
        if calls <= 0:
            continue
        measured = sum(float(sample.get("device_components", {}).get(name, 0.0))
                       for name in recovery_component_names)
        source_weight = 1.0 / max(1, source_context_counts[sample["source"]])
        recovery_cost_numerator += source_weight * measured
        recovery_cost_denominator += source_weight * calls
    recovery_component_ms_per_call = (
        recovery_cost_numerator / recovery_cost_denominator
        if recovery_cost_denominator > 0 else 0.0
    )
    if (component_models or recovery_component_names) and r_model:
        models["component_device_proxy"] = {
            "kind": "component_proxy", "r_model": r_model,
            "component_models": component_models,
            "recovery_call_model": recovery_call_model if recovery_component_names else None,
            "recovery_component_ms_per_call": recovery_component_ms_per_call,
        }
    return models


def _fit_diagnostics(train: list[dict[str, Any]], test: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    names = list(BASE_FEATURES) + [f"action_{a}" for a in ACTIONS]
    names.extend(sorted(set().union(*(s["features"].keys() for s in train)) - set(names)))
    for action in ACTIONS:
        names.extend((f"{action}_x_prefix", f"{action}_x_last_gap",
                      f"{action}_x_read_bucket", f"{action}_x_node_count"))
    names = list(dict.fromkeys(names))
    transform = lambda s: action_feature(s, s["action"])
    fields = ("R", "C", "recovery_calls", "recovery_input_tokens", "target_calls", "target_input_tokens",
              "draft_calls", "draft_input_tokens", "ranker_calls")
    result = {}
    for field in fields:
        model = _fit_target(train, field, alpha, names, transform)
        if model:
            pairs = [(float(s[field]), model.predict(transform(s))) for s in test if s.get(field) is not None]
            errors = [pred - actual for actual, pred in pairs]
            result[field] = {
                "n": len(pairs),
                "mae": statistics.fmean(abs(e) for e in errors) if errors else None,
                "rmse": math.sqrt(statistics.fmean(e * e for e in errors)) if errors else None,
                "mean_actual": statistics.fmean(actual for actual, _ in pairs) if pairs else None,
                "mean_predicted": statistics.fmean(pred for _, pred in pairs) if pairs else None,
            }
    recovery_model = _fit_target(train, "recovery_calls", alpha, names, transform)
    recovery_cost_model = _fit_target(
        train, "C", alpha, _recovery_cost_feature_names(names),
        lambda sample: _recovery_cost_feature(
            sample, sample["action"], float(sample.get("recovery_calls") or 0.0)
        ),
    )
    if recovery_model is not None and recovery_cost_model is not None:
        pairs = []
        for sample in test:
            actual = sample.get("C")
            if actual is None:
                continue
            expected_calls = recovery_model.predict(transform(sample))
            max_calls = (1.0 if sample["scope"] == "single_round"
                         else max(1.0, float(sample["request_budget_tokens"]) / 2.0))
            expected_calls = min(max_calls, max(0.0, expected_calls))
            predicted_cost = recovery_cost_model.predict(
                _recovery_cost_feature(sample, sample["action"], expected_calls)
            )
            pairs.append((float(actual), predicted_cost))
        errors = [predicted - actual for actual, predicted in pairs]
        result["C_recovery_event_cost_predicted_Erec"] = {
            "n": len(pairs),
            "mae": statistics.fmean(abs(error) for error in errors) if errors else None,
            "rmse": math.sqrt(statistics.fmean(error * error for error in errors)) if errors else None,
            "mean_actual": statistics.fmean(actual for actual, _ in pairs) if pairs else None,
            "mean_predicted": statistics.fmean(predicted for _, predicted in pairs) if pairs else None,
            "test_recovery_counts_used_for_prediction": False,
        }
    return result


def _source_metrics(samples, choices, policy_name):
    by_source_context = defaultdict(dict)
    for sample in samples:
        key = (sample["source"], sample["context"])
        by_source_context[key][sample["action"]] = sample
    source_rows = defaultdict(lambda: {"R": 0.0, "C": 0.0, "contexts": 0, "regret_log": 0.0})
    choices_count = Counter()
    for (source, context), action_map in by_source_context.items():
        action = choices.get((source, context), "W1")
        selected = action_map[action]
        source_rows[source]["R"] += selected["R"]
        source_rows[source]["C"] += selected["C"]
        source_rows[source]["contexts"] += 1
        oracle = max(row["ratio"] for row in action_map.values())
        regret = max(0.0, math.log(max(oracle, 1e-12) / max(selected["ratio"], 1e-12)))
        source_rows[source]["regret_log"] += regret
        choices_count[action] += 1
    per_source = {}
    speedups = []
    regrets = []
    for source, value in source_rows.items():
        speed = value["R"] / value["C"] if value["C"] > 0 else None
        per_source[source] = {
            "contexts": value["contexts"], "mean_tokens_per_context": value["R"] / max(1, value["contexts"]),
            "mean_cost_ms_per_context": value["C"] / max(1, value["contexts"]),
            "tokens_per_ms": speed,
            "mean_log_regret_to_sample_oracle": value["regret_log"] / max(1, value["contexts"]),
        }
        if speed is not None:
            speedups.append(speed)
        regrets.append(per_source[source]["mean_log_regret_to_sample_oracle"])
    return per_source, dict(choices_count), statistics.fmean(speedups) if speedups else None, regrets


def _paired_source_lift(selected_source, w1_source, samples=2000, seed=37161):
    common = sorted(set(selected_source) & set(w1_source))
    ratios = []
    for source in common:
        a = selected_source[source].get("tokens_per_ms")
        b = w1_source[source].get("tokens_per_ms")
        if a and b and a > 0 and b > 0:
            ratios.append((source, math.log(a / b)))
    if not ratios:
        return {"sources": 0, "geomean_speedup": None, "source_bootstrap_95pct": None}
    mean_log = statistics.fmean(x for _, x in ratios)
    rng = random.Random(seed)
    draws = []
    n = len(ratios)
    for _ in range(int(samples)):
        chosen = [item[1] for item in rng.choices(ratios, k=n)]
        draws.append(math.exp(statistics.fmean(chosen)))
    draws.sort()
    low = draws[int(0.025 * (len(draws) - 1))]
    high = draws[int(0.975 * (len(draws) - 1))]
    return {"sources": n, "geomean_speedup": math.exp(mean_log),
            "source_bootstrap_95pct": [low, high],
            "source_log_speedups": {source: value for source, value in ratios}}


def _paired_seed_lift(samples, choices, bootstrap_samples=2000, seed=84173):
    grouped = defaultdict(dict)
    for sample in samples:
        grouped[(sample["source"], sample["context"])][sample["action"]] = sample
    contexts = [(key, actions) for key, actions in grouped.items()
                if all(action in actions for action in ACTIONS)]
    seed_sets = []
    for _, actions in contexts:
        sets = [set(actions[action]["seed_R"]) & set(actions[action]["seed_C"])
                for action in ACTIONS]
        seed_sets.append(set.intersection(*sets))
    common = sorted(set.intersection(*seed_sets)) if seed_sets else []
    if len(common) < 2:
        return {"common_seeds": len(common), "geomean_speedup": None,
                "paired_seed_bootstrap_95pct": None}

    def source_logs(draw):
        totals = defaultdict(lambda: {"selected_R": 0.0, "selected_C": 0.0,
                                      "w1_R": 0.0, "w1_C": 0.0})
        for key, actions in contexts:
            action = choices.get(key, "W1")
            selected = actions[action]
            w1 = actions["W1"]
            source = key[0]
            for selected_seed in draw:
                if selected_seed not in selected["seed_R"] or selected_seed not in w1["seed_R"]:
                    continue
                totals[source]["selected_R"] += selected["seed_R"][selected_seed]
                totals[source]["selected_C"] += selected["seed_C"][selected_seed]
                totals[source]["w1_R"] += w1["seed_R"][selected_seed]
                totals[source]["w1_C"] += w1["seed_C"][selected_seed]
        logs = []
        for value in totals.values():
            chosen_rate = value["selected_R"] / value["selected_C"] if value["selected_C"] > 0 else 0.0
            w1_rate = value["w1_R"] / value["w1_C"] if value["w1_C"] > 0 else 0.0
            if chosen_rate > 0 and w1_rate > 0:
                logs.append(math.log(chosen_rate / w1_rate))
        return logs

    center_logs = source_logs(common)
    if not center_logs:
        return {"common_seeds": len(common), "geomean_speedup": None,
                "paired_seed_bootstrap_95pct": None}
    rng = random.Random(seed)
    draws = []
    for _ in range(int(bootstrap_samples)):
        selected = rng.choices(common, k=len(common))
        logs = source_logs(selected)
        if logs:
            draws.append(math.exp(statistics.fmean(logs)))
    draws.sort()
    if not draws:
        interval = None
    else:
        interval = [draws[int(0.025 * (len(draws) - 1))],
                    draws[int(0.975 * (len(draws) - 1))]]
    return {"common_seeds": len(common),
            "geomean_speedup": math.exp(statistics.fmean(center_logs)),
            "paired_seed_bootstrap_95pct": interval,
            "resampling": "resample paired seed IDs across all contexts and actions; source speed ratios remain equally weighted"}


def _balanced_seed_selection(samples):
    by_context = defaultdict(dict)
    for sample in samples:
        by_context[(sample["source"], sample["context"])][sample["action"]] = sample
    source_lifts = defaultdict(list)
    oracle_lifts = defaultdict(list)
    evaluated = 0
    split_count_hist = Counter()
    for (source, _), action_map in by_context.items():
        if any(action not in action_map for action in ACTIONS):
            continue
        seed_sets = [set(action_map[a]["seed_R"]) & set(action_map[a]["seed_C"]) for a in ACTIONS]
        seeds = sorted(set.intersection(*seed_sets))
        if len(seeds) < 4:
            continue
        half = len(seeds) // 2
        combinations = list(itertools.combinations(seeds, half))
        if len(combinations) > 500:
            combinations = combinations[::max(1, len(combinations) // 500)][:500]
        split_count_hist[str(len(combinations))] += 1
        for train_seeds in combinations:
            train_set = set(train_seeds)
            test_seeds = [seed for seed in seeds if seed not in train_set]
            train_speed = {}
            test_speed = {}
            for action in ACTIONS:
                sample = action_map[action]
                tr = sum(sample["seed_R"][s] for s in train_seeds) / sum(sample["seed_C"][s] for s in train_seeds)
                te = sum(sample["seed_R"][s] for s in test_seeds) / sum(sample["seed_C"][s] for s in test_seeds)
                train_speed[action] = tr
                test_speed[action] = te
            picked = _choose(train_speed)
            w1 = test_speed["W1"]
            source_lifts[source].append(math.log(max(test_speed[picked], 1e-12) / max(w1, 1e-12)))
            evaluated += 1
        full_speed = {a: action_map[a]["ratio"] for a in ACTIONS}
        full_oracle = max(full_speed.values())
        full_w1 = full_speed["W1"]
        if full_w1 > 0:
            oracle_lifts[source].append(math.log(full_oracle / full_w1))
    source_log_means = [statistics.fmean(values) for values in source_lifts.values() if values]
    crossfit_lift = math.exp(statistics.fmean(source_log_means)) if source_log_means else None
    oracle_source_means = [statistics.fmean(values) for values in oracle_lifts.values() if values]
    in_sample_lift = math.exp(statistics.fmean(oracle_source_means)) if oracle_source_means else None
    return {
        "contexts": sum(len(values) for values in oracle_lifts.values()),
        "sources": len(source_lifts), "balanced_splits_evaluated": evaluated,
        "split_count_histogram_per_context": dict(split_count_hist),
        "heldout_half_seed_action_lift_over_W1_geomean": crossfit_lift,
        "full_seed_per_context_best_over_W1_geomean": in_sample_lift,
        "apparent_selection_optimism_ratio": (in_sample_lift / crossfit_lift
                                               if in_sample_lift and crossfit_lift else None),
        "interpretation": (
            "The full-seed per-context best picks and scores on the same finite seed set, so it is optimistic. "
            "Balanced half-splits select using one half and score on the other; this estimates finite-seed selection "
            "noise, with remaining uncertainty from the limited seed count and source set. Neither value is online speed."
        ),
    }


def _make_policy_choices(samples, alpha):
    by_source = defaultdict(list)
    for sample in samples:
        by_source[sample["source"]].append(sample)
    oof = defaultdict(dict)
    oof_rankings = defaultdict(dict)
    diagnostics = {}
    score_times = defaultdict(list)
    invalid_cost_predictions = Counter()
    recovery_cost_fold_slopes = {}
    for source in sorted(by_source):
        test = by_source[source]
        train = [sample for sample in samples if sample["source"] != source]
        if not train:
            continue
        models = _fit_policy_models(train, alpha)
        if "recovery_event_cost" in models:
            recovery_cost_fold_slopes[source] = _export_model_spec(models["recovery_event_cost"])[
                "recovery_slope_by_action"
            ]
        grouped = defaultdict(dict)
        for sample in test:
            grouped[(sample["source"], sample["context"])][sample["action"]] = sample
        for context_key, action_map in grouped.items():
            if any(action not in action_map for action in ACTIONS):
                continue
            predictions, elapsed = _predict_for_context(models, action_map)
            for name, result in predictions.items():
                oof[name][context_key] = result["action"]
                oof_rankings[name][context_key] = [
                    {"action": action, "score": result["scores"][action]}
                    for action in sorted(ACTIONS, key=lambda item: result["scores"][item], reverse=True)
                ]
                score_times[name].append(elapsed[name])
                invalid_cost_predictions[name] += result.get("invalid_cost_predictions", 0)
        diagnostics[source] = _fit_diagnostics(train, test, alpha)
    final_models = _fit_policy_models(samples, alpha)
    all_fit_exports = {name: _export_model_spec(spec) for name, spec in final_models.items()}
    return (oof, oof_rankings, diagnostics, score_times, all_fit_exports,
            dict(invalid_cost_predictions), recovery_cost_fold_slopes)


def _export_model_spec(spec):
    output = {"kind": spec["kind"]}
    if spec["kind"] == "fixed":
        output["scores"] = spec["scores"]
    elif spec["kind"] == "progress":
        output["R_model"] = spec["r_model"].export()
    elif spec["kind"] == "node_cost":
        output["action_R_means"] = spec["action_r"]
        output["C_model"] = spec["c_model"].export()
    elif spec["kind"] == "tree_proxy":
        output["R_means_by_action"] = spec["action_r"]
        output["K1_C_mean"] = spec["action_c"]["K1"]
        output["tree_verify_device_ms_model_for_W_actions"] = spec["tree_model"].export()
    elif spec["kind"] == "ratio":
        output["R_over_C_model"] = spec["ratio_model"].export()
    elif spec["kind"] == "separate":
        output["R_model"] = spec["r_model"].export()
        output["C_model"] = spec["c_model"].export()
    elif spec["kind"] == "recovery_cost":
        raw = spec["c_model"].raw_coefficients()
        expected_raw = spec["recovery_model"].raw_coefficients()
        output["R_model"] = spec["r_model"].export()
        output["expected_recovery_calls_model"] = spec["recovery_model"].export()
        output["C_model_with_training_recovery_calls"] = spec["c_model"].export()
        output["recovery_slope_by_action"] = {
            action: {
                "shared_per_call": raw.get("recovery_calls", 0.0),
                "action_intercept": raw.get(f"recovery_x_action_{action}", 0.0),
                "log1p_prefix_slope": raw.get(f"recovery_x_action_{action}_x_prefix", 0.0),
                "formula": "shared_per_call + action_intercept + log1p(prefix_len) * log1p_prefix_slope",
            }
            for action in ACTIONS
        }
        output["expected_recovery_call_coefficients_raw"] = expected_raw
    elif spec["kind"] == "accepted":
        output["accepted_count_model"] = spec["a_model"].export()
        output["C_model"] = spec["c_model"].export()
    elif spec["kind"] == "component_proxy":
        output["R_model"] = spec["r_model"].export()
        output["component_device_ms_models"] = {name: model.export()
                                                  for name, model in spec["component_models"].items()}
        output["recovery_cost_per_call_from_training_components_ms"] = spec["recovery_component_ms_per_call"]
        output["expected_recovery_calls_model"] = (
            spec["recovery_call_model"].export() if spec["recovery_call_model"] else None
        )
    return output


def _metrics_for_mode(samples, alpha, bootstrap_samples, seed_bootstrap_samples):
    if not samples:
        return {"error": "no complete matched source/context/action samples"}
    (oof, oof_rankings, diagnostic_folds, score_times, full_models,
     invalid_cost_predictions, recovery_cost_fold_slopes) = _make_policy_choices(samples, alpha)
    grouped = defaultdict(dict)
    for sample in samples:
        grouped[(sample["source"], sample["context"])][sample["action"]] = sample
    complete_keys = [key for key, value in grouped.items() if all(a in value for a in ACTIONS)]
    if not complete_keys:
        return {"error": "no context has all three actions"}

    policy_names = sorted(set(oof) | {"fixed_w1"})
    choices_by_policy = {}
    metric_rows = {}
    w1_choices = {key: "W1" for key in complete_keys}
    w1_source, w1_counts, _, _ = _source_metrics(samples, w1_choices, "fixed_w1")
    for name in policy_names:
        if name == "fixed_w1":
            choices = w1_choices
        else:
            choices = oof.get(name, {})
            choices = {key: choices.get(key, "W1") for key in complete_keys}
        choices_by_policy[name] = choices
        source_detail, frequencies, mean_speed, regrets = _source_metrics(samples, choices, name)
        lift = _paired_source_lift(source_detail, w1_source, samples=bootstrap_samples)
        log_regret = statistics.fmean(math.log(max(1.0, math.exp(v))) for v in regrets) if regrets else None
        metric_rows[name] = {
            "source_equal": {
                "sources": len(source_detail),
                "equal_source_mean_tokens_per_ms": statistics.fmean(
                    row["tokens_per_ms"] for row in source_detail.values()
                ) if source_detail else None,
                "equal_source_geomean_tokens_per_ms": math.exp(statistics.fmean(
                    math.log(max(row["tokens_per_ms"], 1e-12)) for row in source_detail.values()
                    if row["tokens_per_ms"] is not None
                )) if source_detail else None,
                "per_source": source_detail,
            },
            "relative_to_fixed_W1": lift,
            "paired_seed_uncertainty_vs_fixed_W1": _paired_seed_lift(
                samples, choices, bootstrap_samples=seed_bootstrap_samples
            ),
            "action_frequencies": frequencies,
            "action_rankings": {
                f"{source}::{context}": oof_rankings.get(name, {}).get(
                    (source, context), [{"action": "W1", "score": 1.0}]
                )
                for source, context in complete_keys
            },
            "mean_log_sample_oracle_regret": log_regret,
            "choices": {f"{source}::{context}": action for (source, context), action in choices.items()},
        }
    selector_cost = {}
    for name, values in score_times.items():
        ordered = sorted(values)
        selector_cost[name] = {
            "contexts_scored": len(values),
            "median_microseconds": statistics.median(values) if values else None,
            "p95_microseconds": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))] if values else None,
            "measurement": "local CPU time for Python feature-vector scoring across three actions; excluded from decoder time",
        }
    rate_diagnostics = _fit_diagnostics([], [], alpha)
    # Cross-fit recovery/call prediction quality, pooled without letting a source appear in its own fit.
    pooled_diag = defaultdict(list)
    for row in diagnostic_folds.values():
        for name, values in row.items():
            if values.get("mae") is not None:
                pooled_diag[name].append(values)
    recovery_report = {}
    for name, values in pooled_diag.items():
        recovery_report[name] = {
            "source_fold_mean_mae": statistics.fmean(v["mae"] for v in values),
            "source_fold_mean_rmse": statistics.fmean(v["rmse"] for v in values),
            "source_folds": len(values),
        }
    progress_cost_keys = ("R", "C", "C_recovery_event_cost_predicted_Erec")
    progress_cost_report = {name: recovery_report[name] for name in progress_cost_keys
                            if name in recovery_report}
    slope_report = {}
    for action in ACTIONS:
        fold_values = [values[action] for values in recovery_cost_fold_slopes.values()]
        by_prefix = {}
        for prefix in (64, 256, 512, 1024):
            values = [row["shared_per_call"] + row["action_intercept"] +
                      math.log1p(prefix) * row["log1p_prefix_slope"] for row in fold_values]
            by_prefix[str(prefix)] = {
                "fold_values_ms_per_recovery": values,
                "mean": statistics.fmean(values) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "negative_folds": sum(value < 0 for value in values),
            }
        slope_report[action] = by_prefix
    return {
        "matched": {"contexts": len(complete_keys), "sources": len(set(k[0] for k in complete_keys)),
                    "actions": list(ACTIONS), "total_context_action_rows": len(samples),
                    "seed_counts": sorted(set(s["seed_count"] for s in samples))},
        "seed_variability": {
            "mean_context_action_R_sd": statistics.fmean(s["seed_R_sd"] for s in samples),
            "mean_context_action_C_ms_sd": statistics.fmean(s["seed_C_sd"] for s in samples),
            "meaning": "standard deviation across paired decode seeds after timing repeats were aggregated",
        },
        "policies": metric_rows,
        "selector_scoring_cost": selector_cost,
        "nonpositive_predicted_cost_counts": invalid_cost_predictions,
        "recovery_and_call_predictions": recovery_report,
        "heldout_progress_cost_prediction_errors": progress_cost_report,
        "recovery_cost_slope_by_action_and_prefix": slope_report,
        "recovery_cost_slope_by_heldout_source": recovery_cost_fold_slopes,
        "finite_seed_action_selection": _balanced_seed_selection(samples),
        "models_fit_on_all_selected_development_rows": full_models,
        "diagnostic_source_folds": diagnostic_folds,
        "notes": [
            "Every source is held out as a whole; contexts and seeds from a source never enter its training fold.",
            "Timing repeats are reduced to a median per action, context, seed and cold/warm state before seed averaging.",
            "R and outer cost are averaged over paired seeds; R/C is computed from those means.",
            "Safe features omit acceptance, EOS, commit path, output tokens, current-action measured latency and call counts.",
            "The Ranker P1 candidate count is a paid proposal result available after P1 ranking; it is not a probability.",
            "The decision is evaluated offline on held-out sources and does not establish online speed or deployment benefit.",
            "The accepted-count model predicts accepted proposed nodes from training data; proposal opportunities differ among K1, W1 and W2, so this target is an algorithm comparison, not a common token-progress measure.",
            "CPU component host times may overlap outer execution and are not added to device event times.",
            "The component proxy estimates recovery cost as predicted recovery-call count times a training-fold mean device-event cost per call.",
            "The recovery-event cost model estimates C from training-fold recovery counts, then substitutes predicted expected recovery calls at selection time. Its fitted recovery slopes are reported without sign constraints.",
        ],
    }


def _eos_summary(rows):
    eos_rows = [r for r in rows if r["eos"]]
    stages = Counter(str(r["eos_stage"] or "unspecified") for r in eos_rows)
    return {"records_after_repeat_aggregation": len(rows), "eos_records": len(eos_rows),
            "eos_fraction": len(eos_rows) / len(rows) if rows else None,
            "eos_stage_counts": dict(stages),
            "eos_R_mean": statistics.fmean(r["progress"] for r in eos_rows) if eos_rows else None,
            "non_eos_R_mean": statistics.fmean(r["progress"] for r in rows if not r["eos"])
                              if any(not r["eos"] for r in rows) else None}


def analyze(rows, *, scope="both", thermal="both", alpha=1.0,
            bootstrap_samples=2000, seed_bootstrap_samples=1000):
    raw_selected = [r for r in rows if scope == "both" or r["scope"] == scope]
    repeated, repeat_info = aggregate_repeats(raw_selected)
    mode_report = {}
    mode_names = ("all",) if thermal == "all" else (("cold", "warm") if thermal == "both" else (thermal,))
    for selected_scope in sorted(set(r["scope"] for r in repeated)):
        scope_rows = [r for r in repeated if r["scope"] == selected_scope]
        scope_report = {"rows_after_repeat_aggregation": len(scope_rows),
                        "cold_warm_eos": {}, "thermal_modes": {}}
        for mode in mode_names:
            if mode == "all":
                mode_rows_raw = [r for r in raw_selected if r["scope"] == selected_scope]
                mode_rows, _ = aggregate_repeats(mode_rows_raw)
            else:
                mode_rows = [r for r in scope_rows if r["thermal"] == mode]
            samples, sample_info = build_context_samples(mode_rows)
            scope_report["cold_warm_eos"][mode] = _eos_summary(mode_rows)
            action_costs = {}
            for action in ACTIONS:
                action_rows = [r for r in mode_rows if r["action"] == action]
                action_costs[action] = {
                    "records": len(action_rows),
                    "mean_selected_cost_ms": _mean_available(action_rows, "cost_ms"),
                    "mean_setup_ms": _mean_available(action_rows, "setup_ms"),
                    "mean_prefill_ms": _mean_available(action_rows, "prefill_ms"),
                    "mean_arm_constructor_ms": _mean_available(action_rows, "arm_constructor_ms"),
                    "mean_decode_outer_ms": _mean_available(action_rows, "decode_outer_ms"),
                    "mean_outer_with_arm_constructor_ms": _mean_available(
                        action_rows, "outer_wall_including_arm_constructor_ms"
                    ),
                    "mean_cold_graph_ms": _mean_available(action_rows, "cold_graph_ms"),
                    "mean_ranker_calls": _mean_available(action_rows, "ranker_calls"),
                    "mean_paid_P1_candidate_count": _mean_available(action_rows, "ranker_p1_candidate_count"),
                    "planned_node_count": ACTION_NODES[action],
                    "actual_node_count_histogram": dict(Counter(
                        "missing" if r["actual_node_count"] is None else str(int(r["actual_node_count"]))
                        for r in action_rows
                    )),
                    "actual_read_bucket_histogram": dict(Counter(
                        "missing" if r["actual_read_bucket"] is None else str(int(r["actual_read_bucket"]))
                        for r in action_rows
                    )),
                }
            scope_report["cold_warm_eos"][mode]["per_action_cost_components"] = action_costs
            scope_report["thermal_modes"][mode] = {
                "sample_coverage": sample_info,
                **_metrics_for_mode(samples, alpha, bootstrap_samples, seed_bootstrap_samples),
            }
        mode_report[selected_scope] = scope_report
    return {
        "schema_version": 1,
        "analysis": "source-held-out matched action choice from cheap pre-action features",
        "configuration": {"scope_filter": scope, "thermal_filter": thermal, "ridge_alpha": alpha,
                          "source_bootstrap_samples": bootstrap_samples,
                          "paired_seed_bootstrap_samples": seed_bootstrap_samples,
                          "feature_names": list(BASE_FEATURES) + [f"action_{a}" for a in ACTIONS] +
                                           [f"{a}_x_{suffix}" for a in ACTIONS
                                            for suffix in ("prefix", "last_gap", "read_bucket", "node_count")]},
        "input": {"normalized_rows": len(rows), "selected_rows": len(raw_selected),
                  "timing_repeat_aggregation": repeat_info},
        "scope_results": mode_report,
        "feature_policy": {
            "available_before_action": ["prefix_len", "cache_pos_before", "remaining_budget_before",
                                        "action_budget_tokens", "tree_read_bucket", "node_count",
                                        "last_position_rank_norm", "prior_position_rank_norm",
                                        "last_gap_rank_norm", "last known Feature category"],
            "excluded_outcomes": ["accepted_p1", "accepted_feature", "accepted_p2", "eos", "eos_stage",
                                  "commit_path", "actual_tokens", "round_wall_ms", "full_request_wall_ms",
                                  "proposed_p1", "proposed_p2", "candidate_features", "actual_node_count",
                                  "actual_read_bucket", "ranker_p1_candidate_count", "target_calls",
                                  "recovery_calls", "draft_calls", "component_device_ms"],
            "availability_semantics": "cache_pos_before is derived as prefix_len. Node count and read bucket are computed from each candidate action's planned shape, not the executed shape or any Ranker fallback result.",
            "ranker_candidate_count_semantics": "The input records include choose_position's P1 candidate count, but this first model leaves it out because its added call cost is not separately assigned.",
            "action_timing": "These first models use prompt-derived context and planned action shape only. Feature candidates and proposed Position values are excluded because they depend on the selected action.",
        },
    }


def self_test():
    synthetic = []
    # Three sources and four contexts each; one source-specific pattern is held out as a whole.
    for source_index in range(3):
        source = f"s{source_index}"
        for context_index, prefix in enumerate((64, 256, 512, 1024)):
            context = f"{source}-p{prefix}"
            for seed in range(8):
                for action in ACTIONS:
                    action_extra = {"K1": 0.0, "W1": 1.0, "W2": 2.0}[action]
                    progress = 2.0 + 0.0005 * prefix + action_extra * (0.2 + source_index * 0.1)
                    cost = 1.0 + 0.0003 * prefix + (0.1 if action == "K1" else 0.2 if action == "W1" else 0.45)
                    cost += (seed % 2) * 0.005
                    for repeat in range(3):
                        synthetic.append({
                            "source_id": source, "context_id": context, "prefix_len": prefix,
                            "seed": 46101 + seed, "scope": "single_round", "action_id": action,
                            "action_budget_tokens": 2 if action == "K1" else 4,
                            "repeat_index": repeat, "remaining_budget_before": 2 if action == "K1" else 4,
                            "R": progress, "eos": seed == 7 and context_index == 2 and action == "W2",
                            "eos_stage": "position_residual_p2" if seed == 7 and context_index == 2 and action == "W2" else None,
                            "round_wall_ms": cost + repeat * 0.001,
                            "cold_status": "cold" if repeat == 0 else "warm",
                            "tree_width": ACTION_WIDTH[action], "node_count": ACTION_NODES[action],
                            "tree_read_bucket": 512 if prefix < 512 else 2048,
                            "ranker_p1_candidate_count": 6 + (prefix // 256),
                            "accepted_p1": action != "K1", "accepted_feature": action == "W1",
                            "accepted_p2": action == "W1", "recovery_target_calls": int(action == "W2"),
                            "recovery_target_input_tokens": int(action == "W2"),
                            "component_device_ms": {"tree_verify": 0.1 * (ACTION_NODES[action] + 1),
                                                    "target_recovery": 0.03 * int(action == "W2")},
                        })
    normalized = [normalize_row(row, i + 1) for i, row in enumerate(synthetic)]
    report = analyze(normalized, scope="single_round", thermal="both", alpha=1.0, bootstrap_samples=100)
    for scope_data in report["scope_results"].values():
        for mode_data in scope_data["thermal_modes"].values():
            assert mode_data["matched"]["contexts"] == 12
            assert "fixed_w1" in mode_data["policies"]
            assert "full_separate_R_C" in mode_data["policies"]
            assert "recovery_event_cost" in mode_data["policies"]
            assert "component_device_proxy" in mode_data["policies"]
            assert "tree_verify_time_proxy" in mode_data["policies"]
            assert mode_data["policies"]["full_separate_R_C"]["paired_seed_uncertainty_vs_fixed_W1"][
                "common_seeds"
            ] == 8
            assert mode_data["finite_seed_action_selection"]["balanced_splits_evaluated"] > 0
            fold = next(iter(mode_data["diagnostic_source_folds"].values()))
            assert {"R", "C", "C_recovery_event_cost_predicted_Erec"}.issubset(fold)
            assert fold["C_recovery_event_cost_predicted_Erec"][
                "test_recovery_counts_used_for_prediction"
            ] is False
            assert {"R", "C", "C_recovery_event_cost_predicted_Erec"}.issubset(
                mode_data["heldout_progress_cost_prediction_errors"]
            )
    assert "accepted_p1" not in report["feature_policy"]["available_before_action"]
    assert "last_gap_rank_norm" in report["feature_policy"]["available_before_action"]
    seed_samples = []
    for action in ACTIONS:
        seed_r = {
            f"seed-{seed_index}": (
                1.0 if action == "K1" else 4.0 if action == "W1" else float(seed_index + 1)
            )
            for seed_index in range(8)
        }
        seed_samples.append({
            "source": "bootstrap-source", "context": "bootstrap-context",
            "action": action, "seed_R": seed_r,
            "seed_C": {seed_id: 1.0 for seed_id in seed_r},
        })
    seed_interval = _paired_seed_lift(
        seed_samples, {("bootstrap-source", "bootstrap-context"): "W2"},
        bootstrap_samples=300, seed=1234,
    )["paired_seed_bootstrap_95pct"]
    assert seed_interval is not None and seed_interval[0] < seed_interval[1], seed_interval

    source_selected = {
        f"source-{i}": {"tokens_per_ms": float(rate)}
        for i, rate in enumerate((0.5, 1.0, 2.0))
    }
    source_w1 = {f"source-{i}": {"tokens_per_ms": 1.0} for i in range(3)}
    source_interval = _paired_source_lift(
        source_selected, source_w1, samples=300, seed=4321,
    )["source_bootstrap_95pct"]
    assert source_interval is not None and source_interval[0] < source_interval[1], source_interval
    with tempfile.TemporaryDirectory(prefix="recovery_model_selftest_") as temporary:
        root = Path(temporary)
        prompt_path = root / "prompts.json"
        ranks_path = root / "ranks.json"
        records_path = root / "records.jsonl"
        prompt_path.write_text(json.dumps({"contexts": [{
            "context_id": "s0:p0004", "source_id": "s0", "prefix_len": 4,
            "prompt": [12096, 1, 8005, 3, 8012],
        }]}), encoding="utf-8")
        ranks = [-1] * 8
        ranks[1], ranks[3] = 1, 3
        ranks_path.write_text(json.dumps({"position_rank_map": ranks}), encoding="utf-8")
        records_path.write_text("", encoding="utf-8")
        prompt_map, prompt_info = _prompt_feature_map([records_path], prompt_path, ranks_path)
        row = normalize_row({
            "source_id": "s0", "context_id": "s0:p0004", "prefix_len": 4,
            "seed": 1, "scope": "single_round", "action_id": "W2",
            "action_budget_tokens": 4, "R": 4, "round_wall_ms": 1.0,
            "node_count": 1, "tree_read_bucket": 8, "cache_pos_before": 3,
            "ranker_p1_candidate_count": 12,
        }, 1, prompt_map["s0:p0004"])
        assert prompt_info["position_context_missing"] == 0
        assert row["features"]["last_position_rank_norm"] == 1.0
        assert abs(row["features"]["last_gap_rank_norm"] - (2.0 / 3.0)) < 1e-12
        assert row["features"]["node_count"] == 6.0
        assert row["features"]["cache_pos_before"] == 4.0
        assert abs(row["features"]["log_read_bucket"] - math.log1p(16.0)) < 1e-12
        assert "ranker_p1_candidate_count" not in row["features"]
        assert row["features"]["known_last_feature_8012"] == 1.0
    json.dumps(report, allow_nan=False)
    return {"passed": True, "synthetic_rows": len(synthetic),
            "sources": 3, "contexts": 12, "seeds_per_context_action": 8,
            "timing_repeats": 3, "thermal_modes": ["cold", "warm"],
            "equal_progress_repeats_aggregated": True, "heldout_by_source": True,
            "prompt_rank_features": True, "planned_action_shapes_only": True,
            "ranker_candidate_count_excluded": True,
            "bootstrap_resampling_non_degenerate": True,
            "seed_bootstrap_interval": seed_interval,
            "source_bootstrap_interval": source_interval}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", default=[], help="JSONL input; may be repeated")
    parser.add_argument("--prompts", type=Path, help="Known prompt table; defaults to prompts.json beside the input")
    parser.add_argument("--ranks", type=Path, help="Rank-map JSON, usually a plan.json with position_rank_map")
    parser.add_argument("--out", type=Path, help="write the complete JSON report here")
    parser.add_argument("--scope", choices=("single_round", "full_request", "both"), default="both")
    parser.add_argument("--thermal", choices=("all", "cold", "warm", "both"), default="both")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed-bootstrap-samples", type=int, default=1000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return
    if not args.input or args.out is None:
        parser.error("--input and --out are required unless --self-test is used")
    rows, input_info = read_jsonl(args.input, args.prompts, args.ranks)
    report = analyze(rows, scope=args.scope, thermal=args.thermal, alpha=args.alpha,
                     bootstrap_samples=args.bootstrap_samples,
                     seed_bootstrap_samples=args.seed_bootstrap_samples)
    report["input"].update(input_info)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "rows": len(rows), "scopes": list(report["scope_results"])}))


if __name__ == "__main__":
    main()
