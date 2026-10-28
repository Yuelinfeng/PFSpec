#!/usr/bin/env python3
"""Recompute local PF proposal and recovery evidence from saved records."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from summarize_runs import read_jsonl, summarize_feature_overlap, summarize_position_overlap


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "docs/project/artifacts"


def _mean(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def _median(values):
    values = list(values)
    return statistics.median(values) if values else None


def _new_graphs(row):
    events = []
    for key, value in row.items():
        if key.endswith("_delta") and isinstance(value, dict):
            events.extend(value.get("new_graph_events") or [])
    events.extend(row.get("new_graph_events") or [])
    return events


def _warm(row):
    return (not row.get("runtime_created", False)
            and not _new_graphs(row)
            and row.get("cold_status", "warm") == "warm")


def _stage_counts(rows):
    p1_trials = sum(row.get("accepted_p1") is not None for row in rows)
    p1_pass = sum(row.get("accepted_p1") is True for row in rows)
    feature_trials = sum(row.get("accepted_feature") is not None for row in rows)
    feature_pass = sum(row.get("accepted_feature") is True for row in rows)
    p2_trials = sum(row.get("accepted_p2") is not None for row in rows)
    p2_pass = sum(row.get("accepted_p2") is True for row in rows)
    return dict(
        rows=len(rows),
        P1=dict(passed=p1_pass, trials=p1_trials,
                rate=p1_pass / p1_trials if p1_trials else None),
        F1=dict(passed=feature_pass, trials=feature_trials,
                conditional_rate=feature_pass / feature_trials if feature_trials else None),
        P2=dict(passed=p2_pass, trials=p2_trials,
                conditional_rate=p2_pass / p2_trials if p2_trials else None),
        recoveries=sum(int(row.get("recovery_calls", 0)) for row in rows),
        mean_generated_per_round=_mean(float(row.get("R", 0)) for row in rows),
    )


def _single_round(path):
    rows = read_jsonl(path)
    actions = sorted({str(row["action_id"]) for row in rows})
    result = dict(input=str(path), row_count=len(rows), repeat_indices=dict(
        sorted(Counter(int(row.get("repeat_index", 0)) for row in rows).items())))
    result["actions"] = {}
    for action in actions:
        action_rows = [row for row in rows if str(row["action_id"]) == action]
        outcomes = [row for row in action_rows if int(row.get("repeat_index", 0)) == 0]
        groups = defaultdict(list)
        for row in action_rows:
            if _warm(row):
                key = (row.get("context_id"), row.get("prefix_len"), row.get("seed"))
                groups[key].append(float(row["round_wall_ms"]))
        medians = [_median(values) for values in groups.values()]
        summary = dict(
            outcome_rows=len(outcomes),
            warm_timing_rows=sum(len(values) for values in groups.values()),
            warm_state_seed_groups=len(groups),
            warm_median_group_ms_mean=_mean(medians),
            warm_median_group_ms_median=_median(medians),
            recovery_calls=sum(int(row.get("recovery_calls", 0)) for row in outcomes),
            mean_R=_mean(float(row.get("R", 0)) for row in outcomes),
        )
        if action in {"W1", "W2"}:
            summary["stages"] = _stage_counts(outcomes)
        by_prefix = {}
        for prefix in sorted({int(row["prefix_len"]) for row in outcomes}):
            local = [row for row in outcomes if int(row["prefix_len"]) == prefix]
            by_prefix[str(prefix)] = _stage_counts(local)
        summary["by_prefix"] = by_prefix
        result["actions"][action] = summary
    return result


def _component_cost(path):
    rows = read_jsonl(path)
    by_action = {}
    for action in sorted({str(row["action_id"]) for row in rows}):
        all_rows = [row for row in rows if str(row["action_id"]) == action]
        warm = [row for row in all_rows if _warm(row)]
        recovered = [row for row in warm if int(row.get("recovery_calls", 0)) > 0]
        no_recovery = [row for row in warm if int(row.get("recovery_calls", 0)) == 0]
        event_values = [float(row["component_device_ms"]["target_recovery"])
                        for row in recovered
                        if "target_recovery" in row.get("component_device_ms", {})]
        verify_values = [float(row["component_device_ms"]["tree_verify"])
                         for row in warm
                         if "tree_verify" in row.get("component_device_ms", {})]
        by_action[action] = dict(
            all_rows=len(all_rows), warm_rows=len(warm),
            warm_recovery_rows=len(recovered),
            target_recovery_event_ms_median=_median(event_values),
            target_recovery_event_ms_mean=_mean(event_values),
            tree_verify_event_ms_median=_median(verify_values),
            round_wall_ms_median_with_recovery=_median(
                float(row["round_wall_ms"]) for row in recovered),
            round_wall_ms_median_without_recovery=_median(
                float(row["round_wall_ms"]) for row in no_recovery),
        )
    return dict(input=str(path), row_count=len(rows), by_action=by_action)


def _full_decode(path):
    rows = read_jsonl(path)
    methods = sorted({str(row["method"]) for row in rows})
    result = dict(input=str(path), row_count=len(rows), methods={})
    for method in methods:
        local = [row for row in rows if str(row["method"]) == method]
        counters = [row.get("counters") or {} for row in local]
        total_wall = math.fsum(float(row["outer_wall_seconds"]) for row in local)
        total_tokens = sum(int(row["generated_tokens"]) for row in local)
        summary = dict(
            rows=len(local), source_count=len({str(row["source"]) for row in local}),
            generated_tokens=total_tokens,
            outer_wall_seconds=total_wall,
            pooled_tokens_per_second=total_tokens / total_wall if total_wall else None,
        )
        counter_names = (
            "target_calls", "target_input_tokens", "tree_verify_calls",
            "candidate_tree_nodes", "target_tree_nodes", "feature_attempts",
            "feature_branch_hits", "recovery_calls", "k1_steps",
            "position_fallbacks",
        )
        for name in counter_names:
            summary[name] = sum(int(counter.get(name, 0)) for counter in counters)
        accepted_tree = []
        accepted_tail = []
        for counter in counters:
            values = counter.get("accepted_lengths", [])
            k1_steps = max(0, int(counter.get("k1_steps", 0)))
            if k1_steps:
                accepted_tail.extend(int(value) for value in values[-k1_steps:])
                accepted_tree.extend(int(value) for value in values[:-k1_steps])
            else:
                accepted_tree.extend(int(value) for value in values)
        if accepted_tree and summary["tree_verify_calls"]:
            if len(accepted_tree) != summary["tree_verify_calls"]:
                raise ValueError(
                    f"{method}: tree accepted lengths differ from tree calls"
                )
            hist = Counter(accepted_tree)
            p1_pass = sum(value >= 1 for value in accepted_tree)
            feature_pass = sum(value >= 2 for value in accepted_tree)
            p2_pass = sum(value >= 3 for value in accepted_tree)
            summary["tree_rounds"] = len(accepted_tree)
            summary["accepted_length_counts"] = dict(sorted(hist.items()))
            summary["stages"] = dict(
                P1=dict(passed=p1_pass, trials=len(accepted_tree),
                        rate=p1_pass / len(accepted_tree)),
                F1=dict(passed=feature_pass, trials=p1_pass,
                        conditional_rate=feature_pass / p1_pass if p1_pass else None),
                P2=dict(passed=p2_pass, trials=feature_pass,
                        conditional_rate=p2_pass / feature_pass if feature_pass else None),
                tree_P1_rejections=sum(value == 0 for value in accepted_tree),
                tree_P2_rejections=sum(value == 2 for value in accepted_tree),
            )
            summary["expected_tree_recovery_events"] = (
                summary["stages"]["tree_P1_rejections"]
                + summary["stages"]["tree_P2_rejections"]
            )
            summary["tokens_per_tree_round_from_acceptance"] = (
                2.0 + 2.0 * feature_pass / len(accepted_tree)
            )
        if accepted_tail and summary["tree_verify_calls"]:
            summary["k1_tail_accepted_length_counts"] = dict(
                sorted(Counter(accepted_tail).items()))
        result["methods"][method] = summary

    w1 = result["methods"].get("tree_w1")
    w2 = result["methods"].get("tree_w2")
    if w1 and w2:
        result["W2_vs_W1"] = dict(
            wall_ratio=w2["outer_wall_seconds"] / w1["outer_wall_seconds"],
            target_calls_ratio=w2["target_calls"] / w1["target_calls"],
            target_input_tokens_ratio=(
                w2["target_input_tokens"] / w1["target_input_tokens"]),
            target_tree_nodes_ratio=(
                w2["target_tree_nodes"] / w1["target_tree_nodes"]),
        )
    return result


def build_summary():
    dev_path = ARTIFACTS / "recovery_model_20261021/dev_run01/records.jsonl"
    component_path = ARTIFACTS / "recovery_model_20261021/components_run01/records.jsonl"
    full_path = ARTIFACTS / "feature_tree_20261021/main_run02/records.jsonl"
    position_path = ARTIFACTS / "spec_next_20261022/raw/position_overlap_run01.json"
    feature_path = ARTIFACTS / "spec_next_20261022/raw/feature_reuse_overlap_run01.json"
    dev = _single_round(dev_path)
    components = _component_cost(component_path)
    full = _full_decode(full_path)
    position = summarize_position_overlap(position_path)
    feature = summarize_feature_overlap(feature_path)

    full_w1 = full["methods"]["tree_w1"]
    recovery_ms = components["by_action"]["W1"]["target_recovery_event_ms_median"]
    position_backup = position["metrics"]["backup_mass"]["source_equal"]
    point_tree_reject = full_w1["stages"]["tree_P1_rejections"]
    tree_rounds = full_w1["tree_rounds"]
    throughput_per_ms = full_w1["pooled_tokens_per_second"] / 1000.0
    feature_metrics = feature["metrics"]
    return dict(
        scope="Read-only CPU recomputation from saved JSONL and JSON rows; no new model or GPU run.",
        single_round_dev=dev,
        component_events=components,
        full_decode=full,
        position_overlap=position,
        feature_reuse_overlap=feature,
        cost_bounds=dict(
            W1_full_decode_recovery_calls=full_w1["recovery_calls"],
            W1_tree_P1_rejections=point_tree_reject,
            W1_tree_rounds=tree_rounds,
            W1_tree_P1_reject_rate=(point_tree_reject / tree_rounds),
            warm_single_round_recovery_event_median_ms=recovery_ms,
            all_recovery_event_median_estimate_ms=(
                full_w1["recovery_calls"] * recovery_ms),
            all_recovery_fraction_of_W1_outer_wall=(
                full_w1["recovery_calls"] * recovery_ms
                / full_w1["outer_wall_seconds"] / 1000.0),
            backup_tree_savable_event_ceiling_ms=(
                point_tree_reject * recovery_ms),
            backup_tree_outer_wall_fraction_ceiling=(
                point_tree_reject * recovery_ms
                / full_w1["outer_wall_seconds"] / 1000.0),
            initial_prefix_backup_mass_per_tree_round=position_backup,
            initial_prefix_break_even_added_cost_ms=(position_backup * recovery_ms),
            generated_state_backup_mass_upper_bound=(point_tree_reject / tree_rounds),
            generated_state_break_even_added_cost_ms_upper_bound=(
                point_tree_reject / tree_rounds * recovery_ms),
            tree_verify_event_median_ms=(
                components["by_action"]["W1"]["tree_verify_event_ms_median"]),
            deepest_chain_reach_rate_after_P2=(
                full_w1["stages"]["P2"]["passed"] / tree_rounds),
            deepest_chain_two_token_cost_budget_ms_if_always_reached_and_accepted=(
                2.0 * full_w1["stages"]["P2"]["passed"] / tree_rounds
                / throughput_per_ms),
            feature_mix_lambda_025_overlap_gain=(
                feature_metrics["0.25"]["source_equal_overlap"]
                - feature_metrics["0.0"]["source_equal_overlap"]),
            feature_mix_lambda_025_p1_joint_gain=(
                feature_metrics["0.25"]["source_equal_p1_joint"]
                - feature_metrics["0.0"]["source_equal_p1_joint"]),
            feature_mix_lambda_025_initial_prefix_extra_tokens_per_round=(
                2.0 * (feature_metrics["0.25"]["source_equal_p1_joint"]
                       - feature_metrics["0.0"]["source_equal_p1_joint"])),
        ),
    )


def main():
    result = build_summary()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
