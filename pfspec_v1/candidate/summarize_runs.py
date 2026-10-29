#!/usr/bin/env python3
"""Summarize matched decoding runs and saved proposal-distribution rows."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_BOOTSTRAP = 10000
DEFAULT_SEED = 402101


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def _arm_name(row):
    if row.get("arm") is not None:
        return str(row["arm"])
    if row.get("verification_role") is not None:
        return str(row["verification_role"])
    method = row.get("method")
    mode = row.get("position_mode")
    if method == "prob" and mode:
        return str(mode)
    if row.get("execution") and method:
        return f"{row['execution']}:{method}"
    action = row.get("action_id")
    if action and action != method:
        return str(action)
    if method is not None:
        return str(method)
    if action is not None:
        return str(action)
    raise ValueError("record has no arm, method, or action_id")


def _pair_key(row):
    pair = row.get("pair_id")
    if pair is None:
        fields = (row.get("source", row.get("source_id", "unknown")),
                  row.get("prefix"), row.get("seed"))
        pair = "|".join(str(value) for value in fields)
    return str(pair), str(row.get("budget", ""))


def _source_name(row):
    source = row.get("source", row.get("source_id", row.get("scene")))
    return str(source) if source is not None else "unknown"


def _tokens_per_second(row):
    value = row.get("tokens_per_second")
    if value is not None and float(value) > 0.0:
        return float(value)
    count = row.get("generated_tokens")
    seconds = row.get("outer_wall_seconds")
    if count is None or seconds is None or float(seconds) <= 0.0:
        return None
    return float(count) / float(seconds)


def _generated(row):
    value = row.get("generated_tokens")
    if value is not None:
        return int(value)
    output = row.get("output_tokens", row.get("output"))
    if isinstance(output, list):
        return len(output)
    return None


def _has_new_graph(row):
    if bool(row.get("runtime_created")):
        return True
    try:
        if float(row.get("graph_setup_ms", 0.0) or 0.0) > 0.0:
            return True
    except (TypeError, ValueError):
        pass
    for key, value in row.items():
        if not key.endswith("_delta") or not isinstance(value, dict):
            continue
        events = value.get("new_graph_events") or []
        if events:
            return True
    return bool(row.get("new_graph_events"))


def _graph_observation(row):
    calls = Counter()
    fallbacks = Counter()
    event_count = 0
    for key, value in row.items():
        if not key.endswith("_delta") or not isinstance(value, dict):
            continue
        for name, count in (value.get("calls") or {}).items():
            if "graph" in str(name).lower():
                calls[str(name)] += int(count)
        for name, count in (value.get("fallbacks") or {}).items():
            fallbacks[str(name)] += int(count)
        event_count += len(value.get("new_graph_events") or [])
    event_count += len(row.get("new_graph_events") or [])
    return calls, fallbacks, event_count


def _log_gm(rows, bootstrap, seed):
    by_source = defaultdict(list)
    skipped = 0
    for row in rows:
        ratio = row.get("ratio")
        if ratio is None or not math.isfinite(float(ratio)) or float(ratio) <= 0.0:
            skipped += 1
            continue
        by_source[str(row["source"])].append(math.log(float(ratio)))
    source_logs = {
        source: statistics.fmean(values)
        for source, values in by_source.items() if values
    }
    if not source_logs:
        return dict(pair_count=len(rows), source_count=0, ratio=None, ci95=None,
                    per_source={}, skipped_pairs=skipped)
    names = sorted(source_logs)
    point = math.exp(statistics.fmean(source_logs.values()))
    interval = None
    if len(names) > 1 and int(bootstrap) > 0:
        rng = random.Random(int(seed))
        samples = []
        for _ in range(int(bootstrap)):
            selected = [source_logs[rng.choice(names)] for _ in names]
            samples.append(math.exp(statistics.fmean(selected)))
        samples.sort()
        lo_index = max(0, int(math.floor(0.025 * (len(samples) - 1))))
        hi_index = min(len(samples) - 1,
                       int(math.ceil(0.975 * (len(samples) - 1))))
        interval = [samples[lo_index], samples[hi_index]]
    return dict(pair_count=len(rows), source_count=len(source_logs), ratio=point,
                ci95=interval,
                per_source={name: math.exp(value) for name, value in source_logs.items()},
                skipped_pairs=skipped)


def _row_full(row):
    generated = _generated(row)
    budget = row.get("budget")
    if generated is None or budget is None:
        return False
    return generated == int(budget)


def _pair_ratio_rows(pair_groups, arm, reference, *, all_arms=None,
                     full_only=False, no_new_graph=False):
    selected = []
    for (pair_id, _budget), arms in pair_groups.items():
        left = arms.get(arm)
        right = arms.get(reference)
        if left is None or right is None:
            continue
        if full_only and not (_row_full(left) and _row_full(right)):
            continue
        if all_arms is not None:
            if any(name not in arms for name in all_arms):
                continue
            if full_only and not all(_row_full(arms[name]) for name in all_arms):
                continue
        if no_new_graph:
            checked = (all_arms if all_arms is not None else (arm, reference))
            if any(_has_new_graph(arms[name]) for name in checked):
                continue
        left_tps = _tokens_per_second(left)
        right_tps = _tokens_per_second(right)
        if left_tps is None or right_tps is None or right_tps <= 0.0:
            continue
        left_source = _source_name(left)
        right_source = _source_name(right)
        selected.append(dict(pair_id=pair_id, source=left_source,
                             source_match=(left_source == right_source),
                             ratio=left_tps / right_tps))
    return selected


def _resolve_reference(arms, requested):
    if requested:
        if requested not in arms:
            raise ValueError(f"reference arm {requested!r} not found; found {arms}")
        return requested
    aliases = ("target", "target_only", "original_target", "original_target:target")
    for name in aliases:
        if name in arms:
            return name
    for name in arms:
        lower = name.lower()
        if "original_target" in lower or lower.endswith(":target"):
            return name
    return None


def _counter_summary(rows):
    totals = Counter()
    graph_calls = Counter()
    graph_fallbacks = Counter()
    graph_events = 0
    created = 0
    setup_rows = 0
    output_lengths = []
    eos_rows = 0
    for row in rows:
        counters = row.get("counters") or row.get("stats") or {}
        if isinstance(counters, dict):
            for key, value in counters.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    totals[str(key)] += value
        calls, fallbacks, events = _graph_observation(row)
        graph_calls.update(calls)
        graph_fallbacks.update(fallbacks)
        graph_events += events
        created += int(bool(row.get("runtime_created")))
        setup_rows += int(_has_new_graph(row))
        count = _generated(row)
        if count is not None:
            output_lengths.append(count)
        eos_rows += int(bool(row.get("eos")))

    result = dict(rows=len(rows), generated_tokens=sum(output_lengths),
                  eos_rows=eos_rows, runtime_created_rows=created,
                  rows_with_new_graph=setup_rows,
                  graph_events=graph_events,
                  graph_calls=dict(sorted(graph_calls.items())),
                  graph_fallbacks=dict(sorted(graph_fallbacks.items())))
    names = ("target_calls", "target_input_tokens", "tree_verify_calls",
             "candidate_tree_nodes", "target_tree_nodes", "position_verified",
             "position_accepted", "feature_attempts", "feature_branch_hits",
             "feature_verified", "feature_accepted", "recovery_calls", "k1_steps")
    for name in names:
        if name in totals:
            result[name] = totals[name]
    if totals.get("position_verified", 0):
        result["position_acceptance_rate_all_stages"] = (
            totals.get("position_accepted", 0) / totals["position_verified"]
        )
    attempts = totals.get("feature_attempts", 0)
    if attempts:
        result["feature_hit_per_proposal_attempt"] = (
            totals.get("feature_branch_hits", 0) / attempts
        )
    if totals.get("tree_verify_calls", 0):
        result["recovery_calls_per_tree_verify"] = (
            totals.get("recovery_calls", 0) / totals["tree_verify_calls"]
        )
        result["feature_hits_per_tree_verify"] = (
            totals.get("feature_branch_hits", 0) / totals["tree_verify_calls"]
        )
    tree_lengths = []
    tail_lengths = []
    for row in rows:
        counters = row.get("counters") or row.get("stats") or {}
        if not isinstance(counters, dict):
            continue
        accepted = counters.get("accepted_lengths")
        if not isinstance(accepted, list):
            continue
        k1_steps = max(0, int(counters.get("k1_steps", 0) or 0))
        if k1_steps:
            tail_lengths.extend(int(value) for value in accepted[-k1_steps:])
            tree_lengths.extend(int(value) for value in accepted[:-k1_steps])
        else:
            tree_lengths.extend(int(value) for value in accepted)
    if tree_lengths and totals.get("tree_verify_calls", 0):
        result["tree_round_count_from_accepted_lengths"] = len(tree_lengths)
        result["accepted_length_counts"] = dict(
            sorted(Counter(tree_lengths).items()))
        p1_pass = sum(value >= 1 for value in tree_lengths)
        feature_pass = sum(value >= 2 for value in tree_lengths)
        p2_pass = sum(value >= 3 for value in tree_lengths)
        result["tree_acceptance_stages"] = dict(
            p1_pass=p1_pass,
            p1_trials=len(tree_lengths),
            p1_rate=p1_pass / len(tree_lengths),
            feature_any_pass=feature_pass,
            feature_trials=p1_pass,
            feature_given_p1_rate=feature_pass / p1_pass if p1_pass else None,
            p2_pass=p2_pass,
            p2_trials=feature_pass,
            p2_given_feature_rate=p2_pass / feature_pass if feature_pass else None,
            tree_p1_rejections=sum(value == 0 for value in tree_lengths),
            tree_p2_rejections=sum(value == 2 for value in tree_lengths),
        )
    if tail_lengths:
        result["k1_tail_accepted_length_counts"] = dict(
            sorted(Counter(tail_lengths).items()))
    return result


def summarize_run(run_path: Path, *, reference=None, bootstrap=DEFAULT_BOOTSTRAP,
                  seed=DEFAULT_SEED):
    run_path = run_path.resolve()
    records_path = run_path if run_path.is_file() else run_path / "records.jsonl"
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    run_dir = records_path.parent
    rows = read_jsonl(records_path)
    if not rows:
        raise ValueError(f"No records in {records_path}")
    plan_path = run_dir / "plan.json"
    pipeline_path = run_dir / "pipeline.json"
    plan = read_json(plan_path) if plan_path.exists() else {}
    budgets = plan.get("budgets") or []
    default_budget = int(budgets[0]) if len(budgets) == 1 else None
    if default_budget is None and plan.get("budget") is not None:
        default_budget = int(plan["budget"])
    if default_budget is not None:
        for row in rows:
            row.setdefault("budget", default_budget)
    arms = sorted({_arm_name(row) for row in rows})
    ref = _resolve_reference(arms, reference)
    groups = defaultdict(dict)
    duplicates = []
    by_arm_rows = defaultdict(list)
    for row in rows:
        name = _arm_name(row)
        key = _pair_key(row)
        if name in groups[key]:
            duplicates.append(dict(pair_id=key[0], budget=key[1], arm=name))
        groups[key][name] = row
        by_arm_rows[name].append(row)

    complete = {key: group for key, group in groups.items()
                if all(name in group for name in arms)}
    incomplete = [dict(pair_id=key[0], budget=key[1], arms=sorted(group))
                  for key, group in groups.items() if len(group) != len(arms)]
    summary = dict(
        run=str(run_dir), records=str(records_path), row_count=len(rows),
        arms=arms, reference=ref, complete_pair_count=len(complete),
        expected_runs=None, expected_pair_count=None,
        duplicate_arm_rows=duplicates, incomplete_pairs=incomplete,
        counter_summary={name: _counter_summary(by_arm_rows[name]) for name in arms},
        comparisons={}, output_parity={},
    )
    if plan:
        summary["expected_runs"] = plan.get("expected_runs")
        summary["expected_pair_count"] = plan.get("prompt_count")
        summary["plan_methods"] = plan.get("methods")
        summary["plan_sources"] = plan.get("sources")
    if pipeline_path.exists():
        pipeline = read_json(pipeline_path)
        summary["pipeline_status"] = pipeline.get("status")
        summary["pipeline_expected_runs"] = pipeline.get("expected_runs")
        summary["pipeline_completed_runs"] = pipeline.get("completed_runs")

    if ref is None:
        summary["comparison_note"] = (
            "No original Target arm was found. Pass --reference to select a denominator."
        )
        return summary

    common_full = {
        key: group for key, group in complete.items()
        if all(_row_full(group[name]) for name in arms)
    }
    common_stable = {
        key: group for key, group in complete.items()
        if all(not _has_new_graph(group[name]) for name in arms)
    }
    common_full_stable = {
        key: group for key, group in complete.items()
        if all(_row_full(group[name]) and not _has_new_graph(group[name])
               for name in arms)
    }

    for arm in arms:
        if arm == ref:
            continue
        pools = {
            "all_complete": _pair_ratio_rows(complete, arm, ref),
            "both_full_budget": _pair_ratio_rows(
                complete, arm, ref, full_only=True),
            "all_arms_full_budget": _pair_ratio_rows(
                complete, arm, ref, all_arms=arms, full_only=True),
            "both_without_new_graphs": _pair_ratio_rows(
                complete, arm, ref, no_new_graph=True),
            "all_arms_without_new_graphs": _pair_ratio_rows(
                complete, arm, ref, all_arms=arms, no_new_graph=True),
            "all_arms_full_and_without_new_graphs": _pair_ratio_rows(
                complete, arm, ref, all_arms=arms, full_only=True,
                no_new_graph=True),
        }
        summary["comparisons"][arm] = {
            name: _log_gm(values, bootstrap, seed)
            for name, values in pools.items()
        }

        matched = [group for group in complete.values()
                   if arm in group and ref in group]
        output_equal = 0
        count_equal = 0
        eos_equal = 0
        eos_compared = 0
        count_compared = 0
        seq_compared = 0
        for group in matched:
            left, right = group[arm], group[ref]
            left_count, right_count = _generated(left), _generated(right)
            if left_count is not None and right_count is not None:
                count_compared += 1
                count_equal += int(left_count == right_count)
            left_output = left.get("output_tokens", left.get("output"))
            right_output = right.get("output_tokens", right.get("output"))
            if left_output is not None and right_output is not None:
                seq_compared += 1
                output_equal += int(left_output == right_output)
            if "eos" in left and "eos" in right:
                eos_compared += 1
                eos_equal += int(bool(left["eos"]) == bool(right["eos"]))
        summary["output_parity"][arm] = dict(
            pair_count=len(matched),
            generated_count_pairs=count_compared,
            same_generated_count=count_equal,
            output_sequence_pairs=seq_compared,
            same_output_sequence=output_equal,
            eos_pairs=eos_compared, same_eos=eos_equal,
        )

    summary["subset_counts"] = dict(
        complete_pairs=len(complete),
        all_arms_full_budget=len(common_full),
        all_arms_without_new_graphs=len(common_stable),
        all_arms_full_and_without_new_graphs=len(common_full_stable),
    )
    summary["bootstrap"] = dict(method="source-cluster percentile bootstrap",
                                samples=int(bootstrap), seed=int(seed))
    return summary


def _normalize(values, label):
    result = [float(value) for value in values]
    if not result or any(not math.isfinite(value) or value < 0.0 for value in result):
        raise ValueError(f"{label} must be finite, nonnegative, and nonempty")
    mass = math.fsum(result)
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"{label} must have positive finite mass")
    return [value / mass for value in result], mass


def _source_equal(rows, field):
    by_source = defaultdict(list)
    for row in rows:
        by_source[str(row["source"])].append(float(row[field]))
    per_source = {source: statistics.fmean(values)
                  for source, values in by_source.items() if values}
    return mean(per_source.values()), per_source


def summarize_feature_overlap(path: Path):
    data = read_json(path)
    rows = data.get("rows") or []
    if not rows:
        raise ValueError(f"No rows in {path}")
    computed = []
    max_norm_error = 0.0
    max_reported_error = 0.0
    for index, row in enumerate(rows):
        q_key = "q" if "q" in row else "q_mlp"
        previous_key = "previous_policy" if "previous_policy" in row else "p_prev"
        target_key = "target_policy" if "target_policy" in row else "p_current"
        q, q_mass = _normalize(row[q_key], f"row {index} {q_key}")
        previous, previous_mass = _normalize(row[previous_key],
                                             f"row {index} {previous_key}")
        target, target_mass = _normalize(row[target_key],
                                         f"row {index} {target_key}")
        if not (len(q) == len(previous) == len(target)):
            raise ValueError(f"row {index} feature vectors have different lengths")
        max_norm_error = max(max_norm_error, abs(q_mass - 1.0),
                             abs(previous_mass - 1.0), abs(target_mass - 1.0))
        p1_mass = float(row.get("p1_mass", row.get("p1_acceptance", 1.0)))
        if not math.isfinite(p1_mass) or not 0.0 <= p1_mass <= 1.0 + 1e-6:
            raise ValueError(f"row {index} has invalid P1 mass")
        lambdas = sorted((float(key) for key in (row.get("overlaps") or {"0": 0})),
                         key=float)
        overlaps = {}
        joints = {}
        for weight in lambdas:
            mixed = [(1.0 - weight) * a + weight * b
                     for a, b in zip(q, previous)]
            overlap = math.fsum(min(p, q_value)
                                for p, q_value in zip(target, mixed))
            key = str(weight)
            overlaps[key] = overlap
            joints[key] = p1_mass * overlap
            stored = (row.get("overlaps") or {}).get(key)
            if stored is not None:
                max_reported_error = max(max_reported_error,
                                         abs(float(stored) - overlap))
        computed.append(dict(source=str(row.get("source", "unknown")),
                             p1_mass=p1_mass, overlaps=overlaps, joints=joints,
                             q_mass=q_mass, previous_mass=previous_mass,
                             target_mass=target_mass,
                             prefix=row.get("prefix")))

    weights = sorted(computed[0]["overlaps"], key=float)
    metrics = {}
    per_source = {}
    source_names = sorted({row["source"] for row in computed})
    for source in source_names:
        local = [row for row in computed if row["source"] == source]
        per_source[source] = dict(rows=len(local),
                                  p1_mass_mean=statistics.fmean(r["p1_mass"] for r in local),
                                  overlap={w: statistics.fmean(r["overlaps"][w] for r in local)
                                           for w in weights},
                                  p1_joint={w: statistics.fmean(r["joints"][w] for r in local)
                                            for w in weights},
                                  p1_weighted_overlap={
                                      w: (sum(r["joints"][w] for r in local)
                                          / sum(r["p1_mass"] for r in local)
                                          if sum(r["p1_mass"] for r in local) else None)
                                      for w in weights})
    for weight in weights:
        source_overlap = [per_source[name]["overlap"][weight]
                          for name in source_names]
        source_joint = [per_source[name]["p1_joint"][weight]
                        for name in source_names]
        source_conditional = [per_source[name]["p1_weighted_overlap"][weight]
                              for name in source_names
                              if per_source[name]["p1_weighted_overlap"][weight] is not None]
        row_overlap = [r["overlaps"][weight] for r in computed]
        row_joint = [r["joints"][weight] for r in computed]
        metrics[weight] = dict(
            source_equal_overlap=statistics.fmean(source_overlap),
            row_mean_overlap=statistics.fmean(row_overlap),
            source_equal_p1_joint=statistics.fmean(source_joint),
            row_mean_p1_joint=statistics.fmean(row_joint),
            source_equal_p1_weighted_overlap=(
                statistics.fmean(source_conditional) if source_conditional else None),
        )
    reference_weight = "0.0" if "0.0" in metrics else weights[0]
    directions = {}
    direction_by_source = {}
    for weight in weights:
        direction_by_source[weight] = {
            source: dict(
                overlap_delta=(per_source[source]["overlap"][weight]
                               - per_source[source]["overlap"][reference_weight]),
                p1_joint_delta=(per_source[source]["p1_joint"][weight]
                                - per_source[source]["p1_joint"][reference_weight]),
            )
            for source in source_names
        }
        directions[weight] = dict(
            source_overlap_improved=sum(
                per_source[s]["overlap"][weight] > per_source[s]["overlap"][reference_weight]
                for s in source_names),
            source_overlap_declined=sum(
                per_source[s]["overlap"][weight] < per_source[s]["overlap"][reference_weight]
                for s in source_names),
            source_p1_joint_improved=sum(
                per_source[s]["p1_joint"][weight] > per_source[s]["p1_joint"][reference_weight]
                for s in source_names),
            source_p1_joint_declined=sum(
                per_source[s]["p1_joint"][weight] < per_source[s]["p1_joint"][reference_weight]
                for s in source_names),
        )
    reported_source = data.get("source_mean") or {}
    max_source_mean_error = 0.0
    for weight in weights:
        stored = reported_source.get(weight, {})
        if not stored:
            continue
        max_source_mean_error = max(
            max_source_mean_error,
            abs(float(stored.get("overlap", metrics[weight]["source_equal_overlap"]))
                - metrics[weight]["source_equal_overlap"]),
            abs(float(stored.get("joint", metrics[weight]["source_equal_p1_joint"]))
                - metrics[weight]["source_equal_p1_joint"]),
        )
    return dict(kind="feature_policy_mixture", input=str(path), rows=len(rows),
                sources=len(source_names), mixture_definition="(1-lambda)*q_mlp + lambda*pF_prev",
                normalized_vector_max_abs_mass_error=max_norm_error,
                recomputed_overlap_max_abs_difference_from_rows=max_reported_error,
                recomputed_source_mean_max_abs_difference=max_source_mean_error,
                metrics=metrics, directions_vs_lambda_0=directions,
                direction_by_source_vs_lambda_0=direction_by_source,
                per_source=per_source,
                reported_source_mean=data.get("source_mean"),
                recomputed_matches_reported=(max_reported_error <= 2e-6))


def summarize_position_overlap(path: Path):
    data = read_json(path)
    rows = data.get("rows") or []
    if not rows:
        raise ValueError(f"No rows in {path}")
    computed = []
    max_policy_norm_error = 0.0
    max_candidate_q_norm_error = 0.0
    for index, row in enumerate(rows):
        target, target_mass = _normalize(row["target_policy"],
                                         f"row {index} target_policy")
        ids = [int(value) for value in row.get("ids", [])]
        scores = [float(value) for value in row.get("scores", [])]
        proposal = [float(value) for value in row.get("q", [])]
        if len(ids) != len(scores) or len(ids) != len(proposal):
            raise ValueError(f"row {index} candidate arrays differ in length")
        if len(set(ids)) != len(ids) or any(not 0 <= token < len(target) for token in ids):
            raise ValueError(f"row {index} has duplicate or invalid candidate ids")
        proposal, proposal_mass = _normalize(proposal, f"row {index} q")
        max_policy_norm_error = max(max_policy_norm_error, abs(target_mass - 1.0))
        max_candidate_q_norm_error = max(max_candidate_q_norm_error,
                                        abs(proposal_mass - 1.0))
        point = int(row["point"])
        point_mass = target[point]
        support_mass = math.fsum(target[token] for token in ids)
        overlap = math.fsum(min(target[token], q_value)
                            for token, q_value in zip(ids, proposal))
        alternatives = [i for i, token in enumerate(ids) if token != point]
        if alternatives:
            best_alt = max(alternatives, key=lambda i: scores[i])
            backup_token = ids[best_alt]
            backup_mass = target[backup_token]
            backup_score = scores[best_alt]
        else:
            backup_token = None
            backup_mass = 0.0
            backup_score = None
        reject_mass = max(0.0, 1.0 - point_mass)
        computed.append(dict(source=str(row.get("source", "unknown")),
                             point_acceptance=point_mass,
                             reject_mass=max(0.0, 1.0 - point_mass),
                             probability_overlap=overlap,
                             candidate_mass=support_mass,
                             backup_mass=backup_mass,
                             backup_conditional=(backup_mass / reject_mass
                                                 if reject_mass > 0 else 0.0),
                             candidate_count=len(ids), backup_token=backup_token,
                             backup_score=backup_score))

    source_names = sorted({row["source"] for row in computed})
    per_source = {}
    for source in source_names:
        local = [row for row in computed if row["source"] == source]
        per_source[source] = dict(
            rows=len(local),
            **{key: statistics.fmean(row[key] for row in local)
               for key in ("point_acceptance", "probability_overlap", "candidate_mass",
                           "backup_mass")},
            reject_mass=statistics.fmean(1.0 - row["point_acceptance"]
                                         for row in local),
            backup_conditional=(
                sum(row["backup_mass"] for row in local)
                / sum(1.0 - row["point_acceptance"] for row in local)
                if sum(1.0 - row["point_acceptance"] for row in local) > 0 else 0.0),
        )
    metrics = {}
    for key in ("point_acceptance", "probability_overlap", "candidate_mass",
                "backup_mass", "reject_mass", "backup_conditional"):
        metrics[key] = dict(
            source_equal=statistics.fmean(per_source[s][key] for s in source_names),
            row_mean=statistics.fmean(row[key] for row in computed),
        )
    reject_average = statistics.fmean(per_source[s]["reject_mass"]
                                      for s in source_names)
    reject_row_mean = statistics.fmean(1.0 - row["point_acceptance"]
                                       for row in computed)
    backup_average = metrics["backup_mass"]["source_equal"]
    return dict(kind="position_candidate_overlap", input=str(path), rows=len(rows),
                sources=len(source_names), metrics=metrics,
                source_equal_point_reject_mass=reject_average,
                row_mean_point_reject_mass=reject_row_mean,
                source_equal_fraction_of_rejects_captured=(
                    backup_average / reject_average if reject_average > 0 else None),
                per_source=per_source,
                normalization=dict(target_max_abs_mass_error=max_policy_norm_error,
                                   candidate_q_max_abs_mass_error=max_candidate_q_norm_error),
                reported_source_mean=data.get("source_mean"))


def summarize_overlap(path: Path):
    data = read_json(path)
    rows = data.get("rows") or []
    if rows and ("previous_policy" in rows[0] or "p_prev" in rows[0]):
        return summarize_feature_overlap(path)
    return summarize_position_overlap(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        help="Run directory or records.jsonl path")
    parser.add_argument("--overlap-json", type=Path,
                        help="Saved Position or Feature proposal overlap rows")
    parser.add_argument("--reference", help="Denominator arm name")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path,
                        help="Optional new JSON path; existing files are kept")
    args = parser.parse_args()
    if args.run is None and args.overlap_json is None:
        parser.error("supply --run, --overlap-json, or both")
    result = dict(bootstrap=dict(samples=int(args.bootstrap), seed=int(args.seed)))
    if args.run is not None:
        result["run_summary"] = summarize_run(
            args.run, reference=args.reference,
            bootstrap=args.bootstrap, seed=args.seed)
    if args.overlap_json is not None:
        result["overlap_summary"] = summarize_overlap(args.overlap_json.resolve())
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(str(output))


if __name__ == "__main__":
    main()
