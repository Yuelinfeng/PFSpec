#!/usr/bin/env python3
"""Independently recompute a completed PF/Target decode run."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 381012
CONTRASTS = {
    "original_pf_over_original_target": ("original_pf", "original_target"),
    "candidate_pf_over_candidate_target": ("candidate_pf", "candidate_target"),
    "candidate_pf_over_original_target": ("candidate_pf", "original_target"),
    "candidate_pf_over_original_pf": ("candidate_pf", "original_pf"),
    "candidate_target_over_original_target": ("candidate_target", "original_target"),
}
PARITY = {"target": ("original_target", "candidate_target"),
          "pf": ("original_pf", "candidate_pf")}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def quantile(values, probability):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def source_gm(log_values):
    means = {source: math.fsum(values) / len(values)
             for source, values in log_values.items() if values}
    names = sorted(means)
    if not names:
        return dict(pair_count=0, source_count=0, source_equal_geometric_mean=None,
                    source_bootstrap_95_ci=None, per_source={})
    rng = random.Random(BOOTSTRAP_SEED)
    samples = [math.exp(math.fsum(means[rng.choice(names)] for _ in names) / len(names))
               for _ in range(BOOTSTRAP_N)] if len(names) > 1 else []
    return dict(
        pair_count=sum(len(values) for values in log_values.values()),
        source_count=len(names),
        source_equal_geometric_mean=math.exp(math.fsum(means.values()) / len(names)),
        source_bootstrap_seed=BOOTSTRAP_SEED,
        source_bootstrap_95_ci=([quantile(samples, .025), quantile(samples, .975)]
                                if samples else None),
        per_source={source: math.exp(value) for source, value in means.items()},
    )


def audit(run_dir):
    paths = {name: run_dir / name for name in
             ("plan.json", "prompts.json", "records.jsonl", "summary.json")}
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("run_dir must contain plan.json, prompts.json, records.jsonl, summary.json")
    plan, prompts, summary = (load_json(paths[name]) for name in
                              ("plan.json", "prompts.json", "summary.json"))
    if summary.get("status") != "complete":
        raise ValueError("refusing to audit an incomplete run")
    records = [json.loads(line) for line in paths["records.jsonl"].read_text(
        encoding="utf-8").splitlines() if line.strip()]
    contexts = prompts.get("contexts", [])
    methods = list(plan["methods"])
    expected_runs = int(plan.get("planned_runs", plan.get("expected_runs", -1)))
    expected_pairs = int(plan.get("prompt_count", len(contexts)))
    expected_scenes = len(plan.get("sources", [])) or len({c["source"] for c in contexts})
    if expected_runs < 0 or not contexts or len(set(methods)) != len(methods):
        raise ValueError("plan or prompt counts are missing")

    checks = []

    def add(name, passed, **detail):
        checks.append(dict(name=name, passed=bool(passed), **detail))

    prompt_by_pair = {str(c["pair_id"]): c for c in contexts}
    groups = {}
    duplicates = []
    for row in records:
        pair, method = str(row["pair_id"]), str(row["method"])
        group = groups.setdefault(pair, {})
        if method in group:
            duplicates.append([pair, method])
        group[method] = row
    scenes = {str(row.get("source")) for row in records}
    planned_scenes = {str(source) for source in plan.get("sources", [])}
    prompt_scenes = {str(c.get("source")) for c in contexts}
    missing_pairs = sorted(set(prompt_by_pair) - set(groups))
    extra_pairs = sorted(set(groups) - set(prompt_by_pair))
    bad_arms = {pair: sorted(set(methods) ^ set(group))
                for pair, group in groups.items() if set(methods) != set(group)}
    add("run_pair_scene_counts", expected_runs == len(records)
        and expected_runs == len(contexts) * len(methods)
        and len(contexts) == expected_pairs and len(groups) == expected_pairs
        and len(scenes) == expected_scenes and scenes == prompt_scenes
        and (not planned_scenes or scenes == planned_scenes),
        expected_runs=expected_runs, observed_runs=len(records),
        expected_pairs=expected_pairs, observed_pairs=len(groups),
        expected_scenes=expected_scenes, observed_scenes=len(scenes))
    add("four_arms_each_pair", len(methods) == 4 and not missing_pairs and not extra_pairs
        and not bad_arms and not duplicates,
        methods=methods, missing_pairs=missing_pairs, extra_pairs=extra_pairs,
        bad_arms=bad_arms, duplicate_rows=duplicates)

    prompt_mismatches, length_issues, lengths = [], [], {}
    eos_id = next((source[key] for source in (plan, prompts, summary)
                   for key in ("eos_id", "eos_token_id", "bos_id")
                   if isinstance(source.get(key), int)
                   and not isinstance(source.get(key), bool)), None)
    for row in records:
        pair, method = str(row["pair_id"]), str(row["method"])
        context = prompt_by_pair.get(pair, {})
        for key in ("source", "prefix", "seed", "budget", "input_path"):
            if key in context and key in row and context[key] != row[key]:
                prompt_mismatches.append([pair, method, key])
        tokens = row.get("output_tokens")
        if not isinstance(tokens, list):
            length_issues.append([pair, method, "missing output_tokens"])
            continue
        length = len(tokens)
        budget = int(row.get("requested_new_tokens",
                             context.get("budget", row.get("budget", plan.get("budget", -1)))))
        eos = bool(row.get("eos", False))
        per_method = lengths.setdefault(method, dict(lengths={}, eos_count=0,
                                                      full_budget_count=0))
        per_method["lengths"][str(length)] = per_method["lengths"].get(str(length), 0) + 1
        per_method["eos_count"] += int(eos)
        per_method["full_budget_count"] += int(length == budget)
        if int(row.get("generated_tokens", -1)) != length or budget <= 0 or length > budget:
            length_issues.append([pair, method, "length/budget mismatch"])
        if "full_budget" in row and bool(row["full_budget"]) != (length == budget):
            length_issues.append([pair, method, "full_budget mismatch"])
        if isinstance(row.get("stats"), dict):
            stats = row["stats"]
            if "eos" in stats and bool(stats["eos"]) != eos:
                length_issues.append([pair, method, "stats EOS mismatch"])
            if "generated_tokens" in stats and int(stats["generated_tokens"]) != length:
                length_issues.append([pair, method, "stats length mismatch"])
        if eos_id is not None and bool(tokens and int(tokens[-1]) == eos_id) != eos:
            length_issues.append([pair, method, "terminal token/EOS mismatch"])
        if eos and eos_id is None:
            length_issues.append([pair, method, "EOS id unavailable"])
    add("lengths_and_eos", not length_issues and not prompt_mismatches,
        eos_id=eos_id, eos_marker_source=("plan/prompts/summary" if eos_id is not None else None),
        per_method=lengths, issues=length_issues[:40],
        prompt_mismatch_count=len(prompt_mismatches))

    method_totals, total_issues = {}, []
    for method in methods:
        chosen = [r for r in records if r.get("method") == method]
        seconds = math.fsum(float(r.get("outer_wall_seconds", 0)) for r in chosen)
        tokens = sum(int(r.get("generated_tokens", 0)) for r in chosen)
        counters = {}
        for row in chosen:
            for key, value in (row.get("counters") or {}).items():
                counters[key] = counters.get(key, 0) + int(value)
        actual = dict(runs=len(chosen), generated_tokens=tokens,
                      outer_wall_seconds=seconds,
                      pooled_tokens_per_second=tokens / seconds if seconds else None,
                      eos_count=sum(bool(r.get("eos")) for r in chosen),
                      output_lengths=lengths.get(method, {}).get("lengths", {}), counters=counters)
        method_totals[method] = actual
        reported = summary.get("by_method", {}).get(method, {})
        for key in ("runs", "generated_tokens", "eos_count"):
            if int(reported.get(key, -1)) != actual[key]:
                total_issues.append(f"{method}.{key}")
        for key in ("outer_wall_seconds", "pooled_tokens_per_second"):
            if not math.isclose(float(reported.get(key, -1)), float(actual[key]), rel_tol=1e-9, abs_tol=1e-9):
                total_issues.append(f"{method}.{key}")
        if {str(k): int(v) for k, v in reported.get("output_lengths", {}).items()} != actual["output_lengths"]:
            total_issues.append(f"{method}.output_lengths")
        if {str(k): int(v) for k, v in reported.get("counters", {}).items()} != counters:
            total_issues.append(f"{method}.counters")
    add("raw_method_totals_match_summary", not total_issues, differences=total_issues)

    parity, parity_issues = {}, []
    for label, (old_name, new_name) in PARITY.items():
        paired = [(pair, group[old_name], group[new_name]) for pair, group in groups.items()
                  if old_name in group and new_name in group]
        mismatches = []
        for pair, old, new in paired:
            fields = {
                "tokens": old.get("output_tokens") == new.get("output_tokens"),
                "counts": old.get("counters") == new.get("counters"),
                "length": old.get("generated_tokens") == new.get("generated_tokens"),
                "eos": old.get("eos") == new.get("eos"),
            }
            if not all(fields.values()):
                mismatches.append(dict(pair_id=pair, **fields))
        parity[label] = dict(pair_count=len(paired), equal_pair_count=len(paired) - len(mismatches),
                             mismatches=mismatches)
        if len(paired) != expected_pairs or mismatches:
            parity_issues.append(label)
    add("same_algorithm_old_new_parity", not parity_issues,
        parity=parity, failures=parity_issues)

    source_for_pair = {str(c["pair_id"]): str(c["source"]) for c in contexts}
    ratios, ratio_issues = {}, []
    for name, (up_method, down_method) in CONTRASTS.items():
        all_logs, full_logs = {}, {}
        for pair, group in groups.items():
            if up_method not in group or down_method not in group or pair not in source_for_pair:
                continue
            up, down = group[up_method], group[down_method]
            ut, dt = int(up["generated_tokens"]), int(down["generated_tokens"])
            uw, dw = float(up["outer_wall_seconds"]), float(down["outer_wall_seconds"])
            if min(ut, dt, uw, dw) <= 0:
                continue
            source = source_for_pair[pair]
            all_logs.setdefault(source, []).append(math.log((ut / uw) / (dt / dw)))
            budget = int(up.get("requested_new_tokens", up.get("budget", plan.get("budget", 0))))
            if ut == dt == budget and budget > 0:
                full_logs.setdefault(source, []).append(math.log(dw / uw))
        ratios[name] = dict(all_output_lengths=source_gm(all_logs),
                            common_full_budget=source_gm(full_logs))
        reported = summary.get("source_equal_geometric_speed_ratios", {}).get(name)
        if not isinstance(reported, dict):
            ratio_issues.append(f"{name}: missing")
            continue
        for subset, value in ratios[name].items():
            target = reported.get(subset, {})
            if int(target.get("source_count", -1)) != value["source_count"]:
                ratio_issues.append(f"{name}.{subset}.source_count")
            if not math.isclose(float(target.get("source_equal_geometric_mean") or 0),
                                float(value["source_equal_geometric_mean"] or 0),
                                rel_tol=1e-9, abs_tol=1e-9):
                ratio_issues.append(f"{name}.{subset}.mean")
            if value["source_count"] and int(target.get("source_bootstrap_seed", -1)) != BOOTSTRAP_SEED:
                ratio_issues.append(f"{name}.{subset}.seed")
            ci, target_ci = value["source_bootstrap_95_ci"], target.get("source_bootstrap_95_ci")
            if ci != target_ci and (ci is None or target_ci is None or len(ci) != len(target_ci)
                                    or any(not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
                                           for a, b in zip(ci, target_ci))):
                ratio_issues.append(f"{name}.{subset}.CI")
            ps = {str(k): float(v) for k, v in target.get("per_source", {}).items()}
            if set(ps) != set(value["per_source"]) or any(
                    not math.isclose(ps[k], v, rel_tol=1e-9, abs_tol=1e-9)
                    for k, v in value["per_source"].items() if k in ps):
                ratio_issues.append(f"{name}.{subset}.per_source")
    add("five_speed_ratios_match_summary", not ratio_issues,
        bootstrap_replicates=BOOTSTRAP_N, bootstrap_seed=BOOTSTRAP_SEED,
        differences=ratio_issues, values=ratios)

    candidates = [r for r in records if str(r.get("method", "")).startswith("candidate_")]
    candidate_methods = {m for m in methods if m.startswith("candidate_")}
    expected_candidates = expected_pairs * len(candidate_methods)
    dynamic = summary.get("dynamic_graph_runtime") or {}
    setup = summary.get("dynamic_graph_setup") or {}
    calls, fallbacks = dynamic.get("calls", {}), dynamic.get("fallbacks", {})
    graph_setup = math.fsum(float((r.get("dynamic_graph") or {}).get("new_graph_setup_ms", 0) or 0)
                            for r in candidates)
    runtime_init = math.fsum(float((r.get("dynamic_graph") or {}).get("runtime_init_ms", 0) or 0)
                             for r in candidates)
    cache_alloc = math.fsum(float((r.get("dynamic_graph") or {}).get(
        "new_cache_allocation_host_enqueue_ms", 0) or 0) for r in candidates)
    setup_rows = [r for r in candidates if r.get("dynamic_graph_runtime_created")]
    setup_wall_ok = bool(setup_rows) and all(
        sum(float((r.get("dynamic_graph") or {}).get(key, 0) or 0) for key in (
            "new_graph_setup_ms", "runtime_init_ms", "new_cache_allocation_host_enqueue_ms"))
        <= float(r.get("outer_wall_seconds", 0)) * 1000 + .1 for r in candidates)
    setup_values_ok = (
        math.isclose(float(setup.get("graph_prepare_warmup_capture_ms", -1)), graph_setup,
                     rel_tol=1e-6, abs_tol=.1)
        and math.isclose(float(setup.get("runtime_init_ms", -1)), runtime_init,
                         rel_tol=1e-6, abs_tol=.1)
        and math.isclose(float(setup.get("cache_allocation_host_enqueue_ms", -1)), cache_alloc,
                         rel_tol=1e-6, abs_tol=.1)
        and math.isclose(float(setup.get("cold_setup_breakdown_ms", -1)),
                         graph_setup + runtime_init + cache_alloc, rel_tol=1e-6, abs_tol=.1)
        and int(setup.get("candidate_decode_count", -1)) == len(candidates) == expected_candidates
    )
    note = " ".join((str(summary.get("dynamic_graph_setup_accounting", "")),
                     str(setup.get("accounting", "")))).lower()
    setup_note_ok = "outer" in note and ("inside" in note or "included" in note) and (
        "do not add" in note or "already included" in note or "not add" in note)
    unexpected = {key: int(value) for key, value in fallbacks.items()
                  if key != "unsupported_query_tokens" and int(value) != 0}
    prefill_count = len(candidates)
    fallback_ok = (not unexpected
                   and int(fallbacks.get("unsupported_query_tokens", 0)) == prefill_count
                   and int(calls.get("original_forward", 0)) == prefill_count)
    graph_ok = (bool(plan.get("dynamic_graph_enabled"))
                and len(candidates) == expected_candidates
                and all(bool(r.get("dynamic_graph_enabled")) for r in candidates)
                and int(dynamic.get("graph_count", 0)) > 0
                and int(calls.get("graph_forward", 0)) > 0
                and not dynamic.get("disabled_graphs", {}) and fallback_ok)
    add("dynamic_graph_fallback_and_setup", graph_ok and setup_values_ok
        and setup_wall_ok and setup_note_ok,
        candidate_runs=len(candidates), graph_count=dynamic.get("graph_count"),
        graph_forward_calls=calls.get("graph_forward"), fallbacks=fallbacks,
        unexpected_fallbacks=unexpected,
        setup_ms=dict(records=graph_setup + runtime_init + cache_alloc,
                      summary=setup.get("cold_setup_breakdown_ms")),
        setup_in_outer_wall=setup_wall_ok and setup_values_ok and setup_note_ok)

    summary_counts_ok = (int(summary.get("expected_runs", -1)) == expected_runs
                         and int(summary.get("completed_runs", -1)) == len(records)
                         and int(summary.get("prompts", -1)) == len(contexts))
    add("summary_completion_counts", summary_counts_ok)
    passed = all(row["passed"] for row in checks)
    return dict(
        status="passed" if passed else "failed", run_dir=str(run_dir),
        expected=dict(runs=expected_runs, pairs=expected_pairs, scenes=expected_scenes,
                      arms=len(methods), methods=methods),
        observed=dict(runs=len(records), pairs=len(groups), scenes=len(scenes),
                      arms=len({str(r.get("method")) for r in records})),
        per_method=method_totals, parity=parity, speed_ratios=ratios,
        checks=checks,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path,
                        help="defaults to RUN_DIR/audit.json; existing files are kept")
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else run_dir / "audit.json"
    if output.exists():
        parser.error("output already exists: " + str(output))
    try:
        result = audit(run_dir)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(dict(status=result["status"], output=str(output),
                          passed=sum(item["passed"] for item in result["checks"]),
                          total=len(result["checks"])), ensure_ascii=False))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
