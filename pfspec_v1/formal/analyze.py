#!/usr/bin/env python3
"""Summarize completed request records using source-level paired resampling."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics


def percentile(values, fraction):
    values = sorted(values)
    x = (len(values) - 1) * fraction
    lo = int(x)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def describe(values):
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "p05": percentile(values, .05), "p95": percentile(values, .95)}


def paired_ratio(pairs, metric, iterations, seed):
    by_source = defaultdict(list)
    for first, second in pairs:
        if metric == "generation":
            ta, tb = first["stages"]["generation"], second["stages"]["generation"]
        else:
            ta, tb = first["total_seconds"], second["total_seconds"]
        na, nb = first["generated_tokens"], second["generated_tokens"]
        if min(na, nb, ta, tb) > 0:
            by_source[first["source"]].append(math.log((na / ta) / (nb / tb)))
    means = [statistics.mean(x) for x in by_source.values()]
    if not means:
        return {"sources": 0, "pairs": 0, "ratio": None, "interval95": None}
    rng = random.Random(seed)
    draws = [math.exp(statistics.mean(rng.choices(means, k=len(means)))) for _ in range(iterations)]
    return {"sources": len(means), "pairs": sum(map(len, by_source.values())),
            "ratio": math.exp(statistics.mean(means)),
            "interval95": [percentile(draws, .025), percentile(draws, .975)]}


def mechanism(group):
    fields = ("rounds", "target_calls", "target_input_tokens", "draft_calls", "draft_input_tokens",
              "tree_verify_calls", "candidate_tree_nodes", "target_tree_nodes", "recovery_calls",
              "position_proposed", "position_verified", "position_accepted", "feature_proposed",
              "feature_verified", "feature_accepted", "feature_branch_hits", "feature_all_rejected",
              "position_backup_nodes", "position_rescue_hits", "k1_steps")
    totals = {key: sum(r["stats"].get(key, 0) for r in group) for key in fields
              if any(key in r["stats"] for r in group)}
    if totals.get("rounds") and all("round_generated_lengths" in r["stats"] for r in group):
        totals["generated_per_round"] = sum(sum(r["stats"]["round_generated_lengths"]) for r in group) / totals["rounds"]
        totals["generation_seconds_per_round_including_request_setup"] = sum(r["stages"]["generation"] for r in group) / totals["rounds"]
    if "feature_branch_hits" in totals:
        p2_verified = totals["feature_branch_hits"]
        p2_accepted = sum(sum(length == 3 for length in r["stats"]["accepted_lengths"]) for r in group)
        p1_verified = totals["position_verified"] - p2_verified
        p1_accepted = totals["position_accepted"] - p2_accepted
        feature_rounds = p2_verified + totals["feature_all_rejected"]
        totals["p1_acceptance"] = {"accepted": p1_accepted, "verified": p1_verified}
        totals["feature_any_branch_acceptance_given_p1_and_tree"] = {"accepted": p2_verified, "verified": feature_rounds}
        totals["p2_acceptance_given_feature"] = {"accepted": p2_accepted, "verified": p2_verified}
    return totals


def analyze(directory, iterations, seed):
    plan = json.loads((directory / "plan.json").read_text())
    progress = json.loads((directory / "progress.json").read_text())
    rows = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines() if line.strip()]
    grouped = defaultdict(list)
    indexes = {}
    for row in rows:
        key = (row["cohort"], row["pair_id"], row["method"])
        if key in indexes:
            raise ValueError(f"duplicate record: {key}")
        indexes[key] = row
        cohort = row["cohort"]
        if cohort == "natural_eos":
            cohort += "_bos" if row["prefix"] == 0 else "_completion"
        grouped[(cohort, row["method"])].append(row)
    result = {"directory": str(directory), "status": progress["status"],
              "expected": plan["expected"], "recorded": len(rows),
              "cohorts": {}, "comparisons": {},
              "scope": "Throughput uses actual generated tokens. End-to-end time includes generation, Gaussian export and six PNG views; loading and warmup are excluded.",
              "bootstrap": {"unit": "scene for completion/fixed; seed for BOS", "iterations": iterations, "seed": seed}}
    for (cohort, method), group in sorted(grouped.items()):
        times = [r["total_seconds"] for r in group]
        counts = [r["generated_tokens"] for r in group]
        result["cohorts"].setdefault(cohort, {})[method] = {
            "requests": len(group), "sources": len({r["source"] for r in group}),
            "eos": sum(bool(r["eos"]) for r in group),
            "capped": sum(bool(r["capped"]) for r in group),
            "mean_total_seconds": statistics.mean(times), "median_total_seconds": statistics.median(times),
            "mean_generated_tokens": statistics.mean(counts),
            "pooled_generation_tokens_per_second": sum(counts) / sum(r["stages"]["generation"] for r in group),
            "pooled_total_tokens_per_second": sum(counts) / sum(times),
            "quality": {
                "gaussians": describe([r["quality"]["gaussians"] for r in group]),
                "mean_opacity": describe([r["quality"]["mean_opacity"] for r in group]),
                "scale_clipped_fraction": describe([r["quality"]["scale_clipped_fraction"] for r in group]),
                "alpha_coverage": describe([r["alpha_coverage"] for r in group])},
            "mechanism": mechanism(group)}
    comparisons = [("opt_target", "target"), ("w1", "target"), ("rescue", "target"),
                   ("ranker_only", "opt_target"), ("w1", "opt_target"), ("rescue", "opt_target"),
                   ("w1", "ranker_only"), ("rescue", "w1"), ("w2", "w1")]
    for cohort in sorted({c for c, _ in grouped}):
        for method, reference in comparisons:
            left, right = grouped.get((cohort, method), []), grouped.get((cohort, reference), [])
            lookup = {(r["pair_id"], r["source"]): r for r in right}
            pairs = [(r, lookup[(r["pair_id"], r["source"])]) for r in left
                     if (r["pair_id"], r["source"]) in lookup]
            if pairs:
                comparison = {
                    "generation": paired_ratio(pairs, "generation", iterations, seed),
                    "total": paired_ratio(pairs, "total", iterations, seed),
                    "both_eos": sum(a["eos"] and b["eos"] for a, b in pairs),
                    "same_generated_length": sum(a["generated_tokens"] == b["generated_tokens"] for a, b in pairs)}
                if (method, reference) == ("rescue", "w1"):
                    fields = ("rounds", "proposed", "verified", "accepted", "position_proposed",
                              "position_verified", "position_accepted", "feature_proposed",
                              "feature_verified", "feature_accepted", "accepted_lengths",
                              "round_generated_lengths")
                    comparison["same_output"] = sum(a["output_tokens"] == b["output_tokens"] for a, b in pairs)
                    comparison["same_semantic_counts"] = sum(
                        all(a["stats"].get(k) == b["stats"].get(k) for k in fields) for a, b in pairs)
                    comparison["different_pairs"] = [a["pair_id"] for a, b in pairs
                        if a["output_tokens"] != b["output_tokens"] or
                        any(a["stats"].get(k) != b["stats"].get(k) for k in fields)]
                result["comparisons"].setdefault(cohort, {})[f"{method}/{reference}"] = comparison
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    args.out.write_text(json.dumps(analyze(args.directory, args.iterations, args.seed), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
