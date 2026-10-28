#!/usr/bin/env python3
"""Report paired source-group differences for the first adaptation seed."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from summarize_online_results import sum_fields


METHODS = ("target_only", "p_only", "step12000", "adapt_A", "adapt_B")
PAIRS = (("adapt_A", "step12000"), ("adapt_B", "step12000"), ("adapt_B", "adapt_A"))


def interval(values: list[float], seed: int, *, geometric: bool) -> dict:
    values_array = np.asarray(values, dtype=np.float64)
    transformed = np.log(values_array) if geometric else values_array
    estimate = float(transformed.mean())
    if geometric:
        estimate = float(np.exp(estimate))
    if len(values) < 2:
        return dict(estimate=estimate, ci95=None)
    rng = np.random.default_rng(seed)
    samples = transformed[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    if geometric:
        samples = np.exp(samples)
    return dict(estimate=estimate, ci95=[float(x) for x in np.quantile(samples, [0.025, 0.975])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    result = json.loads((args.eval_root / "online.json").read_text())
    if result["status"] != "complete":
        raise ValueError("The complete interleaved development run is required")
    plan, rows = result["parameters"], result["rows"]
    with args.catalog.open(newline="") as handle:
        groups = {row["scene_id"]: row["group_id"] for row in csv.DictReader(handle)}
    inputs = defaultdict(dict)
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["name"], row["prefix"], row["seed_base"])
        method = row["variant"]
        if method in inputs[key]:
            raise ValueError(f"Repeated method for development input: {key}/{method}")
        inputs[key][method] = row
        grouped[groups[row["scene"]]][method].append(row)
    if any(set(methods) != set(METHODS) for methods in inputs.values()):
        raise ValueError("Every development input must contain T, P, old PF, PF A, and PF B")
    totals = {method: sum_fields([row for row in rows if row["variant"] == method], plan["budget"])
              for method in METHODS}
    per_group = {group: {method: sum_fields(method_rows[method], plan["budget"])
                        for method in METHODS}
                 for group, method_rows in grouped.items()}
    comparisons = {}
    for numerator, denominator in PAIRS:
        values = {}
        for group, metrics in per_group.items():
            new, old = metrics[numerator], metrics[denominator]
            values[group] = dict(
                speed_ratio=new["tokens_per_second"] / old["tokens_per_second"],
                feature_verified_difference=new["feature_acceptance_verified"] - old["feature_acceptance_verified"],
                feature_proposed_difference=new["feature_acceptance_proposed"] - old["feature_acceptance_proposed"],
            )
        comparisons[f"{numerator}_vs_{denominator}"] = dict(
            equal_group_speed_ratio=interval([item["speed_ratio"] for item in values.values()], args.seed, geometric=True),
            micro_speed_ratio=totals[numerator]["tokens_per_second"] / totals[denominator]["tokens_per_second"],
            feature_verified_difference=interval([item["feature_verified_difference"] for item in values.values()], args.seed, geometric=False),
            feature_proposed_difference=interval([item["feature_proposed_difference"] for item in values.values()], args.seed, geometric=False),
            groups_faster=sum(item["speed_ratio"] > 1 for item in values.values()),
            per_source_group=values,
        )

    fixed = json.loads((args.eval_root / "fixed_real_overlap.json").read_text())
    crop_scene = {crop["name"]: crop["scene"] for crop in plan["crops"]}
    fixed_group = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for crop in fixed["crops"]:
        group = groups[crop_scene[crop["name"]]]
        for method in ("step12000", "adapt_A", "adapt_B"):
            fixed_group[group][method][0] += crop[method]["pooled_overlap"] * crop["rows"]
            fixed_group[group][method][1] += crop["rows"]
    fixed_values = {group: {method: total / count for method, (total, count) in metrics.items()}
                    for group, metrics in fixed_group.items()}
    fixed_comparisons = {
        f"{numerator}_vs_{denominator}": interval(
            [values[numerator] - values[denominator] for values in fixed_values.values()],
            args.seed, geometric=False,
        )
        for numerator, denominator in PAIRS
    }
    summary = dict(
        status="complete", precision=plan["precision"], runs=len(rows),
        paired_inputs=len(inputs), source_groups=len(per_group),
        models=plan["models"], original_target=plan["original_target"],
        seed=args.seed, overall=totals, comparisons=comparisons,
        fixed_overlap=dict(rows=fixed["rows"], per_source_group=fixed_values,
                           equal_group_differences=fixed_comparisons),
        interpretation="Intervals resample whole source groups; each group retains all paired crops, prefixes, seeds, and methods. Acceptance differences are fractions.",
    )
    path = args.eval_root / "adaptation_comparison.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "runs", "paired_inputs", "source_groups", "comparisons")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
