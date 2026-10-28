#!/usr/bin/env python3
"""Aggregate paired multiscene decoding results and draw a development figure."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


LABELS = {
    "target_only": "T",
    "p_only": "P",
    "step12000": "PF",
    "d8_feature": "PF-D8",
    "d8_joint": "GPT-D8",
}


def sum_fields(rows: list[dict], budget: int) -> dict:
    fields = [
        "generated_tokens", "seconds", "position_proposed", "position_verified",
        "position_accepted", "feature_proposed", "feature_verified",
        "feature_accepted", "target_calls", "draft_calls", "rounds",
    ]
    totals = {key: sum(row.get(key, 0) or 0 for row in rows) for key in fields}
    totals["runs"] = len(rows)
    totals["tokens_per_second"] = totals["generated_tokens"] / totals["seconds"] if totals["seconds"] else None
    totals["eos_runs"] = sum(bool(row.get("eos")) for row in rows)
    totals["eos_fraction"] = totals["eos_runs"] / len(rows) if rows else None
    totals["budget_capped_runs"] = sum(
        not bool(row.get("eos")) and row.get("generated_tokens", 0) >= budget for row in rows
    )
    totals["mean_generated_tokens"] = totals["generated_tokens"] / len(rows) if rows else None
    accepted_total = sum(sum(row.get("accepted_lengths") or []) for row in rows)
    totals["mean_accepted_length"] = accepted_total / totals["rounds"] if totals["rounds"] else None
    for kind in ("position", "feature"):
        for denominator in ("verified", "proposed"):
            count = totals[f"{kind}_{denominator}"]
            totals[f"{kind}_acceptance_{denominator}"] = (
                totals[f"{kind}_accepted"] / count if count else None
            )
    return totals


def geometric_mean(values: list[float]) -> float | None:
    return math.exp(float(np.log(values).mean())) if values else None


def bootstrap_ci(group_ratios: dict[str, float], seed: int, draws: int = 10000) -> list[float] | None:
    values = np.asarray(list(group_ratios.values()), dtype=np.float64)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(values), size=(draws, len(values)))
    samples = np.exp(np.log(values)[sample_indices].mean(axis=1))
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--split-label", default="development")
    args = parser.parse_args()
    data = json.loads((args.eval_root / "online.json").read_text())
    rows = data["rows"]
    plan = data["parameters"]
    observed = list(dict.fromkeys(row["variant"] for row in rows))
    mode_labels = plan.get("mode_labels", LABELS)
    order = [mode for mode in mode_labels if mode in observed]
    order.extend(mode for mode in observed if mode not in order)
    scene_group = {}
    with args.catalog.open(newline="") as handle:
        scene_group = {row["scene_id"]: row["group_id"] for row in csv.DictReader(handle)}

    per_scene = defaultdict(lambda: defaultdict(list))
    per_group = defaultdict(lambda: defaultdict(list))
    per_prefix = defaultdict(lambda: defaultdict(list))
    for row in rows:
        scene = row["scene"]
        group = scene_group[scene]
        mode = row["variant"]
        per_scene[scene][mode].append(row)
        per_group[group][mode].append(row)
        per_prefix[row["prefix"]][mode].append(row)

    target_total = sum_fields([row for row in rows if row["variant"] == "target_only"], plan["budget"])
    mode_summary = {}
    per_scene_summary = {}
    per_group_summary = {}
    for mode in order:
        aggregate = sum_fields([row for row in rows if row["variant"] == mode], plan["budget"])
        aggregate["speedup_micro"] = (
            aggregate["tokens_per_second"] / target_total["tokens_per_second"]
            if aggregate["tokens_per_second"] and target_total["tokens_per_second"]
            else None
        )
        ratios = {}
        for group, methods in per_group.items():
            method_totals = sum_fields(methods.get(mode, []), plan["budget"])
            target_group = sum_fields(methods.get("target_only", []), plan["budget"])
            if method_totals["tokens_per_second"] and target_group["tokens_per_second"]:
                ratios[group] = method_totals["tokens_per_second"] / target_group["tokens_per_second"]
        aggregate["source_group_geomean_speedup"] = geometric_mean(list(ratios.values()))
        aggregate["source_group_median_speedup"] = float(np.median(list(ratios.values()))) if ratios else None
        aggregate["source_group_speedup_ci95"] = bootstrap_ci(ratios, args.seed)
        aggregate["source_groups_below_1x"] = sum(value < 1.0 for value in ratios.values())
        aggregate["source_group_count"] = len(ratios)
        mode_summary[mode] = aggregate
        for scene, methods in per_scene.items():
            values = sum_fields(methods.get(mode, []), plan["budget"])
            target_scene = sum_fields(methods.get("target_only", []), plan["budget"])
            values["speedup_vs_target"] = (
                values["tokens_per_second"] / target_scene["tokens_per_second"]
                if values["tokens_per_second"] and target_scene["tokens_per_second"]
                else None
            )
            per_scene_summary.setdefault(scene, {})[mode] = values
        for group, methods in per_group.items():
            values = sum_fields(methods.get(mode, []), plan["budget"])
            target_group = sum_fields(methods.get("target_only", []), plan["budget"])
            values["speedup_vs_target"] = (
                values["tokens_per_second"] / target_group["tokens_per_second"]
                if values["tokens_per_second"] and target_group["tokens_per_second"]
                else None
            )
            per_group_summary.setdefault(group, {})[mode] = values

    prefix_summary = {}
    for prefix, methods in sorted(per_prefix.items()):
        prefix_summary[str(prefix)] = {}
        target_prefix = sum_fields(methods["target_only"], plan["budget"])
        for mode in order:
            values = sum_fields(methods.get(mode, []), plan["budget"])
            values["speedup_micro"] = (
                values["tokens_per_second"] / target_prefix["tokens_per_second"]
                if values["tokens_per_second"] and target_prefix["tokens_per_second"]
                else None
            )
            prefix_summary[str(prefix)][mode] = values

    summary = {
        "status": data.get("status"),
        "evaluation_split": args.split_label,
        "precision": plan.get("precision", "unspecified"),
        "expected_runs": sum(
            len([prefix for prefix in plan["prefixes"] if crop["body_tokens"] >= prefix])
            for crop in plan["crops"]
        ) * len(plan["seeds"]) * len(order),
        "completed_runs": len(rows),
        "unique_scenes": plan["unique_scenes"],
        "unique_crops": plan["unique_crops"],
        "prefixes": plan["prefixes"],
        "seeds": plan["seeds"],
        "budget": plan["budget"],
        "modes": {mode: mode_labels.get(mode, LABELS.get(mode, mode)) for mode in order},
        "overall": mode_summary,
        "by_prefix": prefix_summary,
        "per_scene": per_scene_summary,
        "per_source_group": per_group_summary,
        "limitations": [
            plan["uncertainty"],
            "Speedup intervals resample source groups; crops, prefixes, seeds, and modes stay paired within each group.",
            "Micro throughput pools all runs and is supplementary to the equal-group geometric mean.",
        ],
    }
    output = args.eval_root / "multiscene_summary.json"
    output.write_text(json.dumps(summary, indent=2))
    draw_figure(summary, args.eval_root / "multiscene_comparison.png")
    print(json.dumps({
        "status": summary["status"],
        "completed_runs": summary["completed_runs"],
        "unique_scenes": summary["unique_scenes"],
        "source_group_count": len(per_group),
        "speedups": {
            summary["modes"][mode]: {
                "geomean": value["source_group_geomean_speedup"],
                "median": value["source_group_median_speedup"],
                "ci95": value["source_group_speedup_ci95"],
                "micro": value["speedup_micro"],
                "position_acceptance_verified": value["position_acceptance_verified"],
                "feature_acceptance_verified": value["feature_acceptance_verified"],
            }
            for mode, value in mode_summary.items()
        },
    }, indent=2))


def draw_figure(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    order = list(summary["overall"])
    labels = [summary["modes"][mode] for mode in order]
    values = [summary["overall"][mode]["source_group_geomean_speedup"] for mode in order]
    intervals = [summary["overall"][mode]["source_group_speedup_ci95"] for mode in order]
    yerr = np.asarray([
        [value - interval[0], interval[1] - value] if interval and value is not None else [0, 0]
        for value, interval in zip(values, intervals)
    ]).T
    x = np.arange(len(labels))
    colors = ["#5470C6", "#91CC75", "#FAC858", "#EE6666", "#73C0DE"]
    axes[0].bar(x, values, color=[colors[index % len(colors)] for index in range(len(order))])
    axes[0].errorbar(x, values, yerr=yerr, fmt="none", ecolor="#303030", capsize=4, linewidth=1.3)
    axes[0].axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Geometric mean speedup vs T")
    axes[0].set_title("Equal source-group weight; 95% bootstrap interval")
    axes[0].grid(axis="y", alpha=0.22)

    acceptance_modes = [mode for mode in order if mode != "target_only"]
    acceptance_labels = [summary["modes"][mode] for mode in acceptance_modes]
    width = 0.19
    categories = [
        ("position_acceptance_verified", "P verified"),
        ("position_acceptance_proposed", "P proposed"),
        ("feature_acceptance_verified", "F verified"),
        ("feature_acceptance_proposed", "F proposed"),
    ]
    for index, (key, label) in enumerate(categories):
        heights = [summary["overall"][mode].get(key) or 0 for mode in acceptance_modes]
        axes[1].bar(x[:len(acceptance_modes)] + (index - 1.5) * width, heights, width, label=label)
    axes[1].set_xticks(x[:len(acceptance_modes)], acceptance_labels)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Acceptance rate")
    axes[1].set_title("P/F accepted tokens over verified or proposed")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].grid(axis="y", alpha=0.22)

    fig.suptitle(
        f"GaussianGPT P/F speculative decoding — {summary.get('evaluation_split', 'development')} {summary['precision']}",
        fontsize=13,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
