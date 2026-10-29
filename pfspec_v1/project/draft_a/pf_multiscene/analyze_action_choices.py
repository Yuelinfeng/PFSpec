#!/usr/bin/env python3
"""Analyze whether development context groups favor different P/F actions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


METHODS = ("p_only", "step12000", "d8_feature", "d8_joint")


def totals(rows: list[dict]) -> dict:
    tokens = sum(row["generated_tokens"] for row in rows)
    seconds = sum(row["seconds"] for row in rows)
    return {
        "runs": len(rows),
        "generated_tokens": tokens,
        "seconds": seconds,
        "tokens_per_second": tokens / seconds if seconds else None,
        "position_acceptance_verified": (
            sum(row.get("position_accepted", 0) for row in rows)
            / sum(row.get("position_verified", 0) for row in rows)
            if sum(row.get("position_verified", 0) for row in rows) else None
        ),
        "feature_acceptance_verified": (
            sum(row.get("feature_accepted", 0) for row in rows)
            / sum(row.get("feature_verified", 0) for row in rows)
            if sum(row.get("feature_verified", 0) for row in rows) else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.online.read_text())
    rows = data["rows"]
    source_groups = {}
    with args.catalog.open(newline="") as handle:
        source_groups = {
            row["scene_id"]: row["group_id"] for row in csv.DictReader(handle)
        }

    scene_prefix = defaultdict(lambda: defaultdict(list))
    scene_method = defaultdict(lambda: defaultdict(list))
    prefix_method = defaultdict(lambda: defaultdict(list))
    paired = defaultdict(dict)
    for row in rows:
        scene, prefix, mode = row["scene"], row["prefix"], row["variant"]
        scene_prefix[(scene, prefix)][mode].append(row)
        scene_method[scene][mode].append(row)
        prefix_method[prefix][mode].append(row)
        paired[(row["name"], prefix, row["seed_base"])][mode] = row

    context_cells = []
    context_wins = Counter()
    for (scene, prefix), modes in sorted(scene_prefix.items()):
        values = {mode: totals(modes[mode]) for mode in METHODS}
        target = totals(modes["target_only"])
        speedups = {
            mode: values[mode]["tokens_per_second"] / target["tokens_per_second"]
            for mode in METHODS
        }
        winner = max(speedups, key=speedups.get)
        context_wins[winner] += 1
        context_cells.append({
            "scene": scene,
            "source_group": source_groups[scene],
            "prefix": prefix,
            "target_tokens_per_second": target["tokens_per_second"],
            "methods": values,
            "speedup_vs_target": speedups,
            "fastest_method": winner,
            "p_only_vs_pf_speedup_ratio": (
                values["p_only"]["tokens_per_second"]
                / values["step12000"]["tokens_per_second"]
            ),
        })

    paired_wins = Counter()
    paired_rows = []
    for (crop, prefix, seed), modes in sorted(paired.items()):
        speed = {mode: modes[mode]["tokens_per_second"] for mode in METHODS}
        winner = max(speed, key=speed.get)
        paired_wins[winner] += 1
        paired_rows.append({
            "crop": crop,
            "prefix": prefix,
            "seed_base": seed,
            "fastest_method": winner,
            "tokens_per_second": speed,
            "p_only_vs_pf_speedup_ratio": speed["p_only"] / speed["step12000"],
            "pf_feature_acceptance_verified": modes["step12000"].get(
                "feature_acceptance_verified"
            ),
        })

    prefix_summary = {}
    for prefix, modes in sorted(prefix_method.items()):
        values = {mode: totals(modes[mode]) for mode in ("target_only", *METHODS)}
        target_tps = values["target_only"]["tokens_per_second"]
        prefix_summary[str(prefix)] = {
            "target_tokens_per_second": target_tps,
            "methods": values,
            "speedup_vs_target": {
                mode: values[mode]["tokens_per_second"] / target_tps
                for mode in METHODS
            },
        }

    scene_summary = {}
    for scene, modes in sorted(scene_method.items()):
        values = {mode: totals(modes[mode]) for mode in ("target_only", *METHODS)}
        target_tps = values["target_only"]["tokens_per_second"]
        scene_summary[scene] = {
            "source_group": source_groups[scene],
            "speedup_vs_target": {
                mode: values[mode]["tokens_per_second"] / target_tps
                for mode in METHODS
            },
        }

    pf_wins = context_wins["step12000"]
    action = "keep_static_pf" if pf_wins == len(context_cells) else "evaluate_scheduler"
    result = {
        "status": "complete",
        "scope": "development FP32; scene-prefix cells pool crops and three seeds",
        "available_context_signal": ["scene", "prefix length"],
        "unlogged_signals": ["per-round Position confidence", "per-round action history", "scheduler overhead"],
        "scene_prefix_cell_count": len(context_cells),
        "scene_prefix_fastest_method_counts": dict(context_wins),
        "paired_run_count": len(paired_rows),
        "paired_run_fastest_method_counts": dict(paired_wins),
        "static_action_decision": action,
        "reason": (
            f"PF (step12000) was fastest in {pf_wins}/{len(context_cells)} scene-prefix cells; "
            "the measured context signals do not identify a cell where switching to P-only helps."
            if pf_wins == len(context_cells) else
            "At least one scene-prefix cell favored another method; inspect paired rows before considering a scheduler."
        ),
        "by_prefix": prefix_summary,
        "by_scene": scene_summary,
        "scene_prefix_cells": context_cells,
        "paired_runs": paired_rows,
        "limitations": [
            "This analysis does not estimate a per-round decision rule.",
            "A scheduler based on unlogged signals would require a separate replay and cost measurement.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    selection = {
        "status": "static_method_retained" if action == "keep_static_pf" else "scheduler_candidate_review",
        "selected_method": "step12000" if action == "keep_static_pf" else None,
        "context_signals_used": ["scene", "prefix length"],
        "scene_prefix_cell_count": len(context_cells),
        "scene_prefix_fastest_method_counts": dict(context_wins),
        "paired_run_fastest_method_counts": dict(paired_wins),
        "scheduler_implemented": False,
        "reason": result["reason"],
        "limitations": result["limitations"],
    }
    args.out.with_name("scheduler_selection.json").write_text(
        json.dumps(selection, indent=2)
    )
    print(json.dumps({
        "status": result["status"],
        "scene_prefix_cell_count": len(context_cells),
        "scene_prefix_fastest_method_counts": dict(context_wins),
        "paired_run_count": len(paired_rows),
        "paired_run_fastest_method_counts": dict(paired_wins),
        "static_action_decision": action,
    }, indent=2))


if __name__ == "__main__":
    main()
