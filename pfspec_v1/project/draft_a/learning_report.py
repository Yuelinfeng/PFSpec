"""Describe available budget points using only the Python standard library."""

import argparse
import csv
import io
import json
import math
from pathlib import Path
import random
import statistics


METRICS = (
    "token_weighted_ce", "target_token_weighted_ce",
    "target_token_weighted_raw_entropy", "pooled_overlap",
    "equal_trajectory_mean_overlap", "pentropy", "qentropy", "top1_match",
)
PAIRED_METRICS = (
    "token_weighted_ce", "pooled_overlap", "equal_trajectory_mean_overlap",
)
POINT_FIELDS = (
    "size", "seed", "phase", "budget_target_tokens", "actual_tokens", "overshoot",
    "step", "training_seconds", "wall_seconds", "evaluation_seconds",
)
BOOTSTRAP = dict(draws=1000, random_seed=113, confidence=0.95,
                 method="equal trajectory percentile interval within one training seed")


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _stats(values):
    values = [v for v in values if _number(v)]
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                std=statistics.stdev(values) if len(values) > 1 else None)


def _percentile(values, fraction):
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _interval(inputs, previous=None):
    if previous and previous.get("inputs") == inputs and previous.get("settings") == BOOTSTRAP:
        return previous
    values = [value for _, value in inputs]
    result = dict(settings=dict(BOOTSTRAP), inputs=inputs, n=len(values),
                  mean=statistics.mean(values) if values else None, low=None, high=None)
    if values:
        rng = random.Random(113)
        count = len(values)
        draws = sorted(math.fsum(rng.choices(values, k=count)) / count for _ in range(1000))
        result.update(low=_percentile(draws, 0.025), high=_percentile(draws, 0.975))
    return result


def _replace_text(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _read_point(row, size, seed, budgets):
    phase = row["phase"]
    if type(phase) is not int or not 0 <= phase <= len(budgets):
        raise ValueError("Phase is outside the study budget range")
    budget = 0 if phase == 0 else budgets[phase - 1]
    if row.get("budget_target_tokens", budget) != budget:
        raise ValueError("Recorded phase budget differs from the plan")
    for key, value in (("size", size), ("seed", seed)):
        if row.get(key, value) != value:
            raise ValueError(f"Recorded {key} differs from its run directory")
    actual = row.get("trained_tokens", row.get("target_tokens"))
    if type(actual) is not int or actual < budget or (phase == 0 and actual != 0):
        raise ValueError("Actual token count is invalid for this budget point")
    if row.get("target_tokens", actual) != actual:
        raise ValueError("Recorded cumulative token counts disagree")
    overshoot = actual - budget
    if row.get("overshoot", overshoot) != overshoot:
        raise ValueError("Recorded token excess differs from actual minus budget")
    if phase and "last_update_tokens" in row and overshoot >= row["last_update_tokens"]:
        raise ValueError("Token excess must be smaller than the final complete update")
    result = dict(size=size, seed=seed, phase=phase, budget_target_tokens=budget,
                  actual_tokens=actual, overshoot=overshoot)
    for key in ("step", "training_seconds", "wall_seconds", "evaluation_seconds"):
        result[key] = row.get(key)
    return result


def build_report(study_root):
    """Write curves.csv and report.json, returning the report dictionary."""
    study_root = Path(study_root)
    plan = json.loads((study_root / "plan.json").read_text())
    budgets = plan["token_budgets"]
    if (not budgets or any(type(v) is not int or v <= 0 for v in budgets)
            or budgets != sorted(set(budgets))):
        raise ValueError("Plan needs strictly increasing positive token budgets")
    seeds = plan["seeds"]
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Plan needs distinct integer training seeds")
    sizes = sorted(int(size) for size in plan["subsets"])
    if sizes != [100, 200, 400]:
        raise ValueError("Expected training sizes 100, 200, 400")
    report_path = study_root / "report.json"
    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
    cached = {(r["size"], r["seed"], r["phase"], r["split"], r["kind"]): r["interval"]
              for r in previous.get("trajectory_intervals", [])}
    curves, points, intervals, availability, skipped = [], [], [], [], []
    metric_names = set(METRICS)
    for size in sizes:
        for seed in seeds:
            path = study_root / f"train_n{size}_seed{seed}" / "validation.jsonl"
            phases = set()
            if path.exists():
                for line_number, line in enumerate(path.read_text().splitlines(), 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    metrics = row.get("metrics", {})
                    if (row.get("smoke") or row.get("smoke_endpoint") or metrics.get("limited")
                            or metrics.get("complete") is False):
                        skipped.append(dict(size=size, seed=seed, line=line_number,
                                            reason="limited evaluation"))
                        continue
                    point = _read_point(row, size, seed, budgets)
                    phase = point["phase"]
                    if phase in phases:
                        raise ValueError(f"Repeated phase in {path}: {phase}")
                    phases.add(phase)
                    splits = metrics.get("splits")
                    if not isinstance(splits, dict) or not splits:
                        raise ValueError(f"Missing split metrics in {path}:{line_number}")
                    trajectories = {}
                    for item in metrics.get("per_sequence", []):
                        split, kind = item["split"], item["kind"]
                        if split not in ("train", "val"):
                            raise ValueError("Only train and val may enter this report")
                        value = item.get("overlap")
                        if not _number(value) or item.get("sample_count", 1) <= 0:
                            continue
                        bucket = trajectories.setdefault((split, kind), {})
                        name = item["name"]
                        if name in bucket:
                            raise ValueError("A trajectory repeats within a split and token type")
                        bucket[name] = value
                    points.append(point)
                    for split, kinds in sorted(splits.items()):
                        if split not in ("train", "val"):
                            raise ValueError("Only train and val may enter this report")
                        for kind, values in sorted(kinds.items()):
                            if not isinstance(values, dict):
                                raise ValueError("Token-type metrics must be a dictionary")
                            inputs = [[name, value] for name, value in
                                      sorted(trajectories.get((split, kind), {}).items())]
                            identity = (size, seed, phase, split, kind)
                            interval = _interval(inputs, cached.get(identity))
                            intervals.append(dict(size=size, seed=seed, phase=phase,
                                                  split=split, kind=kind, interval=interval))
                            flat = dict(point, split=split, kind=kind,
                                        trajectory_overlap_n=interval["n"],
                                        trajectory_overlap_mean=interval["mean"],
                                        trajectory_overlap_low95=interval["low"],
                                        trajectory_overlap_high95=interval["high"])
                            for name in METRICS:
                                flat[name] = values.get(name)
                            for name, value in values.items():
                                if _number(value) or value is None:
                                    flat[name] = value
                                    metric_names.add(name)
                            curves.append(flat)
            availability.append(dict(size=size, seed=seed, phases=sorted(phases),
                                     missing_phases=sorted(set(range(len(budgets) + 1)) - phases)))
    grouped = {}
    for row in curves:
        key = (row["size"], row["phase"], row["budget_target_tokens"], row["split"], row["kind"])
        grouped.setdefault(key, []).append(row)
    summaries = []
    for (size, phase, budget, split, kind), rows in sorted(grouped.items()):
        present = sorted(r["seed"] for r in rows)
        summaries.append(dict(size=size, phase=phase, budget_target_tokens=budget,
                              split=split, kind=kind, n=len(rows), seeds=present,
                              missing_seeds=sorted(set(seeds) - set(present)),
                              statistics={name: _stats(r.get(name) for r in rows)
                                          for name in sorted(metric_names)},
                              actual_tokens=_stats(r["actual_tokens"] for r in rows),
                              overshoot=_stats(r["overshoot"] for r in rows)))
    indexed = {(r["size"], r["seed"], r["phase"], r["split"], r["kind"]): r for r in curves}
    paired = []
    for row in curves:
        for smaller in sizes:
            if smaller >= row["size"]:
                continue
            other = indexed.get((smaller, row["seed"], row["phase"], row["split"], row["kind"]))
            if other is None:
                continue
            paired.append(dict(size_large=row["size"], size_small=smaller, seed=row["seed"],
                               phase=row["phase"], budget_target_tokens=row["budget_target_tokens"],
                               split=row["split"], kind=row["kind"],
                               actual_tokens_large=row["actual_tokens"],
                               actual_tokens_small=other["actual_tokens"],
                               actual_token_difference=row["actual_tokens"] - other["actual_tokens"],
                               differences={name: row[name] - other[name]
                                            if _number(row.get(name)) and _number(other.get(name)) else None
                                            for name in PAIRED_METRICS}))
    paired_groups = {}
    for row in paired:
        key = tuple(row[k] for k in ("size_large", "size_small", "phase", "budget_target_tokens", "split", "kind"))
        paired_groups.setdefault(key, []).append(row)
    paired_summary = []
    for key, rows in sorted(paired_groups.items()):
        value = dict(zip(("size_large", "size_small", "phase", "budget_target_tokens", "split", "kind"), key))
        present = sorted(r["seed"] for r in rows)
        value.update(n=len(rows), seeds=present, missing_seeds=sorted(set(seeds) - set(present)),
                     differences={name: _stats(r["differences"][name] for r in rows) for name in PAIRED_METRICS},
                     actual_token_difference=_stats(r["actual_token_difference"] for r in rows))
        paired_summary.append(value)
    report = dict(version=1, study_root=str(study_root), token_budgets=budgets,
                  planned_seeds=seeds, availability=availability, points=points,
                  seed_summary=summaries, paired_seed_differences=paired,
                  paired_summary=paired_summary, trajectory_intervals=intervals,
                  skipped_rows=skipped, bootstrap_settings=dict(BOOTSTRAP),
                  difference_definition="larger training size minus smaller training size; same seed and phase",
                  std_definition="sample standard deviation across available training seeds; null for n < 2",
                  interpretation=[
                      "Only existing completed evaluation rows are summarized; runs may still be in progress",
                      "Three training seeds provide descriptive evidence; no significance claim is made",
                      "Trajectory intervals condition on one trained model and do not estimate training seed uncertainty",
                      "The same evaluation trajectories recur across seeds; they are not pooled as independent observations",
                      "Budget points retain their actual cumulative tokens and final-update excess",
                      "Absent EOS or unavailable metrics remain null; raw CE comparisons stay within token type",
                  ])
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    fields = list(POINT_FIELDS) + ["split", "kind"] + sorted(metric_names) + [
        "trajectory_overlap_n", "trajectory_overlap_mean", "trajectory_overlap_low95", "trajectory_overlap_high95"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(curves)
    _replace_text(study_root / "curves.csv", buffer.getvalue())
    _replace_text(report_path, encoded)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.study_root)
    print(json.dumps(dict(points=len(report["points"]), groups=len(report["seed_summary"]),
                          curves_csv=str(args.study_root / "curves.csv"),
                          report_json=str(args.study_root / "report.json")), allow_nan=False))


if __name__ == "__main__":
    main()
