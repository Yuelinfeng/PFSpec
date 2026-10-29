"""Summarize paired CE / Feature KD curves and reproduce exposure counts."""

import argparse
import json
import math
from pathlib import Path
import random
import statistics


OBJECTIVES = ("ce", "feature_kd")
KINDS = ("all", "position", "feature", "eos")
SPLITS = ("train", "val")
METRICS = (
    "token_weighted_ce", "target_token_weighted_ce",
    "target_token_weighted_raw_entropy", "pooled_overlap",
    "equal_trajectory_mean_overlap", "pentropy", "qentropy", "top1_match",
)
MATCH_FIELDS = (
    "step", "trained_tokens", "budget_target_tokens", "last_update_tokens", "overshoot",
)
COLORS = {"ce": "#0072B2", "feature_kd": "#D55E00"}
LABELS = {"ce": "CE", "feature_kd": "Feature KD"}
PANELS = (
    ("Feature: validation overlap", "Overlap (%)", "val", "feature", "pooled_overlap", 100),
    ("Position: validation overlap", "Overlap (%)", "val", "position", "pooled_overlap", 100),
    ("Feature: validation sampling entropy", "Entropy (nats)", "val", "feature", "qentropy", 1),
    ("Feature: training hard-label CE", "CE (nats)", "train", "feature", "token_weighted_ce", 1),
    ("Feature: validation hard-label CE", "CE (nats)", "val", "feature", "token_weighted_ce", 1),
    ("Paired Feature validation overlap", "KD minus CE (percentage points)", "val", "feature", "pooled_overlap", 100),
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _stats(values):
    values = list(values)
    observed = [v for v in values if v is not None]
    _require(all(_number(v) for v in observed), "Statistics require finite numbers or null")
    return dict(n=len(observed), missing_n=len(values) - len(observed),
                mean=statistics.mean(observed) if observed else None,
                std=statistics.stdev(observed) if len(observed) > 1 else None)


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save(path, value):
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _plans(study_root, reference_root):
    plan, reference = _load(study_root / "plan.json"), _load(reference_root / "plan.json")
    _require(plan.get("objective") == "feature_kd", "Expected objective feature_kd")
    _require(reference.get("objective", "ce") == "ce", "Reference objective must be ce")
    _require(plan.get("size", 400) == 400, "Only size400 is supported")
    seeds, budgets = plan.get("seeds"), plan.get("token_budgets")
    _require(isinstance(seeds, list) and len(seeds) == 3 and
             all(type(s) is int for s in seeds) and len(set(seeds)) == 3,
             "Expected three distinct integer seeds")
    _require(isinstance(budgets, list) and len(budgets) == 6 and
             all(type(b) is int and b > 0 for b in budgets) and
             budgets == sorted(set(budgets)), "Expected six increasing positive budgets")
    for key in ("seeds", "token_budgets", "val_files", "train_eval_files"):
        _require(plan.get(key) == reference.get(key), f"Plan differs from CE: {key}")
    names = plan.get("subsets", {}).get("400")
    _require(isinstance(names, list) and len(names) == 400 and len(set(names)) == 400,
             "Expected 400 distinct training filenames")
    _require(names == reference.get("subsets", {}).get("400"), "Ordered training400 lists differ")
    for key, count in (("val_files", 50), ("train_eval_files", 32)):
        values = plan[key]
        _require(isinstance(values, list) and len(values) == count and len(set(values)) == count,
                 f"Wrong evaluation file count: {key}")
    _require(not set(names).intersection(plan["val_files"]), "Training and validation files overlap")
    _require(set(plan["train_eval_files"]).issubset(names), "Training evaluation is outside training400")
    for name in names + plan["val_files"] + plan["train_eval_files"]:
        _require(isinstance(name, str) and Path(name).name == name and name.endswith(".pt"),
                 "Corpus entries must be plain .pt filenames")
    for key, expected in (("lr", 3e-4), ("weight_decay", 0.01),
                          ("gradient_clip", 1.0), ("trajectories_per_update", 2)):
        _require(plan.get(key) == reference.get(key) == expected, f"Training setting differs: {key}")
    _require(Path(plan["source_root"]).resolve() == Path(reference["source_root"]).resolve(),
             "Source roots differ")
    if "reference_root" in plan:
        _require(Path(plan["reference_root"]).resolve() == reference_root.resolve(),
                 "reference_root differs from the new plan")
    return plan


def _numeric_tree(value, location):
    if isinstance(value, dict):
        for key, item in value.items():
            _numeric_tree(item, f"{location}/{key}")
    else:
        _require(value is None or _number(value), f"Expected finite numeric metric: {location}")


def _validate_row(row, seed, objective, plan, location):
    phase = row.get("phase")
    _require(type(phase) is int and 0 <= phase <= 6, f"Invalid phase: {location}")
    _require(row.get("seed") == seed and row.get("size") == 400, f"Run identity differs: {location}")
    _require(row.get("objective", objective) == objective, f"Objective differs: {location}")
    for key in MATCH_FIELDS:
        _require(type(row.get(key)) is int and row[key] >= 0, f"Missing or invalid {key}: {location}")
    budget = 0 if phase == 0 else plan["token_budgets"][phase - 1]
    _require(row["budget_target_tokens"] == budget and row["trained_tokens"] >= budget,
             f"Wrong budget or cumulative tokens: {location}")
    _require(row["overshoot"] == row["trained_tokens"] - budget,
             f"Wrong final-update excess: {location}")
    _require(row.get("target_tokens", row["trained_tokens"]) == row["trained_tokens"],
             f"Cumulative token fields disagree: {location}")
    if phase == 0:
        _require(all(row[key] == 0 for key in MATCH_FIELDS), f"Initial point is not zero: {location}")
    else:
        _require(row["step"] > 0 and row["last_update_tokens"] > row["overshoot"],
                 f"Point is not the first full update reaching its budget: {location}")
    if objective == "feature_kd":
        _require(type(row.get("trained_feature_tokens")) is int and row["trained_feature_tokens"] >= 0,
                 f"Missing or invalid trained_feature_tokens: {location}")
    for key in ("training_seconds", "evaluation_seconds", "wall_seconds"):
        _require(_number(row.get(key)) and row[key] >= 0, f"Missing or invalid {key}: {location}")
    metrics = row["metrics"]
    _require(metrics.get("complete") is True and metrics.get("limited") is False and
             metrics.get("max_records") is None, f"Evaluation is not explicitly complete: {location}")
    _require(metrics.get("files") == dict(val=plan["val_files"], train=plan["train_eval_files"]),
             f"Evaluation filenames differ: {location}")
    _require(metrics.get("checkpoint_step") == row["step"], f"Evaluation step differs: {location}")
    _require(metrics.get("target_precision") == metrics.get("draft_precision") == "FP32",
             f"Evaluation precision differs: {location}")
    policy = metrics.get("policy", {})
    _require(policy.get("temperature") == 0.9 and policy.get("top_p") == 0.9,
             f"Evaluation sampling settings differ: {location}")
    splits = metrics.get("splits")
    _require(isinstance(splits, dict) and set(splits) == set(SPLITS), f"Missing split metrics: {location}")
    _numeric_tree(splits, location)
    for split in SPLITS:
        _require(isinstance(splits[split], dict) and set(splits[split]) == set(KINDS),
                 f"Missing token types: {location}/{split}")
        for kind in KINDS:
            values = splits[split][kind]
            _require(isinstance(values, dict) and all(key in values for key in METRICS),
                     f"Missing metrics: {location}/{split}/{kind}")
            _require(values.get("sequence_count") == len(metrics["files"][split]),
                     f"Evaluation sequence count differs: {location}/{split}/{kind}")
            for key in ("pooled_overlap", "equal_trajectory_mean_overlap", "top1_match"):
                value = values[key]
                _require(value is None or 0 <= value <= 1, f"Probability metric outside [0,1]: {location}")
            for key in METRICS[:3] + ("pentropy", "qentropy"):
                value = values[key]
                _require(value is None or value >= 0, f"Negative CE or entropy: {location}")
            if kind != "eos":
                _require(all(_number(values[key]) for key in METRICS),
                         f"Required P/F/all metric is unavailable: {location}/{split}/{kind}")
    return dict(row, objective=objective)


def _read_rows(root, seed, objective, plan):
    run = root / f"train_n400_seed{seed}"
    path = run / "validation.jsonl"
    marker = run / "complete.json"
    declared_complete = False
    if marker.exists():
        declared_complete = _load(marker).get("status") == "complete"
    if not path.exists():
        _require(objective != "ce" and not declared_complete, f"Missing evaluation history: {path}")
        return [], [], dict(objective=objective, seed=seed, phases=[], missing_phases=list(range(7)),
                            complete=False, declared_complete=False, state="not_started")
    rows, skipped, phases = [], [], set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        location = f"{path}:{line_number}"
        row = json.loads(line)
        _require(isinstance(row, dict) and isinstance(row.get("metrics"), dict),
                 f"Missing evaluation row or metrics: {location}")
        metrics = row["metrics"]
        if (row.get("smoke") or row.get("smoke_endpoint") or metrics.get("limited") or
                metrics.get("complete") is False or metrics.get("max_records") is not None):
            skipped.append(dict(objective=objective, seed=seed, line=line_number,
                                phase=row.get("phase"), reason="nonformal evaluation"))
            continue
        value = _validate_row(row, seed, objective, plan, location)
        phase = value["phase"]
        _require(phase not in phases, f"Duplicate phase {phase}: {path}")
        _require(phase == len(rows), f"Missing or out-of-order phase before {phase}: {path}")
        if rows:
            previous = rows[-1]
            _require(value["step"] > previous["step"] and value["trained_tokens"] > previous["trained_tokens"],
                     f"Training progress does not increase: {location}")
            _require(value["training_seconds"] >= previous["training_seconds"] and
                     value["wall_seconds"] >= previous["wall_seconds"], f"Cumulative time decreases: {location}")
        rows.append(value)
        phases.add(phase)
    missing = sorted(set(range(7)) - phases)
    _require(not missing or (objective != "ce" and not declared_complete),
             f"Missing required phases {missing}: {path}")
    return rows, skipped, dict(objective=objective, seed=seed, phases=sorted(phases),
                              missing_phases=missing, complete=not missing,
                              declared_complete=declared_complete,
                              state="complete" if not missing else "in_progress")


def _training_counts(plan):
    """Read the original ordered training records; no model is constructed."""
    from draft_a.analyze_v2 import read_records

    source_root = Path(plan["source_root"])
    entries = _load(source_root / "plan.json")["entries"]
    by_name = {entry["name"]: entry for entry in entries}
    _require(len(by_name) == len(entries), "Source plan repeats a filename")
    names = plan["subsets"]["400"]
    _require(all(name in by_name and by_name[name]["split"] == "train" for name in names),
             "Every exposure record must belong to the source training split")
    records = read_records(source_root, [by_name[name] for name in names])
    _require([record["name"] for record in records] == names, "Read training order differs from the plan")
    counts = []
    for record in records:
        body = len(record["tokens"])
        _require(body > 0 and body % 2 == 0, f"Training body must contain complete P/F pairs: {record['name']}")
        eos = int(bool(record["has_eos"]))
        counts.append(dict(feature=body // 2, position=body // 2, eos=eos, total=body + eos))
    return counts


def _replay(counts, seed, steps):
    _require(len(counts) == 400, "Exposure replay requires 400 trajectories")
    steps = set(steps)
    _require(all(type(step) is int and step >= 0 for step in steps), "Invalid replay step")
    rng = random.Random(seed + 1)
    order, cursor = [], 0
    cumulative = dict(feature=0, position=0, eos=0, total=0)
    result = {0: dict(cumulative, last_update_tokens=0)} if 0 in steps else {}
    for step in range(1, max(steps, default=0) + 1):
        update = dict(feature=0, position=0, eos=0, total=0)
        for _ in range(2):
            if cursor >= len(order):
                order = list(range(len(counts)))
                rng.shuffle(order)
                cursor = 0
            item = counts[order[cursor]]
            cursor += 1
            for key in cumulative:
                update[key] += item[key]
        for key in cumulative:
            cumulative[key] += update[key]
        if step in steps:
            result[step] = dict(cumulative, last_update_tokens=update["total"])
    return result


def _validate_exposure(row, counts):
    location = f"{row['objective']}/seed{row['seed']}/phase{row['phase']}"
    _require(counts["total"] == row["trained_tokens"] and
             counts["last_update_tokens"] == row["last_update_tokens"],
             f"Exposure replay differs from recorded tokens: {location}")
    if row["objective"] == "feature_kd":
        _require(counts["feature"] == row["trained_feature_tokens"],
                 f"Exposure replay differs from the Feature counter: {location}")
    return dict(objective=row["objective"], seed=row["seed"], phase=row["phase"],
                step=row["step"], trained_feature_tokens=counts["feature"],
                trained_position_tokens=counts["position"], trained_eos_tokens=counts["eos"],
                trained_tokens=counts["total"], last_update_tokens=counts["last_update_tokens"])


def _metric_stats(rows, field):
    result = {}
    for split in SPLITS:
        result[split] = {}
        for kind in KINDS:
            values = [row[field][split][kind] for row in rows]
            keys = sorted(set().union(*(v.keys() for v in values))) if values else list(METRICS)
            result[split][kind] = {key: _stats(v.get(key) for v in values) for key in keys}
    return result


def _pair(ce, kd):
    location = f"seed{kd['seed']}/phase{kd['phase']}"
    for key in MATCH_FIELDS:
        _require(ce[key] == kd[key], f"Paired {key} differs: {location}")
    a, b = ce["metrics"], kd["metrics"]
    for key in ("files", "policy", "scope", "entropy_units", "ce_definition", "target_precision", "draft_precision"):
        _require(a.get(key) == b.get(key), f"Paired evaluation {key} differs: {location}")
    if kd["phase"] == 0:
        _require(a["splits"] == b["splits"], f"Initial split metrics differ: {location}")
    differences = {}
    for split in SPLITS:
        differences[split] = {}
        for kind in KINDS:
            left, right = a["splits"][split][kind], b["splits"][split][kind]
            _require(set(left) == set(right), f"Paired metric fields differ: {location}/{split}/{kind}")
            values = {}
            for name in left:
                x, y = left[name], right[name]
                _require((x is None) == (y is None), f"Paired metric availability differs: {location}/{name}")
                if name.startswith("target_") or name == "pentropy" or name.endswith("count"):
                    _require(x == y, f"Target or evaluation counts differ: {location}/{split}/{kind}/{name}")
                values[name] = y - x if x is not None else None
            differences[split][kind] = values
    return dict(seed=kd["seed"], phase=kd["phase"],
                **{key: kd[key] for key in MATCH_FIELDS},
                trained_feature_tokens=kd["trained_feature_tokens"], differences=differences,
                training_seconds_difference=kd["training_seconds"] - ce["training_seconds"],
                evaluation_seconds_difference=kd["evaluation_seconds"] - ce["evaluation_seconds"])


def _cache_cost(root, directory):
    path = root / directory / "summary.json"
    if not path.exists():
        return dict(path=str(path), available=False, seconds=None, bytes_on_disk=None)
    summary = _load(path)
    seconds = summary.get("seconds")
    _require(seconds is None or (_number(seconds) and seconds >= 0), f"Invalid cache seconds: {path}")
    byte_count = sum(p.stat().st_size for p in (root / directory).iterdir()
                     if p.is_file() and p.suffix == ".pt")
    return dict(path=str(path), available=True, seconds=seconds, bytes_on_disk=byte_count,
                summary=summary,
                time_scope="Recorded preparation invocation; reuse or earlier failed attempts may be unaccounted for")


def _costs(records, paired, study_root, reference_root):
    objectives = {}
    for objective in OBJECTIVES:
        rows = [row for row in records if row["objective"] == objective]
        latest = {}
        for row in rows:
            latest[row["seed"]] = row
        values = [dict(seed=seed, phase=row["phase"], step=row["step"],
                       training_seconds=row["training_seconds"], wall_seconds=row["wall_seconds"],
                       evaluation_seconds=sum(r["evaluation_seconds"] for r in rows if r["seed"] == seed))
                  for seed, row in sorted(latest.items())]
        objectives[objective] = dict(latest_by_seed=values,
                                    training_seconds=_stats(v["training_seconds"] for v in values),
                                    training_seconds_total=sum(v["training_seconds"] for v in values),
                                    evaluation_seconds_total=sum(v["evaluation_seconds"] for v in values),
                                    scope="Latest available point per seed; objective totals may cover different progress")
    paired_time = []
    for phase in range(7):
        rows = [row for row in paired if row["phase"] == phase]
        paired_time.append(dict(phase=phase, n=len(rows),
                                training_seconds_difference=_stats(r["training_seconds_difference"] for r in rows),
                                evaluation_seconds_difference=_stats(r["evaluation_seconds_difference"] for r in rows)))
    return dict(objectives=objectives, paired_by_phase=paired_time,
                feature_label_preparation=_cache_cost(study_root, "feature_labels"),
                shared_evaluation_preparation=_cache_cost(reference_root, "teacher_cache"),
                note="Cumulative training time is never summed across phases; shared evaluation cache is charged once; unavailable cost remains null")


def build_report(study_root: Path, reference_root: Path) -> dict:
    """Write comparison_summary.json and comparison_report.json after pairing."""
    study_root, reference_root = Path(study_root), Path(reference_root)
    plan = _plans(study_root, reference_root)
    seeds, budgets = plan["seeds"], plan["token_budgets"]
    records, availability, skipped = [], [], []
    for objective, root in (("ce", reference_root), ("feature_kd", study_root)):
        for seed in seeds:
            rows, omitted, state = _read_rows(root, seed, objective, plan)
            records.extend(rows)
            skipped.extend(omitted)
            availability.append(state)
    counts = _training_counts(plan)
    exposure = []
    for seed in seeds:
        rows = [row for row in records if row["seed"] == seed]
        replay = _replay(counts, seed, [row["step"] for row in rows])
        exposure.extend(_validate_exposure(row, replay[row["step"]]) for row in rows)
    index = {(row["objective"], row["seed"], row["phase"]): row for row in records}
    paired = [_pair(index[("ce", row["seed"], row["phase"])], row)
              for row in records if row["objective"] == "feature_kd"]
    summaries, paired_summary = [], []
    for phase in range(7):
        budget = 0 if phase == 0 else budgets[phase - 1]
        for objective in OBJECTIVES:
            rows = [row for row in records if row["objective"] == objective and row["phase"] == phase]
            present = [row["seed"] for row in rows]
            summaries.append(dict(objective=objective, phase=phase, budget_target_tokens=budget,
                                  n=len(rows), seeds=present, missing_seeds=[s for s in seeds if s not in present],
                                  trained_tokens=_stats(row["trained_tokens"] for row in rows),
                                  trained_feature_tokens=_stats(
                                      item["trained_feature_tokens"] for item in exposure
                                      if item["objective"] == objective and item["phase"] == phase),
                                  training_seconds=_stats(row["training_seconds"] for row in rows),
                                  metrics=_metric_stats([dict(row, values=row["metrics"]["splits"]) for row in rows], "values")))
        rows = [row for row in paired if row["phase"] == phase]
        present = [row["seed"] for row in rows]
        paired_summary.append(dict(phase=phase, budget_target_tokens=budget, n=len(rows), seeds=present,
                                   missing_seeds=[s for s in seeds if s not in present],
                                   trained_tokens=_stats(row["trained_tokens"] for row in rows),
                                   differences=_metric_stats(rows, "differences")))
    completed = sum(a["complete"] for a in availability if a["objective"] == "feature_kd")
    status = "COMPLETE" if completed == 3 else "PARTIAL"
    common = dict(version=1, status=status, completed_seed_count=completed, planned_seed_count=3,
                  study_root=str(study_root), reference_root=str(reference_root),
                  seeds=seeds, size=400, token_budgets=budgets)
    summary = dict(common, records=records, skipped_rows=skipped)
    report = dict(common, availability=availability, seed_summary=summaries,
                  paired_seed_differences=paired, paired_summary=paired_summary,
                  exposure=exposure, exposure_per_pass={key: sum(c[key] for c in counts)
                                                      for key in ("feature", "position", "eos", "total")},
                  costs=_costs(records, paired, study_root, reference_root), skipped_rows=skipped,
                  difference_definition="Feature KD minus CE; same seed and phase; overlap differences stored as fractions",
                  std_definition="Sample SD across available training seeds; null for n < 2; no confidence intervals",
                  interpretation=[
                      "Unstarted seeds and trailing phases remain explicitly unavailable",
                      "Three completed seeds are required for COMPLETE",
                      "Overlap uses teacher prefixes with temperature0.9/top_p0.9; Feature training uses temperature1",
                      "Training and validation CE refer to observed hard labels in both objectives",
                      "Equal Feature exposure does not imply equal information or computational cost",
                      "These development curves do not establish online throughput, scene generalization or shared-layer interference",
                  ])
    _save(study_root / "comparison_summary.json", summary)
    _save(study_root / "comparison_report.json", report)
    return report


def _plot_series(report, panel_index):
    _, _, split, kind, metric, scale = PANELS[panel_index]
    series = []
    for objective in (("feature_kd",) if panel_index == 5 else OBJECTIVES):
        rows = report["paired_summary"] if panel_index == 5 else [
            row for row in report["seed_summary"] if row["objective"] == objective]
        points = []
        for row in sorted(rows, key=lambda r: r["phase"]):
            value = row["differences" if panel_index == 5 else "metrics"][split][kind][metric]
            if value["n"] == 0:
                continue
            _require(value["n"] == row["n"] and row["trained_tokens"]["n"] == row["n"],
                     "Plot metric and token counts differ")
            points.append(dict(phase=row["phase"], n=value["n"], seeds=row["seeds"],
                               x=row["trained_tokens"]["mean"] / 1_000_000,
                               trained_tokens=row["trained_tokens"],
                               mean=value["mean"] * scale,
                               std=None if value["std"] is None else value["std"] * scale))
        series.append(dict(objective=objective, label="Paired KD minus CE" if panel_index == 5 else LABELS[objective],
                           color=COLORS[objective], points=points))
    return series


def _draw_series(ax, series, linestyle="-"):
    points = series["points"]
    if not points:
        return
    x, y = [p["x"] for p in points], [p["mean"] for p in points]
    ax.plot(x, y, color=series["color"], label=series["label"], linestyle=linestyle,
            marker="o", markersize=4.5, markerfacecolor="white", linewidth=2)
    # Null SD stays absent rather than becoming a fabricated zero-width interval.
    band_x, band_low, band_high = [], [], []
    for point in points:
        deviation = point["std"]
        band_x.append(point["x"])
        band_low.append(math.nan if deviation is None else point["mean"] - deviation)
        band_high.append(math.nan if deviation is None else point["mean"] + deviation)
    ax.fill_between(band_x, band_low, band_high, color=series["color"], alpha=0.16)
    available = [p for p in points if p["std"] is not None]
    if available:
        ax.errorbar([p["x"] for p in available], [p["mean"] for p in available],
                    yerr=[p["std"] for p in available], fmt="none", ecolor=series["color"],
                    elinewidth=0.8, capsize=2, alpha=0.8)


def plot_report(study_root: Path) -> dict:
    """Render a 2x3 comparison figure from the most recently built report."""
    study_root = Path(study_root)
    report = _load(study_root / "comparison_report.json")
    _require(report.get("planned_seed_count") == 3 and report.get("size") == 400,
             "Expected a three-seed size400 comparison report")
    completed = report["completed_seed_count"]
    _require(type(completed) is int and 0 <= completed <= 3 and
             report["status"] == ("COMPLETE" if completed == 3 else "PARTIAL"),
             "Report completion status is inconsistent")
    panels = []
    for index, (title, ylabel, *_) in enumerate(PANELS):
        panels.append(dict(title=title, ylabel=ylabel, series=_plot_series(report, index)))
    # Target is one shared reference, not a third trained model or training seed.
    target_points = []
    for row in report["seed_summary"]:
        if row["objective"] != "ce" or not row["n"]:
            continue
        value = row["metrics"]["val"]["feature"]["pentropy"]
        _require(value["n"] == row["n"], "Target entropy is unavailable")
        target_points.append(dict(phase=row["phase"], x=row["trained_tokens"]["mean"] / 1_000_000,
                                  mean=value["mean"], std=value["std"], n=value["n"]))
    _require(target_points and all(p["mean"] == target_points[0]["mean"] and p["std"] == 0
                                  for p in target_points), "Shared Target entropy changes across seeds or phases")
    paths = [study_root / "comparison_curves.png", study_root / "comparison_curves.pdf"]
    details = dict(version=1, status=report["status"], completed_seed_count=completed,
                   planned_seed_count=3, seeds=report["seeds"], outputs=[str(p) for p in paths],
                   x_definition="Phase means of actual cumulative training tokens, in millions",
                   dispersion="Mean plus/minus one sample SD across training seeds; n shown per phase; no confidence intervals",
                   panels=panels, target_pentropy=target_points)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as error:
        raise RuntimeError("matplotlib is unavailable; no packages were installed") from error
    style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
             "axes.spines.top": False, "axes.spines.right": False,
             "pdf.fonttype": 42, "savefig.facecolor": "white"}
    with plt.rc_context(style):
        fig, grid = plt.subplots(2, 3, figsize=(16.8, 9.6), sharex=True)
        axes = grid.ravel()
        try:
            fig.suptitle("Joint P/F Draft: CE vs Feature distribution training", fontsize=17, y=0.98)
            fig.text(0.5, 0.945, f"{report['status']} | {completed}/3 completed KD seeds | 400 training trajectories",
                     ha="center", fontsize=11)
            legends = [Line2D([0], [0], color=COLORS[o], marker="o", markerfacecolor="white",
                              linewidth=2, label=LABELS[o]) for o in OBJECTIVES]
            legends.append(Line2D([0], [0], color=COLORS["ce"], linestyle=":", label="Shared Target sampling entropy"))
            fig.legend(handles=legends, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, frameon=False)
            for index, (ax, panel) in enumerate(zip(axes, panels)):
                for series in panel["series"]:
                    _draw_series(ax, series)
                if index == 2:
                    ax.axhline(target_points[0]["mean"], color=COLORS["ce"], linestyle=":", linewidth=1.5)
                if index == 5:
                    ax.axhline(0, color=COLORS["ce"], linestyle=":", linewidth=1)
                ax.set_title(f"{chr(65 + index)}. {panel['title']}", loc="left", pad=10)
                ax.set_ylabel(panel["ylabel"])
                ax.set_xlim(left=0)
                if index >= 3:
                    ax.set_xlabel("Actual training tokens (millions)")
                ax.grid(axis="y", alpha=0.22, linewidth=0.6)
                ax.set_axisbelow(True)
                if index == 1:
                    ax.set_ylim(0, 100)
                count_labels = []
                for series in panel["series"]:
                    counts = ",".join(str(p["n"]) for p in series["points"])
                    phases = ",".join(str(p["phase"]) for p in series["points"])
                    count_labels.append(f"{series['label']}: phases [{phases}], n [{counts}]")
                ax.text(0.01, -0.26, "\n".join(count_labels), transform=ax.transAxes,
                        fontsize=7.5, ha="left", va="top")
            ce_bounds = [p["mean"] + sign * (p["std"] or 0)
                         for index in (3, 4) for series in panels[index]["series"]
                         for p in series["points"] for sign in (-1, 1)]
            if ce_bounds:
                low, high = min(0, min(ce_bounds)), max(ce_bounds)
                pad = max(0.1, (high - low) * 0.06)
                for index in (3, 4):
                    axes[index].set_ylim(low - pad, high + pad)
            fig.text(0.5, 0.035, "Mean +/-1 sample SD across 3 seeds when complete; available n varies while PARTIAL. n < 2: SD absent.",
                     ha="center", fontsize=9)
            fig.text(0.5, 0.014, "Teacher-prefix overlap at temperature0.9 / top-p0.9; hard-label CE. SD is not a confidence interval; no online speed claim.",
                     ha="center", fontsize=9)
            fig.subplots_adjust(left=0.065, right=0.985, top=0.84, bottom=0.20, wspace=0.31, hspace=0.72)
            for path in paths:
                fig.savefig(path, dpi=210, bbox_inches="tight", pad_inches=0.15)
        finally:
            plt.close(fig)
    _save(study_root / "plot_data.json", details)
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.reference_root is None and not args.plot:
        parser.error("--reference-root is required unless --plot is selected")
    if args.reference_root is not None:
        build_report(args.study_root, args.reference_root)
    result = plot_report(args.study_root) if args.plot else _load(args.study_root / "comparison_report.json")
    print(json.dumps(dict(status=result["status"], completed_seed_count=result["completed_seed_count"],
                          study_root=str(args.study_root)), allow_nan=False))


if __name__ == "__main__":
    main()
