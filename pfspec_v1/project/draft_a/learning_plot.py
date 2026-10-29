"""Plot size-wise means and sample deviations across training seeds."""

import argparse
import json
import math
from pathlib import Path
import statistics


PANELS = (
    ("Position: validation overlap", "Overlap (%) - higher is better",
     (("val", "position", "pooled_overlap", "P", "o", 100.0),)),
    ("Feature: validation overlap", "Overlap (%) - higher is better",
     (("val", "feature", "pooled_overlap", "F", "o", 100.0),)),
    ("Feature: training cross entropy", "CE (nats) - lower is better",
     (("train", "feature", "token_weighted_ce", "Train F", "o", 1.0),)),
    ("Feature: validation cross entropy", "CE (nats) - lower is better",
     (("val", "feature", "token_weighted_ce", "Val F", "o", 1.0),)),
)
EXPECTED_RUNS = {(size, seed) for size in (100, 200, 400)
                 for seed in (24681, 24682, 24683)}
COLORS = {100: "#0072B2", 200: "#009E73", 400: "#D55E00"}


def load_curves(summary_path):
    """Read actual cumulative tokens; retain each observed size and seed."""
    summary = json.loads(Path(summary_path).read_text())
    groups = {}
    identities = set()
    skipped = 0
    for row in summary["records"]:
        metrics = row["metrics"]
        if (row.get("smoke") or row.get("smoke_endpoint") or metrics.get("limited")
                or metrics.get("complete") is False):
            skipped += 1
            continue
        size, seed, phase = row["size"], row["seed"], row["phase"]
        if any(type(value) is not int for value in (size, seed, phase)):
            raise ValueError("Size, seed and phase must be integers")
        identity = (size, seed, phase)
        if identity in identities:
            raise ValueError(f"Repeated curve point: {identity}")
        identities.add(identity)
        actual = row.get("trained_tokens", row.get("target_tokens"))
        if type(actual) is not int or actual < 0:
            raise ValueError("Each point needs a nonnegative actual token count")
        if row.get("target_tokens", actual) != actual:
            raise ValueError("Cumulative token fields disagree")
        point = dict(phase=phase, actual_tokens=actual, x=actual / 1_000_000,
                     values=[], evaluation_counts={})
        for _, _, series in PANELS:
            values = []
            for split, kind, field, _, _, scale in series:
                value = metrics["splits"][split][kind][field]
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(f"Unavailable plot value: {identity}, {split}, {kind}, {field}")
                if field == "pooled_overlap" and not 0 <= value <= 1:
                    raise ValueError("Overlap must lie between zero and one")
                values.append(value * scale)
                point["evaluation_counts"][split] = metrics["splits"][split][kind].get("sequence_count")
            point["values"].append(values)
        groups.setdefault((size, seed), []).append(point)
    if not groups:
        raise ValueError("No complete evaluation points are available")
    for points in groups.values():
        points.sort(key=lambda point: point["phase"])
        if any(a["actual_tokens"] >= b["actual_tokens"] for a, b in zip(points, points[1:])):
            raise ValueError("Actual training tokens must increase within each curve")
    return groups, skipped


def summarize_curves(groups):
    """Average each phase across available seeds without inventing dispersion."""
    by_size = {}
    for (size, seed), points in sorted(groups.items()):
        for point in points:
            by_size.setdefault(size, {}).setdefault(point["phase"], []).append((seed, point))
    curves = []
    for size, phases in sorted(by_size.items()):
        points = []
        for phase, rows in sorted(phases.items()):
            seed_ids = [seed for seed, _ in rows]
            actual = [point["actual_tokens"] for _, point in rows]
            values, deviations = [], []
            for panel_index, (_, _, series) in enumerate(PANELS):
                means, stds = [], []
                for series_index in range(len(series)):
                    samples = [point["values"][panel_index][series_index] for _, point in rows]
                    means.append(statistics.mean(samples))
                    stds.append(statistics.stdev(samples) if len(samples) > 1 else None)
                values.append(means)
                deviations.append(stds)
            points.append(dict(phase=phase, seed_count=len(rows), seeds=seed_ids,
                               actual_tokens=statistics.mean(actual),
                               actual_tokens_by_seed=dict(zip(seed_ids, actual)),
                               x=statistics.mean(actual) / 1_000_000,
                               values=values, std=deviations))
        if any(a["x"] >= b["x"] for a, b in zip(points, points[1:])):
            raise ValueError("Mean actual tokens must increase across phases")
        curves.append(dict(size=size, seed_count=len({seed for rows in phases.values() for seed, _ in rows}),
                           plot_points=len(points), points=points))
    return curves


def plot_curves(summary_path, output_dir=None):
    """Save learning_curves.png and learning_curves.pdf; return numeric details."""
    summary_path = Path(summary_path)
    output_dir = summary_path.parent if output_dir is None else Path(output_dir)
    groups, skipped = load_curves(summary_path)
    curves = summarize_curves(groups)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is unavailable; no figures were created and no packages were installed") from error

    seeds = sorted({seed for _, seed in groups})
    partial = (set(groups) != EXPECTED_RUNS or any(
        {point["phase"] for point in points} != set(range(7)) for points in groups.values()))
    status = "PARTIAL STUDY" if partial else "COMPLETE STUDY"
    if not partial:
        seed_label = "mean ±1 SD across 3 training seeds"
    elif len(seeds) == 1:
        seed_label = "SINGLE TRAINING SEED; no SD available"
    else:
        seed_label = "mean ±1 SD across available training seeds; counts vary by phase"
    title = "P / F learning by training-set size"
    subtitle = f"{status} | {len(groups)}/9 training curves | {seed_label}"
    f_lower = min(point["values"][1][0] - (point["std"][1][0] or 0)
                  for curve in curves for point in curve["points"])
    f_upper = max(point["values"][1][0] + (point["std"][1][0] or 0)
                  for curve in curves for point in curve["points"])
    ce_upper = max(point["values"][index][0] + (point["std"][index][0] or 0)
                   for curve in curves for point in curve["points"] for index in (2, 3))
    ce_lower = min(point["values"][index][0] - (point["std"][index][0] or 0)
                   for curve in curves for point in curve["points"] for index in (2, 3))
    style = {
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
    paths = [output_dir / "learning_curves.png", output_dir / "learning_curves.pdf"]
    with plt.rc_context(style):
        fig, axes_grid = plt.subplots(2, 2, figsize=(14.8, 9.4), sharex=True)
        axes = axes_grid.ravel()
        try:
            fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
            fig.text(0.5, 0.94, subtitle, ha="center", fontsize=10, color="#666666")
            for panel_index, (ax, (panel_title, ylabel, series)) in enumerate(zip(axes, PANELS)):
                drawn_points = 0
                for curve in curves:
                    size, points = curve["size"], curve["points"]
                    x = [point["x"] for point in points]
                    for series_index, (_, _, _, label, marker, _) in enumerate(series):
                        y = [point["values"][panel_index][series_index] for point in points]
                        color = COLORS.get(size, "#666666")
                        line, = ax.plot(x, y, color=color, linestyle="-",
                                linewidth=2.3, marker=marker, markersize=5.5,
                                markerfacecolor="white", markeredgewidth=1.5,
                                label=f"{size} training trajectories")
                        drawn_points += len(line.get_xdata())
                        available = [(point["x"], mean, point["std"][panel_index][series_index])
                                     for point, mean in zip(points, y)
                                     if point["std"][panel_index][series_index] is not None]
                        if available:
                            ax.errorbar([item[0] for item in available], [item[1] for item in available],
                                        yerr=[item[2] for item in available], fmt="none", ecolor=color,
                                        elinewidth=1.1, capsize=3.5, capthick=1.1, alpha=0.85)
                ax.set_title(f"{chr(65 + panel_index)}. {panel_title}", loc="left", pad=12)
                if panel_index >= 2:
                    ax.set_xlabel("Mean actual training tokens (millions)")
                ax.set_ylabel(ylabel)
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)
                if panel_index == 0:
                    ax.set_ylim(0, 100)
                elif panel_index == 1:
                    ax.set_ylim(min(0, f_lower - 0.5), max(2, math.ceil((f_upper + 1) / 2) * 2))
                else:
                    ax.set_ylim(min(0, ce_lower - 0.1), math.ceil(ce_upper) + 0.5)
                ax.grid(axis="y", color="#DDDDDD", linewidth=0.65, alpha=0.8)
                ax.set_axisbelow(True)
                expected = len(series) * sum(curve["plot_points"] for curve in curves)
                if drawn_points != expected:
                    raise RuntimeError("Rendered point count differs from the summary")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
                       ncol=min(3, len(curves)), frameon=False, fontsize=11)
            for index, curve in enumerate(curves):
                size, points = curve["size"], curve["points"]
                peak = max(points, key=lambda point: point["values"][1][0])
                last = points[-1]
                text = (f"n={size}\nPeak mean: {peak['values'][1][0]:.2f}% at {peak['x']:.1f}M\n"
                        f"Final mean: {last['values'][1][0]:.2f}% at {last['x']:.1f}M")
                axes[1].text(1.04, 0.94 - index * 0.32, text, transform=axes[1].transAxes,
                             ha="left", va="top", fontsize=9, color=COLORS.get(size, "#666666"),
                             linespacing=1.5, clip_on=False)
            counts = {split: sorted({point["evaluation_counts"][split]
                                    for points in groups.values() for point in points
                                    if point["evaluation_counts"][split] is not None})
                      for split in ("train", "val")}
            count_text = "; ".join(f"{split}: {','.join(str(n) for n in values)} trajectories"
                                   for split, values in counts.items() if values)
            fig.text(0.5, 0.035, "Teacher-prefix overlap (expected acceptance); FP32. " + count_text,
                     ha="center", fontsize=9, color="#444444")
            fig.text(0.5, 0.012, "Bars: sample SD across training seeds, not confidence intervals or evaluation-trajectory uncertainty. CE panels share a scale.",
                     ha="center", fontsize=9, color="#444444")
            fig.subplots_adjust(left=0.075, right=0.78, top=0.83, bottom=0.12, wspace=0.29, hspace=0.29)
            output_dir.mkdir(parents=True, exist_ok=True)
            for path in paths:
                fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.18)
        finally:
            plt.close(fig)
    details = dict(input=str(summary_path), outputs=[str(path) for path in paths],
                   partial=partial, training_seeds=seeds, skipped_rows=skipped,
                   completed_runs=sum({point["phase"] for point in points} == set(range(7))
                                      for points in groups.values()),
                   input_points=sum(len(points) for points in groups.values()),
                   error_bar_definition="sample standard deviation across training seeds; null for a single seed",
                   points_per_panel=[len(series) * sum(curve["plot_points"] for curve in curves)
                                     for _, _, series in PANELS],
                   curves=curves)
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path,
                        default=Path(__file__).with_name("curve_summary.json"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        details = plot_curves(args.summary, args.output_dir)
    except RuntimeError as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps(details, allow_nan=False))


if __name__ == "__main__":
    main()
