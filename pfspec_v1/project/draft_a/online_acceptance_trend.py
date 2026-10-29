"""Plot online acceptance trends across prefix lengths."""

import argparse
import json
from pathlib import Path


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.summary.read_text())
    groups = data["online"]["groups"]
    prefixes = sorted({row["prompt_tokens"] for row in groups})
    gammas = sorted({row["gamma"] for row in groups})
    colors = {2: "#1565c0", 4: "#d97706"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, layout="constrained")
    series = [
        ("feature", "Feature acceptance rate (%)", 30, "30% target"),
        ("position", "Position acceptance rate (%)", None, None),
    ]
    for ax, (kind, ylabel, reference, reference_label) in zip(axes, series):
        for gamma in gammas:
            rows = sorted(
                (row for row in groups if row["gamma"] == gamma),
                key=lambda row: row["prompt_tokens"],
            )
            values = [100 * row[kind]["accepted_over_verified"] for row in rows]
            intervals = [
                [100 * value for value in row[kind]["accepted_over_verified_ci95"]]
                for row in rows
            ]
            lower = [value - interval[0] for value, interval in zip(values, intervals)]
            upper = [interval[1] - value for value, interval in zip(values, intervals)]
            color = colors[gamma]
            ax.plot(
                prefixes,
                values,
                "o-",
                color=color,
                linewidth=2.4,
                markersize=7,
                label=f"Gamma={gamma}",
            )
            ax.errorbar(
                prefixes,
                values,
                yerr=[lower, upper],
                fmt="none",
                ecolor=color,
                elinewidth=1.2,
                capsize=4,
                alpha=0.75,
            )
            for x, value in zip(prefixes, values):
                ax.annotate(
                    f"{value:.1f}%",
                    (x, value),
                    xytext=(0, 9 if gamma == 2 else -16),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if gamma == 2 else "top",
                    fontsize=9,
                    color=color,
                )
        if reference is not None:
            ax.axhline(
                reference,
                color="#b91c1c",
                linestyle="--",
                linewidth=1.5,
                label=reference_label,
            )
            ax.set_ylim(25, 65)
        else:
            ax.set_ylim(90, 100)
        ax.set_title("Feature" if kind == "feature" else "Position")
        ax.set_xlabel("Prefix length (tokens)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(prefixes)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="best", frameon=False)

    fig.suptitle(
        "Online acceptance trend on 12 independent scenes\n"
        "95% scene-bootstrap intervals; temperature=0.9, top-p=0.9",
        fontsize=12,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir / f"online_acceptance_trend.{extension}", dpi=200)
    print(json.dumps({"files": ["online_acceptance_trend.png", "online_acceptance_trend.pdf"]}))


if __name__ == "__main__":
    main()
