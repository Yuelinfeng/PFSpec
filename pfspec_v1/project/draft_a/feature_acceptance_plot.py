"""Plot measured fine-tuning quality without interpolated observations."""

import argparse
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    args = ap.parse_args()
    rows = json.loads((args.run_root / "adapt_d8/history.json").read_text())
    x = [r["step"] for r in rows]
    f = [100 * r["feature"] for r in rows]
    p = [100 * r["position"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, values, title, color in zip(axes, [f, p], ["Feature", "Position"], ["#2374ab", "#198754"]):
        ax.plot(x, values, "o-", color=color, linewidth=2, markersize=6)
        for step, value in zip(x, values):
            ax.annotate(f"{value:.2f}%", (step, value), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("Fine-tuning updates")
        ax.set_ylabel("Validation probability overlap (%)")
        ax.set_xticks(x)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].axhline(30, color="#b45436", linestyle="--", label="Requested 30%")
    axes[0].axhline(20.788057, color="#777777", linestyle=":", label="Earlier best small-corpus Draft")
    axes[0].set_ylim(15, max(f) + 10)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].set_ylim(90, 100)
    fig.suptitle("Pretrained 8-layer Draft + full Feature distribution training\n"
                 "50 validation trajectories; fixed Target and Draft temperature=0.9, top-p=0.9", fontsize=11)
    for extension in ("png", "pdf"):
        fig.savefig(args.run_root / f"feature_improvement.{extension}", dpi=180)
    print(json.dumps(dict(measurements=len(rows), files=["feature_improvement.png", "feature_improvement.pdf"])))


if __name__ == "__main__":
    main()
