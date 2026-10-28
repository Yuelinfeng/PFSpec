"""Plot measured validation and fixed training-subset learning curves."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    history = json.loads((args.root/"history.json").read_text())
    selected = json.loads((args.root/"selected.json").read_text())
    steps = [r["step"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    metrics = [("pooled_overlap", "Full validation set", "#157f99"),
               ("training_probe_overlap", "Fixed training subset", "#d78132")]
    for ax, (field, title, color) in zip(axes, metrics):
        values = [100*r[field] for r in history]
        ax.plot(steps, values, "o-", color=color, linewidth=2, markersize=5)
        ax.set(title=title, xlabel="Cumulative optimizer steps", ylabel="Feature policy overlap (%)")
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        for step, value in zip(steps, values):
            ax.annotate(f"{value:.3f}", (step, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
        ax.margins(y=.18)
    chosen = next(r for r in history if r["step"] == selected["step"])
    axes[0].scatter([chosen["step"]], [100*chosen["pooled_overlap"]], marker="*", s=160, color="#903039", zorder=5,
                    label=f"Selected step {selected['step']}")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Direct-Target Feature Draft: additional training", fontsize=13)
    fig.savefig(args.root/"training_curve.png", dpi=180)
    fig.savefig(args.root/"training_curve.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
