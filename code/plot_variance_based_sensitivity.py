"""Plot six highest-ST uptake parameters per independent-original stage (Figure S1)."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-table", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    table = pd.read_csv(args.input_table, sep="\t")
    selected = {}
    for stage in ["egg", "L1", "L2", "L3"]:
        rows = table[(table.stage == stage) & table.independent_original].copy()
        rows["rank"] = rows[["ST_original", "ST_fixed"]].max(axis=1)
        selected[stage] = rows.sort_values("rank", ascending=False).head(6)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)
    for axis, (stage, rows) in zip(axes.flat, selected.items()):
        positions = np.arange(len(rows))
        for position, before, after in zip(positions, rows.ST_original, rows.ST_fixed):
            axis.plot(
                [before, after],
                [position - 0.11, position + 0.11],
                color="0.65",
                linewidth=1,
            )
        for version, offset, color, marker, label in [
            ("original", -0.11, "#245A81", "o", "Before unblocking"),
            ("fixed", 0.11, "#D16A27", "s", "After unblocking"),
        ]:
            axis.errorbar(
                rows["ST_" + version],
                positions + offset,
                xerr=rows["ST_" + version + "_CI95_halfwidth"],
                fmt=marker,
                color=color,
                capsize=3,
                label=label,
            )
        labels = [
            name.removeprefix("Exchange of ") + "\n" + reaction_id
            for name, reaction_id in zip(rows["name"], rows.reaction_id)
        ]
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_title("Egg" if stage == "egg" else stage, loc="left")
        axis.axvline(0, color="0.6", linewidth=0.8)
        axis.grid(axis="x", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("Total sensitivity index (ST; dimensionless)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
    )
    fig.suptitle("Variance-based sensitivity of biomass production", fontsize=20)
    fig.text(
        0.5,
        0.035,
        "Six parameters ranked by larger original/final ST. "
        "Error bars: 95% bootstrap normal-approximation intervals.\n"
        "Independent uptake capacities: 0–10 model flux units, "
        "matched within each model pair. ST includes interactions.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(
        left=0.17, right=0.98, top=0.85, bottom=0.15, wspace=0.8, hspace=0.38
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".svg"]:
        fig.savefig(args.output_prefix.with_suffix(suffix), dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
