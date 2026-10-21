from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from contextlib import nullcontext

RESULT_DIR = Path("outputs")


def main():
    base = RESULT_DIR
    out = base / "whole_model_enrichment_results"
    d = pd.read_csv(out / "subsystem_enrichment.tsv", sep="\t")
    models = ["egg", "L1", "L2", "L3", "L4f", "L4m", "Am", "Af", "L4union"]
    display_categories = pd.read_csv(
        Path(__file__).with_name("plot_subsystem_selection.tsv"), sep="\t"
    )["category"]
    d = d[d.category.isin(display_categories)].copy()
    categories = sorted(d.category.unique(), key=str.casefold)
    assert len(d) == len(categories) * len(models)
    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    with nullcontext():
        for page, chunk in enumerate(np.array_split(categories, 3), 1):
            fig, ax = plt.subplots(figsize=(12, 18))
            fig.subplots_adjust(left=0.48, right=0.97, top=0.91, bottom=0.14)
            ax.set_facecolor("#fafafa")
            for y, cat in enumerate(chunk):
                for x, model in enumerate(models):
                    r = d[(d.category == cat) & (d.model == model)].iloc[0]
                    if r.observed_k == 0:
                        ax.scatter(
                            x, y, marker="x", s=22, color="#777777", linewidths=0.8
                        )
                    else:
                        strength = min(-np.log10(max(float(r.q_BH_global), 1e-300)), 10)
                        ax.scatter(
                            x,
                            y,
                            s=22 + strength * 13,
                            c=[np.clip(np.log2(r.fold_enrichment), -3, 3)],
                            cmap="RdBu_r",
                            norm=norm,
                            edgecolors="black" if r.enriched else "#bbbbbb",
                            linewidths=1.1 if r.enriched else 0.35,
                        )
            ax.set_xticks(
                range(9),
                ["Egg", "L1", "L2", "L3", "L4f", "L4m", "Am", "Af", "L4 union¹"],
                rotation=45,
                ha="left",
            )
            ax.xaxis.tick_top()
            ax.tick_params(axis="x", length=0, pad=8)
            ax.set_yticks(range(len(chunk)), chunk, fontsize=8.5)
            ax.tick_params(axis="y", length=0, pad=8)
            ax.set_ylim(len(chunk) - 0.5, -0.5)
            ax.set_xlim(-0.6, 8.6)
            ax.set_xticks(np.arange(-0.5, 9, 0.999999), minor=True)
            ax.grid(axis="x", which="minor", color="#e9e9e9", linewidth=0.5)
            ax.axvline(7.5, color="#777777", linestyle="--", lw=0.8)
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.suptitle(
                "Subsystem representation relative to full HaeconGEM\n"
                f"{len(categories)} subsystems · panel {page}/3",
                fontsize=15,
                y=0.98,
            )
            sm = plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
            cb = fig.colorbar(
                sm,
                ax=ax,
                orientation="horizontal",
                fraction=0.022,
                pad=0.045,
                aspect=30,
            )
            cb.set_label("log₂(observed / expected); colour capped at ±3", fontsize=10)
            cb.set_ticks([-3, -2, -1, 0, 1, 2, 3])
            cb.set_ticklabels(["≤−3", "−2", "−1", "0", "1", "2", "≥3"])
            handles = [
                Line2D(
                    [],
                    [],
                    marker="o",
                    ls="",
                    markersize=np.sqrt(22 + v * 13),
                    markerfacecolor="#cccccc",
                    markeredgecolor="#999999",
                    label=f"q = {10 ** (-v):g}" if v < 10 else "q ≤ 10⁻¹⁰",
                )
                for v in [0, 2, 5, 10]
            ]
            handles += [
                Line2D(
                    [],
                    [],
                    marker="o",
                    ls="",
                    markersize=8,
                    markerfacecolor="#cccccc",
                    markeredgecolor="black",
                    label="Significant enrichment",
                ),
                Line2D(
                    [], [], marker="x", ls="", color="#777777", label="Absent (k = 0)"
                ),
            ]
            fig.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.57, 0.055),
                ncol=3,
                fontsize=9,
                frameon=False,
            )
            fig.text(
                0.03,
                0.018,
                "Expected = model size × subsystem size / 11,873. Outline: "
                "fold > 1 and global BH q < 0.05 (1,413 tests).\nDot size: "
                "−log₁₀(q), capped at 10. Blue denotes lower representation; "
                "depletion significance was not tested.\n¹Synthetic L4f ∪ L4m "
                "reference. Uniform reaction-subset null; reaction "
                "composition, not flux activity.",
                fontsize=9,
                va="bottom",
            )
            fig.savefig(out / f"all_subsystem_enrichment_panel{page}.png", dpi=170)
            fig.savefig(out / f"all_subsystem_enrichment_panel{page}.svg")
            plt.close(fig)


if __name__ == "__main__":
    main()
