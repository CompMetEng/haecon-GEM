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
    models = ["egg", "L1", "L2", "L3", "L4f", "L4m", "Am", "Af"]
    # Retain the recorded subsystem display selection.
    display_categories = """
Acyl-CoA hydrolysis
Acylglycerides metabolism
Alanine, aspartate and glutamate metabolism
Alkaloids biosynthesis
Amino sugar and nucleotide sugar metabolism
Aminoacyl-tRNA biosynthesis
Androgen metabolism
Arachidonic acid metabolism
Arginine and proline metabolism
Artificial reactions
Ascaroside transport and biosynthesis
Ascorbate and aldarate metabolism
Beta oxidation of branched-chain fatty acids (mitochondrial)
Beta oxidation of di-unsaturated fatty acids (n-6) (mitochondrial)
Beta oxidation of di-unsaturated fatty acids (n-6) (peroxisomal)
Beta oxidation of even-chain fatty acids (mitochondrial)
Beta oxidation of even-chain fatty acids (peroxisomal)
Beta oxidation of odd-chain fatty acids (mitochondrial)
Beta oxidation of odd-chain fatty acids (peroxisomal)
Beta oxidation of phytanic acid (peroxisomal)
Beta oxidation of poly-unsaturated fatty acids (mitochondrial)
Beta oxidation of unsaturated fatty acids (n-7) (mitochondrial)
Beta oxidation of unsaturated fatty acids (n-7) (peroxisomal)
Beta oxidation of unsaturated fatty acids (n-9) (mitochondrial)
Beta oxidation of unsaturated fatty acids (n-9) (peroxisomal)
Beta-alanine metabolism
Bile acid biosynthesis
Bile acid recycling
Biopterin metabolism
Biotin metabolism
Blood group biosynthesis
Butanoate metabolism
C5-branched dibasic acid metabolism
Carnitine shuttle (cytosolic)
Carnitine shuttle (endoplasmic reticular)
Carnitine shuttle (mitochondrial)
Carnitine shuttle (peroxisomal)
Cholesterol biosynthesis 1 (Bloch pathway)
Cholesterol biosynthesis 2
Cholesterol biosynthesis 3 (Kandustch-Russell pathway)
Cholesterol metabolism
Chondroitin / heparan sulfate biosynthesis
Chondroitin sulfate degradation
Cysteine and methionine metabolism
Dietary fiber binding
Drug metabolism
Eicosanoid metabolism
Endoplasmic reticular transport
Estrogen metabolism
Ether lipid metabolism
Exchange/demand reactions
Extracellular transport
Fatty acid activation (cytosolic)
Fatty acid activation (endoplasmic reticular)
Fatty acid biosynthesis
Fatty acid biosynthesis (even-chain)
Fatty acid biosynthesis (odd-chain)
Fatty acid biosynthesis (unsaturated)
Fatty acid degradation
Fatty acid desaturation (even-chain)
Fatty acid desaturation (odd-chain)
Fatty acid elongation (even-chain)
Fatty acid elongation (odd-chain)
Fatty acid metabolism
Fatty acid oxidation
Folate metabolism
Formation and hydrolysis of cholesterol esters
Fructose and mannose metabolism
Galactose metabolism
Glucocorticoid biosynthesis
Glutathione metabolism
Glycerolipid metabolism
Glycerophospholipid metabolism
Glycine, serine and threonine metabolism
Glycolysis / Gluconeogenesis
Glycosphingolipid biosynthesis-ganglio series
Glycosphingolipid biosynthesis-globo series
Glycosphingolipid biosynthesis-lacto and neolacto series
Glycosphingolipid metabolism
Glycosylphosphatidylinositol (GPI)-anchor biosynthesis
Glyoxylate and dicarboxylate metabolism
Golgi transport
Heme degradation
Heparan sulfate degradation
Hippurate metabolism
Histidine metabolism
Inositol phosphate metabolism
Insect hormone biosynthesis
Intracellular transport
Isolated
Keratan sulfate biosynthesis
Keratan sulfate degradation
Leukotriene metabolism
Linoleate metabolism
Lipoic acid metabolism
Lysine metabolism
Lysosomal transport
Metabolism of other amino acids
Miscellaneous
Mitochondrial transport
N-glycan metabolism
Nicotinate and nicotinamide metabolism
Nuclear transport
Nucleotide metabolism
O-glycan metabolism
Octane oxidation
Omega-3 fatty acid metabolism
Omega-6 fatty acid metabolism
Oxidative phosphorylation
Pantothenate and CoA metabolism
Pentose and glucuronate interconversions
Pentose phosphate pathway
Peptide metabolism
Peroxisomal transport
Phenylalanine metabolism
Phenylalanine, tyrosine and tryptophan biosynthesis
Phosphatidylinositol phosphate metabolism
Pool reactions
Porphyrin metabolism
Propanoate metabolism
Prostaglandin biosynthesis
Protein assembly
Protein degradation
Protein modification
Purine metabolism
Pyrimidine metabolism
Pyruvate metabolism
Riboflavin metabolism
ROS detoxification
Serotonin and melatonin biosynthesis
Sphingolipid metabolism
Starch and sucrose metabolism
Steroid metabolism
Sulfur metabolism
Terpenoid backbone biosynthesis
Thiamine metabolism
Toluene degradation
Triacylglycerol synthesis
Tricarboxylic acid cycle and glyoxylate/dicarboxylate metabolism
Tryptophan metabolism
Tyrosine metabolism
Ubiquinone synthesis
Unassigned
Urea cycle
Valine, leucine, and isoleucine metabolism
Vitamin A metabolism
Vitamin B12 metabolism
Vitamin B2 metabolism
Vitamin B6 metabolism
Vitamin C metabolism
Vitamin D metabolism
Vitamin E metabolism
Xenobiotics metabolism
""".strip().splitlines()
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
                range(len(models)),
                ["Egg", "L1", "L2", "L3", "L4f", "L4m", "Am", "Af"],
                rotation=45,
                ha="left",
            )
            ax.xaxis.tick_top()
            ax.tick_params(axis="x", length=0, pad=8)
            ax.set_yticks(range(len(chunk)), chunk, fontsize=8.5)
            ax.tick_params(axis="y", length=0, pad=8)
            ax.set_ylim(len(chunk) - 0.5, -0.5)
            ax.set_xlim(-0.6, len(models) - 0.4)
            ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
            ax.grid(axis="x", which="minor", color="#e9e9e9", linewidth=0.5)
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
                "fold > 1 and global BH q < 0.05 (1,256 tests).\nDot size: "
                "−log₁₀(q), capped at 10. Blue denotes lower representation; "
                "depletion significance was not tested.\nUniform reaction-subset null; reaction "
                "composition, not flux activity.",
                fontsize=9,
                va="bottom",
            )
            fig.savefig(out / f"all_subsystem_enrichment_panel{page}.png", dpi=170)
            fig.savefig(out / f"all_subsystem_enrichment_panel{page}.svg")
            plt.close(fig)


if __name__ == "__main__":
    main()
