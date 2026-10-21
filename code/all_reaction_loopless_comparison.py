(
    "All-reaction ordinary/explicit-loopless FVA and comparison "
    "with existing paper samples.\n\nNo new sampling. Individual "
    "loopless intervals provide necessary, not "
    "sufficient,\nconditions for a complete flux vector to be "
    "loopless.\n"
)

from pathlib import Path
import numpy as np, pandas as pd, cobra
from cobra.flux_analysis.loopless import add_loopless
from cobra.flux_analysis import flux_variability_analysis

INPUT_DIR = Path("inputs")
OUTPUT_DIR = Path("outputs/all_reaction_loopless_results")
PAIR_DIR = Path("outputs/CHRR_medians_loopless_ranges")
SOLVER = "gurobi"
PAIRS = [
    ("egg", "L1"),
    ("L1", "L2"),
    ("L2", "L3"),
    ("L3", "L4union"),
    ("L4f", "Af"),
    ("L4m", "Am"),
    ("Af", "Am"),
]


def pair_tables(results):
    """Retain contracted entries, counterparts and explicit absence in seven pairs."""
    all_long, all_wide, all_pairs = ([], [], [])
    tolerance = 0.0001
    PAIR_DIR.mkdir(parents=True, exist_ok=True)
    for a, b in PAIRS:
        left = results[results.model.eq(a)].set_index("reaction_id")
        right = results[results.model.eq(b)].set_index("reaction_id")
        ids = sorted(
            set(left.index[left.range_contracted])
            | set(right.index[right.range_contracted])
        )
        if not ids:
            continue
        pair = (
            left.reindex(ids).add_suffix("_A").join(right.reindex(ids).add_suffix("_B"))
        )
        pair.index.name = "reaction_id"
        pair["comparison"] = a + "_vs_" + b
        pair["model_A"], pair["model_B"] = (a, b)
        wide = pair[["comparison", "model_A", "model_B"]].copy()
        for side, stage, frame in [("A", a, left), ("B", b, right)]:
            present = pair.index.isin(frame.index)
            sampled = pair["sampled_" + side].eq(True)
            complete = pair["fva_status_" + side].eq("complete")
            low, high = (
                pair["loopless_minimum_" + side],
                pair["loopless_maximum_" + side],
            )
            median = pair["sample_median_" + side]
            status = np.select(
                [
                    ~present,
                    ~sampled,
                    ~complete,
                    (median >= low - tolerance) & (median <= high + tolerance),
                ],
                ["reaction_absent", "no_CHRR_samples", "unresolved", "inside"],
                default="outside",
            )
            detail = pd.DataFrame(
                {
                    "comparison": a + "_vs_" + b,
                    "model": stage,
                    "reaction_present": present,
                    "CHRR_minimum": pair["sample_minimum_" + side].where(sampled),
                    "CHRR_median": median.where(sampled),
                    "CHRR_maximum": pair["sample_maximum_" + side].where(sampled),
                    "loopless_FVA_minimum": low,
                    "loopless_FVA_maximum": high,
                    "CHRR_median_in_loopless_range": status,
                    "range_contracted": pair["range_contracted_" + side],
                    "CHRR_median_inside_ordinary_FVA": pair[
                        "median_outside_ordinary_" + side
                    ]
                    .eq(False)
                    .where(sampled & complete),
                    "reaction_name": pair["name_" + side],
                    "subsystem": pair["subsystem_" + side],
                },
                index=pair.index,
            )
            all_long.append(detail.reset_index())
            for column in detail.columns:
                if column not in ["comparison", "model", "reaction_name", "subsystem"]:
                    wide[column + "_" + side] = detail[column]
        complete = pair.fva_status_A.eq("complete") & pair.fva_status_B.eq("complete")
        pair["minimum_possible_signed_difference_B_minus_A"] = (
            pair.loopless_minimum_B - pair.loopless_maximum_A
        ).where(complete)
        pair["maximum_possible_signed_difference_B_minus_A"] = (
            pair.loopless_maximum_B - pair.loopless_minimum_A
        ).where(complete)
        for side in ["A", "B"]:
            low, high = (
                pair["loopless_minimum_" + side],
                pair["loopless_maximum_" + side],
            )
            pair["magnitude_min_" + side] = np.where(
                (low <= 0) & (high >= 0), 0, np.minimum(abs(low), abs(high))
            )
            pair["magnitude_max_" + side] = np.maximum(abs(low), abs(high))
        pair["minimum_possible_magnitude_difference_B_minus_A"] = (
            pair.magnitude_min_B - pair.magnitude_max_A
        ).where(complete)
        pair["maximum_possible_magnitude_difference_B_minus_A"] = (
            pair.magnitude_max_B - pair.magnitude_min_A
        ).where(complete)
        pair["CHRR_median_difference_B_minus_A"] = (
            pair.sample_median_B - pair.sample_median_A
        )
        pair["CHRR_magnitude_difference_B_minus_A"] = abs(pair.sample_median_B) - abs(
            pair.sample_median_A
        )
        pair["loopless_intervals_overlap"] = (
            np.maximum(pair.loopless_minimum_A, pair.loopless_minimum_B)
            <= np.minimum(pair.loopless_maximum_A, pair.loopless_maximum_B) + tolerance
        ).where(complete)
        pair["comparison_status"] = np.where(
            complete, "comparable", "unresolved_or_absent"
        )
        wide.to_csv(PAIR_DIR / f"{a}_vs_{b}_CHRR_and_loopless_ranges.tsv", sep="\t")
        all_wide.append(wide.reset_index())
        all_pairs.append(pair.reset_index())
    if all_wide:
        pd.concat(all_wide, ignore_index=True).to_csv(
            PAIR_DIR / "all_relevant_pairs_CHRR_and_loopless_ranges.tsv",
            sep="\t",
            index=False,
        )
        pd.concat(all_long, ignore_index=True).to_csv(
            PAIR_DIR / "all_relevant_pairs_CHRR_and_loopless_ranges_long.tsv",
            sep="\t",
            index=False,
        )
        pd.concat(all_pairs, ignore_index=True).to_csv(
            PAIR_DIR / "all_affected_reactions_relevant_pairs.tsv",
            sep="\t",
            index=False,
        )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS = {
        s: INPUT_DIR / "CHRR_sampling/models" / f"{s}_c.xml"
        for s in ["egg", "L1", "L2", "L3", "L4f", "L4m", "Am", "Af"]
    }
    MODELS["L4union"] = INPUT_DIR / "CHRR_sampling/models/L4mf_union_c.xml"
    TOL = 0.0001
    FLOOR_PATH = (
        INPUT_DIR / "editorial_reviewer/coment3/biomass_optima_and_10pct_bounds.tsv"
    )
    floors = (
        pd.read_csv(FLOOR_PATH, sep="\t")
        .set_index("model")
        .recorded_sampling_lower_bound.to_dict()
    )
    all_rows = []
    summary = []
    for stage, path in MODELS.items():
        if stage == "HaeconGEM1":
            continue
        sample_path = (
            INPUT_DIR / "CHRR_sampling/pca" / f"{stage}_c_samples.csv"
            if stage != "L4union"
            else INPUT_DIR / "CHRR_sampling/union_L4fm/L4mf_union_c_samples.csv"
        )
        m = cobra.io.read_sbml_model(str(path))
        m.solver = SOLVER
        m.solver.configuration.timeout = 30
        m.solver.configuration.tolerances.feasibility = 1e-09
        m.solver.configuration.tolerances.optimality = 1e-09
        floor = float(floors["L4mf_union" if stage == "L4union" else stage])
        growth = m.reactions.get_by_id("MAR04413")
        growth.lower_bound = max(growth.lower_bound, floor)
        m.objective = 0
        samples = pd.read_csv(sample_path)
        assert (
            not samples.columns.duplicated().any()
            and np.isfinite(samples.to_numpy()).all()
        )
        rids = [r.id for r in m.reactions]
        assert not set(samples.columns) - set(rids), (
            stage,
            sorted(set(samples.columns) - set(rids)),
        )
        values = {}
        solver_status = {}
        for family in ["ordinary", "loopless"]:
            if family == "loopless":
                add_loopless(m, method="fastSNP")
            try:
                ranges = flux_variability_analysis(
                    m, reaction_list=rids, fraction_of_optimum=0, processes=1
                )
            except cobra.exceptions.OptimizationError:
                ranges = pd.DataFrame(index=rids, columns=["minimum", "maximum"])
            family_rows = []
            for rid, row in ranges.iterrows():
                for direction, column in [("min", "minimum"), ("max", "maximum")]:
                    value = row[column]
                    status = "optimal" if pd.notna(value) else "unresolved"
                    values[family, rid, direction] = value
                    solver_status[family, rid, direction] = status
                    family_rows.append(
                        {
                            "model": stage,
                            "reaction_id": rid,
                            "direction": direction,
                            "value": value,
                            "status": status,
                        }
                    )
            pd.DataFrame(family_rows).to_csv(
                OUTPUT_DIR / f"{stage}_{family}_endpoints.tsv", sep="\t", index=False
            )
        stage_rows = []
        sample_order = list(samples.columns)
        byid = {}
        for r in m.reactions:
            rid = r.id
            omin = values["ordinary", rid, "min"]
            omax = values["ordinary", rid, "max"]
            lmin = values["loopless", rid, "min"]
            lmax = values["loopless", rid, "max"]
            complete = all(
                (
                    solver_status[f, rid, d] == "optimal"
                    for f in ["ordinary", "loopless"]
                    for d in ["min", "max"]
                )
            )
            if complete:
                assert lmin >= omin - TOL and lmax <= omax + TOL, (
                    stage,
                    rid,
                    omin,
                    omax,
                    lmin,
                    lmax,
                )
            ow = omax - omin
            lw = lmax - lmin
            row = dict(
                model=stage,
                reaction_id=rid,
                name=r.name,
                subsystem=r.subsystem,
                boundary=r.boundary,
                biomass_floor=floor,
                ordinary_minimum=omin,
                ordinary_maximum=omax,
                loopless_minimum=lmin,
                loopless_maximum=lmax,
                ordinary_width=ow,
                loopless_width=lw,
                width_reduction_percent=(
                    100 * (ow - lw) / ow if complete and ow > TOL else np.nan
                ),
                range_contracted=bool(
                    complete and (lmin > omin + TOL or lmax < omax - TOL)
                ),
                fva_status="complete" if complete else "unresolved",
                sampled=rid in samples.columns,
            )
            for f in ["ordinary", "loopless"]:
                for direction in ["min", "max"]:
                    row[f + "_" + direction + "_status"] = solver_status[
                        f, rid, direction
                    ]
            if rid in samples:
                v = samples[rid].to_numpy()
                median = float(np.median(v))
                outsideO = (v < omin - TOL) | (v > omax + TOL) if complete else None
                outsideL = (v < lmin - TOL) | (v > lmax + TOL) if complete else None
                row.update(
                    samples_n=len(v),
                    sample_minimum=float(v.min()),
                    sample_median=median,
                    sample_maximum=float(v.max()),
                    ordinary_samples_outside_n=(
                        int(outsideO.sum()) if complete else np.nan
                    ),
                    loopless_samples_outside_n=(
                        int(outsideL.sum()) if complete else np.nan
                    ),
                    loopless_samples_outside_percent=(
                        100 * float(outsideL.mean()) if complete else np.nan
                    ),
                    ordinary_compatible_but_loopless_outside_n=(
                        int((outsideL & ~outsideO).sum()) if complete else np.nan
                    ),
                    median_outside_ordinary=(
                        bool(median < omin - TOL or median > omax + TOL)
                        if complete
                        else None
                    ),
                    median_outside_loopless=(
                        bool(median < lmin - TOL or median > lmax + TOL)
                        if complete
                        else None
                    ),
                    median_ordinary_compatible_but_loopless_outside=(
                        bool(
                            (median < lmin - TOL or median > lmax + TOL)
                            and (not (median < omin - TOL or median > omax + TOL))
                        )
                        if complete
                        else None
                    ),
                )
            stage_rows.append(row)
            byid[rid] = row
        g = pd.DataFrame(stage_rows)
        g.to_csv(OUTPUT_DIR / f"{stage}_reaction_comparison.tsv", sep="\t", index=False)
        all_rows.extend(stage_rows)
        comparable = [
            rid for rid in sample_order if byid[rid]["fva_status"] == "complete"
        ]
        X = samples[comparable].to_numpy()
        ol = np.array([byid[r]["ordinary_minimum"] for r in comparable])
        ou = np.array([byid[r]["ordinary_maximum"] for r in comparable])
        ll = np.array([byid[r]["loopless_minimum"] for r in comparable])
        lu = np.array([byid[r]["loopless_maximum"] for r in comparable])
        outsideO = (X < ol - TOL) | (X > ou + TOL)
        outsideL = (X < ll - TOL) | (X > lu + TOL)
        sampled = g[g.sampled & g.fva_status.eq("complete")]
        summary.append(
            dict(
                model=stage,
                reactions=len(g),
                complete_fva_reactions=int(g.fva_status.eq("complete").sum()),
                unresolved_fva_reactions=int(g.fva_status.eq("unresolved").sum()),
                range_contracted_reactions=int(g.range_contracted.sum()),
                range_contracted_percent=100 * float(g.range_contracted.mean()),
                sampled_reactions=len(samples.columns),
                sampled_reactions_with_complete_fva=len(sampled),
                samples_per_reaction=len(samples),
                reactions_with_any_sample_outside_ordinary=int(
                    sampled.ordinary_samples_outside_n.gt(0).sum()
                ),
                reactions_with_any_sample_outside_loopless=int(
                    sampled.loopless_samples_outside_n.gt(0).sum()
                ),
                medians_outside_ordinary=int(
                    sampled.median_outside_ordinary.eq(True).sum()
                ),
                medians_outside_loopless=int(
                    sampled.median_outside_loopless.eq(True).sum()
                ),
                medians_outside_loopless_but_inside_ordinary=int(
                    sampled.median_ordinary_compatible_but_loopless_outside.eq(
                        True
                    ).sum()
                ),
                sample_values_outside_ordinary=int(outsideO.sum()),
                sample_values_outside_loopless=int(outsideL.sum()),
                sample_values_outside_loopless_percent=100 * float(outsideL.mean()),
                sample_vectors_with_any_marginal_loopless_violation=int(
                    outsideL.any(axis=1).sum()
                ),
                sample_vectors_with_ordinary_compatible_loopless_violations=int(
                    (outsideL & ~outsideO).any(axis=1).sum()
                ),
            )
        )
    results = pd.DataFrame(all_rows)
    results.to_csv(OUTPUT_DIR / "all_reaction_comparison.tsv", sep="\t", index=False)
    pd.DataFrame(summary).to_csv(
        OUTPUT_DIR / "model_summary.tsv", sep="\t", index=False
    )
    pair_tables(results)


if __name__ == "__main__":
    main()
