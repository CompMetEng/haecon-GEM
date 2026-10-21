"""Adult blood-condition pFBA and loopless FVA, with common carbon budgets.

Set the paths below before running. Fluxes are in model flux units; carbon
uptake is in model carbon units. Input SBML files are not overwritten.
"""

from pathlib import Path

import cobra
import numpy as np
import pandas as pd
from cobra.flux_analysis import flux_variability_analysis, pfba
from cobra.flux_analysis.loopless import add_loopless

MODELS = {
    "Af": Path("inputs/models/Af_c_RQ_FDR_min03.xml"),
    "Am": Path("inputs/models/Am_c_RQ_reduced_repair.xml"),
}
CONSTRAINTS = {
    "Af": Path("inputs/Af_blood_constraints.csv"),
    "Am": Path("inputs/Am_blood_constraints.csv"),
}
BASELINE_SUMMARY = Path("inputs/adult_baseline_summary.tsv")
REACTION_TARGETS = Path("inputs/adult_reaction_targets.tsv")
OUTPUT = Path("results/adult_comparison")
BIOMASS = "MAR04413"
FRACTIONS = {
    "uncapped": None,
    "25pct": 0.25,
    "50pct": 0.5,
    "100pct": 1.0,
    "200pct": 2.0,
}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(BASELINE_SUMMARY, sep="\t").set_index("model")
    reference = float(baseline.loc[["Af", "Am"], "total_carbon_uptake"].min())
    targets = pd.read_csv(REACTION_TARGETS, sep="\t").reaction_id.tolist()
    summaries, fluxes, uptake_rows, intervals = [], [], [], []

    for stage, path in MODELS.items():
        model = cobra.io.read_sbml_model(str(path))
        model.solver = "gurobi"
        model.solver.configuration.timeout = 30
        model.solver.configuration.tolerances.feasibility = 1e-9
        model.solver.configuration.tolerances.optimality = 1e-9
        for row in pd.read_csv(CONSTRAINTS[stage]).itertuples():
            reaction = model.reactions.get_by_id(row.exchange_reaction_id)
            if row.action == "blood_carbon_source_kept_6_over_nC":
                reaction.lower_bound = -6.0 / row.n_carbon
            elif row.action == "non_blood_carbon_source_closed":
                reaction.lower_bound = 0
            elif row.action != "growth_essential_carbon_source_kept":
                raise ValueError(f"Unknown exchange action: {row.action}")
        for reaction_id in ["MAR06918", "MAR04652"]:
            assert model.reactions.get_by_id(reaction_id).bounds == (0, 0)
        model.objective = BIOMASS
        model.objective_direction = "max"

        exchanges = []
        for reaction in model.exchanges:
            metabolite, coefficient = next(iter(reaction.metabolites.items()))
            if not metabolite.formula:
                raise ValueError(f"Missing exchange formula: {reaction.id}")
            carbon = float(metabolite.elements.get("C", 0))
            exchanges.append((reaction, metabolite.id, coefficient, carbon))

        for scenario, fraction in FRACTIONS.items():
            cap = None if fraction is None else reference * fraction
            with model:
                if cap is not None:
                    carbon_uptake = sum(
                        carbon
                        * abs(coefficient)
                        * (
                            reaction.reverse_variable
                            if coefficient < 0
                            else reaction.forward_variable
                        )
                        for reaction, _, coefficient, carbon in exchanges
                        if carbon > 0
                    )
                    model.add_cons_vars(
                        model.problem.Constraint(
                            carbon_uptake, lb=0, ub=cap, name="total_carbon_uptake"
                        )
                    )
                try:
                    solution = pfba(model, fraction_of_optimum=1.0)
                except cobra.exceptions.OptimizationError:
                    summaries.append(
                        {
                            "model": stage,
                            "scenario": scenario,
                            "status": model.solver.status,
                        }
                    )
                    continue
                biomass = float(solution.fluxes[BIOMASS])
                uptake = []
                for reaction, metabolite, coefficient, carbon in exchanges:
                    rate = max(0.0, coefficient * float(solution.fluxes[reaction.id]))
                    uptake.append(
                        {
                            "model": stage,
                            "scenario": scenario,
                            "reaction_id": reaction.id,
                            "metabolite_id": metabolite,
                            "uptake": rate,
                            "carbon_number": carbon,
                            "carbon_uptake": rate * carbon,
                        }
                    )
                uptake = pd.DataFrame(uptake)
                total_carbon = float(uptake.carbon_uptake.sum())
                paper_carbon = float(
                    uptake.loc[
                        uptake.carbon_number.gt(0) & uptake.carbon_number.le(50),
                        "carbon_uptake",
                    ].sum()
                )
                if min(total_carbon, paper_carbon) <= 0:
                    raise ValueError("Carbon normalisation requires positive uptake")
                summaries.append(
                    {
                        "model": stage,
                        "scenario": scenario,
                        "status": solution.status,
                        "reference_carbon_uptake": reference,
                        "reference_fraction": fraction,
                        "carbon_budget": cap,
                        "biomass_optimum": biomass,
                        "actual_total_carbon": total_carbon,
                        "paper_carbon_denominator": paper_carbon,
                        "FDR_flux": float(solution.fluxes["FDR"]),
                        "RHQR_flux": float(solution.fluxes["RHQR"]),
                    }
                )
                uptake_rows.extend(uptake.to_dict("records"))
                for reaction in model.reactions:
                    value = float(solution.fluxes[reaction.id])
                    fluxes.append(
                        {
                            "model": stage,
                            "scenario": scenario,
                            "reaction_id": reaction.id,
                            "flux": value,
                            "flux_per_total_carbon": value / total_carbon,
                            "flux_per_paper_carbon": value / paper_carbon,
                        }
                    )

                biomass_reaction = model.reactions.get_by_id(BIOMASS)
                biomass_reaction.lower_bound = max(
                    biomass_reaction.lower_bound, 0.1 * biomass
                )
                model.objective = 0
                add_loopless(model, method="fastSNP")
                reaction_ids = (
                    model.reactions.list_attr("id") if cap is None else targets
                )
                present = [rid for rid in reaction_ids if rid in model.reactions]
                try:
                    ranges = flux_variability_analysis(
                        model, reaction_list=present, fraction_of_optimum=0, processes=1
                    )
                    ranges["status"] = np.where(
                        ranges[["minimum", "maximum"]].notna().all(axis=1),
                        "complete",
                        "unresolved",
                    )
                except cobra.exceptions.OptimizationError:
                    ranges = pd.DataFrame(index=present, columns=["minimum", "maximum"])
                    ranges["status"] = "unresolved"
                for reaction_id in reaction_ids:
                    row = {
                        "model": stage,
                        "scenario": scenario,
                        "reaction_id": reaction_id,
                        "biomass_floor": 0.1 * biomass,
                    }
                    row.update(
                        ranges.loc[reaction_id].to_dict()
                        if reaction_id in ranges.index
                        else {"status": "absent"}
                    )
                    intervals.append(row)

            for filename, rows in [
                ("summary.tsv", summaries),
                ("pfba.tsv", fluxes),
                ("carbon_uptake.tsv", uptake_rows),
                ("loopless_fva.tsv", intervals),
            ]:
                pd.DataFrame(rows).to_csv(OUTPUT / filename, sep="\t", index=False)

    comparison = pd.DataFrame(fluxes).pivot(
        index=["scenario", "reaction_id"],
        columns="model",
        values=["flux", "flux_per_total_carbon"],
    )
    comparison.columns = [f"{measure}_{stage}" for measure, stage in comparison.columns]
    difference = (
        comparison["flux_per_total_carbon_Af"].abs()
        - comparison["flux_per_total_carbon_Am"].abs()
    )
    present = (
        comparison[["flux_per_total_carbon_Af", "flux_per_total_carbon_Am"]]
        .notna()
        .all(axis=1)
    )
    inactive = comparison[["flux_Af", "flux_Am"]].abs().le(1e-4).all(axis=1)
    comparison["larger_absolute_normalised_flux"] = np.select(
        [~present, inactive, difference > 1e-8, difference < -1e-8],
        ["absent from one model", "both inactive", "Af", "Am"],
        default="equal",
    )
    comparison.to_csv(OUTPUT / "paired_pfba.tsv", sep="\t")


if __name__ == "__main__":
    main()
