"""Extended mapped energy-carrier assays, with separate proton-path diagnostics.

All mapped carrier drains are present together; proton paths are tested one at
 a time with the carrier drains disabled. These are the final extended assays,
not a general cycle-enumeration or thermodynamic-certification procedure.
"""

from pathlib import Path
import hashlib
import cobra
import numpy as np
import pandas as pd

INPUT_DIR = Path("inputs")
OUTPUT_DIR = Path("outputs/internal_energy_cycle_results")
REFERENCE_PATH = INPUT_DIR / "haeconGEM1.xml"
SOLVER = "gurobi"
TOL = 1e-07
NUM_TOL = 1e-06
MODEL_PATHS = {
    stage: INPUT_DIR / "CHRR_sampling/models" / f"{stage}_c.xml"
    for stage in ["egg", "L1", "L2", "L3", "L4f", "L4m", "Am", "Af"]
}
MODEL_PATHS["L4union"] = INPUT_DIR / "CHRR_sampling/models/L4mf_union_c.xml"
MODEL_PATHS["HaeconGEM1"] = REFERENCE_PATH


def specs():
    out = {}
    for a, b in [
        ("ATP", "ADP"),
        ("CTP", "CDP"),
        ("GTP", "GDP"),
        ("UTP", "UDP"),
        ("ITP", "IDP"),
    ]:
        out[a] = [(a, -1), ("H2O", -1), (b, 1), ("H+", 1), ("Pi", 1)]
    for a, b, h in [
        ("NADH", "NAD+", 1),
        ("NADPH", "NADP+", 1),
        ("FADH2", "FAD", 2),
        ("FMNH2", "FMN", 2),
        ("ubiquinol", "ubiquinone", 2),
        ("Menaquinol-8", "Menaquinone-8", 2),
        ("2-Demethylmenaquinol-8", "2-Demethylmenaquinone-8", 2),
        ("rhodoquinone reduced", "rhodoquinone", 2),
    ]:
        out[a] = [(a, -1), (b, 1), ("H+", h)]
    out["acetyl-CoA"] = [
        ("acetyl-CoA", -1),
        ("H2O", -1),
        ("H+", 1),
        ("acetate", 1),
        ("CoA", 1),
    ]
    out["glutamate"] = [
        ("glutamate", -1),
        ("H2O", -1),
        ("AKG", 1),
        ("NH4+", 1),
        ("H+", 2),
    ]
    out["PPi"] = [("PPi", -1), ("H2O", -1), ("Pi", 2), ("H+", 1)]
    return out


def close(m, scenario):
    changes = []
    for r in m.reactions:
        old = r.bounds
        if r.boundary:
            if scenario == "all_boundaries_closed":
                new = (0, 0)
            else:
                assert bool(r.reactants) != bool(r.products)
                new = (0, max(0, old[1])) if r.reactants else (min(0, old[0]), 0)
        else:
            new = (min(0, old[0]), max(0, old[1]))
        if new != old:
            changes.append(
                dict(
                    reaction_id=r.id,
                    old_lb=old[0],
                    old_ub=old[1],
                    new_lb=new[0],
                    new_ub=new[1],
                )
            )
            r.bounds = new
    return changes


def numerical(m, sol):
    if sol.status != "optimal":
        return (np.nan, np.nan)
    residual = {met.id: 0.0 for met in m.metabolites}
    for r in m.reactions:
        v = float(sol.fluxes[r.id])
        for met, c in r.metabolites.items():
            residual[met.id] += c * v
    err = max((abs(x) for x in residual.values()))
    violation = max(
        [0.0]
        + [
            max(r.lower_bound - sol.fluxes[r.id], sol.fluxes[r.id] - r.upper_bound)
            for r in m.reactions
        ]
    )
    assert np.isfinite(err) and np.isfinite(violation)
    return (err, violation)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reference = cobra.io.read_sbml_model(str(REFERENCE_PATH))
    index = {(m.name.casefold(), m.compartment): m.id for m in reference.metabolites}
    akg = [
        m
        for m in reference.metabolites
        if m.compartment == "c"
        and m.name.casefold()
        in [
            "akg",
            "2-oxoglutarate",
            "alpha-ketoglutarate",
            "alpha-ketoglutaric acid",
            "2-oxoglutaric acid",
        ]
    ]
    if not akg:
        akg = [
            m
            for m in reference.metabolites
            if m.compartment == "c" and m.formula in ["C5H4O5", "C5H6O5"]
        ]
    if len(akg) == 1:
        stem = akg[0].id[:-1]
        for metabolite in reference.metabolites:
            if metabolite.id.startswith(stem) and metabolite.name == akg[0].name:
                index["akg", metabolite.compartment] = metabolite.id
    tests, coverage, changes, manifest = ([], [], [], [])
    for stage, path in MODEL_PATHS.items():
        model = cobra.io.read_sbml_model(str(path))
        model.solver = SOLVER
        model.solver.configuration.timeout = 30
        model.solver.configuration.tolerances.feasibility = 1e-09
        model.solver.configuration.tolerances.optimality = 1e-09
        manifest.append(
            dict(
                model=stage,
                path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
        mapped = []
        for compartment in sorted({m.compartment for m in model.metabolites} - {"e"}):
            for currency, definition in specs().items():
                mapping = [
                    (index.get((name.casefold(), compartment)), coefficient)
                    for name, coefficient in definition
                ]
                missing = [
                    name
                    for (name, _), (mid, _) in zip(definition, mapping)
                    if mid is None or mid not in model.metabolites
                ]
                coverage.append(
                    dict(
                        model=stage,
                        currency=currency,
                        compartment=compartment,
                        missing=";".join(missing),
                        coverage="missing_metabolites" if missing else "tested",
                    )
                )
                if missing:
                    continue
                drain = cobra.Reaction(f"EDR_{len(mapped)}")
                drain.bounds = (0, 1000)
                drain.add_metabolites(
                    {
                        model.metabolites.get_by_id(mid): coefficient
                        for mid, coefficient in mapping
                    }
                )
                mapped.append((currency, compartment, drain))
        for scenario in ["no_uptake", "all_boundaries_closed"]:
            with model:
                changes.extend(
                    (
                        dict(model=stage, scenario=scenario, **row)
                        for row in close(model, scenario)
                    )
                )
                model.objective = 0
                model.add_reactions([drain.copy() for _, _, drain in mapped])
                assays = [
                    (currency, compartment, model.reactions.get_by_id(drain.id))
                    for currency, compartment, drain in mapped
                ]
                protons = {
                    m.compartment: m for m in model.metabolites if m.name == "H+"
                }
                for left, right in [
                    ("i", "m"),
                    ("c", "m"),
                    ("e", "c"),
                    ("c", "r"),
                    ("c", "g"),
                    ("c", "l"),
                    ("c", "x"),
                    ("c", "n"),
                ]:
                    if left not in protons or right not in protons:
                        continue
                    for origin, destination in [(left, right), (right, left)]:
                        drain = cobra.Reaction(f"PROTON_{origin}_{destination}")
                        drain.bounds = (0, 1000)
                        drain.add_metabolites(
                            {protons[origin]: -1, protons[destination]: 1}
                        )
                        assays.append(
                            ("proton_transfer", origin + "->" + destination, drain)
                        )
                for currency, compartment, assay in assays:
                    with model:
                        if assay.id not in model.reactions:
                            model.add_reactions([assay.copy()])
                        if currency == "proton_transfer":
                            for drain in model.reactions:
                                if drain.id.startswith("EDR_"):
                                    drain.bounds = (0, 0)
                        model.objective = model.reactions.get_by_id(assay.id)
                        model.objective_direction = "max"
                        row = dict(
                            model=stage,
                            scenario=scenario,
                            currency=currency,
                            compartment=compartment,
                            assay_class=(
                                "proton_path_diagnostic"
                                if currency == "proton_transfer"
                                else "energy_carrier_dissipation"
                            ),
                            status="unresolved",
                            assessment="unresolved",
                            max_dissipation=np.nan,
                        )
                        try:
                            solution = model.optimize()
                            row["status"] = solution.status
                            if solution.status == "optimal":
                                value = float(solution.objective_value)
                                residual, violation = numerical(model, solution)
                                row.update(
                                    max_dissipation=value,
                                    steady_state_residual=residual,
                                    bound_violation=violation,
                                )
                                if (
                                    not np.isfinite(value)
                                    or max(residual, violation) > NUM_TOL
                                ):
                                    row["assessment"] = "unresolved_numerical_error"
                                elif currency == "proton_transfer":
                                    row["assessment"] = (
                                        "proton_transfer_supported"
                                        if value > TOL
                                        else "proton_transfer_not_supported"
                                    )
                                else:
                                    row["assessment"] = (
                                        "candidate_energy_generation"
                                        if value > TOL
                                        else "not_detected"
                                    )
                        except Exception as error:
                            row.update(status="error", error=str(error))
                        tests.append(row)
    results = pd.DataFrame(tests)
    results.to_csv(OUTPUT_DIR / "extended_energy_tests.tsv", sep="\t", index=False)
    pd.DataFrame(coverage).to_csv(
        OUTPUT_DIR / "energy_assay_coverage.tsv", sep="\t", index=False
    )
    pd.DataFrame(changes).to_csv(
        OUTPUT_DIR / "extended_energy_bound_changes.tsv", sep="\t", index=False
    )
    pd.DataFrame(manifest).to_csv(
        OUTPUT_DIR / "extended_input_manifest.tsv", sep="\t", index=False
    )
    summary = (
        results.groupby(["model", "scenario", "assay_class"], sort=False)
        .agg(
            tests=("assessment", "size"),
            candidates=(
                "assessment",
                lambda x: sum(x == "candidate_energy_generation"),
            ),
            proton_paths_supported=(
                "assessment",
                lambda x: sum(x == "proton_transfer_supported"),
            ),
            unresolved=("assessment", lambda x: sum(x.str.startswith("unresolved"))),
            largest_dissipation=("max_dissipation", "max"),
        )
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "extended_energy_summary.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
