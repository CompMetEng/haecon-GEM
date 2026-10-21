"""Assemble reference-model quality metrics without modifying the model.

Reuse hash-matched saved FVA and energy tests; calculate structural diagnostics
and use MEMOTE's implementation of stoichiometric consistency. This is a
targeted assessment, not a complete MEMOTE run or composite MEMOTE score.
"""

import csv
import hashlib
import importlib.metadata
import json
import re
from collections import defaultdict
from pathlib import Path
import cobra
from memote.support import consistency, consistency_helpers, helpers

INPUT_DIR = Path("inputs")
OUTPUT_DIR = Path("outputs/model_quality")
REFERENCE_PATH = INPUT_DIR / "haeconGEM1.xml"
COUNTS_PATH = INPUT_DIR / "final_model_reaction_counts_open_exchanges.json"
ENERGY_MANIFEST_PATH = INPUT_DIR / "extended_input_manifest.tsv"
ENERGY_TESTS_PATH = INPUT_DIR / "extended_energy_tests.tsv"
SOLVER = "gurobi"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL = REFERENCE_PATH
OUTPUT = OUTPUT_DIR / "model_quality_metrics.tsv"
DETAILS = OUTPUT_DIR / "model_quality_details.json"
FIELDS = ["Category", "Metric", "Result", "Percentage (%)", "Assessment status"]


def read_tsv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main():
    digest = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    saved = json.loads(COUNTS_PATH.read_text())
    assert digest == saved["sha256"], "Saved FVA input hash differs."
    cobra.Configuration().solver = SOLVER
    model = cobra.io.read_sbml_model(str(MODEL))
    model.solver.configuration.timeout = 60
    boundary = set(model.boundary)
    nonboundary = [r for r in model.reactions if r not in boundary]
    biomass = helpers.find_biomass_reaction(model)
    internal = sorted(consistency_helpers.get_internals(model), key=lambda r: r.id)
    assert len(model.reactions) == saved["total_reactions"]
    rows = []
    details = {
        "model": str(MODEL),
        "sha256": digest,
        "cobra_version": cobra.__version__,
        "memote_version": importlib.metadata.version("memote"),
        "biomass_excluded_by_memote": [r.id for r in biomass],
        "balance_tolerance": 1e-07,
        "assessment": "Targeted MEMOTE-style assessment; no composite score",
    }

    def add(category, metric, value, denominator=None, status="completed"):
        rows.append(
            {
                "Category": category,
                "Metric": metric,
                "Result": value,
                "Percentage (%)": (
                    100 * value / denominator
                    if denominator and isinstance(value, (int, float))
                    else ""
                ),
                "Assessment status": status,
            }
        )

    add("Inventory", "Reactions", len(model.reactions))
    add("Inventory", "Metabolites", len(model.metabolites))
    add("Inventory", "Genes", len(model.genes))
    add("Inventory", "Compartments", len(model.compartments))
    add("Inventory", "Exchange reactions", len(model.exchanges))
    demands = [
        r
        for r in boundary
        if len(r.metabolites) == 1
        and (not any((m.compartment == "e" for m in r.metabolites)))
    ]
    add("Inventory", "Intracellular boundary reactions", len(demands))
    transports = [
        r for r in nonboundary if len({m.compartment for m in r.metabolites}) > 1
    ]
    add("Inventory", "Transport reactions", len(transports), len(nonboundary))
    add("Inventory", "Nonboundary reactions", len(nonboundary))
    associated = [r for r in nonboundary if r.genes]
    add(
        "Gene associations",
        "Gene-associated nonboundary reactions",
        len(associated),
        len(nonboundary),
    )
    add(
        "Gene associations",
        "Gene-less nonboundary reactions",
        len(nonboundary) - len(associated),
        len(nonboundary),
    )
    formulas, charges, unresolved = ([], [], [])
    formula_pattern = re.compile("(?:[A-Z][a-z]?[0-9]*(?:\\.[0-9]+)?)+")
    for met in model.metabolites:
        if not met.formula:
            formulas.append(met.id)
        elif not formula_pattern.fullmatch(met.formula) or re.search(
            "(?:^|[^a-z])[RX](?:[^a-z]|$)", met.formula
        ):
            unresolved.append(met.id)
        if met.charge is None:
            charges.append(met.id)
    details.update(
        missing_formula=formulas, unresolved_formula=unresolved, missing_charge=charges
    )
    add(
        "Chemical annotation",
        "Metabolites missing formulas",
        len(formulas),
        len(model.metabolites),
    )
    add(
        "Chemical annotation",
        "Metabolites with unresolved formulas",
        len(unresolved),
        len(model.metabolites),
    )
    add(
        "Chemical annotation",
        "Metabolites missing charges",
        len(charges),
        len(model.metabolites),
    )
    zero_charges = sum((m.charge == 0 for m in model.metabolites))
    details["zero_stored_charges"] = zero_charges
    add(
        "Chemical annotation",
        "Metabolites with stored charge zero",
        zero_charges,
        len(model.metabolites),
    )
    mass_bad, charge_bad, mass_unknown, charge_unknown = ({}, {}, [], [])
    bad_formula_ids = set(formulas + unresolved)
    for rxn in internal:
        if any((m.id in bad_formula_ids for m in rxn.metabolites)):
            mass_unknown.append(rxn.id)
        else:
            residual = defaultdict(float)
            for met, coefficient in rxn.metabolites.items():
                for element, number in met.elements.items():
                    residual[element] += coefficient * number
            residual = {k: v for k, v in residual.items() if abs(v) > 1e-07}
            if residual:
                mass_bad[rxn.id] = residual
        if any((m.charge is None for m in rxn.metabolites)):
            charge_unknown.append(rxn.id)
        else:
            residual = sum((m.charge * c for m, c in rxn.metabolites.items()))
            if abs(residual) > 1e-07:
                charge_bad[rxn.id] = residual
    details.update(
        mass_imbalance=mass_bad,
        charge_imbalance=charge_bad,
        mass_not_assessable=mass_unknown,
        charge_not_assessable=charge_unknown,
    )
    for label, bad, unknown in [
        ("Mass", mass_bad, mass_unknown),
        ("Charge", charge_bad, charge_unknown),
    ]:
        eligible = len(internal) - len(unknown)
        balance_status = (
            "uninformative_all_charges_zero"
            if label == "Charge" and zero_charges == len(model.metabolites)
            else "completed"
        )
        add(
            "Balance",
            label + "-balance-assessable reactions",
            eligible,
            len(internal),
            status=balance_status,
        )
        add(
            "Balance",
            label + "-imbalanced reactions",
            len(bad),
            eligible,
            status=balance_status,
        )
        add(
            "Balance",
            label + "-balanced reactions",
            eligible - len(bad),
            eligible,
            status=balance_status,
        )
        add(
            "Balance",
            label + "-balance-unassessable reactions",
            len(unknown),
            len(internal),
        )
    for title, fn in [
        ("Only-consumed metabolites (orphans)", consistency.find_orphans),
        ("Only-produced metabolites (dead ends)", consistency.find_deadends),
        ("Disconnected metabolites", consistency.find_disconnected),
    ]:
        found = [m.id for m in fn(model)]
        details[title] = found
        add("Connectivity", title, len(found), len(model.metabolites))
    groups = defaultdict(list)
    for rxn in nonboundary:
        signature = tuple(
            sorted(((m.id, float(c)) for m, c in rxn.metabolites.items()))
        )
        groups[signature].append(rxn.id)
    duplicates = [v for v in groups.values() if len(v) > 1]
    details["identical_stoichiometry_groups"] = duplicates
    add("Duplicates", "Identical-stoichiometry candidate groups", len(duplicates))
    add(
        "Duplicates",
        "Reactions in identical-stoichiometry candidate groups",
        sum((len(g) for g in duplicates)),
        len(nonboundary),
    )
    for title, objects in [
        ("Reactions", model.reactions),
        ("Metabolites", model.metabolites),
        ("Genes", model.genes),
    ]:
        annotated = [x.id for x in objects if x.annotation]
        details[title + "_annotation_namespaces"] = sorted(
            {key for obj in objects for key in obj.annotation}
        )
        add(
            "Identifier coverage",
            title + " with any annotation",
            len(annotated),
            len(objects),
        )
        external = [
            x
            for x in objects
            if any(
                (
                    key not in {"sbo", "SBO", " SBO"} and bool(value)
                    for key, value in x.annotation.items()
                )
            )
        ]
        add(
            "Identifier coverage",
            title + " with non-SBO annotation",
            len(external),
            len(objects),
        )
    add(
        "Flux consistency",
        "Blocked nonboundary reactions",
        saved["blocked_nonboundary"],
        len(nonboundary),
    )
    add(
        "Flux consistency",
        "Blocked gene-associated nonboundary reactions",
        saved["blocked_gene_associated_nonboundary"],
        len(associated),
    )
    manifest = read_tsv(ENERGY_MANIFEST_PATH)
    assert next((x for x in manifest if x["model"] == "HaeconGEM1"))["sha256"] == digest
    assays = [
        x
        for x in read_tsv(ENERGY_TESTS_PATH)
        if x["model"] == "HaeconGEM1"
        and x["assay_class"] == "energy_carrier_dissipation"
    ]
    positives = sum(
        (
            x["status"] == "optimal" and float(x["max_dissipation"]) > 1e-07
            for x in assays
        )
    )
    details["energy_tests"] = {
        "n": len(assays),
        "positive": positives,
        "largest_dissipation": max((float(x["max_dissipation"]) for x in assays)),
    }
    add(
        "Energy cycles",
        "Positive mapped energy-dissipation tests",
        positives,
        len(assays),
        status=(
            "completed"
            if all((x["status"] == "optimal" for x in assays))
            else "unresolved"
        ),
    )
    add(
        "Energy cycles",
        "Largest mapped energy-dissipation flux",
        details["energy_tests"]["largest_dissipation"],
    )

    def save():
        with OUTPUT.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        DETAILS.write_text(json.dumps(details, indent=2) + "\n")

    save()
    try:
        outcome = consistency.check_stoichiometric_consistency(model)
        details["stoichiometric_consistency"] = bool(outcome)
        add(
            "Stoichiometric consistency",
            "Positive conserved mass vector",
            "Consistent" if outcome else "Inconsistent",
        )
    except Exception as exc:
        details["stoichiometric_consistency_error"] = str(exc)
        add(
            "Stoichiometric consistency",
            "Positive conserved mass vector",
            "Unresolved",
            status="solver_unresolved",
        )
    details["input_model_unchanged"] = (
        digest == hashlib.sha256(MODEL.read_bytes()).hexdigest()
    )
    assert details["input_model_unchanged"]
    save()


if __name__ == "__main__":
    main()
