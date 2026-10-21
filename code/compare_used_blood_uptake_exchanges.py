"""Actual uptake exchanges in saved blood-condition pFBA, and other-model transporters."""

from pathlib import Path
import pandas as pd, cobra

INPUT_DIR = Path("inputs")
OUTPUT_DIR = Path("outputs/used_blood_exchange_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def same_species(ext, internal):
    stem = lambda met: (
        met.id[: -len(met.compartment)]
        if met.compartment and met.id.endswith(met.compartment)
        else met.id
    )
    return stem(ext) == stem(internal) or (
        ext.name == internal.name and ext.formula == internal.formula
    )


def find_transport(m, eid):
    if eid not in m.metabolites:
        return []
    ext = m.metabolites.get_by_id(eid)
    found = []
    for r in ext.reactions:
        if r.boundary:
            continue
        inside = [met for met in r.metabolites if met.compartment != "e"]
        if not inside:
            continue
        ce = r.metabolites[ext]
        identical = [
            met
            for met in inside
            if same_species(ext, met) and ce * r.metabolites[met] < 0
        ]
        exact = bool(identical)
        direction = []
        if exact:
            if ce < 0 and r.upper_bound > 0:
                direction.append("positive_flux")
            if ce > 0 and r.lower_bound < 0:
                direction.append("negative_flux")
        else:
            if (
                ce < 0
                and r.upper_bound > 0
                and any((r.metabolites[met] > 0 for met in inside))
            ):
                direction.append("positive_flux")
            if (
                ce > 0
                and r.lower_bound < 0
                and any((r.metabolites[met] < 0 for met in inside))
            ):
                direction.append("negative_flux")
        found.append(
            dict(
                reaction=r,
                kind=(
                    "same_metabolite_translocation"
                    if exact
                    else "other_cross_compartment_reaction"
                ),
                import_directions=direction,
            )
        )
    return found


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TOL = 0.0001
    models = {
        s: cobra.io.read_sbml_model(
            str(INPUT_DIR / "CHRR_sampling/models" / f"{s}_c.xml")
        )
        for s in ["Af", "Am"]
    }
    flux = {
        s: pd.read_csv(INPUT_DIR / "CHRR_sampling" / f"{s}_b_c_pfba_fluxes.csv")
        .set_index("reaction_id")
        .flux
        for s in models
    }
    constraints = {
        s: pd.read_csv(
            INPUT_DIR / "fma_pFBA_Yields" / f"{s}_b_c_blood_constraints.csv"
        ).set_index("exchange_reaction_id")
        for s in models
    }
    used = {}
    exchanges = {}
    for s, m in models.items():
        exchanges[s] = {r.id: r for r in m.exchanges}
        used[s] = {}
        for rid, r in exchanges[s].items():
            assert len(r.metabolites) == 1
            met, coef = next(iter(r.metabolites.items()))
            v = float(flux[s].get(rid, 0))
            rate = coef * v
            if rate > TOL:
                used[s][rid] = (met, rate, v)
    rows = []
    missing = []
    transport = []
    for rid in sorted(set(used["Af"]) | set(used["Am"])):
        src = next((s for s in models if rid in used[s]))
        met, rate, v = used[src][rid]
        row = dict(
            exchange_reaction_id=rid,
            metabolite_id=met.id,
            metabolite_name=met.name,
            formula=met.formula,
        )
        for stage, m in models.items():
            present = rid in exchanges[stage]
            active = rid in used[stage]
            r = exchanges[stage].get(rid)
            fv = float(flux[stage].get(rid, 0)) if present else float("nan")
            cr = constraints[stage]
            action = (
                cr.loc[rid, "action"]
                if rid in cr.index
                else "not_in_carbon_constraint_table"
            )
            carriers = find_transport(m, met.id)
            direct = [
                x for x in carriers if x["kind"] == "same_metabolite_translocation"
            ]
            importers = [x for x in direct if x["import_directions"]]
            row.update(
                {
                    stage + "_exchange_present": present,
                    stage + "_exchange_flux": fv,
                    stage + "_uptake_used": active,
                    stage + "_uptake_rate": used[stage][rid][1] if active else 0.0,
                    stage + "_constraint_action": action,
                    stage
                    + "_retained_direct_transporter_IDs": ";".join(
                        sorted((x["reaction"].id for x in direct))
                    ),
                    stage
                    + "_import_permitted_transporter_IDs": ";".join(
                        sorted((x["reaction"].id for x in importers))
                    ),
                }
            )
            for item in direct:
                tr = item["reaction"]
                transport.append(
                    dict(
                        exchange_reaction_id=rid,
                        metabolite_name=met.name,
                        model=stage,
                        transporter_reaction_id=tr.id,
                        import_permitted=bool(item["import_directions"]),
                        equation=tr.build_reaction_string(use_metabolite_names=True),
                        gpr=tr.gene_reaction_rule,
                    )
                )
            if not present:
                other = "Am" if stage == "Af" else "Af"
                assert rid in used[other]
                missing.append(
                    dict(
                        used_by_model=other,
                        missing_from_model=stage,
                        exchange_reaction_id=rid,
                        metabolite_id=met.id,
                        metabolite_name=met.name,
                        uptake_rate_in_using_model=used[other][rid][1],
                        using_model_constraint_action=(
                            constraints[other].loc[rid, "action"]
                            if rid in constraints[other].index
                            else "not_in_carbon_constraint_table"
                        ),
                        transporter_present_in_missing_model=bool(direct),
                        import_permitted_transporter_present=bool(importers),
                        transporter_IDs=";".join(
                            sorted((x["reaction"].id for x in direct))
                        ),
                        interpretation=(
                            "missing_exchange_despite_retained_importer"
                            if importers
                            else (
                                "exchange_and_direct_transporter_absent"
                                if not direct
                                else "transporter_retained_but_no_import_direction_permitted"
                            )
                        ),
                    )
                )
        rows.append(row)
    rows = pd.DataFrame(rows)
    missing = pd.DataFrame(
        missing,
        columns=[
            "used_by_model",
            "missing_from_model",
            "exchange_reaction_id",
            "metabolite_id",
            "metabolite_name",
            "uptake_rate_in_using_model",
            "using_model_constraint_action",
            "transporter_present_in_missing_model",
            "import_permitted_transporter_present",
            "transporter_IDs",
            "interpretation",
        ],
    )
    rows.to_csv(
        OUTPUT_DIR / "actually_used_uptake_exchanges.tsv", sep="\t", index=False
    )
    missing.to_csv(
        OUTPUT_DIR / "used_exchanges_missing_from_other_model.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(transport).to_csv(
        OUTPUT_DIR / "transporters_for_actually_used_exchanges.tsv",
        sep="\t",
        index=False,
    )
    blood_ids = set()
    for stage, cr in constraints.items():
        blood_ids |= set(cr[cr.action.eq("blood_carbon_source_kept_6_over_nC")].index)
    rows["designated_blood_component_in_either_model"] = rows.exchange_reaction_id.isin(
        blood_ids
    )
    rows.to_csv(
        OUTPUT_DIR / "actually_used_uptake_exchanges.tsv", sep="\t", index=False
    )
    blood = rows[rows.designated_blood_component_in_either_model]
    blood.to_csv(
        OUTPUT_DIR / "actually_used_designated_blood_component_exchanges.tsv",
        sep="\t",
        index=False,
    )
    missing_blood = missing[missing.exchange_reaction_id.isin(blood_ids)]
    missing_blood.to_csv(
        OUTPUT_DIR / "used_designated_blood_exchanges_missing_from_other_model.tsv",
        sep="\t",
        index=False,
    )
    summary = []
    for label, g in [
        ("all_actual_uptake", rows),
        ("designated_blood_components", blood),
    ]:
        ids = set(g.exchange_reaction_id)
        ms = missing[missing.exchange_reaction_id.isin(ids)]
        summary.append(
            dict(
                scope=label,
                Af_used=int(g.Af_uptake_used.sum()),
                Am_used=int(g.Am_uptake_used.sum()),
                used_in_both=int((g.Af_uptake_used & g.Am_uptake_used).sum()),
                union_used=len(g),
                used_exchanges_absent_from_other_model=len(ms),
                absent_exchange_and_transporter=int(
                    (~ms.transporter_present_in_missing_model).sum()
                ),
                absent_exchange_with_retained_importer=int(
                    ms.import_permitted_transporter_present.sum()
                ),
            )
        )
    summary = pd.DataFrame(summary)
    summary.to_csv(OUTPUT_DIR / "used_exchange_summary.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
