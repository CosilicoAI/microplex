"""Build a spec-driven US eCPS-replacement candidate and optionally score it.

Pipeline (v1 architecture, see _MISSION_JOURNAL.md):

1. Load ASEC persons/households with raw pointer columns.
2. Construct the six-entity unit structure (microunit tax engine).
3. Aggregate persons to tax units -> the spine base frame.
4. run_spec: seeded 50/50 support spine + PUF donor imputation
   (steps lifted from packs/us/specs/us-2024.yaml).
5. Assign block geography per household.
6. Re-attach persons; allocate tax-unit-imputed amounts to heads.
7. Export a USSingleYearDataset H5 gated by the eCPS export contract.
8. (--score) Run the sound eCPS-replacement comparison via the legacy
   harness in ~/CosilicoAI/microplex-us.

Smoke: .venv/bin/python scripts/build_us_candidate.py --mode smoke
Full:  .venv/bin/python scripts/build_us_candidate.py --mode full --score
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from microplex.data_sources.cps import load_cps_asec  # noqa: E402
from microplex.data_sources.us_registry import (  # noqa: E402
    create_us_asec_puf_source_registry,
)
from microplex.export import (  # noqa: E402
    ExportContract,
    export_policyengine_us_dataset,
)
from microplex.run import run_spec  # noqa: E402
from microplex.spec import load_spec_dict  # noqa: E402
from microplex.units import assign_us_unit_structure  # noqa: E402

US_DATA_STORAGE = Path(
    "~/PolicyEngine/policyengine-us-data/policyengine_us_data/storage"
).expanduser()
OLD_WORKTREE = Path("~/CosilicoAI/microplex-us").expanduser()
BLOCK_CROSSWALK = Path(
    "~/CosilicoAI/microplex/data/block_probabilities.parquet"
).expanduser()

MICROUNIT_RAW_COLUMNS = (
    "A_LINENO",
    "A_AGE",
    "A_MARITL",
    "A_SPOUSE",
    "PEPAR1",
    "PEPAR2",
    "A_EXPRRP",
    "SPM_ID",
    "PF_SEQ",
    "A_HSCOL",
)

# Person-level ASEC harmonized income columns that sum to tax-unit totals.
PERSON_INCOME_COLUMNS = (
    "employment_income",
    "self_employment_income",
    "taxable_interest_income",
    "rental_income",
    "social_security",
    "taxable_pension_income",
    "unemployment_compensation",
)


# Donor-named variables the PUF source actually carries, imputed at
# tax-unit grain. CPS-measured ones (also on the spine base) are listed in
# CPS_MEASURED; the rest are PUF-only detail imputed onto both halves.
PUF_IMPUTE_VARS = (
    "employment_income",
    "self_employment_income",
    "social_security",
    "taxable_pension_income",
    "taxable_interest_income",
    "unemployment_compensation",
    "rental_income",
    "partnership_s_corp_income",
    "farm_income",
    "tax_exempt_interest_income",
    "qualified_dividend_income",
    "ordinary_dividend_income",
    "short_term_capital_gains",
    "long_term_capital_gains",
    "taxable_pension_income",
    "total_pension_income",
    "ira_distributions",
    "alimony_received",
    "charitable_cash",
    "charitable_noncash",
    "mortgage_interest_paid",
    "real_estate_tax_paid",
)
CPS_MEASURED = (
    "employment_income",
    "self_employment_income",
    "social_security",
    "taxable_pension_income",
    "taxable_interest_income",
    "unemployment_compensation",
    "rental_income",
)

# donor/common name -> PolicyEngine contract name at person allocation.
DONOR_TO_PE = {
    "unemployment_compensation": "taxable_unemployment_compensation",
    "ira_distributions": "taxable_ira_distributions",
    "alimony_received": "alimony_income",
    "charitable_cash": "charitable_cash_donations",
    "charitable_noncash": "charitable_non_cash_donations",
    "mortgage_interest_paid": "interest_deduction",
    "real_estate_tax_paid": "real_estate_taxes",
    "ordinary_dividend_income": "non_qualified_dividend_income",
}

SHARED_PREDICTORS = ("age", "is_joint", "n_people", "n_children")


def _build_imputation_steps() -> list[dict]:
    """v1 ASEC+PUF imputation graph over donor-available variables."""
    puf_vars = list(dict.fromkeys(PUF_IMPUTE_VARS))
    puf_only = [v for v in puf_vars if v not in CPS_MEASURED]
    return [
        {"onto": "synthetic_puf", "from": "puf", "vars": puf_vars,
         "order": "spine_first"},
        {"onto": "cps_keep", "from": "puf", "vars": puf_only,
         "condition_on": ["demographics", *CPS_MEASURED]},
    ]


def _aggregate_tax_units(person: pd.DataFrame, tax_unit: pd.DataFrame) -> pd.DataFrame:
    """Aggregate persons to the tax-unit-grain spine base frame."""
    g = person.groupby("person_tax_unit_id", sort=True)
    base = pd.DataFrame(index=g.size().index)
    base["tax_unit_id"] = base.index
    base["household_id"] = g["household_id"].first()
    base["n_people"] = g.size().astype(float)
    is_head = person["tax_unit_role_input"] == "HEAD"
    head_age = (
        person.loc[is_head]
        .groupby("person_tax_unit_id")["A_AGE"]
        .max()
        .astype(float)
    )
    base["age"] = head_age.reindex(base.index).fillna(
        g["A_AGE"].max().astype(float)
    )
    base["n_children"] = (
        person.assign(_child=(person["A_AGE"] < 18).astype(float))
        .groupby("person_tax_unit_id")["_child"]
        .sum()
        .reindex(base.index)
        .fillna(0.0)
    )
    base["is_joint"] = (
        person.assign(_sp=(person["tax_unit_role_input"] == "SPOUSE").astype(float))
        .groupby("person_tax_unit_id")["_sp"]
        .max()
        .reindex(base.index)
        .fillna(0.0)
    )
    for col in PERSON_INCOME_COLUMNS:
        if col in person.columns:
            base[col] = g[col].sum()
    fs = tax_unit.set_index("tax_unit_id")["filing_status_input"]
    base["filing_status_input"] = base["tax_unit_id"].map(fs)
    base = base.reset_index(drop=True)
    return base


def _attach_household_columns(
    base: pd.DataFrame, households: pd.DataFrame
) -> pd.DataFrame:
    keep = ["household_id", "state_fips", "household_weight"]
    have = [c for c in keep if c in households.columns]
    return base.merge(
        households[have].drop_duplicates("household_id"),
        on="household_id",
        how="left",
    )


def _assign_blocks(
    households: pd.DataFrame, crosswalk_path: Path, seed: int
) -> pd.DataFrame:
    """Probability-weighted census block assignment per household by state."""
    xw = pd.read_parquet(crosswalk_path)
    rng = np.random.default_rng(seed)
    households = households.copy()
    households["block_geoid"] = ""
    households["county_fips"] = ""
    households["congressional_district_geoid"] = ""
    xw["state_fips"] = xw["state_fips"].astype(int)
    for state, idx in households.groupby(
        households["state_fips"].astype(int)
    ).groups.items():
        pool = xw[xw["state_fips"] == state]
        if pool.empty:
            pool = xw
        p = pool["prob"].to_numpy()
        p = p / p.sum()
        draw = rng.choice(len(pool), size=len(idx), p=p)
        chosen = pool.iloc[draw]
        households.loc[idx, "block_geoid"] = chosen["geoid"].astype(str).to_numpy()
        households.loc[idx, "county_fips"] = (
            chosen["geoid"].astype(str).str[:5].to_numpy()
        )
        households.loc[idx, "congressional_district_geoid"] = (
            chosen["cd_id"].astype(str).to_numpy()
        )
    return households


def _allocate_to_persons(
    person: pd.DataFrame,
    spine: pd.DataFrame,
    imputed_vars: list[str],
    half_col: str,
) -> pd.DataFrame:
    """Re-attach spine tax-unit values to persons.

    cps_keep persons keep their ASEC person-level values for CPS-measured
    columns; PUF-only imputed amounts go to the unit head. synthetic_puf
    persons get every imputed variable head-allocated (others zero).
    """
    spine_idx = spine.set_index("tax_unit_id")
    person = person.copy()
    person["_half"] = person["person_tax_unit_id"].map(spine_idx[half_col])
    person = person[person["_half"].notna()].copy()
    is_head = person["tax_unit_role_input"] == "HEAD"
    synthetic = person["_half"] == "synthetic_puf"

    for var in imputed_vars:
        if var not in spine_idx.columns:
            continue
        unit_value = person["person_tax_unit_id"].map(spine_idx[var])
        head_alloc = np.where(is_head, unit_value.fillna(0.0), 0.0)
        if var in person.columns:
            # CPS-measured: keep person values on cps_keep, head-allocate
            # the synthetic draw on the synthetic half.
            person[var] = np.where(
                synthetic, head_alloc, person[var].fillna(0.0)
            )
        else:
            person[var] = head_alloc
    return person


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--asec-year", type=int, default=2025)
    ap.add_argument("--calendar-year", type=int, default=2024)
    ap.add_argument("--puf-year", type=int, default=2024)
    ap.add_argument("--max-tax-units", type=int, default=None)
    ap.add_argument("--max-puf-rows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260529)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--score", action="store_true")
    ap.add_argument(
        "--baseline-h5",
        type=Path,
        default=US_DATA_STORAGE / "enhanced_cps_2024.h5",
    )
    args = ap.parse_args()

    smoke = args.mode == "smoke"
    max_units = args.max_tax_units or (4000 if smoke else None)
    max_puf = args.max_puf_rows or (8000 if smoke else None)
    out = args.output_dir or (
        REPO / "artifacts" / f"spec_candidate_{args.mode}_{args.calendar_year}"
    )
    out.mkdir(parents=True, exist_ok=True)
    log = lambda *a: print("[build]", *a, flush=True)  # noqa: E731

    # ---- Stage A: ASEC persons + households -------------------------------
    log("stage A: loading ASEC persons/households")
    ds = load_cps_asec(
        year=args.asec_year,
        extra_person_columns=list(MICROUNIT_RAW_COLUMNS) + ["PH_SEQ"],
    )
    persons = ds.persons.to_pandas()
    households = ds.households.to_pandas()
    log(f"  persons={len(persons):,} households={len(households):,}")

    # ---- Stage B: unit structure ------------------------------------------
    log("stage B: unit assignment (microunit)")
    units = assign_us_unit_structure(persons, year=args.calendar_year)
    person = units.person
    log(
        f"  tax_units={len(units.tax_unit):,} spm={len(units.spm_unit):,} "
        f"families={len(units.family):,} marital={len(units.marital_unit):,}"
    )

    # ---- Stage C: tax-unit spine base --------------------------------------
    log("stage C: aggregating tax-unit spine base")
    units_tu = units.tax_unit.rename(columns={"TAX_ID": "tax_unit_id"})
    base = _aggregate_tax_units(person, units_tu)
    base = _attach_household_columns(base, households)
    if max_units is not None:
        keep_units = base["tax_unit_id"].head(max_units)
        base = base[base["tax_unit_id"].isin(keep_units)].copy()
        person = person[person["person_tax_unit_id"].isin(keep_units)].copy()
    log(f"  spine base tax units: {len(base):,}")

    # ---- Stage D: run_spec --------------------------------------------------
    log("stage D: run_spec (support spine + PUF imputation)")
    steps = _build_imputation_steps()
    imputed_vars = sorted({v for s in steps for v in s.get("vars", [])})
    registry = create_us_asec_puf_source_registry(
        asec_year=args.asec_year,
        calendar_year=args.calendar_year,
        puf_year=args.puf_year,
        puf_path=US_DATA_STORAGE / "puf_2015.csv",
        puf_demographics_path=US_DATA_STORAGE / "demographics_2015.csv",
    )
    id_keep = [
        "tax_unit_id",
        "household_id",
        "household_weight",
        "state_fips",
        "filing_status_input",
    ]
    spec = load_spec_dict(
        {
            "meta": {"country": "us", "model_year": args.calendar_year},
            "sources": {
                "cps_asec": {
                    "dataset": (
                        f"cps_asec_{args.asec_year}_calendar_{args.calendar_year}"
                    ),
                    "role": "spine",
                },
                "puf": {"dataset": f"puf_{args.puf_year}", "role": "donor"},
            },
            "spine": {
                "base": "cps_asec",
                "method": "support_spine",
                "support": {"seed": args.seed},
                "halves": [
                    {"name": "cps_keep", "keep": "all"},
                    {
                        "name": "synthetic_puf",
                        "strip_to": ["demographics", *id_keep],
                    },
                ],
            },
            "imputation": steps,
        }
    )
    puf = registry.resolve_source(spec, "puf")
    if max_puf is not None:
        puf = puf.head(max_puf).copy()
    # Harmonize donor names + derive the shared predictor surface.
    puf = puf.rename(columns={"gross_social_security": "social_security"})
    puf["is_joint"] = (puf["filing_status"] == "JOINT").astype(float)
    puf["n_people"] = puf["exemptions_count"].clip(lower=1).astype(float)
    puf["n_children"] = puf["ctc_children"].fillna(0).astype(float)
    puf["age"] = puf["age"].astype(float)
    log(f"  puf donor rows: {len(puf):,}")
    result = run_spec(
        spec,
        {"cps_asec": base, "puf": puf},
        demographic_columns=SHARED_PREDICTORS,
        spine_keywords=(
            "employment_income",
            "self_employment_income",
            "social_security",
            "taxable_pension_income",
            "taxable_interest_income",
        ),
    )
    spine = result.frame
    half_col = [c for c in spine.columns if c.startswith("_spine")][0]
    log(
        f"  spine rows={len(spine):,} halves="
        f"{spine[half_col].value_counts().to_dict()}"
    )

    # ---- Stage E: geography -------------------------------------------------
    log("stage E: block geography")
    households = _assign_blocks(
        households[households["household_id"].isin(person["household_id"])],
        BLOCK_CROSSWALK,
        args.seed,
    )

    # ---- Stage F: entity assembly ------------------------------------------
    log("stage F: person re-attach + entity tables")
    person = _allocate_to_persons(person, spine, imputed_vars, half_col)
    # Donor/common names -> PolicyEngine contract names; derived splits.
    person = person.rename(columns=DONOR_TO_PE)
    if {"total_pension_income", "taxable_pension_income"} <= set(person.columns):
        person["tax_exempt_pension_income"] = (
            person["total_pension_income"] - person["taxable_pension_income"]
        ).clip(lower=0.0)
        person = person.drop(columns=["total_pension_income"])
    if {
        "non_qualified_dividend_income",
        "qualified_dividend_income",
    } <= set(person.columns):
        person["non_qualified_dividend_income"] = (
            person["non_qualified_dividend_income"]
            - person["qualified_dividend_income"]
        ).clip(lower=0.0)
    person["person_household_id"] = person["household_id"]
    person["person_id"] = np.arange(1, len(person) + 1, dtype=np.int64)
    person["age"] = person["A_AGE"].astype(float)

    hh_ids = person["person_household_id"].unique()
    hh = households[households["household_id"].isin(hh_ids)].copy()

    def unit_table(id_col: str, source: pd.DataFrame | None = None) -> pd.DataFrame:
        ids = np.sort(person[f"person_{id_col}"].unique())
        t = pd.DataFrame({id_col: ids})
        if source is not None:
            extra = source.rename(columns={"TAX_ID": id_col})
            t = t.merge(extra, on=id_col, how="left")
        return t

    class _Key:
        def __init__(self, value: str):
            self.value = value

    entity_frames = {
        _Key("person"): person,
        _Key("household"): hh,
        _Key("tax_unit"): unit_table("tax_unit_id", units_tu),
        _Key("spm_unit"): unit_table("spm_unit_id"),
        _Key("family"): unit_table("family_id"),
        _Key("marital_unit"): unit_table("marital_unit_id"),
    }

    # ---- Stage G: export ----------------------------------------------------
    log("stage G: export")
    contract = ExportContract.from_path(
        REPO / "packs/us/manifests/ecps_export_contract.json"
    )
    defaults = json.loads(
        (REPO / "packs/us/manifests/export_defaults.json").read_text()
    )
    defaults.pop("_source", None)
    candidate_h5 = out / "candidate_policyengine_us.h5"
    gate = export_policyengine_us_dataset(
        entity_frames,
        period=args.calendar_year,
        output_path=candidate_h5,
        contract=contract,
        defaults=defaults,
        allow_incomplete=smoke,
    )
    (out / "export_gate.json").write_text(json.dumps(gate.to_dict(), indent=2))
    log(
        f"  gate passed={gate.passed} missing={len(gate.missing_required)} "
        f"defaulted={len(gate.defaulted)} dropped={len(gate.dropped)}"
    )
    if gate.missing_required:
        log(f"  missing (first 25): {list(gate.missing_required)[:25]}")

    # ---- Stage H: score -----------------------------------------------------
    if args.score and candidate_h5.exists():
        log("stage H: sound eCPS comparison (legacy harness)")
        cmd = [
            str(OLD_WORKTREE / ".venv/bin/python"),
            "-m",
            "microplex_us.pipelines.ecps_replacement_comparison",
            "--candidate-dataset",
            str(candidate_h5),
            "--baseline-dataset",
            str(args.baseline_h5),
            "--output-dir",
            str(out / "sound_comparison"),
            "--period",
            str(args.calendar_year),
        ]
        log("  " + " ".join(cmd))
        proc = subprocess.run(cmd, cwd=OLD_WORKTREE, capture_output=True, text=True)
        (out / "score_stdout.log").write_text(proc.stdout)
        (out / "score_stderr.log").write_text(proc.stderr)
        log(f"  harness exit: {proc.returncode}")
        result_json = out / "sound_comparison" / "sound_ecps_replacement_comparison.json"
        if result_json.exists():
            payload = json.loads(result_json.read_text())
            log(json.dumps(payload.get("headline", payload), indent=2)[:2000])
        return proc.returncode

    return 0 if (gate.passed or smoke) else 1


if __name__ == "__main__":
    raise SystemExit(main())
