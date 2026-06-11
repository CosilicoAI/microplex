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

# Raw ASEC columns backing CPS-threadable contract variables (mapping rules
# mirror the eCPS loader in policyengine-us-data cps.py).
EXTRA_ASEC_COLUMNS = (
    # v2 parity additions (rules mirror usdata cps.py; see V2_PLAN.md)
    "ED_VAL",
    "FIN_VAL",
    "SRVS_VAL",
    "VET_VAL",
    "WC_VAL",
    "OI_VAL",
    "OI_OFF",
    "A_HRS1",
    "WKSWORK",
    "POCCU2",
    "PEIOOCC",
    "RETCB_VAL",
    "WSAL_VAL",
    "SEMP_VAL",
    "CSP_VAL",
    "CHSP_VAL",
    "DIS_VAL1",
    "DIS_VAL2",
    "DIS_SC1",
    "DIS_SC2",
    "NOW_GRP",
    "NOW_MRK",
    "WICYN",
    "PHIP_VAL",
    "POTC_VAL",
    "PMED_VAL",
    "PEDISDRS",
    "PEDISEAR",
    "PEDISEYE",
    "PEDISOUT",
    "PEDISPHY",
    "PEDISREM",
    "RESNSS1",
    "RESNSS2",
    "SPM_ENGVAL",
    "SPM_CHILDCAREXPNS",
    "NOW_OWNGRP",
    "NOW_HIPAID",
    "NOW_GRPFTYP",
    "PERIDNUM",
    "LKWEEKS",
    "DST_SC1",
    "DST_SC2",
    "DST_SC1_YNG",
    "DST_SC2_YNG",
    "DST_VAL1",
    "DST_VAL2",
    "DST_VAL1_YNG",
    "DST_VAL2_YNG",
    "SPM_CAPWKCCXPNS",
    # v3 parity additions (ASEC raw sources for eCPS-populated layers)
    "A_FTPT",
    "P_SEQ",
    "NOW_NONM",
    "NOW_CAID",
    "NOW_OTHMT",
    "NOW_MIL",
    "NOW_CHAMPVA",
    "NOW_VACARE",
    "NOW_IHSFLG",
    "SPM_CAPHOUSESUB",
)

# Contract-required columns eCPS sources from PUF/SIPP/SCF detail our v1
# donor surface does not carry. Explicit zero/false defaults, recorded in
# the export gate's `defaulted` accounting. Iterate in v2.
V1_ZERO_DEFAULTS: dict[str, object] = {
    **{
        c: 0.0
        for c in (
            "bank_account_assets",
            "bond_assets",
            "stock_assets",
            "partnership_se_income",
            "tip_income",
            "employer_sponsored_insurance_premiums",
            "roth_401k_contributions",
            "roth_ira_contributions",
            "traditional_401k_contributions",
            "self_employed_pension_contributions",
        )
    },
    "receives_housing_assistance": False,
    "takes_up_housing_assistance_if_eligible": False,
    "takes_up_medicare_if_eligible": True,
    "reported_owns_employer_sponsored_health_insurance_at_interview": False,
    "is_surviving_spouse": False,
}


def _derive_person_columns(person: pd.DataFrame) -> pd.DataFrame:
    """Derive CPS-threadable contract columns (eCPS loader rules)."""
    p = person.copy()
    num = lambda c: pd.to_numeric(p.get(c, 0), errors="coerce").fillna(0)  # noqa: E731
    p["child_support_received"] = num("CSP_VAL").astype(float)
    p["child_support_expense"] = num("CHSP_VAL").astype(float)
    p["disability_benefits"] = (
        num("DIS_VAL1") * (num("DIS_SC1") != 1)
        + num("DIS_VAL2") * (num("DIS_SC2") != 1)
    ).astype(float)
    p["has_esi"] = (num("NOW_GRP") == 1).astype(bool)
    p["has_marketplace_health_coverage"] = (num("NOW_MRK") == 1).astype(bool)
    p["receives_wic"] = (num("WICYN") == 1).astype(bool)
    p["health_insurance_premiums_without_medicare_part_b"] = num(
        "PHIP_VAL"
    ).astype(float)
    p["over_the_counter_health_expenses"] = num("POTC_VAL").astype(float)
    p["other_medical_expenses"] = num("PMED_VAL").astype(float)
    dis_flags = ["PEDISDRS", "PEDISEAR", "PEDISEYE", "PEDISOUT", "PEDISPHY", "PEDISREM"]
    p["is_disabled"] = (
        pd.concat([num(c) == 1 for c in dis_flags], axis=1).any(axis=1)
    ).astype(bool)
    p["cps_race"] = num("race").astype(int)
    p["is_female"] = (num("sex") == 2).astype(bool)
    p["is_hispanic"] = (num("hispanic") == 1).astype(bool)
    p["is_household_head"] = num("A_EXPRRP").isin([1, 2]).astype(bool)
    p["is_separated"] = (num("A_MARITL") == 6).astype(bool)
    p["is_unmarried_partner_of_household_head"] = (
        num("A_EXPRRP") == 13
    ).astype(bool)
    # own children: count persons naming me as parent within the household.
    key = p["household_id"].astype(str)
    me = key + ":" + num("A_LINENO").astype(int).astype(str)
    par1 = key + ":" + num("PEPAR1").astype(int).astype(str)
    par2 = key + ":" + num("PEPAR2").astype(int).astype(str)
    counts = par1.value_counts().add(par2.value_counts(), fill_value=0)
    p["own_children_in_household"] = me.map(counts).fillna(0).astype(float)
    # household child counts (eCPS groups by household).
    u18 = (num("A_AGE") < 18).groupby(p["household_id"]).transform("sum")
    u6 = (num("A_AGE") < 6).groupby(p["household_id"]).transform("sum")
    p["count_under_18"] = u18.astype(float)
    p["count_under_6"] = u6.astype(float)
    # Social security split by RESNSS reason codes, age-62 fallback.
    ss = pd.to_numeric(p.get("social_security", 0), errors="coerce").fillna(0)
    r1, r2 = num("RESNSS1"), num("RESNSS2")
    retired = (r1 == 1) | (r2 == 1)
    disabled = ((r1 == 2) | (r2 == 2)) & ~retired
    survivor = (r1.isin([3, 5]) | r2.isin([3, 5])) & ~retired & ~disabled
    dependent = (
        (r1.isin([4, 6, 7]) | r2.isin([4, 6, 7]))
        & ~retired
        & ~disabled
        & ~survivor
    )
    unclassified = (ss > 0) & ~(retired | disabled | survivor | dependent)
    age = num("A_AGE")
    retired = retired | (unclassified & (age >= 62))
    disabled = disabled | (unclassified & (age < 62))
    p["social_security_retirement"] = (ss * retired).astype(float)
    p["social_security_disability"] = (ss * disabled).astype(float)
    p["social_security_survivors"] = (ss * survivor).astype(float)
    p["social_security_dependents"] = (ss * dependent).astype(float)
    # ---- v2 parity derivations (rules mirror usdata cps.py; V2_PLAN.md) ----
    p["educational_assistance"] = num("ED_VAL").astype(float)
    p["financial_assistance"] = num("FIN_VAL").astype(float)
    p["survivor_benefits"] = num("SRVS_VAL").astype(float)
    p["veterans_benefits"] = num("VET_VAL").astype(float)
    p["workers_compensation"] = num("WC_VAL").astype(float)
    oi_val, oi_off = num("OI_VAL"), num("OI_OFF")
    strike = oi_off == 12
    alimony_oi = oi_off == 20
    p["strike_benefits"] = (oi_val * strike).astype(float)
    p["miscellaneous_income"] = (
        oi_val * ~(strike | alimony_oi)
    ).astype(float)
    p["hours_worked_last_week"] = num("A_HRS1").clip(lower=0).astype(float)
    p["weeks_worked"] = num("WKSWORK").clip(0, 52).astype(float)
    p["detailed_occupation_recode"] = num("POCCU2").astype(float)
    # Weeks looking for work (LKWEEKS: -1 NIU -> 0), mirroring usdata.
    p["weeks_unemployed"] = num("LKWEEKS").clip(lower=0).astype(float)
    # v3 parity: ASEC raw derivations mirroring usdata cps.py exactly.
    p["is_blind"] = (num("PEDISEYE") == 1).astype(bool)
    p["is_surviving_spouse"] = (num("A_MARITL") == 4).astype(bool)
    p["is_full_time_college_student"] = (
        (num("A_HSCOL") == 2) & (num("A_FTPT") == 1)
    ).astype(bool)
    # Household head: person sequence 1 within the household (usdata P_SEQ rule).
    p["is_household_head"] = (num("P_SEQ") == 1).astype(bool)
    # Current health coverage at interview (ASEC NOW_* flags; 1 = covered).
    for _pe_name, _now in {
        "has_marketplace_health_coverage_at_interview": "NOW_MRK",
        "has_non_marketplace_direct_purchase_health_coverage_at_interview": "NOW_NONM",
        "has_medicaid_health_coverage_at_interview": "NOW_CAID",
        "has_other_means_tested_health_coverage_at_interview": "NOW_OTHMT",
        "has_tricare_health_coverage_at_interview": "NOW_MIL",
        "has_champva_health_coverage_at_interview": "NOW_CHAMPVA",
        "has_va_health_coverage_at_interview": "NOW_VACARE",
        "has_indian_health_service_coverage_at_interview": "NOW_IHSFLG",
    }.items():
        p[_pe_name] = (num(_now) == 1).astype(bool)
    # SPM-reported housing assistance (capped housing subsidy > 0).
    p["receives_housing_assistance"] = (num("SPM_CAPHOUSESUB") > 0).astype(bool)
    # FLSA overtime occupation flags from POCCU2 (codes shipped by the engine).
    from policyengine_us.data.cps import (
        CPS_FLSA_EXECUTIVE_ADMINISTRATIVE_PROFESSIONAL_OCCUPATION_CODES as _EXEC,
        CPS_FLSA_OVERTIME_OCCUPATION_CODES as _OCC,
    )

    for _flag, _code in _OCC.items():
        p[_flag] = (num("POCCU2") == _code).astype(bool)
    p["is_executive_administrative_professional"] = (
        num("POCCU2").isin(list(_EXEC)).astype(bool)
    )
    # RETCB proportional split (usdata cps.py:1505-1552; shares from
    # imputation_parameters.yaml — BEA/FRED + IRS SOI administrative shares).
    retcb = num("RETCB_VAL").clip(lower=0)
    has_wages = num("WSAL_VAL") > 0
    has_se = num("SEMP_VAL") > 0
    has_earned = has_wages | has_se
    se_pension = retcb * 0.046 * has_se
    p["self_employed_pension_contributions_desired"] = se_pension.astype(float)
    remaining = (retcb - se_pension).clip(lower=0)
    dc_pool = remaining * 0.908 * has_wages
    ira_pool = (remaining - dc_pool) * has_earned
    p["traditional_401k_contributions_desired"] = (dc_pool * 0.85).astype(float)
    p["roth_401k_contributions_desired"] = (dc_pool * 0.15).astype(float)
    p["traditional_ira_contributions_desired"] = (ira_pool * 0.392).astype(float)
    p["roth_ira_contributions_desired"] = (ira_pool * 0.608).astype(float)
    return p

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
    "student_loan_interest",
    "ira_deduction",
    "farm_income",
    # v2 parity fields (donor names from microplex puf.py field map)
    "alimony_expense",
    "casualty_loss",
    "domestic_production_ald",
    "educator_expense",
    "estate_income",
    "health_savings_account_ald",
    "long_term_capital_gains_on_collectibles",
    "unreimbursed_business_employee_expenses",
    "qualified_tuition_expenses",
    "business_is_sstb",
    "sstb_self_employment_income_would_be_qualified",
    "farm_rent_income",
    "self_employed_pension_contribution_ald",
    "unrecaptured_section_1250_gain",
    "puf_miscellaneous_income",
    "salt_refund_income",
    "investment_income_elected_form_4952",
    "capital_gains_distributions",
    # v3 QBI/partnership block (derived in the PUF loader from usdata rules)
    "partnership_se_income",
    "w2_wages_from_qualified_business",
    "unadjusted_basis_qualified_property",
    "sstb_self_employment_income",
    "sstb_w2_wages_from_qualified_business",
    "sstb_unadjusted_basis_qualified_property",
    "qualified_bdc_income",
    "qualified_reit_and_ptp_income",
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
    "ira_distributions": "taxable_ira_distributions",
    "alimony_received": "alimony_income",
    "charitable_cash": "charitable_cash_donations",
    "charitable_noncash": "charitable_non_cash_donations",
    "mortgage_interest_paid": "interest_deduction",
    "real_estate_tax_paid": "real_estate_taxes",
    "ordinary_dividend_income": "non_qualified_dividend_income",
    "ira_deduction": "traditional_ira_contributions",
    "puf_miscellaneous_income": "miscellaneous_income",
    "capital_gains_distributions": "non_sch_d_capital_gains",
    "sstb_self_employment_income": "sstb_self_employment_income_before_lsr",
}

SHARED_PREDICTORS = ("age", "is_joint", "n_people", "n_children")


def _build_imputation_steps(*, weighted: bool = True) -> list[dict]:
    """ASEC+PUF imputation graph over donor-available variables.

    weighted=True fits donors with the PUF design weight — the verified
    landmine fix (microplex#76): the PUF oversamples high incomes, so
    unweighted fits draw from the sample distribution (measured here:
    full-loss 130.1 vs eCPS 1.41, candidate LTCG $200T vs $257B).
    """
    puf_vars = list(dict.fromkeys(PUF_IMPUTE_VARS))
    puf_only = [v for v in puf_vars if v not in CPS_MEASURED]
    w = {"weights": "weight"} if weighted else {}
    return [
        {"onto": "synthetic_puf", "from": "puf", "vars": puf_vars,
         "order": "spine_first", **w},
        {"onto": "cps_keep", "from": "puf", "vars": puf_only,
         "condition_on": ["demographics", *CPS_MEASURED], **w},
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
    keep = ["household_id", "state_fips", "household_weight", "tenure_type"]
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
    xw["state_fips"] = xw["state_fips"].astype(int)
    block = pd.Series("", index=households.index, dtype=object)
    county = pd.Series("", index=households.index, dtype=object)
    cd = pd.Series(0, index=households.index, dtype="int64")
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
        geoid = chosen["geoid"].astype(str).to_numpy()
        block.loc[idx] = geoid
        county.loc[idx] = [g[:5] for g in geoid]
        district = (
            chosen["cd_id"]
            .astype(str)
            .str.extract(r"(\d+)$")[0]
            .fillna("0")
            .astype(int)
            .to_numpy()
        )
        cd.loc[idx] = int(state) * 100 + district
    households["block_geoid"] = block.astype(str)
    households["county_fips"] = county.astype(str)
    households["congressional_district_geoid"] = cd.astype(np.int32)
    return households


def _build_unit_map(
    base: pd.DataFrame, halves: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Original->engine unit-id mapping recovered via the halves' row index.

    The support spine re-identifies the synthetic half's tax_unit_id (they
    are new synthetic units) and zeroes its weights; the per-half frames
    keep the base frame's positional index, which is the join key back to
    the original ids.
    """
    min_id = int(base["tax_unit_id"].min())
    offset = int(base["tax_unit_id"].max()) - min(0, min_id) + 1
    pieces = []
    for half_name, frame in halves.items():
        new_ids = frame["tax_unit_id"].to_numpy()
        orig_ids = new_ids - offset if half_name == "synthetic_puf" else new_ids
        pieces.append(
            pd.DataFrame(
                {
                    "orig_tax_unit_id": orig_ids,
                    "new_tax_unit_id": new_ids,
                    "_half": half_name,
                }
            )
        )
    unit_map = pd.concat(pieces, ignore_index=True)
    base_ids = set(base["tax_unit_id"].tolist())
    recovered = set(unit_map["orig_tax_unit_id"].tolist())
    if unit_map["orig_tax_unit_id"].duplicated().any():
        raise ValueError("unit map has duplicated original tax unit ids")
    if recovered != base_ids:
        raise ValueError(
            "recovered original ids do not partition the base "
            f"(missing {len(base_ids - recovered)}, "
            f"extra {len(recovered - base_ids)})"
        )
    return unit_map


def _allocate_to_persons(
    person: pd.DataFrame,
    spine: pd.DataFrame,
    imputed_vars: list[str],
    unit_map: pd.DataFrame,
) -> pd.DataFrame:
    """Re-attach spine tax-unit values to persons via the unit map.

    cps_keep persons keep their ASEC person-level values for CPS-measured
    columns; PUF-only imputed amounts go to the unit head. synthetic_puf
    persons get every imputed variable head-allocated (others zero).
    """
    person = person.copy()
    m = unit_map.set_index("orig_tax_unit_id")
    person["_half"] = person["person_tax_unit_id"].map(m["_half"])
    person["new_tax_unit_id"] = person["person_tax_unit_id"].map(
        m["new_tax_unit_id"]
    )
    unmapped = int(person["_half"].isna().sum())
    if unmapped:
        raise ValueError(f"{unmapped} persons failed to map to a spine half")
    spine_idx = spine.set_index("tax_unit_id")
    is_head = person["tax_unit_role_input"] == "HEAD"
    synthetic = person["_half"] == "synthetic_puf"

    for var in imputed_vars:
        if var not in spine_idx.columns:
            continue
        unit_value = person["new_tax_unit_id"].map(spine_idx[var])
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
        default=OLD_WORKTREE / "artifacts/baselines/enhanced_cps_2024_hf_main.h5",
    )
    ap.add_argument(
        "--usdata-repo",
        type=Path,
        default=Path("~/.claude-worktrees/usdata-f7458313").expanduser(),
    )
    ap.add_argument(
        "--tail-units",
        type=int,
        default=15000,
        help="Top-income PUF returns carried verbatim as a tail-support "
        "stratum at design weights (0 disables).",
    )
    args = ap.parse_args()

    # The usdata package provides primary-source rules (tipped occupations,
    # QBI simulators, SCF/ACS loaders); make it importable for every stage.
    sys.path.insert(0, str(args.usdata_repo))

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
        extra_person_columns=(
            list(MICROUNIT_RAW_COLUMNS) + ["PH_SEQ"] + list(EXTRA_ASEC_COLUMNS)
        ),
        extra_household_columns=["H_TENURE", "GTCO"],
    )
    persons = ds.persons.to_pandas()
    households = ds.households.to_pandas()
    if "GTCO" in households.columns:
        # NYC boroughs by state*1000+county FIPS (usdata cps.py NYC list).
        _scf = (
            pd.to_numeric(households.get("state_fips", 0), errors="coerce").fillna(0)
            * 1000
            + pd.to_numeric(households["GTCO"], errors="coerce").fillna(0)
        )
        households["in_nyc"] = _scf.isin([36005, 36047, 36061, 36081, 36085]).astype(bool)
    if "H_TENURE" in households.columns:
        # Same base map as the eCPS loader (usdata cps.py:418).
        households["tenure_type"] = (
            pd.to_numeric(households["H_TENURE"], errors="coerce")
            .fillna(0)
            .map({0: "NONE", 1: "OWNED_WITH_MORTGAGE", 2: "RENTED", 3: "NONE"})
        )
    log(f"  persons={len(persons):,} households={len(households):,}")

    # ---- Stage B: unit structure ------------------------------------------
    log("stage B: unit assignment (microunit)")
    units = assign_us_unit_structure(persons, year=args.calendar_year)
    person = units.person.rename(
        columns={
            "wage_income": "employment_income",
            "interest_income": "taxable_interest_income",
        }
    )
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
    # Tail-support stratum: the top returns by income proxy, kept verbatim at
    # design weights BEFORE the population resample discards them. Free
    # support for top-bracket targets; calibration owns how much to use.
    income_proxy = sum(
        puf[c].clip(lower=0)
        for c in (
            "employment_income",
            "self_employment_income",
            "partnership_s_corp_income",
            "taxable_interest_income",
            "qualified_dividend_income",
            "ordinary_dividend_income",
            "long_term_capital_gains",
            "short_term_capital_gains",
            "taxable_pension_income",
        )
        if c in puf.columns
    )
    tail = (
        puf.loc[income_proxy.nlargest(min(args.tail_units, len(puf) // 20)).index].copy()
        if args.tail_units
        else puf.head(0).copy()
    )
    log(f"  tail stratum: {len(tail):,} units, weight {tail['weight'].sum()/1e6 if len(tail) else 0:.2f}M")

    # Population-resample the donor by its design weight: the PUF oversamples
    # high incomes, and microimpute's canonical Imputer silently ignores
    # weight_col (verified 2026-06-10), so importance-resampling is the
    # correct way to get population conditionals from the fit.
    rng = np.random.default_rng(args.seed)
    pw = puf["weight"].to_numpy(dtype=float)
    pw = pw / pw.sum()
    puf = puf.iloc[
        rng.choice(len(puf), size=len(puf), replace=True, p=pw)
    ].reset_index(drop=True)
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
    unit_map = _build_unit_map(base, dict(result.halves))
    orig_household_size = person.groupby("household_id").size()
    person = _allocate_to_persons(person, spine, imputed_vars, unit_map)
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
    person = _derive_person_columns(person)
    # Tipped-occupation codes via the eCPS's own mapping (usdata module).
    from policyengine_us_data.datasets.cps.tipped_occupation import (
        derive_is_tipped_occupation,
        derive_treasury_tipped_occupation_code,
    )

    ttoc = derive_treasury_tipped_occupation_code(person["PEIOOCC"])
    person["treasury_tipped_occupation_code"] = ttoc.astype(float)
    person["is_tipped_occupation"] = derive_is_tipped_occupation(ttoc)

    # Retirement-account distributions: ASEC DST_SC/DST_VAL pairs by source
    # code, split by the usdata taxable fractions (imputation_parameters.yaml).
    import yaml as _yaml

    _ipar = _yaml.safe_load(
        (args.usdata_repo / "policyengine_us_data" / "datasets" / "cps"
         / "imputation_parameters.yaml").read_text()
    )
    _codes = {1: "401k", 2: "403b", 6: "sep"}
    # Codes without a taxable split: keogh (5) is its own PE input; roth IRA
    # (3) is tax-exempt by assumption (usdata cps.py RETIREMENT_CODES).
    for _code, _pe in ((5, "keogh_distributions"), (3, "tax_exempt_ira_distributions")):
        _tot = 0
        for _i in ("1", "2", "1_YNG", "2_YNG"):
            _sc = pd.to_numeric(person.get(f"DST_SC{_i}", 0), errors="coerce").fillna(0)
            _val = pd.to_numeric(person.get(f"DST_VAL{_i}", 0), errors="coerce").fillna(0)
            _tot = _tot + (_sc == _code) * _val
        person[_pe] = pd.Series(_tot, index=person.index).astype(float)
    for _code, _name in _codes.items():
        _tot = 0
        for _i in ("1", "2", "1_YNG", "2_YNG"):
            _sc = pd.to_numeric(person.get(f"DST_SC{_i}", 0), errors="coerce").fillna(0)
            _val = pd.to_numeric(person.get(f"DST_VAL{_i}", 0), errors="coerce").fillna(0)
            _tot = _tot + (_sc == _code) * _val
        _frac = float(_ipar[f"taxable_{_name}_distribution_fraction"])
        person[f"taxable_{_name}_distributions"] = (_tot * _frac).astype(float)
        person[f"tax_exempt_{_name}_distributions"] = (_tot * (1 - _frac)).astype(float)
    log("  retirement distributions split (DST codes x usdata taxable fractions)")
    # Pre-response copies and aliases the contract requires alongside the
    # base variables.
    person["employment_income_before_lsr"] = person["employment_income"]
    person["self_employment_income_before_lsr"] = person[
        "self_employment_income"
    ]
    person["long_term_capital_gains_before_response"] = person[
        "long_term_capital_gains"
    ]
    person["taxable_unemployment_compensation"] = person[
        "unemployment_compensation"
    ]
    person["farm_operations_income"] = person["farm_income"]
    person["taxable_private_pension_income"] = person["taxable_pension_income"]
    person["tax_exempt_private_pension_income"] = person[
        "tax_exempt_pension_income"
    ]
    # Re-key every unit system per (original id, half): the synthetic half's
    # units are new synthetic entities, and a multi-unit household whose
    # units land in different halves splits into per-half export households.
    person["person_tax_unit_id"] = person["new_tax_unit_id"].astype(np.int64)
    for unit in ("spm_unit", "family", "marital_unit"):
        key = (
            person[f"person_{unit}_id"].astype(str) + "|" + person["_half"]
        )
        person[f"person_{unit}_id"] = (pd.factorize(key)[0] + 1).astype(
            np.int64
        )
    person["_orig_household_id"] = person["household_id"]
    hh_key = person["household_id"].astype(str) + "|" + person["_half"]
    person["person_household_id"] = (pd.factorize(hh_key)[0] + 1).astype(
        np.int64
    )
    person["person_id"] = np.arange(1, len(person) + 1, dtype=np.int64)
    person["age"] = person["A_AGE"].astype(float)
    # eCPS source flags: synthetic_puf half maps to the PUF-clone marker the
    # loss surface's nation/source/* household-count targets read.
    person["person_is_puf_clone"] = (person["_half"] == "synthetic_puf").astype(
        bool
    )

    # ---- Tail-support stratum persons ------------------------------------
    if len(tail):
        t = tail.reset_index(drop=True).copy()
        t = t.rename(columns={"gross_social_security": "social_security"})
        t = t.rename(columns=DONOR_TO_PE)
        if {"total_pension_income", "taxable_pension_income"} <= set(t.columns):
            t["tax_exempt_pension_income"] = (
                t["total_pension_income"] - t["taxable_pension_income"]
            ).clip(lower=0.0)
        if {
            "non_qualified_dividend_income",
            "qualified_dividend_income",
        } <= set(t.columns):
            t["non_qualified_dividend_income"] = (
                t["non_qualified_dividend_income"]
                - t["qualified_dividend_income"]
            ).clip(lower=0.0)
        n = len(t)
        joint = (t["filing_status"] == "JOINT").to_numpy()
        base_ids = {
            "tax": int(person["person_tax_unit_id"].max()),
            "hh": int(person["person_household_id"].max()),
            "spm": int(person["person_spm_unit_id"].max()),
            "fam": int(person["person_family_id"].max()),
            "mar": int(person["person_marital_unit_id"].max()),
            "pid": int(person["person_id"].max()),
        }
        head = pd.DataFrame(index=range(n))
        head["person_tax_unit_id"] = base_ids["tax"] + 1 + np.arange(n)
        head["person_household_id"] = base_ids["hh"] + 1 + np.arange(n)
        head["person_spm_unit_id"] = base_ids["spm"] + 1 + np.arange(n)
        head["person_family_id"] = base_ids["fam"] + 1 + np.arange(n)
        head["person_marital_unit_id"] = base_ids["mar"] + 1 + np.arange(n)
        head["age"] = (
            pd.to_numeric(t["age"], errors="coerce").fillna(50).clip(18, 85)
        )
        head["A_AGE"] = head["age"]
        head["tax_unit_role_input"] = "HEAD"
        head["_half"] = "tail_puf"
        value_cols = [
            c
            for c in t.columns
            if c in set(person.columns)
            and c
            not in ("age", "weight", "tax_unit_id", "filing_status", "_survey")
            and pd.api.types.is_numeric_dtype(t[c])
        ]
        for c in value_cols:
            head[c] = pd.to_numeric(t[c], errors="coerce").fillna(0.0)
        head["own_children_in_household"] = (
            pd.to_numeric(t.get("ctc_children", 0), errors="coerce")
            .fillna(0)
            .to_numpy()
        )
        head["count_under_18"] = head["own_children_in_household"]
        spouse = head[joint].copy()
        for c in value_cols:
            spouse[c] = 0.0
        spouse["tax_unit_role_input"] = "SPOUSE"
        spouse["own_children_in_household"] = 0.0
        tail_person = pd.concat([head, spouse], ignore_index=True)
        tail_person["person_id"] = (
            base_ids["pid"] + 1 + np.arange(len(tail_person))
        )
        tail_person["person_is_puf_clone"] = True
        for col in (
            "employment_income_before_lsr",
            "self_employment_income_before_lsr",
        ):
            src = col.replace("_before_lsr", "")
            if src in tail_person.columns:
                tail_person[col] = tail_person[src]
        if "long_term_capital_gains" in tail_person.columns:
            tail_person["long_term_capital_gains_before_response"] = (
                tail_person["long_term_capital_gains"]
            )
        if "unemployment_compensation" in tail_person.columns:
            tail_person["taxable_unemployment_compensation"] = tail_person[
                "unemployment_compensation"
            ]
        if "taxable_pension_income" in tail_person.columns:
            tail_person["taxable_private_pension_income"] = tail_person[
                "taxable_pension_income"
            ]
        if "tax_exempt_pension_income" in tail_person.columns:
            tail_person["tax_exempt_private_pension_income"] = tail_person[
                "tax_exempt_pension_income"
            ]
        if "farm_income" in tail_person.columns:
            tail_person["farm_operations_income"] = tail_person["farm_income"]
        tail_person = tail_person.reindex(columns=person.columns)
        for c in person.columns:
            if tail_person[c].isna().all():
                dt = person[c].dtype
                if pd.api.types.is_bool_dtype(dt):
                    tail_person[c] = False
                elif pd.api.types.is_numeric_dtype(dt):
                    tail_person[c] = 0
                else:
                    tail_person[c] = ""
        tail_person["_orig_household_id"] = np.nan
        log(
            f"  tail persons: {len(tail_person):,} "
            f"({int(joint.sum()):,} spouses added)"
        )
    else:
        tail_person = person.head(0)


    # Export households = (original household, half) pieces. Attributes come
    # from the original household; weight is prorated by the piece's person
    # share so person-mass totals are preserved exactly (full ASEC scale).
    piece = (
        person.groupby("person_household_id")
        .agg(
            _orig=("_orig_household_id", "first"),
            _half=("_half", "first"),
            _n=("person_id", "size"),
        )
        .reset_index()
        .rename(columns={"person_household_id": "household_id"})
    )
    hh_attrs = households.drop_duplicates("household_id").rename(
        columns={"household_id": "_orig", "household_weight": "_orig_weight"}
    )
    hh = piece.merge(hh_attrs, on="_orig", how="left")
    hh["household_weight"] = (
        hh["_orig_weight"]
        * hh["_n"]
        / hh["_orig"].map(orig_household_size).to_numpy()
    ).astype(float)
    hh["tract_geoid"] = hh["block_geoid"].astype(str).str[:11]
    hh["household_is_puf_clone"] = (hh["_half"] == "synthetic_puf").astype(bool)
    hh = hh.drop(columns=["_orig", "_half", "_n", "_orig_weight"])

    # Tail-stratum households: design weights, geography sampled from the
    # main household distribution (the PUF has no state).
    if len(tail):
        rng_t = np.random.default_rng(args.seed + 1)
        donor_geo = hh.sample(
            n=len(tail),
            replace=True,
            weights=hh["household_weight"].clip(lower=1e-9),
            random_state=int(rng_t.integers(0, 2**31)),
        ).reset_index(drop=True)
        tail_hh = donor_geo.copy()
        tail_hh["household_id"] = (
            tail_person["person_household_id"].unique()
        )
        tail_hh["household_weight"] = tail["weight"].to_numpy(dtype=float)
        tail_hh["household_is_puf_clone"] = True
        hh = pd.concat([hh, tail_hh], ignore_index=True)
        person = pd.concat([person, tail_person], ignore_index=True)
        log(f"  households incl. tail: {len(hh):,}")

    # ---- Stage F2: primary-source imputation (v3, eCPS-free) ---------------
    # Wealth from Fed SCF, tips from SIPP, wages from CPS-ORG. The enhanced
    # CPS is never a build input; it remains only the scoring benchmark.
    log("stage F2: primary-source imputation (SCF / SIPP / ORG)")
    sys.path.insert(0, str(args.usdata_repo))
    import primary_source_impute as psi

    person, hh = psi.add_scf_wealth(person, hh, seed=args.seed, log=log)
    person = psi.add_sipp_tips(person, log=log)
    person = psi.add_org_wages(person, hh, args.calendar_year, log=log)
    person = psi.add_meps_esi_premiums(person, log=log)
    person = psi.add_prior_year_income(person, args.asec_year, log=log)
    person = psi.add_mortgage_conversion(person, hh, args.calendar_year, log=log)
    person, hh = psi.add_acs_rent(person, hh, seed=args.seed, log=log)
    person, hh = psi.add_vehicle_assets(person, hh, log=log)

    def _group_clone_flag(id_col: str) -> pd.Series:
        share = person.groupby(person[id_col])["person_is_puf_clone"].mean()
        return share > 0.5

    clone_flags = {
        c: _group_clone_flag(f"person_{c.split('_is_')[0]}_id")
        for c in (
            "household_is_puf_clone",
            "tax_unit_is_puf_clone",
            "spm_unit_is_puf_clone",
            "family_is_puf_clone",
        )
    }

    def unit_table(id_col: str, source: pd.DataFrame | None = None) -> pd.DataFrame:
        ids = np.sort(person[f"person_{id_col}"].unique())
        t = pd.DataFrame({id_col: ids})
        if source is not None:
            extra = source.rename(columns={"TAX_ID": id_col})
            t = t.merge(extra, on=id_col, how="left")
        flag = f"{id_col.rsplit('_id', 1)[0]}_is_puf_clone"
        if flag in clone_flags:
            t[flag] = t[id_col].map(clone_flags[flag]).fillna(False).astype(bool)
        return t

    spm = unit_table("spm_unit_id")
    # SPM-record childcare expenses: raw ASEC columns, max per SPM unit
    # (constant within unit on the SPM record), mirroring the energy subsidy.
    for raw, pe_name in (
        ("SPM_CHILDCAREXPNS", "spm_unit_pre_subsidy_childcare_expenses"),
        ("SPM_CAPWKCCXPNS", "spm_unit_capped_work_childcare_expenses"),
    ):
        if raw in person.columns:
            agg = (
                pd.to_numeric(person[raw], errors="coerce")
                .fillna(0.0)
                .groupby(person["person_spm_unit_id"])
                .max()
            )
            spm[pe_name] = (
                spm["spm_unit_id"].map(agg).fillna(0.0).astype(float)
            )
    if "SPM_ENGVAL" in person.columns:
        eng = (
            pd.to_numeric(person["SPM_ENGVAL"], errors="coerce")
            .fillna(0.0)
            .groupby(person["person_spm_unit_id"])
            .max()
        )
        spm["spm_unit_energy_subsidy"] = (
            spm["spm_unit_id"].map(eng).fillna(0.0).astype(float)
        )

    class _Key:
        def __init__(self, value: str):
            self.value = value

    # Group-owned id columns must exist only on their group tables; the
    # person table carries them as join keys until this point.
    person = person.drop(
        columns=[
            c
            for c in (
                "household_id",
                "tax_unit_id",
                "spm_unit_id",
                "family_id",
                "marital_unit_id",
                "household_weight",
            )
            if c in person.columns
        ]
    )

    units_tu_new = (
        unit_map.rename(columns={"orig_tax_unit_id": "tax_unit_id"})
        .merge(units_tu, on="tax_unit_id", how="left")
        .drop(columns=["tax_unit_id", "_half"])
        .rename(columns={"new_tax_unit_id": "TAX_ID"})
    )
    # ---- v3: support guard anchored to the PUF's OWN realized ranges -------
    # (The v2 guard clipped to the eCPS baseline's ranges — an eCPS
    # contamination and, per the architecture review, the wrong reference.
    # Structural heavy-tail control belongs to signed calibration targets;
    # this clip only enforces the donor's own support.)
    from microplex.data_sources.puf import PUF_VARIABLE_MAP as PUF_FIELD_MAP
    from microplex.data_sources.puf import UPRATING_FACTORS

    _puf_csv = Path.home() / ".cache" / "microplex" / "puf_2015.csv"
    _header = pd.read_csv(_puf_csv, nrows=0).columns
    _by_upper = {c.upper(): c for c in _header}
    _want = {
        _by_upper[k.upper()]: v
        for k, v in PUF_FIELD_MAP.items()
        if k != "S006" and k.upper() in _by_upper
    }
    _puf_raw = pd.read_csv(_puf_csv, usecols=list(_want))
    _puf_raw.columns = [c for c in _puf_raw.columns]
    PUF_FIELD_MAP = {c: _want[c] for c in _want}
    _ranges: dict[str, tuple[float, float]] = {}
    for _raw, _donor in PUF_FIELD_MAP.items():
        if _raw == "S006" or _raw not in _puf_raw.columns:
            continue
        _up = UPRATING_FACTORS.get(_donor, 1.0)
        _v = pd.to_numeric(_puf_raw[_raw], errors="coerce").dropna()
        _pe = DONOR_TO_PE.get(_donor, _donor)
        _lo, _hi = float(_v.min()) * _up, float(_v.max()) * _up
        _ranges[_pe] = (min(_lo, _hi), max(_lo, _hi))
    if "_estate_income_gross" in _puf_raw.columns or True:
        # estate_income = E26390 - E26400 (uprated): bound by the rowwise diff.
        try:
            _est = pd.read_csv(
                Path.home() / ".cache" / "microplex" / "puf_2015.csv",
                usecols=["E26390", "E26400"],
            )
            _diff = (
                pd.to_numeric(_est["E26390"], errors="coerce").fillna(0)
                - pd.to_numeric(_est["E26400"], errors="coerce").fillna(0)
            ) * UPRATING_FACTORS.get("estate_income", 1.0)
            _ranges["estate_income"] = (float(_diff.min()), float(_diff.max()))
        except Exception:
            pass
    for _c in list(person.columns):
        if _c.startswith("person_") or _c not in _ranges:
            continue
        _vals = pd.to_numeric(person[_c], errors="coerce")
        if _vals.isna().all():
            continue
        _lo, _hi = _ranges[_c]
        _clipped = _vals.clip(_lo, _hi)
        _n = int((_clipped != _vals).sum())
        if _n:
            log(f"  puf-support-guard {_c}: clipped {_n} to [{_lo:,.0f}, {_hi:,.0f}]")
            person[_c] = _clipped

    # ---- v3: AOTC factual inputs from the PUF tuition signal ----------------
    # usdata extended_cps: with no credit signal, the AOTC student mask is
    # simply qualified_tuition_expenses > 0; the five factual eligibility
    # flags are set for those students.
    _aotc = (
        pd.to_numeric(
            person.get("qualified_tuition_expenses", 0), errors="coerce"
        ).fillna(0)
        > 0
    )
    for _flag in (
        "is_pursuing_credential_for_american_opportunity_credit",
        "attends_eligible_educational_institution_for_american_opportunity_credit",
        "is_enrolled_at_least_half_time_for_american_opportunity_credit",
        "has_american_opportunity_credit_1098_t_or_exception",
        "has_american_opportunity_credit_institution_ein",
    ):
        person[_flag] = _aotc.astype(bool)
    log(f"  AOTC factual inputs: {_aotc.mean()*100:.1f}% of persons flagged")
    # QRF-drawn boolean flags can land as mixed-object columns; normalize to
    # clean bools so HDF serialization and PE casting are deterministic.
    for _bcol in (
        "business_is_sstb",
        "sstb_self_employment_income_would_be_qualified",
        "self_employment_income_would_be_qualified",
    ):
        if _bcol in person.columns:
            person[_bcol] = (
                pd.to_numeric(person[_bcol], errors="coerce").fillna(0) > 0.5
            ).astype(bool)

    # ---- v2: place variables at their PolicyEngine entity ------------------
    # The PUF and donor stages leave tax-unit and SPM-entity amounts on the
    # person/household frames (head-carried); PE rejects inputs stored at the
    # wrong entity length, so move each to its owning table.
    from policyengine_us.system import system as _pe_system

    def _pe_entity(col: str) -> str | None:
        var = _pe_system.variables.get(col)
        return var.entity.key if var is not None else None

    tu_moves = [
        c for c in person.columns
        if not c.startswith("person_") and _pe_entity(c) == "tax_unit"
    ]
    # Aggregate person-stored tax-unit amounts per unit id; applied onto the
    # BUILT tax-unit table below (whose ids derive from persons, so the
    # aggregation covers every unit — attaching to units_tu_new would leave
    # NaNs for tail-stratum units absent from the spine unit map).
    tu_agg = {
        c: (
            pd.to_numeric(person[c], errors="coerce")
            .fillna(0.0)
            .groupby(person["person_tax_unit_id"])
            .sum()
        )
        for c in tu_moves
    }
    person = person.drop(columns=tu_moves)
    if tu_moves:
        log(f"  moved to tax_unit entity: {tu_moves}")
    person_spm_moves = [
        c for c in person.columns
        if not c.startswith("person_") and _pe_entity(c) == "spm_unit"
    ]
    spm_agg = {
        c: person[c].groupby(person["person_spm_unit_id"]).first()
        for c in person_spm_moves
    }
    person = person.drop(columns=person_spm_moves)
    if person_spm_moves:
        log(f"  moved person->spm_unit entity: {person_spm_moves}")
    spm_moves = [c for c in hh.columns if _pe_entity(c) == "spm_unit"]
    for c in spm_moves:
        val = dict(zip(hh["household_id"], hh[c]))
        per_person = person["person_household_id"].map(val)
        agg = per_person.groupby(person["person_spm_unit_id"]).first()
        spm[c] = spm["spm_unit_id"].map(agg).fillna(0.0).astype(float)
        hh = hh.drop(columns=[c])
    if spm_moves:
        log(f"  moved to spm_unit entity: {spm_moves}")
    for c, agg in spm_agg.items():
        mapped = spm["spm_unit_id"].map(agg)
        if pd.api.types.is_bool_dtype(person.dtypes.get(c, bool)) or mapped.dtype == object:
            spm[c] = mapped.fillna(False).astype(bool)
        else:
            spm[c] = pd.to_numeric(mapped, errors="coerce").fillna(0.0)

    tax_unit_tbl = unit_table("tax_unit_id", units_tu_new)
    for c, agg in tu_agg.items():
        tax_unit_tbl[c] = (
            tax_unit_tbl["tax_unit_id"].map(agg).fillna(0.0).astype(float)
        )
    entity_frames = {
        _Key("person"): person,
        _Key("household"): hh,
        _Key("tax_unit"): tax_unit_tbl,
        _Key("spm_unit"): spm,
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
    for column, value in V1_ZERO_DEFAULTS.items():
        defaults.setdefault(column, value)
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
    # Sibling export in the eCPS time-period layout ({variable}/{period}
    # datasets) — the format the comparison harness and HF artifacts use.
    import h5py

    tp_h5 = out / "candidate_timeperiod.h5"
    allowed = set(contract.required) | set(contract.optional)
    from policyengine_us.data import USSingleYearDataset

    saved = USSingleYearDataset(file_path=str(candidate_h5))
    saved_tables = [
        saved.person,
        saved.household,
        saved.tax_unit,
        saved.spm_unit,
        saved.family,
        saved.marital_unit,
    ]
    with h5py.File(tp_h5, "w") as handle:
        seen: set[str] = set()
        for frame in saved_tables:
            if frame is None or len(frame) == 0:
                continue
            for column in frame.columns:
                if column in seen or column not in allowed:
                    continue
                seen.add(column)
                values = frame[column].to_numpy()
                if values.dtype.kind in {"U", "O"}:
                    values = values.astype("S")
                elif values.dtype.kind == "b":
                    values = values.astype(bool)
                grp = handle.create_group(column)
                grp.create_dataset(str(args.calendar_year), data=values)
        missing_tp = sorted(set(contract.required) - seen)
    log(f"  time-period export: {tp_h5.name} cols={len(seen)} missing={len(missing_tp)}")
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
            str(out / "candidate_timeperiod.h5"),
            "--baseline-dataset",
            str(args.baseline_h5),
            "--output-dir",
            str(out / "sound_comparison"),
            "--period",
            str(args.calendar_year),
            "--force",
            "--policyengine-us-data-repo",
            str(args.usdata_repo),
            "--policyengine-us-data-python",
            str(args.usdata_repo / ".venv/bin/python"),
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
