"""Primary-source imputation stages (v3, eCPS-free).

Replaces ecps_donor_impute.py: every layer draws from its primary survey via
the usdata loaders in the worktree (Fed SCF for wealth, SIPP for tips, CPS-ORG
for hourly wage, MEPS-IC parameters for ESI premiums). The enhanced CPS
appears nowhere — it is only ever the benchmark in scoring. Each imputed
block is support-guarded to ITS OWN donor's realized per-record range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: SCF-sourced wealth block (household grain in the pool; donor is the
#: summarized Fed SCF with CPS-comparable predictor names).
SCF_TARGETS = [
    "net_worth",
    "scf_primary_residence_value",
    "scf_retirement_assets",
    "scf_business_equity",
    "scf_mortgage_debt",
    "scf_other_residential_real_estate",
    "scf_nonresidential_real_estate_equity",
    "scf_other_residential_debt",
    "scf_other_financial_assets",
    "scf_other_nonfinancial_assets",
    "scf_other_managed_assets",
    "scf_cash_value_life_insurance",
    "scf_certificates_of_deposit",
    "scf_savings_bonds",
    "scf_credit_card_debt",
    "scf_student_loan_debt",
    "scf_vehicle_installment_debt",
    "scf_other_installment_debt",
    "scf_other_lines_of_credit",
    "scf_other_debt",
    "household_vehicles_owned",
    "household_vehicles_value",
    "auto_loan_balance",
    "auto_loan_interest",
    "bank_account_assets",
    "bond_assets",
    "stock_assets",
]


def _support_guard(values: np.ndarray, donor: np.ndarray, name: str, log) -> np.ndarray:
    lo, hi = float(np.nanmin(donor)), float(np.nanmax(donor))
    clipped = np.clip(values, lo, hi)
    n = int((clipped != values).sum())
    if n:
        log(f"  support-guard {name}: clipped {n} to donor range [{lo:,.0f}, {hi:,.0f}]")
    return clipped


def add_scf_wealth(person: pd.DataFrame, hh: pd.DataFrame, seed: int, log) -> pd.DataFrame:
    """Impute the wealth block onto households from SCF 2022 (usdata blueprint).

    Mirrors usdata cps.py's own SCF stage: SCF_2022 donor, `wgt` weights, the
    same predictor list, target lists from the same helper functions; imputed
    at household-head grain and attached to households, support-guarded to the
    SCF's own realized ranges.
    """
    from microimpute import Imputer
    from policyengine_us_data.datasets.cps.cps import (
        add_scf_financial_asset_targets,
        add_scf_household_asset_targets,
        add_scf_net_worth_component_targets,
        add_scf_net_worth_target,
    )
    from policyengine_us_data.datasets.scf.scf import SCF_2022

    scf_raw = SCF_2022().load_dataset()
    scf = pd.DataFrame({k: scf_raw[k] for k in scf_raw.keys()})
    targets = list(
        dict.fromkeys(
            list(add_scf_net_worth_target(scf))
            + ["auto_loan_balance", "auto_loan_interest"]
            + list(add_scf_financial_asset_targets(scf))
            + list(add_scf_household_asset_targets(scf))
            + list(add_scf_net_worth_component_targets(scf))
        )
    )
    PREDICTORS = [
        "age", "is_female", "cps_race", "is_married",
        "own_children_in_household", "employment_income",
        "interest_dividend_income", "social_security_pension_income",
    ]
    log(f"  SCF 2022 donor: {len(scf):,} rows, {len(targets)} targets")

    num = lambda c: pd.to_numeric(person.get(c, 0), errors="coerce").fillna(0)  # noqa: E731
    head = person.get("is_household_head", pd.Series(False, index=person.index)).astype(bool)
    pf = pd.DataFrame(
        {
            "hh": person["person_household_id"],
            "head": head,
            "age": num("A_AGE") if "A_AGE" in person.columns else num("age"),
            "is_female": person.get("is_female", False),
            "cps_race": num("cps_race"),
            "is_married": person.get("A_MARITL", pd.Series(0, index=person.index)).isin([1, 2]).astype(float) if "A_MARITL" in person.columns else 0.0,
            "own_children_in_household": num("own_children_in_household"),
            "employment_income": num("employment_income"),
            "interest_dividend_income": num("taxable_interest_income") + num("dividend_income") + num("qualified_dividend_income") + num("non_qualified_dividend_income"),
            "social_security_pension_income": num("social_security") + num("taxable_pension_income"),
        }
    )
    heads = pf[pf["head"]].drop_duplicates("hh").set_index("hh")
    # Households with no flagged head: use the eldest member.
    missing = set(hh["household_id"]) - set(heads.index)
    if missing:
        eldest = (
            pf[pf["hh"].isin(missing)]
            .sort_values("age", ascending=False)
            .drop_duplicates("hh")
            .set_index("hh")
        )
        heads = pd.concat([heads, eldest])
    recv = heads.reindex(hh["household_id"]).fillna(0.0)

    donor_cols = [c for c in PREDICTORS if c in scf.columns]
    targets = [t for t in targets if t in scf.columns]
    donor = scf[donor_cols + targets + ["wgt"]].dropna()
    fitted = Imputer(seed=seed, log_level="WARNING").fit(
        donor, donor_cols, targets, weight_col="wgt"
    )
    draws = fitted.predict(recv[donor_cols].copy().reset_index(drop=True))
    hh = hh.copy()
    for t in targets:
        vals = np.asarray(draws[t], dtype=np.float64)
        hh[t] = _support_guard(vals, scf[t].to_numpy(dtype=np.float64), t, log)
    if "household_vehicles_owned" in hh.columns:
        hh["household_vehicles_owned"] = hh["household_vehicles_owned"].clip(lower=0).round()
    log(f"  SCF wealth block: {len(targets)} variables imputed (weighted=wgt)")
    return hh


def add_sipp_tips(person: pd.DataFrame, log) -> pd.DataFrame:
    """Tips from the SIPP-trained model (usdata get_tip_model)."""
    from policyengine_us_data.datasets.sipp import get_tip_model

    model = get_tip_model()
    x = pd.DataFrame(index=person.index)
    emp = pd.to_numeric(person["employment_income"], errors="coerce").fillna(0)
    x["employment_income"] = emp
    x["is_tipped_occupation"] = person.get("is_tipped_occupation", False)
    x["age"] = pd.to_numeric(person.get("age", person.get("A_AGE", 0)), errors="coerce").fillna(0)
    # usdata's call site builds pension/retirement/non-SSI aggregates first;
    # provide every feature the model declares, defaulting absent ones to 0.
    needed = list(getattr(model, "predictors", []) or [])
    for c in needed:
        if c not in x.columns:
            src = person.get(c)
            x[c] = pd.to_numeric(src, errors="coerce").fillna(0) if src is not None else 0.0
    try:
        person = person.copy()
        person["tip_income"] = np.asarray(
            model.predict(X_test=x, mean_quantile=0.5).tip_income.values
        )
        person.loc[~person.get("is_tipped_occupation", pd.Series(False, index=person.index)).astype(bool), "tip_income"] = 0.0
        log(f"  SIPP tips: nz {(person['tip_income']>0).mean()*100:.1f}%")
    except Exception as exc:
        log(f"  SIPP tips FAILED ({exc}); leaving zeros")
        person["tip_income"] = 0.0
    return person


def add_org_wages(person: pd.DataFrame, hh: pd.DataFrame, year: int, log) -> pd.DataFrame:
    """Hourly wage / hourly-pay status / overtime from CPS-ORG donors.

    usdata's add_org_labor_market_inputs operates on an h5-like mapping of
    arrays; a plain dict satisfies its read/write protocol.
    """
    from policyengine_us_data.datasets.cps.cps import add_org_labor_market_inputs

    hh_state = pd.to_numeric(hh.get("state_fips", 0), errors="coerce").fillna(0)

    class _ZeroFallback(dict):
        """h5-like mapping: unknown reads return zeros (logged once)."""

        def __init__(self, n, *a, **k):
            super().__init__(*a, **k)
            self._n = n
            self._missed = set()

        def __getitem__(self, key):
            if key in self:
                return super().__getitem__(key)
            if key not in self._missed:
                self._missed.add(key)
            return np.zeros(self._n, dtype=np.float32)

    n_persons = len(person)
    cps = _ZeroFallback(n_persons)
    cps.update({
        "age": pd.to_numeric(person.get("age", person.get("A_AGE", 0)), errors="coerce").fillna(0).to_numpy(np.float32),
        "household_id": hh["household_id"].to_numpy(np.int64),
        "person_household_id": person["person_household_id"].to_numpy(np.int64),
        "state_fips": hh_state.to_numpy(np.float32),
        "employment_income": pd.to_numeric(person["employment_income"], errors="coerce").fillna(0).to_numpy(np.float32),
        "is_female": person.get("is_female", pd.Series(False, index=person.index)).astype(bool).to_numpy(),
        "cps_race": pd.to_numeric(person.get("cps_race", 0), errors="coerce").fillna(0).to_numpy(np.float32),
        "weekly_hours_worked": pd.to_numeric(person.get("hours_worked_last_week", 0), errors="coerce").fillna(0).to_numpy(np.float32),
        "hours_worked_last_week": pd.to_numeric(person.get("hours_worked_last_week", 0), errors="coerce").fillna(0).to_numpy(np.float32),
        "weeks_worked": pd.to_numeric(person.get("weeks_worked", 0), errors="coerce").fillna(0).to_numpy(np.float32),
        "is_hispanic": person.get("is_hispanic", pd.Series(False, index=person.index)).astype(bool).to_numpy(),
    })
    # Occupation flags the ORG models read — pass real pool values when present.
    for flag in ("has_never_worked", "is_computer_scientist",
                 "is_executive_administrative_professional",
                 "is_farmer_fisher", "is_military"):
        if flag in person.columns:
            cps[flag] = person[flag].astype(bool).to_numpy()
    try:
        add_org_labor_market_inputs(cps, year)
        if cps._missed:
            log(f"  ORG zero-fallback keys: {sorted(cps._missed)}")
        person = person.copy()
        for out in ("hourly_wage", "is_paid_hourly", "is_union_member_or_covered",
                    "weekly_hours_worked_before_lsr", "fsla_overtime_premium"):
            if out in cps:
                person[out] = np.asarray(cps[out])
        log("  ORG labor-market inputs imputed")
    except Exception as exc:
        log(f"  ORG stage FAILED ({exc}); leaving defaults")
    return person


def add_meps_esi_premiums(person: pd.DataFrame, log) -> pd.DataFrame:
    """ESI premiums from MEPS-IC plan-type parameters (usdata rule, verbatim)."""
    from policyengine_us_data.datasets.cps.cps import (
        impute_employer_sponsored_insurance_premiums,
    )

    person = person.copy()
    person["employer_sponsored_insurance_premiums"] = (
        impute_employer_sponsored_insurance_premiums(person)
    )
    nz = float((person["employer_sponsored_insurance_premiums"] > 0).mean())
    log(f"  MEPS ESI premiums: nz {nz*100:.1f}%")
    return person


def add_prior_year_income(person: pd.DataFrame, asec_year: int, log) -> pd.DataFrame:
    """Prior-year earnings via the consecutive-ASEC PERIDNUM join (usdata rule).

    Maps last year's WSAL_VAL/SEMP_VAL onto matched persons; sentinel values
    {-1, -9999} mean unavailable.
    """
    from microplex.data_sources.cps import load_cps_asec

    person = person.copy()
    if "PERIDNUM" not in person.columns:
        log("  prior-year: PERIDNUM missing from pool; skipping")
        return person
    prior = load_cps_asec(
        year=asec_year - 1,
        extra_person_columns=["PERIDNUM", "WSAL_VAL", "SEMP_VAL"],
    ).persons.to_pandas()
    prior = prior.drop_duplicates("PERIDNUM").set_index("PERIDNUM")
    sentinels = {-1, -9999}
    cur_ids = person["PERIDNUM"]
    emp = cur_ids.map(prior["WSAL_VAL"]) if "WSAL_VAL" in prior.columns else pd.Series(np.nan, index=person.index)
    se = cur_ids.map(prior["SEMP_VAL"]) if "SEMP_VAL" in prior.columns else pd.Series(np.nan, index=person.index)
    matched = emp.notna() & se.notna() & ~emp.isin(sentinels) & ~se.isin(sentinels)
    person["employment_income_last_year"] = pd.to_numeric(emp, errors="coerce").where(matched, 0.0).fillna(0.0)
    person["self_employment_income_last_year"] = pd.to_numeric(se, errors="coerce").where(matched, 0.0).fillna(0.0)
    person["previous_year_income_available"] = matched.astype(bool)
    log(f"  prior-year join: matched {matched.mean()*100:.1f}% of persons (ASEC {asec_year-1})")
    return person
