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
    """Impute the wealth block onto households from the Fed SCF (weighted)."""
    from microimpute import Imputer
    from policyengine_us_data.datasets.scf.fed_scf import SummarizedFedSCF

    scf = SummarizedFedSCF().load()
    log(f"  SCF donor: {len(scf):,} rows, cols sample {list(scf.columns)[:8]}")
    # Predictors present on both sides (SCF is renamed to CPS-comparable names).
    cand_preds = [
        "age", "employment_income", "self_employment_income",
        "social_security", "is_female",
    ]
    # Household-grain receiver aggregates from persons.
    g = person.groupby(person["person_household_id"])
    recv = pd.DataFrame({
        "age": g.apply(lambda x: x.loc[x["is_household_head"].astype(bool), "A_AGE"].max() if x["is_household_head"].astype(bool).any() else x["A_AGE"].max()) if "A_AGE" in person.columns else g["age"].max(),
    })
    for c in ("employment_income", "self_employment_income", "social_security"):
        recv[c] = pd.to_numeric(person[c], errors="coerce").fillna(0).groupby(person["person_household_id"]).sum()
    recv["is_female"] = person.groupby(person["person_household_id"])["is_female"].first().astype(float) if "is_female" in person.columns else 0.0
    preds = [p for p in cand_preds if p in scf.columns and p in recv.columns]
    targets = [t for t in SCF_TARGETS if t in scf.columns]
    skipped = sorted(set(SCF_TARGETS) - set(targets))
    if skipped:
        log(f"  SCF donor lacks (skipping): {skipped}")
    wcol = next((c for c in ("household_weight", "weight", "wgt") if c in scf.columns), None)
    fitted = Imputer(seed=seed, log_level="WARNING").fit(
        scf.dropna(subset=preds + targets + ([wcol] if wcol else [])),
        preds, targets, weight_col=wcol,
    )
    draws = fitted.predict(recv[preds].copy())
    hh = hh.copy()
    hh_order = hh["household_id"]
    for t in targets:
        vals = pd.Series(np.asarray(draws[t]), index=recv.index)
        aligned = hh_order.map(vals).fillna(0.0).to_numpy(dtype=np.float64)
        hh[t] = _support_guard(aligned, scf[t].to_numpy(dtype=np.float64), t, log)
    if "household_vehicles_owned" in hh.columns:
        hh["household_vehicles_owned"] = hh["household_vehicles_owned"].clip(lower=0).round()
    log(f"  SCF wealth block: {len(targets)} variables imputed (weighted={bool(wcol)})")
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
    cps = {
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
    }
    try:
        add_org_labor_market_inputs(cps, year)
        person = person.copy()
        for out in ("hourly_wage", "is_paid_hourly", "is_union_member_or_covered",
                    "weekly_hours_worked_before_lsr", "fsla_overtime_premium"):
            if out in cps:
                person[out] = np.asarray(cps[out])
        log("  ORG labor-market inputs imputed")
    except Exception as exc:
        log(f"  ORG stage FAILED ({exc}); leaving defaults")
    return person
