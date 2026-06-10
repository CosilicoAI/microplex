"""eCPS-donor imputation stage (v2 parity).

Imputes the variable blocks the v1 spec defaulted to zero — SCF balance
sheets, mortgages, vehicles/auto loans, insurance premiums, tips, hourly
wage, prior-year income — by training weighted regime-gated QRF models on
the frozen enhanced-CPS baseline and drawing for the populace pool.

Using the incumbent as a donor is parity-legitimate: these blocks are
survey-imputed in the eCPS itself (SIPP/SCF/ACS models), and the comparison
gate stays the administrative target surface, which never sees donor values
directly.
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd

PERIOD = "2024"

#: Person-grain targets drawn from the donor, in chain order.
PERSON_TARGETS = [
    "employer_sponsored_insurance_premiums",
    "other_health_insurance_premiums",
    "tip_income",
    "hourly_wage",
    "is_paid_hourly",
    "weekly_hours_worked_before_lsr",
    "fsla_overtime_premium",
    "employment_income_last_year",
    "self_employment_income_last_year",
    "previous_year_income_available",
    "bank_account_assets",
    "bond_assets",
    "stock_assets",
]

#: Household-grain targets (SPM childcare is aggregated to household on the
#: donor side and broadcast back to SPM units at export).
HOUSEHOLD_TARGETS = [
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
    "first_home_mortgage_balance",
    "first_home_mortgage_interest",
    "first_home_mortgage_origination_year",
    "spm_unit_pre_subsidy_childcare_expenses",
]

PERSON_PREDICTORS = [
    "age",
    "is_female",
    "is_household_head",
    "employment_income",
    "self_employment_income",
    "social_security",
    "taxable_pension_income",
]

HOUSEHOLD_PREDICTORS = [
    "hh_size",
    "hh_employment_income",
    "hh_self_employment_income",
    "hh_social_security",
    "head_age",
    "tenure_type",
]


def _donor_frames(baseline_h5: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build donor person + household frames from the flat eCPS baseline."""
    with h5py.File(baseline_h5) as f:
        def col(name):
            v = f[name][PERIOD][:]
            if v.dtype.kind == "S":
                return np.char.decode(v.astype("S32"))
            return v

        keys = set(f.keys())

        def first_of(*names):
            for n in names:
                if n in keys:
                    return col(n)
            raise KeyError(f"none of {names} in donor")

        # The eCPS stores pre-LSR earnings and split social security; map to
        # the predictor names the pool uses.
        ss = sum(
            col(n)
            for n in (
                "social_security_retirement",
                "social_security_disability",
                "social_security_survivors",
                "social_security_dependents",
            )
            if n in keys
        )
        person = pd.DataFrame(
            {
                "person_household_id": col("person_household_id"),
                "age": col("age"),
                "is_female": col("is_female").astype(bool),
                "is_household_head": col("is_household_head").astype(bool),
                "employment_income": first_of(
                    "employment_income", "employment_income_before_lsr"
                ),
                "self_employment_income": first_of(
                    "self_employment_income", "self_employment_income_before_lsr"
                ),
                "social_security": ss,
                "taxable_pension_income": first_of(
                    "taxable_pension_income", "taxable_private_pension_income"
                ),
            }
        )
        for t in PERSON_TARGETS:
            person[t] = col(t)
        hh = pd.DataFrame(
            {
                "household_id": col("household_id"),
                "household_weight": col("household_weight"),
                "tenure_type": col("tenure_type"),
            }
        )
        hh_targets = [
            t for t in HOUSEHOLD_TARGETS
            if t != "spm_unit_pre_subsidy_childcare_expenses"
        ]
        for t in hh_targets:
            hh[t] = col(t)
        # SPM childcare -> household grain (first SPM unit per household).
        spm_cc = col("spm_unit_pre_subsidy_childcare_expenses")
        spm_id = col("spm_unit_id")
        p_spm = col("person_spm_unit_id")
        spm_map = dict(zip(spm_id.tolist(), spm_cc.tolist()))
        cc_by_hh = (
            pd.DataFrame(
                {
                    "hh": person["person_household_id"],
                    "cc": [spm_map.get(s, 0.0) for s in p_spm.tolist()],
                }
            )
            .groupby("hh")["cc"]
            .max()
        )
        hh["spm_unit_pre_subsidy_childcare_expenses"] = (
            hh["household_id"].map(cc_by_hh).fillna(0.0)
        )
    return person, hh


def _household_aggregates(person: pd.DataFrame, hh_id_col: str) -> pd.DataFrame:
    g = person.groupby(person[hh_id_col])
    out = pd.DataFrame(
        {
            "hh_size": g.size(),
            "hh_employment_income": g["employment_income"].sum(),
            "hh_self_employment_income": g["self_employment_income"].sum(),
            "hh_social_security": g["social_security"].sum(),
        }
    )
    head_age = (
        person[person["is_household_head"]]
        .groupby(person.loc[person["is_household_head"], hh_id_col])["age"]
        .max()
    )
    out["head_age"] = head_age.reindex(out.index).fillna(
        g["age"].max()
    )
    return out


def run(
    person: pd.DataFrame,
    hh: pd.DataFrame,
    baseline_h5: str,
    seed: int,
    log=print,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute the donor blocks onto the pool person/household frames."""
    from microimpute import Imputer

    d_person, d_hh = _donor_frames(baseline_h5)
    log(f"  donor: {len(d_person):,} persons, {len(d_hh):,} households")

    # Person weights = household weight of the person's household.
    wmap = dict(
        zip(d_hh["household_id"].tolist(), d_hh["household_weight"].tolist())
    )
    d_person["_w"] = (
        d_person["person_household_id"].map(wmap).fillna(0.0).astype(float)
    )

    # ---- person block -----------------------------------------------------
    recv_p = person.copy()
    for c in PERSON_PREDICTORS:
        if c not in recv_p.columns:
            raise ValueError(f"pool person frame missing predictor {c!r}")
        recv_p[c] = pd.to_numeric(recv_p[c], errors="coerce").fillna(0)
    fitted = Imputer(seed=seed, log_level="WARNING").fit(
        d_person,
        PERSON_PREDICTORS,
        PERSON_TARGETS,
        weight_col="_w",
    )
    draws = fitted.predict(recv_p[PERSON_PREDICTORS].copy())
    for t in PERSON_TARGETS:
        person[t] = np.asarray(draws[t])
    log(f"  person block imputed: {len(PERSON_TARGETS)} variables")

    # ---- household block ---------------------------------------------------
    d_aggr = _household_aggregates(d_person, "person_household_id")
    d_hh = d_hh.merge(
        d_aggr, left_on="household_id", right_index=True, how="left"
    ).fillna({c: 0 for c in d_aggr.columns})

    r_aggr = _household_aggregates(person, "person_household_id")
    recv_h = hh.merge(
        r_aggr, left_on="household_id", right_index=True, how="left"
    ).fillna({c: 0 for c in r_aggr.columns})
    if "tenure_type" not in recv_h.columns:
        raise ValueError("pool household frame missing tenure_type")
    recv_h["tenure_type"] = recv_h["tenure_type"].fillna("NONE").astype(str)
    d_hh["tenure_type"] = d_hh["tenure_type"].astype(str)

    fitted_h = Imputer(seed=seed + 1, log_level="WARNING").fit(
        d_hh,
        HOUSEHOLD_PREDICTORS,
        HOUSEHOLD_TARGETS,
        weight_col="household_weight",
    )
    draws_h = fitted_h.predict(recv_h[HOUSEHOLD_PREDICTORS].copy())
    for t in HOUSEHOLD_TARGETS:
        hh[t] = np.asarray(draws_h[t])
    # Vehicle counts and origination years are integral quantities.
    hh["household_vehicles_owned"] = (
        hh["household_vehicles_owned"].clip(lower=0).round()
    )
    hh["first_home_mortgage_origination_year"] = np.where(
        hh["first_home_mortgage_balance"] > 0,
        hh["first_home_mortgage_origination_year"].clip(1960, 2024).round(),
        0,
    )
    log(f"  household block imputed: {len(HOUSEHOLD_TARGETS)} variables")
    return person, hh
