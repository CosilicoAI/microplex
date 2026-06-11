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
    if "is_household_head" not in person.columns:
        raise RuntimeError(
            "head-carry requires is_household_head on the person frame "
            "(derive it from ASEC P_SEQ == 1 before stage F2); refusing to "
            "head-carry onto an all-False mask."
        )
    head = person["is_household_head"].astype(bool)
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

    def hh_sum(cols):
        present = [c for c in cols if c in hh.columns]
        return sum(hh[c].to_numpy(dtype=np.float64) for c in present) if present else None

    # The usdata target helpers impute PE-shaped names under an scf_ prefix
    # (scf_bank_account_assets etc.); person-entity assets head-carry from
    # those. Loud: a missing source column is a build bug, not a skip.
    PE_FROM_SCF = {
        "bank_account_assets": "scf_bank_account_assets",
        "stock_assets": "scf_stock_assets",
        "bond_assets": "scf_bond_assets",
    }
    head_carry_to_person = {}
    for pe_name, scf_name in PE_FROM_SCF.items():
        if scf_name not in hh.columns:
            raise RuntimeError(
                f"SCF stage expected imputed column {scf_name!r} for "
                f"{pe_name!r}; imputed scf columns: "
                f"{[c for c in hh.columns if c.startswith('scf_')][:8]}..."
            )
        head_carry_to_person[pe_name] = hh[scf_name].to_numpy(dtype=np.float64)
    # net worth = sum of imputed SCF components (usdata computes it the same way).
    # Net worth is usdata's own imputed measure — scf_net_worth already nets
    # assets against debts. Summing the scf_ component columns ON TOP of it
    # double-counts (the +34% miss on nation/net_worth/total); the vehicle
    # value joins later in the SIPP vehicle stage, mirroring usdata's
    # net_worth_components assembly.
    if "scf_net_worth" not in hh.columns:
        raise RuntimeError(
            "SCF stage expected imputed column 'scf_net_worth'; got "
            f"{[c for c in hh.columns if c.startswith('scf_')][:8]}..."
        )
    hh["net_worth"] = hh["scf_net_worth"].to_numpy(dtype=np.float64)
    # person-entity assets: head-carry onto persons (export entity mover is
    # person->tax_unit only; person columns export directly).
    if "is_household_head" not in person.columns:
        raise RuntimeError(
            "head-carry requires is_household_head on the person frame "
            "(derive it from ASEC P_SEQ == 1 before stage F2); refusing to "
            "head-carry onto an all-False mask."
        )
    headp = person["is_household_head"].astype(bool)
    hmap = dict(zip(hh["household_id"].tolist(), range(len(hh))))
    pidx = person["person_household_id"].map(hmap)
    for pe_name, vals in head_carry_to_person.items():
        person[pe_name] = np.where(
            headp & pidx.notna(), np.asarray(vals)[pidx.fillna(0).astype(int)], 0.0
        )
    log(f"  SCF wealth block: {len(targets)} scf vars + net_worth + {sorted(head_carry_to_person)} (weighted=wgt)")
    return person, hh


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


def add_mortgage_conversion(person: pd.DataFrame, hh: pd.DataFrame, year: int, log) -> pd.DataFrame:
    """Structural mortgages from SCF hints + PUF deductible interest.

    Ports usdata's two-step conversion (extended_cps.py:1183-1190): SCF-donor
    balance hints, then conversion of the PUF-imputed interest deduction into
    tax-unit mortgage balances/interest/origination plus person-level
    home_mortgage_interest and the investment_interest_expense residual.
    Failures raise — no silent zero fallbacks (charter rule).
    """
    from policyengine_us_data.utils.mortgage_interest import (
        convert_mortgage_interest_to_structural_inputs,
        impute_tax_unit_mortgage_balance_hints,
    )

    person = person.copy()
    tp = year
    p_tu = person["person_tax_unit_id"].to_numpy()
    tu_ids = np.sort(np.unique(p_tu))
    tu_index = {t: i for i, t in enumerate(tu_ids.tolist())}
    p_tu_idx = np.fromiter((tu_index[t] for t in p_tu.tolist()), dtype=np.int64)

    def person_col(name, default=0.0):
        return pd.to_numeric(person.get(name, default), errors="coerce").fillna(0.0).to_numpy(np.float32)

    # Tax-unit interest deduction from the head-carried person values.
    itd_person = person_col("interest_deduction")
    itd_tu = np.zeros(len(tu_ids), dtype=np.float32)
    np.add.at(itd_tu, p_tu_idx, itd_person)

    class _PersonZeroFallback(dict):
        """Missing person-grain reads return person-length zeros (logged)."""

        def __init__(self, n):
            super().__init__()
            self._n = n
            self.missed: set[str] = set()

        def get(self, key, default=None):
            if key in self:
                return super().__getitem__(key)
            self.missed.add(key)
            return {tp: np.zeros(self._n, dtype=np.float32)}

        def __getitem__(self, key):
            if key in self:
                return super().__getitem__(key)
            self.missed.add(key)
            return {tp: np.zeros(self._n, dtype=np.float32)}

    data = _PersonZeroFallback(len(person))
    data.update({
        "interest_deduction": {tp: itd_tu},
        "deductible_mortgage_interest": {tp: np.maximum(itd_person, 0)},
        "person_tax_unit_id": {tp: p_tu},
        "tax_unit_id": {tp: tu_ids},
        "person_id": {tp: person["person_id"].to_numpy()},
        "age": {tp: person_col("A_AGE" if "A_AGE" in person.columns else "age")},
        "employment_income": {tp: person_col("employment_income")},
        "self_employment_income": {tp: person_col("self_employment_income")},
        "social_security": {tp: person_col("social_security")},
        "taxable_pension_income": {tp: person_col("taxable_pension_income")},
        "taxable_interest_income": {tp: person_col("taxable_interest_income")},
        "is_female": {tp: person.get("is_female", pd.Series(False, index=person.index)).astype(bool).to_numpy()},
        "cps_race": {tp: person_col("cps_race")},
        "is_tax_unit_head": {tp: (person.get("tax_unit_role_input", "") == "HEAD").to_numpy()},
        "is_tax_unit_spouse": {tp: (person.get("tax_unit_role_input", "") == "SPOUSE").to_numpy()},
        "person_household_id": {tp: person["person_household_id"].to_numpy()},
        "person_spm_unit_id": {tp: person["person_spm_unit_id"].to_numpy()},
        "household_id": {tp: hh["household_id"].to_numpy()},
        "spm_unit_id": {tp: np.sort(person["person_spm_unit_id"].unique())},
        "tenure_type": {tp: hh.get("tenure_type", pd.Series("NONE", index=hh.index)).fillna("NONE").astype(str).to_numpy()},
    })
    # filing_status per tax unit: JOINT when a spouse is present, else SINGLE
    # (the converter uses it only for the mortgage debt-cap split).
    has_spouse_tu = np.zeros(len(tu_ids), dtype=bool)
    np.add.at(
        has_spouse_tu,
        p_tu_idx,
        (person.get("tax_unit_role_input", "") == "SPOUSE").to_numpy(),
    )
    data["filing_status"] = {tp: np.where(has_spouse_tu, "JOINT", "SINGLE")}
    # spm_unit_tenure_type: the SPM unit inherits its household's tenure.
    hh_tenure = dict(zip(hh["household_id"], data["tenure_type"][tp]))
    spm_ids = data["spm_unit_id"][tp]
    spm_hh = (
        pd.DataFrame({"spm": person["person_spm_unit_id"], "hh": person["person_household_id"]})
        .drop_duplicates("spm").set_index("spm")["hh"]
    )
    data["spm_unit_tenure_type"] = {tp: np.array([
        hh_tenure.get(spm_hh.get(sid), "NONE") for sid in spm_ids.tolist()
    ])}
    before = set(data)
    data = impute_tax_unit_mortgage_balance_hints(data, tp)
    data = convert_mortgage_interest_to_structural_inputs(data, tp)

    if data.missed:
        log(f"  mortgage converter zero-fallback keys: {sorted(data.missed)}")
    tu_outputs = []
    for key in set(data) - before | {"interest_deduction"}:
        arr = np.asarray(data[key][tp])
        if key.startswith("imputed_"):
            continue
        if len(arr) == len(person):
            person[key] = arr
        elif len(arr) == len(tu_ids):
            # Head-carry tax-unit outputs onto persons; the export-time entity
            # mover places them on the tax-unit table.
            head = (person.get("tax_unit_role_input", "") == "HEAD").to_numpy()
            vals = arr[p_tu_idx]
            person[key] = np.where(head, vals, 0.0)
            tu_outputs.append(key)
        else:
            raise ValueError(f"mortgage conversion output {key!r} has odd length {len(arr)}")
    log(f"  mortgage conversion: person outputs + head-carried {sorted(tu_outputs)}")
    inv = person.get("investment_interest_expense")
    if inv is not None:
        log(f"  investment_interest_expense: nz {(pd.to_numeric(inv, errors='coerce').fillna(0)>0).mean()*100:.1f}%")
    return person


def add_acs_rent(person: pd.DataFrame, hh: pd.DataFrame, seed: int, log):
    """Rent + vehicle ownership from the Census ACS 2022 artifact (usdata
    storage), imputed at household grain with ACS household weights;
    pre_subsidy_rent is person-entity and head-carried."""
    import h5py
    from microimpute import Imputer

    acs_path = (
        "/Users/maxghenis/.claude-worktrees/usdata-populace/"
        "policyengine_us_data/storage/acs_2022.h5"
    )
    with h5py.File(acs_path) as f:
        def col(name):
            v = f[name][:]
            return v
        # rent is stored at person grain in the ACS artifact (head-carried);
        # everything else here is household grain — aggregate before framing.
        d_pers = pd.DataFrame({
            "person_household_id": col("person_household_id"),
            "is_household_head": col("is_household_head").astype(bool),
            "employment_income": col("employment_income"),
            "rent": col("rent"),
        })
        d_hh = pd.DataFrame({
            "household_id": col("household_id"),
            "household_weight": col("household_weight"),
            "state_fips": col("household_state_fips"),
            "household_vehicles_owned": col("household_vehicles_owned"),
        })
    g = d_pers.groupby("person_household_id")
    d_hh = d_hh.merge(
        pd.DataFrame({
            "hh_employment_income": g["employment_income"].sum(),
            "hh_size": g.size(),
            "rent": g["rent"].sum(),
        }),
        left_on="household_id", right_index=True, how="left",
    ).fillna({"hh_employment_income": 0, "hh_size": 1, "rent": 0})
    d_hh = d_hh[d_hh["household_weight"] > 0]

    PRED = ["state_fips", "hh_employment_income", "hh_size"]
    fitted = Imputer(seed=seed, log_level="WARNING").fit(
        d_hh.dropna(subset=PRED + ["rent"]),
        PRED, ["rent"],
        weight_col="household_weight",
    )
    pg = person.groupby(person["person_household_id"])
    recv = pd.DataFrame({
        "hh_employment_income": pd.to_numeric(person["employment_income"], errors="coerce").fillna(0).groupby(person["person_household_id"]).sum(),
        "hh_size": pg.size(),
    })
    recv = recv.reindex(hh["household_id"]).fillna(0)
    recv["state_fips"] = pd.to_numeric(hh.get("state_fips", 0), errors="coerce").fillna(0).to_numpy()
    draws = fitted.predict(recv[PRED].reset_index(drop=True))

    hh = hh.copy()
    rent = _support_guard(np.asarray(draws["rent"], dtype=np.float64), d_hh["rent"].to_numpy(np.float64), "rent", log)
    # Rent applies to renter households only (tenure from the CPS H_TENURE map).
    tenure = hh.get("tenure_type", pd.Series("NONE", index=hh.index)).astype(str)
    rent = np.where(tenure.str.upper() == "RENTED", rent, 0.0)
    # pre_subsidy_rent is person-entity: head-carry the household rent.
    if "is_household_head" not in person.columns:
        raise RuntimeError(
            "head-carry requires is_household_head on the person frame "
            "(derive it from ASEC P_SEQ == 1 before stage F2); refusing to "
            "head-carry onto an all-False mask."
        )
    headp = person["is_household_head"].astype(bool)
    hmap = dict(zip(hh["household_id"].tolist(), range(len(hh))))
    pidx = person["person_household_id"].map(hmap)
    person = person.copy()
    person["pre_subsidy_rent"] = np.where(
        headp & pidx.notna(), rent[pidx.fillna(0).astype(int)], 0.0
    )
    log(f"  ACS rent: renter-hh rent nz {(rent>0).mean()*100:.1f}%")
    return person, hh


def add_vehicle_assets(person: pd.DataFrame, hh: pd.DataFrame, log):
    """Household vehicles (count + value) from the SIPP-trained QRF donor
    (usdata get_vehicle_model + receiver builder), mirroring usdata's
    auto-loan/vehicle imputation. Writes household grain."""
    from policyengine_us_data.datasets.sipp import get_vehicle_model
    from policyengine_us_data.utils.asset_imputation import (
        build_household_vehicle_receiver,
    )

    model = get_vehicle_model()
    # Build the receiver from a controlled column set: the person frame may
    # already carry a household_id-named column, and a duplicate name breaks
    # the builder's groupby.
    receiver_cols = [
        c
        for c in (
            "employment_income",
            "interest_income",
            "dividend_income",
            "interest_dividend_income",
            "rental_income",
            "age",
            "is_female",
            "is_married",
            "is_household_head",
        )
        if c in person.columns
    ]
    receiver_person = person[receiver_cols].copy()
    receiver_person["household_id"] = person["person_household_id"].to_numpy()
    tenure = hh.get("tenure_type")
    receiver = build_household_vehicle_receiver(
        receiver_person,
        tenure_type=(np.asarray(tenure) if tenure is not None else None),
    )
    pred = model.predict(X_test=receiver, mean_quantile=0.5)
    owned = np.clip(
        np.rint(np.asarray(pred["household_vehicles_owned"], dtype=np.float64)),
        0,
        None,
    )
    value = np.clip(
        np.asarray(pred["household_vehicles_value"], dtype=np.float64), 0, None
    )
    # receiver rows are one per household in hh order (builder groups by
    # household_id of the persons); align defensively by id.
    rid = np.asarray(receiver["household_id"]) if "household_id" in receiver else None
    hh = hh.copy()
    if rid is not None:
        owned_by = dict(zip(rid.tolist(), owned.tolist()))
        value_by = dict(zip(rid.tolist(), value.tolist()))
        hh["household_vehicles_owned"] = (
            hh["household_id"].map(owned_by).fillna(0.0).astype(float)
        )
        hh["household_vehicles_value"] = (
            hh["household_id"].map(value_by).fillna(0.0).astype(float)
        )
    else:
        if len(owned) != len(hh):
            raise RuntimeError(
                f"vehicle receiver rows ({len(owned)}) != households "
                f"({len(hh)}) and no household_id to align by."
            )
        hh["household_vehicles_owned"] = owned
        hh["household_vehicles_value"] = value
    # usdata folds the vehicle value into net worth (cps.py net_worth
    # components assembly); mirror that here, where the value is imputed.
    if "net_worth" in hh.columns:
        hh["net_worth"] = (
            hh["net_worth"].to_numpy(dtype=np.float64)
            + hh["household_vehicles_value"].to_numpy(dtype=np.float64)
        )
    log(
        f"  SIPP vehicles: owned nz {(hh['household_vehicles_owned']>0).mean()*100:.1f}%, "
        f"value nz {(hh['household_vehicles_value']>0).mean()*100:.1f}%, "
        f"folded into net_worth"
    )
    return person, hh
