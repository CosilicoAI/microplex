"""Post-calibration enrichment: simulation-dependent factual inputs.

Three eCPS layers need a baseline simulation on the finished artifact (their
rules read eligibility or credit aggregates), so they are written after the
calibrated artifact exists — mirroring usdata's own derivations exactly:

- ``takes_up_housing_assistance_if_eligible`` (SPM grain): reported
  recipients first, then a seeded fill to the national receipt rate among
  the simulation-eligible.
- ``is_pregnant`` (person): seeded state-rate draws for women 15-44
  (CDC/Census rates via usdata's etl).
- ``would_file_taxes_voluntarily`` (tax unit): usdata's demographic rate
  table over children/wage/age bins, excluding EITC claimants.

Weights are untouched; the script verifies them byte-identical after save.
"""

import sys
from pathlib import Path

import numpy as np

USDATA = Path.home() / ".claude-worktrees" / "usdata-populace"
sys.path.insert(0, str(USDATA))

ART = Path.home() / ".claude-worktrees" / "microplex-spec-build" / "artifacts"
OUT = ART / "populace_us_2024.h5"
OUT_TP = ART / "populace_us_2024_timeperiod.h5"
YEAR = 2024


def main() -> int:
    from policyengine_us import Microsimulation
    from policyengine_us.data import USSingleYearDataset
    from policyengine_us_data.datasets.cps.cps import (
        _voluntary_filing_age_bin,
        _voluntary_filing_children_bin,
        _voluntary_filing_rate_by_tax_unit,
        _voluntary_filing_wage_income_bin,
    )
    from policyengine_us_data.datasets.cps.takeup import (
        prioritize_reported_recipients,
    )
    from policyengine_us_data.db.etl_pregnancy import get_state_pregnancy_rates
    from policyengine_us_data.parameters import load_take_up_rate
    from policyengine_us_data.utils.randomness import seeded_rng

    ds = USSingleYearDataset(file_path=str(OUT))
    weights_before = np.asarray(ds.household["household_weight"], dtype=np.float64)
    sim = Microsimulation(dataset=str(OUT))

    # --- housing assistance take-up (SPM grain) ----------------------------
    rate = load_take_up_rate("housing_assistance", YEAR)
    reported = np.asarray(
        ds.spm_unit["receives_housing_assistance"], dtype=bool
    )
    eligible = np.asarray(
        sim.calculate("is_eligible_for_housing_assistance", YEAR).values,
        dtype=bool,
    )
    rng = seeded_rng("takes_up_housing_assistance_if_eligible")
    takes_up = prioritize_reported_recipients(
        reported, rate, rng.random(len(reported)), eligible_mask=eligible
    )
    ds.spm_unit["takes_up_housing_assistance_if_eligible"] = np.asarray(
        takes_up, dtype=bool
    )
    print(
        f"takes_up_housing_assistance_if_eligible: rate={rate:.3f} "
        f"reported nz={reported.mean()*100:.1f}% -> takeup nz={np.mean(takes_up)*100:.1f}%"
    )

    # --- pregnancy (person) -------------------------------------------------
    rates = get_state_pregnancy_rates(cdc_year=YEAR, acs_year=YEAR)
    national = 0.041
    state = np.asarray(
        sim.calculate("state_fips", YEAR, map_to="person").values, dtype=int
    )
    by_person = np.array([rates.get(int(s), national) for s in state])
    age = np.asarray(ds.person["age"], dtype=float)
    is_female = np.asarray(ds.person["is_female"], dtype=bool)
    eligible_preg = is_female & (age >= 15) & (age <= 44)
    rng = seeded_rng("is_pregnant")
    ds.person["is_pregnant"] = eligible_preg & (
        rng.random(len(age)) < by_person
    )
    print(f"is_pregnant: nz={np.mean(ds.person['is_pregnant'])*100:.2f}%")

    # --- voluntary filing (tax unit) ----------------------------------------
    voluntary_rates = load_take_up_rate("voluntary_filing", YEAR)
    takes_up_eitc = np.asarray(ds.tax_unit["takes_up_eitc"], dtype=bool)
    eitc = np.asarray(sim.calculate("eitc", YEAR).values, dtype=float)
    claims_eitc = takes_up_eitc & (eitc > 0)
    children = np.asarray(
        sim.calculate("tax_unit_child_dependents", YEAR).values, dtype=float
    )
    wage = np.asarray(
        sim.calculate("employment_income", YEAR, map_to="tax_unit").values,
        dtype=float,
    )
    age_head = np.asarray(sim.calculate("age_head", YEAR).values, dtype=float)
    rate_by_unit = _voluntary_filing_rate_by_tax_unit(
        voluntary_rates,
        _voluntary_filing_children_bin(children),
        _voluntary_filing_wage_income_bin(wage),
        _voluntary_filing_age_bin(age_head),
    )
    rng = seeded_rng("would_file_taxes_voluntarily")
    ds.tax_unit["would_file_taxes_voluntarily"] = ~claims_eitc & (
        rng.random(len(rate_by_unit)) < np.asarray(rate_by_unit, dtype=float)
    )
    print(
        "would_file_taxes_voluntarily: "
        f"nz={np.mean(ds.tax_unit['would_file_taxes_voluntarily'])*100:.2f}%"
    )

    ds.save(str(OUT))
    check = USSingleYearDataset(file_path=str(OUT))
    weights_after = np.asarray(
        check.household["household_weight"], dtype=np.float64
    )
    assert np.array_equal(weights_before, weights_after), "weights changed!"

    # propagate the three new layers into the timeperiod export
    import h5py

    with h5py.File(OUT_TP, "a") as f:
        for var, values in (
            (
                "takes_up_housing_assistance_if_eligible",
                np.asarray(
                    check.spm_unit["takes_up_housing_assistance_if_eligible"]
                ),
            ),
            ("is_pregnant", np.asarray(check.person["is_pregnant"])),
            (
                "would_file_taxes_voluntarily",
                np.asarray(check.tax_unit["would_file_taxes_voluntarily"]),
            ),
        ):
            key = f"{var}/{YEAR}"
            if key in f:
                del f[key]
            f.create_dataset(key, data=values)
    print("enriched + verified (weights byte-identical)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
