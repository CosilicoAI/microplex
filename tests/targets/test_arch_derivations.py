"""Tests for generic Arch target derivations."""

from __future__ import annotations

from dataclasses import replace

from microplex.targets.arch_derivations import (
    ArchTargetRecord,
    SOIAgingFactors,
    age_soi_records,
    bea_national_wages_record,
    bea_state_employment_income_before_lsr,
    component_sum_records,
    default_total_scope,
    is_ssa_carry_forward_candidate,
    latest_carry_forward,
    soi_aging_factors,
    soi_amount_aging_factor,
    soi_count_aging_factor,
    ssa_carry_forward_rank,
    state_to_national_rollup,
    sum_state_records_to_national,
    with_component_sum_records,
)

SALT_MAP = {
    "salt_amount": (
        "state_local_income_or_sales_tax_amount",
        "real_estate_taxes_amount",
    ),
}


def _rec(variable: str, value: float, **kw) -> ArchTargetRecord:
    base = dict(
        variable=variable,
        period=2024,
        value=value,
        target_type="AMOUNT",
        geographic_level="STATE",
        geography_id="06",
        source="IRS_SOI",
        source_table="soi",
    )
    base.update(kw)
    return ArchTargetRecord(**base)


def test_component_sum_emits_composite_when_all_components_present():
    records = [
        _rec("state_local_income_or_sales_tax_amount", 100.0),
        _rec("real_estate_taxes_amount", 40.0),
    ]
    composites = component_sum_records(records, component_sum_map=SALT_MAP)
    assert len(composites) == 1
    composite = composites[0]
    assert composite.variable == "salt_amount"
    assert composite.value == 140.0
    assert composite.concept_relation == "sum_of_components"
    assert composite.concept_authority == "policyengine_us"
    # Synthesized records get deterministic negative ids and a tagged source id.
    assert composite.target_id < 0
    assert composite.source_record_id.startswith("microplex_component_sum:")


def test_with_component_sum_records_appends_composite():
    records = [
        _rec("state_local_income_or_sales_tax_amount", 100.0),
        _rec("real_estate_taxes_amount", 40.0),
    ]
    out = with_component_sum_records(records, component_sum_map=SALT_MAP)
    assert len(out) == 3
    assert out[:2] == records
    assert out[2].variable == "salt_amount"


def test_no_composite_when_components_incomplete():
    records = [_rec("state_local_income_or_sales_tax_amount", 100.0)]
    assert component_sum_records(records, component_sum_map=SALT_MAP) == []


def test_skip_when_output_already_exists_at_cell():
    records = [
        _rec("state_local_income_or_sales_tax_amount", 100.0),
        _rec("real_estate_taxes_amount", 40.0),
        _rec("salt_amount", 999.0),  # output already present at this cell
    ]
    # The existing salt_amount key matches the would-be composite key -> skip.
    assert component_sum_records(records, component_sum_map=SALT_MAP) == []


def test_duplicate_component_at_cell_drops_group():
    records = [
        _rec("state_local_income_or_sales_tax_amount", 100.0),
        _rec("state_local_income_or_sales_tax_amount", 50.0),  # duplicate component
        _rec("real_estate_taxes_amount", 40.0),
    ]
    # Ambiguous duplicate -> never double-count -> no composite.
    assert component_sum_records(records, component_sum_map=SALT_MAP) == []


def test_components_in_different_cells_do_not_merge():
    records = [
        _rec("state_local_income_or_sales_tax_amount", 100.0, geography_id="06"),
        _rec("real_estate_taxes_amount", 40.0, geography_id="36"),  # different state
    ]
    assert component_sum_records(records, component_sum_map=SALT_MAP) == []


def test_non_amount_records_ignored():
    records = [
        _rec("state_local_income_or_sales_tax_amount", 100.0, target_type="COUNT"),
        _rec("real_estate_taxes_amount", 40.0, target_type="COUNT"),
    ]
    assert component_sum_records(records, component_sum_map=SALT_MAP) == []


def test_distinct_periods_do_not_merge():
    records = [
        _rec("state_local_income_or_sales_tax_amount", 100.0, period=2024),
        _rec("real_estate_taxes_amount", 40.0, period=2023),
    ]
    assert component_sum_records(records, component_sum_map=SALT_MAP) == []


def test_value_sums_three_way_when_mapped():
    three_map = {"total_abc": ("a", "b", "c")}
    records = [_rec("a", 1.0), _rec("b", 2.0), _rec("c", 3.0)]
    composites = component_sum_records(records, component_sum_map=three_map)
    assert len(composites) == 1
    assert composites[0].variable == "total_abc"
    assert composites[0].value == 6.0


# --- latest carry-forward ---


def _cell(record: ArchTargetRecord):
    return (record.variable, record.geography_id)


def _carry(record: ArchTargetRecord, year: int) -> ArchTargetRecord:
    return replace(record, period=year)


def test_carry_forward_keeps_highest_rank_per_cell_and_carries_stale_forward():
    records = [
        _rec("ssi_total_payments", 10.0, period=2022, target_id=1),
        _rec("ssi_total_payments", 20.0, period=2023, target_id=2),
    ]
    out = latest_carry_forward(
        records,
        target_year=2024,
        is_candidate=lambda r: True,
        cell_key=_cell,
        rank=lambda r: (r.period,),
        carry_forward=_carry,
    )
    assert len(out) == 1
    assert out[0].value == 20.0  # latest period kept
    assert out[0].period == 2024  # stale 2023 carried forward


def test_carry_forward_keeps_target_year_record_unchanged():
    rec = _rec("ssi_total_payments", 5.0, period=2024)
    out = latest_carry_forward(
        [rec],
        target_year=2024,
        is_candidate=lambda r: True,
        cell_key=_cell,
        rank=lambda r: (r.period,),
        carry_forward=_carry,
    )
    assert out == [rec]


def test_carry_forward_skips_future_periods():
    rec = _rec("x", 1.0, period=2025)
    out = latest_carry_forward(
        [rec],
        target_year=2024,
        is_candidate=lambda r: True,
        cell_key=_cell,
        rank=lambda r: (r.period,),
        carry_forward=_carry,
    )
    assert out == []


def test_carry_forward_excludes_non_candidates_and_none_cells():
    a = _rec("a", 1.0, period=2023)
    b = _rec("b", 2.0, period=2023)
    out = latest_carry_forward(
        [a, b],
        target_year=2024,
        is_candidate=lambda r: r.variable == "a",
        cell_key=lambda r: None if r.variable == "a" else (r.variable,),
        rank=lambda r: (r.period,),
        carry_forward=_carry,
    )
    assert out == []  # a -> None cell, b -> not a candidate


def test_carry_forward_sort_key_orders_output():
    records = [
        _rec("z", 1.0, period=2024, geography_id="06"),
        _rec("a", 1.0, period=2024, geography_id="36"),
    ]
    out = latest_carry_forward(
        records,
        target_year=2024,
        is_candidate=lambda r: True,
        cell_key=_cell,
        rank=lambda r: (r.period,),
        carry_forward=_carry,
        sort_key=lambda r: r.variable,
    )
    assert [r.variable for r in out] == ["a", "z"]


def test_ssa_rank_prefers_latest_period_then_asr():
    older = _rec("x", 1.0, period=2022, target_id=1)
    newer = _rec("x", 1.0, period=2023, target_id=2)
    assert ssa_carry_forward_rank(newer) > ssa_carry_forward_rank(older)
    asr = _rec(
        "x",
        1.0,
        period=2023,
        source_table="Annual Statistical Report 2023",
        target_id=1,
    )
    plain = _rec("x", 1.0, period=2023, source_table="other", target_id=99)
    assert ssa_carry_forward_rank(asr) > ssa_carry_forward_rank(plain)


def test_is_ssa_candidate():
    vars_ = ["ssi_total_payments", "oasdi_beneficiaries"]
    assert is_ssa_carry_forward_candidate(
        _rec("ssi_total_payments", 1.0, source="SSA"), variables=vars_
    )
    assert not is_ssa_carry_forward_candidate(
        _rec("ssi_total_payments", 1.0, source="IRS_SOI"), variables=vars_
    )
    assert not is_ssa_carry_forward_candidate(
        _rec("other_var", 1.0, source="SSA"), variables=vars_
    )
    assert not is_ssa_carry_forward_candidate(
        _rec("ssi_total_payments", 1.0, source="SSA", target_type="RATIO"),
        variables=vars_,
    )


# --- state -> national rollup ---

REQUIRED = ("06", "36")


def _state(variable: str, value: float, fips: str, **kw) -> ArchTargetRecord:
    return _rec(variable, value, geographic_level="STATE", geography_id=fips, **kw)


def test_state_rollup_sums_complete_state_set_to_national():
    records = [_state("x", 100.0, "06"), _state("x", 40.0, "36")]
    out = state_to_national_rollup(
        records, required_states=REQUIRED, group_key=lambda r: r.variable
    )
    assert len(out) == 1
    nat = out[0]
    assert nat.value == 140.0
    assert nat.geographic_level is None
    assert nat.geography_id is None
    assert nat.target_id < 0
    assert nat.source_record_id.startswith("microplex_state_rollup:")


def test_state_rollup_skips_incomplete_state_set():
    records = [_state("x", 100.0, "06")]  # missing 36
    assert (
        state_to_national_rollup(
            records, required_states=REQUIRED, group_key=lambda r: r.variable
        )
        == []
    )


def test_state_rollup_skips_duplicate_state():
    records = [
        _state("x", 100.0, "06"),
        _state("x", 1.0, "06"),  # duplicate state
        _state("x", 40.0, "36"),
    ]
    assert (
        state_to_national_rollup(
            records, required_states=REQUIRED, group_key=lambda r: r.variable
        )
        == []
    )


def test_state_rollup_skips_when_national_already_exists():
    records = [
        _state("x", 100.0, "06"),
        _state("x", 40.0, "36"),
        _rec("x", 999.0, geographic_level="NATIONAL", geography_id=None),
    ]
    assert (
        state_to_national_rollup(
            records, required_states=REQUIRED, group_key=lambda r: r.variable
        )
        == []
    )


def test_state_rollup_ignores_states_outside_required_set():
    records = [
        _state("x", 100.0, "06"),
        _state("x", 40.0, "36"),
        _state("x", 5.0, "72"),  # PR excluded from required set
    ]
    out = state_to_national_rollup(
        records, required_states=REQUIRED, group_key=lambda r: r.variable
    )
    assert len(out) == 1
    assert out[0].value == 140.0  # PR (72) not summed


def test_state_rollup_ignores_non_state_records():
    records = [_rec("x", 100.0, geographic_level="COUNTY", geography_id="06001")]
    assert (
        state_to_national_rollup(
            records, required_states=REQUIRED, group_key=lambda r: r.variable
        )
        == []
    )


def test_state_rollup_builder_can_strip_constraints():
    constraints = (("region", "==", "z"),)
    records = [
        _state("x", 10.0, "06", constraints=constraints),
        _state("x", 20.0, "36", constraints=constraints),
    ]
    out = state_to_national_rollup(
        records,
        required_states=REQUIRED,
        group_key=lambda r: r.variable,
        build_national=lambda key, recs: sum_state_records_to_national(
            key, recs, non_state_constraints=lambda c: ()
        ),
    )
    assert out[0].constraints == ()


# --- SOI aging ---


def _ref(source: str, variable: str, period: int, value: float) -> ArchTargetRecord:
    return _rec(variable, value, source=source, period=period)


def test_age_soi_records_applies_factor_by_target_type_and_stamps_period():
    def stub_factors(refs, *, source_year, target_year, **kw):
        return SOIAgingFactors(source_year, target_year, 2.0, 3.0, "c", "a")

    records = [
        _rec("returns", 10.0, period=2020, target_type="COUNT"),
        _rec("agi", 100.0, period=2020, target_type="AMOUNT"),
    ]
    aged = age_soi_records(
        records, target_year=2024, reference_records=[], factors_for=stub_factors
    )
    by_var = {r.variable: r for r in aged}
    assert by_var["returns"].value == 20.0  # x2 count factor
    assert by_var["agi"].value == 300.0  # x3 amount factor
    assert all(r.period == 2024 for r in aged)
    assert all(r.source_period == 2020 for r in aged)
    assert all(r.aging_factors is not None for r in aged)


def test_age_soi_records_passes_through_same_year():
    rec = _rec("agi", 100.0, period=2024, target_type="AMOUNT")
    aged = age_soi_records([rec], target_year=2024, reference_records=[])
    assert aged == [rec]


def test_soi_count_factor_uses_bls_labor_force_ratio():
    refs = [
        _ref("BLS", "labor_force_count", 2020, 100.0),
        _ref("BLS", "labor_force_count", 2024, 110.0),
    ]
    factor, method = soi_count_aging_factor(refs, source_year=2020, target_year=2024)
    assert round(factor, 6) == 1.1
    assert method == "bls_labor_force_ratio"


def test_soi_count_factor_falls_back_to_cbo_for_target():
    refs = [
        _ref("BLS", "labor_force_count", 2020, 100.0),
        _ref("CBO", "labor_force", 2024, 120.0),  # no BLS at target -> CBO
    ]
    factor, method = soi_count_aging_factor(refs, source_year=2020, target_year=2024)
    assert round(factor, 6) == 1.2
    assert method == "cbo_labor_force_ratio"


def test_soi_count_factor_falls_back_to_soi_return_count():
    refs = [
        _ref("IRS_SOI", "tax_unit_count", 2020, 50.0),
        _ref("IRS_SOI", "tax_unit_count", 2024, 60.0),
    ]
    factor, method = soi_count_aging_factor(refs, source_year=2020, target_year=2024)
    assert round(factor, 6) == 1.2
    assert method == "soi_total_return_count_ratio"


def test_soi_count_factor_carry_forward_when_no_reference():
    factor, method = soi_count_aging_factor([], source_year=2020, target_year=2024)
    assert factor == 1.0
    assert "no_count_reference" in method


def test_soi_amount_factor_exact_agi_ratio():
    refs = [
        _ref("IRS_SOI", "adjusted_gross_income", 2020, 1000.0),
        _ref("IRS_SOI", "adjusted_gross_income", 2024, 1500.0),
    ]
    factor, method = soi_amount_aging_factor(refs, source_year=2020, target_year=2024)
    assert round(factor, 6) == 1.5
    assert method == "soi_total_agi_ratio"


def test_soi_amount_factor_extrapolates_when_target_year_absent():
    refs = [
        _ref("IRS_SOI", "adjusted_gross_income", 2020, 1000.0),
        _ref("IRS_SOI", "adjusted_gross_income", 2022, 1100.0),  # 1.1/yr growth
    ]
    factor, method = soi_amount_aging_factor(refs, source_year=2020, target_year=2024)
    # latest 2022 (1100), growth 1.1/yr, 2 yrs fwd -> 1331; /1000 source -> 1.331
    assert round(factor, 6) == 1.331
    assert method == "soi_total_agi_last_growth_extrapolation"


def test_soi_amount_factor_carry_forward_when_insufficient():
    refs = [_ref("IRS_SOI", "adjusted_gross_income", 2020, 1000.0)]
    factor, method = soi_amount_aging_factor(refs, source_year=2020, target_year=2024)
    assert factor == 1.0
    assert "no_amount_reference" in method


def test_soi_aging_factors_identity_same_year():
    factors = soi_aging_factors([], source_year=2024, target_year=2024)
    assert factors.count_factor == 1.0
    assert factors.amount_factor == 1.0
    assert factors.count_method == "identity"


def test_soi_aging_factors_not_required():
    factors = soi_aging_factors(
        [],
        source_year=2020,
        target_year=2024,
        needs_count_factor=False,
        needs_amount_factor=False,
    )
    assert factors.count_factor == 1.0
    assert factors.count_method == "not_required"
    assert factors.amount_method == "not_required"


def test_default_total_scope():
    assert default_total_scope(_rec("x", 1.0))  # no constraints
    assert default_total_scope(
        _rec("x", 1.0, constraints=(("is_tax_filer", "==", "1"),))
    )
    assert not default_total_scope(_rec("x", 1.0, constraints=(("age", ">=", "65"),)))


# --- BEA employment_income_before_lsr ---

WAGE_COMPONENTS = {
    "w": "wages",
    "s": "supplements",
    "c": "contributions",
    "ra": "residence_adjustment",
}
BEA_OUTPUT = "employment_income_before_lsr"


def _bea(variable: str, value: float, fips: str) -> ArchTargetRecord:
    return _rec(
        variable, value, source="BEA", geographic_level="STATE", geography_id=fips
    )


def _national_wages(value: float) -> ArchTargetRecord:
    return _rec(
        BEA_OUTPUT,
        value,
        source="BEA",
        geographic_level="NATIONAL",
        geography_id=None,
        concept="bea_nipa.wages_and_salaries",
    )


def _bea_state_records(fips, w, s, c, ra) -> list[ArchTargetRecord]:
    return [
        _bea("w", w, fips),
        _bea("s", s, fips),
        _bea("c", c, fips),
        _bea("ra", ra, fips),
    ]


def test_bea_synthesizes_residence_adjusted_scaled_to_national():
    records = [
        *_bea_state_records("06", 100.0, 20.0, 10.0, 5.0),
        *_bea_state_records("36", 200.0, 40.0, 20.0, 10.0),
    ]
    out = bea_state_employment_income_before_lsr(
        records,
        national_wages=_national_wages(400.0),
        required_states=("06", "36"),
        wage_component_variables=WAGE_COMPONENTS,
        output_variable=BEA_OUTPUT,
    )
    assert len(out) == 2
    assert all(r.variable == BEA_OUTPUT for r in out)
    assert {r.geography_id for r in out} == {"06", "36"}
    # residence-adjusted state values are scaled to match the national total.
    assert round(sum(r.value for r in out), 6) == 400.0
    assert all(r.value > 0 for r in out)
    assert all(r.concept == "policyengine_us.employment_income_before_lsr" for r in out)


def test_bea_returns_empty_when_state_missing_a_component():
    records = [
        *_bea_state_records("06", 100.0, 20.0, 10.0, 5.0),
        _bea("w", 200.0, "36"),  # 36 missing s/c/ra
        _bea("s", 40.0, "36"),
        _bea("c", 20.0, "36"),
    ]
    out = bea_state_employment_income_before_lsr(
        records,
        national_wages=_national_wages(400.0),
        required_states=("06", "36"),
        wage_component_variables=WAGE_COMPONENTS,
        output_variable=BEA_OUTPUT,
    )
    assert out == []


def test_bea_returns_empty_when_a_required_state_absent():
    records = _bea_state_records("06", 100.0, 20.0, 10.0, 5.0)  # only 06
    out = bea_state_employment_income_before_lsr(
        records,
        national_wages=_national_wages(400.0),
        required_states=("06", "36"),
        wage_component_variables=WAGE_COMPONENTS,
        output_variable=BEA_OUTPUT,
    )
    assert out == []


def test_bea_returns_empty_on_nonpositive_denominator():
    records = [
        *_bea_state_records("06", 0.0, 0.0, 0.0, 5.0),  # denom 0
        *_bea_state_records("36", 200.0, 40.0, 20.0, 10.0),
    ]
    out = bea_state_employment_income_before_lsr(
        records,
        national_wages=_national_wages(400.0),
        required_states=("06", "36"),
        wage_component_variables=WAGE_COMPONENTS,
        output_variable=BEA_OUTPUT,
    )
    assert out == []


def test_bea_national_wages_record_finds_by_concept():
    national = _national_wages(400.0)
    records = [
        national,
        _bea("w", 100.0, "06"),  # noise: state component
        _rec(BEA_OUTPUT, 999.0, source="IRS_SOI", geographic_level="NATIONAL"),
    ]
    found = bea_national_wages_record(records, output_variable=BEA_OUTPUT)
    assert found is national
    assert bea_national_wages_record([], output_variable=BEA_OUTPUT) is None
