"""Tests for generic Arch target derivations."""

from __future__ import annotations

from dataclasses import replace

from microplex.targets.arch_derivations import (
    ArchTargetRecord,
    component_sum_records,
    is_ssa_carry_forward_candidate,
    latest_carry_forward,
    ssa_carry_forward_rank,
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
