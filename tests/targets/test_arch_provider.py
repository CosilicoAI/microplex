"""Tests for the Arch provider boundary adapter (ArchTargetRecord -> TargetSpec)."""

from __future__ import annotations

from microplex.core import EntityType
from microplex.targets.arch_derivations import ArchTargetRecord
from microplex.targets.arch_provider import (
    arch_records_to_target_set,
    arch_target_record_to_target_spec,
    default_arch_target_name,
)
from microplex.targets.spec import TargetAggregation


def _rec(variable, value, **kw) -> ArchTargetRecord:
    base = dict(
        variable=variable,
        period=2024,
        value=value,
        target_type="AMOUNT",
        geographic_level="NATIONAL",
        source="IRS_SOI",
    )
    base.update(kw)
    return ArchTargetRecord(**base)


def test_amount_record_converts_to_sum_target_with_measure():
    spec = arch_target_record_to_target_spec(
        _rec("employment_income", 1000.0, unit="usd", notes="n"), entity="person"
    )
    assert spec.aggregation is TargetAggregation.SUM
    assert spec.measure == "employment_income"
    assert spec.value == 1000.0
    assert spec.period == 2024
    assert spec.entity is EntityType.PERSON
    assert spec.source == "IRS_SOI"
    assert spec.units == "usd"
    assert spec.description == "n"
    assert spec.metadata["arch_variable"] == "employment_income"


def test_count_record_converts_to_count_target_without_measure():
    spec = arch_target_record_to_target_spec(
        _rec("tax_unit_count", 50.0, target_type="COUNT"), entity="tax_unit"
    )
    assert spec.aggregation is TargetAggregation.COUNT
    assert spec.measure is None
    assert spec.entity is EntityType.TAX_UNIT


def test_constraints_become_target_filters():
    spec = arch_target_record_to_target_spec(
        _rec("agi", 1.0, constraints=(("state_fips", "==", "06"),)),
        entity="tax_unit",
    )
    assert len(spec.filters) == 1
    assert spec.filters[0].feature == "state_fips"
    assert spec.filters[0].value == "06"


def test_explicit_name_overrides_default():
    rec = _rec("agi", 1.0, geography_id="06")
    assert default_arch_target_name(rec) == "IRS_SOI/agi/06"
    spec = arch_target_record_to_target_spec(rec, entity="tax_unit", name="custom")
    assert spec.name == "custom"


def test_lineage_preserved_in_metadata():
    spec = arch_target_record_to_target_spec(
        _rec(
            "agi",
            1.0,
            source_record_id="rid-1",
            concept="irs_soi.agi",
            source_table="soi_table",
            target_id=42,
        ),
        entity="tax_unit",
    )
    assert spec.metadata["source_record_id"] == "rid-1"
    assert spec.metadata["concept"] == "irs_soi.agi"
    assert spec.metadata["source_table"] == "soi_table"
    assert spec.metadata["target_id"] == 42


def test_records_to_target_set_applies_entity_and_skip():
    records = [
        _rec("employment_income", 1000.0),
        _rec("agi", 2000.0, target_type="COUNT"),
        _rec("dropme", 9.0),
    ]
    entity_of = {"employment_income": "person", "agi": "tax_unit", "dropme": "person"}
    target_set = arch_records_to_target_set(
        records,
        entity_of=lambda v: entity_of[v],
        skip=lambda r: r.variable == "dropme",
    )
    assert len(target_set.targets) == 2
    by_name = {t.name: t for t in target_set.targets}
    assert any(t.entity is EntityType.PERSON for t in target_set.targets)
    assert any(t.entity is EntityType.TAX_UNIT for t in target_set.targets)
    assert all("dropme" not in t.name for t in target_set.targets)
    assert by_name  # names are populated


def test_records_to_target_set_measure_override():
    records = [_rec("employment_income_amount", 1000.0)]
    target_set = arch_records_to_target_set(
        records,
        entity_of=lambda v: "person",
        measure_of=lambda v: "employment_income",  # measure differs from variable
    )
    assert target_set.targets[0].measure == "employment_income"
