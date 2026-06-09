"""Tests for the Arch provider boundary adapter (ArchTargetRecord -> TargetSpec)."""

from __future__ import annotations

from microplex.core import EntityType
from microplex.targets.arch_derivations import ArchTargetRecord
from microplex.targets.arch_provider import (
    ArchPipelineConfig,
    ArchTargetProvider,
    arch_records_to_target_set,
    arch_target_record_to_target_spec,
    default_arch_target_name,
    latest_soi_records_by_composition,
    run_arch_derivation_pipeline,
)
from microplex.targets.provider import TargetQuery
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


def test_count_record_converts_to_sum_of_entity_count_measure():
    # a count is a sum of an entity-count measure (1 per record at that level)
    spec = arch_target_record_to_target_spec(
        _rec("returns", 50.0, target_type="COUNT"), entity="tax_unit"
    )
    assert spec.aggregation is TargetAggregation.SUM
    assert spec.measure == "tax_unit_count"
    assert spec.entity is EntityType.TAX_UNIT


def test_rate_target_type_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        arch_target_record_to_target_spec(
            _rec("poverty_rate", 0.1, target_type="RATE"), entity="person"
        )


def test_subnational_record_gets_geography_filter():
    spec = arch_target_record_to_target_spec(
        _rec("agi", 1.0, geographic_level="STATE", geography_id="06"),
        entity="tax_unit",
    )
    geo = [f for f in spec.filters if f.feature == "state_fips"]
    assert len(geo) == 1
    assert geo[0].value == "06"


def test_national_record_gets_no_geography_filter():
    spec = arch_target_record_to_target_spec(
        _rec("agi", 1.0, geographic_level="NATIONAL", geography_id=None),
        entity="tax_unit",
    )
    assert spec.filters == ()


def test_existing_geo_constraint_not_duplicated():
    spec = arch_target_record_to_target_spec(
        _rec(
            "agi",
            1.0,
            geographic_level="STATE",
            geography_id="06",
            constraints=(("state_fips", "==", "06"),),
        ),
        entity="tax_unit",
    )
    assert len([f for f in spec.filters if f.feature == "state_fips"]) == 1


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


# --- pipeline orchestrator + provider ---


def test_latest_soi_by_composition_keeps_latest_period():
    records = [
        _rec("agi", 100.0, period=2022),
        _rec("agi", 120.0, period=2024),
    ]
    out = latest_soi_records_by_composition(records, target_year=2024)
    assert len(out) == 1
    assert out[0].period == 2024
    assert out[0].value == 120.0


def test_pipeline_runs_component_sum_over_soi_records():
    records = [
        _rec("state_local_income_or_sales_tax_amount", 100.0),
        _rec("real_estate_taxes_amount", 40.0),
    ]
    config = ArchPipelineConfig(
        target_year=2024,
        component_sum_map={
            "salt_amount": (
                "state_local_income_or_sales_tax_amount",
                "real_estate_taxes_amount",
            )
        },
        age_soi=False,
    )
    out = run_arch_derivation_pipeline(records, config=config)
    assert any(r.variable == "salt_amount" and r.value == 140.0 for r in out)


def test_pipeline_skip_filter_applied():
    records = [_rec("agi", 100.0), _rec("dropme", 1.0)]
    config = ArchPipelineConfig(
        target_year=2024, age_soi=False, skip=lambda r: r.variable == "dropme"
    )
    out = run_arch_derivation_pipeline(records, config=config)
    assert all(r.variable != "dropme" for r in out)
    assert any(r.variable == "agi" for r in out)


def test_provider_produces_target_set_and_applies_query():
    records = [_rec("employment_income", 1000.0, period=2024)]
    config = ArchPipelineConfig(target_year=2024, age_soi=False)
    provider = ArchTargetProvider(
        records=records, config=config, entity_of=lambda v: "person"
    )
    target_set = provider.load_target_set()
    assert len(target_set.targets) == 1
    assert target_set.targets[0].measure == "employment_income"
    # query filtering by period
    assert len(provider.load_target_set(TargetQuery(period=2024)).targets) == 1
    assert len(provider.load_target_set(TargetQuery(period=1999)).targets) == 0
