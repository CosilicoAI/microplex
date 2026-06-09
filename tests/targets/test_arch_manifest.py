"""Tests for manifest-backed Arch consumer fact mapping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from microplex.core import EntityType
from microplex.targets.arch import ArchConsumerFact, load_arch_target_records
from microplex.targets.arch_manifest import (
    ArchTargetManifest,
    arch_target_provider_from_consumer_facts,
    load_arch_target_manifest,
    load_manifest_arch_target_records,
)
from microplex.targets.arch_provider import ArchTargetProvider

ROOT = Path(__file__).resolve().parents[2]
US_ARCH_MANIFEST = ROOT / "packs/us/manifests/arch_targets.json"


def _fact_row(
    concept: str,
    *,
    aggregation: str = "sum",
    canonical_concept: str | None = None,
    geography_level: str = "country",
    geography_id: str = "0100000US",
    constraints: list[dict[str, Any]] | None = None,
    period: int = 2024,
    value: float = 1.0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "aggregation": {"method": aggregation},
        "geography": {"id": geography_id, "level": geography_level},
        "observed_measure": {
            "source_concept": concept,
            "source_name": "irs_soi",
            "source_table": "fixture",
            "unit": "count" if aggregation == "count" else "usd",
        },
        "period": {"type": "tax_year", "value": period},
        "schema_version": "arch.consumer_fact.v1",
        "source": {"source_name": "irs_soi", "source_table": "fixture"},
        "universe_constraints": {"constraints": constraints or []},
        "value": value,
    }
    if canonical_concept is not None:
        row["concept_alignment"] = {
            "canonical_concept": canonical_concept,
            "source_concept": concept,
        }
    return row


def _manifest() -> ArchTargetManifest:
    return load_arch_target_manifest(US_ARCH_MANIFEST)


def test_us_arch_manifest_loads_and_pack_remains_content_only():
    manifest = _manifest()
    assert manifest.payload["country"] == "us"
    assert not list((ROOT / "packs/us").rglob("*.py"))


def test__given_us_arch_manifest__then_legacy_mapping_surface_is_expanded():
    manifest = _manifest()

    assert len(manifest.payload["target_mappings"]) >= 160
    assert len(manifest.payload["amount_measures"]) >= 60
    assert len(manifest.payload["count_aliases"]) >= 21
    assert (
        manifest.payload["target_mappings"]["irs_soi.total_income_tax"]["variable"]
        == "income_tax_liability"
    )
    assert (
        manifest.payload["target_mappings"]["cms_medicaid.total_chip_enrollment"][
            "target_type"
        ]
        == "SKIP"
    )
    assert (
        "irs_soi.returns_with_state_and_local_taxes"
        in manifest.payload["skip_concepts"]
    )


def test_manifest_maps_soi_state_agi_count_row_through_jsonl(tmp_path: Path):
    manifest = _manifest()
    path = tmp_path / "consumer_facts.jsonl"
    row = _fact_row(
        "irs_soi.individual_income_tax_returns",
        aggregation="count",
        geography_level="state",
        geography_id="0400000US01",
        constraints=[
            {
                "variable": "us:statutes/26/62#adjusted_gross_income",
                "operator": "<",
                "value": 1,
            }
        ],
        period=2022,
        value=32590,
    )
    path.write_text(json.dumps(row) + "\n")

    records = load_arch_target_records(
        [path],
        variable_of=manifest.variable_of,
        target_type_of=manifest.target_type_of,
        constraints_of=manifest.constraints_of,
        geography_level_of=manifest.geography_level_of,
        geography_id_of=manifest.geography_id_of,
    )

    assert len(records) == 1
    record = records[0]
    assert record.variable == "tax_unit_count"
    assert record.target_type == "COUNT"
    assert record.geographic_level == "STATE"
    assert record.geography_id is None
    assert ("adjusted_gross_income", "<", 1) in record.constraints
    assert ("state_fips", "==", "01") in record.constraints

    provider = ArchTargetProvider(
        records=records,
        config=manifest.pipeline_config(target_year=2022),
        entity_of=manifest.entity_of,
        measure_of=manifest.measure_of,
        geo_feature=manifest.geo_feature,
        count_measure=manifest.count_measure,
    )
    target = provider.load_target_set().targets[0]
    assert target.entity is EntityType.TAX_UNIT
    assert target.measure == "tax_unit_count"
    assert sum(filter_.feature == "state_fips" for filter_ in target.filters) == 1


def test__given_arch_source_suite_directory__then_manifest_loads_target_records(
    tmp_path: Path,
) -> None:
    suite_dir = tmp_path / "arch_source_suites" / "soi-table-1-2-2024"
    suite_dir.mkdir(parents=True)
    path = suite_dir / "consumer_facts.jsonl"
    path.write_text(
        json.dumps(
            _fact_row(
                "irs_soi.individual_income_tax_returns",
                aggregation="count",
                value=123,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = _manifest()

    records = load_manifest_arch_target_records(
        manifest, tmp_path / "arch_source_suites"
    )

    assert len(records) == 1
    assert records[0].variable == "tax_unit_count"
    assert records[0].target_type == "COUNT"


def test__given_arch_source_suite_directory__then_manifest_builds_provider(
    tmp_path: Path,
) -> None:
    suite_dir = tmp_path / "arch_source_suites" / "soi-table-1-2-2024"
    suite_dir.mkdir(parents=True)
    path = suite_dir / "consumer_facts.jsonl"
    path.write_text(
        json.dumps(
            _fact_row(
                "irs_soi.individual_income_tax_returns",
                aggregation="count",
                value=123,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    provider = arch_target_provider_from_consumer_facts(
        US_ARCH_MANIFEST,
        tmp_path / "arch_source_suites",
        target_year=2024,
    )
    target_set = provider.load_target_set()

    assert len(target_set.targets) == 1
    assert target_set.targets[0].measure == "tax_unit_count"


def test__given_empty_arch_source_suite_directory__then_manifest_loader_fails_closed(
    tmp_path: Path,
) -> None:
    suite_dir = tmp_path / "arch_source_suites"
    suite_dir.mkdir()

    import pytest

    with pytest.raises(FileNotFoundError, match="consumer_facts.jsonl"):
        load_manifest_arch_target_records(_manifest(), suite_dir)


def test_manifest_adds_legacy_positive_filter_for_count_alias():
    manifest = _manifest()
    fact = ArchConsumerFact(
        _fact_row("irs_soi.returns_with_total_wages", aggregation="count")
    )

    assert manifest.variable_of(fact) == "wages_salaries_returns"
    assert manifest.entity_of("wages_salaries_returns") is EntityType.TAX_UNIT
    assert ("employment_income", ">", 0) in manifest.constraints_of(fact)


def test__given_legacy_count_alias__then_manifest_adds_explicit_positive_filter():
    manifest = _manifest()
    fact = ArchConsumerFact(
        _fact_row("irs_soi.returns_with_premium_tax_credit", aggregation="count")
    )

    assert manifest.variable_of(fact) == "aca_ptc_returns"
    assert manifest.entity_of("aca_ptc_returns") is EntityType.TAX_UNIT
    assert manifest.count_measure(EntityType.TAX_UNIT) == "tax_unit_count"
    assert ("aca_ptc", ">", 0) in manifest.constraints_of(fact)


def test_manifest_maps_amount_variable_to_support_measure():
    manifest = _manifest()
    fact = ArchConsumerFact(_fact_row("irs_soi.total_wages"))

    row_record = manifest.variable_of(fact)
    assert row_record == "wages_salaries_amount"
    assert manifest.entity_of(row_record) is EntityType.PERSON
    assert manifest.measure_of(row_record) == "employment_income"

    records = load_arch_target_records_from_facts(manifest, [fact])
    provider = ArchTargetProvider(
        records=records,
        config=manifest.pipeline_config(target_year=2024),
        entity_of=manifest.entity_of,
        measure_of=manifest.measure_of,
        geo_feature=manifest.geo_feature,
        count_measure=manifest.count_measure,
    )
    target = provider.load_target_set().targets[0]
    assert target.entity is EntityType.PERSON
    assert target.measure == "employment_income"


def test__given_legacy_amount_alias__then_provider_uses_support_measure():
    manifest = _manifest()
    fact = ArchConsumerFact(_fact_row("irs_soi.total_income_tax"))

    row_record = manifest.variable_of(fact)
    assert row_record == "income_tax_liability"
    assert manifest.entity_of(row_record) is EntityType.TAX_UNIT
    assert manifest.measure_of(row_record) == "income_tax"

    records = load_arch_target_records_from_facts(manifest, [fact])
    provider = ArchTargetProvider(
        records=records,
        config=manifest.pipeline_config(target_year=2024),
        entity_of=manifest.entity_of,
        measure_of=manifest.measure_of,
        geo_feature=manifest.geo_feature,
        count_measure=manifest.count_measure,
    )
    target = provider.load_target_set().targets[0]
    assert target.entity is EntityType.TAX_UNIT
    assert target.measure == "income_tax"


def test__given_positive_amount_domain__then_filter_uses_measure_alias():
    manifest = _manifest()
    fact = ArchConsumerFact(_fact_row("irs_soi.taxable_net_capital_gains"))

    assert manifest.variable_of(fact) == "net_capital_gains_amount"
    assert manifest.measure_of("net_capital_gains_amount") == "net_capital_gains"
    assert ("net_capital_gains", ">", 0) in manifest.constraints_of(fact)


def test__given_positive_constraint_alias__then_manifest_maps_truthy_and_falsey():
    manifest = _manifest()
    receiving = ArchConsumerFact(
        _fact_row(
            "usda_snap.total_benefits",
            constraints=[
                {
                    "variable": "snap_receipt_status",
                    "operator": "eq",
                    "value": "receiving_food_stamps_snap",
                }
            ],
        )
    )
    not_receiving = ArchConsumerFact(
        _fact_row(
            "usda_snap.total_benefits",
            constraints=[
                {
                    "variable": "snap_receipt_status",
                    "operator": "eq",
                    "value": "not_receiving_food_stamps_snap",
                }
            ],
        )
    )

    assert ("snap", ">", 0) in manifest.constraints_of(receiving)
    assert ("snap", "==", 0) in manifest.constraints_of(not_receiving)


def test__given_ignored_constraint__then_manifest_drops_and_normalizes_filters():
    manifest = _manifest()
    fact = ArchConsumerFact(
        _fact_row(
            "irs_soi.adjusted_gross_income",
            constraints=[
                {
                    "variable": "amount_basis",
                    "operator": "=",
                    "value": "nominal",
                },
                {
                    "variable": "us:statutes/26/62#adjusted_gross_income",
                    "operator": "eq",
                    "value": 1,
                },
            ],
        )
    )

    constraints = manifest.constraints_of(fact)
    assert ("adjusted_gross_income", "==", 1) in constraints
    assert all(variable != "amount_basis" for variable, _, _ in constraints)


def test__given_unsupported_legacy_count__then_provider_skips_record():
    manifest = _manifest()
    fact = ArchConsumerFact(
        _fact_row("cms_medicaid.total_chip_enrollment", aggregation="count")
    )

    assert manifest.variable_of(fact) == "chip_total_enrollment"
    assert manifest.target_type_of(fact) == "SKIP"

    records = load_arch_target_records_from_facts(manifest, [fact])
    provider = ArchTargetProvider(
        records=records,
        config=manifest.pipeline_config(target_year=2024),
        entity_of=manifest.entity_of,
        measure_of=manifest.measure_of,
        geo_feature=manifest.geo_feature,
        count_measure=manifest.count_measure,
    )
    assert len(provider.load_target_set().targets) == 0


def test__given_declared_skip_concept__then_provider_skips_record():
    manifest = _manifest()
    fact = ArchConsumerFact(
        _fact_row("cbo.adjusted_gross_income_projection", aggregation="sum")
    )

    assert manifest.variable_of(fact) == "cbo.adjusted_gross_income_projection"
    assert manifest.target_type_of(fact) == "SKIP"

    records = load_arch_target_records_from_facts(manifest, [fact])
    provider = ArchTargetProvider(
        records=records,
        config=manifest.pipeline_config(target_year=2024),
        entity_of=manifest.entity_of,
        measure_of=manifest.measure_of,
        geo_feature=manifest.geo_feature,
        count_measure=manifest.count_measure,
    )
    assert len(provider.load_target_set().targets) == 0


def test_manifest_maps_state_cms_count_and_provider_uses_person_count():
    manifest = _manifest()
    fact = ArchConsumerFact(
        _fact_row(
            "cms_aca.marketplace_plan_selections",
            aggregation="count",
            canonical_concept="cms_aca.marketplace_plan_selections",
            geography_level="state",
            geography_id="0400000US02",
            period=2024,
            value=27464,
        )
    )
    mapped = manifest.variable_of(fact)
    assert mapped == "aca_marketplace_plan_selections"
    assert ("state_fips", "==", "02") in manifest.constraints_of(fact)

    records = load_arch_target_records_from_facts(manifest, [fact])
    provider = ArchTargetProvider(
        records=records,
        config=manifest.pipeline_config(target_year=2024),
        entity_of=manifest.entity_of,
        measure_of=manifest.measure_of,
        geo_feature=manifest.geo_feature,
        count_measure=manifest.count_measure,
    )
    target = provider.load_target_set().targets[0]
    assert target.entity is EntityType.PERSON
    assert target.measure == "person_count"
    assert any(
        filter_.feature == "state_fips" and filter_.value == "02"
        for filter_ in target.filters
    )


def test_manifest_rate_target_is_carried_but_not_consumed_by_default():
    manifest = _manifest()
    fact = ArchConsumerFact(
        _fact_row(
            "cms_aca.average_monthly_aptc",
            aggregation="mean",
            canonical_concept="cms_aca.average_monthly_aptc",
            geography_level="state",
            geography_id="0400000US02",
        )
    )
    records = load_arch_target_records_from_facts(manifest, [fact])
    assert records[0].target_type == "RATE"

    provider = ArchTargetProvider(
        records=records,
        config=manifest.pipeline_config(target_year=2024),
        entity_of=manifest.entity_of,
        measure_of=manifest.measure_of,
        geo_feature=manifest.geo_feature,
        count_measure=manifest.count_measure,
    )
    assert len(provider.load_target_set().targets) == 0


def test_strict_manifest_rejects_unmapped_concepts():
    payload = {
        "schema_version": "microplex.arch_targets.v1",
        "model_year": 2024,
        "require_target_mapping": True,
    }
    manifest = ArchTargetManifest.from_dict(payload)
    fact = ArchConsumerFact(_fact_row("unknown.concept"))

    import pytest

    with pytest.raises(ValueError):
        manifest.variable_of(fact)


def load_arch_target_records_from_facts(
    manifest: ArchTargetManifest, facts: list[ArchConsumerFact]
):
    from microplex.targets.arch import arch_consumer_fact_to_target_record

    return tuple(
        arch_consumer_fact_to_target_record(
            fact,
            variable_of=manifest.variable_of,
            target_type_of=manifest.target_type_of,
            constraints_of=manifest.constraints_of,
            geography_level_of=manifest.geography_level_of,
            geography_id_of=manifest.geography_id_of,
        )
        for fact in facts
    )
