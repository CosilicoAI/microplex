"""Tests for neutral Arch target artifact helpers."""

from __future__ import annotations

import json

import pytest

from microplex.targets import (
    ArchConsumerFact,
    arch_consumer_fact_concept,
    arch_consumer_fact_constraints,
    arch_consumer_fact_geography_id,
    arch_consumer_fact_geography_level,
    arch_consumer_fact_numeric_value,
    arch_consumer_fact_period,
    arch_consumer_fact_source_record_id,
    arch_consumer_fact_target_type,
    arch_consumer_fact_to_target_record,
    load_arch_consumer_fact_jsonl_rows,
    load_arch_consumer_facts,
    load_arch_target_records,
)


def _consumer_fact(
    key: str,
    *,
    period: dict | None = None,
    aggregation: dict | None = None,
    concept_alignment: dict | None = None,
    geography: dict | None = None,
    universe_constraints: dict | None = None,
) -> dict:
    return {
        "schema_version": "arch.consumer_fact.v1",
        "aggregate_fact_key": f"arch.aggregate_fact.v2:{key}",
        "semantic_fact_key": f"arch.semantic_fact.v2:{key}",
        "value": "123.5",
        "period": period or {"type": "calendar_year", "value": 2024},
        "aggregation": aggregation or {"method": "sum"},
        "geography": geography or {"level": "country", "id": "0100000US", "name": "US"},
        "observed_measure": {
            "source_concept": "publisher.population",
            "source_name": "publisher",
            "source_table": "Table 1",
            "unit": "count",
        },
        "concept_alignment": concept_alignment or {},
        "source": {"source_name": "publisher", "source_table": "Table 1"},
        "lineage": {
            "source_record_id": f"publisher.{key}",
            "source_cell_keys": [f"arch.source_cell.v1:{key}"],
            "source_row_keys": [f"arch.source_row.v1:{key}"],
        },
        "universe_constraints": universe_constraints or {},
    }


def test_load_arch_consumer_facts_validates_and_parses_rows(tmp_path) -> None:
    path = tmp_path / "consumer_facts.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    _consumer_fact(
                        "state",
                        concept_alignment={
                            "canonical_concept": "canonical.population",
                        },
                    ),
                    sort_keys=True,
                ),
                "",
                json.dumps(
                    _consumer_fact(
                        "month",
                        period={"type": "month", "value": "2025-01"},
                    ),
                    sort_keys=True,
                ),
            ]
        )
        + "\n"
    )

    rows = load_arch_consumer_fact_jsonl_rows((path,), period=2024)
    facts = load_arch_consumer_facts((path,))

    assert len(rows) == 1
    assert len(facts) == 2
    assert isinstance(facts[0], ArchConsumerFact)
    assert facts[0].concept == "canonical.population"
    assert facts[0].period == 2024
    assert facts[0].value == 123.5
    assert facts[0].source_record_id == "publisher.state"
    assert facts[0].path == str(path)
    assert facts[0].line_number == 1
    assert facts[1].period == 2025


def test_arch_consumer_fact_accessors_fall_back_to_observed_concept() -> None:
    row = _consumer_fact("fallback")

    assert arch_consumer_fact_concept(row) == "publisher.population"
    assert arch_consumer_fact_target_type(row) == "AMOUNT"
    assert arch_consumer_fact_geography_level(row) == "NATIONAL"
    assert arch_consumer_fact_geography_id(row) is None
    assert arch_consumer_fact_constraints(row) == ()
    assert arch_consumer_fact_period(row) == 2024
    assert arch_consumer_fact_source_record_id(row) == "publisher.fallback"
    assert arch_consumer_fact_numeric_value("42") == 42


def test_arch_consumer_fact_to_target_record_preserves_constraints_and_lineage() -> (
    None
):
    row = _consumer_fact(
        "state-count",
        aggregation={"method": "count"},
        concept_alignment={
            "canonical_concept": "canonical.return_count",
            "authority": "arch",
            "relation": "exact",
            "notes": "aligned in source manifest",
            "evidence_url": "https://example.com/concepts",
        },
        geography={"level": "state", "id": "0400000US01", "name": "Alabama"},
        universe_constraints={
            "constraints": [
                {
                    "variable": "us:statutes/26/62#adjusted_gross_income",
                    "operator": "<",
                    "value": 1,
                    "unit": "usd",
                }
            ]
        },
    )

    record = arch_consumer_fact_to_target_record(ArchConsumerFact(row))

    assert record.variable == "canonical.return_count"
    assert record.target_type == "COUNT"
    assert record.period == 2024
    assert record.value == 123.5
    assert record.geographic_level == "STATE"
    assert record.geography_id == "0400000US01"
    assert record.constraints == (("us:statutes/26/62#adjusted_gross_income", "<", 1),)
    assert record.source == "publisher"
    assert record.source_table == "Table 1"
    assert record.unit == "count"
    assert record.source_record_id == "publisher.state-count"
    assert record.source_cell_keys == ("arch.source_cell.v1:state-count",)
    assert record.source_row_keys == ("arch.source_row.v1:state-count",)
    assert record.aggregate_fact_key == "arch.aggregate_fact.v2:state-count"
    assert record.semantic_fact_key == "arch.semantic_fact.v2:state-count"
    assert record.source_concept == "publisher.population"
    assert record.concept == "canonical.return_count"
    assert record.concept_relation == "exact"
    assert record.concept_authority == "arch"
    assert record.concept_evidence_notes == "aligned in source manifest"
    assert record.concept_evidence_url == "https://example.com/concepts"


def test_load_arch_target_records_filters_period_and_uses_injected_mappers(
    tmp_path,
) -> None:
    path = tmp_path / "consumer_facts.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(_consumer_fact("current"), sort_keys=True),
                json.dumps(
                    _consumer_fact(
                        "future",
                        period={"type": "calendar_year", "value": 2025},
                    ),
                    sort_keys=True,
                ),
            ]
        )
        + "\n"
    )

    records = load_arch_target_records(
        (path,),
        period=2024,
        variable_of=lambda fact: f"mapped.{fact.concept}",
        geography_id_of=lambda _fact: None,
    )

    assert len(records) == 1
    assert records[0].variable == "mapped.publisher.population"
    assert records[0].period == 2024
    assert records[0].geography_id is None


def test_load_arch_consumer_facts_rejects_wrong_schema(tmp_path) -> None:
    path = tmp_path / "consumer_facts.jsonl"
    row = _consumer_fact("bad")
    row["schema_version"] = "arch.consumer_fact.v0"
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="line 1"):
        load_arch_consumer_fact_jsonl_rows((path,))


def test_arch_consumer_fact_numeric_value_rejects_bool() -> None:
    with pytest.raises(ValueError, match="not numeric"):
        arch_consumer_fact_numeric_value(True)


def test_arch_consumer_fact_to_target_record_rejects_missing_variable() -> None:
    row = _consumer_fact("missing-variable")
    row["observed_measure"] = {}

    with pytest.raises(ValueError, match="no target variable"):
        arch_consumer_fact_to_target_record(ArchConsumerFact(row))


def test_arch_consumer_fact_constraints_reject_malformed_payload() -> None:
    row = _consumer_fact(
        "bad-constraints",
        universe_constraints={"constraints": [{"variable": "x"}]},
    )

    with pytest.raises(ValueError, match="missing variable, operator, or value"):
        arch_consumer_fact_constraints(row)
