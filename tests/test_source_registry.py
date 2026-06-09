from __future__ import annotations

import pandas as pd
import pytest

from microplex.core import EntityType
from microplex.core.sources import (
    EntityObservation,
    ObservationFrame,
    Shareability,
    SourceDescriptor,
    StaticSourceProvider,
    TimeStructure,
)
from microplex.source_registry import SourceRegistry
from microplex.spec import load_spec_dict


def _frame() -> ObservationFrame:
    source = SourceDescriptor(
        name="cps_asec_2025_calendar_2024",
        shareability=Shareability.PUBLIC,
        time_structure=TimeStructure.CROSS_SECTION,
        observations=(
            EntityObservation(
                entity=EntityType.HOUSEHOLD,
                key_column="household_id",
                variable_names=("household_weight", "state_fips"),
                weight_column="household_weight",
            ),
            EntityObservation(
                entity=EntityType.PERSON,
                key_column="person_id",
                variable_names=("age", "employment_income"),
            ),
        ),
    )
    return ObservationFrame(
        source=source,
        tables={
            EntityType.HOUSEHOLD: pd.DataFrame(
                {
                    "household_id": [1, 2],
                    "household_weight": [10.0, 20.0],
                    "state_fips": ["06", "36"],
                }
            ),
            EntityType.PERSON: pd.DataFrame(
                {
                    "person_id": [11, 12],
                    "age": [30, 40],
                    "employment_income": [100.0, 200.0],
                }
            ),
        },
    )


def _spec(entity: str | None = "person") -> dict:
    source = {"dataset": "cps_asec_2025_calendar_2024", "role": "spine"}
    if entity is not None:
        source["entity"] = entity
    return {
        "meta": {"country": "us", "model_year": 2024},
        "sources": {"cps_asec": source},
        "spine": {
            "base": "cps_asec",
            "method": "support_spine",
            "support": {"seed": 42},
            "halves": [
                {"name": "keep", "keep": "all"},
                {"name": "synth", "strip_to": ["age"]},
            ],
        },
        "imputation": [],
    }


def test_source_registry_resolves_declared_entity_table() -> None:
    registry = SourceRegistry().register(
        "cps_asec_2025_calendar_2024",
        StaticSourceProvider(_frame()),
    )
    spec = load_spec_dict(_spec(entity="person"))

    frames = registry.resolve_sources(spec)

    assert list(frames) == ["cps_asec"]
    assert frames["cps_asec"].columns.tolist() == [
        "person_id",
        "age",
        "employment_income",
    ]


def test_source_registry_resolves_full_observation_frame() -> None:
    registry = SourceRegistry().register(
        "cps_asec_2025_calendar_2024",
        StaticSourceProvider(_frame()),
    )
    spec = load_spec_dict(_spec(entity="person"))

    frame = registry.resolve_observation_frame(spec, "cps_asec")

    assert set(frame.tables) == {EntityType.HOUSEHOLD, EntityType.PERSON}
    assert frame.tables[EntityType.HOUSEHOLD]["household_id"].tolist() == [1, 2]
    assert frame.tables[EntityType.PERSON]["person_id"].tolist() == [11, 12]


def test_source_registry_resolves_all_observation_frames() -> None:
    registry = SourceRegistry().register(
        "cps_asec_2025_calendar_2024",
        StaticSourceProvider(_frame()),
    )
    spec = load_spec_dict(_spec(entity="person"))

    frames = registry.resolve_observation_frames(spec)

    assert list(frames) == ["cps_asec"]
    assert set(frames["cps_asec"].tables) == {
        EntityType.HOUSEHOLD,
        EntityType.PERSON,
    }


def test_source_registry_uses_registered_default_entity() -> None:
    registry = SourceRegistry().register(
        "cps_asec_2025_calendar_2024",
        StaticSourceProvider(_frame()),
        default_entity=EntityType.HOUSEHOLD,
    )
    spec = load_spec_dict(_spec(entity=None))

    frames = registry.resolve_sources(spec)

    assert frames["cps_asec"].columns.tolist() == [
        "household_id",
        "household_weight",
        "state_fips",
    ]


def test_source_registry_requires_entity_for_multi_table_provider() -> None:
    registry = SourceRegistry().register(
        "cps_asec_2025_calendar_2024",
        StaticSourceProvider(_frame()),
    )
    spec = load_spec_dict(_spec(entity=None))

    with pytest.raises(ValueError, match="materialized multiple entity tables"):
        registry.resolve_sources(spec)


def test_source_registry_rejects_missing_dataset_provider() -> None:
    spec = load_spec_dict(_spec(entity="person"))

    with pytest.raises(KeyError, match="No SourceRegistry provider"):
        SourceRegistry().resolve_sources(spec)


def test_source_registry_rejects_unavailable_entity() -> None:
    registry = SourceRegistry().register(
        "cps_asec_2025_calendar_2024",
        StaticSourceProvider(_frame()),
    )
    spec = load_spec_dict(_spec(entity="tax_unit"))

    with pytest.raises(KeyError, match="requested entity"):
        registry.resolve_sources(spec)
