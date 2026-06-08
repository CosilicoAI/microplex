from __future__ import annotations

import pandas as pd
import polars as pl

from microplex.core import EntityType
from microplex.core.sources import SourceQuery
from microplex.data_sources import us_registry
from microplex.data_sources.cps import (
    CPSAsecSourceProvider,
    CPSDataset,
    get_available_years,
    load_cps_asec,
)
from microplex.data_sources.puf import SHARED_VARS, PUFSourceProvider
from microplex.data_sources.us_registry import create_us_asec_puf_source_registry
from microplex.spec import load_spec_dict


def _cps_dataset(*, year: int, cache_dir=None, download: bool = True) -> CPSDataset:
    del cache_dir, download
    households = pl.DataFrame(
        {
            "household_id": [1, 2],
            "household_weight": [100.0, 200.0],
            "state_fips": [6, 36],
            "cbsa": [41860, 35620],
            "household_size": [2, 1],
            "household_total_income": [70_000.0, 20_000.0],
        }
    )
    persons = pl.DataFrame(
        {
            "household_id": [1, 1, 2],
            "person_number": [1, 2, 1],
            "age": [40, 38, 21],
            "sex": [1, 2, 2],
            "race": [1, 1, 2],
            "marital_status": [1, 1, 6],
            "wage_income": [50_000.0, 15_000.0, 20_000.0],
            "self_employment_income": [5_000.0, 0.0, 0.0],
            "interest_income": [100.0, 50.0, 0.0],
            "dividend_income": [200.0, 25.0, 0.0],
            "rental_income": [10.0, 3.0, 0.0],
            "social_security": [1000.0, 0.0, 500.0],
            "taxable_pension_income": [40.0, 5.0, 20.0],
            "unemployment_compensation": [0.0, 25.0, 10.0],
            "weight": [10.0, 10.0, 20.0],
            "is_child": [False, False, False],
            "is_adult": [True, True, True],
        }
    )
    return CPSDataset(
        persons=persons,
        households=households,
        year=year,
        source="test-cps",
    )


def _cps_dataset_out_of_order_ref(
    *,
    year: int,
    cache_dir=None,
    download: bool = True,
) -> CPSDataset:
    del cache_dir, download
    households = pl.DataFrame(
        {
            "household_id": [1],
            "household_weight": [100.0],
            "state_fips": [6],
            "cbsa": [41860],
        }
    )
    persons = pl.DataFrame(
        {
            "household_id": [1, 1],
            "person_number": [2, 1],
            "age": [38, 40],
            "sex": [2, 1],
            "race": [2, 1],
            "marital_status": [6, 1],
            "wage_income": [15_000.0, 50_000.0],
            "self_employment_income": [0.0, 5_000.0],
            "interest_income": [50.0, 100.0],
            "dividend_income": [25.0, 200.0],
            "social_security": [0.0, 1_000.0],
            "taxable_pension_income": [5.0, 40.0],
            "weight": [10.0, 10.0],
            "family_relationship": [2, 1],
        }
    )
    return CPSDataset(
        persons=persons,
        households=households,
        year=year,
        source="test-cps-out-of-order-ref",
    )


def _cps_dataset_raw_income_columns(
    *,
    year: int,
    cache_dir=None,
    download: bool = True,
) -> CPSDataset:
    del cache_dir, download
    households = pl.DataFrame(
        {
            "household_id": [1],
            "household_weight": [100.0],
            "state_fips": [6],
            "cbsa": [41860],
        }
    )
    persons = pl.DataFrame(
        {
            "household_id": [1, 1],
            "PH_SEQ": [1, 1],
            "A_LINENO": [1, 2],
            "A_AGE": [40, 38],
            "A_SEX": [1, 2],
            "PRDTRACE": [1, 1],
            "A_MARITL": [1, 1],
            "WSAL_VAL": [50_000.0, 15_000.0],
            "SEMP_VAL": [5_000.0, 0.0],
            "INT_VAL": [100.0, 50.0],
            "DIV_VAL": [200.0, 25.0],
            "SS_VAL": [1_000.0, 0.0],
            "PNSN_VAL": [40.0, 5.0],
            "A_FNLWGT": [10.0, 10.0],
        }
    )
    return CPSDataset(
        persons=persons,
        households=households,
        year=year,
        source="test-cps-raw-income-columns",
    )


def _cps_dataset_relationship_children(
    *,
    year: int,
    cache_dir=None,
    download: bool = True,
) -> CPSDataset:
    del cache_dir, download
    households = pl.DataFrame(
        {
            "household_id": [1],
            "household_weight": [100.0],
            "state_fips": [6],
            "cbsa": [41860],
        }
    )
    persons = pl.DataFrame(
        {
            "household_id": [1, 1, 1],
            "person_number": [1, 2, 3],
            "age": [40, 10, 12],
            "sex": [1, 2, 1],
            "race": [1, 1, 1],
            "marital_status": [6, 6, 6],
            "wage_income": [50_000.0, 0.0, 0.0],
            "self_employment_income": [0.0, 0.0, 0.0],
            "interest_income": [0.0, 0.0, 0.0],
            "dividend_income": [0.0, 0.0, 0.0],
            "social_security": [0.0, 0.0, 0.0],
            "taxable_pension_income": [0.0, 0.0, 0.0],
            "weight": [10.0, 10.0, 10.0],
            "is_child": [False, True, True],
            "family_relationship": [1, 3, 4],
        }
    )
    return CPSDataset(
        persons=persons,
        households=households,
        year=year,
        source="test-cps-relationship-children",
    )


def _puf_frame(*, target_year: int, expand_persons: bool = False, cache_dir=None):
    del cache_dir
    assert target_year == 2024
    assert expand_persons is False
    return pd.DataFrame(
        {
            "employment_income": [60_000.0, 10_000.0],
            "self_employment_income": [4_000.0, 0.0],
            "taxable_interest_income": [500.0, 0.0],
            "ordinary_dividend_income": [600.0, 0.0],
            "long_term_capital_gains": [5_000.0, 0.0],
            "rental_income": [700.0, 0.0],
            "gross_social_security": [800.0, 0.0],
            "taxable_pension_income": [300.0, 0.0],
            "unemployment_compensation": [900.0, 0.0],
            "age": [45, 30],
            "filing_status": ["JOINT", "SINGLE"],
            "weight": [75.0, 25.0],
        }
    )


def test_cps_provider_materializes_valid_entity_tables() -> None:
    provider = CPSAsecSourceProvider(
        asec_year=2025,
        calendar_year=2024,
        loader=_cps_dataset,
    )

    frame = provider.load_frame(SourceQuery(period=2024))

    assert set(frame.tables) == {
        EntityType.HOUSEHOLD,
        EntityType.PERSON,
        EntityType.TAX_UNIT,
    }
    tax_units = frame.tables[EntityType.TAX_UNIT]
    assert tax_units["tax_unit_id"].tolist() == [1, 2]
    assert tax_units["household_weight"].tolist() == [100.0, 200.0]
    assert tax_units["weight"].tolist() == [100.0, 200.0]
    assert tax_units["state_fips"].tolist() == [6, 36]
    assert tax_units["cbsa"].tolist() == [41860, 35620]
    assert tax_units["is_female"].tolist() == [False, True]
    assert tax_units["cps_race"].tolist() == [1, 2]
    assert tax_units["is_married"].tolist() == [True, False]
    assert tax_units["own_children_in_household"].tolist() == [0, 0]
    assert tax_units["employment_income"].tolist() == [65_000.0, 20_000.0]
    assert tax_units["self_employment_income"].tolist() == [5_000.0, 0.0]
    assert tax_units["taxable_interest_income"].tolist() == [150.0, 0.0]
    assert tax_units["ordinary_dividend_income"].tolist() == [225.0, 0.0]
    assert tax_units["rental_income"].tolist() == [13.0, 0.0]
    assert tax_units["gross_social_security"].tolist() == [1000.0, 500.0]
    assert tax_units["taxable_pension_income"].tolist() == [45.0, 20.0]
    assert tax_units["interest_dividend_income"].tolist() == [375.0, 0.0]
    assert tax_units["social_security_pension_income"].tolist() == [1045.0, 520.0]
    assert tax_units["unemployment_compensation"].tolist() == [25.0, 10.0]
    assert tax_units["year"].tolist() == [2024, 2024]
    assert {
        (relationship.parent_entity, relationship.child_entity)
        for relationship in frame.relationships
    } == {
        (EntityType.HOUSEHOLD, EntityType.PERSON),
        (EntityType.HOUSEHOLD, EntityType.TAX_UNIT),
    }


def test_cps_provider_uses_processed_person_number_for_reference_person() -> None:
    provider = CPSAsecSourceProvider(
        asec_year=2025,
        calendar_year=2024,
        loader=_cps_dataset_out_of_order_ref,
    )

    frame = provider.load_frame(SourceQuery(period=2024))

    tax_units = frame.tables[EntityType.TAX_UNIT]
    assert tax_units["age"].tolist() == [40]
    assert tax_units["is_female"].tolist() == [False]
    assert tax_units["cps_race"].tolist() == [1]
    assert tax_units["is_married"].tolist() == [True]


def test_cps_provider_preserves_raw_transform_income_predictor_sums() -> None:
    provider = CPSAsecSourceProvider(
        asec_year=2025,
        calendar_year=2024,
        loader=_cps_dataset_raw_income_columns,
    )

    frame = provider.load_frame(SourceQuery(period=2024))

    tax_units = frame.tables[EntityType.TAX_UNIT]
    assert tax_units["interest_dividend_income"].tolist() == [375.0]
    assert tax_units["social_security_pension_income"].tolist() == [1045.0]
    assert tax_units["agi_proxy"].tolist() == [70_375.0]


def test_cps_provider_counts_own_children_from_relationship_codes() -> None:
    provider = CPSAsecSourceProvider(
        asec_year=2025,
        calendar_year=2024,
        loader=_cps_dataset_relationship_children,
    )

    frame = provider.load_frame(SourceQuery(period=2024))

    tax_units = frame.tables[EntityType.TAX_UNIT]
    assert tax_units["own_children_in_household"].tolist() == [1]
    assert tax_units["ctc_qualifying_children"].tolist() == [1]


def test_puf_provider_materializes_valid_tax_unit_table() -> None:
    provider = PUFSourceProvider(target_year=2024, loader=_puf_frame)

    frame = provider.load_frame(SourceQuery(period=2024))

    table = frame.tables[EntityType.TAX_UNIT]
    assert table["tax_unit_id"].tolist() == [0, 1]
    assert table["employment_income"].tolist() == [60_000.0, 10_000.0]
    assert table["year"].tolist() == [2024, 2024]


def test_puf_provider_forwards_explicit_restricted_access_paths(tmp_path) -> None:
    puf_path = tmp_path / "puf_2015.csv"
    demographics_path = tmp_path / "demographics_2015.csv"
    puf_path.write_text("MARS,S006\n1,100\n")
    demographics_path.write_text("RECID,age\n1,40\n")
    captured = {}

    def _recording_puf_loader(**kwargs):
        captured.update(kwargs)
        return _puf_frame(
            target_year=kwargs["target_year"],
            expand_persons=kwargs["expand_persons"],
            cache_dir=kwargs["cache_dir"],
        )

    provider = PUFSourceProvider(
        target_year=2024,
        puf_path=puf_path,
        demographics_path=demographics_path,
        loader=_recording_puf_loader,
    )

    provider.load_frame(SourceQuery(period=2024))

    assert captured["puf_path"] == puf_path
    assert captured["demographics_path"] == demographics_path


def test_us_asec_puf_registry_uses_spec_dataset_ids(monkeypatch) -> None:
    monkeypatch.setattr(
        "microplex.data_sources.us_registry.CPSAsecSourceProvider",
        lambda **kwargs: CPSAsecSourceProvider(loader=_cps_dataset, **kwargs),
    )
    monkeypatch.setattr(
        "microplex.data_sources.us_registry.PUFSourceProvider",
        lambda **kwargs: PUFSourceProvider(loader=_puf_frame, **kwargs),
    )
    registry = create_us_asec_puf_source_registry(download_cps=False)
    spec = load_spec_dict(
        {
            "meta": {"country": "us", "model_year": 2024},
            "sources": {
                "cps_asec": {
                    "dataset": "cps_asec_2025_calendar_2024",
                    "role": "spine",
                },
                "puf": {"dataset": "puf_2024", "role": "donor"},
            },
            "spine": {
                "base": "cps_asec",
                "method": "support_spine",
                "support": {"seed": 42},
                "halves": [
                    {"name": "cps_keep", "keep": "all"},
                    {"name": "synthetic_puf", "strip_to": ["tax_unit_id"]},
                ],
            },
            "imputation": [],
        }
    )

    frames = registry.resolve_sources(spec)

    assert set(frames) == {"cps_asec", "puf"}
    assert frames["cps_asec"]["tax_unit_id"].tolist() == [1, 2]
    assert frames["puf"]["tax_unit_id"].tolist() == [0, 1]
    assert "employment_income" in frames["cps_asec"]
    assert "employment_income" in frames["puf"]


def test_registry_resolved_asec_puf_shared_vars_overlap(monkeypatch) -> None:
    monkeypatch.setattr(
        "microplex.data_sources.us_registry.CPSAsecSourceProvider",
        lambda **kwargs: CPSAsecSourceProvider(loader=_cps_dataset, **kwargs),
    )
    monkeypatch.setattr(
        "microplex.data_sources.us_registry.PUFSourceProvider",
        lambda **kwargs: PUFSourceProvider(loader=_puf_frame, **kwargs),
    )
    registry = create_us_asec_puf_source_registry(download_cps=False)
    spec = load_spec_dict(
        {
            "meta": {"country": "us", "model_year": 2024},
            "sources": {
                "cps_asec": {
                    "dataset": "cps_asec_2025_calendar_2024",
                    "role": "spine",
                },
                "puf": {"dataset": "puf_2024", "role": "donor"},
            },
            "spine": {
                "base": "cps_asec",
                "method": "support_spine",
                "support": {"seed": 42},
                "halves": [
                    {"name": "cps_keep", "keep": "all"},
                    {"name": "synthetic_puf", "strip_to": ["tax_unit_id"]},
                ],
            },
            "imputation": [],
        }
    )

    frames = registry.resolve_sources(spec)

    shared = set(SHARED_VARS)
    assert shared.issubset(frames["cps_asec"].columns)
    assert shared.issubset(frames["puf"].columns)


def test_provider_descriptors_advertise_loaded_entity_schema() -> None:
    cps_descriptor = CPSAsecSourceProvider().descriptor
    puf_descriptor = PUFSourceProvider().descriptor

    assert set(cps_descriptor.observed_entities) == {
        EntityType.HOUSEHOLD,
        EntityType.PERSON,
        EntityType.TAX_UNIT,
    }
    assert set(SHARED_VARS).issubset(cps_descriptor.variables_for(EntityType.TAX_UNIT))
    assert set(SHARED_VARS).issubset(puf_descriptor.variables_for(EntityType.TAX_UNIT))
    assert not puf_descriptor.shareability.allows_direct_release


def test_cps_loader_knows_current_asec_release_url() -> None:
    assert 2025 in get_available_years()


def test_cps_processed_cache_requires_household_weight_cache(tmp_path) -> None:
    persons = pl.DataFrame(
        {
            "household_id": [1, 2],
            "weight": [10.0, 20.0],
        }
    )
    households = pl.DataFrame(
        {
            "household_id": [1, 2],
            "household_weight": [100.0, 200.0],
        }
    )
    persons.write_parquet(tmp_path / "cps_asec_2025_processed.parquet")
    households.write_parquet(tmp_path / "cps_asec_2025_households_processed.parquet")

    cached = load_cps_asec(year=2025, cache_dir=tmp_path, download=False)

    assert cached.households["household_weight"].to_list() == [100.0, 200.0]


def test_cps_processed_cache_rejects_stale_person_only_cache(tmp_path) -> None:
    persons = pl.DataFrame(
        {
            "household_id": [1, 2],
            "weight": [10.0, 20.0],
        }
    )
    persons.write_parquet(tmp_path / "cps_asec_2025_processed.parquet")

    try:
        load_cps_asec(year=2025, cache_dir=tmp_path, download=False)
    except FileNotFoundError as exc:
        assert "household cache is missing" in str(exc)
    else:  # pragma: no cover - defensive assertion for fail-closed behavior
        raise AssertionError("stale person-only CPS cache should fail closed")


def test_us_registry_forwards_explicit_puf_paths(monkeypatch, tmp_path) -> None:
    puf_path = tmp_path / "puf_2015.csv"
    demographics_path = tmp_path / "demographics_2015.csv"
    puf_path.write_text("MARS,S006\n1,100\n")
    demographics_path.write_text("RECID,age\n1,40\n")
    captured = {}

    class RecordingPUFProvider:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(us_registry, "PUFSourceProvider", RecordingPUFProvider)

    us_registry.create_us_asec_puf_source_registry(
        puf_path=puf_path,
        puf_demographics_path=demographics_path,
    )

    assert captured["puf_path"] == puf_path
    assert captured["demographics_path"] == demographics_path
