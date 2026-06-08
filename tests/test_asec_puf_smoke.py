from __future__ import annotations

import json

import pandas as pd
import polars as pl
import pytest

from microplex.core import EntityType
from microplex.data_sources.asec_puf_smoke import (
    build_asec_puf_support_spine_spec,
    main,
    run_asec_puf_support_spine_smoke,
)
from microplex.data_sources.cps import CPSAsecSourceProvider, CPSDataset
from microplex.data_sources.puf import PUFSourceProvider, download_puf, load_puf
from microplex.source_registry import SourceRegistry


def _cps_dataset(*, year: int, cache_dir=None, download: bool = True) -> CPSDataset:
    del year, cache_dir, download
    households = pl.DataFrame(
        {
            "household_id": [1, 2, 3, 4],
            "household_weight": [100.0, 200.0, 300.0, 400.0],
            "state_fips": [6, 36, 48, 12],
            "household_size": [2, 1, 3, 1],
            "household_total_income": [70_000.0, 20_000.0, 120_000.0, 12_000.0],
        }
    )
    persons = pl.DataFrame(
        {
            "household_id": [1, 1, 2, 3, 3, 3, 4],
            "person_number": [1, 2, 1, 1, 2, 3, 1],
            "age": [40, 38, 21, 50, 48, 16, 77],
            "marital_status": [1, 1, 6, 1, 1, 6, 4],
            "wage_income": [50_000.0, 15_000.0, 20_000.0, 80_000.0, 30_000.0, 0.0, 0.0],
            "self_employment_income": [5_000.0, 0.0, 0.0, 2_000.0, 0.0, 0.0, 0.0],
            "interest_income": [100.0, 50.0, 0.0, 600.0, 20.0, 0.0, 2.0],
            "dividend_income": [200.0, 25.0, 0.0, 700.0, 30.0, 0.0, 0.0],
            "rental_income": [10.0, 3.0, 0.0, 1_000.0, 0.0, 0.0, 0.0],
            "social_security": [1_000.0, 0.0, 500.0, 0.0, 0.0, 0.0, 12_000.0],
            "taxable_pension_income": [40.0, 5.0, 20.0, 0.0, 0.0, 0.0, 5_000.0],
            "unemployment_compensation": [0.0, 25.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            "weight": [10.0, 10.0, 20.0, 30.0, 30.0, 30.0, 40.0],
            "is_child": [False, False, False, False, False, True, False],
            "is_adult": [True, True, True, True, True, False, True],
        }
    )
    return CPSDataset(
        persons=persons,
        households=households,
        year=2025,
        source="fixture-cps",
    )


def _puf_frame(*, target_year: int, expand_persons: bool = False, cache_dir=None):
    del target_year, expand_persons, cache_dir
    return pd.DataFrame(
        {
            "employment_income": [60_000.0, 10_000.0, 150_000.0],
            "self_employment_income": [4_000.0, 0.0, 1_000.0],
            "taxable_interest_income": [500.0, 0.0, 900.0],
            "ordinary_dividend_income": [600.0, 0.0, 700.0],
            "long_term_capital_gains": [5_000.0, 0.0, 30_000.0],
            "rental_income": [700.0, 0.0, 1_000.0],
            "gross_social_security": [800.0, 0.0, 0.0],
            "taxable_pension_income": [300.0, 0.0, 0.0],
            "unemployment_compensation": [900.0, 0.0, 0.0],
            "age": [45, 30, 55],
            "filing_status": ["JOINT", "SINGLE", "JOINT"],
            "weight": [75.0, 25.0, 50.0],
        }
    )


def _registry() -> SourceRegistry:
    return (
        SourceRegistry()
        .register(
            "cps_asec_2025_calendar_2024",
            CPSAsecSourceProvider(loader=_cps_dataset),
            default_entity=EntityType.TAX_UNIT,
        )
        .register(
            "puf_2024",
            PUFSourceProvider(loader=_puf_frame),
            default_entity=EntityType.TAX_UNIT,
        )
    )


def test_build_asec_puf_support_spine_spec_matches_registry_ids() -> None:
    spec = build_asec_puf_support_spine_spec()

    assert spec.sources["cps_asec"].dataset == "cps_asec_2025_calendar_2024"
    assert spec.sources["puf"].dataset == "puf_2024"
    assert spec.spine.method.value == "support_spine"


def test_asec_puf_smoke_loads_sources_and_builds_support_spine() -> None:
    result = run_asec_puf_support_spine_smoke(registry=_registry())

    assert result.diagnostics["source_rows"] == {"cps_asec": 4, "puf": 3}
    assert result.diagnostics["output_rows"] == 4
    assert result.diagnostics["half_counts"] == {"cps_keep": 2, "synthetic_puf": 2}
    assert result.diagnostics["synthetic_puf_household_weight_sum"] == 0.0
    assert result.diagnostics["shared_missing"] == {"cps_asec": [], "puf": []}


def test_asec_puf_smoke_caps_loaded_source_rows() -> None:
    result = run_asec_puf_support_spine_smoke(
        registry=_registry(),
        max_cps_rows=2,
        max_puf_rows=2,
    )

    assert result.diagnostics["source_rows"] == {"cps_asec": 2, "puf": 2}
    assert result.diagnostics["half_counts"] == {"cps_keep": 1, "synthetic_puf": 1}


def test_asec_puf_smoke_cli_writes_json(tmp_path, monkeypatch) -> None:
    puf_path = tmp_path / "puf_2015.csv"
    demographics_path = tmp_path / "demographics_2015.csv"
    captured = {}

    def _registry_factory(**kwargs):
        captured.update(kwargs)
        return _registry()

    monkeypatch.setattr(
        "microplex.data_sources.asec_puf_smoke.create_us_asec_puf_source_registry",
        _registry_factory,
    )
    output_path = tmp_path / "smoke.json"

    exit_code = main(
        [
            "--puf-path",
            str(puf_path),
            "--puf-demographics-path",
            str(demographics_path),
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert captured["puf_path"] == puf_path
    assert captured["puf_demographics_path"] == demographics_path
    payload = json.loads(output_path.read_text())
    assert payload["source_rows"] == {"cps_asec": 4, "puf": 3}
    assert payload["half_counts"] == {"cps_keep": 2, "synthetic_puf": 2}


def test_load_puf_accepts_explicit_restricted_access_paths(tmp_path) -> None:
    puf_path = tmp_path / "puf_2015.csv"
    demo_path = tmp_path / "demographics_2015.csv"
    pd.DataFrame(
        {
            "RECID": [1],
            "MARS": [1],
            "XTOT": [1],
            "E00200": [10_000.0],
            "E00300": [25.0],
            "E00400": [0.0],
            "E00600": [15.0],
            "E00900": [0.0],
            "E01700": [0.0],
            "E02300": [0.0],
            "E02400": [0.0],
            "E25850": [100.0],
            "E25860": [-50.0],
            "S006": [12345.0],
        }
    ).to_csv(puf_path, index=False)
    pd.DataFrame({"RECID": [1], "age": [44], "is_male": [1]}).to_csv(
        demo_path, index=False
    )

    loaded = load_puf(
        target_year=2024,
        expand_persons=False,
        puf_path=puf_path,
        demographics_path=demo_path,
    )

    assert loaded["employment_income"].tolist() == [14500.0]
    assert loaded["rental_income"].tolist() == [50.0]
    assert loaded["weight"].tolist() == [123.45]


def test_download_puf_explains_restricted_local_file_requirement(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("microplex.data_sources.puf.HF_AVAILABLE", True)

    def _raise_auth_error(**kwargs):
        del kwargs
        raise RuntimeError("401")

    monkeypatch.setattr("microplex.data_sources.puf.hf_hub_download", _raise_auth_error)

    with pytest.raises(FileNotFoundError, match="restricted IRS PUF files"):
        download_puf(tmp_path)


def test_download_puf_without_huggingface_uses_restricted_file_guidance(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("microplex.data_sources.puf.HF_AVAILABLE", False)

    with pytest.raises(FileNotFoundError, match="restricted IRS PUF files"):
        download_puf(tmp_path)
