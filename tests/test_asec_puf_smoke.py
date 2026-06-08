from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import polars as pl
import pytest

from microplex.core import EntityType
from microplex.data_sources.asec_puf_smoke import (
    build_asec_puf_support_spine_spec,
    main,
    run_asec_puf_support_spine_smoke,
    write_asec_puf_support_spine_stage_artifacts,
)
from microplex.data_sources.cps import CPSAsecSourceProvider, CPSDataset
from microplex.data_sources.puf import PUFSourceProvider, download_puf, load_puf
from microplex.source_registry import SourceRegistry
from microplex.stage_manifest import load_stage_manifest, validate_stage_manifest


def _cps_dataset(*, year: int, cache_dir=None, download: bool = True) -> CPSDataset:
    del year, cache_dir, download
    households = pl.DataFrame(
        {
            "household_id": [1, 2, 3, 4],
            "household_weight": [100.0, 200.0, 300.0, 400.0],
            "state_fips": [6, 36, 48, 12],
            "cbsa": [41860, 35620, 19100, 33100],
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


def _cps_dataset_with_missing_geography(
    *,
    year: int,
    cache_dir=None,
    download: bool = True,
) -> CPSDataset:
    dataset = _cps_dataset(year=year, cache_dir=cache_dir, download=download)
    households = dataset.households.with_columns(
        [
            pl.when(pl.col("household_id") == 1)
            .then(None)
            .otherwise(pl.col("state_fips"))
            .alias("state_fips"),
            pl.when(pl.col("household_id") == 1)
            .then(None)
            .otherwise(pl.col("cbsa"))
            .alias("cbsa"),
        ]
    )
    return CPSDataset(
        persons=dataset.persons,
        households=households,
        year=dataset.year,
        source=dataset.source,
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


def _block_crosswalk_path(tmp_path) -> str:
    path = tmp_path / "block_crosswalk.csv"
    pd.DataFrame(
        {
            "block_geoid": [
                "060010201001000",
                "060010201001001",
                "120860001001000",
                "360610001001000",
                "480010001001000",
            ],
            "sldu": ["009", "009", "040", "030", "003"],
            "sldl": ["018", "018", "110", "065", "011"],
            "place_fips": ["53000", "", "45000", "51000", ""],
        }
    ).to_csv(path, index=False)
    return str(path)


def _source_impute_manifest_path(
    tmp_path: Path,
    *,
    include_sipp: bool = False,
) -> Path:
    blocks: dict[str, Any] = {
        "scf": {
            "survey_name": "scf",
            "default_year": 2022,
            "dataset_loader": {
                "class_name": "SCF_2022",
                "builder_kind": "single_person_households",
                "direct_person_columns": {
                    "age": "age",
                    "net_worth": "net_worth",
                    "weight": "wgt",
                },
                "constant_person_columns": {
                    "state_fips": 0,
                    "tenure": 0,
                },
            },
            "household_variables": ["state_fips", "tenure"],
            "person_variables": ["age", "net_worth", "weight"],
            "target_variables": ["net_worth"],
            "predictors": ["age"],
        }
    }
    if include_sipp:
        blocks["sipp_assets"] = {
            "survey_name": "sipp",
            "default_year": 2023,
            "dataset_loader": {
                "class_name": "SIPP_2023",
                "builder_kind": "single_person_households",
                "direct_person_columns": {
                    "age": "age",
                    "bank_account_assets": "bank_account_assets",
                    "weight": "wgt",
                },
            },
            "household_variables": [],
            "person_variables": ["age", "bank_account_assets", "weight"],
            "target_variables": ["bank_account_assets"],
            "predictors": ["age"],
        }
    path = tmp_path / "pe_source_impute_blocks.json"
    path.write_text(json.dumps({"blocks": blocks}))
    return path


def _source_impute_spec_path(
    tmp_path: Path,
    *,
    include_sipp: bool = False,
) -> Path:
    sources: dict[str, Any] = {
        "cps_asec": {
            "dataset": "cps_asec_2025_calendar_2024",
            "role": "spine",
            "entity": "tax_unit",
        },
        "scf": {
            "dataset": "scf_2022",
            "role": "donor",
            "entity": "person",
        },
    }
    variables: dict[str, Any] = {
        "net_worth": {
            "mp_spec": {
                "method": "impute from scf",
                "operation": {
                    "kind": "impute",
                    "source": "scf",
                    "imputation_step": "scf_source_impute",
                },
            }
        }
    }
    if include_sipp:
        sources["sipp"] = {
            "dataset": "sipp_2023",
            "role": "donor",
            "entity": "person",
        }
        variables["bank_account_assets"] = {
            "mp_spec": {
                "method": "impute from sipp",
                "operation": {
                    "kind": "impute",
                    "source": "sipp",
                    "imputation_step": "sipp_source_impute",
                },
            }
        }
    path = tmp_path / "us-2024-source-impute.yaml"
    path.write_text(
        json.dumps(
            {
                "meta": {"country": "us", "model_year": 2024},
                "sources": sources,
                "spine": {
                    "base": "cps_asec",
                    "method": "support_spine",
                    "support": {"seed": 42},
                    "halves": [
                        {"name": "cps_keep", "keep": "all"},
                        {"name": "synthetic_puf", "strip_to": ["demographics"]},
                    ],
                },
                "imputation": [],
                "variables": variables,
            }
        )
    )
    return path


def _source_impute_scf_h5_path(tmp_path: Path) -> Path:
    path = tmp_path / "scf_2022.h5"
    with h5py.File(path, "w") as h5:
        h5["age"] = np.array([25, 40, 65, 80])
        h5["net_worth"] = np.array([10_000.0, 100_000.0, 250_000.0, 500_000.0])
        h5["wgt"] = np.array([1.0, 2.0, 3.0, 4.0])
    return path


class _RecordingSourceImputer:
    def __init__(self) -> None:
        self.fit_kwargs: dict[str, Any] | None = None
        self.regimes_: dict[str, str] = {}

    def fit(self, **kwargs):
        self.fit_kwargs = kwargs
        self.regimes_ = {variable: "TEST" for variable in kwargs["imputed_variables"]}
        return self

    def predict(self, target: pd.DataFrame) -> pd.DataFrame:
        assert self.fit_kwargs is not None
        return pd.DataFrame(
            {
                variable: np.arange(len(target), dtype=float) + 1000.0
                for variable in self.fit_kwargs["imputed_variables"]
            },
            index=target.index,
        )


def test_build_asec_puf_support_spine_spec_matches_registry_ids() -> None:
    spec = build_asec_puf_support_spine_spec()

    assert spec.sources["cps_asec"].dataset == "cps_asec_2025_calendar_2024"
    assert spec.sources["puf"].dataset == "puf_2024"
    assert spec.spine.method.value == "support_spine"
    assert spec.spine.synthetic_half.strip_to == [
        "demographics",
        "state_fips",
        "cbsa",
    ]


def test_asec_puf_smoke_loads_sources_and_builds_support_spine() -> None:
    result = run_asec_puf_support_spine_smoke(registry=_registry())

    assert result.diagnostics["source_rows"] == {"cps_asec": 4, "puf": 3}
    assert result.diagnostics["output_rows"] == 4
    assert result.diagnostics["half_counts"] == {"cps_keep": 2, "synthetic_puf": 2}
    assert result.diagnostics["synthetic_puf_household_weight_sum"] == 0.0
    assert result.diagnostics["shared_missing"] == {"cps_asec": [], "puf": []}
    assert result.diagnostics["geography_constraint_columns"] == [
        "state_fips",
        "cbsa",
    ]
    synthetic = result.run_result.halves["synthetic_puf"]
    assert set(synthetic["state_fips"]) <= {6, 36, 48, 12}
    assert set(synthetic["cbsa"]) <= {41860, 35620, 19100, 33100}
    assert synthetic[["state_fips", "cbsa"]].notna().any(axis=1).all()


def test_asec_puf_smoke_can_assign_census_blocks(tmp_path) -> None:
    result = run_asec_puf_support_spine_smoke(
        registry=_registry(),
        block_crosswalk_path=Path(_block_crosswalk_path(tmp_path)),
    )

    block_geography = result.diagnostics["block_geography"]
    assert block_geography["assigned"] is True
    assert block_geography["assigned_rows"] == 4
    assert block_geography["partition_counts"] == {"state_fips": 4}
    assert set(block_geography["columns"]) == {
        "block_geoid",
        "state_fips",
        "county_fips",
        "tract_geoid",
    }
    assert result.run_result.frame["block_geoid"].str.len().eq(15).all()
    assert result.run_result.frame["state_fips"].isin(["06", "12", "36", "48"]).all()


def test_asec_puf_smoke_rejects_source_impute_before_block_assignment(
    tmp_path: Path,
) -> None:
    _source_impute_scf_h5_path(tmp_path)

    with pytest.raises(ValueError, match="requires block geography assignment"):
        run_asec_puf_support_spine_smoke(
            registry=_registry(),
            source_impute_spec_path=_source_impute_spec_path(tmp_path),
            source_impute_manifest_path=_source_impute_manifest_path(tmp_path),
            source_impute_storage_dir=tmp_path,
            source_impute_blocks=("scf",),
        )


def test_asec_puf_smoke_runs_source_impute_after_block_assignment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _source_impute_scf_h5_path(tmp_path)
    recorder = _RecordingSourceImputer()
    monkeypatch.setattr(
        "microplex.imputation.ImputationRunner._make_imputer",
        lambda self: recorder,
    )

    result = run_asec_puf_support_spine_smoke(
        registry=_registry(),
        block_crosswalk_path=Path(_block_crosswalk_path(tmp_path)),
        source_impute_spec_path=_source_impute_spec_path(tmp_path),
        source_impute_manifest_path=_source_impute_manifest_path(tmp_path),
        source_impute_storage_dir=tmp_path,
        source_impute_blocks=("scf",),
    )

    source_imputation = result.diagnostics["source_imputation"]
    assert source_imputation["enabled"] is True
    assert source_imputation["source_rows"] == {"scf": 4}
    assert len(source_imputation["results"]) == 2
    assert {item["onto"] for item in source_imputation["results"]} == {
        "cps_keep",
        "synthetic_puf",
    }
    assert all(
        item["donor"] == "scf" and item["imputed"] == ["net_worth"]
        for item in source_imputation["results"]
    )
    assert result.source_impute_sources["scf"]["net_worth"].notna().all()
    assert result.run_result.frame["block_geoid"].notna().all()
    assert result.run_result.frame["net_worth"].notna().all()
    assert recorder.fit_kwargs is not None
    assert recorder.fit_kwargs["predictors"] == ["age"]


def test_asec_puf_smoke_block_filter_limits_source_impute_compilation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _source_impute_scf_h5_path(tmp_path)
    recorder = _RecordingSourceImputer()
    monkeypatch.setattr(
        "microplex.imputation.ImputationRunner._make_imputer",
        lambda self: recorder,
    )

    result = run_asec_puf_support_spine_smoke(
        registry=_registry(),
        block_crosswalk_path=Path(_block_crosswalk_path(tmp_path)),
        source_impute_spec_path=_source_impute_spec_path(tmp_path, include_sipp=True),
        source_impute_manifest_path=_source_impute_manifest_path(
            tmp_path,
            include_sipp=True,
        ),
        source_impute_storage_dir=tmp_path,
        source_impute_blocks=("scf",),
    )

    source_imputation = result.diagnostics["source_imputation"]
    assert source_imputation["source_rows"] == {"scf": 4}
    assert [item["donor"] for item in source_imputation["results"]] == [
        "scf",
        "scf",
    ]
    assert "bank_account_assets" not in result.run_result.frame.columns


def test_asec_puf_smoke_step_filter_still_applies_with_block_filter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _source_impute_scf_h5_path(tmp_path)
    recorder = _RecordingSourceImputer()
    monkeypatch.setattr(
        "microplex.imputation.ImputationRunner._make_imputer",
        lambda self: recorder,
    )

    result = run_asec_puf_support_spine_smoke(
        registry=_registry(),
        block_crosswalk_path=Path(_block_crosswalk_path(tmp_path)),
        source_impute_spec_path=_source_impute_spec_path(tmp_path),
        source_impute_manifest_path=_source_impute_manifest_path(tmp_path),
        source_impute_storage_dir=tmp_path,
        source_impute_blocks=("scf",),
        source_impute_imputation_steps=(),
    )

    source_imputation = result.diagnostics["source_imputation"]
    assert source_imputation == {
        "enabled": False,
        "source_rows": {},
        "results": [],
    }
    assert "net_worth" not in result.run_result.frame.columns
    assert recorder.fit_kwargs is None


def test_asec_puf_smoke_writes_stage_artifacts(tmp_path: Path) -> None:
    result = run_asec_puf_support_spine_smoke(
        registry=_registry(),
        block_crosswalk_path=Path(_block_crosswalk_path(tmp_path)),
    )
    output_dir = tmp_path / "stage"

    manifest = write_asec_puf_support_spine_stage_artifacts(result, output_dir)

    assert (output_dir / "support_frame.parquet").is_file()
    assert (output_dir / "diagnostics.json").is_file()
    assert (output_dir / "stage_manifest.json").is_file()
    assert manifest.stage_id == "us_asec_puf_support_spine"
    assert set(manifest.artifacts) == {"support_frame", "diagnostics"}
    assert manifest.metadata["output_rows"] == 4
    assert validate_stage_manifest(manifest, root=output_dir) == []
    loaded = load_stage_manifest(output_dir / "stage_manifest.json")
    assert loaded == manifest
    frame = pd.read_parquet(output_dir / "support_frame.parquet")
    assert len(frame) == result.diagnostics["output_rows"]
    diagnostics = json.loads((output_dir / "diagnostics.json").read_text())
    assert diagnostics["block_geography"]["assigned"] is True


def test_asec_puf_smoke_caps_loaded_source_rows() -> None:
    result = run_asec_puf_support_spine_smoke(
        registry=_registry(),
        max_cps_rows=2,
        max_puf_rows=2,
    )

    assert result.diagnostics["source_rows"] == {"cps_asec": 2, "puf": 2}
    assert result.diagnostics["half_counts"] == {"cps_keep": 1, "synthetic_puf": 1}


def test_asec_puf_smoke_rejects_loaded_cps_rows_without_geography() -> None:
    registry = (
        SourceRegistry()
        .register(
            "cps_asec_2025_calendar_2024",
            CPSAsecSourceProvider(loader=_cps_dataset_with_missing_geography),
            default_entity=EntityType.TAX_UNIT,
        )
        .register(
            "puf_2024",
            PUFSourceProvider(loader=_puf_frame),
            default_entity=EntityType.TAX_UNIT,
        )
    )

    with pytest.raises(ValueError, match="no geography constraint values"):
        run_asec_puf_support_spine_smoke(registry=registry)


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
    output_dir = tmp_path / "stage"

    exit_code = main(
        [
            "--puf-path",
            str(puf_path),
            "--puf-demographics-path",
            str(demographics_path),
            "--output-json",
            str(output_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert captured["puf_path"] == puf_path
    assert captured["puf_demographics_path"] == demographics_path
    payload = json.loads(output_path.read_text())
    assert payload["source_rows"] == {"cps_asec": 4, "puf": 3}
    assert payload["half_counts"] == {"cps_keep": 2, "synthetic_puf": 2}
    manifest = load_stage_manifest(output_dir / "stage_manifest.json")
    assert set(manifest.artifacts) == {"support_frame", "diagnostics"}
    assert (output_dir / "support_frame.parquet").is_file()


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
