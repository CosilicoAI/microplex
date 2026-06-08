from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from microplex.core import EntityType
from microplex.core.sources import SourceQuery
from microplex.data_sources.source_impute import (
    ManifestSourceImputeProvider,
    SourceImputeManifest,
    load_source_impute_block_table,
)
from microplex.data_sources.us_registry import register_us_source_impute_blocks
from microplex.source_registry import SourceRegistry
from microplex.spec import load_spec_dict


def _manifest_path(tmp_path: Path) -> Path:
    path = tmp_path / "pe_source_impute_blocks.json"
    path.write_text(
        json.dumps(
            {
                "blocks": {
                    "scf": {
                        "survey_name": "scf",
                        "block_name": None,
                        "default_year": 2022,
                        "archetype": "wealth",
                        "dataset_loader": {
                            "class_name": "SCF_2022",
                            "builder_kind": "single_person_households",
                            "direct_person_columns": {
                                "age": "age",
                                "credit_score": "credit_score",
                                "employment_income": "employment_income",
                                "weight": "wgt",
                            },
                            "boolean_person_columns": {"is_female": "is_female"},
                            "fallback_person_columns": {
                                "net_worth": ["net_worth", "networth"]
                            },
                            "copy_person_columns": {"income": "employment_income"},
                            "constant_person_columns": {
                                "state_fips": 0,
                                "tenure": 0,
                            },
                            "int_person_columns": [
                                "age",
                                "sex",
                                "state_fips",
                                "tenure",
                            ],
                            "sex_from_boolean_source": "is_female",
                            "sex_true_value": 2,
                            "sex_false_value": 1,
                        },
                        "household_variables": ["state_fips", "tenure"],
                        "person_variables": [
                            "age",
                            "sex",
                            "is_female",
                            "employment_income",
                            "income",
                            "net_worth",
                        ],
                        "target_variables": ["net_worth"],
                        "predictors": [
                            "age",
                            "credit_score",
                            "is_female",
                            "employment_income",
                        ],
                    }
                }
            }
        )
    )
    return path


def _scf_h5_path(tmp_path: Path) -> Path:
    path = tmp_path / "scf_2022.h5"
    with h5py.File(path, "w") as h5:
        h5["age"] = np.array([40, 65, 30])
        h5["credit_score"] = np.array([680, 720, 640])
        h5["employment_income"] = np.array([50_000.0, 0.0, 80_000.0])
        h5["wgt"] = np.array([10.0, 20.0, 30.0])
        h5["is_female"] = np.array([False, True, True])
        h5["networth"] = np.array([100_000.0, 250_000.0, -5_000.0])
    return path


def _real_scf_manifest_path() -> Path:
    return Path("packs/us/manifests/pe_source_impute_blocks.json")


def _real_scf_h5_path(tmp_path: Path) -> Path:
    block = SourceImputeManifest.from_path(_real_scf_manifest_path()).block("scf")
    assert block.dataset_loader is not None
    loader: dict[str, Any] = dict(block.dataset_loader)
    direct_sources = set(loader["direct_person_columns"].values())
    boolean_sources = set(loader["boolean_person_columns"].values())
    fallback_sources = {
        sources[-1] for sources in loader["fallback_person_columns"].values()
    }

    path = tmp_path / "scf_2022.h5"
    with h5py.File(path, "w") as h5:
        for source in sorted(direct_sources):
            h5[source] = np.array([1.0, 2.0, 3.0])
        for source in sorted(boolean_sources):
            h5[source] = np.array([0, 1, 0])
        for source in sorted(fallback_sources):
            h5[source] = np.array([10.0, 20.0, 30.0])
    return path


def test_source_impute_manifest_loads_block(tmp_path: Path) -> None:
    manifest = SourceImputeManifest.from_path(_manifest_path(tmp_path))

    block = manifest.block("scf")

    assert block.survey_name == "scf"
    assert block.default_year == 2022
    assert block.target_variables == ("net_worth",)


def test_source_impute_block_table_uses_manifest_mappings(tmp_path: Path) -> None:
    block = SourceImputeManifest.from_path(_manifest_path(tmp_path)).block("scf")

    table = load_source_impute_block_table(
        block,
        dataset_path=_scf_h5_path(tmp_path),
        max_rows=2,
        period=2024,
    )

    assert table["person_id"].tolist() == [0, 1]
    assert table["household_id"].tolist() == [0, 1]
    assert table["tax_unit_id"].tolist() == [0, 1]
    assert table["year"].tolist() == [2024, 2024]
    assert table["age"].tolist() == [40, 65]
    assert table["credit_score"].tolist() == [680, 720]
    assert table["is_female"].tolist() == [False, True]
    assert table["sex"].tolist() == [1, 2]
    assert table["state_fips"].tolist() == [0, 0]
    assert table["tenure"].tolist() == [0, 0]
    assert table["income"].tolist() == [50_000.0, 0.0]
    assert table["net_worth"].tolist() == [100_000.0, 250_000.0]
    assert table["weight"].tolist() == [10.0, 20.0]


def test_source_impute_block_table_parses_string_boolean_values(
    tmp_path: Path,
) -> None:
    block = SourceImputeManifest.from_path(_manifest_path(tmp_path)).block("scf")
    path = _scf_h5_path(tmp_path)
    with h5py.File(path, "a") as h5:
        del h5["is_female"]
        h5["is_female"] = np.array([b"False", b"0", b"True"])

    table = load_source_impute_block_table(block, dataset_path=path, period=2024)

    assert table["is_female"].tolist() == [False, False, True]
    assert table["sex"].tolist() == [1, 1, 2]


def test_manifest_source_impute_provider_materializes_person_frame(
    tmp_path: Path,
) -> None:
    provider = ManifestSourceImputeProvider(
        manifest_path=_manifest_path(tmp_path),
        block_name="scf",
        dataset_path=_scf_h5_path(tmp_path),
        max_rows=2,
    )

    frame = provider.load_frame(SourceQuery(period=2024))

    assert frame.source.name == "scf"
    assert frame.source.observed_entities == (EntityType.PERSON,)
    assert "state_fips" in provider.descriptor.variables_for(EntityType.PERSON)
    table = frame.tables[EntityType.PERSON]
    assert len(table) == 2
    assert "net_worth" in frame.source.variables_for(EntityType.PERSON)


def test_manifest_source_impute_provider_accepts_string_period_query(
    tmp_path: Path,
) -> None:
    provider = ManifestSourceImputeProvider(
        manifest_path=_manifest_path(tmp_path),
        block_name="scf",
        dataset_path=_scf_h5_path(tmp_path),
        max_rows=2,
    )

    frame = provider.load_frame(SourceQuery(period="2024"))

    table = frame.tables[EntityType.PERSON]
    assert len(table) == 2
    assert table["year"].tolist() == [2024, 2024]


def test_register_us_source_impute_blocks_resolves_scf_dataset(
    tmp_path: Path,
) -> None:
    _scf_h5_path(tmp_path)
    registry = SourceRegistry()
    register_us_source_impute_blocks(
        registry,
        manifest_path=_manifest_path(tmp_path),
        storage_dir=tmp_path,
        max_rows=2,
    )
    spec = load_spec_dict(
        {
            "meta": {"country": "us", "model_year": 2024},
            "sources": {
                "scf": {
                    "dataset": "scf_2022",
                    "role": "spine",
                    "entity": "person",
                }
            },
            "spine": {
                "base": "scf",
                "method": "support_spine",
                "support": {"seed": 42},
                "halves": [
                    {"name": "keep", "keep": "all"},
                    {"name": "strip", "strip_to": ["person_id"]},
                ],
            },
            "imputation": [],
        }
    )

    frames = registry.resolve_sources(spec)

    assert frames["scf"]["person_id"].tolist() == [0, 1]
    assert frames["scf"]["net_worth"].tolist() == [100_000.0, 250_000.0]


def test_register_us_source_impute_blocks_rejects_unsupported_real_blocks() -> None:
    registry = SourceRegistry()

    with pytest.raises(NotImplementedError, match="household_rows"):
        register_us_source_impute_blocks(
            registry,
            manifest_path=_real_scf_manifest_path(),
            blocks=("acs",),
        )


def test_register_us_source_impute_blocks_rejects_mixed_blocks_atomically() -> None:
    registry = SourceRegistry()

    with pytest.raises(NotImplementedError, match="household_rows"):
        register_us_source_impute_blocks(
            registry,
            manifest_path=_real_scf_manifest_path(),
            blocks=("scf", "acs"),
        )
    with pytest.raises(KeyError, match="scf_2022"):
        registry.provider_for("scf_2022")


def test_register_us_source_impute_blocks_rejects_duplicates_atomically() -> None:
    registry = SourceRegistry()

    with pytest.raises(ValueError, match="Duplicate source-impute dataset"):
        register_us_source_impute_blocks(
            registry,
            manifest_path=_real_scf_manifest_path(),
            blocks=("scf", "scf"),
        )
    with pytest.raises(KeyError, match="scf_2022"):
        registry.provider_for("scf_2022")


def test_source_impute_loader_rejects_missing_required_h5_array(
    tmp_path: Path,
) -> None:
    block = SourceImputeManifest.from_path(_manifest_path(tmp_path)).block("scf")
    path = _scf_h5_path(tmp_path)
    with h5py.File(path, "a") as h5:
        del h5["employment_income"]

    with pytest.raises(ValueError, match="missing required arrays"):
        load_source_impute_block_table(block, dataset_path=path, period=2024)


def test_source_impute_loader_rejects_empty_required_h5_arrays(
    tmp_path: Path,
) -> None:
    block = SourceImputeManifest.from_path(_manifest_path(tmp_path)).block("scf")
    path = _scf_h5_path(tmp_path)
    with h5py.File(path, "a") as h5:
        for name in list(h5.keys()):
            del h5[name]
        h5["age"] = np.array([])
        h5["credit_score"] = np.array([])
        h5["employment_income"] = np.array([])
        h5["wgt"] = np.array([])
        h5["is_female"] = np.array([], dtype=bool)
        h5["networth"] = np.array([])

    with pytest.raises(ValueError, match="at least one row"):
        load_source_impute_block_table(block, dataset_path=path, period=2024)


def test_source_impute_loader_rejects_malformed_integer_columns(
    tmp_path: Path,
) -> None:
    block = SourceImputeManifest.from_path(_manifest_path(tmp_path)).block("scf")
    path = _scf_h5_path(tmp_path)
    with h5py.File(path, "a") as h5:
        del h5["age"]
        h5["age"] = np.array([b"40", b"not an age", b"30"])

    with pytest.raises(ValueError, match="non-numeric"):
        load_source_impute_block_table(block, dataset_path=path, period=2024)


def test_source_impute_loader_rejects_unrecognized_boolean_values(
    tmp_path: Path,
) -> None:
    block = SourceImputeManifest.from_path(_manifest_path(tmp_path)).block("scf")
    path = _scf_h5_path(tmp_path)
    with h5py.File(path, "a") as h5:
        del h5["is_female"]
        h5["is_female"] = np.array([b"False", b"unknown", b"True"])

    with pytest.raises(ValueError, match="unrecognized values"):
        load_source_impute_block_table(block, dataset_path=path, period=2024)


def test_source_impute_loader_rejects_missing_declared_variables(
    tmp_path: Path,
) -> None:
    manifest = json.loads(_manifest_path(tmp_path).read_text())
    manifest["blocks"]["scf"]["person_variables"].append("unmapped_person_variable")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    block = SourceImputeManifest.from_path(path).block("scf")

    with pytest.raises(ValueError, match="missing declared variables"):
        load_source_impute_block_table(
            block,
            dataset_path=_scf_h5_path(tmp_path),
            period=2024,
        )


def test_source_impute_loader_rejects_unsupported_builder_kind(
    tmp_path: Path,
) -> None:
    manifest = json.loads(_manifest_path(tmp_path).read_text())
    manifest["blocks"]["scf"]["dataset_loader"]["builder_kind"] = "household_rows"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    block = SourceImputeManifest.from_path(path).block("scf")

    with pytest.raises(NotImplementedError, match="household_rows"):
        load_source_impute_block_table(
            block,
            dataset_path=_scf_h5_path(tmp_path),
            period=2024,
        )


def test_source_impute_loader_rejects_non_vector_h5_arrays(
    tmp_path: Path,
) -> None:
    block = SourceImputeManifest.from_path(_manifest_path(tmp_path)).block("scf")
    path = _scf_h5_path(tmp_path)
    with h5py.File(path, "a") as h5:
        del h5["credit_score"]
        h5["credit_score"] = np.array([[680], [720], [640]])

    with pytest.raises(ValueError, match="one-dimensional"):
        load_source_impute_block_table(block, dataset_path=path, period=2024)


def test_real_us_scf_manifest_block_retains_declared_targets_and_predictors(
    tmp_path: Path,
) -> None:
    block = SourceImputeManifest.from_path(_real_scf_manifest_path()).block("scf")

    table = load_source_impute_block_table(
        block,
        dataset_path=_real_scf_h5_path(tmp_path),
        period=2024,
    )

    assert set(block.target_variables).issubset(table.columns)
    assert set(block.predictors).issubset(table.columns)
    assert set(block.household_variables).issubset(table.columns)
