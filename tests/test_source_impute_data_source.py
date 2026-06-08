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
from microplex.spec import load_spec, load_spec_dict


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


def _acs_manifest_path(tmp_path: Path) -> Path:
    path = tmp_path / "pe_source_impute_blocks.json"
    path.write_text(
        json.dumps(
            {
                "blocks": {
                    "acs": {
                        "survey_name": "acs",
                        "block_name": None,
                        "default_year": 2022,
                        "archetype": "housing",
                        "dataset_loader": {
                            "class_name": "ACS_2022",
                            "builder_kind": "household_rows",
                            "household_index_key": "household_id",
                            "person_household_key": "person_household_id",
                            "person_id_key": "person_id",
                            "direct_person_columns": {
                                "age": "age",
                                "employment_income": "employment_income",
                                "self_employment_income": "self_employment_income",
                                "social_security": "social_security",
                                "taxable_pension_income": "taxable_private_pension_income",
                                "rent": "rent",
                                "real_estate_taxes": "real_estate_taxes",
                            },
                            "boolean_person_columns": {
                                "is_male": "is_male",
                                "is_household_head": "is_household_head",
                            },
                            "row_indexed_person_columns": {
                                "state_fips": "state_fips",
                                "weight": "household_weight",
                            },
                            "mapped_row_person_columns": {
                                "tenure_type": "tenure_type",
                            },
                            "mapped_value_tables": {
                                "tenure_type": {
                                    "OWNED_OUTRIGHT": 1,
                                    "RENTED": 2,
                                    "NONE": 0,
                                }
                            },
                            "fallback_person_columns": {},
                            "copy_person_columns": {
                                "pension_income": "taxable_pension_income",
                                "tenure": "tenure_type",
                            },
                            "constant_person_columns": {},
                            "income_sum_columns": [
                                "employment_income",
                                "self_employment_income",
                                "social_security",
                                "taxable_pension_income",
                            ],
                            "group_count_person_columns": {
                                "household_size": "person_household_id",
                            },
                            "int_person_columns": [
                                "age",
                                "sex",
                                "tenure_type",
                                "tenure",
                                "state_fips",
                            ],
                            "sex_from_boolean_source": "is_male",
                            "sex_true_value": 1,
                            "sex_false_value": 2,
                        },
                        "household_variables": ["state_fips", "tenure"],
                        "person_variables": [
                            "age",
                            "sex",
                            "is_male",
                            "is_household_head",
                            "tenure_type",
                            "employment_income",
                            "self_employment_income",
                            "social_security",
                            "taxable_pension_income",
                            "rent",
                            "real_estate_taxes",
                            "income",
                        ],
                        "target_variables": ["rent", "real_estate_taxes"],
                        "predictors": [
                            "is_household_head",
                            "age",
                            "is_male",
                            "tenure_type",
                            "employment_income",
                            "self_employment_income",
                            "social_security",
                            "pension_income",
                            "household_size",
                            "state_fips",
                        ],
                    }
                }
            }
        )
    )
    return path


def _acs_h5_path(tmp_path: Path) -> Path:
    path = tmp_path / "acs_2022.h5"
    with h5py.File(path, "w") as h5:
        h5["person_id"] = np.array([101, 102, 201, 301])
        h5["person_household_id"] = np.array([10, 10, 20, 30])
        h5["age"] = np.array([40, 38, 65, 22])
        h5["employment_income"] = np.array([50_000.0, 10_000.0, 0.0, 20_000.0])
        h5["self_employment_income"] = np.array([5_000.0, 0.0, 0.0, 2_000.0])
        h5["social_security"] = np.array([0.0, 0.0, 18_000.0, 0.0])
        h5["taxable_private_pension_income"] = np.array([100.0, 50.0, 7_000.0, 0.0])
        h5["rent"] = np.array([0.0, 0.0, 12_000.0, 0.0])
        h5["real_estate_taxes"] = np.array([3_000.0, 0.0, 0.0, 0.0])
        h5["is_male"] = np.array([1, 0, 0, 1])
        h5["is_household_head"] = np.array([1, 0, 1, 1])
        h5["household_id"] = np.array([10, 20, 30])
        h5["state_fips"] = np.array([6, 36, 48])
        h5["household_weight"] = np.array([100.0, 200.0, 300.0])
        h5["tenure_type"] = np.array([b"OWNED_OUTRIGHT", b"RENTED", b"NONE"])
    return path


def _real_scf_manifest_path() -> Path:
    return Path("packs/us/manifests/pe_source_impute_blocks.json")


def _real_us_spec_path() -> Path:
    return Path("packs/us/specs/us-2024.yaml")


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


def test_source_impute_household_rows_maps_household_columns(
    tmp_path: Path,
) -> None:
    block = SourceImputeManifest.from_path(_acs_manifest_path(tmp_path)).block("acs")

    table = load_source_impute_block_table(
        block,
        dataset_path=_acs_h5_path(tmp_path),
        period=2024,
    )

    assert table["person_id"].tolist() == [101, 102, 201, 301]
    assert table["household_id"].tolist() == [10, 10, 20, 30]
    assert table["tax_unit_id"].tolist() == [10, 10, 20, 30]
    assert table["state_fips"].tolist() == [6, 6, 36, 48]
    assert table["weight"].tolist() == [100.0, 100.0, 200.0, 300.0]
    assert table["tenure_type"].tolist() == [1, 1, 2, 0]
    assert table["tenure"].tolist() == [1, 1, 2, 0]
    assert table["household_size"].tolist() == [2, 2, 1, 1]
    assert table["pension_income"].tolist() == [100.0, 50.0, 7_000.0, 0.0]
    assert table["income"].tolist() == [55_100.0, 10_050.0, 25_000.0, 22_000.0]
    assert table["sex"].tolist() == [1, 2, 2, 1]
    assert table["year"].tolist() == [2024, 2024, 2024, 2024]


def test_source_impute_household_rows_uses_full_group_counts_with_max_rows(
    tmp_path: Path,
) -> None:
    block = SourceImputeManifest.from_path(_acs_manifest_path(tmp_path)).block("acs")
    path = _acs_h5_path(tmp_path)
    with h5py.File(path, "a") as h5:
        del h5["person_household_id"]
        h5["person_household_id"] = np.array([10, 20, 10, 20])

    table = load_source_impute_block_table(
        block,
        dataset_path=path,
        max_rows=2,
        period=2024,
    )

    assert table["person_id"].tolist() == [101, 102]
    assert table["household_id"].tolist() == [10, 20]
    assert table["household_size"].tolist() == [2, 2]


def test_source_impute_household_rows_rejects_missing_household_mapping(
    tmp_path: Path,
) -> None:
    block = SourceImputeManifest.from_path(_acs_manifest_path(tmp_path)).block("acs")
    path = _acs_h5_path(tmp_path)
    with h5py.File(path, "a") as h5:
        del h5["person_household_id"]
        h5["person_household_id"] = np.array([10, 10, 99, 30])

    with pytest.raises(ValueError, match="missing household ids"):
        load_source_impute_block_table(block, dataset_path=path, period=2024)


def test_source_impute_household_rows_rejects_duplicate_household_ids(
    tmp_path: Path,
) -> None:
    block = SourceImputeManifest.from_path(_acs_manifest_path(tmp_path)).block("acs")
    path = _acs_h5_path(tmp_path)
    with h5py.File(path, "a") as h5:
        del h5["household_id"]
        h5["household_id"] = np.array([10, 10, 30])

    with pytest.raises(ValueError, match="duplicate ids"):
        load_source_impute_block_table(block, dataset_path=path, period=2024)


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


def test_register_us_source_impute_blocks_resolves_acs_dataset(
    tmp_path: Path,
) -> None:
    _acs_h5_path(tmp_path)
    registry = SourceRegistry()
    register_us_source_impute_blocks(
        registry,
        manifest_path=_real_scf_manifest_path(),
        storage_dir=tmp_path,
        max_rows=2,
        blocks=("acs",),
    )
    spec = load_spec_dict(
        {
            "meta": {"country": "us", "model_year": 2024},
            "sources": {
                "acs": {
                    "dataset": "acs_2024",
                    "role": "spine",
                    "entity": "person",
                }
            },
            "spine": {
                "base": "acs",
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

    assert frames["acs"]["person_id"].tolist() == [101, 102]
    assert frames["acs"]["household_size"].tolist() == [2, 2]
    assert frames["acs"]["state_fips"].tolist() == [6, 6]


def test_register_us_source_impute_blocks_matches_real_us_spec_acs_dataset(
    tmp_path: Path,
) -> None:
    _acs_h5_path(tmp_path)
    spec = load_spec(_real_us_spec_path())
    registry = SourceRegistry()
    register_us_source_impute_blocks(
        registry,
        manifest_path=_real_scf_manifest_path(),
        storage_dir=tmp_path,
        blocks=("acs",),
    )

    assert spec.sources["acs"].dataset == "acs_2024"
    provider = registry.provider_for(spec.sources["acs"].dataset)
    frame = provider.load_frame(SourceQuery(period=2024))

    assert frame.tables[EntityType.PERSON]["person_id"].tolist() == [
        101,
        102,
        201,
        301,
    ]


def test_register_us_source_impute_blocks_rejects_unsupported_real_blocks() -> None:
    registry = SourceRegistry()

    with pytest.raises(NotImplementedError, match="has no dataset_loader"):
        register_us_source_impute_blocks(
            registry,
            manifest_path=_real_scf_manifest_path(),
            blocks=("sipp_tips",),
        )


def test_register_us_source_impute_blocks_rejects_mixed_blocks_atomically() -> None:
    registry = SourceRegistry()

    with pytest.raises(NotImplementedError, match="has no dataset_loader"):
        register_us_source_impute_blocks(
            registry,
            manifest_path=_real_scf_manifest_path(),
            blocks=("scf", "sipp_tips"),
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
    manifest["blocks"]["scf"]["dataset_loader"]["builder_kind"] = "unsupported_kind"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    block = SourceImputeManifest.from_path(path).block("scf")

    with pytest.raises(NotImplementedError, match="unsupported_kind"):
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


def test_real_us_acs_manifest_block_retains_declared_targets_and_predictors(
    tmp_path: Path,
) -> None:
    block = SourceImputeManifest.from_path(_real_scf_manifest_path()).block("acs")
    expected_predictors = {
        "is_household_head",
        "age",
        "is_male",
        "tenure_type",
        "employment_income",
        "self_employment_income",
        "social_security",
        "pension_income",
        "household_size",
        "state_fips",
    }

    assert set(block.target_variables) == {"rent", "real_estate_taxes"}
    assert set(block.predictors) == expected_predictors

    table = load_source_impute_block_table(
        block,
        dataset_path=_acs_h5_path(tmp_path),
        period=2024,
    )

    assert set(block.target_variables).issubset(table.columns)
    assert set(block.predictors).issubset(table.columns)
    assert set(block.household_variables).issubset(table.columns)
