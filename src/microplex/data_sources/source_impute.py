"""Manifest-backed source-imputation donor loaders."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microplex.core import EntityType
from microplex.core.sources import (
    EntityObservation,
    ObservationFrame,
    Shareability,
    SourceArchetype,
    SourceDescriptor,
    SourceQuery,
    TimeStructure,
    apply_source_query,
)

_SUPPORTED_BUILDER_KINDS = frozenset({"single_person_households", "household_rows"})


@dataclass(frozen=True)
class SourceImputeBlock:
    """One source-imputation block from ``pe_source_impute_blocks.json``."""

    name: str
    survey_name: str
    default_year: int
    dataset_id: str | None
    archetype: str | None
    dataset_loader: Mapping[str, Any] | None
    household_variables: tuple[str, ...]
    person_variables: tuple[str, ...]
    target_variables: tuple[str, ...]
    predictors: tuple[str, ...]

    @classmethod
    def from_mapping(cls, name: str, raw: Mapping[str, Any]) -> SourceImputeBlock:
        """Build a typed block descriptor from manifest JSON."""
        dataset_loader = raw.get("dataset_loader")
        if dataset_loader is not None and not isinstance(dataset_loader, Mapping):
            raise ValueError(
                f"source-impute block {name!r} dataset_loader must be an object"
            )
        return cls(
            name=name,
            survey_name=str(raw["survey_name"]),
            default_year=int(raw["default_year"]),
            dataset_id=str(raw["dataset_id"]) if raw.get("dataset_id") else None,
            archetype=raw.get("archetype"),
            dataset_loader=dataset_loader,
            household_variables=tuple(raw.get("household_variables") or ()),
            person_variables=tuple(raw.get("person_variables") or ()),
            target_variables=tuple(raw.get("target_variables") or ()),
            predictors=tuple(raw.get("predictors") or ()),
        )


@dataclass(frozen=True)
class SourceImputeManifest:
    """Typed view of a source-imputation block manifest."""

    blocks: Mapping[str, SourceImputeBlock]

    @classmethod
    def from_path(cls, path: str | Path) -> SourceImputeManifest:
        """Load ``pe_source_impute_blocks.json`` from disk."""
        manifest_path = Path(path)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"source-impute manifest not found: {manifest_path}"
            )
        raw = json.loads(manifest_path.read_text())
        raw_blocks = raw.get("blocks")
        if not isinstance(raw_blocks, Mapping) or not raw_blocks:
            raise ValueError("source-impute manifest must contain non-empty blocks")
        return cls(
            blocks={
                str(name): SourceImputeBlock.from_mapping(str(name), block)
                for name, block in raw_blocks.items()
            }
        )

    def block(self, name: str) -> SourceImputeBlock:
        """Return one source-imputation block by manifest key."""
        try:
            return self.blocks[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.blocks))
            raise KeyError(
                f"source-impute block {name!r} not found; available: {available}"
            ) from exc


@dataclass
class ManifestSourceImputeProvider:
    """Source provider backed by the pack source-imputation manifest."""

    manifest_path: str | Path
    block_name: str
    storage_dir: str | Path | None = None
    dataset_path: str | Path | None = None
    max_rows: int | None = None
    source_name: str | None = None

    @property
    def descriptor(self) -> SourceDescriptor:
        block = self._block()
        return _descriptor_for_block(block, self.source_name or block.survey_name)

    def load_frame(self, query: SourceQuery | None = None) -> ObservationFrame:
        """Load the source block into a validated observation frame."""
        block = self._block()
        period = int(query.period) if query is not None and query.period else None
        normalized_query = (
            SourceQuery(
                period=period,
                provider_filters=dict(query.provider_filters),
            )
            if query is not None and query.period is not None
            else query
        )
        table = load_source_impute_block_table(
            block,
            dataset_path=self._dataset_path(block),
            max_rows=self.max_rows,
            period=period or block.default_year,
        )
        source_name = self.source_name or block.survey_name
        frame = ObservationFrame(
            source=_descriptor_for_table(block, table, source_name),
            tables={EntityType.PERSON: table},
        )
        frame.validate()
        return apply_source_query(frame, normalized_query)

    def _block(self) -> SourceImputeBlock:
        return SourceImputeManifest.from_path(self.manifest_path).block(self.block_name)

    def _dataset_path(self, block: SourceImputeBlock) -> Path:
        if self.dataset_path is not None:
            path = Path(self.dataset_path)
        else:
            if block.dataset_loader is None:
                raise NotImplementedError(
                    f"source-impute block {block.name!r} has no dataset_loader"
                )
            storage_dir = Path(self.storage_dir) if self.storage_dir else Path.cwd()
            class_name = str(block.dataset_loader["class_name"])
            filename = _dataset_filename_from_class_name(class_name)
            path = storage_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"source-impute dataset not found: {path}")
        return path


def load_source_impute_block_table(
    block: SourceImputeBlock,
    *,
    dataset_path: str | Path,
    max_rows: int | None = None,
    period: int | None = None,
) -> pd.DataFrame:
    """Load one manifest block from an array-style HDF5 dataset."""
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive when supplied")
    loader = _supported_loader(block)
    builder_kind = str(loader.get("builder_kind"))
    if builder_kind == "single_person_households":
        table = _load_single_person_household_table(
            loader,
            dataset_path=Path(dataset_path),
            max_rows=max_rows,
        )
    elif builder_kind == "household_rows":
        table = _load_household_rows_table(
            loader,
            dataset_path=Path(dataset_path),
            max_rows=max_rows,
        )
    else:  # pragma: no cover - guarded by _supported_loader
        raise NotImplementedError(
            f"source-impute builder_kind {builder_kind!r} is not implemented"
        )
    return _finalize_source_impute_table(block, loader, table, period=period)


def validate_source_impute_block_supported(block: SourceImputeBlock) -> None:
    """Fail unless ``block`` has a loader shape implemented by Microplex."""
    _supported_loader(block)


def _supported_loader(block: SourceImputeBlock) -> Mapping[str, Any]:
    loader = block.dataset_loader
    if loader is None:
        raise NotImplementedError(
            f"source-impute block {block.name!r} has no dataset_loader"
        )
    builder_kind = str(loader.get("builder_kind") or "")
    if builder_kind not in _SUPPORTED_BUILDER_KINDS:
        raise NotImplementedError(
            f"source-impute builder_kind {builder_kind!r} is not implemented"
        )
    return loader


def _load_single_person_household_table(
    loader: Mapping[str, Any],
    *,
    dataset_path: Path,
    max_rows: int | None,
) -> pd.DataFrame:
    direct_columns = _string_mapping(loader.get("direct_person_columns"))
    boolean_columns = _string_mapping(loader.get("boolean_person_columns"))
    fallback_columns = _fallback_mapping(loader.get("fallback_person_columns"))
    available, values = _read_h5_mapped_arrays(
        dataset_path,
        direct_columns=direct_columns,
        boolean_columns=boolean_columns,
        fallback_columns=fallback_columns,
        max_rows=max_rows,
    )
    length = _common_length(values)
    table = pd.DataFrame(index=pd.RangeIndex(length))

    for target, source in direct_columns.items():
        table[target] = values[source]
    for target, source in boolean_columns.items():
        table[target] = _coerce_boolean_array(values[source], column=target)
    for target, sources in fallback_columns.items():
        source = next(
            (candidate for candidate in sources if candidate in available), None
        )
        if source is None:
            raise ValueError(
                f"No fallback source columns found for {target!r}: {list(sources)}"
            )
        table[target] = values[source]

    for column, value in (loader.get("constant_person_columns") or {}).items():
        table[str(column)] = value

    _add_income_sum_columns(table, loader)
    _add_group_count_person_columns(table, loader)
    _copy_person_columns(table, loader)
    _add_sex_from_boolean_source(table, loader)
    _add_single_person_ids(table)
    return table


def _load_household_rows_table(
    loader: Mapping[str, Any],
    *,
    dataset_path: Path,
    max_rows: int | None,
) -> pd.DataFrame:
    person_id_key = _required_loader_string(loader, "person_id_key")
    person_household_key = _required_loader_string(loader, "person_household_key")
    household_index_key = _required_loader_string(loader, "household_index_key")
    direct_columns = _string_mapping(loader.get("direct_person_columns"))
    boolean_columns = _string_mapping(loader.get("boolean_person_columns"))
    fallback_columns = _fallback_mapping(loader.get("fallback_person_columns"))
    row_indexed_columns = _string_mapping(loader.get("row_indexed_person_columns"))
    mapped_row_columns = _string_mapping(loader.get("mapped_row_person_columns"))
    mapped_value_tables = _nested_mapping(loader.get("mapped_value_tables"))
    group_count_columns = _string_mapping(loader.get("group_count_person_columns"))

    person_sources = (
        {person_id_key, person_household_key}
        | set(direct_columns.values())
        | set(boolean_columns.values())
        | set(group_count_columns.values())
    )
    fallback_sources = {
        source for sources in fallback_columns.values() for source in sources
    }
    household_sources = (
        {household_index_key}
        | set(row_indexed_columns.values())
        | set(mapped_row_columns.values())
    )
    available, person_values, household_values, full_person_values = (
        _read_h5_person_household_arrays(
            dataset_path,
            person_sources=person_sources,
            fallback_person_sources=fallback_sources,
            household_sources=household_sources,
            full_person_sources=set(group_count_columns.values()),
            max_person_rows=max_rows,
        )
    )
    length = _common_length(person_values)
    _common_length(household_values)
    if full_person_values:
        _common_length(full_person_values)
    table = pd.DataFrame(index=pd.RangeIndex(length))

    table["person_id"] = person_values[person_id_key]
    table["household_id"] = person_values[person_household_key]
    table["tax_unit_id"] = table["household_id"]
    if person_household_key not in table.columns:
        table[person_household_key] = person_values[person_household_key]

    for target, source in direct_columns.items():
        table[target] = person_values[source]
    for target, source in boolean_columns.items():
        table[target] = _coerce_boolean_array(person_values[source], column=target)
    for target, sources in fallback_columns.items():
        source = next(
            (candidate for candidate in sources if candidate in available), None
        )
        if source is None:
            raise ValueError(
                f"No fallback source columns found for {target!r}: {list(sources)}"
            )
        table[target] = person_values[source]

    household_index = _household_index(
        household_values[household_index_key],
        column=household_index_key,
    )
    positions = _household_positions(
        person_household_ids=table["household_id"],
        household_index=household_index,
        person_household_key=person_household_key,
    )
    for target, source in row_indexed_columns.items():
        table[target] = household_values[source][positions]
    for target, source in mapped_row_columns.items():
        raw_values = household_values[source][positions]
        table[target] = _apply_mapped_value_table(
            raw_values,
            target=target,
            source=source,
            mapped_value_tables=mapped_value_tables,
        )

    for column, value in (loader.get("constant_person_columns") or {}).items():
        table[str(column)] = value

    _add_income_sum_columns(table, loader)
    _add_group_count_person_columns(
        table,
        loader,
        full_source_values=full_person_values,
    )
    _copy_person_columns(table, loader)
    _add_sex_from_boolean_source(table, loader)
    return table


def _finalize_source_impute_table(
    block: SourceImputeBlock,
    loader: Mapping[str, Any],
    table: pd.DataFrame,
    *,
    period: int | None,
) -> pd.DataFrame:
    table["year"] = int(period if period is not None else block.default_year)
    for column in loader.get("int_person_columns") or ():
        column = str(column)
        if column in table.columns:
            table[column] = _coerce_int_column(table[column], column=column)

    declared_columns = {"person_id", "household_id", "tax_unit_id", "year", "weight"}
    expected = (
        set(block.person_variables)
        | set(block.household_variables)
        | set(block.target_variables)
        | set(block.predictors)
        | declared_columns
    )
    missing_targets = sorted(set(block.target_variables) - set(table.columns))
    if missing_targets:
        raise ValueError(
            f"source-impute block {block.name!r} missing target variables: "
            f"{missing_targets}"
        )
    missing_predictors = sorted(set(block.predictors) - set(table.columns))
    if missing_predictors:
        raise ValueError(
            f"source-impute block {block.name!r} missing predictor variables: "
            f"{missing_predictors}"
        )
    missing_declared = sorted(expected - set(table.columns))
    if missing_declared:
        raise ValueError(
            f"source-impute block {block.name!r} missing declared variables: "
            f"{missing_declared}"
        )
    return table.loc[:, [column for column in table.columns if column in expected]]


def _read_h5_mapped_arrays(
    path: Path,
    *,
    direct_columns: Mapping[str, str],
    boolean_columns: Mapping[str, str],
    fallback_columns: Mapping[str, tuple[str, ...]],
    max_rows: int | None,
) -> tuple[frozenset[str], dict[str, np.ndarray]]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised only without dependency
        raise ImportError("h5py is required to load array-style HDF5 sources") from exc

    with h5py.File(path, "r") as h5:
        available = frozenset(str(key) for key in h5.keys())
        required = set(direct_columns.values()) | set(boolean_columns.values())
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"source-impute H5 missing required arrays: {missing}")

        fallback_sources = {
            source
            for sources in fallback_columns.values()
            for source in sources
            if source in available
        }
        values = {
            source: _read_h5_dataset(h5[source], max_rows=max_rows)
            for source in sorted(required | fallback_sources)
        }
    return available, values


def _read_h5_person_household_arrays(
    path: Path,
    *,
    person_sources: set[str],
    fallback_person_sources: set[str],
    household_sources: set[str],
    full_person_sources: set[str],
    max_person_rows: int | None,
) -> tuple[
    frozenset[str],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised only without dependency
        raise ImportError("h5py is required to load array-style HDF5 sources") from exc

    with h5py.File(path, "r") as h5:
        available = frozenset(str(key) for key in h5.keys())
        required = person_sources | household_sources | full_person_sources
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"source-impute H5 missing required arrays: {missing}")

        person_values = {
            source: _read_h5_dataset(h5[source], max_rows=max_person_rows)
            for source in sorted(
                person_sources
                | {source for source in fallback_person_sources if source in available}
            )
        }
        household_values = {
            source: _read_h5_dataset(h5[source], max_rows=None)
            for source in sorted(household_sources)
        }
        full_person_values = {
            source: (
                person_values[source]
                if max_person_rows is None and source in person_values
                else _read_h5_dataset(h5[source], max_rows=None)
            )
            for source in sorted(full_person_sources)
        }
    return available, person_values, household_values, full_person_values


def _read_h5_dataset(dataset, *, max_rows: int | None) -> np.ndarray:
    if len(dataset.shape) != 1:
        raise ValueError(
            f"source-impute array {dataset.name!r} must be one-dimensional"
        )
    values = dataset[:max_rows] if max_rows is not None else dataset[()]
    array = np.asarray(values)
    if array.dtype.kind == "S":
        return np.char.decode(array, "utf-8")
    if array.dtype.kind == "O":
        return np.asarray(
            [
                value.decode("utf-8") if isinstance(value, bytes) else value
                for value in array
            ]
        )
    return array


def _common_length(values: Mapping[str, np.ndarray]) -> int:
    lengths = {len(value) for value in values.values()}
    if not lengths:
        raise ValueError("source-impute block loaded no arrays")
    if len(lengths) != 1:
        raise ValueError(f"source-impute arrays have inconsistent lengths: {lengths}")
    length = lengths.pop()
    if length < 1:
        raise ValueError("source-impute arrays must contain at least one row")
    return length


def _coerce_boolean_array(values: np.ndarray, *, column: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind == "b":
        return array.astype(bool)
    if array.dtype.kind in {"i", "u", "f"}:
        numeric = pd.Series(array)
        if numeric.isna().any():
            raise ValueError(f"Boolean source-impute column {column!r} contains nulls")
        invalid = ~numeric.isin([0, 1])
        if invalid.any():
            bad_values = sorted({str(value) for value in numeric[invalid].head(5)})
            raise ValueError(
                f"Boolean source-impute column {column!r} must contain only 0/1; "
                f"found {bad_values}"
            )
        return numeric.astype(bool).to_numpy()

    true_values = {"1", "true", "t", "yes", "y"}
    false_values = {"0", "false", "f", "no", "n"}
    parsed: list[bool] = []
    invalid_values: list[str] = []
    for value in array:
        if isinstance(value, bytes):
            text = value.decode("utf-8")
        else:
            text = str(value)
        normalized = text.strip().lower()
        if normalized in true_values:
            parsed.append(True)
        elif normalized in false_values:
            parsed.append(False)
        else:
            invalid_values.append(text)
            parsed.append(False)
    if invalid_values:
        bad_values = sorted(set(invalid_values[:5]))
        raise ValueError(
            f"Boolean source-impute column {column!r} has unrecognized values: "
            f"{bad_values}"
        )
    return np.asarray(parsed, dtype=bool)


def _coerce_int_column(values: pd.Series, *, column: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        bad_values = sorted({str(value) for value in values[numeric.isna()].head(5)})
        raise ValueError(
            f"Integer source-impute column {column!r} contains non-numeric values: "
            f"{bad_values}"
        )
    numeric_array = numeric.to_numpy(dtype=float)
    if not np.isfinite(numeric_array).all():
        raise ValueError(
            f"Integer source-impute column {column!r} contains non-finite values"
        )
    rounded = np.rint(numeric_array)
    if not np.isclose(numeric_array, rounded).all():
        bad_values = sorted(
            {
                str(value)
                for value in numeric_array[~np.isclose(numeric_array, rounded)][:5]
            }
        )
        raise ValueError(
            f"Integer source-impute column {column!r} contains non-integer values: "
            f"{bad_values}"
        )
    return rounded.astype("int64")


def _household_index(values: np.ndarray, *, column: str) -> pd.Index:
    index = pd.Index(values, name=column)
    if index.hasnans:
        raise ValueError(f"source-impute household index {column!r} contains nulls")
    if not index.is_unique:
        duplicates = sorted({str(value) for value in index[index.duplicated()][:5]})
        raise ValueError(
            f"source-impute household index {column!r} contains duplicate ids: "
            f"{duplicates}"
        )
    return index


def _household_positions(
    *,
    person_household_ids: pd.Series,
    household_index: pd.Index,
    person_household_key: str,
) -> np.ndarray:
    positions = household_index.get_indexer(person_household_ids)
    missing_mask = positions < 0
    if missing_mask.any():
        missing = sorted(
            {str(value) for value in person_household_ids[missing_mask].head(5)}
        )
        raise ValueError(
            f"source-impute person household key {person_household_key!r} "
            f"references missing household ids: {missing}"
        )
    return positions


def _add_income_sum_columns(
    table: pd.DataFrame,
    loader: Mapping[str, Any],
) -> None:
    columns = tuple(str(column) for column in loader.get("income_sum_columns") or ())
    if not columns or "income" in table.columns:
        return
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"source-impute income_sum_columns missing: {missing}")
    table["income"] = table.loc[:, list(columns)].sum(axis=1)


def _add_group_count_person_columns(
    table: pd.DataFrame,
    loader: Mapping[str, Any],
    *,
    full_source_values: Mapping[str, np.ndarray] | None = None,
) -> None:
    for target, source in _string_mapping(
        loader.get("group_count_person_columns")
    ).items():
        if source not in table.columns:
            raise ValueError(
                f"Cannot derive source-impute group count {target!r}; "
                f"source {source!r} missing"
            )
        if full_source_values is not None and source in full_source_values:
            full_values = pd.Series(full_source_values[source])
            if full_values.isna().any():
                raise ValueError(
                    f"Cannot derive source-impute group count {target!r}; "
                    f"source {source!r} contains nulls"
                )
            counts = full_values.value_counts(sort=False)
            mapped_counts = table[source].map(counts)
            if mapped_counts.isna().any():
                missing = sorted(
                    {
                        str(value)
                        for value in table[source][mapped_counts.isna()].head(5)
                    }
                )
                raise ValueError(
                    f"Cannot derive source-impute group count {target!r}; "
                    f"source {source!r} values missing from full counts: {missing}"
                )
            table[target] = mapped_counts.astype("int64")
        else:
            table[target] = (
                table.groupby(source, sort=False)[source]
                .transform("size")
                .astype("int64")
            )


def _copy_person_columns(table: pd.DataFrame, loader: Mapping[str, Any]) -> None:
    for target, source in _string_mapping(loader.get("copy_person_columns")).items():
        if source not in table.columns:
            raise ValueError(
                f"Cannot copy source-impute column {source!r} to {target!r}; source missing"
            )
        table[target] = table[source]


def _apply_mapped_value_table(
    values: np.ndarray,
    *,
    target: str,
    source: str,
    mapped_value_tables: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    mapping = mapped_value_tables.get(target) or mapped_value_tables.get(source)
    if mapping is None:
        raise ValueError(
            f"mapped row source-impute column {target!r} from {source!r} "
            "has no mapped_value_tables entry"
        )
    normalized_values = pd.Series([_mapping_key(value) for value in values])
    missing_mask = ~normalized_values.isin(mapping)
    if missing_mask.any():
        bad_values = sorted(
            {str(value) for value in normalized_values[missing_mask].head(5)}
        )
        raise ValueError(
            f"mapped row source-impute column {target!r} has unmapped values: "
            f"{bad_values}"
        )
    return normalized_values.map(mapping).to_numpy()


def _mapping_key(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _add_sex_from_boolean_source(
    table: pd.DataFrame, loader: Mapping[str, Any]
) -> None:
    source = loader.get("sex_from_boolean_source")
    if not source or "sex" in table.columns:
        return
    source = str(source)
    if source not in table.columns:
        raise ValueError(f"sex_from_boolean_source {source!r} missing from table")
    true_value = int(loader.get("sex_true_value", 1))
    false_value = int(loader.get("sex_false_value", 0))
    table["sex"] = np.where(table[source].astype(bool), true_value, false_value)


def _add_single_person_ids(table: pd.DataFrame) -> None:
    ids = pd.RangeIndex(len(table))
    for column in ("household_id", "person_id", "tax_unit_id"):
        if column not in table.columns:
            table[column] = ids


def _descriptor_for_block(
    block: SourceImputeBlock, source_name: str
) -> SourceDescriptor:
    variables = tuple(
        dict.fromkeys(
            (
                *block.person_variables,
                *block.household_variables,
                *block.predictors,
                *block.target_variables,
                "household_id",
                "tax_unit_id",
            )
        )
    )
    return SourceDescriptor(
        name=source_name,
        shareability=Shareability.PUBLIC,
        time_structure=TimeStructure.CROSS_SECTION,
        archetype=_source_archetype(block.archetype),
        population=f"US {block.survey_name.upper()} source-imputation donor",
        observations=(
            EntityObservation(
                entity=EntityType.PERSON,
                key_column="person_id",
                variable_names=tuple(
                    name for name in variables if name not in {"person_id", "weight"}
                ),
                weight_column="weight",
                period_column="year",
            ),
        ),
    )


def _descriptor_for_table(
    block: SourceImputeBlock,
    table: pd.DataFrame,
    source_name: str,
) -> SourceDescriptor:
    excluded = {"person_id", "weight", "year"}
    variable_names = tuple(column for column in table.columns if column not in excluded)
    return SourceDescriptor(
        name=source_name,
        shareability=Shareability.PUBLIC,
        time_structure=TimeStructure.CROSS_SECTION,
        archetype=_source_archetype(block.archetype),
        population=f"US {block.survey_name.upper()} source-imputation donor",
        observations=(
            EntityObservation(
                entity=EntityType.PERSON,
                key_column="person_id",
                variable_names=variable_names,
                weight_column="weight",
                period_column="year",
            ),
        ),
    )


def _source_archetype(value: str | None) -> SourceArchetype | None:
    if value is None:
        return None
    try:
        return SourceArchetype(value)
    except ValueError:
        return None


def _dataset_filename_from_class_name(class_name: str) -> str:
    text = class_name.strip()
    if not text:
        raise ValueError("dataset_loader.class_name must be non-empty")
    return f"{text.lower()}.h5"


def _string_mapping(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("source-impute mapping must be an object")
    return {str(key): str(value) for key, value in raw.items()}


def _nested_mapping(raw: Any) -> dict[str, dict[str, Any]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("source-impute nested mapping must be an object")
    result: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"source-impute mapping for {key!r} must be an object")
        result[str(key)] = {
            str(inner_key): inner_value for inner_key, inner_value in value.items()
        }
    return result


def _required_loader_string(loader: Mapping[str, Any], key: str) -> str:
    value = loader.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"dataset_loader.{key} must be non-empty")
    return str(value)


def _fallback_mapping(raw: Any) -> dict[str, tuple[str, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("source-impute fallback mapping must be an object")
    result: dict[str, tuple[str, ...]] = {}
    for key, values in raw.items():
        if not isinstance(values, Sequence) or isinstance(values, str):
            raise ValueError(f"fallback values for {key!r} must be a list")
        result[str(key)] = tuple(str(value) for value in values)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-load one manifest-backed source-imputation donor block."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--block", default="scf")
    parser.add_argument("--storage-dir", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--period", type=int, default=2024)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for source-imputation donor smoke loads."""
    args = _build_parser().parse_args(argv)
    provider = ManifestSourceImputeProvider(
        manifest_path=args.manifest,
        block_name=args.block,
        storage_dir=args.storage_dir,
        dataset_path=args.dataset_path,
        max_rows=args.max_rows,
    )
    frame = provider.load_frame(SourceQuery(period=args.period))
    table = frame.tables[EntityType.PERSON]
    payload = {
        "block": args.block,
        "rows": int(len(table)),
        "columns": list(table.columns),
        "target_variables_present": sorted(
            set(provider._block().target_variables) & set(table.columns)
        ),
        "predictors_present": sorted(
            set(provider._block().predictors) & set(table.columns)
        ),
        "weight_sum": float(table["weight"].sum()) if "weight" in table else None,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
