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
from microplex.spec import (
    BOTH_TOKEN,
    ImputationOrder,
    ImputationPhase,
    ImputationStep,
    MicroplexSpec,
    VariableOperationKind,
)

_SUPPORTED_BUILDER_KINDS = frozenset(
    {"single_person_households", "household_rows", "raw_person_rows"}
)
_RAW_CSV_CHUNK_ROWS = 200_000


@dataclass(frozen=True)
class SourceImputeBlock:
    """One source-imputation block from ``pe_source_impute_blocks.json``."""

    name: str
    survey_name: str
    default_year: int
    dataset_id: str | None
    archetype: str | None
    dataset_loader: Mapping[str, Any] | None
    raw_loader: Mapping[str, Any] | None
    required_monthcode: int | None
    annualized_variables: tuple[str, ...]
    household_count_variables: tuple[str, ...]
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
        raw_loader = raw.get("raw_loader")
        if raw_loader is not None and not isinstance(raw_loader, Mapping):
            raise ValueError(
                f"source-impute block {name!r} raw_loader must be an object"
            )
        required_monthcode = raw.get("required_monthcode")
        return cls(
            name=name,
            survey_name=str(raw["survey_name"]),
            default_year=int(raw["default_year"]),
            dataset_id=str(raw["dataset_id"]) if raw.get("dataset_id") else None,
            archetype=raw.get("archetype"),
            dataset_loader=dataset_loader,
            raw_loader=raw_loader,
            required_monthcode=(
                int(required_monthcode) if required_monthcode is not None else None
            ),
            annualized_variables=tuple(raw.get("annualized_variables") or ()),
            household_count_variables=tuple(raw.get("household_count_variables") or ()),
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
    block_names: tuple[str, ...] | None = None

    @property
    def descriptor(self) -> SourceDescriptor:
        blocks = self._blocks()
        source_name = self.source_name or blocks[0].survey_name
        if len(blocks) == 1:
            return _descriptor_for_block(blocks[0], source_name)
        return _descriptor_for_blocks(blocks, source_name)

    def load_frame(self, query: SourceQuery | None = None) -> ObservationFrame:
        """Load the source block into a validated observation frame."""
        blocks = self._blocks()
        if not blocks:
            raise ValueError("ManifestSourceImputeProvider requires at least one block")
        block = blocks[0]
        period = int(query.period) if query is not None and query.period else None
        normalized_query = (
            SourceQuery(
                period=period,
                provider_filters=dict(query.provider_filters),
            )
            if query is not None and query.period is not None
            else query
        )
        tables = [
            load_source_impute_block_table(
                source_block,
                dataset_path=self._dataset_path(source_block),
                max_rows=self.max_rows,
                period=period or source_block.default_year,
            )
            for source_block in blocks
        ]
        table = (
            tables[0]
            if len(tables) == 1
            else _concat_source_impute_tables(blocks, tables)
        )
        source_name = self.source_name or block.survey_name
        frame = ObservationFrame(
            source=_descriptor_for_table_blocks(blocks, table, source_name),
            tables={EntityType.PERSON: table},
        )
        frame.validate()
        return apply_source_query(frame, normalized_query)

    def _block(self) -> SourceImputeBlock:
        return self._blocks()[0]

    def _blocks(self) -> tuple[SourceImputeBlock, ...]:
        manifest = SourceImputeManifest.from_path(self.manifest_path)
        block_names = self.block_names or (self.block_name,)
        return tuple(manifest.block(block_name) for block_name in block_names)

    def _dataset_path(self, block: SourceImputeBlock) -> Path:
        if self.dataset_path is not None:
            if self.block_names and len(self.block_names) > 1:
                raise ValueError(
                    "dataset_path can only be used with a single source-impute block"
                )
            path = Path(self.dataset_path)
        else:
            loader = _supported_loader(block)
            if "filename" in loader:
                filename = str(loader["filename"])
            elif "class_name" in loader:
                class_name = str(loader["class_name"])
                filename = _dataset_filename_from_class_name(class_name)
            else:
                raise NotImplementedError(
                    f"source-impute block {block.name!r} has no dataset filename"
                )
            storage_dir = Path(self.storage_dir) if self.storage_dir else Path.cwd()
            path = storage_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"source-impute dataset not found: {path}")
        return path


def compile_source_impute_steps_from_manifest(
    spec: MicroplexSpec,
    manifest: SourceImputeManifest | str | Path,
    *,
    imputation_steps: Sequence[str] | None = None,
    onto: str = BOTH_TOKEN,
    at: ImputationPhase = ImputationPhase.HALVES,
    order: ImputationOrder = ImputationOrder.AS_DECLARED,
    weights: str | None = None,
    synthesize: bool = False,
) -> list[ImputationStep]:
    """Compile executable source-impute variable operations into steps.

    Country packs declare source-imputation behavior in ``variables[*].mp_spec``
    while the donor block manifest declares each source frame's target and
    predictor surface. This helper joins those two declarative inputs and
    returns ordinary :class:`~microplex.spec.ImputationStep` objects that the
    generic runner can execute after the support spine is built.

    Only ``kind: impute`` rows are executable here. ``open_decision`` rows stay
    inert until the pack resolves them to a concrete operation, so loading an
    ACS donor surface for rent/property-tax does not make those variables run
    automatically.
    """
    source_manifest = (
        manifest
        if isinstance(manifest, SourceImputeManifest)
        else SourceImputeManifest.from_path(manifest)
    )
    requested_steps = set(imputation_steps) if imputation_steps is not None else None
    blocks_by_survey: dict[str, list[SourceImputeBlock]] = {}
    for block in source_manifest.blocks.values():
        blocks_by_survey.setdefault(block.survey_name, []).append(block)

    grouped_variables: dict[tuple[str, str], list[str]] = {}
    grouped_blocks: dict[tuple[str, str], SourceImputeBlock] = {}
    unresolved: list[str] = []

    for variable_name, variable in spec.variables.items():
        operation = variable.mp_spec.operation if variable.mp_spec else None
        if operation is None or operation.kind is not VariableOperationKind.IMPUTE:
            continue
        if operation.imputation_step is None or operation.source is None:
            continue
        if (
            requested_steps is not None
            and operation.imputation_step not in requested_steps
        ):
            continue

        candidate_blocks = blocks_by_survey.get(operation.source, [])
        if not candidate_blocks:
            if requested_steps is not None:
                unresolved.append(
                    f"{variable_name} ({operation.imputation_step} from "
                    f"{operation.source}: no manifest block)"
                )
            continue
        matching_blocks = [
            block
            for block in candidate_blocks
            if variable_name in block.target_variables
        ]
        if not matching_blocks:
            unresolved.append(
                f"{variable_name} ({operation.imputation_step} from "
                f"{operation.source}: not a manifest target)"
            )
            continue
        if len(matching_blocks) > 1:
            block_names = [block.name for block in matching_blocks]
            raise ValueError(
                "source-impute variable operation is ambiguous across manifest "
                f"blocks: {variable_name} appears in {block_names}"
            )

        block = matching_blocks[0]
        key = (operation.imputation_step, block.name)
        grouped_variables.setdefault(key, []).append(variable_name)
        grouped_blocks[key] = block

    if unresolved:
        raise ValueError(
            "source-impute variable operations are not backed by manifest target "
            f"variables: {unresolved}"
        )

    steps: list[ImputationStep] = []
    for key, variables in grouped_variables.items():
        block = grouped_blocks[key]
        steps.append(
            ImputationStep(
                onto=onto,
                **{"from": block.survey_name},
                vars=variables,
                condition_on=list(block.predictors),
                at=at,
                order=order,
                weights=weights,
                synthesize=synthesize,
            )
        )
    return steps


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
    elif builder_kind == "raw_person_rows":
        table = _load_raw_person_rows_table(
            block,
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
    if block.dataset_loader is not None:
        loader = block.dataset_loader
    elif block.raw_loader is not None:
        loader = {"builder_kind": "raw_person_rows", **block.raw_loader}
        if "int_columns" in loader and "int_person_columns" not in loader:
            loader["int_person_columns"] = loader["int_columns"]
    else:
        raise NotImplementedError(
            f"source-impute block {block.name!r} has no dataset_loader or raw_loader"
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


def _load_raw_person_rows_table(
    block: SourceImputeBlock,
    loader: Mapping[str, Any],
    *,
    dataset_path: Path,
    max_rows: int | None,
) -> pd.DataFrame:
    household_count_maps = _raw_household_count_maps(
        block,
        loader=loader,
        dataset_path=dataset_path,
        max_rows=max_rows,
    )
    raw = _read_raw_csv_person_rows(
        dataset_path,
        loader=loader,
        max_rows=max_rows,
        required_monthcode=block.required_monthcode,
    )
    raw = _filter_required_monthcode(raw, block.required_monthcode)
    if raw.empty:
        raise ValueError(
            f"source-impute raw block {block.name!r} contains no rows after filters"
        )
    if max_rows is not None:
        raw = raw.head(max_rows).copy()

    direct_columns = _string_mapping(loader.get("direct_columns"))
    table = pd.DataFrame(index=raw.index)
    for target, source in direct_columns.items():
        _require_raw_columns(raw, [source], context=f"direct column {target!r}")
        table[target] = raw[source]

    for target, token in _string_mapping(loader.get("sum_columns_contains")).items():
        matching = [column for column in raw.columns if token in str(column)]
        if not matching:
            raise ValueError(
                f"source-impute raw block {block.name!r} found no columns "
                f"containing {token!r} for {target!r}"
            )
        table[target] = raw.loc[:, matching].fillna(0).sum(axis=1)

    for target, spec in _nested_mapping(loader.get("indicator_columns")).items():
        source = str(spec.get("column", ""))
        if not source:
            raise ValueError(f"indicator column {target!r} missing source column")
        _require_raw_columns(raw, [source], context=f"indicator column {target!r}")
        table[target] = raw[source] == spec.get("equals")

    _fillna_columns(table, loader)

    for column in block.annualized_variables:
        column = str(column)
        if column in table.columns:
            table[column] = table[column] * 12

    household_id_parts = _string_sequence(loader.get("household_id_parts"))
    person_id_parts = _string_sequence(loader.get("person_id_parts"))
    _require_raw_columns(raw, household_id_parts, context="household_id_parts")
    _require_raw_columns(raw, person_id_parts, context="person_id_parts")
    table["household_id"] = _compose_key(raw, household_id_parts, column="household_id")
    table["person_id"] = _compose_key(raw, person_id_parts, column="person_id")
    table["tax_unit_id"] = table["household_id"]

    for column, value in (loader.get("constant_columns") or {}).items():
        table[str(column)] = value

    _add_raw_household_count_variables(
        block,
        raw=raw,
        table=table,
        household_id_parts=household_id_parts,
        count_maps=household_count_maps,
    )
    _copy_person_columns(table, {"copy_person_columns": loader.get("copy_columns")})
    return table.reset_index(drop=True)


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


def _read_raw_csv_person_rows(
    path: Path,
    *,
    loader: Mapping[str, Any],
    max_rows: int | None,
    required_monthcode: int | None,
) -> pd.DataFrame:
    read_kwargs = _raw_csv_read_kwargs(loader)
    if max_rows is not None and required_monthcode is None:
        read_kwargs["nrows"] = max_rows
        return pd.read_csv(path, **read_kwargs)
    if max_rows is not None:
        chunks: list[pd.DataFrame] = []
        empty_template: pd.DataFrame | None = None
        for chunk in pd.read_csv(path, chunksize=_RAW_CSV_CHUNK_ROWS, **read_kwargs):
            filtered = _filter_required_monthcode(chunk, required_monthcode)
            if empty_template is None:
                empty_template = filtered.head(0)
            if filtered.empty:
                continue
            chunks.append(filtered)
            if sum(len(chunk) for chunk in chunks) >= max_rows:
                break
        if not chunks:
            return empty_template if empty_template is not None else pd.DataFrame()
        return pd.concat(chunks, ignore_index=True).head(max_rows)
    return pd.read_csv(path, **read_kwargs)


def _raw_csv_read_kwargs(loader: Mapping[str, Any]) -> dict[str, Any]:
    usecols = _string_sequence(loader.get("usecols"))
    read_kwargs: dict[str, Any] = {}
    delimiter = loader.get("delimiter")
    if delimiter:
        read_kwargs["sep"] = str(delimiter)
    if usecols:
        read_kwargs["usecols"] = usecols
    return read_kwargs


def _filter_required_monthcode(
    raw: pd.DataFrame, required_monthcode: int | None
) -> pd.DataFrame:
    if required_monthcode is None:
        return raw
    if "MONTHCODE" not in raw.columns:
        raise ValueError(
            "source-impute raw loader requires MONTHCODE for required_monthcode"
        )
    month = pd.to_numeric(raw["MONTHCODE"], errors="coerce")
    return raw.loc[month == required_monthcode].copy()


def _raw_household_count_maps(
    block: SourceImputeBlock,
    *,
    loader: Mapping[str, Any],
    dataset_path: Path,
    max_rows: int | None,
) -> dict[str, dict[Any, int]]:
    if not block.household_count_variables or max_rows is None:
        return {}
    direct_columns = _string_mapping(loader.get("direct_columns"))
    age_source = direct_columns.get("age")
    if age_source is None:
        raise ValueError(
            f"source-impute block {block.name!r} cannot derive household counts "
            "without a raw age source"
        )
    household_id_parts = _string_sequence(loader.get("household_id_parts"))
    needed_columns = set(household_id_parts) | {age_source}
    if block.required_monthcode is not None:
        needed_columns.add("MONTHCODE")
    read_kwargs = _raw_csv_read_kwargs({**loader, "usecols": sorted(needed_columns)})
    counts = {variable: {} for variable in block.household_count_variables}
    saw_rows = False
    for chunk in pd.read_csv(
        dataset_path, chunksize=_RAW_CSV_CHUNK_ROWS, **read_kwargs
    ):
        chunk = _filter_required_monthcode(chunk, block.required_monthcode)
        if chunk.empty:
            continue
        saw_rows = True
        _require_raw_columns(chunk, household_id_parts, context="household_id_parts")
        _require_raw_columns(chunk, [age_source], context="household count age")
        household_key = _compose_key(chunk, household_id_parts, column="household_id")
        age = pd.to_numeric(chunk[age_source], errors="coerce")
        if age.isna().any():
            raise ValueError(
                f"source-impute block {block.name!r} cannot derive household counts "
                "from non-numeric ages"
            )
        for variable in block.household_count_variables:
            threshold = _count_under_threshold(str(variable))
            grouped = (
                (age < threshold)
                .groupby(household_key, sort=False)
                .sum()
                .astype("int64")
            )
            variable_counts = counts[str(variable)]
            for key, value in grouped.items():
                variable_counts[key] = variable_counts.get(key, 0) + int(value)
    if not saw_rows:
        raise ValueError(
            f"source-impute raw block {block.name!r} contains no rows after filters"
        )
    return {str(variable): mapping for variable, mapping in counts.items()}


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


def _compose_key(
    raw: pd.DataFrame, parts: tuple[str, ...], *, column: str
) -> pd.Series:
    if not parts:
        raise ValueError(f"source-impute {column} requires at least one part")
    if raw.loc[:, list(parts)].isna().any(axis=None):
        raise ValueError(f"source-impute {column} contains null keys")
    if len(parts) == 1:
        key = raw[parts[0]]
    else:
        key = raw.loc[:, list(parts)].astype(str).agg("|".join, axis=1)
    return pd.Series(key, index=raw.index)


def _add_raw_household_count_variables(
    block: SourceImputeBlock,
    *,
    raw: pd.DataFrame,
    table: pd.DataFrame,
    household_id_parts: tuple[str, ...],
    count_maps: Mapping[str, Mapping[Any, int]] | None = None,
) -> None:
    if not block.household_count_variables:
        return
    household_key = _compose_key(raw, household_id_parts, column="household_id")
    if count_maps:
        for variable in block.household_count_variables:
            variable = str(variable)
            counts = pd.Series(count_maps.get(variable, {}))
            mapped_counts = household_key.map(counts)
            if mapped_counts.isna().any():
                missing = sorted(
                    {
                        str(value)
                        for value in household_key[mapped_counts.isna()].head(5)
                    }
                )
                raise ValueError(
                    f"source-impute block {block.name!r} missing full household "
                    f"counts for {variable!r}: {missing}"
                )
            table[variable] = mapped_counts.astype("int64")
        return
    if "age" not in table.columns:
        raise ValueError(
            f"source-impute block {block.name!r} cannot derive household counts "
            "without an age column"
        )
    age = pd.to_numeric(table["age"], errors="coerce")
    if age.isna().any():
        raise ValueError(
            f"source-impute block {block.name!r} cannot derive household counts "
            "from non-numeric ages"
        )
    for variable in block.household_count_variables:
        threshold = _count_under_threshold(str(variable))
        counts = (age < threshold).groupby(household_key, sort=False).transform("sum")
        table[str(variable)] = counts.astype("int64")


def _count_under_threshold(variable: str) -> int:
    prefix = "count_under_"
    if not variable.startswith(prefix):
        raise ValueError(
            f"Unsupported household_count_variables entry {variable!r}; "
            "expected count_under_<age>"
        )
    threshold = variable.removeprefix(prefix)
    if not threshold.isdigit():
        raise ValueError(
            f"Unsupported household_count_variables entry {variable!r}; "
            "expected numeric age threshold"
        )
    return int(threshold)


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


def _fillna_columns(table: pd.DataFrame, loader: Mapping[str, Any]) -> None:
    fill_values = loader.get("fillna_columns")
    if fill_values is None:
        return
    if not isinstance(fill_values, Mapping):
        raise ValueError("source-impute fillna_columns must be an object")
    for column, value in fill_values.items():
        column = str(column)
        if column not in table.columns:
            raise ValueError(
                f"Cannot fill source-impute column {column!r}; column missing"
            )
        table[column] = table[column].fillna(value)


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


def _concat_source_impute_tables(
    blocks: Sequence[SourceImputeBlock],
    tables: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    table = pd.concat(tables, ignore_index=True, sort=False)
    if table.empty:
        block_names = [block.name for block in blocks]
        raise ValueError(f"source-impute blocks {block_names} loaded no rows")
    if table["person_id"].isna().any():
        raise ValueError("combined source-impute table contains null person_id")
    if table["person_id"].duplicated().any():
        duplicates = sorted(
            {
                str(value)
                for value in table["person_id"][table["person_id"].duplicated()].head(5)
            }
        )
        raise ValueError(
            "combined source-impute table contains duplicate person_id values: "
            f"{duplicates}"
        )
    return table


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


def _descriptor_for_blocks(
    blocks: Sequence[SourceImputeBlock], source_name: str
) -> SourceDescriptor:
    variables = tuple(
        dict.fromkeys(
            variable
            for block in blocks
            for variable in (
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
        archetype=_source_archetype(blocks[0].archetype),
        population=f"US {blocks[0].survey_name.upper()} source-imputation donor",
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
    return _descriptor_for_table_blocks((block,), table, source_name)


def _descriptor_for_table_blocks(
    blocks: Sequence[SourceImputeBlock],
    table: pd.DataFrame,
    source_name: str,
) -> SourceDescriptor:
    excluded = {"person_id", "weight", "year"}
    variable_names = tuple(column for column in table.columns if column not in excluded)
    return SourceDescriptor(
        name=source_name,
        shareability=Shareability.PUBLIC,
        time_structure=TimeStructure.CROSS_SECTION,
        archetype=_source_archetype(blocks[0].archetype),
        population=f"US {blocks[0].survey_name.upper()} source-imputation donor",
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


def _string_sequence(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise ValueError("source-impute sequence must be a list")
    return tuple(str(value) for value in raw)


def _require_raw_columns(
    raw: pd.DataFrame,
    columns: Sequence[str],
    *,
    context: str,
) -> None:
    missing = sorted(set(columns) - set(raw.columns))
    if missing:
        raise ValueError(f"source-impute raw {context} missing columns: {missing}")


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
