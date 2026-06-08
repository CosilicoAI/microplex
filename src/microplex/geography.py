"""
Atomic geography helpers and provider protocols for microplex.

Core `microplex` owns the generic crosswalk, assignment, provider
abstractions, and reusable concrete loaders. Country content packages stay
declarative and supply specs/manifests rather than runtime Python.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

# Kept as compatibility constants for callers that still use US Census GEOIDs.
STATE_LEN = 2
COUNTY_LEN = 3
TRACT_LEN = 6
BLOCK_LEN = 4

STATE_GEOID_LEN = STATE_LEN
COUNTY_GEOID_LEN = STATE_LEN + COUNTY_LEN
TRACT_GEOID_LEN = STATE_LEN + COUNTY_LEN + TRACT_LEN
BLOCK_GEOID_LEN = STATE_LEN + COUNTY_LEN + TRACT_LEN + BLOCK_LEN

PartitionKey = tuple[Any, ...]
PartitionFallbackResolver = Callable[
    [PartitionKey, tuple[PartitionKey, ...]], PartitionKey
]
PartitionNormalizer = Callable[[Any], Any]


@dataclass(frozen=True)
class AtomicGeographyCrosswalk:
    """Crosswalk from atomic geography units to materialized parent geographies."""

    data: pd.DataFrame
    atomic_id_column: str
    geography_columns: tuple[str, ...] = ()
    probability_column: str | None = None

    def __post_init__(self) -> None:
        if self.atomic_id_column not in self.data.columns:
            raise ValueError(
                f"Atomic geography column '{self.atomic_id_column}' not found in crosswalk"
            )
        if self.data[self.atomic_id_column].duplicated().any():
            raise ValueError("Atomic geography crosswalk must have unique atomic ids")
        if (
            self.probability_column is not None
            and self.probability_column not in self.data.columns
        ):
            raise ValueError(
                f"Probability column '{self.probability_column}' not found in crosswalk"
            )

        geography_columns = self.geography_columns or tuple(
            column
            for column in self.data.columns
            if column not in {self.atomic_id_column, self.probability_column}
        )
        missing_columns = [
            column for column in geography_columns if column not in self.data.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Geography columns not found in crosswalk: {sorted(missing_columns)}"
            )
        object.__setattr__(self, "geography_columns", tuple(geography_columns))

    def lookup(
        self,
        atomic_ids: pd.Series | np.ndarray | list[Any],
        *,
        columns: tuple[str, ...] | list[str] | None = None,
    ) -> pd.DataFrame:
        """Lookup parent geographies for atomic ids."""
        requested_columns = tuple(columns or self.geography_columns)
        lookup = self.data[[self.atomic_id_column, *requested_columns]].copy()
        requested_ids = pd.DataFrame({self.atomic_id_column: list(atomic_ids)})
        return requested_ids.merge(lookup, on=self.atomic_id_column, how="left")

    def materialize(
        self,
        frame: pd.DataFrame,
        *,
        columns: tuple[str, ...] | list[str] | None = None,
        atomic_id_column: str | None = None,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """Attach parent geography columns to a frame that already has atomic ids."""
        join_column = atomic_id_column or self.atomic_id_column
        if join_column not in frame.columns:
            raise ValueError(
                f"Atomic geography column '{join_column}' not found in frame"
            )
        requested_columns = tuple(columns or self.geography_columns)
        if not overwrite:
            requested_columns = tuple(
                column
                for column in requested_columns
                if column not in frame.columns or column == join_column
            )
        if not requested_columns:
            return frame.copy()
        lookup = self.data[[self.atomic_id_column, *requested_columns]].copy()
        if join_column != self.atomic_id_column:
            lookup = lookup.rename(columns={self.atomic_id_column: join_column})
        if overwrite:
            columns_to_drop = [
                column
                for column in requested_columns
                if column != join_column and column in frame.columns
            ]
            base_frame = frame.drop(columns=columns_to_drop)
        else:
            base_frame = frame
        return base_frame.merge(lookup, on=join_column, how="left")


def materialize_geographies(
    frame: pd.DataFrame,
    crosswalk: AtomicGeographyCrosswalk,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    atomic_id_column: str | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Attach geography columns from an atomic-geography crosswalk."""
    return crosswalk.materialize(
        frame,
        columns=columns,
        atomic_id_column=atomic_id_column,
        overwrite=overwrite,
    )


def nearest_numeric_partition_key(
    requested_key: PartitionKey,
    available_keys: tuple[PartitionKey, ...],
) -> PartitionKey:
    """Resolve a missing partition key to the nearest numeric available key."""
    if len(requested_key) != 1:
        raise ValueError(
            "nearest_numeric_partition_key only supports one-column partitions"
        )
    if not available_keys:
        raise ValueError("Cannot resolve nearest key without available partitions")

    requested_value = int(round(float(requested_key[0])))
    distances = np.array(
        [
            abs(int(round(float(candidate[0]))) - requested_value)
            for candidate in available_keys
        ]
    )
    return available_keys[int(np.argmin(distances))]


def normalize_us_state_fips(value: Any) -> str:
    """Compatibility helper for US state FIPS normalization."""
    return str(int(round(float(value)))).zfill(2)


def normalize_string_code(value: Any) -> str:
    """Normalize integer-like categorical geography codes to strings."""
    if isinstance(value, str):
        text = value.strip()
        try:
            numeric = float(text)
        except ValueError:
            return text
        if numeric.is_integer():
            return str(int(numeric))
        return text
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _normalize_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _is_missing_partition_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (np.ndarray, pd.Series, pd.Index, list, tuple)):
        return False
    return bool(missing)


def _partition_values_equal(left: Any, right: Any) -> bool:
    if _is_missing_partition_value(left) or _is_missing_partition_value(right):
        return False
    return bool(left == right)


@dataclass
class ProbabilisticAtomicGeographyAssigner:
    """Assign atomic geography ids from grouped probability distributions."""

    crosswalk: AtomicGeographyCrosswalk
    partition_columns: tuple[str, ...]
    probability_column: str | None = None
    partition_normalizers: dict[str, PartitionNormalizer] = field(default_factory=dict)
    fallback_resolver: PartitionFallbackResolver | None = None

    def __post_init__(self) -> None:
        missing_columns = [
            column
            for column in self.partition_columns
            if column not in self.crosswalk.data.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Partition columns not found in crosswalk: {sorted(missing_columns)}"
            )
        if self.probability_column is None:
            self.probability_column = self.crosswalk.probability_column
        if self.probability_column is None:
            raise ValueError(
                "A probability column is required for probabilistic assignment"
            )
        if self.probability_column not in self.crosswalk.data.columns:
            raise ValueError(
                f"Probability column '{self.probability_column}' not found in crosswalk"
            )
        self._group_lookup = self._build_group_lookup()

    def assign(
        self,
        frame: pd.DataFrame,
        *,
        atomic_id_column: str | None = None,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """Assign atomic geography ids to each row of a frame."""
        missing_columns = [
            column for column in self.partition_columns if column not in frame.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Partition columns not found in frame: {sorted(missing_columns)}"
            )

        rng = np.random.default_rng(random_state)
        assigned_atomic_ids: list[Any] = []
        available_keys = tuple(self._group_lookup.keys())

        for raw_key in frame.loc[:, self.partition_columns].itertuples(
            index=False,
            name=None,
        ):
            normalized_key = self._normalize_partition_key(raw_key)
            lookup_key = normalized_key
            if lookup_key not in self._group_lookup:
                if self.fallback_resolver is None:
                    raise ValueError(
                        f"No atomic geography distribution available for partition key {lookup_key}"
                    )
                lookup_key = self.fallback_resolver(lookup_key, available_keys)
            group = self._group_lookup[lookup_key]
            sampled_index = rng.choice(
                len(group["atomic_ids"]), p=group["probabilities"]
            )
            assigned_atomic_ids.append(group["atomic_ids"][sampled_index])

        result = frame.copy()
        result[atomic_id_column or self.crosswalk.atomic_id_column] = (
            assigned_atomic_ids
        )
        return result

    def _build_group_lookup(self) -> dict[PartitionKey, dict[str, np.ndarray]]:
        group_lookup: dict[PartitionKey, dict[str, np.ndarray]] = {}
        grouped = self.crosswalk.data.groupby(
            list(self.partition_columns), dropna=False, sort=False
        )
        for raw_key, group in grouped:
            key_tuple = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            normalized_key = self._normalize_partition_key(key_tuple)
            probabilities = pd.to_numeric(
                group[self.probability_column],
                errors="coerce",
            ).fillna(0.0)
            total_probability = float(probabilities.sum())
            if total_probability <= 0:
                raise ValueError(
                    f"Partition {normalized_key} has non-positive total probability"
                )
            group_lookup[normalized_key] = {
                "atomic_ids": group[self.crosswalk.atomic_id_column].to_numpy(),
                "probabilities": (probabilities / total_probability).to_numpy(
                    dtype=float
                ),
            }
        return group_lookup

    def _normalize_partition_key(self, raw_key: tuple[Any, ...]) -> PartitionKey:
        normalized: list[Any] = []
        for column, value in zip(self.partition_columns, raw_key, strict=False):
            normalizer = self.partition_normalizers.get(column)
            normalized_value = normalizer(value) if normalizer is not None else value
            normalized.append(_normalize_scalar(normalized_value))
        return tuple(normalized)


@dataclass(frozen=True)
class LowestAvailableGeographyAssignmentPlan:
    """Assign atomic geography using the most-specific available row geography.

    ``partition_columns`` is ordered from most specific to least specific. For
    each row, the assigner samples from the first non-missing column whose value
    exists in the crosswalk. This models "assign a block within the lowest
    available CPS geography" without hard-coding a country's geography names.
    """

    partition_columns: tuple[str, ...]
    atomic_id_column: str
    geography_columns: tuple[str, ...] = ()
    probability_column: str | None = None
    partition_normalizers: dict[str, PartitionNormalizer] = field(default_factory=dict)
    assignment_level_column: str | None = "_geography_partition_column"
    assignment_key_column: str | None = "_geography_partition_key"
    sync_partition_columns: bool = True

    def __post_init__(self) -> None:
        if not self.partition_columns:
            raise ValueError("partition_columns must list at least one geography level")

    def requested_geography_columns(self) -> tuple[str, ...]:
        """Columns to materialize from the assigned atomic geography."""
        ordered_columns: list[str] = []
        if self.sync_partition_columns:
            ordered_columns.extend(self.partition_columns)
        ordered_columns.extend(self.geography_columns)
        return tuple(dict.fromkeys(ordered_columns))


@dataclass
class LowestAvailableAtomicGeographyAssigner:
    """Sample atomic ids from the most-specific row geography with support."""

    crosswalk: AtomicGeographyCrosswalk
    plan: LowestAvailableGeographyAssignmentPlan

    def __post_init__(self) -> None:
        missing_columns = [
            column
            for column in self.plan.partition_columns
            if column not in self.crosswalk.data.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Partition columns not found in crosswalk: {sorted(missing_columns)}"
            )
        if self.plan.atomic_id_column != self.crosswalk.atomic_id_column:
            raise ValueError(
                "plan.atomic_id_column must match the crosswalk atomic_id_column "
                f"('{self.crosswalk.atomic_id_column}')"
            )
        self._probability_column = (
            self.plan.probability_column or self.crosswalk.probability_column
        )
        if self._probability_column is None:
            raise ValueError(
                "A probability column is required for geography assignment"
            )
        if self._probability_column not in self.crosswalk.data.columns:
            raise ValueError(
                f"Probability column '{self._probability_column}' not found in crosswalk"
            )
        self._normalized_partition_values = self._build_normalized_partition_values()
        self._level_lookup = self._build_level_lookup()

    def assign(
        self,
        frame: pd.DataFrame,
        *,
        random_state: int | None = None,
        materialize: bool = True,
        overwrite_geography: bool = True,
    ) -> pd.DataFrame:
        """Assign atomic geography ids and optionally materialize parent columns."""
        available_columns = [
            column for column in self.plan.partition_columns if column in frame.columns
        ]
        if not available_columns:
            raise ValueError(
                "No assignment partition columns found in frame; expected one of "
                f"{list(self.plan.partition_columns)}"
            )

        rng = np.random.default_rng(random_state)
        assigned_atomic_ids: list[Any] = []
        assigned_levels: list[str] = []
        assigned_keys: list[Any] = []

        for row_index, row in frame.iterrows():
            choice = self._choose_distribution(row)
            if choice is None:
                raise ValueError(
                    "No atomic geography distribution available for row "
                    f"{row_index}; checked columns {list(self.plan.partition_columns)}"
                )
            column, key, group = choice
            sampled_index = rng.choice(
                len(group["atomic_ids"]),
                p=group["probabilities"],
            )
            assigned_atomic_ids.append(group["atomic_ids"][sampled_index])
            assigned_levels.append(column)
            assigned_keys.append(key)

        result = frame.copy()
        result[self.plan.atomic_id_column] = assigned_atomic_ids
        if self.plan.assignment_level_column is not None:
            result[self.plan.assignment_level_column] = assigned_levels
        if self.plan.assignment_key_column is not None:
            result[self.plan.assignment_key_column] = assigned_keys

        if not materialize:
            return result
        return self.crosswalk.materialize(
            result,
            columns=self.plan.requested_geography_columns(),
            atomic_id_column=self.plan.atomic_id_column,
            overwrite=overwrite_geography,
        )

    def _build_normalized_partition_values(self) -> dict[str, pd.Series]:
        return {
            column: self.crosswalk.data[column].map(
                lambda value, column=column: self._normalize_value(column, value)
            )
            for column in self.plan.partition_columns
        }

    def _build_level_lookup(self) -> dict[str, dict[Any, dict[str, np.ndarray]]]:
        lookup: dict[str, dict[Any, dict[str, np.ndarray]]] = {}
        for column in self.plan.partition_columns:
            prepared = pd.DataFrame(
                {
                    "_microplex_partition_key": self._normalized_partition_values[
                        column
                    ],
                    "_microplex_row_position": np.arange(len(self.crosswalk.data)),
                }
            )
            prepared = prepared.loc[
                ~prepared["_microplex_partition_key"].map(_is_missing_partition_value)
            ]
            column_lookup: dict[Any, dict[str, np.ndarray]] = {}
            for key, index_group in prepared.groupby(
                "_microplex_partition_key",
                dropna=False,
                sort=False,
            ):
                row_positions = index_group["_microplex_row_position"].to_numpy()
                self._distribution_from_positions(
                    row_positions,
                    error_context=f"Partition {column}={key!r}",
                )
                column_lookup[_normalize_scalar(key)] = {"row_positions": row_positions}
            lookup[column] = column_lookup
        return lookup

    def _choose_distribution(
        self,
        row: pd.Series,
    ) -> tuple[str, Any, dict[str, np.ndarray]] | None:
        row_values = self._row_partition_values(row)
        for level_index, column in enumerate(self.plan.partition_columns):
            key = row_values.get(column)
            if key is None or _is_missing_partition_value(key):
                continue
            group = self._level_lookup[column].get(key)
            if group is None:
                continue
            constraints = {
                constraint_column: constraint_key
                for constraint_column in self.plan.partition_columns[level_index:]
                if (
                    constraint_column in row_values
                    and not _is_missing_partition_value(row_values[constraint_column])
                )
                for constraint_key in (row_values[constraint_column],)
            }
            constrained = self._constrain_group(
                group,
                constraints,
                error_context=(
                    f"row geography constraints for {column}={key!r}: {constraints}"
                ),
            )
            return column, key, constrained
        return None

    def _row_partition_values(self, row: pd.Series) -> dict[str, Any]:
        return {
            column: self._normalize_value(column, row[column])
            for column in self.plan.partition_columns
            if column in row.index
        }

    def _constrain_group(
        self,
        group: dict[str, np.ndarray],
        constraints: dict[str, Any],
        *,
        error_context: str,
    ) -> dict[str, np.ndarray]:
        row_positions = group["row_positions"]
        selected = np.ones(len(row_positions), dtype=bool)
        for column, key in constraints.items():
            values = self._normalized_partition_values[column].iloc[row_positions]
            matches = np.array(
                [_partition_values_equal(value, key) for value in values],
                dtype=bool,
            )
            selected &= matches
        constrained_positions = row_positions[selected]
        if len(constrained_positions) == 0:
            raise ValueError(
                f"No atomic geography distribution satisfies {error_context}"
            )
        return self._distribution_from_positions(
            constrained_positions,
            error_context=error_context,
        )

    def _distribution_from_positions(
        self,
        row_positions: np.ndarray,
        *,
        error_context: str,
    ) -> dict[str, np.ndarray]:
        rows = self.crosswalk.data.iloc[row_positions]
        probabilities = pd.to_numeric(
            rows[self._probability_column],
            errors="coerce",
        ).fillna(0.0)
        total_probability = float(probabilities.sum())
        if total_probability <= 0:
            raise ValueError(f"{error_context} has non-positive total probability")
        return {
            "atomic_ids": rows[self.crosswalk.atomic_id_column].to_numpy(),
            "probabilities": (probabilities / total_probability).to_numpy(dtype=float),
        }

    def _normalize_value(self, column: str, value: Any) -> Any:
        value = _normalize_scalar(value)
        if _is_missing_partition_value(value):
            return value
        normalizer = self.plan.partition_normalizers.get(column)
        normalized = normalizer(value) if normalizer is not None else value
        return _normalize_scalar(normalized)


@dataclass(frozen=True)
class GeographyQuery:
    """Generic query parameters for atomic-geography providers."""

    geography_columns: tuple[str, ...] = ()
    partition_columns: tuple[str, ...] = ()
    probability_column: str | None = None
    partition_normalizers: dict[str, PartitionNormalizer] = field(default_factory=dict)
    fallback_resolver: PartitionFallbackResolver | None = None


@dataclass(frozen=True)
class GeographyAssignmentPlan:
    """How a model should assign atomic geography ids during synthesis."""

    partition_columns: tuple[str, ...]
    atomic_id_column: str
    geography_columns: tuple[str, ...] = ()
    probability_column: str | None = None
    partition_normalizers: dict[str, PartitionNormalizer] = field(default_factory=dict)
    fallback_resolver: PartitionFallbackResolver | None = None
    sync_partition_columns: bool = True

    def requested_geography_columns(self) -> tuple[str, ...]:
        """Columns that should be materialized after assignment."""
        ordered_columns: list[str] = []
        if self.sync_partition_columns:
            ordered_columns.extend(self.partition_columns)
        ordered_columns.extend(self.geography_columns)
        return tuple(dict.fromkeys(ordered_columns))

    def to_query(self) -> GeographyQuery:
        """Convert the assignment plan into a provider query."""
        return GeographyQuery(
            geography_columns=self.requested_geography_columns(),
            partition_columns=self.partition_columns,
            probability_column=self.probability_column,
            partition_normalizers=dict(self.partition_normalizers),
            fallback_resolver=self.fallback_resolver,
        )


@runtime_checkable
class GeographyProvider(Protocol):
    """Protocol for providers of atomic geography crosswalks and assigners."""

    def load_crosswalk(
        self,
        query: GeographyQuery | None = None,
    ) -> AtomicGeographyCrosswalk:
        """Load an atomic geography crosswalk."""

    def load_assigner(
        self,
        query: GeographyQuery | None = None,
    ) -> ProbabilisticAtomicGeographyAssigner:
        """Load a probabilistic atomic geography assigner."""


@dataclass
class StaticGeographyProvider:
    """A geography provider backed by an in-memory atomic crosswalk."""

    crosswalk: AtomicGeographyCrosswalk
    default_partition_columns: tuple[str, ...] = ()
    default_partition_normalizers: dict[str, PartitionNormalizer] = field(
        default_factory=dict
    )
    default_fallback_resolver: PartitionFallbackResolver | None = None

    def load_crosswalk(
        self,
        query: GeographyQuery | None = None,
    ) -> AtomicGeographyCrosswalk:
        query = query or GeographyQuery()
        geography_columns = query.geography_columns or self.crosswalk.geography_columns
        probability_column = (
            query.probability_column or self.crosswalk.probability_column
        )
        return AtomicGeographyCrosswalk(
            data=self.crosswalk.data.copy(),
            atomic_id_column=self.crosswalk.atomic_id_column,
            geography_columns=tuple(geography_columns),
            probability_column=probability_column,
        )

    def load_assigner(
        self,
        query: GeographyQuery | None = None,
    ) -> ProbabilisticAtomicGeographyAssigner:
        query = query or GeographyQuery()
        partition_columns = query.partition_columns or self.default_partition_columns
        if not partition_columns:
            raise ValueError(
                "partition_columns are required to build a geography assigner"
            )
        probability_column = (
            query.probability_column or self.crosswalk.probability_column
        )
        return ProbabilisticAtomicGeographyAssigner(
            crosswalk=self.load_crosswalk(query),
            partition_columns=tuple(partition_columns),
            probability_column=probability_column,
            partition_normalizers=(
                query.partition_normalizers or self.default_partition_normalizers
            ),
            fallback_resolver=(
                query.fallback_resolver
                if query.fallback_resolver is not None
                else self.default_fallback_resolver
            ),
        )


def _raise_missing_us_geography() -> ModuleNotFoundError:
    return ModuleNotFoundError(
        "US block geography helpers moved to the separate `microplex-us` package. "
        "Install or add `microplex-us`, then import `microplex_us.geography`."
    )


def _load_us_geography_symbol(name: str) -> Any:
    try:
        module = import_module("microplex_us.geography")
    except ModuleNotFoundError as exc:
        if exc.name == "microplex_us":
            raise _raise_missing_us_geography() from exc
        raise
    return getattr(module, name)


def load_block_probabilities(*args: Any, **kwargs: Any) -> pd.DataFrame:
    return _load_us_geography_symbol("load_block_probabilities")(*args, **kwargs)


def derive_geographies(*args: Any, **kwargs: Any) -> pd.DataFrame:
    return _load_us_geography_symbol("derive_geographies")(*args, **kwargs)


class _USBlockGeographyProxy(type):
    def __getattr__(cls, name: str) -> Any:
        return getattr(_load_us_geography_symbol("BlockGeography"), name)

    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, _load_us_geography_symbol("BlockGeography"))

    def __subclasscheck__(cls, subclass: type) -> bool:
        return issubclass(subclass, _load_us_geography_symbol("BlockGeography"))


class BlockGeography(metaclass=_USBlockGeographyProxy):
    """Compatibility proxy for the moved US block geography adapter."""

    @classmethod
    def from_data(cls, data: pd.DataFrame) -> Any:
        return _load_us_geography_symbol("BlockGeography").from_data(data)

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        return _load_us_geography_symbol("BlockGeography")(*args, **kwargs)
