"""Census block crosswalk loading for geography assignment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from microplex.geography import (
    AtomicGeographyCrosswalk,
    GeographyProvider,
    GeographyQuery,
    ProbabilisticAtomicGeographyAssigner,
    normalize_us_state_fips,
)

BLOCK_GEOID_COLUMN = "block_geoid"
DEFAULT_BLOCK_PROBABILITY_COLUMN = "prob"
DEFAULT_BLOCK_GEOGRAPHY_COLUMNS: tuple[str, ...] = (
    "state_fips",
    "county_fips",
    "tract_geoid",
    "sldu",
    "sldl",
    "place_fips",
    "vtd",
    "puma",
    "zcta",
)


def load_census_block_crosswalk(
    path: str | Path,
    *,
    state_fips: Sequence[Any] | None = None,
    max_rows: int | None = None,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Load a Census block crosswalk and derive parent GEOID columns.

    The production PolicyEngine-US crosswalk stores one row per block with a
    `block_geoid` and selected parent geography codes. It does not currently
    include sampling probabilities, so this loader adds a uniform `prob` column
    unless one already exists.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Census block crosswalk not found: {path}")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive when supplied")

    state_filter = _normalize_state_filter(state_fips)
    if state_filter or max_rows is not None:
        return _load_crosswalk_chunks(
            path,
            state_filter=state_filter,
            max_rows=max_rows,
            chunksize=chunksize,
        )
    return prepare_census_block_crosswalk(pd.read_csv(path, dtype=str))


def prepare_census_block_crosswalk(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw block crosswalk frame into assigner-ready columns."""
    if "geoid" in frame.columns and BLOCK_GEOID_COLUMN not in frame.columns:
        frame = frame.rename(columns={"geoid": BLOCK_GEOID_COLUMN})
    if BLOCK_GEOID_COLUMN not in frame.columns:
        raise ValueError("Census block crosswalk must include block_geoid or geoid")

    result = frame.copy()
    normalized_geoids = (
        result[BLOCK_GEOID_COLUMN]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    valid_numeric = normalized_geoids.str.fullmatch(r"\d+").fillna(False)
    raw_lengths = normalized_geoids.str.len()
    valid_length = raw_lengths.eq(15) | (
        raw_lengths.eq(14) & normalized_geoids.str.zfill(15).str.startswith("0")
    )
    invalid = normalized_geoids.isna() | ~valid_numeric | ~valid_length
    if invalid.any():
        raise ValueError(
            "Census block crosswalk has invalid Census block GEOIDs: "
            f"{int(invalid.sum())} rows"
        )
    result[BLOCK_GEOID_COLUMN] = normalized_geoids.str.zfill(15)

    result["state_fips"] = result[BLOCK_GEOID_COLUMN].str[:2]
    result["county_fips"] = result[BLOCK_GEOID_COLUMN].str[:5]
    result["tract_geoid"] = result[BLOCK_GEOID_COLUMN].str[:11]
    if DEFAULT_BLOCK_PROBABILITY_COLUMN not in result.columns:
        result[DEFAULT_BLOCK_PROBABILITY_COLUMN] = 1.0
    else:
        result[DEFAULT_BLOCK_PROBABILITY_COLUMN] = pd.to_numeric(
            result[DEFAULT_BLOCK_PROBABILITY_COLUMN],
            errors="coerce",
        ).fillna(0.0)
    if result[DEFAULT_BLOCK_PROBABILITY_COLUMN].lt(0).any():
        raise ValueError("Census block crosswalk probabilities must be non-negative")
    if result[BLOCK_GEOID_COLUMN].duplicated().any():
        raise ValueError("Census block crosswalk must have unique block_geoids")
    return result


@dataclass
class CensusBlockCrosswalkProvider(GeographyProvider):
    """File-backed atomic geography provider for Census block crosswalks."""

    path: str | Path | None = None
    data: pd.DataFrame | None = None
    state_fips: Sequence[Any] | None = None
    max_rows: int | None = None

    @classmethod
    def from_data(cls, data: pd.DataFrame) -> CensusBlockCrosswalkProvider:
        """Create a provider from an in-memory fixture."""
        return cls(data=data.copy())

    def load_crosswalk(
        self,
        query: GeographyQuery | None = None,
    ) -> AtomicGeographyCrosswalk:
        """Load the block crosswalk as an atomic geography crosswalk."""
        query = query or GeographyQuery()
        data = self._load_data()
        geography_columns = tuple(
            query.geography_columns
            or tuple(
                column for column in DEFAULT_BLOCK_GEOGRAPHY_COLUMNS if column in data
            )
        )
        probability_column = (
            query.probability_column or DEFAULT_BLOCK_PROBABILITY_COLUMN
        )
        return AtomicGeographyCrosswalk(
            data=data.copy(),
            atomic_id_column=BLOCK_GEOID_COLUMN,
            geography_columns=geography_columns,
            probability_column=probability_column,
        )

    def load_assigner(
        self,
        query: GeographyQuery | None = None,
    ) -> ProbabilisticAtomicGeographyAssigner:
        """Build a probabilistic assigner from the loaded crosswalk."""
        query = query or GeographyQuery(partition_columns=("state_fips",))
        partition_columns = tuple(query.partition_columns) or ("state_fips",)
        normalizers = dict(query.partition_normalizers)
        normalizers.setdefault("state_fips", normalize_us_state_fips)
        return ProbabilisticAtomicGeographyAssigner(
            crosswalk=self.load_crosswalk(query),
            partition_columns=partition_columns,
            probability_column=query.probability_column
            or DEFAULT_BLOCK_PROBABILITY_COLUMN,
            partition_normalizers=normalizers,
            fallback_resolver=query.fallback_resolver,
        )

    def _load_data(self) -> pd.DataFrame:
        if self.data is not None:
            return prepare_census_block_crosswalk(self.data)
        if self.path is None:
            raise ValueError("CensusBlockCrosswalkProvider requires path or data")
        return load_census_block_crosswalk(
            self.path,
            state_fips=self.state_fips,
            max_rows=self.max_rows,
        )


def _load_crosswalk_chunks(
    path: Path,
    *,
    state_filter: frozenset[str],
    max_rows: int | None,
    chunksize: int,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    remaining = max_rows
    for chunk in pd.read_csv(path, dtype=str, chunksize=chunksize):
        prepared = prepare_census_block_crosswalk(chunk)
        if state_filter:
            prepared = prepared.loc[prepared["state_fips"].isin(state_filter)]
        if prepared.empty:
            continue
        if remaining is not None:
            prepared = prepared.head(remaining)
            remaining -= len(prepared)
        chunks.append(prepared)
        if remaining == 0:
            break
    if not chunks:
        raise ValueError("Census block crosswalk contains no rows after filtering")
    result = pd.concat(chunks, ignore_index=True)
    if result[BLOCK_GEOID_COLUMN].duplicated().any():
        raise ValueError("Census block crosswalk must have unique block_geoids")
    return result


def _normalize_state_filter(values: Sequence[Any] | None) -> frozenset[str]:
    if values is None:
        return frozenset()
    return frozenset(normalize_us_state_fips(value) for value in values)
