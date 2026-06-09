"""Tests for the CPS ASEC data source loader (``microplex.data_sources.cps``).

Focus: the opt-in ``extra_person_columns`` / ``extra_household_columns`` path
that carries raw CPS CSV columns through verbatim, the cache-key separation that
keeps distinct extra-column sets in distinct processed-parquet files, and the
unknown-column validation error.

These tests never download. They synthesize a tiny zip fixture in ``tmp_path``
that mimics the real ``pppub*.csv`` / ``hhpub*.csv`` structure inside the
``asecpub*csv.zip`` archive, then point ``load_cps_asec(cache_dir=...)`` at it.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl
import pytest

from microplex.data_sources.cps import (
    CPSAsecSourceProvider,
    _extra_columns_cache_token,
    _normalize_extra_columns,
    load_cps_asec,
)

# Extra raw columns the eCPS-replacement pipeline needs verbatim. These are real
# ASEC public-use column names; the loader must pass them through untouched.
EXTRA_PERSON_COLUMNS = [
    "A_LINENO",
    "A_MARITL",
    "A_SPOUSE",
    "PEPAR1",
    "PEPAR2",
    "A_EXPRRP",
    "SPM_ID",
    "PF_SEQ",
    "A_HSCOL",
    "PH_SEQ",
]


def _person_frame() -> pl.DataFrame:
    """A tiny person frame with the harmonized inputs plus raw extra columns.

    The extra columns carry integer sentinels (-1) and zero values so the test
    can assert dtypes and sentinel semantics survive verbatim.
    """
    return pl.DataFrame(
        {
            # --- columns the harmonized mapping consumes ---
            "PH_SEQ": [1, 1, 2, 3],
            "PF_SEQ": [1, 1, 1, 1],
            "A_LINENO": [1, 2, 1, 1],
            "A_AGE": [40, 38, 21, 67],
            "A_SEX": [1, 2, 2, 1],
            "A_MARITL": [1, 1, 6, 4],
            "WSAL_VAL": [50_000, 15_000, 20_000, 0],
            "SS_VAL": [0, 0, 0, 12_000],
            "A_FNLWGT": [1_050_000, 1_050_000, 2_010_000, 4_000_000],
            "MARSUPWT": [1_050_000, 1_050_000, 2_010_000, 4_000_000],
            # --- raw extras that the default path drops ---
            "A_SPOUSE": [2, 1, 0, 0],
            "PEPAR1": [-1, -1, -1, -1],
            "PEPAR2": [-1, -1, -1, -1],
            "A_EXPRRP": [1, 2, 1, 1],
            "SPM_ID": [101, 101, 201, 301],
            "A_HSCOL": [0, 0, 1, 0],
        }
    )


def _household_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "H_SEQ": [1, 2, 3],
            "GESTFIPS": [6, 36, 48],
            "GTCBSA": [41860, 35620, 19100],
            "H_NUMPER": [2, 1, 1],
            "HSUP_WGT": [1_050_000, 2_010_000, 4_000_000],
            # raw extras on the household side
            "HRNUMHOU": [2, 1, 1],
            "HTOTVAL": [70_000, 20_000, 12_000],
        }
    )


def _write_fixture_zip(
    cache_dir: Path,
    year: int = 2025,
    *,
    persons: pl.DataFrame | None = None,
    households: pl.DataFrame | None = None,
) -> Path:
    """Write a ``cps_asec_{year}.zip`` with pppub/hhpub CSVs into ``cache_dir``.

    Mirrors the real archive layout so ``load_cps_asec`` finds the files by the
    ``pppub*`` / ``hhpub*`` name globs.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    persons = _person_frame() if persons is None else persons
    households = _household_frame() if households is None else households

    suffix = str(year)[-2:]
    zip_path = cache_dir / f"cps_asec_{year}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"pppub{suffix}.csv", persons.write_csv())
        zf.writestr(f"hhpub{suffix}.csv", households.write_csv())
    return zip_path


# ---------------------------------------------------------------------------
# Cache-key unit tests
# ---------------------------------------------------------------------------


def test_normalize_extra_columns_sorts_and_dedupes() -> None:
    assert _normalize_extra_columns(["B", "A", "B"]) == ("A", "B")
    assert _normalize_extra_columns([]) == ()


def test_cache_token_empty_for_default_request() -> None:
    # Default set must yield no suffix so existing caches stay valid.
    assert _extra_columns_cache_token((), ()) == ""


def test_cache_token_order_independent_and_distinct() -> None:
    token_a = _extra_columns_cache_token(["A_LINENO", "A_MARITL"], [])
    token_a_reordered = _extra_columns_cache_token(["A_MARITL", "A_LINENO"], [])
    token_b = _extra_columns_cache_token(["A_LINENO"], [])

    # Sorting makes the token order-independent...
    assert token_a == token_a_reordered
    # ...but different sets produce different tokens, and tokens are namespaced.
    assert token_a != token_b
    assert token_a.startswith("__x")
    assert token_b.startswith("__x")


def test_cache_token_separates_person_vs_household_namespace() -> None:
    # Same column name on the person vs household side must not collide.
    person_side = _extra_columns_cache_token(["HTOTVAL"], [])
    household_side = _extra_columns_cache_token([], ["HTOTVAL"])
    assert person_side != household_side


# ---------------------------------------------------------------------------
# Default path: backward compatibility
# ---------------------------------------------------------------------------


def test_default_load_omits_extra_columns_and_uses_unsuffixed_cache(
    tmp_path: Path,
) -> None:
    _write_fixture_zip(tmp_path)

    dataset = load_cps_asec(year=2025, cache_dir=tmp_path, download=False)

    # Harmonized columns present, raw extras absent by default.
    assert "age" in dataset.persons.columns
    assert "household_id" in dataset.persons.columns
    assert "A_LINENO" not in dataset.persons.columns
    assert "A_MARITL" not in dataset.persons.columns

    # Default cache uses the original, unsuffixed filename.
    assert (tmp_path / "cps_asec_2025_processed.parquet").exists()
    assert (
        tmp_path / "cps_asec_2025_households_processed.parquet"
    ).exists()
    # No suffixed file should have been written for the default request.
    suffixed = list(tmp_path.glob("cps_asec_2025_processed__x*.parquet"))
    assert suffixed == []


# ---------------------------------------------------------------------------
# Extra columns: presence + verbatim values/dtypes
# ---------------------------------------------------------------------------


def test_extra_person_columns_present_and_verbatim(tmp_path: Path) -> None:
    _write_fixture_zip(tmp_path)
    dataset = load_cps_asec(
        year=2025,
        cache_dir=tmp_path,
        download=False,
        extra_person_columns=EXTRA_PERSON_COLUMNS,
    )

    raw = _person_frame()
    for column in EXTRA_PERSON_COLUMNS:
        assert column in dataset.persons.columns, f"missing extra column {column}"

    # PH_SEQ is both renamed to household_id AND kept raw.
    assert "household_id" in dataset.persons.columns
    assert dataset.persons["PH_SEQ"].to_list() == raw["PH_SEQ"].to_list()
    assert dataset.persons["household_id"].to_list() == raw["PH_SEQ"].to_list()

    # Values are byte-identical to the raw CSV (no sentinel rewriting): the
    # -1 PEPAR sentinels and 0/coded values survive untouched.
    for column in EXTRA_PERSON_COLUMNS:
        assert dataset.persons[column].to_list() == raw[column].to_list()

    # Dtypes stay integral (ints stay ints).
    for column in ("A_LINENO", "A_MARITL", "PEPAR1", "SPM_ID"):
        assert dataset.persons[column].dtype.is_integer(), column

    # Harmonized income sentinel logic is unaffected for the harmonized column.
    assert dataset.persons["wage_income"].to_list() == [
        50_000,
        15_000,
        20_000,
        0,
    ]


def test_extra_household_columns_present_and_verbatim(tmp_path: Path) -> None:
    _write_fixture_zip(tmp_path)
    dataset = load_cps_asec(
        year=2025,
        cache_dir=tmp_path,
        download=False,
        extra_household_columns=["HTOTVAL", "HRNUMHOU"],
    )

    raw = _household_frame()
    assert "HTOTVAL" in dataset.households.columns
    assert "HRNUMHOU" in dataset.households.columns
    assert dataset.households["HTOTVAL"].to_list() == raw["HTOTVAL"].to_list()
    assert dataset.households["HRNUMHOU"].to_list() == raw["HRNUMHOU"].to_list()
    # Harmonized household_weight is still scaled (HSUP_WGT / 100), proving the
    # passthrough does not disturb the harmonized pipeline.
    assert dataset.households["household_weight"].to_list() == [
        10_500.0,
        20_100.0,
        40_000.0,
    ]


# ---------------------------------------------------------------------------
# Cache-key separation: two different extra sets -> two cache files
# ---------------------------------------------------------------------------


def test_distinct_extra_sets_write_distinct_cache_files(tmp_path: Path) -> None:
    _write_fixture_zip(tmp_path)

    set_a = ["A_LINENO", "A_MARITL"]
    set_b = ["A_SPOUSE", "A_EXPRRP"]

    load_cps_asec(
        year=2025, cache_dir=tmp_path, download=False, extra_person_columns=set_a
    )
    load_cps_asec(
        year=2025, cache_dir=tmp_path, download=False, extra_person_columns=set_b
    )

    token_a = _extra_columns_cache_token(set_a, [])
    token_b = _extra_columns_cache_token(set_b, [])
    file_a = tmp_path / f"cps_asec_2025_processed{token_a}.parquet"
    file_b = tmp_path / f"cps_asec_2025_processed{token_b}.parquet"

    assert file_a.exists()
    assert file_b.exists()
    assert file_a != file_b

    # Each cache file carries only the columns it was built for.
    persons_a = pl.read_parquet(file_a)
    persons_b = pl.read_parquet(file_b)
    assert {"A_LINENO", "A_MARITL"} <= set(persons_a.columns)
    assert "A_SPOUSE" not in persons_a.columns
    assert {"A_SPOUSE", "A_EXPRRP"} <= set(persons_b.columns)
    assert "A_LINENO" not in persons_b.columns

    # Three distinct processed-person parquet files now exist for 2025:
    # the two suffixed sets here (the default unsuffixed file is only written
    # by a default-set load, which this test does not perform).
    suffixed = sorted(tmp_path.glob("cps_asec_2025_processed__x*.parquet"))
    assert len(suffixed) == 2


def test_cache_is_reused_for_same_extra_set(tmp_path: Path) -> None:
    _write_fixture_zip(tmp_path)
    columns = ["A_LINENO", "A_MARITL"]

    first = load_cps_asec(
        year=2025, cache_dir=tmp_path, download=False, extra_person_columns=columns
    )
    token = _extra_columns_cache_token(columns, [])
    cache_file = tmp_path / f"cps_asec_2025_processed{token}.parquet"
    assert cache_file.exists()

    # Remove the source zip; a second call with the same set must be served from
    # the processed cache (proving the cache key matched), not re-parsed.
    (tmp_path / "cps_asec_2025.zip").unlink()
    second = load_cps_asec(
        year=2025, cache_dir=tmp_path, download=False, extra_person_columns=columns
    )

    assert first.persons["A_LINENO"].to_list() == second.persons["A_LINENO"].to_list()
    assert "A_MARITL" in second.persons.columns


# ---------------------------------------------------------------------------
# Unknown column validation
# ---------------------------------------------------------------------------


def test_unknown_extra_person_column_raises_value_error(tmp_path: Path) -> None:
    _write_fixture_zip(tmp_path)

    with pytest.raises(ValueError, match="NOT_A_REAL_COLUMN"):
        load_cps_asec(
            year=2025,
            cache_dir=tmp_path,
            download=False,
            extra_person_columns=["A_LINENO", "NOT_A_REAL_COLUMN"],
        )


def test_unknown_extra_household_column_raises_value_error(tmp_path: Path) -> None:
    _write_fixture_zip(tmp_path)

    with pytest.raises(ValueError, match="BOGUS_HH_COL"):
        load_cps_asec(
            year=2025,
            cache_dir=tmp_path,
            download=False,
            extra_household_columns=["BOGUS_HH_COL"],
        )


# ---------------------------------------------------------------------------
# Provider threading
# ---------------------------------------------------------------------------


def test_provider_threads_extra_columns_into_frame(tmp_path: Path) -> None:
    _write_fixture_zip(tmp_path)

    provider = CPSAsecSourceProvider(
        asec_year=2025,
        calendar_year=2024,
        cache_dir=tmp_path,
        download=False,
        extra_person_columns=["A_LINENO", "A_MARITL", "PH_SEQ"],
        extra_household_columns=["HTOTVAL"],
    )
    frame = provider.load_frame()

    from microplex.core import EntityType

    persons = frame.tables[EntityType.PERSON]
    households = frame.tables[EntityType.HOUSEHOLD]

    assert "A_LINENO" in persons.columns
    assert "A_MARITL" in persons.columns
    assert "PH_SEQ" in persons.columns
    assert "HTOTVAL" in households.columns

    # The runtime descriptor should advertise the passed-through columns as
    # observed person variables.
    person_obs = next(
        obs
        for obs in frame.source.observations
        if obs.entity == EntityType.PERSON
    )
    assert "A_LINENO" in person_obs.variable_names
    assert "A_MARITL" in person_obs.variable_names
