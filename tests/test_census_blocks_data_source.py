from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from microplex.data_sources.census_blocks import (
    CensusBlockCrosswalkProvider,
    load_census_block_crosswalk,
    prepare_census_block_crosswalk,
)
from microplex.geography import GeographyQuery


def _raw_crosswalk() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "block_geoid": [
                "060010201001000",
                "060010201001001",
                "360610001001000",
            ],
            "sldu": ["009", "009", "030"],
            "sldl": ["018", "018", "065"],
            "place_fips": ["53000", "", "51000"],
        }
    )


def test_prepare_census_block_crosswalk_derives_parent_geoids() -> None:
    prepared = prepare_census_block_crosswalk(_raw_crosswalk())

    assert prepared["state_fips"].tolist() == ["06", "06", "36"]
    assert prepared["county_fips"].tolist() == ["06001", "06001", "36061"]
    assert prepared["tract_geoid"].tolist() == [
        "06001020100",
        "06001020100",
        "36061000100",
    ]
    assert prepared["prob"].tolist() == [1.0, 1.0, 1.0]


def test_prepare_census_block_crosswalk_accepts_leading_zero_loss() -> None:
    raw = _raw_crosswalk()
    raw["block_geoid"] = raw["block_geoid"].astype(int)

    prepared = prepare_census_block_crosswalk(raw)

    assert prepared["block_geoid"].tolist()[0] == "060010201001000"


def test_prepare_census_block_crosswalk_rejects_short_or_nonnumeric_geoids() -> None:
    raw = _raw_crosswalk()
    raw.loc[0, "block_geoid"] = "12345"
    raw.loc[1, "block_geoid"] = "not-a-geoid"

    with pytest.raises(ValueError, match="invalid Census block GEOIDs"):
        prepare_census_block_crosswalk(raw)


def test_load_census_block_crosswalk_filters_states_from_compressed_csv(
    tmp_path: Path,
) -> None:
    path = tmp_path / "block_crosswalk.csv.gz"
    _raw_crosswalk().to_csv(path, index=False, compression="gzip")

    loaded = load_census_block_crosswalk(path, state_fips=[36])

    assert loaded["block_geoid"].tolist() == ["360610001001000"]
    assert loaded["state_fips"].tolist() == ["36"]


def test_load_census_block_crosswalk_rejects_empty_state_filter(
    tmp_path: Path,
) -> None:
    path = tmp_path / "block_crosswalk.csv.gz"
    _raw_crosswalk().to_csv(path, index=False, compression="gzip")

    with pytest.raises(ValueError, match="no rows after filtering"):
        load_census_block_crosswalk(path, state_fips=[12])


def test_load_census_block_crosswalk_checks_duplicates_after_chunking(
    tmp_path: Path,
) -> None:
    path = tmp_path / "block_crosswalk.csv"
    pd.concat([_raw_crosswalk().iloc[[0]], _raw_crosswalk().iloc[[0]]]).to_csv(
        path,
        index=False,
    )

    with pytest.raises(ValueError, match="unique block_geoids"):
        load_census_block_crosswalk(path, max_rows=2, chunksize=1)


def test_prepare_census_block_crosswalk_rejects_negative_probabilities() -> None:
    frame = _raw_crosswalk()
    frame["prob"] = [1.0, -0.1, 1.0]

    with pytest.raises(ValueError, match="non-negative"):
        prepare_census_block_crosswalk(frame)


def test_census_block_crosswalk_provider_builds_assigner() -> None:
    provider = CensusBlockCrosswalkProvider.from_data(_raw_crosswalk())
    assigner = provider.load_assigner(GeographyQuery(partition_columns=("state_fips",)))

    assigned = assigner.assign(
        pd.DataFrame({"state_fips": [6, 36]}),
        random_state=4,
    )

    assert assigned["block_geoid"].str.len().tolist() == [15, 15]
    assert assigned["block_geoid"].str[:2].tolist() == ["06", "36"]
