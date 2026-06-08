from __future__ import annotations

import pandas as pd
import pytest

from microplex.geography import (
    AtomicGeographyCrosswalk,
    LowestAvailableAtomicGeographyAssigner,
    LowestAvailableGeographyAssignmentPlan,
    normalize_string_code,
    normalize_us_state_fips,
)


def _crosswalk() -> AtomicGeographyCrosswalk:
    return AtomicGeographyCrosswalk(
        data=pd.DataFrame(
            {
                "block_geoid": [
                    "060010001001001",
                    "060010001001002",
                    "060750101001001",
                    "360610001001001",
                ],
                "state_fips": ["06", "06", "06", "36"],
                "cbsa": ["41860", "41860", "41860", "35620"],
                "tract_geoid": [
                    "06001000100",
                    "06001000100",
                    "06075010100",
                    "36061000100",
                ],
                "county_fips": ["06001", "06001", "06075", "36061"],
                "prob": [0.4, 0.6, 1.0, 1.0],
            }
        ),
        atomic_id_column="block_geoid",
        geography_columns=("state_fips", "cbsa", "tract_geoid", "county_fips"),
        probability_column="prob",
    )


def _assigner() -> LowestAvailableAtomicGeographyAssigner:
    return LowestAvailableAtomicGeographyAssigner(
        crosswalk=_crosswalk(),
        plan=LowestAvailableGeographyAssignmentPlan(
            partition_columns=("tract_geoid", "cbsa", "state_fips"),
            atomic_id_column="block_geoid",
            geography_columns=("county_fips",),
            partition_normalizers={
                "cbsa": normalize_string_code,
                "state_fips": normalize_us_state_fips,
            },
        ),
    )


def test_lowest_available_geography_assignment_prefers_specific_columns() -> None:
    frame = pd.DataFrame(
        {
            "tax_unit_id": [1, 2, 3],
            "tract_geoid": ["06001000100", pd.NA, pd.NA],
            "cbsa": ["41860", 35620, pd.NA],
            "state_fips": [6, 36, 6],
        }
    )

    result = _assigner().assign(frame, random_state=3)

    assert result.loc[0, "_geography_partition_column"] == "tract_geoid"
    assert result.loc[0, "block_geoid"].startswith("06001000100")
    assert result.loc[1, "_geography_partition_column"] == "cbsa"
    assert result.loc[1, "block_geoid"] == "360610001001001"
    assert result.loc[2, "_geography_partition_column"] == "state_fips"
    assert result.loc[2, "state_fips"] == "06"
    assert result.loc[2, "block_geoid"].startswith("06")
    assert result.loc[0, "county_fips"] == "06001"
    assert result.loc[1, "county_fips"] == "36061"
    assert result.loc[2, "county_fips"] in {"06001", "06075"}


def test_lowest_available_geography_assignment_falls_back_when_key_unsupported() -> (
    None
):
    frame = pd.DataFrame(
        {
            "tax_unit_id": [1],
            "tract_geoid": ["99999999999"],
            "cbsa": [pd.NA],
            "state_fips": [36],
        }
    )

    result = _assigner().assign(frame, random_state=7)

    assert result.loc[0, "_geography_partition_column"] == "state_fips"
    assert result.loc[0, "_geography_partition_key"] == "36"
    assert result.loc[0, "block_geoid"] == "360610001001001"


def test_lowest_available_geography_assignment_intersects_coarser_constraints() -> None:
    crosswalk = AtomicGeographyCrosswalk(
        data=pd.DataFrame(
            {
                "block_geoid": ["340170001001001", "360610001001001"],
                "cbsa": ["35620", "35620"],
                "state_fips": ["34", "36"],
                "prob": [0.9, 0.1],
            }
        ),
        atomic_id_column="block_geoid",
        geography_columns=("cbsa", "state_fips"),
        probability_column="prob",
    )
    assigner = LowestAvailableAtomicGeographyAssigner(
        crosswalk=crosswalk,
        plan=LowestAvailableGeographyAssignmentPlan(
            partition_columns=("cbsa", "state_fips"),
            atomic_id_column="block_geoid",
            partition_normalizers={
                "cbsa": normalize_string_code,
                "state_fips": normalize_us_state_fips,
            },
        ),
    )

    result = assigner.assign(
        pd.DataFrame({"tax_unit_id": [1], "cbsa": [35620], "state_fips": [36]}),
        random_state=7,
    )

    assert result.loc[0, "_geography_partition_column"] == "cbsa"
    assert result.loc[0, "block_geoid"] == "360610001001001"
    assert result.loc[0, "state_fips"] == "36"


def test_lowest_available_geography_assignment_rejects_inconsistent_constraints() -> (
    None
):
    frame = pd.DataFrame(
        {
            "tax_unit_id": [1],
            "tract_geoid": [pd.NA],
            "cbsa": [35620],
            "state_fips": [6],
        }
    )

    with pytest.raises(ValueError, match="No atomic geography distribution satisfies"):
        _assigner().assign(frame, random_state=7)


def test_lowest_available_geography_assignment_requires_supported_row_geography() -> (
    None
):
    frame = pd.DataFrame({"tax_unit_id": [1], "state_fips": [99]})

    with pytest.raises(ValueError, match="No atomic geography distribution available"):
        _assigner().assign(frame, random_state=7)


def test_lowest_available_geography_assignment_skips_missing_normalized_values() -> (
    None
):
    frame = pd.DataFrame(
        {
            "tax_unit_id": [1],
            "tract_geoid": [pd.NA],
            "cbsa": [pd.NA],
            "state_fips": [pd.NA],
        }
    )

    with pytest.raises(ValueError, match="No atomic geography distribution available"):
        _assigner().assign(frame, random_state=7)


def test_lowest_available_geography_assignment_rejects_missing_frame_columns() -> None:
    frame = pd.DataFrame({"tax_unit_id": [1]})

    with pytest.raises(ValueError, match="No assignment partition columns"):
        _assigner().assign(frame, random_state=7)


def test_lowest_available_geography_assignment_rejects_zero_probability_groups() -> (
    None
):
    crosswalk = AtomicGeographyCrosswalk(
        data=pd.DataFrame(
            {
                "block_geoid": ["010010001001001"],
                "state_fips": ["01"],
                "prob": [0.0],
            }
        ),
        atomic_id_column="block_geoid",
        geography_columns=("state_fips",),
        probability_column="prob",
    )

    with pytest.raises(ValueError, match="non-positive total probability"):
        LowestAvailableAtomicGeographyAssigner(
            crosswalk=crosswalk,
            plan=LowestAvailableGeographyAssignmentPlan(
                partition_columns=("state_fips",),
                atomic_id_column="block_geoid",
            ),
        )
