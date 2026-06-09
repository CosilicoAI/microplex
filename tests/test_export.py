"""Tests for the PolicyEngine-US dataset export stage.

These tests use a tiny synthetic two-household world and a fabricated
mini-contract so they stay fast and independent of the full packs manifest.
Round-trip verification goes only as far as reloading the saved
``USSingleYearDataset`` (not a full ``Microsimulation`` run), which is the
expensive part.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from microplex.core.entities import EntityType
from microplex.export import (
    ExportContract,
    ExportGateResult,
    export_policyengine_us_dataset,
)

# policyengine-us is an optional dependency; every test here needs it for the
# USSingleYearDataset round-trip, so skip the whole module if it is absent.
pytest.importorskip("policyengine_us")

PACKS_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "packs"
    / "us"
    / "manifests"
    / "ecps_export_contract.json"
)


@dataclass(frozen=True)
class _MaritalUnitKey:
    """Stand-in entity key for the marital_unit table.

    ``EntityType`` has no ``MARITAL_UNIT`` member, but the export resolves a
    frame's PolicyEngine table name from any key exposing a ``.value``. This
    mirrors how an orchestrator supplies the marital_unit frame.
    """

    value: str = "marital_unit"


MARITAL_UNIT = _MaritalUnitKey()


def _synthetic_world() -> dict[object, pd.DataFrame]:
    """Build a consistent two-household synthetic world.

    Household 0 is a married couple (persons 0 and 1) sharing one tax unit, spm
    unit, and family; the two spouses form one marital unit. Household 1 is a
    single adult (person 2) in their own units.
    """
    person = pd.DataFrame(
        {
            "person_id": [0, 1, 2],
            "person_household_id": [0, 0, 1],
            "person_tax_unit_id": [0, 0, 1],
            "person_spm_unit_id": [0, 0, 1],
            "person_family_id": [0, 0, 1],
            "person_marital_unit_id": [0, 0, 1],
            "age": [40.0, 38.0, 25.0],
            "employment_income": [50000.0, 30000.0, 20000.0],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [0, 1],
            "household_weight": [1000.0, 1500.0],
        }
    )
    tax_unit = pd.DataFrame({"tax_unit_id": [0, 1]})
    spm_unit = pd.DataFrame({"spm_unit_id": [0, 1]})
    family = pd.DataFrame({"family_id": [0, 1]})
    marital_unit = pd.DataFrame({"marital_unit_id": [0, 1]})
    return {
        EntityType.PERSON: person,
        EntityType.HOUSEHOLD: household,
        EntityType.TAX_UNIT: tax_unit,
        EntityType.SPM_UNIT: spm_unit,
        EntityType.FAMILY: family,
        MARITAL_UNIT: marital_unit,
    }


# Membership/id columns every group entity contributes (and that the engine
# always supplies). Used as the baseline required set for the mini-contract.
_ID_COLUMNS = (
    "person_id",
    "household_id",
    "tax_unit_id",
    "spm_unit_id",
    "family_id",
    "marital_unit_id",
    "person_household_id",
    "person_tax_unit_id",
    "person_spm_unit_id",
    "person_family_id",
    "person_marital_unit_id",
    "household_weight",
)


def _mini_contract(
    *,
    extra_required: tuple[str, ...] = (),
    forbidden: tuple[str, ...] = (),
    formula_owned_excluded: tuple[str, ...] = (),
) -> ExportContract:
    """A small fabricated contract built around the id/membership columns."""
    return ExportContract(
        required=_ID_COLUMNS + ("age", "employment_income") + extra_required,
        forbidden=forbidden,
        optional=("person_is_puf_clone",),
        formula_owned_excluded=formula_owned_excluded,
    )


def test_contract_from_path_ignores_metadata_keys(tmp_path: Path) -> None:
    payload = {
        "_description": "metadata that must be ignored",
        "_categories": {"required": "doc"},
        "required": ["age", "person_id"],
        "forbidden": ["snap_reported"],
        "ecps_internal_optional": ["person_is_puf_clone"],
        "formula_owned_excluded": ["weeks_worked"],
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    contract = ExportContract.from_path(path)

    assert contract.required == ("age", "person_id")
    assert contract.forbidden == ("snap_reported",)
    assert contract.optional == ("person_is_puf_clone",)
    assert contract.formula_owned_excluded == ("weeks_worked",)


def test_packs_contract_has_252_required() -> None:
    """The frozen packs manifest must expose exactly 252 required columns."""
    contract = ExportContract.from_path(PACKS_CONTRACT_PATH)

    assert len(contract.required) == 252
    assert len(contract.forbidden) == 22
    assert len(contract.optional) == 5
    assert contract.formula_owned_excluded == ("weeks_worked",)
    # Id/membership/weight contract columns are part of the required set.
    for column in _ID_COLUMNS:
        assert column in contract.required


def test_gate_passes_and_round_trips(tmp_path: Path) -> None:
    out = tmp_path / "mp.h5"
    result = export_policyengine_us_dataset(
        _synthetic_world(),
        period=2024,
        output_path=out,
        contract=_mini_contract(),
    )

    assert isinstance(result, ExportGateResult)
    assert result.passed
    assert result.missing_required == ()
    assert result.forbidden_present == ()
    assert out.exists()
    # Exported set is the verified union across all entity tables.
    for column in _ID_COLUMNS + ("age", "employment_income"):
        assert column in result.exported


def test_round_trip_preserves_values(tmp_path: Path) -> None:
    from policyengine_us.data import USSingleYearDataset

    out = tmp_path / "mp.h5"
    export_policyengine_us_dataset(
        _synthetic_world(),
        period=2024,
        output_path=out,
        contract=_mini_contract(),
    )

    reloaded = USSingleYearDataset(file_path=str(out))
    assert list(reloaded.person["person_id"]) == [0, 1, 2]
    assert list(reloaded.person["person_household_id"]) == [0, 0, 1]
    assert list(reloaded.person["age"]) == [40.0, 38.0, 25.0]
    assert list(reloaded.household["household_id"]) == [0, 1]
    assert list(reloaded.household["household_weight"]) == [1000.0, 1500.0]
    assert list(reloaded.marital_unit["marital_unit_id"]) == [0, 1]
    # time_period persists and reloads as a string.
    assert reloaded.time_period == "2024"


def test_gate_fails_on_missing_required(tmp_path: Path) -> None:
    world = _synthetic_world()
    # Drop a required person column entirely.
    world[EntityType.PERSON] = world[EntityType.PERSON].drop(columns=["age"])
    out = tmp_path / "mp.h5"

    result = export_policyengine_us_dataset(
        world,
        period=2024,
        output_path=out,
        contract=_mini_contract(),
    )

    assert not result.passed
    assert "age" in result.missing_required
    # Incomplete exports are not written unless explicitly allowed.
    assert not out.exists()
    assert result.exported == ()


def test_missing_id_column_is_reported_missing(tmp_path: Path) -> None:
    """A group id column the engine omits is reported, never invented."""
    world = _synthetic_world()
    world[EntityType.TAX_UNIT] = world[EntityType.TAX_UNIT].drop(
        columns=["tax_unit_id"]
    )
    out = tmp_path / "mp.h5"

    result = export_policyengine_us_dataset(
        world,
        period=2024,
        output_path=out,
        contract=_mini_contract(),
    )

    assert not result.passed
    assert "tax_unit_id" in result.missing_required
    assert not out.exists()


def test_default_cannot_fabricate_id_when_owning_table_absent(
    tmp_path: Path,
) -> None:
    """A default cannot place an id column when its entity table is missing.

    When the engine omits the marital_unit frame entirely, a default for
    ``marital_unit_id`` has no table to host it, so it is reported missing
    rather than misfiled onto the person table — and nothing is written.
    """
    world = _synthetic_world()
    del world[MARITAL_UNIT]
    out = tmp_path / "mp.h5"

    result = export_policyengine_us_dataset(
        world,
        period=2024,
        output_path=out,
        contract=_mini_contract(),
        defaults={"marital_unit_id": 0},
    )

    assert not result.passed
    assert "marital_unit_id" in result.missing_required
    assert result.defaulted == ()
    # person_marital_unit_id is engine-supplied on the person table, so it is
    # NOT reported missing even though the marital_unit table is absent.
    assert "person_marital_unit_id" not in result.missing_required
    assert not out.exists()


def test_forbidden_columns_dropped(tmp_path: Path) -> None:
    world = _synthetic_world()
    person = world[EntityType.PERSON].copy()
    person["snap_reported"] = [1.0, 2.0, 3.0]
    world[EntityType.PERSON] = person
    out = tmp_path / "mp.h5"

    result = export_policyengine_us_dataset(
        world,
        period=2024,
        output_path=out,
        contract=_mini_contract(forbidden=("snap_reported",)),
    )

    # Forbidden presence fails the gate but the column is still stripped.
    assert not result.passed
    assert "snap_reported" in result.forbidden_present
    assert "snap_reported" in result.dropped
    assert "snap_reported" not in result.exported


def test_formula_owned_columns_silently_dropped(tmp_path: Path) -> None:
    world = _synthetic_world()
    person = world[EntityType.PERSON].copy()
    person["weeks_worked"] = [52, 40, 13]
    world[EntityType.PERSON] = person
    out = tmp_path / "mp.h5"

    result = export_policyengine_us_dataset(
        world,
        period=2024,
        output_path=out,
        contract=_mini_contract(formula_owned_excluded=("weeks_worked",)),
    )

    # Silent drop: removed from output, recorded in dropped, but does NOT fail
    # the gate and is NOT listed as forbidden.
    assert result.passed
    assert "weeks_worked" in result.dropped
    assert "weeks_worked" not in result.forbidden_present
    assert "weeks_worked" not in result.exported


def test_defaults_broadcast_to_correct_entity(tmp_path: Path) -> None:
    from policyengine_us.data import USSingleYearDataset

    out = tmp_path / "mp.h5"
    # state_fips is a household-level variable; in_nyc household; is_blind
    # person. None are present in the synthetic world.
    result = export_policyengine_us_dataset(
        _synthetic_world(),
        period=2024,
        output_path=out,
        contract=_mini_contract(extra_required=("state_fips", "in_nyc", "is_blind")),
        defaults={"state_fips": 6, "in_nyc": False, "is_blind": False},
    )

    assert result.passed
    for column in ("state_fips", "in_nyc", "is_blind"):
        assert column in result.defaulted

    reloaded = USSingleYearDataset(file_path=str(out))
    # Household-level defaults land on the household table (2 rows).
    assert "state_fips" in reloaded.household.columns
    assert list(reloaded.household["state_fips"]) == [6, 6]
    assert "in_nyc" in reloaded.household.columns
    assert "state_fips" not in reloaded.person.columns
    # Person-level default lands on the person table (3 rows).
    assert "is_blind" in reloaded.person.columns
    assert list(reloaded.person["is_blind"]) == [False, False, False]
    assert "is_blind" not in reloaded.household.columns


def test_allow_incomplete_writes_despite_missing(tmp_path: Path) -> None:
    world = _synthetic_world()
    world[EntityType.PERSON] = world[EntityType.PERSON].drop(columns=["age"])
    out = tmp_path / "mp.h5"

    result = export_policyengine_us_dataset(
        world,
        period=2024,
        output_path=out,
        contract=_mini_contract(),
        allow_incomplete=True,
    )

    # Still reported as failing the contract, but the artifact is written.
    assert not result.passed
    assert "age" in result.missing_required
    assert out.exists()
    # Columns that WERE present are verified as exported.
    assert "person_id" in result.exported
    assert "household_weight" in result.exported


def test_requires_person_table(tmp_path: Path) -> None:
    world = _synthetic_world()
    del world[EntityType.PERSON]
    out = tmp_path / "mp.h5"

    with pytest.raises(ValueError, match="person"):
        export_policyengine_us_dataset(
            world,
            period=2024,
            output_path=out,
            contract=_mini_contract(),
        )


def test_rejects_non_h5_output_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\.h5"):
        export_policyengine_us_dataset(
            _synthetic_world(),
            period=2024,
            output_path=tmp_path / "mp.parquet",
            contract=_mini_contract(),
        )


def test_to_dict_is_json_serializable() -> None:
    result = ExportGateResult(
        exported=("age", "person_id"),
        missing_required=("employment_income",),
        forbidden_present=("snap_reported",),
        defaulted=("state_fips",),
        dropped=("weeks_worked",),
    )

    payload = result.to_dict()
    # Round-trips through JSON without error.
    assert json.loads(json.dumps(payload)) == payload
    assert payload["passed"] is False
    assert payload["exported"] == ["age", "person_id"]
