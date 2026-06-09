"""Tests for the US unit-assignment module (``microplex.units``).

Covers the four constructed PolicyEngine unit systems on small synthetic CPS
ASEC households: tax units delegated to microunit (married couple with kids,
single filer, multigenerational household where the adult child is a separate
tax unit), the SPM passthrough and household fallback, family-within-household
splitting, marital-unit spouse pairing, determinism, and the validation errors.

CPS pointer semantics are constructed realistically: ``A_SPOUSE``, ``PEPAR1``,
and ``PEPAR2`` are *line numbers* (``A_LINENO`` values) within the household,
``0`` meaning "none"; ``A_EXPRRP`` uses the Census relationship recode
(:class:`microunit.CPSRelationshipCode`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microplex.units import (
    MICROUNIT_REQUIRED_COLUMNS,
    TAX_UNIT_FILING_STATUS_COLUMN,
    UnitAssignmentResult,
    _microunit_input_frame,
    assign_us_unit_structure,
)

# Census A_EXPRRP relationship-to-reference-person recode values.
# (Mirrors microunit.CPSRelationshipCode; spelled out here so the tests document
#  the exact codes they exercise without coupling to the enum import.)
REF_PERSON_WITH_RELATIVES = 1
REF_PERSON_WITHOUT_RELATIVES = 2
WIFE = 4
OWN_CHILD = 5
GRANDCHILD = 7

# CPS marital-status (A_MARITL) codes used below.
MARRIED_SPOUSE_PRESENT = 1
WIDOWED = 4
NEVER_MARRIED = 7


def _person_row(
    *,
    ph_seq: int,
    line: int,
    age: int,
    marital: int,
    spouse: int = 0,
    parent1: int = 0,
    parent2: int = 0,
    relationship: int,
    spm_id: int | None = None,
    family: int = 1,
    wage: float = 0.0,
) -> dict:
    """Build one CPS-like person record with both raw and harmonized columns."""
    row = {
        "PH_SEQ": ph_seq,
        "A_LINENO": line,
        "A_AGE": age,
        "A_MARITL": marital,
        "A_SPOUSE": spouse,
        "PEPAR1": parent1,
        "PEPAR2": parent2,
        "A_EXPRRP": relationship,
        "PF_SEQ": family,
        "wage_income": wage,
        "household_id": ph_seq,
        "age": age,
    }
    if spm_id is not None:
        row["SPM_ID"] = spm_id
    return row


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Stack person records and attach a dense person_id and a shifted index."""
    frame = pd.DataFrame(rows)
    frame["person_id"] = np.arange(len(frame), dtype="int64")
    # Use a non-trivial (non-range) index to catch index-alignment bugs.
    frame.index = pd.RangeIndex(start=100, stop=100 + len(frame))
    return frame


def _married_with_kids() -> pd.DataFrame:
    """A married couple (lines 1,2) with two own children (lines 3,4)."""
    return _frame(
        [
            _person_row(
                ph_seq=10,
                line=1,
                age=42,
                marital=MARRIED_SPOUSE_PRESENT,
                spouse=2,
                relationship=REF_PERSON_WITH_RELATIVES,
                spm_id=500,
                wage=70000,
            ),
            _person_row(
                ph_seq=10,
                line=2,
                age=40,
                marital=MARRIED_SPOUSE_PRESENT,
                spouse=1,
                relationship=WIFE,
                spm_id=500,
                wage=30000,
            ),
            _person_row(
                ph_seq=10,
                line=3,
                age=12,
                marital=NEVER_MARRIED,
                parent1=1,
                parent2=2,
                relationship=OWN_CHILD,
                spm_id=500,
            ),
            _person_row(
                ph_seq=10,
                line=4,
                age=9,
                marital=NEVER_MARRIED,
                parent1=1,
                parent2=2,
                relationship=OWN_CHILD,
                spm_id=500,
            ),
        ]
    )


def _single_filer() -> pd.DataFrame:
    """A lone adult household (line 1)."""
    return _frame(
        [
            _person_row(
                ph_seq=20,
                line=1,
                age=33,
                marital=NEVER_MARRIED,
                relationship=REF_PERSON_WITHOUT_RELATIVES,
                spm_id=700,
                wage=48000,
            ),
        ]
    )


def _multigenerational() -> pd.DataFrame:
    """Grandparent (1), independent adult child (2), grandchild (3).

    The adult child earns well above the dependent gross-income limit, so the
    tax rules must put them in their own tax unit (as HEAD), claiming the
    grandchild -- not as a dependent of the grandparent. The two families are
    distinguished by PF_SEQ within the single household.
    """
    return _frame(
        [
            _person_row(
                ph_seq=30,
                line=1,
                age=66,
                marital=WIDOWED,
                relationship=REF_PERSON_WITH_RELATIVES,
                spm_id=900,
                family=1,
                wage=20000,
            ),
            _person_row(
                ph_seq=30,
                line=2,
                age=27,
                marital=NEVER_MARRIED,
                parent1=1,
                relationship=OWN_CHILD,
                spm_id=900,
                family=2,
                wage=55000,
            ),
            _person_row(
                ph_seq=30,
                line=3,
                age=3,
                marital=NEVER_MARRIED,
                parent1=2,
                relationship=GRANDCHILD,
                spm_id=900,
                family=2,
            ),
        ]
    )


def _by_line(result: UnitAssignmentResult, ph_seq: int, line: int) -> pd.Series:
    """Return the augmented person row for a given household line number."""
    person = result.person
    mask = (person["PH_SEQ"] == ph_seq) & (person["A_LINENO"] == line)
    matched = person[mask]
    assert len(matched) == 1, f"expected one row for PH_SEQ={ph_seq} line={line}"
    return matched.iloc[0]


class TestTaxUnits:
    def test_married_couple_with_kids_is_one_joint_unit(self) -> None:
        result = assign_us_unit_structure(_married_with_kids(), year=2024)
        head = _by_line(result, 10, 1)
        spouse = _by_line(result, 10, 2)
        child_a = _by_line(result, 10, 3)
        child_b = _by_line(result, 10, 4)

        tax_id = head["person_tax_unit_id"]
        assert spouse["person_tax_unit_id"] == tax_id
        assert child_a["person_tax_unit_id"] == tax_id
        assert child_b["person_tax_unit_id"] == tax_id

        assert head["tax_unit_role_input"] == "HEAD"
        assert spouse["tax_unit_role_input"] == "SPOUSE"
        assert child_a["tax_unit_role_input"] == "DEPENDENT"
        assert child_b["tax_unit_role_input"] == "DEPENDENT"

        filing = result.tax_unit.set_index("tax_unit_id")[TAX_UNIT_FILING_STATUS_COLUMN]
        assert filing.loc[tax_id] == "JOINT"

    def test_single_filer_is_one_single_unit(self) -> None:
        result = assign_us_unit_structure(_single_filer(), year=2024)
        person = _by_line(result, 20, 1)
        assert person["tax_unit_role_input"] == "HEAD"
        filing = result.tax_unit.set_index("tax_unit_id")[TAX_UNIT_FILING_STATUS_COLUMN]
        assert filing.loc[person["person_tax_unit_id"]] == "SINGLE"
        assert len(result.tax_unit) == 1

    def test_independent_adult_child_is_a_separate_tax_unit(self) -> None:
        result = assign_us_unit_structure(_multigenerational(), year=2024)
        grandparent = _by_line(result, 30, 1)
        adult_child = _by_line(result, 30, 2)
        grandchild = _by_line(result, 30, 3)

        # The adult child files separately from the grandparent...
        assert adult_child["person_tax_unit_id"] != grandparent["person_tax_unit_id"]
        assert adult_child["tax_unit_role_input"] == "HEAD"
        assert grandparent["tax_unit_role_input"] == "HEAD"
        # ...and claims the grandchild as a dependent in their own unit.
        assert grandchild["person_tax_unit_id"] == adult_child["person_tax_unit_id"]
        assert grandchild["tax_unit_role_input"] == "DEPENDENT"

    def test_filing_status_decoded_to_str_not_bytes(self) -> None:
        result = assign_us_unit_structure(_married_with_kids(), year=2024)
        values = result.tax_unit[TAX_UNIT_FILING_STATUS_COLUMN].tolist()
        assert all(isinstance(value, str) for value in values)
        roles = result.person["tax_unit_role_input"].tolist()
        assert all(isinstance(value, str) for value in roles)

    def test_tax_unit_ids_globally_dense_across_households(self) -> None:
        frame = pd.concat(
            [_married_with_kids(), _single_filer(), _multigenerational()],
            ignore_index=True,
        )
        result = assign_us_unit_structure(frame, year=2024)
        ids = sorted(result.person["person_tax_unit_id"].unique().tolist())
        # 4 tax units: couple, single, grandparent, adult-child-with-grandchild.
        assert ids == [1, 2, 3, 4]
        assert result.tax_unit["tax_unit_id"].tolist() == [1, 2, 3, 4]

    def test_census_documented_mode_is_forwarded(self) -> None:
        # Mode must reach microunit; census_documented yields a valid partition.
        result = assign_us_unit_structure(
            _married_with_kids(),
            year=2024,
            tax_unit_mode="census_documented",
        )
        head = _by_line(result, 10, 1)
        spouse = _by_line(result, 10, 2)
        assert head["person_tax_unit_id"] == spouse["person_tax_unit_id"]


class TestSpmUnits:
    def test_native_spm_id_is_preserved_as_partition(self) -> None:
        frame = pd.concat([_married_with_kids(), _single_filer()], ignore_index=True)
        result = assign_us_unit_structure(frame, year=2024)
        # The four HH-10 members share one SPM unit; HH-20 is a second.
        hh10 = result.person[result.person["PH_SEQ"] == 10]["person_spm_unit_id"]
        hh20 = result.person[result.person["PH_SEQ"] == 20]["person_spm_unit_id"]
        assert hh10.nunique() == 1
        assert hh20.nunique() == 1
        assert set(hh10) != set(hh20)
        assert result.spm_unit["spm_unit_id"].tolist() == [1, 2]

    def test_distinct_spm_ids_within_household_split(self) -> None:
        # Two SPM units inside one household (e.g. an unrelated subfamily).
        frame = _frame(
            [
                _person_row(
                    ph_seq=40,
                    line=1,
                    age=50,
                    marital=NEVER_MARRIED,
                    relationship=REF_PERSON_WITH_RELATIVES,
                    spm_id=10,
                ),
                _person_row(
                    ph_seq=40,
                    line=2,
                    age=48,
                    marital=NEVER_MARRIED,
                    relationship=REF_PERSON_WITHOUT_RELATIVES,
                    spm_id=20,
                ),
            ]
        )
        result = assign_us_unit_structure(frame, year=2024)
        ids = result.person["person_spm_unit_id"].tolist()
        assert ids[0] != ids[1]
        assert result.spm_unit["spm_unit_id"].tolist() == [1, 2]

    def test_missing_spm_id_falls_back_to_household(self) -> None:
        frame = _married_with_kids().drop(columns=["SPM_ID"])
        result = assign_us_unit_structure(frame, year=2024)
        # All four members share a single SPM unit derived from the household.
        assert result.person["person_spm_unit_id"].nunique() == 1
        assert result.spm_unit["spm_unit_id"].tolist() == [1]

    def test_partial_spm_id_falls_back_to_household(self) -> None:
        frame = _married_with_kids()
        frame.loc[frame.index[1], "SPM_ID"] = np.nan  # one NaN voids passthrough
        result = assign_us_unit_structure(frame, year=2024)
        assert result.person["person_spm_unit_id"].nunique() == 1


class TestFamilies:
    def test_family_splits_within_household_by_pf_seq(self) -> None:
        result = assign_us_unit_structure(_multigenerational(), year=2024)
        grandparent = _by_line(result, 30, 1)
        adult_child = _by_line(result, 30, 2)
        grandchild = _by_line(result, 30, 3)
        # PF_SEQ 1 vs 2 -> two families even though it is one household.
        assert grandparent["person_family_id"] != adult_child["person_family_id"]
        assert grandchild["person_family_id"] == adult_child["person_family_id"]

    def test_family_falls_back_to_household_without_pf_seq(self) -> None:
        frame = _multigenerational().drop(columns=["PF_SEQ"])
        result = assign_us_unit_structure(frame, year=2024)
        # Without PF_SEQ the whole household collapses to one family.
        assert result.person["person_family_id"].nunique() == 1


class TestMaritalUnits:
    def test_spouses_share_a_marital_unit(self) -> None:
        result = assign_us_unit_structure(_married_with_kids(), year=2024)
        head = _by_line(result, 10, 1)
        spouse = _by_line(result, 10, 2)
        assert head["person_marital_unit_id"] == spouse["person_marital_unit_id"]

    def test_children_are_singleton_marital_units(self) -> None:
        result = assign_us_unit_structure(_married_with_kids(), year=2024)
        couple_id = _by_line(result, 10, 1)["person_marital_unit_id"]
        child_a = _by_line(result, 10, 3)["person_marital_unit_id"]
        child_b = _by_line(result, 10, 4)["person_marital_unit_id"]
        assert child_a != couple_id
        assert child_b != couple_id
        assert child_a != child_b

    def test_symmetric_spouse_pointers_pair_into_one_unit(self) -> None:
        # CPS A_SPOUSE is symmetric: each spouse carries the other's line number.
        # Both rows compute the same sorted (low, high) pair key, so they share a
        # marital unit regardless of which row is processed first.
        result = assign_us_unit_structure(_married_with_kids(), year=2024)
        unit_ids = result.person.set_index("A_LINENO")["person_marital_unit_id"]
        assert unit_ids.loc[1] == unit_ids.loc[2]

    def test_dangling_spouse_pointer_does_not_pair_and_does_not_crash(self) -> None:
        # A one-directional/dangling A_SPOUSE (only one partner points) is not a
        # valid CPS marriage record; it must degrade to singletons, not error.
        frame = _married_with_kids()
        frame.loc[frame.index[1], "A_SPOUSE"] = 0  # break the line-2 -> line-1 link
        result = assign_us_unit_structure(frame, year=2024)
        unit_ids = result.person.set_index("A_LINENO")["person_marital_unit_id"]
        assert unit_ids.loc[1] != unit_ids.loc[2]

    def test_single_person_household_is_its_own_marital_unit(self) -> None:
        result = assign_us_unit_structure(_single_filer(), year=2024)
        assert result.marital_unit["marital_unit_id"].tolist() == [1]


class TestStructureInvariants:
    def test_caller_frame_is_not_mutated(self) -> None:
        frame = _married_with_kids()
        before = frame.copy()
        assign_us_unit_structure(frame, year=2024)
        pd.testing.assert_frame_equal(frame, before)

    def test_index_is_preserved(self) -> None:
        frame = _married_with_kids()
        result = assign_us_unit_structure(frame, year=2024)
        assert result.person.index.tolist() == frame.index.tolist()

    def test_all_id_columns_are_int64_and_complete(self) -> None:
        frame = pd.concat(
            [_married_with_kids(), _single_filer(), _multigenerational()],
            ignore_index=True,
        )
        result = assign_us_unit_structure(frame, year=2024)
        for column in (
            "person_tax_unit_id",
            "person_spm_unit_id",
            "person_family_id",
            "person_marital_unit_id",
        ):
            assert result.person[column].dtype == np.int64
            assert result.person[column].notna().all()

    def test_unit_tables_are_exactly_referenced_ids_sorted(self) -> None:
        frame = pd.concat(
            [_married_with_kids(), _single_filer(), _multigenerational()],
            ignore_index=True,
        )
        result = assign_us_unit_structure(frame, year=2024)
        pairs = [
            (result.person["person_tax_unit_id"], result.tax_unit["tax_unit_id"]),
            (result.person["person_spm_unit_id"], result.spm_unit["spm_unit_id"]),
            (result.person["person_family_id"], result.family["family_id"]),
            (
                result.person["person_marital_unit_id"],
                result.marital_unit["marital_unit_id"],
            ),
        ]
        for person_ids, table_ids in pairs:
            expected = sorted(person_ids.unique().tolist())
            assert table_ids.tolist() == expected
            assert table_ids.is_monotonic_increasing
            assert not table_ids.duplicated().any()


class TestDeterminism:
    def test_identical_input_yields_identical_ids(self) -> None:
        frame = pd.concat(
            [_married_with_kids(), _single_filer(), _multigenerational()],
            ignore_index=True,
        )
        first = assign_us_unit_structure(frame.copy(), year=2024)
        second = assign_us_unit_structure(frame.copy(), year=2024)
        pd.testing.assert_frame_equal(first.person, second.person)
        pd.testing.assert_frame_equal(first.tax_unit, second.tax_unit)
        pd.testing.assert_frame_equal(first.spm_unit, second.spm_unit)
        pd.testing.assert_frame_equal(first.family, second.family)
        pd.testing.assert_frame_equal(first.marital_unit, second.marital_unit)


class TestValidation:
    @pytest.mark.parametrize("missing", list(MICROUNIT_REQUIRED_COLUMNS))
    def test_missing_required_column_names_it(self, missing: str) -> None:
        frame = _married_with_kids().drop(columns=[missing])
        with pytest.raises(ValueError, match=missing):
            assign_us_unit_structure(frame, year=2024)

    def test_missing_household_identifier_is_reported(self) -> None:
        # Drop both household_id and PH_SEQ -> PH_SEQ is required, so the
        # required-column check fires first naming PH_SEQ.
        frame = _single_filer().drop(columns=["household_id", "PH_SEQ"])
        with pytest.raises(ValueError, match="PH_SEQ"):
            assign_us_unit_structure(frame, year=2024)

    def test_missing_microunit_raises_clear_error(self, monkeypatch) -> None:
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "microunit" or name.startswith("microunit."):
                raise ImportError("microunit not available")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        with pytest.raises(ImportError, match="microunit"):
            assign_us_unit_structure(_single_filer(), year=2024)


class TestHarmonizedIncomeMapping:
    """The microunit input view sources income from harmonized names.

    microunit reads raw CPS ASEC value columns (``WSAL_VAL``, ``SEMP_VAL``,
    ``SS_VAL``, ...) when estimating dependent gross income. The microplex frame
    may carry only the harmonized names (``wage_income``, ...); the input view
    must copy them onto the ASEC names so the dependency rules see real income.
    """

    def test_harmonized_income_is_mapped_onto_asec_names(self) -> None:
        frame = _frame(
            [
                _person_row(
                    ph_seq=50,
                    line=1,
                    age=50,
                    marital=NEVER_MARRIED,
                    relationship=REF_PERSON_WITH_RELATIVES,
                    wage=80000,
                ),
            ]
        )
        frame["self_employment_income"] = [1200.0]
        frame["social_security"] = [340.0]
        assert "WSAL_VAL" not in frame.columns  # only the harmonized names exist

        view = _microunit_input_frame(frame)
        assert view["WSAL_VAL"].tolist() == [80000.0]
        assert view["SEMP_VAL"].tolist() == [1200.0]
        assert view["SS_VAL"].tolist() == [340.0]
        # Harmonized / non-ASEC columns are not carried into the engine view.
        assert "wage_income" not in view.columns
        assert "household_id" not in view.columns

    def test_raw_asec_income_takes_precedence_over_harmonized(self) -> None:
        frame = _frame(
            [
                _person_row(
                    ph_seq=50,
                    line=1,
                    age=50,
                    marital=NEVER_MARRIED,
                    relationship=REF_PERSON_WITH_RELATIVES,
                    wage=80000,  # harmonized wage_income
                ),
            ]
        )
        frame["WSAL_VAL"] = [12345.0]  # raw ASEC value present too
        view = _microunit_input_frame(frame)
        assert view["WSAL_VAL"].tolist() == [12345.0]

    def test_input_view_does_not_mutate_caller(self) -> None:
        frame = _married_with_kids()
        before = frame.copy()
        _microunit_input_frame(frame)
        pd.testing.assert_frame_equal(frame, before)
