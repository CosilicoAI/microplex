"""Assign PolicyEngine's entity structure onto a CPS ASEC person frame.

PolicyEngine-US groups people into six nested entities:
``person -> {tax_unit, spm_unit, family, marital_unit, household}``. The
microplex pipeline produces a person-level frame that carries the raw CPS ASEC
columns verbatim (Census names such as ``PH_SEQ``, ``A_LINENO``, ``A_SPOUSE``,
``PEPAR1``/``PEPAR2``, ``A_EXPRRP``, ``SPM_ID``, ``PF_SEQ``) alongside
harmonized columns (``household_id``, ``person_id``, ``age``, and income
components). This module turns that frame into the four constructed unit
systems that sit between the person and the household.

The tax-unit system is the one with real federal filing/dependency rules, so it
is delegated to :mod:`microunit` -- the sanctioned tax-unit constructor that was
extracted byte-identically from ``policyengine-us-data``. This module never
re-implements those rules; it maps the microplex frame onto microunit's CPS-like
input contract and reads back its dense ``TAX_ID``, role, and filing-status
output. The remaining three systems (SPM unit, family, marital unit) are
partitions of existing identifiers and are constructed here with deterministic,
vectorized factorization, leaning on :func:`microunit.units.assign_spm_partition`
for the SPM passthrough/fallback.

The :data:`MICROUNIT_REQUIRED_COLUMNS` are the hard contract: the pipeline must
guarantee them on every person row before calling
:func:`assign_us_unit_structure`. Income components are optional -- when only the
harmonized names are present they are mapped onto the raw ASEC names microunit
reads, so the dependency rules still see real income signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "UnitAssignmentResult",
    "assign_us_unit_structure",
    "MICROUNIT_REQUIRED_COLUMNS",
    "TAX_UNIT_FILING_STATUS_COLUMN",
]

#: Raw CPS ASEC columns that :func:`microunit.construct_tax_units` requires on
#: every person row. The microplex frame carries these verbatim from the ASEC.
MICROUNIT_REQUIRED_COLUMNS: tuple[str, ...] = (
    "PH_SEQ",
    "A_LINENO",
    "A_AGE",
    "A_MARITL",
    "A_SPOUSE",
    "PEPAR1",
    "PEPAR2",
    "A_EXPRRP",
)

#: Column on the returned ``tax_unit`` frame carrying the per-unit filing-status
#: input. The name is exactly what :func:`microunit.construct_tax_units` emits;
#: microunit stores it as ``bytes`` (e.g. ``b"JOINT"``) and this module decodes
#: it to ``str`` (``"JOINT"``, ``"HEAD_OF_HOUSEHOLD"``, ``"SINGLE"``,
#: ``"SEPARATE"``, ``"SURVIVING_SPOUSE"``) for downstream PolicyEngine use.
TAX_UNIT_FILING_STATUS_COLUMN = "filing_status_input"

#: Per-row tax-unit role microunit emits, decoded to ``str`` on the person
#: frame (``"HEAD"``, ``"SPOUSE"``, ``"DEPENDENT"``).
_TAX_UNIT_ROLE_COLUMN = "tax_unit_role_input"

#: Whether a person is related to the head/spouse of their tax unit, as emitted
#: by microunit. Carried through verbatim because PolicyEngine filing logic uses
#: it; harmless if downstream ignores it.
_TAX_UNIT_RELATED_FLAG_COLUMN = "is_related_to_head_or_spouse"

#: Person-frame output id columns (dense int64) for the four constructed units.
_PERSON_TAX_UNIT_ID = "person_tax_unit_id"
_PERSON_SPM_UNIT_ID = "person_spm_unit_id"
_PERSON_FAMILY_ID = "person_family_id"
_PERSON_MARITAL_UNIT_ID = "person_marital_unit_id"

#: Mapping from the harmonized income column names microplex carries (see
#: ``microplex.data_sources.cps.PERSON_VARIABLES``) to the raw CPS ASEC value
#: columns that :func:`microunit.construct_tax_units` reads when estimating
#: dependent gross income. When the frame only has the harmonized name, the
#: value is copied onto the ASEC name for the microunit input view so the
#: dependency rules still see real income. Raw ASEC columns already present on
#: the frame always take precedence.
_HARMONIZED_INCOME_TO_ASEC: dict[str, str] = {
    "wage_income": "WSAL_VAL",
    "self_employment_income": "SEMP_VAL",
    "interest_income": "INT_VAL",
    "dividend_income": "DIV_VAL",
    "rental_income": "RNT_VAL",
    "social_security": "SS_VAL",
    "taxable_pension_income": "PNSN_VAL",
    "unemployment_compensation": "UC_VAL",
    "total_person_income": "PTOTVAL",
}


@dataclass(frozen=True)
class UnitAssignmentResult:
    """The constructed PolicyEngine entity structure for a person frame.

    Attributes:
        person: The input frame with four dense ``int64`` id columns appended
            (one per constructed unit system):
            ``person_tax_unit_id``, ``person_spm_unit_id``,
            ``person_family_id``, ``person_marital_unit_id``. The original row
            order and index are preserved. The frame also gains the per-person
            tax-unit role (``tax_unit_role_input``, decoded to ``str``) and the
            ``is_related_to_head_or_spouse`` flag that microunit emits.
        tax_unit: One row per tax unit, keyed by ``tax_unit_id`` (the dense
            ids referenced by ``person.person_tax_unit_id``, in sorted order),
            plus :data:`TAX_UNIT_FILING_STATUS_COLUMN` carrying the
            filing-status input microunit produced (decoded from ``bytes`` to
            ``str``).
        spm_unit: One row per SPM unit, keyed by ``spm_unit_id``.
        family: One row per family, keyed by ``family_id``.
        marital_unit: One row per marital unit, keyed by ``marital_unit_id``.

    Notes:
        Each unit table contains exactly the distinct ids referenced by
        ``person``, in sorted order -- nothing more, nothing less. This is
        enforced internally by :func:`_validate_partition`.
    """

    person: pd.DataFrame
    tax_unit: pd.DataFrame
    spm_unit: pd.DataFrame
    family: pd.DataFrame
    marital_unit: pd.DataFrame


def assign_us_unit_structure(
    person: pd.DataFrame,
    *,
    year: int,
    tax_unit_mode: str = "policyengine",
) -> UnitAssignmentResult:
    """Construct the four PolicyEngine unit systems for a CPS person frame.

    The function is deterministic and side-effect free: the input ``person``
    frame is never mutated, and identical input always yields identical ids.

    Tax units are delegated to :mod:`microunit`'s rules engine
    (``construct_tax_units``), which applies federal filing/dependency rules to
    CPS-like records and returns globally-dense ``TAX_ID`` values, per-person
    roles, and per-unit filing statuses. SPM units, families, and marital units
    are partitions of existing identifiers, factorized here into dense
    ``int64`` ids:

    * **SPM unit**: the native ``SPM_ID`` when fully present (passthrough),
      otherwise ``household_id`` (documented fallback). Delegated to
      :func:`microunit.units.assign_spm_partition`, then densified. A partially
      missing ``SPM_ID`` (any NaN) also triggers the household fallback.
    * **family**: the ``(PH_SEQ, PF_SEQ)`` pair, densely factorized, so two
      families in the same household get distinct ids. Falls back to
      ``household_id`` when ``PF_SEQ`` is absent (documented fallback).
    * **marital unit**: a person is paired with their spouse when
      ``A_SPOUSE > 0`` (pair key = household plus the sorted line-number pair);
      everyone else is a singleton.

    Args:
        person: Person-level frame carrying the raw CPS ASEC columns. The
            columns in :data:`MICROUNIT_REQUIRED_COLUMNS` are mandatory, as is a
            household identifier (``household_id``, or ``PH_SEQ`` from which it
            is derived). Income components are optional: harmonized names (e.g.
            ``wage_income``) are mapped onto the raw ASEC names microunit reads.
        year: Calendar year the records describe. Used by microunit for the
            qualifying-relative gross-income limit and the qualifying-child age
            test.
        tax_unit_mode: microunit tax-unit construction mode. ``"policyengine"``
            (the default) applies the PolicyEngine claiming rules;
            ``"census_documented"`` follows the documented Census tax-model
            flow. Forwarded verbatim to ``construct_tax_units``.

    Returns:
        A :class:`UnitAssignmentResult` with the augmented person frame and the
        four unit tables.

    Raises:
        ImportError: if :mod:`microunit` is not installed.
        ValueError: if any required column is missing (the message names exactly
            which), or if the constructed partitions fail their internal
            consistency check.

    Notes:
        Pipeline guarantees required upstream of this call: every person row
        must carry the :data:`MICROUNIT_REQUIRED_COLUMNS` and a household id.
        ``A_SPOUSE``, ``PEPAR1``, and ``PEPAR2`` must be *line numbers within
        the household* (``A_LINENO`` values), with ``0``/missing meaning "none".
        ``A_EXPRRP`` must use the CPS relationship recode
        (:class:`microunit.CPSRelationshipCode`).
    """
    construct_tax_units = _import_construct_tax_units()

    _require_columns(person, MICROUNIT_REQUIRED_COLUMNS, purpose="unit assignment")
    household_id = _household_id_series(person)

    result = person.copy()

    tax_unit_ids, tax_unit_frame = _construct_tax_units(
        construct_tax_units,
        person,
        year=year,
        mode=tax_unit_mode,
    )
    result[_PERSON_TAX_UNIT_ID] = tax_unit_ids["TAX_ID"]
    result[_TAX_UNIT_ROLE_COLUMN] = tax_unit_ids[_TAX_UNIT_ROLE_COLUMN]
    result[_TAX_UNIT_RELATED_FLAG_COLUMN] = tax_unit_ids[_TAX_UNIT_RELATED_FLAG_COLUMN]

    spm_unit_ids = _assign_spm_unit_ids(person, household_id)
    result[_PERSON_SPM_UNIT_ID] = spm_unit_ids

    family_ids = _assign_family_ids(person, household_id)
    result[_PERSON_FAMILY_ID] = family_ids

    marital_unit_ids = _assign_marital_unit_ids(person, household_id)
    result[_PERSON_MARITAL_UNIT_ID] = marital_unit_ids

    spm_unit = _unit_frame(spm_unit_ids, id_column="spm_unit_id")
    family = _unit_frame(family_ids, id_column="family_id")
    marital_unit = _unit_frame(marital_unit_ids, id_column="marital_unit_id")

    _validate_partition(result[_PERSON_TAX_UNIT_ID], tax_unit_frame["tax_unit_id"])
    _validate_partition(result[_PERSON_SPM_UNIT_ID], spm_unit["spm_unit_id"])
    _validate_partition(result[_PERSON_FAMILY_ID], family["family_id"])
    _validate_partition(
        result[_PERSON_MARITAL_UNIT_ID], marital_unit["marital_unit_id"]
    )

    return UnitAssignmentResult(
        person=result,
        tax_unit=tax_unit_frame,
        spm_unit=spm_unit,
        family=family,
        marital_unit=marital_unit,
    )


# ----------------------------------------------------------------------------
# Tax units (delegated to microunit's rules engine)
# ----------------------------------------------------------------------------


def _import_construct_tax_units():
    """Import ``microunit.construct_tax_units`` with a clear error if absent."""
    try:
        from microunit import construct_tax_units
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "US tax-unit construction requires the 'microunit' package "
            "(the sanctioned tax-unit constructor extracted from "
            "policyengine-us-data). Install it to use "
            "assign_us_unit_structure()."
        ) from exc
    return construct_tax_units


def _construct_tax_units(
    construct_tax_units,
    person: pd.DataFrame,
    *,
    year: int,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run microunit's tax-unit engine and decode its byte-string outputs.

    Returns a ``(person_assignments, tax_unit_frame)`` pair. ``person_assignments``
    is index-aligned to ``person`` with an ``int64`` ``TAX_ID``, a decoded
    ``tax_unit_role_input`` (``str``), and the ``is_related_to_head_or_spouse``
    flag. ``tax_unit_frame`` is keyed by ``tax_unit_id`` (the dense ``TAX_ID``)
    with the decoded :data:`TAX_UNIT_FILING_STATUS_COLUMN`.
    """
    microunit_input = _microunit_input_frame(person)
    assignments, tax_unit = construct_tax_units(microunit_input, year=year, mode=mode)

    assignments = assignments.copy()
    assignments[_TAX_UNIT_ROLE_COLUMN] = _decode_bytes_series(
        assignments[_TAX_UNIT_ROLE_COLUMN]
    )
    # ``TAX_ID`` is already a globally-dense int64 from microunit; align to the
    # caller's index (construct_tax_units preserves the original index).
    assignments["TAX_ID"] = assignments["TAX_ID"].astype("int64")
    assignments = assignments.reindex(person.index)

    tax_unit_frame = tax_unit.copy()
    tax_unit_frame[TAX_UNIT_FILING_STATUS_COLUMN] = _decode_bytes_series(
        tax_unit_frame[TAX_UNIT_FILING_STATUS_COLUMN]
    )
    tax_unit_frame = tax_unit_frame.rename(columns={"TAX_ID": "tax_unit_id"})
    tax_unit_frame["tax_unit_id"] = tax_unit_frame["tax_unit_id"].astype("int64")
    tax_unit_frame = tax_unit_frame.sort_values("tax_unit_id").reset_index(drop=True)
    return assignments, tax_unit_frame


def _microunit_input_frame(person: pd.DataFrame) -> pd.DataFrame:
    """Build the CPS-like view microunit consumes, without touching ``person``.

    Carries the required pointer columns plus any income columns microunit
    reads. Harmonized income columns are copied onto their raw ASEC names when
    the ASEC name is absent, so the dependency rules see real income signal.
    """
    columns = list(MICROUNIT_REQUIRED_COLUMNS)
    # Pass through any ASEC value/flag columns microunit may read if already
    # present (income components, enrollment, disability, PTOTVAL, ...).
    extra_asec = [
        column
        for column in person.columns
        if column not in columns and _is_microunit_optional_column(column)
    ]
    columns.extend(extra_asec)
    view = person.loc[:, [c for c in columns if c in person.columns]].copy()

    for harmonized, asec in _HARMONIZED_INCOME_TO_ASEC.items():
        if asec not in view.columns and harmonized in person.columns:
            view[asec] = pd.to_numeric(person[harmonized], errors="coerce").fillna(0.0)
    return view


def _is_microunit_optional_column(column: str) -> bool:
    """Whether ``column`` is an optional ASEC input microunit reads directly."""
    return (
        column.endswith("_VAL")
        or column in {"PTOTVAL", "A_ENRLW", "A_FTPT", "A_HSCOL"}
        or column.startswith("PEDIS")
    )


def _decode_bytes_series(series: pd.Series) -> pd.Series:
    """Decode a (possibly ``bytes``) string series to ``str``, preserving index."""
    decoded = series.map(
        lambda value: value.decode() if isinstance(value, (bytes, bytearray)) else value
    )
    return decoded.astype(object)


# ----------------------------------------------------------------------------
# SPM units, families, marital units (identifier partitions)
# ----------------------------------------------------------------------------


def _assign_spm_unit_ids(
    person: pd.DataFrame,
    household_id: pd.Series,
) -> pd.Series:
    """Dense SPM-unit ids: native ``SPM_ID`` passthrough, else ``household_id``.

    Delegates the passthrough/fallback decision to
    :func:`microunit.units.assign_spm_partition`. Its family-within-household
    fallback is deliberately disabled (via a non-existent ``family_col``) so the
    fallback goes straight to ``household_id`` as documented, then the raw ids
    are densified to ``int64``.
    """
    from microunit.units import assign_spm_partition

    spm_input = pd.DataFrame(
        {
            "person_id": np.arange(len(person), dtype="int64"),
            "household_id": household_id.to_numpy(),
        },
        index=person.index,
    )
    if "SPM_ID" in person.columns:
        spm_input["SPM_ID"] = person["SPM_ID"].to_numpy()

    partition = assign_spm_partition(
        spm_input,
        family_col="__microplex_no_family_fallback__",
    )
    return _dense_ids(pd.Series(partition.unit_id.to_numpy(), index=person.index))


def _assign_family_ids(
    person: pd.DataFrame,
    household_id: pd.Series,
) -> pd.Series:
    """Dense family ids from the ``(PH_SEQ, PF_SEQ)`` pair.

    Two families in the same household get distinct ids. When ``PF_SEQ`` is
    absent, the family system collapses onto the household (documented
    fallback).
    """
    if "PF_SEQ" not in person.columns:
        return _dense_ids(household_id)
    pair = pd.DataFrame(
        {
            "household": household_id.to_numpy(),
            "family": _integer_codes(person["PF_SEQ"]),
        },
        index=person.index,
    )
    return _dense_pair_ids(pair["household"], pair["family"])


def _assign_marital_unit_ids(
    person: pd.DataFrame,
    household_id: pd.Series,
) -> pd.Series:
    """Dense marital-unit ids: spouse pairs together, everyone else singleton.

    A person is paired with their spouse when ``A_SPOUSE`` is a positive line
    number; the pair key is the household plus the sorted ``(line, spouse_line)``
    pair, so both partners land in the same unit regardless of which row points
    at which. Unpaired people each get their own unit keyed by their own line.
    """
    line_no = _integer_codes(person["A_LINENO"])
    spouse_line = _integer_codes(person["A_SPOUSE"])
    households = household_id.to_numpy()

    has_spouse = spouse_line > 0
    low = np.where(has_spouse, np.minimum(line_no, spouse_line), line_no)
    high = np.where(has_spouse, np.maximum(line_no, spouse_line), line_no)

    pair_key = pd.DataFrame(
        {
            "household": households,
            "low": low,
            "high": high,
        },
        index=person.index,
    )
    return _dense_tuple_ids(pair_key)


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------


def _household_id_series(person: pd.DataFrame) -> pd.Series:
    """Return the household identifier, deriving it from ``PH_SEQ`` if needed."""
    if "household_id" in person.columns:
        source = person["household_id"]
    elif "PH_SEQ" in person.columns:
        source = person["PH_SEQ"]
    else:  # pragma: no cover - PH_SEQ is in MICROUNIT_REQUIRED_COLUMNS
        raise ValueError(
            "Unit assignment requires a household identifier: 'household_id' "
            "or 'PH_SEQ'."
        )
    if source.isna().any():
        raise ValueError("Household identifier contains missing values.")
    return source


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    purpose: str,
) -> None:
    """Raise ``ValueError`` naming exactly which required columns are missing."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) for {purpose}: {', '.join(missing)}."
        )


def _integer_codes(series: pd.Series) -> np.ndarray:
    """Coerce a CPS pointer/line column to a non-negative ``int64`` array.

    Missing values and negatives (CPS "not applicable" sentinels) collapse to
    ``0``, matching microunit's "0 means none" pointer convention.
    """
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    values = np.nan_to_num(values, nan=0.0)
    values = np.where(values < 0, 0.0, values)
    return values.astype("int64")


def _dense_ids(values: pd.Series) -> pd.Series:
    """Factorize ``values`` into dense ``int64`` ids starting at 1.

    Ids are assigned in ascending order of the underlying value so that the
    resulting id ordering is stable and content-determined (not encounter
    order), which keeps unit tables in a meaningful sorted order.
    """
    codes = pd.factorize(values, sort=True)[0]
    return pd.Series((codes + 1).astype("int64"), index=values.index)


def _dense_pair_ids(first: pd.Series, second: pd.Series) -> pd.Series:
    """Dense ``int64`` ids for the ``(first, second)`` pair, sorted by the pair."""
    pair = pd.DataFrame({"first": first.to_numpy(), "second": second.to_numpy()})
    return _dense_tuple_ids(pair.set_axis(first.index))


def _dense_tuple_ids(frame: pd.DataFrame) -> pd.Series:
    """Dense ``int64`` ids for each distinct row tuple, in sorted tuple order."""
    tuples = pd.MultiIndex.from_frame(frame)
    codes, _ = pd.factorize(tuples, sort=True)
    return pd.Series((codes + 1).astype("int64"), index=frame.index)


def _unit_frame(person_unit_ids: pd.Series, *, id_column: str) -> pd.DataFrame:
    """Build a one-row-per-unit frame keyed by the sorted distinct ids."""
    unique_ids = np.sort(np.unique(person_unit_ids.to_numpy()))
    return pd.DataFrame({id_column: unique_ids.astype("int64")})


def _validate_partition(person_unit_ids: pd.Series, unit_ids: pd.Series) -> None:
    """Assert a clean partition: one id per person, unit table = distinct ids.

    Raises:
        ValueError: if any person id is missing, if the unit table is not the
            exact sorted set of ids the persons reference, or if the unit ids
            are not unique.
    """
    if person_unit_ids.isna().any():
        raise ValueError("Every person must receive exactly one unit id.")

    referenced = np.sort(np.unique(person_unit_ids.to_numpy()))
    table = unit_ids.to_numpy()
    if unit_ids.duplicated().any():
        raise ValueError("Unit table contains duplicate ids.")
    sorted_table = np.sort(table)
    if not np.array_equal(sorted_table, table):
        raise ValueError("Unit table ids are not in sorted order.")
    if not np.array_equal(referenced, sorted_table):
        raise ValueError(
            "Unit table must contain exactly the distinct ids referenced by "
            "persons, in sorted order."
        )
