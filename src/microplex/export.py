"""PolicyEngine-US dataset export stage for the Microplex engine.

The engine produces one pandas ``DataFrame`` per entity (person, household,
tax_unit, spm_unit, family, marital_unit). This module persists those frames as
a :class:`policyengine_us.data.USSingleYearDataset` HDF5 file that loads
directly into ``policyengine_us.Microsimulation``, while enforcing a frozen
column-parity contract against the enhanced-CPS (eCPS) baseline.

Layout contract (load-bearing for downstream consumers)
-------------------------------------------------------
``USSingleYearDataset`` flattens every entity table into a single ``{column:
array}`` dict via ``.load()``; ``policyengine-core``'s ``build_from_dataset``
then reconstructs the entity graph. For that reconstruction to succeed the
saved tables must follow PolicyEngine's id/membership conventions exactly:

* The ``person`` table must carry ``person_id`` plus a membership column
  ``person_{group}_id`` for every group entity it belongs to
  (``person_household_id``, ``person_tax_unit_id``, ``person_spm_unit_id``,
  ``person_family_id``, ``person_marital_unit_id``).
* Each group table (``household``, ``tax_unit``, ``spm_unit``, ``family``,
  ``marital_unit``) must carry its own ``{group}_id`` column.
* ``person_{group}_role`` is optional; PolicyEngine defaults absent roles to
  ``"member"``.
* ``household_weight`` is the population weight PolicyEngine reads for
  household-level aggregation.

This module never fabricates id/membership columns: they originate upstream and
their absence is reported as a contract failure, not silently patched.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from microplex.core.entities import EntityType

__all__ = [
    "ExportContract",
    "ExportGateResult",
    "export_policyengine_us_dataset",
]

# USSingleYearDataset table order and names. ``person`` is the only non-group
# table; the remainder are PolicyEngine group entities. Kept as a module
# constant so the export stays consistent with the in-memory dataset the engine
# builds in ``microplex.policyengine_us``.
_PERSON_TABLE = "person"
_GROUP_TABLES: tuple[str, ...] = (
    "household",
    "tax_unit",
    "spm_unit",
    "family",
    "marital_unit",
)
_TABLE_NAMES: tuple[str, ...] = (_PERSON_TABLE, *_GROUP_TABLES)

# Cached lazily so a bare ``import microplex.export`` never pulls in the heavy
# ``policyengine_us`` tax-benefit system. Populated by
# ``_variable_entity_keys`` on first default-broadcast request.
_VARIABLE_ENTITY_KEYS: dict[str, str] | None = None


@dataclass(frozen=True)
class ExportContract:
    """Frozen column-parity contract for a PolicyEngine dataset export.

    The contract is derived from the eCPS baseline H5 export column set. It is
    the authority for which columns a Microplex export must contain to be a
    drop-in eCPS replacement.

    Attributes:
        required: Columns the export MUST contain. A missing required column
            fails the export gate.
        forbidden: Columns the export MUST NOT contain. They are dropped on
            sight and their presence fails the gate.
        optional: eCPS-internal bookkeeping columns that are neither required
            nor forbidden; the export passes them through untouched if present.
        formula_owned_excluded: PolicyEngine formula-owned variables the
            baseline does not persist as inputs. They are silently dropped if
            present so PolicyEngine computes them from its own formulas.
    """

    required: tuple[str, ...]
    forbidden: tuple[str, ...]
    optional: tuple[str, ...]
    formula_owned_excluded: tuple[str, ...]

    @classmethod
    def from_path(cls, path: str | Path) -> ExportContract:
        """Load a contract from a JSON manifest.

        Keys whose name starts with ``"_"`` (documentation/metadata such as
        ``_description`` and ``_categories``) are ignored.

        Args:
            path: Filesystem path to the contract JSON manifest.

        Returns:
            The parsed :class:`ExportContract`.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        sections = {
            key: value for key, value in payload.items() if not key.startswith("_")
        }
        return cls(
            required=_as_str_tuple(sections.get("required", ())),
            forbidden=_as_str_tuple(sections.get("forbidden", ())),
            optional=_as_str_tuple(sections.get("ecps_internal_optional", ())),
            formula_owned_excluded=_as_str_tuple(
                sections.get("formula_owned_excluded", ())
            ),
        )


@dataclass(frozen=True)
class ExportGateResult:
    """Outcome of an export against an :class:`ExportContract`.

    Attributes:
        exported: Every column persisted to (and verified after reloading) the
            output dataset, across all entity tables.
        missing_required: Required columns absent from every entity frame and
            not supplied via defaults.
        forbidden_present: Forbidden columns that were found (and dropped).
        defaulted: Required columns that were broadcast from caller-supplied
            defaults rather than sourced from an entity frame.
        dropped: Columns removed before writing (forbidden plus
            formula-owned-excluded columns that were present).
    """

    exported: tuple[str, ...]
    missing_required: tuple[str, ...]
    forbidden_present: tuple[str, ...]
    defaulted: tuple[str, ...]
    dropped: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Whether the export satisfies the contract.

        ``True`` only when no required column is missing and no forbidden
        column was present.
        """
        return not self.missing_required and not self.forbidden_present

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping of the gate result."""
        return {
            "passed": self.passed,
            "exported": list(self.exported),
            "missing_required": list(self.missing_required),
            "forbidden_present": list(self.forbidden_present),
            "defaulted": list(self.defaulted),
            "dropped": list(self.dropped),
        }


def export_policyengine_us_dataset(
    entity_frames: Mapping[Any, pd.DataFrame],
    *,
    period: int,
    output_path: str | Path,
    contract: ExportContract | None = None,
    defaults: Mapping[str, object] | None = None,
    allow_incomplete: bool = False,
) -> ExportGateResult:
    """Export per-entity frames as a PolicyEngine ``USSingleYearDataset``.

    The export drops forbidden and formula-owned columns, optionally broadcasts
    caller-supplied defaults for contract-required columns that no entity frame
    provides, writes the dataset as HDF5, and verifies the write by reloading
    it and confirming the persisted columns survived.

    Id and membership columns (``person_id``, ``{group}_id``,
    ``person_{group}_id``) are never fabricated — they originate upstream, so
    when they are absent the export records them as ``missing_required`` rather
    than inventing values.

    Args:
        entity_frames: Per-entity frames keyed by
            :class:`microplex.core.entities.EntityType` (or any key whose
            ``.value`` matches a PolicyEngine table name). Recognized table
            names are ``person``, ``household``, ``tax_unit``, ``spm_unit``,
            ``family`` and ``marital_unit``.
        period: Dataset time period (e.g. ``2024``).
        output_path: Destination ``.h5`` path for the saved dataset.
        contract: Column-parity contract to enforce. When ``None`` the gate
            performs no required/forbidden/formula checks and simply persists
            the supplied frames.
        defaults: Optional mapping of column name to scalar value. A default is
            applied only for a contract-required column that no entity frame
            already provides; the value is broadcast onto the correct entity
            table (chosen from PolicyEngine variable metadata) and recorded in
            ``defaulted``.
        allow_incomplete: When ``False`` (the default) the dataset is not
            written if any required column is missing — incomplete exports are
            never persisted. Pass ``True`` to force a write anyway; this is
            intended only for smoke runs that need an artifact even though the
            contract is not yet satisfied.

    Returns:
        An :class:`ExportGateResult` describing what was exported, what was
        missing, dropped, or defaulted, and whether the contract passed.

    Raises:
        ValueError: If the supplied frames do not contain a ``person`` table,
            if ``output_path`` does not end in ``.h5``, or if the written
            dataset fails to round-trip (a column expected after reload is
            absent).
    """
    contract = contract or _EMPTY_CONTRACT
    defaults = dict(defaults or {})
    output_path = Path(output_path)
    if output_path.suffix != ".h5":
        raise ValueError(f"output_path must end with '.h5', got {output_path.name!r}.")

    tables = _frames_by_table_name(entity_frames)
    if _PERSON_TABLE not in tables:
        raise ValueError(
            "entity_frames must include a 'person' table "
            f"(EntityType.PERSON); got tables {sorted(tables)}."
        )

    forbidden_set = set(contract.forbidden)
    drop_on_sight = forbidden_set | set(contract.formula_owned_excluded)

    dropped: set[str] = set()
    forbidden_present: set[str] = set()
    for name, frame in tables.items():
        present_drops = drop_on_sight.intersection(frame.columns)
        if present_drops:
            tables[name] = frame.drop(columns=sorted(present_drops))
        dropped.update(present_drops)
        forbidden_present.update(forbidden_set.intersection(present_drops))

    # Columns sourced directly from the engine frames (post-drop).
    present_columns: set[str] = set()
    for frame in tables.values():
        present_columns.update(frame.columns)

    # Broadcast defaults for required columns that no frame provides.
    defaulted: set[str] = set()
    missing_required: list[str] = []
    for column in contract.required:
        if column in present_columns:
            continue
        if column in defaults:
            target = _broadcast_default(tables, column, defaults[column])
            present_columns.add(column)
            defaulted.add(column)
            if target is None:
                # No frame to host the column (e.g. an empty group table that
                # was not supplied); cannot satisfy the requirement.
                missing_required.append(column)
                present_columns.discard(column)
                defaulted.discard(column)
        else:
            missing_required.append(column)

    result_stub = ExportGateResult(
        exported=(),
        missing_required=tuple(missing_required),
        forbidden_present=tuple(sorted(forbidden_present)),
        defaulted=tuple(sorted(defaulted)),
        dropped=tuple(sorted(dropped)),
    )

    if missing_required and not allow_incomplete:
        # Never persist an incomplete dataset unless explicitly forced.
        return result_stub

    exported = _write_and_verify(tables, period=int(period), output_path=output_path)
    return ExportGateResult(
        exported=exported,
        missing_required=result_stub.missing_required,
        forbidden_present=result_stub.forbidden_present,
        defaulted=result_stub.defaulted,
        dropped=result_stub.dropped,
    )


_EMPTY_CONTRACT = ExportContract(
    required=(),
    forbidden=(),
    optional=(),
    formula_owned_excluded=(),
)


def _as_str_tuple(values: Any) -> tuple[str, ...]:
    """Coerce a JSON list (or any iterable) into a tuple of strings."""
    if values is None:
        return ()
    return tuple(str(value) for value in values)


def _frames_by_table_name(
    entity_frames: Mapping[Any, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Map caller frames to PolicyEngine table names via ``EntityType.value``.

    Accepts keys that are :class:`EntityType` members, raw strings, or any
    object exposing a ``.value`` matching a recognized table name. Frames whose
    resolved name is not a PolicyEngine table are ignored, so callers may pass
    extra entities (e.g. records) without breaking the export.
    """
    recognized = set(_TABLE_NAMES)
    resolved: dict[str, pd.DataFrame] = {}
    for key, frame in entity_frames.items():
        name = _table_name_for_key(key)
        if name in recognized:
            resolved[name] = frame.copy()
    return resolved


def _table_name_for_key(key: Any) -> str | None:
    """Return the PolicyEngine table name a frame key refers to, if any."""
    if isinstance(key, EntityType):
        return key.value
    value = getattr(key, "value", None)
    if isinstance(value, str):
        return value
    if isinstance(key, str):
        return key
    return None


def _broadcast_default(
    tables: dict[str, pd.DataFrame],
    column: str,
    value: object,
) -> str | None:
    """Broadcast ``value`` for ``column`` onto its owning entity table.

    The owning entity is taken from PolicyEngine variable metadata
    (``variable.entity.key``). A variable unknown to PolicyEngine is placed on
    the ``person`` table as a safe default. When the resolved owning table is a
    group entity that was not supplied, the default cannot be placed correctly
    and ``None`` is returned (the caller records the column as missing) rather
    than misfiling it onto another entity.

    Returns:
        The table name written to, or ``None`` if the owning table is absent.
    """
    entity_key = _variable_entity_keys().get(column, _PERSON_TABLE)
    if entity_key not in tables:
        return None
    frame = tables[entity_key]
    frame[column] = value
    return entity_key


def _variable_entity_keys() -> dict[str, str]:
    """Return a cached ``{variable_name: entity_key}`` map from policyengine-us.

    The PolicyEngine ``CountryTaxBenefitSystem`` is instantiated lazily and
    cached at module level so importing this module stays cheap. If
    ``policyengine_us`` is unavailable an empty map is returned, which falls all
    default placement back to the ``person`` table.
    """
    global _VARIABLE_ENTITY_KEYS
    if _VARIABLE_ENTITY_KEYS is None:
        try:
            from policyengine_us import CountryTaxBenefitSystem
        except ImportError:  # pragma: no cover - optional dependency
            _VARIABLE_ENTITY_KEYS = {}
        else:
            system = CountryTaxBenefitSystem()
            _VARIABLE_ENTITY_KEYS = {
                name: variable.entity.key for name, variable in system.variables.items()
            }
    return _VARIABLE_ENTITY_KEYS


def _write_and_verify(
    tables: Mapping[str, pd.DataFrame],
    *,
    period: int,
    output_path: Path,
) -> tuple[str, ...]:
    """Persist tables as a ``USSingleYearDataset`` and verify the round-trip.

    Builds the dataset from the supplied tables (absent group tables default to
    empty), saves it, then reloads it and asserts every persisted column
    survived. Returns the sorted union of verified columns.

    Raises:
        ImportError: If ``policyengine_us`` is not installed.
        ValueError: If a column expected after reload is missing.
    """
    try:
        from policyengine_us.data import USSingleYearDataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "PolicyEngine-US export requires policyengine-us to be installed."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    person = tables[_PERSON_TABLE]
    dataset_kwargs: dict[str, pd.DataFrame] = {_PERSON_TABLE: person}
    for name in _GROUP_TABLES:
        dataset_kwargs[name] = tables.get(name, pd.DataFrame())

    dataset = USSingleYearDataset(time_period=int(period), **dataset_kwargs)
    dataset.save(str(output_path))

    # ``.save`` only writes tables with len > 0; columns from non-empty tables
    # must reappear on reload. This catches a silently dropped/corrupted write.
    expected_columns: set[str] = set()
    for name in _TABLE_NAMES:
        frame = dataset_kwargs[name]
        if len(frame) > 0:
            expected_columns.update(frame.columns)

    reloaded = USSingleYearDataset(file_path=str(output_path))
    persisted_columns: set[str] = set()
    for frame in (
        reloaded.person,
        reloaded.household,
        reloaded.tax_unit,
        reloaded.spm_unit,
        reloaded.family,
        reloaded.marital_unit,
    ):
        persisted_columns.update(frame.columns)

    missing = expected_columns - persisted_columns
    if missing:
        raise ValueError(
            "Export round-trip verification failed; columns absent after "
            f"reload: {sorted(missing)}."
        )
    return tuple(sorted(persisted_columns))
