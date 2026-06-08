"""Generic Arch target derivations.

The Arch target DB / consumer-fact artifacts give *raw* target records (one
per source fact). Producing the calibration target surface requires several
derivations on top of those raw records:

- **component sum** — synthesize a composite target (e.g. SALT) by summing
  declared component targets at the same cell;
- latest **carry-forward** — keep the most recent record per target cell when a
  source publishes the same cell across periods with a lag;
- **state to national rollup** — sum state records into a national total;
- **BEA** state ``employment_income_before_lsr`` synthesis;
- **SOI aging** — age count vs. amount records by source year.

These are country-agnostic *algorithms*; the country-specific inputs (which
variables sum into which composite, which source aliases collapse, how a cell's
geography level is derived) are **injected** so the engine stays generic and the
US pack only declares data. This module is the home for that logic, ported
faithfully from the legacy ``microplex_us.targets.arch`` pipeline and kept
representation-light so it can be wired onto whichever loaded-record type the
target layer settles on.

Records are :class:`ArchTargetRecord` — a frozen dataclass carrying the fields
the derivations read. A loader adapter maps the on-disk Arch DB / consumer-fact
rows into these; the derivations and their tests do not depend on that loader.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha1
from typing import Any

__all__ = [
    "ArchTargetRecord",
    "ComponentSumMap",
    "component_sum_records",
    "with_component_sum_records",
    "default_geo_level",
    "default_normalize_source",
]


@dataclass(frozen=True)
class ArchTargetRecord:
    """A single Arch target record the derivations operate on.

    Mirrors the fields the legacy ``microplex_us`` Arch pipeline reads. Most
    fields default so tests and adapters can build minimal records; the
    derivations only read the subset each needs.
    """

    variable: str
    period: int
    value: float
    target_type: str = "AMOUNT"
    geographic_level: str | None = None
    geography_id: str | None = None
    source: str = ""
    source_table: str | None = None
    source_url: str | None = None
    notes: str | None = None
    jurisdiction: str | None = None
    constraints: tuple[tuple[str, str, str], ...] = ()
    source_period: int | None = None
    aging_factors: Any | None = None
    unit: str | None = None
    target_id: int = 0
    stratum_id: int = 0
    source_record_id: str | None = None
    source_cell_keys: tuple[str, ...] = ()
    source_row_keys: tuple[str, ...] = ()
    aggregate_fact_key: str | None = None
    semantic_fact_key: str | None = None
    source_target_id: int | None = None
    source_stratum_id: int | None = None
    concept: str | None = None
    source_concept: str | None = None
    concept_relation: str | None = None
    concept_authority: str | None = None
    concept_evidence_notes: str | None = None


#: output variable -> the component variables that sum into it.
ComponentSumMap = Mapping[str, Sequence[str]]

GeoLevelFn = Callable[[ArchTargetRecord], str | None]
NormalizeSourceFn = Callable[[str], str]


def default_geo_level(record: ArchTargetRecord) -> str | None:
    """Default cell geography level: the record's declared ``geographic_level``.

    The US pack injects a constraints-aware version; the generic default is the
    declared level, which is sufficient to group records at the same cell when
    constraints are not the discriminating dimension.
    """
    return record.geographic_level


def default_normalize_source(source: str) -> str:
    """Default source normalization: upper-case, dashes to underscores.

    The US pack injects an alias map (e.g. collapsing ``IRS-SOI`` variants); the
    generic default just canonicalizes case/separators.
    """
    return str(source).upper().replace("-", "_")


def _component_sum_record_key(
    record: ArchTargetRecord,
    *,
    output_variable: str,
    geo_level: GeoLevelFn,
    normalize_source: NormalizeSourceFn,
) -> tuple[Any, ...]:
    """Cell key identifying records that may sum into one composite output.

    Same tuple shape as the legacy pipeline: the key pins everything that must
    match for components to belong to the same composite (output variable,
    target type, period, geography, constraints, source, source period, aging,
    unit).
    """
    return (
        output_variable,
        record.target_type,
        record.period,
        geo_level(record),
        record.geography_id,
        tuple(sorted(record.constraints)),
        normalize_source(record.source),
        record.source_period,
        record.aging_factors,
        record.unit,
    )


def component_sum_records(
    records: Sequence[ArchTargetRecord],
    *,
    component_sum_map: ComponentSumMap,
    geo_level: GeoLevelFn = default_geo_level,
    normalize_source: NormalizeSourceFn = default_normalize_source,
) -> list[ArchTargetRecord]:
    """Synthesize composite ``AMOUNT`` targets by summing declared components.

    For each ``output -> (components...)`` in ``component_sum_map``, group the
    ``AMOUNT`` records whose variable is a component by their shared cell key.
    A composite is emitted only when **all** declared components are present at
    that cell and an output record does not already exist there. If the same
    component variable appears twice at a cell the group is dropped (never
    double-counted). Faithful port of the legacy ``_component_sum_records``.

    Returns only the newly synthesized composite records (not the inputs); see
    :func:`with_component_sum_records` to append them.
    """
    # Cells where the output variable already exists -> do not re-synthesize.
    existing_keys = {
        _component_sum_record_key(
            record,
            output_variable=record.variable,
            geo_level=geo_level,
            normalize_source=normalize_source,
        )
        for record in records
        if record.target_type == "AMOUNT"
    }

    grouped: dict[tuple[Any, ...], dict[str, ArchTargetRecord]] = {}
    for record in records:
        if record.target_type != "AMOUNT":
            continue
        for output_variable, component_variables in component_sum_map.items():
            if record.variable not in component_variables:
                continue
            key = _component_sum_record_key(
                record,
                output_variable=output_variable,
                geo_level=geo_level,
                normalize_source=normalize_source,
            )
            if key in existing_keys:
                continue
            components = grouped.setdefault(key, {})
            if record.variable in components:
                # Duplicate component at this cell: ambiguous, drop the group.
                components.clear()
                break
            components[record.variable] = record

    composites: list[ArchTargetRecord] = []
    for key, components_by_variable in grouped.items():
        output_variable = str(key[0])
        component_variables = component_sum_map[output_variable]
        if set(components_by_variable) != set(component_variables):
            # Incomplete component set: cannot form the composite.
            continue
        ordered = [
            components_by_variable[component_variable]
            for component_variable in component_variables
        ]
        composites.append(_component_records_to_sum_record(key, ordered))
    return composites


def with_component_sum_records(
    records: Sequence[ArchTargetRecord],
    *,
    component_sum_map: ComponentSumMap,
    geo_level: GeoLevelFn = default_geo_level,
    normalize_source: NormalizeSourceFn = default_normalize_source,
) -> list[ArchTargetRecord]:
    """Return ``records`` plus any synthesized component-sum composites."""
    composites = component_sum_records(
        records,
        component_sum_map=component_sum_map,
        geo_level=geo_level,
        normalize_source=normalize_source,
    )
    if not composites:
        return list(records)
    return [*records, *composites]


def _component_records_to_sum_record(
    key: tuple[Any, ...],
    records: list[ArchTargetRecord],
) -> ArchTargetRecord:
    """Build one composite record summing ``records`` (faithful port)."""
    first = records[0]
    digest = sha1(repr(key).encode("utf-8")).hexdigest()
    component_labels = ", ".join(record.variable for record in records)
    source_tables = tuple(
        dict.fromkeys(r.source_table for r in records if r.source_table)
    )
    source_urls = tuple(dict.fromkeys(r.source_url for r in records if r.source_url))
    source_row_keys = tuple(
        dict.fromkeys(
            source_row_key
            for record in records
            for source_row_key in (
                record.source_row_keys
                or (str(record.source_target_id or record.target_id),)
            )
        )
    )
    source_cell_keys = tuple(
        dict.fromkeys(
            source_cell_key
            for record in records
            for source_cell_key in record.source_cell_keys
        )
    )
    notes = (
        "Microplex component sum matching PolicyEngine salt sources: "
        f"{component_labels}."
    )
    return replace(
        first,
        target_id=-int(digest[:12], 16),
        stratum_id=-int(digest[12:20], 16),
        variable=str(key[0]),
        value=sum(record.value for record in records),
        source_table=(
            source_tables[0]
            if len(source_tables) == 1
            else "Microplex component sum from Arch source tables"
        ),
        source_url=source_urls[0] if len(source_urls) == 1 else None,
        notes=f"{first.notes} {notes}" if first.notes else notes,
        source_record_id=f"microplex_component_sum:{digest[:16]}",
        source_cell_keys=source_cell_keys,
        source_row_keys=source_row_keys,
        aggregate_fact_key=None,
        semantic_fact_key=None,
        source_target_id=None,
        source_stratum_id=None,
        concept=None,
        source_concept=None,
        concept_relation="sum_of_components",
        concept_authority="policyengine_us",
        concept_evidence_notes=notes,
    )
