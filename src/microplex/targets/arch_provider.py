"""Convert derived Arch target records into the calibration target surface.

The Arch derivation pipeline (:mod:`microplex.targets.arch_derivations`) works
over :class:`~microplex.targets.arch_derivations.ArchTargetRecord` — the
middle representation rich enough to preserve Arch lineage. At the **provider
boundary**, after all derivations and skip filters, those records convert to
the calibration-facing :class:`~microplex.targets.spec.TargetSpec` /
``TargetSet`` (codex's iter280 decision).

This module is that boundary adapter: ``ArchTargetRecord`` -> ``TargetSpec``.
The PE entity for each variable is injected (``entity_of``); aggregation follows
the target type (``COUNT`` -> count with no measure, otherwise sum over the
variable); constraints become target filters; and Arch lineage is preserved in
``TargetSpec.metadata`` so the calibration surface stays auditable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from microplex.core import EntityType
from microplex.targets.arch_derivations import ArchTargetRecord
from microplex.targets.spec import (
    TargetAggregation,
    TargetFilter,
    TargetSet,
    TargetSpec,
)

__all__ = [
    "default_arch_target_name",
    "arch_target_record_to_target_spec",
    "arch_records_to_target_set",
]

EntityOfFn = Callable[[str], EntityType | str]
NameOfFn = Callable[[ArchTargetRecord], str]
MeasureOfFn = Callable[[str], str | None]
SkipFn = Callable[[ArchTargetRecord], bool]


def default_arch_target_name(record: ArchTargetRecord) -> str:
    """A deterministic, cell-unique target name from a record.

    Combines source, variable, geography, and non-empty constraints so distinct
    target cells get distinct names. Packs can inject their own ``name_of``.
    """
    parts: list[str] = [record.source or "arch", record.variable]
    if record.geography_id:
        parts.append(str(record.geography_id))
    elif record.geographic_level:
        parts.append(str(record.geographic_level))
    for variable, operator, value in record.constraints:
        parts.append(f"{variable}{operator}{value}")
    return "/".join(parts)


def arch_target_record_to_target_spec(
    record: ArchTargetRecord,
    *,
    entity: EntityType | str,
    measure: str | None = None,
    name: str | None = None,
    name_of: NameOfFn = default_arch_target_name,
) -> TargetSpec:
    """Convert one derived ``ArchTargetRecord`` into a canonical ``TargetSpec``.

    - ``COUNT`` records become count targets (no measure); all others sum over
      ``measure`` (defaulting to the record's variable).
    - Record constraints become :class:`TargetFilter`s.
    - Arch lineage (ids, concept, source table/url, geography) is kept in
      ``metadata`` so the calibration surface remains auditable.
    """
    is_count = record.target_type == "COUNT"
    aggregation = TargetAggregation.COUNT if is_count else TargetAggregation.SUM
    resolved_measure = (
        None if is_count else (measure if measure is not None else record.variable)
    )
    filters = tuple(
        TargetFilter(feature=variable, operator=operator, value=value)
        for variable, operator, value in record.constraints
    )
    lineage = {
        "arch_variable": record.variable,
        "target_type": record.target_type,
        "geo_level": record.geographic_level,
        "geography_id": record.geography_id,
        "jurisdiction": record.jurisdiction,
        "target_id": record.target_id,
        "stratum_id": record.stratum_id,
        "stratum_name": record.stratum_name,
        "source_table": record.source_table,
        "source_url": record.source_url,
        "source_record_id": record.source_record_id,
        "source_period": record.source_period,
        "concept": record.concept,
        "source_concept": record.source_concept,
        "concept_relation": record.concept_relation,
        "concept_authority": record.concept_authority,
        "aging_factors": record.aging_factors,
    }
    metadata = {key: value for key, value in lineage.items() if value is not None}
    return TargetSpec(
        name=name if name is not None else name_of(record),
        entity=entity,
        value=float(record.value),
        period=record.period,
        measure=resolved_measure,
        aggregation=aggregation,
        filters=filters,
        source=record.source or None,
        units=record.unit,
        description=record.notes,
        metadata=metadata,
    )


def arch_records_to_target_set(
    records: Sequence[ArchTargetRecord],
    *,
    entity_of: EntityOfFn,
    skip: SkipFn | None = None,
    measure_of: MeasureOfFn | None = None,
    name_of: NameOfFn = default_arch_target_name,
) -> TargetSet:
    """Convert derived Arch records into a ``TargetSet`` (the calibration surface).

    ``entity_of`` maps a variable to its PE entity. ``skip`` (e.g.
    :func:`~microplex.targets.arch_derivations.should_skip_target_record`) drops
    records before conversion. ``measure_of`` overrides the measure column per
    variable when it differs from the variable name.
    """
    specs: list[TargetSpec] = []
    for record in records:
        if skip is not None and skip(record):
            continue
        measure = measure_of(record.variable) if measure_of is not None else None
        specs.append(
            arch_target_record_to_target_spec(
                record,
                entity=entity_of(record.variable),
                measure=measure,
                name_of=name_of,
            )
        )
    return TargetSet(specs)
