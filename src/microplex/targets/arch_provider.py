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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from microplex.core import EntityType
from microplex.targets.arch_derivations import (
    ArchTargetRecord,
    age_soi_records,
    bea_national_wages_record,
    default_normalize_source,
    latest_carry_forward,
    ssa_carry_forward_rank,
    state_to_national_rollup,
    with_bea_employment_income_before_lsr,
    with_component_sum_records,
)
from microplex.targets.provider import TargetQuery, apply_target_query
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
    "arch_record_composition_key",
    "latest_soi_records_by_composition",
    "ArchPipelineConfig",
    "run_arch_derivation_pipeline",
    "ArchTargetProvider",
]

EntityOfFn = Callable[[str], EntityType | str]
NameOfFn = Callable[[ArchTargetRecord], str]
MeasureOfFn = Callable[[str], str | None]
SkipFn = Callable[[ArchTargetRecord], bool]


def _normalized_geo_level(record: ArchTargetRecord) -> str:
    return (record.geographic_level or "").lower()


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


def arch_record_composition_key(
    record: ArchTargetRecord,
    *,
    geo_level: Callable[[ArchTargetRecord], str] = _normalized_geo_level,
) -> tuple[str, str, str, tuple[tuple[str, str, str], ...]]:
    """Composition identity: (variable, target type, geo level, constraints)."""
    return (
        record.variable,
        record.target_type,
        geo_level(record),
        tuple(sorted(record.constraints)),
    )


def latest_soi_records_by_composition(
    records: Sequence[ArchTargetRecord],
    *,
    target_year: int,
    geo_level: Callable[[ArchTargetRecord], str] = _normalized_geo_level,
) -> list[ArchTargetRecord]:
    """Per composition cell, keep the record from the latest period not exceeding
    ``target_year`` (faithful port of the legacy SOI composition dedup)."""
    candidates = [record for record in records if record.period <= target_year]
    latest_period: dict[tuple[str, str, str, tuple], int] = {}
    for record in candidates:
        key = arch_record_composition_key(record, geo_level=geo_level)
        latest_period[key] = max(latest_period.get(key, record.period), record.period)
    return [
        record
        for record in candidates
        if record.period
        == latest_period[arch_record_composition_key(record, geo_level=geo_level)]
    ]


@dataclass(frozen=True)
class ArchPipelineConfig:
    """Declarative config for the Arch derivation pipeline (the US pack supplies
    it). Steps run only when their config is present."""

    target_year: int
    component_sum_map: Mapping[str, Sequence[str]] = field(default_factory=dict)
    rollup_required_states: tuple[str, ...] = ()
    rollup_group_key: Callable[[ArchTargetRecord], Any] | None = None
    bea_output_variable: str | None = None
    bea_required_states: tuple[str, ...] = ()
    bea_wage_component_variables: Mapping[str, str] = field(default_factory=dict)
    carry_forward_is_candidate: Callable[[ArchTargetRecord], bool] | None = None
    carry_forward_cell_key: Callable[[ArchTargetRecord], Any] | None = None
    carry_forward_rank: Callable[[ArchTargetRecord], tuple[Any, ...]] = (
        ssa_carry_forward_rank
    )
    age_soi: bool = True
    soi_source: str = "IRS_SOI"
    skip: SkipFn | None = None
    state_fips_of: Callable[[ArchTargetRecord], str | None] = (
        lambda record: record.geography_id
    )
    geo_level: Callable[[ArchTargetRecord], str] = _normalized_geo_level
    normalize_source: Callable[[str], str] = default_normalize_source


def _carry_forward_to_year(record: ArchTargetRecord, year: int) -> ArchTargetRecord:
    return replace(record, period=year, source_period=record.period)


def run_arch_derivation_pipeline(
    records: Sequence[ArchTargetRecord],
    *,
    config: ArchPipelineConfig,
    reference_records: Sequence[ArchTargetRecord] | None = None,
) -> list[ArchTargetRecord]:
    """Compose the Arch derivations in the legacy order into the final records.

    BEA augment → (non-SOI current + latest carry-forward + latest/aged SOI) →
    component sum → state→national rollup → skip filter. Each step runs only when
    its config is present. ``reference_records`` (defaults to ``records``)
    supplies the SOI aging reference series.
    """
    refs = reference_records if reference_records is not None else records
    norm = config.normalize_source
    soi = norm(config.soi_source)
    target_year = config.target_year

    current = [record for record in records if record.period == target_year]
    if config.bea_output_variable and config.bea_wage_component_variables:
        national_wages = bea_national_wages_record(
            current,
            output_variable=config.bea_output_variable,
            normalize_source=norm,
            geo_level=config.geo_level,
        )
        if national_wages is not None:
            current = with_bea_employment_income_before_lsr(
                current,
                national_wages=national_wages,
                required_states=config.bea_required_states,
                wage_component_variables=config.bea_wage_component_variables,
                output_variable=config.bea_output_variable,
                state_fips_of=config.state_fips_of,
                geo_level=config.geo_level,
                normalize_source=norm,
            )

    is_candidate = config.carry_forward_is_candidate
    non_soi_current = [
        record
        for record in current
        if norm(record.source) != soi
        and not (is_candidate(record) if is_candidate is not None else False)
    ]

    carry_forward: list[ArchTargetRecord] = []
    if config.carry_forward_cell_key is not None and is_candidate is not None:
        carry_forward = latest_carry_forward(
            records,
            target_year=target_year,
            is_candidate=is_candidate,
            cell_key=config.carry_forward_cell_key,
            rank=config.carry_forward_rank,
            carry_forward=_carry_forward_to_year,
        )

    soi_records = latest_soi_records_by_composition(
        [record for record in records if norm(record.source) == soi],
        target_year=target_year,
        geo_level=config.geo_level,
    )
    if config.age_soi and soi_records:
        soi_records = age_soi_records(
            soi_records, target_year=target_year, reference_records=refs
        )

    combined = [*non_soi_current, *carry_forward, *soi_records]
    if config.component_sum_map:
        combined = with_component_sum_records(
            combined,
            component_sum_map=config.component_sum_map,
            geo_level=config.geo_level,
            normalize_source=norm,
        )
    if config.rollup_group_key is not None and config.rollup_required_states:
        combined = combined + state_to_national_rollup(
            combined,
            required_states=config.rollup_required_states,
            group_key=config.rollup_group_key,
            state_fips_of=config.state_fips_of,
            geo_level=config.geo_level,
        )
    if config.skip is not None:
        combined = [record for record in combined if not config.skip(record)]
    return combined


@dataclass
class ArchTargetProvider:
    """A ``TargetProvider`` that derives the calibration surface from Arch records.

    Runs :func:`run_arch_derivation_pipeline` over pre-loaded ``records`` and
    converts the result to a ``TargetSet``. The DB/JSONL → ``ArchTargetRecord``
    load adapter is supplied upstream (it needs the real Arch artifact); this
    provider consumes the already-loaded records.
    """

    records: Sequence[ArchTargetRecord]
    config: ArchPipelineConfig
    entity_of: EntityOfFn
    reference_records: Sequence[ArchTargetRecord] | None = None
    convert_skip: SkipFn | None = None
    measure_of: MeasureOfFn | None = None
    name_of: NameOfFn = default_arch_target_name

    def load_target_set(self, query: TargetQuery | None = None) -> TargetSet:
        derived = run_arch_derivation_pipeline(
            self.records,
            config=self.config,
            reference_records=self.reference_records,
        )
        target_set = arch_records_to_target_set(
            derived,
            entity_of=self.entity_of,
            skip=self.convert_skip,
            measure_of=self.measure_of,
            name_of=self.name_of,
        )
        return apply_target_query(target_set, query)
