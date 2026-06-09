"""Convert derived Arch target records into the calibration target surface.

The Arch derivation pipeline (:mod:`microplex.targets.arch_derivations`) works
over :class:`~microplex.targets.arch_derivations.ArchTargetRecord` — the
middle representation rich enough to preserve Arch lineage. At the **provider
boundary**, after all derivations and skip filters, those records convert to
the calibration-facing :class:`~microplex.targets.spec.TargetSpec` /
``TargetSet`` (codex's iter280 decision).

This module is that boundary adapter: ``ArchTargetRecord`` -> ``TargetSpec``.
The PE entity for each variable is injected (``entity_of``). Every target is a
**SUM over a measure with filters** — there is no separate count aggregation: an
``AMOUNT`` record sums its variable, and a ``COUNT`` record sums an entity-count
measure (e.g. ``tax_unit_count``, 1 per record at that level), so a count is a
sum of ones. Record constraints become target filters, and geography is added as
an explicit filter (``state_fips == ...``) so subnational targets are scoped
rather than left national. Other target types (e.g. ``RATE``) are out of scope
and raise. Arch lineage is preserved in ``TargetSpec.metadata`` so the
calibration surface stays auditable.
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
    "default_geo_feature",
    "default_count_measure",
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
GeoFeatureFn = Callable[[str | None], str | None]
CountMeasureFn = Callable[[EntityType], str]

#: geography level -> the microdata feature that identifies it.
DEFAULT_GEO_FEATURES = {
    "state": "state_fips",
    "county": "county_fips",
    "district": "congressional_district",
    "congressional_district": "congressional_district",
    "tract": "tract_geoid",
    "block": "block_geoid",
}
NATIONAL_GEO_LEVELS = {"", "national", "nation"}


def _normalized_geo_level(record: ArchTargetRecord) -> str:
    return (record.geographic_level or "").lower()


def default_geo_feature(geo_level: str | None) -> str | None:
    """The microdata feature identifying a geography level (``None`` = national).

    e.g. ``state`` -> ``state_fips``. Packs can inject their own mapping.
    """
    if geo_level is None:
        return None
    return DEFAULT_GEO_FEATURES.get(geo_level.lower())


def default_count_measure(entity: EntityType) -> str:
    """The entity-count measure that is 1 per record at the entity level.

    A count target is a **sum** of this (e.g. ``tax_unit_count``), so counts and
    sums share one aggregation path. Packs can inject their own mapping.
    """
    return f"{entity.value}_count"


def _is_national_geo_level(geo_level: str | None) -> bool:
    return (geo_level or "").lower() in NATIONAL_GEO_LEVELS


def _geo_filter_matches(filter_: TargetFilter, geography_id: str) -> bool:
    return str(filter_.operator) == "==" and str(filter_.value) == geography_id


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
    geo_feature: GeoFeatureFn = default_geo_feature,
    count_measure: CountMeasureFn = default_count_measure,
) -> TargetSpec:
    """Convert one derived ``ArchTargetRecord`` into a canonical ``TargetSpec``.

    Every target is a **SUM over a measure with filters** — there is no separate
    count aggregation. An ``AMOUNT`` record sums its variable; a ``COUNT`` record
    sums an **entity-count measure** (``count_measure(entity)``, 1 per record at
    that level), so a count is a sum of ones and conditional counts fall out of
    the filters. Any other target type (e.g. ``RATE``) is out of scope and raises.

    Record constraints become :class:`TargetFilter`s, and the record's geography
    is added as an explicit filter (``geo_feature(level) == geography_id``) so a
    state/county/district target is scoped rather than left national — unless
    that feature is already constrained. Arch lineage is kept in ``metadata``.
    """
    entity_type = entity if isinstance(entity, EntityType) else EntityType(entity)
    if record.target_type == "AMOUNT":
        resolved_measure = measure if measure is not None else record.variable
    elif record.target_type == "COUNT":
        if measure is not None:
            raise ValueError(
                "COUNT Arch targets must use count_measure(entity), not a "
                "measure override. Use the count_measure hook to override "
                "entity-count measure names."
            )
        resolved_measure = count_measure(entity_type)
    else:
        raise ValueError(
            f"unsupported Arch target_type {record.target_type!r} for "
            f"{record.variable!r}: only AMOUNT and COUNT are supported "
            "(counts are summed entity-count measures; RATE is out of scope)"
        )

    filters = [
        TargetFilter(feature=variable, operator=operator, value=value)
        for variable, operator, value in record.constraints
    ]
    constrained_features = {variable for variable, _, _ in record.constraints}
    feature = geo_feature(record.geographic_level)
    if not _is_national_geo_level(record.geographic_level):
        if feature is None:
            raise ValueError(
                f"unsupported Arch geography level {record.geographic_level!r} "
                f"for {record.variable!r}; inject a geo_feature mapping before "
                "converting subnational targets"
            )
        if feature in constrained_features:
            # Geography is already carried as a constraint — the dominant Arch
            # pattern (e.g. a STATE record with `state_fips == "06"` in its
            # stratum constraints and no `geography_id`). It is already fully
            # scoped, so do not require `geography_id` and do not duplicate the
            # filter. If a redundant `geography_id` is also present, verify it
            # agrees with the constraint rather than silently diverging.
            if record.geography_id is not None:
                geography_id = str(record.geography_id)
                existing_geo_filters = [
                    filter_ for filter_ in filters if filter_.feature == feature
                ]
                if not any(
                    _geo_filter_matches(filter_, geography_id)
                    for filter_ in existing_geo_filters
                ):
                    raise ValueError(
                        f"conflicting Arch geography for {record.variable!r}: "
                        f"metadata {feature} == {geography_id!r} but constraints "
                        f"include {existing_geo_filters!r}"
                    )
        elif record.geography_id is None:
            # Subnational level but geography is in neither geography_id nor a
            # constraint — genuinely unscoped, fail closed.
            raise ValueError(
                f"Arch geography level {record.geographic_level!r} for "
                f"{record.variable!r} requires geography_id or a "
                f"{feature!r} constraint"
            )
        else:
            filters.append(
                TargetFilter(
                    feature=feature, operator="==", value=str(record.geography_id)
                )
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
        entity=entity_type,
        value=float(record.value),
        period=record.period,
        measure=resolved_measure,
        aggregation=TargetAggregation.SUM,
        filters=tuple(filters),
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
    geo_feature: GeoFeatureFn = default_geo_feature,
    count_measure: CountMeasureFn = default_count_measure,
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
        measure = (
            measure_of(record.variable)
            if measure_of is not None and record.target_type != "COUNT"
            else None
        )
        specs.append(
            arch_target_record_to_target_spec(
                record,
                entity=entity_of(record.variable),
                measure=measure,
                name_of=name_of,
                geo_feature=geo_feature,
                count_measure=count_measure,
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
