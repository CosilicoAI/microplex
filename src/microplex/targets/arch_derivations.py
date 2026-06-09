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
    "latest_carry_forward",
    "ssa_carry_forward_rank",
    "is_ssa_carry_forward_candidate",
    "state_to_national_rollup",
    "sum_state_records_to_national",
    "SOIAgingFactors",
    "age_soi_records",
    "soi_aging_factors",
    "soi_count_aging_factor",
    "soi_amount_aging_factor",
    "reference_total",
    "soi_total_for_year",
    "default_total_scope",
    "bea_state_employment_income_before_lsr",
    "with_bea_employment_income_before_lsr",
    "bea_national_wages_record",
    "should_skip_target_record",
    "should_skip_fact_concept",
    "is_blocked_self_employment_binding",
    "is_bea_regional_country_record",
    "default_bea_regional_lineage",
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
    stratum_name: str | None = None
    concept_evidence_url: str | None = None
    legal_vintage: str | None = None


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


RankFn = Callable[[ArchTargetRecord], tuple[Any, ...]]
CandidateFn = Callable[[ArchTargetRecord], bool]
CellKeyFn = Callable[[ArchTargetRecord], Any]
CarryForwardFn = Callable[[ArchTargetRecord, int], ArchTargetRecord]
SortKeyFn = Callable[[ArchTargetRecord], Any]


def latest_carry_forward(
    records: Sequence[ArchTargetRecord],
    *,
    target_year: int,
    is_candidate: CandidateFn,
    cell_key: CellKeyFn,
    rank: RankFn,
    carry_forward: CarryForwardFn,
    sort_key: SortKeyFn | None = None,
) -> list[ArchTargetRecord]:
    """Keep the highest-ranked candidate per cell, carrying stale cells forward.

    Generic port of the legacy SSA latest-carry-forward: a source publishes the
    same target cell across periods with a lag, so within each cell we keep the
    single highest-``rank`` candidate (period not in the future), then remap any
    kept record whose period predates ``target_year`` via ``carry_forward``.

    Representation-specific pieces are injected: ``is_candidate`` (eligibility,
    e.g. SSA + carry-forward variables), ``cell_key`` (target-cell identity,
    which depends on the canonical target representation; return ``None`` to
    skip a record), ``rank`` (preference within a cell — higher wins; see
    :func:`ssa_carry_forward_rank`), ``carry_forward`` (remap a stale record to
    ``target_year``), and an optional ``sort_key`` for deterministic output.
    """
    latest: dict[Any, tuple[tuple[Any, ...], ArchTargetRecord]] = {}
    for record in records:
        if record.period > target_year:
            continue
        if not is_candidate(record):
            continue
        key = cell_key(record)
        if key is None:
            continue
        record_rank = rank(record)
        current = latest.get(key)
        if current is None or record_rank > current[0]:
            latest[key] = (record_rank, record)
    kept = [record for _, record in latest.values()]
    if sort_key is not None:
        kept.sort(key=sort_key)
    return [
        record if record.period == target_year else carry_forward(record, target_year)
        for record in kept
    ]


def ssa_carry_forward_rank(record: ArchTargetRecord) -> tuple[Any, ...]:
    """Default SSA carry-forward preference rank (higher wins).

    Faithful port: prefer the latest period, then an "annual statistical report"
    source table, then any source table, then the ``ssi_total_payments``
    variable, then the larger target id (stable tiebreak).
    """
    source_table = str(record.source_table or "").lower()
    return (
        int(record.period),
        "annual statistical report" in source_table,
        bool(record.source_table),
        record.variable == "ssi_total_payments",
        int(record.target_id),
    )


def is_ssa_carry_forward_candidate(
    record: ArchTargetRecord,
    *,
    variables: Sequence[str],
    source: str = "SSA",
    normalize_source: NormalizeSourceFn = default_normalize_source,
) -> bool:
    """Default SSA carry-forward eligibility.

    SSA source, a declared carry-forward ``variables`` member, and an
    AMOUNT/COUNT target. ``variables`` is the injected US set.
    """
    return (
        normalize_source(record.source) == source
        and record.variable in set(variables)
        and record.target_type in {"AMOUNT", "COUNT"}
    )


GroupKeyFn = Callable[[ArchTargetRecord], Any]
StateFipsFn = Callable[[ArchTargetRecord], str | None]
NationalBuilderFn = Callable[[Any, "list[ArchTargetRecord]"], ArchTargetRecord]


def _normalized_geo_level(record: ArchTargetRecord) -> str:
    return (record.geographic_level or "").lower()


def state_to_national_rollup(
    records: Sequence[ArchTargetRecord],
    *,
    required_states: Sequence[str],
    group_key: GroupKeyFn,
    state_fips_of: StateFipsFn = lambda record: record.geography_id,
    geo_level: Callable[[ArchTargetRecord], str] = _normalized_geo_level,
    build_national: NationalBuilderFn | None = None,
) -> list[ArchTargetRecord]:
    """Sum complete sets of state records into national totals.

    Generic port of the legacy state→national rollup: group ``state``-level
    records by ``group_key`` and, for each group that covers **every** state in
    ``required_states`` exactly once, emit one national record (default builder
    :func:`sum_state_records_to_national`). Groups missing a state, carrying a
    duplicate state, or whose national total already exists are skipped. The US
    pack injects ``required_states`` (the 51-state set excluding PR fips ``72``),
    ``group_key`` (rollup-variable filter + non-state cell fields), and the
    state-fips / geo-level extractors. Returns only the new national records.
    """
    builder = build_national or sum_state_records_to_national
    required = frozenset(required_states)

    existing_national_keys = {
        key
        for record in records
        if geo_level(record) == "national"
        for key in (group_key(record),)
        if key is not None
    }

    grouped: dict[Any, list[tuple[str, ArchTargetRecord]]] = {}
    for record in records:
        if geo_level(record) != "state":
            continue
        key = group_key(record)
        if key is None or key in existing_national_keys:
            continue
        state_fips = state_fips_of(record)
        if state_fips is None or state_fips not in required:
            continue
        grouped.setdefault(key, []).append((state_fips, record))

    rollups: list[ArchTargetRecord] = []
    for key, state_records in grouped.items():
        records_by_state: dict[str, ArchTargetRecord] = {}
        duplicate = False
        for state_fips, record in state_records:
            if state_fips in records_by_state:
                duplicate = True
                break
            records_by_state[state_fips] = record
        if duplicate or set(records_by_state) != required:
            continue
        ordered = [records_by_state[fips] for fips in sorted(required)]
        rollups.append(builder(key, ordered))
    return rollups


def sum_state_records_to_national(
    key: Any,
    records: list[ArchTargetRecord],
    *,
    non_state_constraints: Callable[
        [tuple[tuple[str, str, str], ...]], tuple[tuple[str, str, str], ...]
    ] = lambda constraints: constraints,
) -> ArchTargetRecord:
    """Default national-rollup builder: sum the state records (faithful port).

    Sums values, nulls the geography, tags a deterministic national id, and
    merges source lineage. ``non_state_constraints`` (injected by the US pack)
    strips state-level constraints from the national record; the generic default
    keeps them.
    """
    first = records[0]
    digest = sha1(repr(key).encode("utf-8")).hexdigest()
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
    notes = f"Microplex national rollup from {len(records)} state targets."
    if first.notes:
        notes = f"{first.notes} {notes}"
    return replace(
        first,
        target_id=-int(digest[:12], 16),
        stratum_id=-int(digest[12:20], 16),
        value=sum(record.value for record in records),
        geographic_level=None,
        geography_id=None,
        constraints=non_state_constraints(first.constraints),
        notes=notes,
        source_record_id=f"microplex_state_rollup:{digest[:16]}",
        source_cell_keys=source_cell_keys,
        source_row_keys=source_row_keys,
        source_target_id=None,
        source_stratum_id=None,
    )


@dataclass(frozen=True)
class SOIAgingFactors:
    """Factors (and method labels) used to age SOI records to a model year."""

    source_year: int
    target_year: int
    count_factor: float
    amount_factor: float
    count_method: str
    amount_method: str


#: US/eCPS default SOI-aging reference series. Aging is *source-backed*: factors
#: are ratios of these reference totals across years, not hardcoded growth rates.
SOI_SOURCE = "IRS_SOI"
COUNT_SOI_FALLBACK_VARIABLE = "tax_unit_count"
AMOUNT_SOI_VARIABLE = "adjusted_gross_income"
#: target-year labor-force lookup order: (source, variable, method-label).
LABOR_FORCE_REFERENCES = (
    ("BLS", "labor_force_count", "bls_labor_force_ratio"),
    ("CBO", "labor_force", "cbo_labor_force_ratio"),
)

TotalScopeFn = Callable[[ArchTargetRecord], bool]


def default_total_scope(record: ArchTargetRecord) -> bool:
    """Whether a record is a jurisdiction-wide TOTAL (usable as a denominator).

    True for unconstrained records, filer-only constraints, or an "all filers"
    stratum. The US pack can inject its own predicate.
    """
    if not record.constraints:
        return True
    if tuple(record.constraints) in {
        (("is_tax_filer", "==", "1"),),
        (("tax_unit_is_filer", "==", "1"),),
    }:
        return True
    return (
        str(getattr(record, "stratum_name", None) or "").lower().endswith("all filers")
    )


def reference_total(
    records: Sequence[ArchTargetRecord],
    *,
    year: int,
    source: str,
    variable: str,
    normalize_source: NormalizeSourceFn = default_normalize_source,
    total_scope: TotalScopeFn | None = None,
) -> float | None:
    """The first matching reference total (period/source/variable), or ``None``.

    When ``total_scope`` is given, only records it accepts are considered.
    """
    normalized = normalize_source(source)
    matches = [
        record
        for record in records
        if record.period == year
        and normalize_source(record.source) == normalized
        and record.variable == variable
    ]
    if total_scope is not None:
        matches = [record for record in matches if total_scope(record)]
    return float(matches[0].value) if matches else None


def soi_total_for_year(
    records: Sequence[ArchTargetRecord],
    *,
    target_year: int,
    variable: str,
    exact_method: str,
    extrapolation_method: str,
    total_scope: TotalScopeFn = default_total_scope,
    normalize_source: NormalizeSourceFn = default_normalize_source,
) -> tuple[float | None, str]:
    """An SOI total at ``target_year``: exact if present, else last-growth
    extrapolation from the two latest available years, else ``None``."""
    exact = reference_total(
        records,
        year=target_year,
        source=SOI_SOURCE,
        variable=variable,
        normalize_source=normalize_source,
        total_scope=total_scope,
    )
    if exact is not None:
        return exact, exact_method
    available: dict[int, float] = {}
    for year in sorted({record.period for record in records}):
        if year > target_year:
            continue
        value = reference_total(
            records,
            year=year,
            source=SOI_SOURCE,
            variable=variable,
            normalize_source=normalize_source,
            total_scope=total_scope,
        )
        if value is not None:
            available[year] = value
    if len(available) < 2:
        return None, f"source_fact_carry_forward_no_{variable}_reference"
    latest_year = max(available)
    previous_year = max(year for year in available if year < latest_year)
    annual_growth = available[latest_year] / available[previous_year]
    years_forward = target_year - latest_year
    return available[latest_year] * annual_growth**years_forward, extrapolation_method


def _labor_force_for_year(
    records: Sequence[ArchTargetRecord],
    *,
    year: int,
    normalize_source: NormalizeSourceFn = default_normalize_source,
) -> tuple[float | None, str]:
    for source, variable, method in LABOR_FORCE_REFERENCES:
        value = reference_total(
            records,
            year=year,
            source=source,
            variable=variable,
            normalize_source=normalize_source,
        )
        if value is not None:
            return value, method
    return None, "source_fact_carry_forward_no_labor_force_reference"


def soi_count_aging_factor(
    records: Sequence[ArchTargetRecord],
    *,
    source_year: int,
    target_year: int,
    normalize_source: NormalizeSourceFn = default_normalize_source,
    total_scope: TotalScopeFn = default_total_scope,
) -> tuple[float, str]:
    """COUNT aging factor: labor-force ratio (BLS source-year vs BLS/CBO
    target-year), else SOI return-count ratio, else 1.0. Faithful port."""
    source_labor_force = reference_total(
        records,
        year=source_year,
        source="BLS",
        variable="labor_force_count",
        normalize_source=normalize_source,
    )
    target_labor_force, labor_force_method = _labor_force_for_year(
        records, year=target_year, normalize_source=normalize_source
    )
    if source_labor_force is not None and target_labor_force is not None:
        return target_labor_force / source_labor_force, labor_force_method

    source_count = reference_total(
        records,
        year=source_year,
        source=SOI_SOURCE,
        variable=COUNT_SOI_FALLBACK_VARIABLE,
        normalize_source=normalize_source,
        total_scope=total_scope,
    )
    target_count, count_method = soi_total_for_year(
        records,
        target_year=target_year,
        variable=COUNT_SOI_FALLBACK_VARIABLE,
        exact_method="soi_total_return_count_ratio",
        extrapolation_method="soi_total_return_count_last_growth_extrapolation",
        total_scope=total_scope,
        normalize_source=normalize_source,
    )
    if source_count is not None and target_count is not None:
        return target_count / source_count, count_method
    return 1.0, "source_fact_carry_forward_no_count_reference"


def soi_amount_aging_factor(
    records: Sequence[ArchTargetRecord],
    *,
    source_year: int,
    target_year: int,
    normalize_source: NormalizeSourceFn = default_normalize_source,
    total_scope: TotalScopeFn = default_total_scope,
) -> tuple[float, str]:
    """AMOUNT aging factor: SOI AGI ratio (exact or last-growth extrapolation),
    else 1.0. Faithful port."""
    source_agi = reference_total(
        records,
        year=source_year,
        source=SOI_SOURCE,
        variable=AMOUNT_SOI_VARIABLE,
        normalize_source=normalize_source,
        total_scope=total_scope,
    )
    target_agi, amount_method = soi_total_for_year(
        records,
        target_year=target_year,
        variable=AMOUNT_SOI_VARIABLE,
        exact_method="soi_total_agi_ratio",
        extrapolation_method="soi_total_agi_last_growth_extrapolation",
        total_scope=total_scope,
        normalize_source=normalize_source,
    )
    if source_agi is not None and target_agi is not None:
        return target_agi / source_agi, amount_method
    return 1.0, "source_fact_carry_forward_no_amount_reference"


CountFactorFn = Callable[..., tuple[float, str]]


def soi_aging_factors(
    reference_records: Sequence[ArchTargetRecord],
    *,
    source_year: int,
    target_year: int,
    needs_count_factor: bool = True,
    needs_amount_factor: bool = True,
    count_factor: CountFactorFn = soi_count_aging_factor,
    amount_factor: CountFactorFn = soi_amount_aging_factor,
) -> SOIAgingFactors:
    """Resolve the count + amount aging factors for one source→target year."""
    if source_year == target_year:
        return SOIAgingFactors(
            source_year, target_year, 1.0, 1.0, "identity", "identity"
        )
    if needs_count_factor:
        count_value, count_method = count_factor(
            reference_records, source_year=source_year, target_year=target_year
        )
    else:
        count_value, count_method = 1.0, "not_required"
    if needs_amount_factor:
        amount_value, amount_method = amount_factor(
            reference_records, source_year=source_year, target_year=target_year
        )
    else:
        amount_value, amount_method = 1.0, "not_required"
    return SOIAgingFactors(
        source_year,
        target_year,
        count_value,
        amount_value,
        count_method,
        amount_method,
    )


AgingFactorsFn = Callable[..., SOIAgingFactors]


def age_soi_records(
    records: Sequence[ArchTargetRecord],
    *,
    target_year: int,
    reference_records: Sequence[ArchTargetRecord],
    factors_for: AgingFactorsFn = soi_aging_factors,
) -> list[ArchTargetRecord]:
    """Age SOI records from each source year to ``target_year``.

    Generic application (faithful port): group records by source year, resolve
    the count/amount factors for that year via ``factors_for`` (default
    :func:`soi_aging_factors`, computed from ``reference_records``), then scale
    each record by its target-type factor and stamp ``period``/``source_period``
    /``aging_factors``. Same-year records pass through unchanged.
    """
    aged: list[ArchTargetRecord] = []
    for source_year in sorted({record.period for record in records}):
        source_records = [r for r in records if r.period == source_year]
        if source_year == target_year:
            aged.extend(source_records)
            continue
        needs_count = any(r.target_type == "COUNT" for r in source_records)
        needs_amount = any(r.target_type == "AMOUNT" for r in source_records)
        factors = factors_for(
            reference_records,
            source_year=source_year,
            target_year=target_year,
            needs_count_factor=needs_count,
            needs_amount_factor=needs_amount,
        )
        for record in source_records:
            if record.target_type == "COUNT":
                factor = factors.count_factor
            elif record.target_type == "AMOUNT":
                factor = factors.amount_factor
            else:
                factor = 1.0
            aged.append(
                replace(
                    record,
                    value=float(record.value) * factor,
                    period=target_year,
                    source_period=record.period,
                    aging_factors=factors,
                )
            )
    return aged


#: BEA NIPA national-wages concept identifiers (US/eCPS defaults).
DEFAULT_BEA_NIPA_WAGE_CONCEPTS = ("bea_nipa.wages_and_salaries",)
DEFAULT_BEA_NIPA_SOURCE_CONCEPTS = ("bea_nipa.a034rc_wages_and_salaries",)

#: variable -> component role (wages/supplements/contributions/residence_adjustment).
BeaComponentMap = Mapping[str, str]
BeaRecordBuilderFn = Callable[..., ArchTargetRecord]


def bea_national_wages_record(
    records: Sequence[ArchTargetRecord],
    *,
    output_variable: str,
    normalize_source: NormalizeSourceFn = default_normalize_source,
    geo_level: Callable[[ArchTargetRecord], str] = _normalized_geo_level,
    wage_concepts: Sequence[str] = DEFAULT_BEA_NIPA_WAGE_CONCEPTS,
    source_concepts: Sequence[str] = DEFAULT_BEA_NIPA_SOURCE_CONCEPTS,
) -> ArchTargetRecord | None:
    """The national BEA NIPA wages_and_salaries total record, or ``None``."""
    wage_concept_set = set(wage_concepts)
    source_concept_set = set(source_concepts)
    candidates = [
        record
        for record in records
        if normalize_source(record.source) == "BEA"
        and record.variable == output_variable
        and record.target_type == "AMOUNT"
        and geo_level(record) in {"national", "country"}
        and (
            record.concept in wage_concept_set
            or record.source_concept in source_concept_set
            or "nipa" in str(record.source_record_id or "").lower()
        )
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda record: (
            record.concept in wage_concept_set,
            bool(record.source_record_id),
            int(record.target_id),
        ),
        reverse=True,
    )[0]


def bea_state_employment_income_before_lsr(
    records: Sequence[ArchTargetRecord],
    *,
    national_wages: ArchTargetRecord,
    required_states: Sequence[str],
    wage_component_variables: BeaComponentMap,
    output_variable: str,
    state_fips_of: StateFipsFn = lambda record: record.geography_id,
    geo_level: Callable[[ArchTargetRecord], str] = _normalized_geo_level,
    normalize_source: NormalizeSourceFn = default_normalize_source,
    build_record: BeaRecordBuilderFn | None = None,
) -> list[ArchTargetRecord]:
    """Synthesize residence-adjusted, nationally-reconciled state wage targets.

    BEA method (faithful port): for each state carrying the full set of wage
    components (wages, supplements, contributions, residence_adjustment),
    allocate the residence adjustment to wages by
    ``wages / (wages + supplements + contributions)``, then scale every state so
    the residence-adjusted total equals the national BEA NIPA wages total. Emits
    one ``output_variable`` AMOUNT record per state. Returns ``[]`` unless all
    ``required_states`` are present with all four component roles, and bails if
    any per-state denominator or the national total is non-positive.
    """
    builder = build_record or _default_bea_state_record
    required = frozenset(required_states)
    required_roles = set(wage_component_variables.values())

    components_by_state: dict[str, dict[str, ArchTargetRecord]] = {}
    for record in records:
        if normalize_source(record.source) != "BEA":
            continue
        role = wage_component_variables.get(record.variable)
        if role is None:
            continue
        if geo_level(record) != "state":
            continue
        state_fips = state_fips_of(record)
        if state_fips is None or state_fips not in required:
            continue
        components_by_state.setdefault(state_fips, {}).setdefault(role, record)

    if not required or set(components_by_state) != required:
        return []
    if any(set(roles) != required_roles for roles in components_by_state.values()):
        return []

    adjusted_by_state: dict[str, float] = {}
    for state_fips, components in components_by_state.items():
        wages = components["wages"].value
        supplements = components["supplements"].value
        contributions = components["contributions"].value
        residence_adjustment = components["residence_adjustment"].value
        denominator = wages + supplements + contributions
        if denominator <= 0:
            return []
        adjusted_by_state[state_fips] = (
            wages + residence_adjustment * wages / denominator
        )

    adjusted_total = sum(adjusted_by_state.values())
    if adjusted_total <= 0:
        return []
    scale_factor = national_wages.value / adjusted_total

    return [
        builder(
            state_fips=state_fips,
            state_components=components_by_state[state_fips],
            national_wages=national_wages,
            output_variable=output_variable,
            value=adjusted_by_state[state_fips] * scale_factor,
            scale_factor=scale_factor,
        )
        for state_fips in sorted(components_by_state)
    ]


def with_bea_employment_income_before_lsr(
    records: Sequence[ArchTargetRecord],
    *,
    national_wages: ArchTargetRecord,
    **kwargs: Any,
) -> list[ArchTargetRecord]:
    """Return ``records`` plus the synthesized BEA state wage records."""
    derived = bea_state_employment_income_before_lsr(
        records, national_wages=national_wages, **kwargs
    )
    if not derived:
        return list(records)
    return [*records, *derived]


def _default_bea_state_record(
    *,
    state_fips: str,
    state_components: dict[str, ArchTargetRecord],
    national_wages: ArchTargetRecord,
    output_variable: str,
    value: float,
    scale_factor: float,
    state_abbr_of: Callable[[str], str] = lambda fips: fips,
) -> ArchTargetRecord:
    """Faithful port of the synthetic BEA state-wage record builder."""
    component_records = tuple(
        state_components[role]
        for role in ("wages", "supplements", "contributions", "residence_adjustment")
    )
    first = component_records[0]
    digest = sha1(
        repr(
            (
                "bea_state_employment_income_before_lsr",
                first.period,
                state_fips,
                tuple(r.source_record_id or r.target_id for r in component_records),
                national_wages.source_record_id or national_wages.target_id,
            )
        ).encode("utf-8")
    ).hexdigest()
    source_cell_keys = tuple(
        dict.fromkeys(
            key
            for record in (*component_records, national_wages)
            for key in record.source_cell_keys
        )
    )
    source_row_keys = tuple(
        dict.fromkeys(
            key
            for record in (*component_records, national_wages)
            for key in (
                record.source_row_keys
                or (str(record.source_record_id or record.target_id),)
            )
        )
    )
    state_abbr = state_abbr_of(state_fips)
    notes = (
        "Microplex derived BEA state employment_income_before_lsr from "
        "SAINC5N line 50 wages, line 60 supplements, line 36 contributions, "
        "and line 42 residence adjustment. Residence adjustment is allocated "
        "to wages by wages / (wages + supplements + contributions), then "
        f"scaled to national BEA NIPA wages with factor {scale_factor:.12g}."
    )
    return replace(
        first,
        target_id=-int(digest[:12], 16),
        stratum_id=-int(digest[12:20], 16),
        variable=output_variable,
        value=float(value),
        target_type="AMOUNT",
        source_table="BEA Regional SAINC5N residence-adjusted state wages",
        source_url=first.source_url or national_wages.source_url,
        notes=notes,
        stratum_name=f"{state_abbr} residence-adjusted wages",
        constraints=(),
        aggregate_fact_key=f"microplex.derived.bea_state_wages.{first.period}.{state_fips}",
        semantic_fact_key=f"microplex.semantic.bea_state_wages.{first.period}.{state_fips}",
        source_record_id=f"microplex.derived.bea_state_wages.{first.period}.{state_fips}",
        source_cell_keys=source_cell_keys,
        source_row_keys=source_row_keys,
        concept="policyengine_us.employment_income_before_lsr",
        source_concept="bea_regional.sainc5n_residence_adjusted_wages_scaled_to_nipa",
        concept_relation="derived",
        concept_authority="microplex-us",
        concept_evidence_url=national_wages.concept_evidence_url
        or first.concept_evidence_url,
        concept_evidence_notes=notes,
        legal_vintage=national_wages.legal_vintage or first.legal_vintage,
        source_target_id=None,
        source_stratum_id=None,
    )


def default_bea_regional_lineage(record: ArchTargetRecord) -> bool:
    """Whether a record carries BEA regional lineage (faithful port)."""
    for value in (record.concept, record.source_concept, record.source_record_id):
        if value is None:
            continue
        text = str(value)
        if (
            text.startswith("bea_regional.")
            or text.startswith("bea-regional.")
            or ".bea-regional-" in text
        ):
            return True
    return False


def is_bea_regional_country_record(
    record: ArchTargetRecord,
    *,
    has_bea_regional_lineage: Callable[
        [ArchTargetRecord], bool
    ] = default_bea_regional_lineage,
    geo_level: Callable[[ArchTargetRecord], str] = _normalized_geo_level,
) -> bool:
    """A national/country BEA regional *component* record — an input the BEA
    derivation consumes, not an exported target."""
    if not has_bea_regional_lineage(record):
        return False
    if str(record.geography_id) == "0100000US":
        return True
    return geo_level(record) in {"national", "country"}


def should_skip_target_record(
    record: ArchTargetRecord,
    *,
    unsupported_variables: Sequence[str] = (),
    is_bea_regional_country: Callable[
        [ArchTargetRecord], bool
    ] = is_bea_regional_country_record,
) -> bool:
    """Drop a target record from the surface: an unsupported ratio/component
    variable, or a national BEA regional input. ``unsupported_variables`` is the
    injected US blocklist."""
    return record.variable in set(unsupported_variables) or is_bea_regional_country(
        record
    )


def should_skip_fact_concept(
    concept: str,
    *,
    skipped_concepts: Sequence[str],
) -> bool:
    """Whether an Arch fact concept is on the injected skip list."""
    return concept in set(skipped_concepts)


def is_blocked_self_employment_binding(
    record: ArchTargetRecord,
    model_variable: str,
    *,
    blocklist: Sequence[str],
    se_variable: str = "self_employment_income",
) -> bool:
    """Whether binding ``record`` to ``model_variable`` hits the broad
    business-income self-employment blocklist (faithful port). The record's
    identity markers (variable/concept/source ids + ``var:value`` constraints)
    are intersected with the injected ``blocklist``."""
    if model_variable != se_variable:
        return False
    markers = {
        str(value)
        for value in (
            record.variable,
            record.concept,
            record.source_concept,
            record.source_record_id,
        )
        if value is not None
    }
    markers.update(
        f"{variable}:{value}"
        for variable, _, value in record.constraints
        if value is not None
    )
    return bool(markers & set(blocklist))


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
