"""Neutral helpers for Arch target artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microplex.targets.arch_derivations import ArchTargetRecord

ARCH_CONSUMER_FACT_SCHEMA_VERSION = "arch.consumer_fact.v1"


@dataclass(frozen=True)
class ArchConsumerFact:
    """Neutral view over one Arch consumer-contract fact row."""

    row: Mapping[str, Any]
    path: str | None = None
    line_number: int | None = None

    @property
    def concept(self) -> str | None:
        """Return the canonical or observed concept for this fact."""
        return arch_consumer_fact_concept(self.row)

    @property
    def period(self) -> int:
        """Return the calendar/model year represented by this fact."""
        return arch_consumer_fact_period(self.row)

    @property
    def value(self) -> float:
        """Return the fact's numeric value."""
        return arch_consumer_fact_numeric_value(self.row.get("value"))

    @property
    def geography(self) -> Mapping[str, Any]:
        """Return the fact geography payload."""
        return mapping_value(self.row.get("geography"))

    @property
    def source(self) -> Mapping[str, Any]:
        """Return the source metadata payload."""
        return mapping_value(self.row.get("source"))

    @property
    def source_record_id(self) -> str | None:
        """Return the source record ID from lineage metadata, when present."""
        return arch_consumer_fact_source_record_id(self.row)


def load_arch_consumer_fact_jsonl_rows(
    paths: Iterable[str | Path],
    *,
    period: int | None = None,
    schema_version: str = ARCH_CONSUMER_FACT_SCHEMA_VERSION,
) -> tuple[dict[str, Any], ...]:
    """Load validated Arch consumer fact JSONL rows from one or more files."""
    rows: list[dict[str, Any]] = []
    for pathlike in paths:
        path = Path(pathlike)
        for fact in iter_arch_consumer_facts(
            path,
            period=period,
            schema_version=schema_version,
        ):
            rows.append(dict(fact.row))
    return tuple(rows)


def load_arch_consumer_facts(
    paths: Iterable[str | Path],
    *,
    period: int | None = None,
    schema_version: str = ARCH_CONSUMER_FACT_SCHEMA_VERSION,
) -> tuple[ArchConsumerFact, ...]:
    """Load validated Arch consumer facts from one or more JSONL files."""
    facts: list[ArchConsumerFact] = []
    for path in paths:
        facts.extend(
            iter_arch_consumer_facts(
                path,
                period=period,
                schema_version=schema_version,
            )
        )
    return tuple(facts)


def load_arch_target_records(
    paths: Iterable[str | Path],
    *,
    period: int | None = None,
    schema_version: str = ARCH_CONSUMER_FACT_SCHEMA_VERSION,
    variable_of: Callable[[ArchConsumerFact], str | None] | None = None,
    target_type_of: Callable[[ArchConsumerFact], str] | None = None,
    constraints_of: Callable[
        [ArchConsumerFact],
        Sequence[tuple[str, str, Any]],
    ]
    | None = None,
    geography_level_of: Callable[[ArchConsumerFact], str | None] | None = None,
    geography_id_of: Callable[[ArchConsumerFact], str | None] | None = None,
) -> tuple[ArchTargetRecord, ...]:
    """Load Arch consumer facts and adapt them to raw target records."""
    records: list[ArchTargetRecord] = []
    for path in paths:
        for fact in iter_arch_consumer_facts(
            path,
            period=period,
            schema_version=schema_version,
        ):
            records.append(
                arch_consumer_fact_to_target_record(
                    fact,
                    variable_of=variable_of,
                    target_type_of=target_type_of,
                    constraints_of=constraints_of,
                    geography_level_of=geography_level_of,
                    geography_id_of=geography_id_of,
                )
            )
    return tuple(records)


def iter_arch_consumer_facts(
    pathlike: str | Path,
    *,
    period: int | None = None,
    schema_version: str = ARCH_CONSUMER_FACT_SCHEMA_VERSION,
) -> Iterable[ArchConsumerFact]:
    """Yield validated Arch consumer facts from one JSONL file."""
    path = Path(pathlike)
    with path.open() as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            observed_schema_version = row.get("schema_version")
            if observed_schema_version != schema_version:
                raise ValueError(
                    "Unsupported Arch consumer fact schema "
                    f"{observed_schema_version!r} in {path} line {line_number}; "
                    f"expected {schema_version!r}."
                )
            if period is not None and arch_consumer_fact_period(row) != int(period):
                continue
            yield ArchConsumerFact(
                row=row,
                path=str(path),
                line_number=line_number,
            )


def arch_consumer_fact_to_target_record(
    fact: ArchConsumerFact,
    *,
    variable_of: Callable[[ArchConsumerFact], str | None] | None = None,
    target_type_of: Callable[[ArchConsumerFact], str] | None = None,
    constraints_of: Callable[
        [ArchConsumerFact],
        Sequence[tuple[str, str, Any]],
    ]
    | None = None,
    geography_level_of: Callable[[ArchConsumerFact], str | None] | None = None,
    geography_id_of: Callable[[ArchConsumerFact], str | None] | None = None,
) -> ArchTargetRecord:
    """Adapt one neutral consumer fact into a raw Arch target record.

    The defaults preserve the neutral Arch contract. Country packs can inject
    variable, constraint, or geography mappers without adding Python to the pack.
    """
    variable = (variable_of or _default_target_variable)(fact)
    if not variable:
        location = _fact_location(fact)
        raise ValueError(f"Arch consumer fact has no target variable{location}.")

    target_type = (target_type_of or _default_target_type)(fact)
    constraints = tuple((constraints_of or _default_constraints)(fact))
    geography_level = (geography_level_of or _default_geography_level)(fact)
    geography_id = (geography_id_of or _default_geography_id)(fact)

    row = fact.row
    observed_measure = mapping_value(row.get("observed_measure"))
    concept_alignment = mapping_value(row.get("concept_alignment"))
    source = mapping_value(row.get("source"))
    lineage = mapping_value(row.get("lineage"))
    layout = mapping_value(row.get("layout"))

    return ArchTargetRecord(
        variable=str(variable),
        period=fact.period,
        value=fact.value,
        target_type=target_type,
        geographic_level=geography_level,
        geography_id=geography_id,
        source=_first_string(
            source.get("source_name"),
            observed_measure.get("source_name"),
        )
        or "",
        source_table=_first_string(
            source.get("source_table"),
            observed_measure.get("source_table"),
        ),
        source_url=_first_string(source.get("url")),
        notes=_first_string(row.get("label"), layout.get("label")),
        unit=_first_string(observed_measure.get("unit")),
        source_record_id=fact.source_record_id,
        source_cell_keys=_string_tuple(lineage.get("source_cell_keys")),
        source_row_keys=_string_tuple(lineage.get("source_row_keys")),
        aggregate_fact_key=_first_string(row.get("aggregate_fact_key")),
        semantic_fact_key=_first_string(row.get("semantic_fact_key")),
        source_concept=_first_string(observed_measure.get("source_concept")),
        concept=_first_string(concept_alignment.get("canonical_concept")),
        concept_relation=_first_string(concept_alignment.get("relation")),
        concept_authority=_first_string(concept_alignment.get("authority")),
        concept_evidence_notes=_first_string(concept_alignment.get("notes")),
        concept_evidence_url=_first_string(concept_alignment.get("evidence_url")),
        legal_vintage=_first_string(source.get("vintage")),
        constraints=constraints,
    )


def arch_consumer_fact_concept(row: Mapping[str, Any]) -> str | None:
    """Return a row's canonical concept, falling back to source concept."""
    concept_alignment = mapping_value(row.get("concept_alignment"))
    observed_measure = mapping_value(row.get("observed_measure"))
    concept = concept_alignment.get("canonical_concept") or observed_measure.get(
        "source_concept"
    )
    return str(concept) if concept is not None else None


def arch_consumer_fact_target_type(row: Mapping[str, Any]) -> str:
    """Return the Arch target type implied by a consumer fact aggregation."""
    aggregation = mapping_value(row.get("aggregation"))
    method = aggregation.get("method")
    normalized = str(method or "").strip().lower()
    if normalized in {"sum", "amount", "total"}:
        return "AMOUNT"
    if normalized == "count":
        return "COUNT"
    if normalized in {"mean", "average"}:
        return "MEAN"
    if normalized in {"rate", "ratio"}:
        return "RATE"
    if normalized:
        return normalized.upper()
    return "AMOUNT"


def arch_consumer_fact_constraints(
    row: Mapping[str, Any],
) -> tuple[tuple[str, str, Any], ...]:
    """Return explicit target constraints from a consumer fact universe payload."""
    universe_constraints = mapping_value(row.get("universe_constraints"))
    constraints = universe_constraints.get("constraints") or ()
    if isinstance(constraints, Mapping) or isinstance(constraints, (str, bytes)):
        raise ValueError("Arch universe constraints must be a sequence of mappings.")

    target_constraints: list[tuple[str, str, Any]] = []
    for index, constraint in enumerate(constraints):
        constraint_mapping = mapping_value(constraint)
        if not constraint_mapping:
            raise ValueError(
                "Arch universe constraint "
                f"{index} must be a mapping, got {constraint!r}."
            )
        variable = constraint_mapping.get("variable")
        operator = constraint_mapping.get("operator")
        if variable is None or operator is None or "value" not in constraint_mapping:
            raise ValueError(
                "Arch universe constraint "
                f"{index} is missing variable, operator, or value."
            )
        target_constraints.append(
            (str(variable), str(operator), constraint_mapping["value"])
        )
    return tuple(target_constraints)


def arch_consumer_fact_geography_level(row: Mapping[str, Any]) -> str | None:
    """Return a normalized geography level for target-record conversion."""
    geography = mapping_value(row.get("geography"))
    level = geography.get("level")
    if level is None:
        return None
    normalized = str(level).strip().lower()
    if normalized in {"country", "national", "nation", "us", "usa"}:
        return "NATIONAL"
    return normalized.upper()


def arch_consumer_fact_geography_id(row: Mapping[str, Any]) -> str | None:
    """Return a geography identifier for target-record conversion."""
    if arch_consumer_fact_geography_level(row) == "NATIONAL":
        return None
    geography_id = mapping_value(row.get("geography")).get("id")
    return str(geography_id) if geography_id not in (None, "") else None


def arch_consumer_fact_period(row: Mapping[str, Any]) -> int:
    """Return a consumer fact period as an integer year."""
    period = mapping_value(row.get("period"))
    value = period["value"]
    if period.get("type") == "month" and isinstance(value, str):
        return int(value.split("-", maxsplit=1)[0])
    return int(value)


def arch_consumer_fact_source_record_id(row: Mapping[str, Any]) -> str | None:
    """Return a source record ID from a consumer fact lineage payload."""
    lineage = mapping_value(row.get("lineage"))
    source_record_id = lineage.get("source_record_id")
    return str(source_record_id) if source_record_id is not None else None


def arch_consumer_fact_numeric_value(value: Any) -> float:
    """Return a numeric consumer fact value."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Arch consumer fact value is not numeric: {value!r}")
    if isinstance(value, (int, float, str)):
        return float(value)
    raise ValueError(f"Arch consumer fact value is not numeric: {value!r}")


def mapping_value(value: Any) -> Mapping[str, Any]:
    """Return a mapping payload, or an empty mapping for malformed/empty values."""
    return value if isinstance(value, Mapping) else {}


def _default_target_variable(fact: ArchConsumerFact) -> str | None:
    return fact.concept


def _default_target_type(fact: ArchConsumerFact) -> str:
    return arch_consumer_fact_target_type(fact.row)


def _default_constraints(fact: ArchConsumerFact) -> tuple[tuple[str, str, Any], ...]:
    return arch_consumer_fact_constraints(fact.row)


def _default_geography_level(fact: ArchConsumerFact) -> str | None:
    return arch_consumer_fact_geography_level(fact.row)


def _default_geography_id(fact: ArchConsumerFact) -> str | None:
    return arch_consumer_fact_geography_id(fact.row)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (str(value),)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    return (str(value),)


def _first_string(*values: Any) -> str | None:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return None


def _fact_location(fact: ArchConsumerFact) -> str:
    if fact.path is None:
        return ""
    if fact.line_number is None:
        return f" in {fact.path}"
    return f" in {fact.path} line {fact.line_number}"


__all__ = [
    "ARCH_CONSUMER_FACT_SCHEMA_VERSION",
    "ArchConsumerFact",
    "arch_consumer_fact_constraints",
    "arch_consumer_fact_concept",
    "arch_consumer_fact_geography_id",
    "arch_consumer_fact_geography_level",
    "arch_consumer_fact_numeric_value",
    "arch_consumer_fact_period",
    "arch_consumer_fact_source_record_id",
    "arch_consumer_fact_target_type",
    "arch_consumer_fact_to_target_record",
    "iter_arch_consumer_facts",
    "load_arch_consumer_fact_jsonl_rows",
    "load_arch_consumer_facts",
    "load_arch_target_records",
    "mapping_value",
]
