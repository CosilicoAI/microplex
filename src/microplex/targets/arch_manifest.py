"""Manifest-backed Arch target mapping helpers.

Country packs provide declarative Arch target manifests. This module turns those
manifests into the callables used by the neutral JSONL loader and the Arch
target provider, so packs can stay content-only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microplex.core import EntityType
from microplex.targets.arch import (
    ArchConsumerFact,
    arch_consumer_fact_constraints,
    arch_consumer_fact_geography_id,
    arch_consumer_fact_geography_level,
    arch_consumer_fact_target_type,
    mapping_value,
)
from microplex.targets.arch_derivations import ArchTargetRecord
from microplex.targets.arch_provider import ArchPipelineConfig

ARCH_TARGET_MANIFEST_SCHEMA_VERSION = "microplex.arch_targets.v1"


@dataclass(frozen=True)
class ArchTargetManifest:
    """Declarative mapping from neutral Arch consumer facts to Microplex targets."""

    payload: Mapping[str, Any]

    @classmethod
    def from_path(cls, pathlike: str | Path) -> ArchTargetManifest:
        path = Path(pathlike)
        with path.open() as file:
            return cls.from_dict(json.load(file))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArchTargetManifest:
        schema_version = payload.get("schema_version")
        if schema_version != ARCH_TARGET_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported Arch target manifest schema "
                f"{schema_version!r}; expected "
                f"{ARCH_TARGET_MANIFEST_SCHEMA_VERSION!r}."
            )
        return cls(dict(payload))

    def variable_of(self, fact: ArchConsumerFact) -> str | None:
        """Return the Microplex target variable for one Arch consumer fact."""
        entry = self.target_mapping_for_fact(fact)
        if entry:
            return _optional_string(entry.get("variable"))
        if _bool(self.payload.get("require_target_mapping"), default=False):
            concept = fact.concept or "<missing concept>"
            raise ValueError(f"No Arch target mapping for {concept!r}.")
        return fact.concept

    def target_type_of(self, fact: ArchConsumerFact) -> str:
        """Return COUNT/AMOUNT/etc. after applying manifest concept mappings."""
        entry = self.target_mapping_for_fact(fact)
        if entry and entry.get("target_type") is not None:
            return str(entry["target_type"]).upper()
        return arch_consumer_fact_target_type(fact.row)

    def constraints_of(
        self, fact: ArchConsumerFact
    ) -> tuple[tuple[str, str, Any], ...]:
        """Return mapped universe constraints plus explicit geography/count filters."""
        constraints: list[tuple[str, str, Any]] = []
        for variable, operator, value in arch_consumer_fact_constraints(fact.row):
            constraints.extend(self._map_constraint(variable, operator, value))

        geography_constraint = self._geography_constraint(fact)
        if geography_constraint is not None:
            constraints.append(geography_constraint)

        variable = self.variable_of(fact)
        target_type = self.target_type_of(fact)
        if variable is not None and target_type == "COUNT":
            measure = self._positive_count_filter_measure(variable)
            if measure is not None:
                constraints.append((measure, ">", 0))
        elif variable is not None and target_type == "AMOUNT":
            measure = self._positive_amount_filter_measure(variable)
            if measure is not None:
                constraints.append((measure, ">", 0))

        return _dedupe_constraints(constraints)

    def geography_level_of(self, fact: ArchConsumerFact) -> str | None:
        return arch_consumer_fact_geography_level(fact.row)

    def geography_id_of(self, fact: ArchConsumerFact) -> str | None:
        """Return metadata geography, omitting IDs already encoded as constraints."""
        if self._geography_constraint(fact) is not None:
            return None
        return arch_consumer_fact_geography_id(fact.row)

    def entity_of(self, variable: str) -> EntityType:
        """Return the PE entity for a target variable."""
        entity = self._entity_for_variable(variable)
        if entity is None:
            default_entity = _optional_string(self.payload.get("default_entity"))
            if default_entity is not None:
                entity = default_entity
        if entity is None:
            raise KeyError(f"No Arch entity mapping for variable {variable!r}.")
        return EntityType(entity)

    def measure_of(self, variable: str) -> str | None:
        """Return an AMOUNT target's measure override, if any."""
        return _optional_string(self._amount_measures().get(variable))

    def count_measure(self, entity: EntityType) -> str:
        """Return the count measure to sum for a COUNT target entity."""
        count_measures = self._count_measures()
        return _optional_string(count_measures.get(entity.value)) or (
            f"{entity.value}_count"
        )

    def geo_feature(self, geo_level: str | None) -> str | None:
        """Return the support feature that scopes a geography level."""
        if geo_level is None:
            return None
        return _optional_string(self._geo_features().get(geo_level.lower()))

    def state_fips_of(self, record: ArchTargetRecord) -> str | None:
        """Extract state FIPS from either explicit constraints or geography ID."""
        state_feature = self.geo_feature("state")
        if state_feature is not None:
            for variable, operator, value in record.constraints:
                if variable == state_feature and operator == "==":
                    return str(value)
        if (record.geographic_level or "").lower() == "state":
            return _state_fips_from_geo_id(record.geography_id, self._geography())
        return None

    def pipeline_config(self, *, target_year: int | None = None) -> ArchPipelineConfig:
        """Build the generic Arch derivation config declared by the manifest."""
        derivations = mapping_value(self.payload.get("derivations"))
        component_sum_map = {
            str(variable): tuple(str(component) for component in components)
            for variable, components in mapping_value(
                derivations.get("component_sum_map")
            ).items()
            if isinstance(components, Sequence) and not isinstance(components, str)
        }
        rollup_required_states = tuple(
            str(state) for state in derivations.get("rollup_required_states", ())
        )
        bea_wage_component_variables = {
            str(key): str(value)
            for key, value in mapping_value(
                derivations.get("bea_wage_component_variables")
            ).items()
        }
        resolved_year = target_year
        if resolved_year is None:
            resolved_year = int(self.payload.get("model_year"))
        return ArchPipelineConfig(
            target_year=resolved_year,
            component_sum_map=component_sum_map,
            rollup_required_states=rollup_required_states,
            bea_output_variable=_optional_string(
                derivations.get("bea_output_variable")
            ),
            bea_required_states=tuple(
                str(state) for state in derivations.get("bea_required_states", ())
            ),
            bea_wage_component_variables=bea_wage_component_variables,
            age_soi=_bool(derivations.get("age_soi"), default=True),
            soi_source=str(derivations.get("soi_source", "IRS_SOI")),
            state_fips_of=self.state_fips_of,
        )

    def target_mapping_for_fact(self, fact: ArchConsumerFact) -> Mapping[str, Any]:
        target_mappings = mapping_value(self.payload.get("target_mappings"))
        for concept in _concept_candidates(fact):
            entry = target_mappings.get(concept)
            if entry is not None:
                return mapping_value(entry)
        return {}

    def _map_constraint(
        self, variable: str, operator: str, value: Any
    ) -> tuple[tuple[str, str, Any], ...]:
        constraints = mapping_value(self.payload.get("constraints"))
        positive_aliases = mapping_value(constraints.get("positive_aliases"))
        if variable in positive_aliases and operator in {"=", "=="}:
            feature = str(positive_aliases[variable])
            if _truthy_constraint_value(value):
                return ((feature, ">", 0),)
            return ((feature, "==", 0),)

        aliases = mapping_value(constraints.get("aliases"))
        return ((str(aliases.get(variable, variable)), operator, value),)

    def _geography_constraint(
        self, fact: ArchConsumerFact
    ) -> tuple[str, str, str] | None:
        level = self.geography_level_of(fact)
        if (level or "").lower() != "state":
            return None
        state_fips = _state_fips_from_geo_id(
            arch_consumer_fact_geography_id(fact.row),
            self._geography(),
        )
        feature = self.geo_feature("state")
        if state_fips is None or feature is None:
            return None
        return (feature, "==", state_fips)

    def _positive_count_filter_measure(self, variable: str) -> str | None:
        count_alias = mapping_value(self._count_aliases().get(variable))
        explicit = _optional_string(count_alias.get("positive_filter_measure"))
        if explicit is not None:
            return explicit

        suffixes = self.payload.get("positive_count_suffixes", ())
        if not isinstance(suffixes, Sequence) or isinstance(suffixes, str):
            return None
        amount_measures = self._amount_measures()
        for suffix_entry in suffixes:
            suffix_mapping = mapping_value(suffix_entry)
            suffix = _optional_string(suffix_mapping.get("suffix"))
            amount_suffix = _optional_string(suffix_mapping.get("amount_suffix"))
            if suffix is None or amount_suffix is None:
                continue
            if variable.endswith(suffix):
                amount_variable = variable[: -len(suffix)] + amount_suffix
                return _optional_string(amount_measures.get(amount_variable))
        return None

    def _positive_amount_filter_measure(self, variable: str) -> str | None:
        positive_amount_filters = self.payload.get("positive_amount_filters", ())
        if not isinstance(positive_amount_filters, Sequence) or isinstance(
            positive_amount_filters, str
        ):
            return None
        if variable not in positive_amount_filters:
            return None
        return self.measure_of(variable) or variable

    def _entity_for_variable(self, variable: str) -> str | None:
        entities = mapping_value(self.payload.get("entities"))
        entity = _optional_string(entities.get(variable))
        if entity is not None:
            return entity

        count_alias = mapping_value(self._count_aliases().get(variable))
        entity = _optional_string(count_alias.get("entity"))
        if entity is not None:
            return entity

        for entry in mapping_value(self.payload.get("target_mappings")).values():
            entry_mapping = mapping_value(entry)
            if entry_mapping.get("variable") == variable:
                entity = _optional_string(entry_mapping.get("entity"))
                if entity is not None:
                    return entity
        return None

    def _amount_measures(self) -> Mapping[str, Any]:
        return mapping_value(self.payload.get("amount_measures"))

    def _count_aliases(self) -> Mapping[str, Any]:
        return mapping_value(self.payload.get("count_aliases"))

    def _count_measures(self) -> Mapping[str, Any]:
        return mapping_value(self.payload.get("count_measures"))

    def _geography(self) -> Mapping[str, Any]:
        return mapping_value(self.payload.get("geography"))

    def _geo_features(self) -> Mapping[str, Any]:
        return mapping_value(self._geography().get("features"))


def load_arch_target_manifest(pathlike: str | Path) -> ArchTargetManifest:
    return ArchTargetManifest.from_path(pathlike)


def _concept_candidates(fact: ArchConsumerFact) -> tuple[str, ...]:
    concept_alignment = mapping_value(fact.row.get("concept_alignment"))
    observed_measure = mapping_value(fact.row.get("observed_measure"))
    candidates = (
        concept_alignment.get("canonical_concept"),
        concept_alignment.get("source_concept"),
        observed_measure.get("source_concept"),
        fact.concept,
    )
    seen: set[str] = set()
    out: list[str] = []
    for candidate in candidates:
        if candidate in (None, ""):
            continue
        text = str(candidate)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return tuple(out)


def _state_fips_from_geo_id(
    geography_id: str | None, geography: Mapping[str, Any]
) -> str | None:
    if geography_id in (None, ""):
        return None
    raw = str(geography_id)
    if raw.isdigit() and len(raw) <= 2:
        return raw.zfill(2)
    prefixes = geography.get("state_geoid_prefixes", ("0400000US",))
    if isinstance(prefixes, str):
        prefixes = (prefixes,)
    if not isinstance(prefixes, Sequence):
        return None
    for prefix in prefixes:
        prefix_text = str(prefix)
        if raw.startswith(prefix_text):
            suffix = raw[len(prefix_text) :]
            return suffix.zfill(2) if suffix.isdigit() else suffix
    return None


def _dedupe_constraints(
    constraints: Sequence[tuple[str, str, Any]],
) -> tuple[tuple[str, str, Any], ...]:
    seen: set[tuple[str, str, str]] = set()
    out: list[tuple[str, str, Any]] = []
    for variable, operator, value in constraints:
        key = (variable, operator, _constraint_value_key(value))
        if key in seen:
            continue
        seen.add(key)
        out.append((variable, operator, value))
    return tuple(out)


def _constraint_value_key(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return repr(value)


def _truthy_constraint_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none"}
    return bool(value)


def _bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _optional_string(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


__all__ = [
    "ARCH_TARGET_MANIFEST_SCHEMA_VERSION",
    "ArchTargetManifest",
    "load_arch_target_manifest",
]
