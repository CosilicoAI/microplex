"""Provider-backed source resolution for spec-driven runs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from microplex.core import EntityType
from microplex.core.sources import ObservationFrame, SourceProvider, SourceQuery
from microplex.spec import MicroplexSpec, SourceSpec


@dataclass(frozen=True)
class RegisteredSourceProvider:
    """One provider registered under a spec dataset id."""

    provider: SourceProvider
    default_entity: EntityType | str | None = None

    def default_entity_type(self) -> EntityType | None:
        if self.default_entity is None:
            return None
        if isinstance(self.default_entity, EntityType):
            return self.default_entity
        return EntityType(self.default_entity)


@dataclass
class SourceRegistry:
    """Resolve spec source declarations into runner-ready frames.

    Country packages declare dataset ids and optional entities in YAML. The
    executable loader remains in Microplex: callers register providers for those
    dataset ids, then :meth:`resolve_sources` loads validated observation frames
    and selects the declared entity table for the existing frame-based runner.
    """

    _providers: dict[str, RegisteredSourceProvider]

    def __init__(
        self,
        providers: dict[str, SourceProvider | RegisteredSourceProvider] | None = None,
    ) -> None:
        self._providers = {}
        for dataset, provider in (providers or {}).items():
            if isinstance(provider, RegisteredSourceProvider):
                self._providers[dataset] = provider
            else:
                self.register(dataset, provider)

    def register(
        self,
        dataset: str,
        provider: SourceProvider,
        *,
        default_entity: EntityType | str | None = None,
    ) -> SourceRegistry:
        """Register a provider for one spec dataset id."""
        dataset_id = dataset.strip()
        if not dataset_id:
            raise ValueError("SourceRegistry dataset id must be non-empty")
        if dataset_id in self._providers:
            raise ValueError(
                f"SourceRegistry already has a provider for {dataset_id!r}"
            )
        self._providers[dataset_id] = RegisteredSourceProvider(
            provider=provider,
            default_entity=default_entity,
        )
        return self

    def provider_for(self, dataset: str) -> SourceProvider:
        """Return the provider registered for one dataset id."""
        return self._registered_provider(dataset).provider

    def resolve_source(
        self,
        spec: MicroplexSpec,
        source_name: str,
    ) -> pd.DataFrame:
        """Load and select one frame declared in ``spec``."""
        observation_frame = self.resolve_observation_frame(spec, source_name)
        source_spec = spec.sources[source_name]
        registered = self._registered_provider(source_spec.dataset)
        entity = self._select_entity(
            source_name=source_name,
            source_spec=source_spec,
            registered=registered,
            observation_frame=observation_frame,
        )
        return observation_frame.tables[entity].copy()

    def resolve_observation_frame(
        self,
        spec: MicroplexSpec,
        source_name: str,
    ) -> ObservationFrame:
        """Load the full observation frame declared for one source."""
        try:
            source_spec = spec.sources[source_name]
        except KeyError as exc:
            available = ", ".join(sorted(spec.sources)) or "<none>"
            raise KeyError(
                f"Spec has no source {source_name!r}. Available sources: {available}."
            ) from exc
        registered = self._registered_provider(source_spec.dataset)
        query = SourceQuery(
            period=spec.meta.model_year,
            provider_filters={
                "dataset": source_spec.dataset,
                "source_name": source_name,
                "role": source_spec.role.value,
            },
        )
        observation_frame = registered.provider.load_frame(query)
        observation_frame.validate()
        return observation_frame

    def resolve_sources(self, spec: MicroplexSpec) -> dict[str, pd.DataFrame]:
        """Load and select frames for every source declared in ``spec``."""
        return {
            source_name: self.resolve_source(spec, source_name)
            for source_name in spec.sources
        }

    def resolve_observation_frames(
        self,
        spec: MicroplexSpec,
    ) -> dict[str, ObservationFrame]:
        """Load full observation frames for every source declared in ``spec``."""
        return {
            source_name: self.resolve_observation_frame(spec, source_name)
            for source_name in spec.sources
        }

    def _registered_provider(self, dataset: str) -> RegisteredSourceProvider:
        try:
            return self._providers[dataset]
        except KeyError as exc:
            available = ", ".join(sorted(self._providers)) or "<none>"
            raise KeyError(
                f"No SourceRegistry provider registered for dataset {dataset!r}. "
                f"Available datasets: {available}."
            ) from exc

    def _select_entity(
        self,
        *,
        source_name: str,
        source_spec: SourceSpec,
        registered: RegisteredSourceProvider,
        observation_frame: ObservationFrame,
    ) -> EntityType:
        explicit_entity = (
            EntityType(source_spec.entity) if source_spec.entity is not None else None
        )
        entity = explicit_entity or registered.default_entity_type()
        if entity is None:
            observed_entities = observation_frame.source.observed_entities
            if len(observed_entities) == 1:
                entity = observed_entities[0]
            else:
                observed = ", ".join(observed.value for observed in observed_entities)
                raise ValueError(
                    f"Source {source_name!r} dataset {source_spec.dataset!r} "
                    "materialized multiple entity tables "
                    f"({observed}); declare sources.{source_name}.entity or "
                    "register a default_entity."
                )

        if entity not in observation_frame.tables:
            available = ", ".join(
                observed.value
                for observed in observation_frame.source.observed_entities
            )
            raise KeyError(
                f"Source {source_name!r} requested entity {entity.value!r}, "
                f"but dataset {source_spec.dataset!r} provides: {available}."
            )
        return entity
