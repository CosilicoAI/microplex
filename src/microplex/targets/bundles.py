"""Shared entity-table bundle primitives for target reweighting."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from microplex.core import EntityType
from microplex.core.sources import EntityRelationship, ObservationFrame


@dataclass(frozen=True)
class EntityTableBinding:
    """How one entity table maps onto a shared reweighting vector."""

    frame: pd.DataFrame
    id_column: str
    weight_link_column: str | None = None
    synced_weight_column: str | None = None

    def __post_init__(self) -> None:
        if self.id_column not in self.frame.columns:
            raise ValueError(f"Entity table is missing id column '{self.id_column}'")
        if (
            self.weight_link_column is not None
            and self.weight_link_column not in self.frame.columns
        ):
            raise ValueError(
                f"Entity table is missing weight linkage column '{self.weight_link_column}'"
            )
        if self.synced_weight_column is not None and self.weight_link_column is None:
            raise ValueError(
                "synced_weight_column requires weight_link_column for non-weight entities"
            )


@dataclass(frozen=True)
class EntityTableBundle:
    """Entity tables aligned to one shared weight entity."""

    weight_entity: EntityType
    weight_column: str
    bindings: dict[EntityType, EntityTableBinding]

    def __post_init__(self) -> None:
        if self.weight_entity not in self.bindings:
            raise ValueError(
                f"Weight entity '{self.weight_entity.value}' must be present in bindings"
            )
        weight_binding = self.bindings[self.weight_entity]
        if self.weight_column not in weight_binding.frame.columns:
            raise ValueError(
                f"Weight entity table is missing weight column '{self.weight_column}'"
            )

    def available_entities(self) -> tuple[EntityType, ...]:
        return tuple(self.bindings)

    def table_for(self, entity: EntityType) -> pd.DataFrame:
        binding = self.bindings.get(entity)
        if binding is None:
            raise KeyError(f"No table available for entity '{entity.value}'")
        return binding.frame

    def entity_frames(self) -> dict[EntityType, pd.DataFrame]:
        return {entity: binding.frame for entity, binding in self.bindings.items()}

    def initial_weights(self) -> pd.Series:
        return self.bindings[self.weight_entity].frame[self.weight_column]

    def entity_weight_indexes(self) -> dict[EntityType, pd.Series]:
        weight_binding = self.bindings[self.weight_entity]
        weight_index = pd.Series(
            np.arange(len(weight_binding.frame), dtype=int),
            index=weight_binding.frame[weight_binding.id_column],
        )
        mappings: dict[EntityType, pd.Series] = {}
        for entity, binding in self.bindings.items():
            link_column = (
                binding.id_column
                if entity is self.weight_entity
                else binding.weight_link_column
            )
            if link_column is None:
                continue
            mappings[entity] = pd.to_numeric(
                binding.frame[link_column].map(weight_index),
                errors="coerce",
            )
        return mappings

    def with_updated_weights(
        self, weights: pd.Series | np.ndarray
    ) -> EntityTableBundle:
        weight_binding = self.bindings[self.weight_entity]
        updated_weights = np.asarray(weights, dtype=float)
        if len(updated_weights) != len(weight_binding.frame):
            raise ValueError("Updated weights must align to the weight entity table")

        updated_bindings = dict(self.bindings)
        updated_weight_frame = weight_binding.frame.copy()
        updated_weight_frame[self.weight_column] = updated_weights
        updated_bindings[self.weight_entity] = replace(
            weight_binding,
            frame=updated_weight_frame,
        )
        weight_by_id = updated_weight_frame.set_index(weight_binding.id_column)[
            self.weight_column
        ]

        for entity, binding in self.bindings.items():
            if entity is self.weight_entity or binding.synced_weight_column is None:
                continue
            if binding.weight_link_column is None:
                raise ValueError(
                    f"Entity '{entity.value}' cannot sync weights without a linkage column"
                )
            updated_frame = binding.frame.copy()
            updated_frame[binding.synced_weight_column] = updated_frame[
                binding.weight_link_column
            ].map(weight_by_id)
            updated_bindings[entity] = replace(binding, frame=updated_frame)

        return EntityTableBundle(
            weight_entity=self.weight_entity,
            weight_column=self.weight_column,
            bindings=updated_bindings,
        )


def entity_table_bundle_from_observation_frame(
    frame: ObservationFrame,
    *,
    weight_entity: EntityType | str,
    weight_column: str | None = None,
    sync_child_weight_columns: bool = True,
) -> EntityTableBundle:
    """Build a calibration bundle from a related observation frame.

    Non-weight entity tables must have a direct relationship to the weight
    entity so their rows can map onto the shared reweighting vector.
    """
    frame.validate()
    resolved_weight_entity = (
        weight_entity
        if isinstance(weight_entity, EntityType)
        else EntityType(weight_entity)
    )
    weight_observation = frame.source.observation_for(resolved_weight_entity)
    resolved_weight_column = weight_column or weight_observation.weight_column
    if resolved_weight_column is None:
        raise ValueError(
            f"weight entity {resolved_weight_entity.value!r} has no weight column; "
            "supply weight_column explicitly"
        )

    direct_relationships = _relationships_from_weight_entity(
        frame.relationships,
        resolved_weight_entity,
    )
    bindings: dict[EntityType, EntityTableBinding] = {}
    for observation in frame.source.observations:
        entity = observation.entity
        table = frame.tables[entity].copy()
        if entity is resolved_weight_entity:
            bindings[entity] = EntityTableBinding(
                frame=table,
                id_column=observation.key_column,
            )
            continue

        relationship = direct_relationships.get(entity)
        if relationship is None:
            raise ValueError(
                f"Entity {entity.value!r} is not directly linked to weight "
                f"entity {resolved_weight_entity.value!r}"
            )
        if relationship.parent_key != weight_observation.key_column:
            raise ValueError(
                f"Entity {entity.value!r} links to weight entity "
                f"{resolved_weight_entity.value!r} through parent key "
                f"{relationship.parent_key!r}, but calibration bundles require "
                f"the weight entity id column {weight_observation.key_column!r}"
            )
        bindings[entity] = EntityTableBinding(
            frame=table,
            id_column=observation.key_column,
            weight_link_column=relationship.child_key,
            synced_weight_column=observation.weight_column
            if sync_child_weight_columns
            else None,
        )

    return EntityTableBundle(
        weight_entity=resolved_weight_entity,
        weight_column=resolved_weight_column,
        bindings=bindings,
    )


def _relationships_from_weight_entity(
    relationships: tuple[EntityRelationship, ...],
    weight_entity: EntityType,
) -> dict[EntityType, EntityRelationship]:
    direct: dict[EntityType, EntityRelationship] = {}
    for relationship in relationships:
        if relationship.parent_entity is not weight_entity:
            continue
        if relationship.child_entity in direct:
            raise ValueError(
                "Observation frame has multiple direct relationships from "
                f"{weight_entity.value!r} to {relationship.child_entity.value!r}"
            )
        direct[relationship.child_entity] = relationship
    return direct
