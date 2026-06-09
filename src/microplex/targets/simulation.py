"""Simulation-backed target compiler adapters.

The sparse calibration path can compile ordinary targets directly from entity
tables. Targets whose measures/filters must first be computed by a tax-benefit
model declare ``TargetSpec.sim_modifiers`` and route through a
``SimulationTargetCompiler``. This module provides a generic compiler that
delegates feature materialization to an injected adapter, then reuses the same
frame-based target compiler for the resulting columns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import pandas as pd

from microplex.core import EntityType
from microplex.targets.reweighting import (
    TargetConstraintCompilationResult,
    compile_target_reweighting_constraints,
)
from microplex.targets.spec import TargetSimulationModifier, TargetSpec

__all__ = [
    "MaterializedSimulationTargetCompiler",
    "SimulationFeatureMaterializer",
]


@runtime_checkable
class SimulationFeatureMaterializer(Protocol):
    """Materialize simulated features needed by a group of target specs."""

    def materialize_simulation_features(
        self,
        *,
        targets: Sequence[TargetSpec],
        entity_frames: Mapping[EntityType, pd.DataFrame],
        modifiers: Sequence[TargetSimulationModifier],
    ) -> Mapping[EntityType, pd.DataFrame]:
        """Return simulated feature tables aligned to the input entity frames."""


@dataclass(frozen=True)
class MaterializedSimulationTargetCompiler:
    """Compile simulator-dependent targets from materialized entity features."""

    materializer: SimulationFeatureMaterializer

    def compile_simulation_target_constraints(
        self,
        *,
        targets: Sequence[TargetSpec],
        entity_frames: Mapping[EntityType, pd.DataFrame],
        entity_weight_indexes: Mapping[EntityType, pd.Series],
    ) -> TargetConstraintCompilationResult:
        constraints = []
        skipped = []
        for modifiers, grouped_targets in _targets_by_modifiers(targets):
            materialized_frames = self.materializer.materialize_simulation_features(
                targets=grouped_targets,
                entity_frames=entity_frames,
                modifiers=modifiers,
            )
            compiled = compile_target_reweighting_constraints(
                targets=[
                    replace(target, sim_modifiers=()) for target in grouped_targets
                ],
                entity_frames=_merged_entity_frames(entity_frames, materialized_frames),
                entity_weight_indexes=dict(entity_weight_indexes),
            )
            constraints.extend(compiled.constraints)
            skipped.extend(compiled.skipped_targets)
        return TargetConstraintCompilationResult(
            constraints=tuple(constraints),
            skipped_targets=tuple(skipped),
        )


def _targets_by_modifiers(
    targets: Sequence[TargetSpec],
) -> tuple[tuple[tuple[TargetSimulationModifier, ...], tuple[TargetSpec, ...]], ...]:
    groups: dict[
        tuple[tuple[str, tuple[tuple[str, str], ...]], ...],
        tuple[tuple[TargetSimulationModifier, ...], list[TargetSpec]],
    ] = {}
    for target in targets:
        modifiers = target.sim_modifiers
        key = _modifier_key(modifiers)
        if key not in groups:
            groups[key] = (modifiers, [])
        groups[key][1].append(target)
    return tuple((modifiers, tuple(group)) for modifiers, group in groups.values())


def _modifier_key(
    modifiers: Sequence[TargetSimulationModifier],
) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    return tuple(
        (
            modifier.name,
            tuple(
                sorted(
                    (str(key), repr(value))
                    for key, value in modifier.parameters.items()
                )
            ),
        )
        for modifier in modifiers
    )


def _merged_entity_frames(
    entity_frames: Mapping[EntityType, pd.DataFrame],
    materialized_frames: Mapping[EntityType, pd.DataFrame],
) -> dict[EntityType, pd.DataFrame]:
    merged = {entity: frame.copy() for entity, frame in entity_frames.items()}
    for entity, simulated in materialized_frames.items():
        base = merged.get(entity)
        if base is None:
            base = pd.DataFrame(index=simulated.index)
            merged[entity] = base
        if len(base) != len(simulated):
            raise ValueError(
                f"simulation materializer returned {len(simulated)} rows for "
                f"{entity.value}, expected {len(base)}"
            )
        simulated = simulated.reset_index(drop=True)
        base = base.reset_index(drop=True)
        for column in simulated.columns:
            base[column] = simulated[column]
        merged[entity] = base
    return merged
