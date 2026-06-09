"""Simulation-aware target compilation helpers.

Country-specific simulator logic stays outside core. The helpers here provide
two generic adapters for targets whose measure or filter features are produced
by a runtime simulator: one that applies modifier handlers target-by-target,
and one that delegates grouped feature materialization to an injected adapter.
Both paths then reuse the ordinary frame-backed target compiler.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from microplex.core import EntityType
from microplex.targets.reweighting import (
    TargetConstraintCompilationResult,
    compile_target_reweighting_constraints,
)
from microplex.targets.spec import TargetSimulationModifier, TargetSpec

__all__ = [
    "MaterializedSimulationTargetCompiler",
    "MaterializingSimulationTargetCompiler",
    "SimulationFeatureMaterializer",
    "SimulationModifierHandler",
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
        entity_weight_indexes: Mapping[EntityType, pd.Series | np.ndarray],
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


@runtime_checkable
class SimulationModifierHandler(Protocol):
    """Handler protocol for one target simulation modifier."""

    def apply_simulation_modifier(
        self,
        entity_frames: Mapping[EntityType, pd.DataFrame],
        *,
        target: TargetSpec,
        modifier: TargetSimulationModifier,
        entity_weight_indexes: Mapping[EntityType, pd.Series | np.ndarray],
    ) -> Mapping[EntityType, pd.DataFrame]:
        """Return entity frames after applying ``modifier`` for ``target``."""


class MaterializingSimulationTargetCompiler:
    """Compile simulator-modified targets through registered handlers.

    Each target is materialized independently, preserving target order and
    letting the existing frame-target compiler produce the final sparse row.
    This is deliberately generic: a handler can run PolicyEngine, rerandomize
    takeup inputs, or apply another simulator-specific transform.
    """

    def __init__(
        self,
        modifier_handlers: Mapping[str, SimulationModifierHandler],
    ) -> None:
        self.modifier_handlers = dict(modifier_handlers)

    def compile_simulation_target_constraints(
        self,
        *,
        targets: Sequence[TargetSpec],
        entity_frames: Mapping[EntityType, pd.DataFrame],
        entity_weight_indexes: Mapping[EntityType, pd.Series | np.ndarray],
    ) -> TargetConstraintCompilationResult:
        constraints = []
        skipped: list[tuple[str, str]] = []
        for target in targets:
            frames = {entity: frame.copy() for entity, frame in entity_frames.items()}
            skip_reason: str | None = None
            for modifier in target.sim_modifiers:
                handler = self.modifier_handlers.get(modifier.name)
                if handler is None:
                    skip_reason = f"missing_simulation_modifier_handler:{modifier.name}"
                    break
                frames = _apply_modifier_handler(
                    handler,
                    frames,
                    target=target,
                    modifier=modifier,
                    entity_weight_indexes=entity_weight_indexes,
                )
            if skip_reason is not None:
                skipped.append((target.name, skip_reason))
                continue

            compilation = compile_target_reweighting_constraints(
                targets=[replace(target, sim_modifiers=())],
                entity_frames=frames,
                entity_weight_indexes=dict(entity_weight_indexes),
            )
            constraints.extend(compilation.constraints)
            skipped.extend(compilation.skipped_targets)

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
        entity_type = entity if isinstance(entity, EntityType) else EntityType(entity)
        if not isinstance(simulated, pd.DataFrame):
            raise TypeError(
                "simulation materializer returned "
                f"{type(simulated).__name__} for entity {entity_type.value!r}; "
                "expected pandas.DataFrame."
            )
        base = merged.get(entity_type)
        if base is None:
            base = pd.DataFrame(index=simulated.index)
            merged[entity_type] = base
        if len(base) != len(simulated):
            raise ValueError(
                f"simulation materializer returned {len(simulated)} rows for "
                f"{entity_type.value}, expected {len(base)}"
            )
        simulated = simulated.reset_index(drop=True)
        base = base.reset_index(drop=True)
        for column in simulated.columns:
            base[column] = simulated[column]
        merged[entity_type] = base
    return merged


def _apply_modifier_handler(
    handler: SimulationModifierHandler,
    entity_frames: Mapping[EntityType, pd.DataFrame],
    *,
    target: TargetSpec,
    modifier: TargetSimulationModifier,
    entity_weight_indexes: Mapping[EntityType, pd.Series | np.ndarray],
) -> dict[EntityType, pd.DataFrame]:
    apply_method = getattr(handler, "apply_simulation_modifier", None)
    if apply_method is not None:
        updated = apply_method(
            entity_frames,
            target=target,
            modifier=modifier,
            entity_weight_indexes=entity_weight_indexes,
        )
    else:
        updated = handler(  # type: ignore[operator]
            entity_frames,
            target=target,
            modifier=modifier,
            entity_weight_indexes=entity_weight_indexes,
        )
    if not isinstance(updated, Mapping):
        raise TypeError(
            "Simulation modifier handler returned "
            f"{type(updated).__name__}; expected a mapping of entity frames."
        )
    merged = dict(entity_frames)
    for entity, frame in updated.items():
        entity_type = entity if isinstance(entity, EntityType) else EntityType(entity)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"Simulation modifier handler returned {type(frame).__name__} "
                f"for entity {entity_type.value!r}; expected pandas.DataFrame."
            )
        if entity_type in entity_frames and len(frame) != len(
            entity_frames[entity_type]
        ):
            raise ValueError(
                "Simulation modifier handler changed row count for entity "
                f"{entity_type.value!r} from {len(entity_frames[entity_type])} "
                f"to {len(frame)}."
            )
        merged[entity_type] = frame
    return merged
