"""Simulation-aware target compilation helpers.

This module keeps country-specific simulator logic outside core. A country
adapter supplies modifier handlers that materialize or transform entity frames;
core applies them and then uses the ordinary linear target compiler.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from microplex.core import EntityType
from microplex.targets.reweighting import (
    TargetConstraintCompilationResult,
    compile_target_reweighting_constraints,
)
from microplex.targets.spec import TargetSimulationModifier, TargetSpec


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

            plain_target = replace(target, sim_modifiers=())
            compilation = compile_target_reweighting_constraints(
                targets=[plain_target],
                entity_frames=frames,
                entity_weight_indexes=dict(entity_weight_indexes),
            )
            constraints.extend(compilation.constraints)
            skipped.extend(compilation.skipped_targets)

        return TargetConstraintCompilationResult(
            constraints=tuple(constraints),
            skipped_targets=tuple(skipped),
        )


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
