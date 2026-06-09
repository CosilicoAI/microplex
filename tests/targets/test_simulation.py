"""Tests for simulator-modified target compilation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microplex.core import EntityType
from microplex.targets import (
    MaterializingSimulationTargetCompiler,
    TargetAggregation,
    TargetFilter,
    TargetSimulationModifier,
    TargetSpec,
)


class SnapMaterializer:
    def apply_simulation_modifier(
        self,
        entity_frames,
        *,
        target,
        modifier,
        entity_weight_indexes,
    ):
        assert target.name == "snap_total"
        assert modifier.parameters == {"variables": ["snap"]}
        assert EntityType.HOUSEHOLD in entity_weight_indexes
        households = entity_frames[EntityType.HOUSEHOLD].copy()
        households["snap"] = np.array([12.0, 0.0])
        return {EntityType.HOUSEHOLD: households}


def test_materializing_simulation_compiler_materializes_missing_measure():
    compiler = MaterializingSimulationTargetCompiler(
        {"materialize_policyengine": SnapMaterializer()}
    )

    compilation = compiler.compile_simulation_target_constraints(
        targets=[
            TargetSpec(
                name="snap_total",
                entity=EntityType.HOUSEHOLD,
                value=12.0,
                period=2024,
                measure="snap",
                aggregation=TargetAggregation.SUM,
                sim_modifiers=(
                    TargetSimulationModifier(
                        name="materialize_policyengine",
                        parameters={"variables": ["snap"]},
                    ),
                ),
            )
        ],
        entity_frames={EntityType.HOUSEHOLD: pd.DataFrame({"household_id": [1, 2]})},
        entity_weight_indexes={EntityType.HOUSEHOLD: pd.Series([0, 1])},
    )

    assert compilation.skipped_targets == ()
    assert len(compilation.constraints) == 1
    constraint = compilation.constraints[0]
    assert constraint.name == "snap_total"
    np.testing.assert_array_equal(constraint.weight_indexes, np.array([0]))
    np.testing.assert_allclose(constraint.coefficients, np.array([12.0]))


def test_materializing_simulation_compiler_materializes_filter_features():
    compiler = MaterializingSimulationTargetCompiler(
        {"materialize_policyengine": SnapMaterializer()}
    )

    compilation = compiler.compile_simulation_target_constraints(
        targets=[
            TargetSpec(
                name="snap_total",
                entity=EntityType.HOUSEHOLD,
                value=1.0,
                period=2024,
                aggregation=TargetAggregation.COUNT,
                filters=(TargetFilter(feature="snap", operator=">", value=0),),
                sim_modifiers=(
                    TargetSimulationModifier(
                        name="materialize_policyengine",
                        parameters={"variables": ["snap"]},
                    ),
                ),
            )
        ],
        entity_frames={EntityType.HOUSEHOLD: pd.DataFrame({"household_id": [1, 2]})},
        entity_weight_indexes={EntityType.HOUSEHOLD: pd.Series([0, 1])},
    )

    assert compilation.skipped_targets == ()
    constraint = compilation.constraints[0]
    np.testing.assert_array_equal(constraint.weight_indexes, np.array([0]))
    np.testing.assert_allclose(constraint.coefficients, np.array([1.0]))


def test_materializing_simulation_compiler_skips_missing_handler():
    compiler = MaterializingSimulationTargetCompiler({})

    compilation = compiler.compile_simulation_target_constraints(
        targets=[
            TargetSpec(
                name="snap_total",
                entity=EntityType.HOUSEHOLD,
                value=12.0,
                period=2024,
                measure="snap",
                sim_modifiers=(
                    TargetSimulationModifier(name="materialize_policyengine"),
                ),
            )
        ],
        entity_frames={EntityType.HOUSEHOLD: pd.DataFrame({"household_id": [1, 2]})},
        entity_weight_indexes={EntityType.HOUSEHOLD: pd.Series([0, 1])},
    )

    assert compilation.constraints == ()
    assert compilation.skipped_targets == (
        (
            "snap_total",
            "missing_simulation_modifier_handler:materialize_policyengine",
        ),
    )


def test_materializing_simulation_compiler_rejects_row_count_changes():
    class BadMaterializer:
        def apply_simulation_modifier(
            self,
            entity_frames,
            *,
            target,
            modifier,
            entity_weight_indexes,
        ):
            _ = (target, modifier, entity_weight_indexes)
            return {EntityType.HOUSEHOLD: entity_frames[EntityType.HOUSEHOLD].iloc[:1]}

    compiler = MaterializingSimulationTargetCompiler(
        {"materialize_policyengine": BadMaterializer()}
    )

    with pytest.raises(ValueError, match="changed row count"):
        compiler.compile_simulation_target_constraints(
            targets=[
                TargetSpec(
                    name="snap_total",
                    entity=EntityType.HOUSEHOLD,
                    value=12.0,
                    period=2024,
                    measure="snap",
                    sim_modifiers=(
                        TargetSimulationModifier(name="materialize_policyengine"),
                    ),
                )
            ],
            entity_frames={
                EntityType.HOUSEHOLD: pd.DataFrame({"household_id": [1, 2]})
            },
            entity_weight_indexes={EntityType.HOUSEHOLD: pd.Series([0, 1])},
        )
