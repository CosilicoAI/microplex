"""Tests for simulator-modified target compilation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microplex.core import EntityType
from microplex.targets import (
    MaterializedSimulationTargetCompiler,
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


class RecordingMaterializer:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def materialize_simulation_features(self, *, targets, entity_frames, modifiers):
        self.calls.append(
            {
                "targets": tuple(target.name for target in targets),
                "modifiers": tuple(modifier.name for modifier in modifiers),
                "entity_rows": {
                    entity.value: len(frame) for entity, frame in entity_frames.items()
                },
            }
        )
        return self.outputs[tuple(modifier.name for modifier in modifiers)]


def test__given_materialized_simulation_features__then_compiler_returns_constraints():
    materializer = RecordingMaterializer(
        {
            ("materialize_policyengine",): {
                EntityType.PERSON: pd.DataFrame({"snap": [0.0, 7.0]})
            }
        }
    )
    compiler = MaterializedSimulationTargetCompiler(materializer)
    target = TargetSpec(
        name="snap_total",
        entity=EntityType.PERSON,
        value=14.0,
        period=2024,
        measure="snap",
        aggregation="sum",
        sim_modifiers=(TargetSimulationModifier("materialize_policyengine"),),
    )

    result = compiler.compile_simulation_target_constraints(
        targets=(target,),
        entity_frames={EntityType.PERSON: pd.DataFrame({"age": [10, 20]})},
        entity_weight_indexes={EntityType.PERSON: np.array([0, 1])},
    )

    assert result.skipped_targets == ()
    assert len(result.constraints) == 1
    constraint = result.constraints[0]
    assert constraint.name == "snap_total"
    assert constraint.weight_indexes.tolist() == [1]
    assert constraint.coefficients.tolist() == [7.0]
    assert materializer.calls == [
        {
            "targets": ("snap_total",),
            "modifiers": ("materialize_policyengine",),
            "entity_rows": {"person": 2},
        }
    ]


def test__given_original_frame_filter__then_compiler_merges_simulated_columns():
    materializer = RecordingMaterializer(
        {
            ("materialize_policyengine",): {
                EntityType.PERSON: pd.DataFrame({"snap": [5.0, 7.0]})
            }
        }
    )
    compiler = MaterializedSimulationTargetCompiler(materializer)
    target = TargetSpec(
        name="california_snap_total",
        entity=EntityType.PERSON,
        value=10.0,
        period=2024,
        measure="snap",
        aggregation="sum",
        filters=(TargetFilter("state_fips", "==", "06"),),
        sim_modifiers=(TargetSimulationModifier("materialize_policyengine"),),
    )

    result = compiler.compile_simulation_target_constraints(
        targets=(target,),
        entity_frames={EntityType.PERSON: pd.DataFrame({"state_fips": ["06", "12"]})},
        entity_weight_indexes={EntityType.PERSON: np.array([0, 1])},
    )

    assert result.skipped_targets == ()
    assert result.constraints[0].weight_indexes.tolist() == [0]
    assert result.constraints[0].coefficients.tolist() == [5.0]


def test__given_different_modifier_sets__then_compiler_materializes_each_group():
    materializer = RecordingMaterializer(
        {
            ("materialize_policyengine",): {
                EntityType.PERSON: pd.DataFrame({"income_tax": [1.0]})
            },
            ("rerandomize_takeup", "materialize_policyengine"): {
                EntityType.PERSON: pd.DataFrame({"snap": [2.0]})
            },
        }
    )
    compiler = MaterializedSimulationTargetCompiler(materializer)

    result = compiler.compile_simulation_target_constraints(
        targets=(
            TargetSpec(
                name="income_tax_total",
                entity=EntityType.PERSON,
                value=1.0,
                period=2024,
                measure="income_tax",
                aggregation="sum",
                sim_modifiers=(TargetSimulationModifier("materialize_policyengine"),),
            ),
            TargetSpec(
                name="snap_total",
                entity=EntityType.PERSON,
                value=2.0,
                period=2024,
                measure="snap",
                aggregation="sum",
                sim_modifiers=(
                    TargetSimulationModifier("rerandomize_takeup"),
                    TargetSimulationModifier("materialize_policyengine"),
                ),
            ),
        ),
        entity_frames={EntityType.PERSON: pd.DataFrame({"person_id": [1]})},
        entity_weight_indexes={EntityType.PERSON: np.array([0])},
    )

    assert tuple(constraint.name for constraint in result.constraints) == (
        "income_tax_total",
        "snap_total",
    )
    assert [call["targets"] for call in materializer.calls] == [
        ("income_tax_total",),
        ("snap_total",),
    ]


def test__given_materializer_omits_required_feature__then_target_is_skipped():
    materializer = RecordingMaterializer(
        {
            ("materialize_policyengine",): {
                EntityType.PERSON: pd.DataFrame(index=range(2))
            }
        }
    )
    compiler = MaterializedSimulationTargetCompiler(materializer)
    target = TargetSpec(
        name="snap_total",
        entity=EntityType.PERSON,
        value=14.0,
        period=2024,
        measure="snap",
        aggregation="sum",
        sim_modifiers=(TargetSimulationModifier("materialize_policyengine"),),
    )

    result = compiler.compile_simulation_target_constraints(
        targets=(target,),
        entity_frames={EntityType.PERSON: pd.DataFrame({"person_id": [1, 2]})},
        entity_weight_indexes={EntityType.PERSON: np.array([0, 1])},
    )

    assert result.constraints == ()
    assert result.skipped_targets == (("snap_total", "missing_features:snap"),)
