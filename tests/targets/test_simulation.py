from __future__ import annotations

import numpy as np
import pandas as pd

from microplex.core import EntityType
from microplex.targets import (
    MaterializedSimulationTargetCompiler,
    TargetFilter,
    TargetSimulationModifier,
    TargetSpec,
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
