"""Tests for optional PolicyEngine-US runtime adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microplex.core import EntityType
from microplex.policyengine_us import (
    POLICYENGINE_US_TAKEUP_HANDLER,
    PolicyEngineUSMicrosimulationMaterializer,
    PolicyEngineUSRuntimeAdapter,
    SeededPolicyEngineUSTakeupRerandomizer,
)
from microplex.runtime_operations import apply_runtime_variable_operations
from microplex.spec import load_spec_dict
from microplex.targets import TargetFilter, TargetSimulationModifier, TargetSpec


def _spec_dict_with_variables(variables: dict) -> dict:
    return {
        "meta": {"country": "us", "model_year": 2024},
        "sources": {"cps": {"dataset": "cps_2024", "role": "spine"}},
        "spine": {
            "base": "cps",
            "method": "support_spine",
            "support": {"seed": 0},
            "halves": [
                {"name": "cps_keep", "keep": "all"},
                {"name": "synthetic", "strip_to": ["demographics"]},
            ],
        },
        "variables": variables,
    }


def _runtime_variable(entity: str, operation: str, handler: str) -> dict:
    return {
        "entity": entity,
        "mp_spec": {
            "method": "runtime test",
            "operation": {
                "kind": operation,
                "handler": handler,
            },
        },
    }


class RecordingPolicyEngineMaterializer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def materialize_policyengine_variables(
        self,
        entity_frames,
        *,
        variables_by_entity,
        period,
    ):
        self.calls.append(
            {
                "variables_by_entity": {
                    entity.value: tuple(variables)
                    for entity, variables in variables_by_entity.items()
                },
                "period": period,
                "has_snap_takeup": "takes_up_snap_if_eligible"
                in entity_frames.get(EntityType.SPM_UNIT, pd.DataFrame()).columns,
            }
        )
        frames = {entity: frame.copy() for entity, frame in entity_frames.items()}
        for entity, variables in variables_by_entity.items():
            frame = frames[entity].copy()
            for variable in variables:
                if variable == "snap" and "takes_up_snap_if_eligible" in frame:
                    frame[variable] = np.where(
                        frame["takes_up_snap_if_eligible"],
                        10.0,
                        0.0,
                    )
                else:
                    frame[variable] = np.arange(len(frame), dtype=float) + 1.0
            frames[entity] = frame
        return frames


def test_runtime_takeup_handler_materializes_declared_spec_variables() -> None:
    spec = load_spec_dict(
        _spec_dict_with_variables(
            {
                "takes_up_snap_if_eligible": _runtime_variable(
                    "spm_unit",
                    "rerandomize_takeup",
                    POLICYENGINE_US_TAKEUP_HANDLER,
                ),
                "takes_up_eitc": _runtime_variable(
                    "tax_unit",
                    "rerandomize_takeup",
                    POLICYENGINE_US_TAKEUP_HANDLER,
                ),
            }
        )
    )
    adapter = PolicyEngineUSRuntimeAdapter(
        takeup_rerandomizer=SeededPolicyEngineUSTakeupRerandomizer(
            rates={
                "takes_up_snap_if_eligible": 1.0,
                "takes_up_eitc": 0.0,
            }
        )
    )

    frame, results = apply_runtime_variable_operations(
        pd.DataFrame({"row_id": [1, 2, 3]}),
        spec=spec,
        handlers=adapter.runtime_variable_operation_handlers(),
    )

    assert frame["takes_up_snap_if_eligible"].tolist() == [True, True, True]
    assert frame["takes_up_eitc"].tolist() == [False, False, False]
    assert [(result.handler, result.variables) for result in results] == [
        (
            POLICYENGINE_US_TAKEUP_HANDLER,
            ("takes_up_snap_if_eligible", "takes_up_eitc"),
        )
    ]


def test_simulation_materializer_materializes_only_missing_target_features() -> None:
    materializer = RecordingPolicyEngineMaterializer()
    adapter = PolicyEngineUSRuntimeAdapter(materializer=materializer)
    target = TargetSpec(
        name="state_snap_total",
        entity=EntityType.SPM_UNIT,
        value=100.0,
        period=2024,
        measure="snap",
        filters=(TargetFilter("state_fips", "==", "06"),),
        sim_modifiers=(
            TargetSimulationModifier(
                "materialize_policyengine",
                {"model": "policyengine-us"},
            ),
        ),
    )

    frames = adapter.materialize_simulation_features(
        targets=(target,),
        entity_frames={EntityType.SPM_UNIT: pd.DataFrame({"state_fips": ["06", "12"]})},
        modifiers=target.sim_modifiers,
    )

    assert frames[EntityType.SPM_UNIT]["snap"].tolist() == [1.0, 2.0]
    assert materializer.calls == [
        {
            "variables_by_entity": {"spm_unit": ("snap",)},
            "period": 2024,
            "has_snap_takeup": False,
        }
    ]


def test_simulation_materializer_runs_takeup_before_policyengine() -> None:
    materializer = RecordingPolicyEngineMaterializer()
    adapter = PolicyEngineUSRuntimeAdapter(
        materializer=materializer,
        takeup_rerandomizer=SeededPolicyEngineUSTakeupRerandomizer({"snap": 1.0}),
    )
    target = TargetSpec(
        name="snap_total",
        entity=EntityType.SPM_UNIT,
        value=100.0,
        period=2024,
        measure="snap",
        sim_modifiers=(
            TargetSimulationModifier("rerandomize_takeup", {"program": "snap"}),
            TargetSimulationModifier(
                "materialize_policyengine",
                {"model": "policyengine-us"},
            ),
        ),
    )

    frames = adapter.materialize_simulation_features(
        targets=(target,),
        entity_frames={EntityType.SPM_UNIT: pd.DataFrame({"spm_unit_id": [1, 2]})},
        modifiers=target.sim_modifiers,
    )

    assert frames[EntityType.SPM_UNIT]["takes_up_snap_if_eligible"].tolist() == [
        True,
        True,
    ]
    assert frames[EntityType.SPM_UNIT]["snap"].tolist() == [10.0, 10.0]
    assert materializer.calls[0]["has_snap_takeup"] is True


def test_simulation_materializer_rejects_non_us_model_modifier() -> None:
    adapter = PolicyEngineUSRuntimeAdapter(
        materializer=RecordingPolicyEngineMaterializer()
    )
    target = TargetSpec(
        name="snap_total",
        entity=EntityType.SPM_UNIT,
        value=100.0,
        period=2024,
        measure="snap",
        sim_modifiers=(
            TargetSimulationModifier(
                "materialize_policyengine",
                {"model": "policyengine-uk"},
            ),
        ),
    )

    with pytest.raises(ValueError, match="model='policyengine-us'"):
        adapter.materialize_simulation_features(
            targets=(target,),
            entity_frames={EntityType.SPM_UNIT: pd.DataFrame({"spm_unit_id": [1]})},
            modifiers=target.sim_modifiers,
        )


def test_microsimulation_materializer_uses_injected_factories() -> None:
    captured: dict = {}

    def dataset_factory(entity_frames, period):
        captured["dataset_period"] = period
        captured["dataset_entities"] = tuple(
            sorted(entity.value for entity in entity_frames)
        )
        return {"entity_frames": entity_frames, "period": period}

    class FakeSimulation:
        def __init__(self, dataset) -> None:
            captured["simulation_dataset"] = dataset

        def calc(self, variable, *, period, map_to, use_weights):
            captured["calculation"] = (variable, period, map_to, use_weights)
            return np.array([3.0, 4.0])

    materializer = PolicyEngineUSMicrosimulationMaterializer(
        dataset_factory=dataset_factory,
        simulation_factory=FakeSimulation,
    )

    frames = materializer.materialize_policyengine_variables(
        {EntityType.TAX_UNIT: pd.DataFrame({"tax_unit_id": [10, 20]})},
        variables_by_entity={EntityType.TAX_UNIT: ("income_tax",)},
        period=2024,
    )

    assert frames[EntityType.TAX_UNIT]["income_tax"].tolist() == [3.0, 4.0]
    assert captured["dataset_period"] == 2024
    assert captured["dataset_entities"] == ("tax_unit",)
    assert captured["calculation"] == ("income_tax", 2024, "tax_unit", False)


def test_microsimulation_materializer_requires_core_us_entities() -> None:
    materializer = PolicyEngineUSMicrosimulationMaterializer()

    with pytest.raises(ValueError, match="missing \\['person', 'household'\\]"):
        materializer.materialize_policyengine_variables(
            {EntityType.TAX_UNIT: pd.DataFrame({"tax_unit_id": [10]})},
            variables_by_entity={EntityType.TAX_UNIT: ("income_tax",)},
            period=2024,
        )
