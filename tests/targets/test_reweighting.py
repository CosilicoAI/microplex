from __future__ import annotations

import numpy as np
import pandas as pd

from microplex.core import EntityType
from microplex.targets import (
    EntityTableBinding,
    EntityTableBundle,
    FilterOperator,
    TargetFilter,
    TargetSimulationModifier,
    TargetSpec,
    apply_filter,
    compile_target_reweighting_constraints,
    constraint_abs_relative_error,
    reweight_entity_table_bundle_targets,
    reweight_to_target_constraints,
)


def test_compile_target_reweighting_constraints_groups_to_shared_weight_vector():
    person = pd.DataFrame(
        {
            "person_household_id": [10, 10, 20],
            "age": [5, 8, 30],
            "employment_income": [0.0, 0.0, 100.0],
            "local_authority_code": ["A", "A", "B"],
        }
    )
    household = pd.DataFrame({"household_id": [10, 20]})
    household_index = pd.Series(np.arange(len(household)), index=household["household_id"])

    targets = [
        TargetSpec(
            name="age_band_count",
            entity=EntityType.PERSON,
            value=4.0,
            period=2024,
            aggregation="count",
            filters=(
                TargetFilter("local_authority_code", FilterOperator.EQ, "A"),
                TargetFilter("age", FilterOperator.GTE, 0),
                TargetFilter("age", FilterOperator.LT, 10),
            ),
        ),
        TargetSpec(
            name="employment_sum",
            entity=EntityType.PERSON,
            value=120.0,
            period=2024,
            measure="employment_income",
            aggregation="sum",
            filters=(TargetFilter("local_authority_code", FilterOperator.EQ, "B"),),
        ),
    ]

    compilation = compile_target_reweighting_constraints(
        targets=targets,
        entity_frames={
            EntityType.PERSON: person,
            EntityType.HOUSEHOLD: household,
        },
        entity_weight_indexes={
            EntityType.PERSON: person["person_household_id"].map(household_index),
            EntityType.HOUSEHOLD: household_index.reindex(household["household_id"]).to_numpy(),
        },
    )

    assert len(compilation.constraints) == 2
    count_constraint = compilation.constraints[0]
    sum_constraint = compilation.constraints[1]
    assert count_constraint.weight_indexes.tolist() == [0]
    assert count_constraint.coefficients.tolist() == [2.0]
    assert sum_constraint.weight_indexes.tolist() == [1]
    assert sum_constraint.coefficients.tolist() == [100.0]


def test_reweight_to_target_constraints_hits_simple_targets():
    person = pd.DataFrame(
        {
            "person_household_id": [10, 10, 20],
            "age": [5, 8, 30],
            "local_authority_code": ["A", "A", "B"],
        }
    )
    household = pd.DataFrame({"household_id": [10, 20]})
    household_index = pd.Series(np.arange(len(household)), index=household["household_id"])
    targets = [
        TargetSpec(
            name="age_band_count",
            entity=EntityType.PERSON,
            value=4.0,
            period=2024,
            aggregation="count",
            filters=(
                TargetFilter("local_authority_code", FilterOperator.EQ, "A"),
                TargetFilter("age", FilterOperator.GTE, 0),
                TargetFilter("age", FilterOperator.LT, 10),
            ),
        )
    ]

    compilation = compile_target_reweighting_constraints(
        targets=targets,
        entity_frames={EntityType.PERSON: person},
        entity_weight_indexes={
            EntityType.PERSON: person["person_household_id"].map(household_index),
        },
    )
    weights, diagnostics = reweight_to_target_constraints(
        np.array([1.0, 1.0]),
        constraints=compilation.constraints,
        max_iter=4,
        tol=1e-6,
    )

    assert weights.tolist() == [2.0, 1.0]
    assert diagnostics.constraint_count == 1
    assert diagnostics.mean_abs_relative_error == 0.0


def test_reweight_to_target_constraints_shrinks_mean_residual_toward_zero():
    person = pd.DataFrame({"income": [0.0, 1.2]})
    compilation = compile_target_reweighting_constraints(
        targets=[
            TargetSpec(
                name="mean_income",
                entity=EntityType.PERSON,
                value=0.5,
                period=2024,
                measure="income",
                aggregation="mean",
            )
        ],
        entity_frames={EntityType.PERSON: person},
        entity_weight_indexes={EntityType.PERSON: np.array([0, 1])},
    )

    constraint = compilation.constraints[0]
    initial_weights = np.array([1.0, 1.0])
    initial_residual = float(np.dot(initial_weights, constraint.coefficients))

    weights, diagnostics = reweight_to_target_constraints(
        initial_weights,
        constraints=compilation.constraints,
        max_iter=1,
        tol=1e-6,
    )

    updated_residual = float(np.dot(weights, constraint.coefficients))

    assert abs(updated_residual) < abs(initial_residual)
    assert diagnostics.mean_abs_relative_error < abs(initial_residual)


def test_reweight_to_target_constraints_does_not_report_converged_when_positive_target_has_nonpositive_current():
    constraint = compile_target_reweighting_constraints(
        targets=[
            TargetSpec(
                name="income_sum",
                entity=EntityType.PERSON,
                value=10.0,
                period=2024,
                measure="income",
                aggregation="sum",
            )
        ],
        entity_frames={EntityType.PERSON: pd.DataFrame({"income": [1.0]})},
        entity_weight_indexes={EntityType.PERSON: np.array([0])},
    ).constraints[0]

    weights, diagnostics = reweight_to_target_constraints(
        np.array([0.0]),
        constraints=(constraint,),
        max_iter=1,
        tol=1e-6,
    )

    assert weights.tolist() == [0.0]
    assert diagnostics.converged is False


def test_constraint_abs_relative_error_uses_finite_zero_target_denominator():
    constraint = compile_target_reweighting_constraints(
        targets=[
            TargetSpec(
                name="zero_target_sum",
                entity=EntityType.PERSON,
                value=0.0,
                period=2024,
                measure="employment_income",
                aggregation="sum",
            )
        ],
        entity_frames={
            EntityType.PERSON: pd.DataFrame({"employment_income": [2.0]}),
        },
        entity_weight_indexes={EntityType.PERSON: np.array([0])},
    ).constraints[0]

    assert constraint_abs_relative_error(constraint, np.array([1.0])) == 2.0


def test_simulation_modifier_targets_are_skipped_until_simulator_compiler_exists():
    compilation = compile_target_reweighting_constraints(
        targets=[
            TargetSpec(
                name="snap_after_takeup",
                entity=EntityType.PERSON,
                value=10.0,
                period=2024,
                measure="snap",
                aggregation="sum",
                sim_modifiers=(
                    TargetSimulationModifier(
                        name="rerandomize_takeup",
                        parameters={"program": "snap"},
                    ),
                ),
            )
        ],
        entity_frames={EntityType.PERSON: pd.DataFrame({"snap": [1.0, 2.0]})},
        entity_weight_indexes={EntityType.PERSON: np.array([0, 1])},
    )

    assert compilation.constraints == ()
    assert compilation.skipped_targets == (
        ("snap_after_takeup", "requires_simulation_modifiers:rerandomize_takeup"),
    )


def test_apply_filter_preserves_numeric_equality_semantics():
    values = pd.Series([1, 1.0, 2, True, None], dtype="object")

    mask = apply_filter(values, FilterOperator.EQ, 1)

    assert mask.tolist() == [True, True, False, True, False]


def test_entity_table_bundle_maps_weight_indexes_and_syncs_dependent_weights():
    bundle = EntityTableBundle(
        weight_entity=EntityType.HOUSEHOLD,
        weight_column="household_weight",
        bindings={
            EntityType.HOUSEHOLD: EntityTableBinding(
                frame=pd.DataFrame(
                    {
                        "household_id": [10, 20],
                        "household_weight": [1.0, 1.0],
                    }
                ),
                id_column="household_id",
            ),
            EntityType.PERSON: EntityTableBinding(
                frame=pd.DataFrame(
                    {
                        "person_id": [1, 2, 3],
                        "person_household_id": [10, 10, 20],
                        "weight": [1.0, 1.0, 1.0],
                    }
                ),
                id_column="person_id",
                weight_link_column="person_household_id",
                synced_weight_column="weight",
            ),
        },
    )

    indexes = bundle.entity_weight_indexes()
    assert indexes[EntityType.HOUSEHOLD].tolist() == [0, 1]
    assert indexes[EntityType.PERSON].tolist() == [0, 0, 1]

    updated = bundle.with_updated_weights(np.array([2.0, 1.0]))

    assert updated.table_for(EntityType.HOUSEHOLD)["household_weight"].tolist() == [2.0, 1.0]
    assert updated.table_for(EntityType.PERSON)["weight"].tolist() == [2.0, 2.0, 1.0]


def test_reweight_entity_table_bundle_targets_updates_bundle_in_one_step():
    bundle = EntityTableBundle(
        weight_entity=EntityType.HOUSEHOLD,
        weight_column="household_weight",
        bindings={
            EntityType.HOUSEHOLD: EntityTableBinding(
                frame=pd.DataFrame(
                    {
                        "household_id": [10, 20],
                        "household_weight": [1.0, 1.0],
                    }
                ),
                id_column="household_id",
            ),
            EntityType.PERSON: EntityTableBinding(
                frame=pd.DataFrame(
                    {
                        "person_id": [1, 2, 3],
                        "person_household_id": [10, 10, 20],
                        "weight": [1.0, 1.0, 1.0],
                        "age": [5, 8, 30],
                        "local_authority_code": ["A", "A", "B"],
                    }
                ),
                id_column="person_id",
                weight_link_column="person_household_id",
                synced_weight_column="weight",
            ),
        },
    )

    result = reweight_entity_table_bundle_targets(
        bundle,
        targets=[
            TargetSpec(
                name="age_band_count",
                entity=EntityType.PERSON,
                value=4.0,
                period=2024,
                aggregation="count",
                filters=(
                    TargetFilter("local_authority_code", FilterOperator.EQ, "A"),
                    TargetFilter("age", FilterOperator.GTE, 0),
                    TargetFilter("age", FilterOperator.LT, 10),
                ),
            )
        ],
    )

    assert result.bundle.table_for(EntityType.HOUSEHOLD)["household_weight"].tolist() == [2.0, 1.0]
    assert result.bundle.table_for(EntityType.PERSON)["weight"].tolist() == [2.0, 2.0, 1.0]
    assert result.compilation.skipped_targets == ()
