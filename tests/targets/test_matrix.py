from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

from scipy import sparse

from microplex.core import EntityType
from microplex.targets import (
    FilterOperator,
    SparseTargetMatrix,
    TargetFilter,
    TargetReweightingConstraint,
    TargetSpec,
    compile_sparse_target_matrix,
    target_constraints_to_sparse_matrix,
)


def test_compile_sparse_target_matrix_emits_csr_target_surface() -> None:
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
    target_matrix = compile_sparse_target_matrix(
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
                metadata={"family": "demographics"},
            ),
            TargetSpec(
                name="employment_sum",
                entity=EntityType.PERSON,
                value=120.0,
                period=2024,
                measure="employment_income",
                aggregation="sum",
                filters=(TargetFilter("local_authority_code", FilterOperator.EQ, "B"),),
                metadata={"family": "income"},
            ),
        ],
        entity_frames={EntityType.PERSON: person, EntityType.HOUSEHOLD: household},
        entity_weight_indexes={
            EntityType.PERSON: person["person_household_id"].map(household_index),
            EntityType.HOUSEHOLD: household_index.reindex(household["household_id"]).to_numpy(),
        },
        n_weights=len(household),
    )

    assert isinstance(target_matrix, SparseTargetMatrix)
    assert sparse.isspmatrix_csr(target_matrix.matrix)
    assert target_matrix.matrix.shape == (2, 2)
    np.testing.assert_array_equal(
        target_matrix.matrix.toarray(),
        np.array([[2.0, 0.0], [0.0, 100.0]]),
    )
    np.testing.assert_array_equal(target_matrix.target_vector, np.array([4.0, 120.0]))
    assert target_matrix.names == ("age_band_count", "employment_sum")
    assert target_matrix.metadata == ({"family": "demographics"}, {"family": "income"})
    assert target_matrix.skipped_targets == ()
    assert target_matrix.n_targets == 2
    assert target_matrix.n_weights == 2


def test_compile_sparse_target_matrix_preserves_skipped_targets_and_shape() -> None:
    target_matrix = compile_sparse_target_matrix(
        targets=[
            TargetSpec(
                name="missing_measure",
                entity=EntityType.PERSON,
                value=1.0,
                period=2024,
                measure="missing_income",
                aggregation="sum",
            )
        ],
        entity_frames={EntityType.PERSON: pd.DataFrame({"person_id": [1, 2]})},
        entity_weight_indexes={EntityType.PERSON: np.array([0, 1])},
        n_weights=2,
    )

    assert target_matrix.matrix.shape == (0, 2)
    assert target_matrix.target_vector.tolist() == []
    assert target_matrix.names == ()
    assert target_matrix.skipped_targets == (
        ("missing_measure", "missing_features:missing_income"),
    )


def test_target_constraints_to_sparse_matrix_preserves_trailing_zero_columns() -> None:
    target_matrix = target_constraints_to_sparse_matrix(
        [
            TargetReweightingConstraint(
                name="first_two",
                entity=EntityType.HOUSEHOLD,
                weight_indexes=np.array([0, 1]),
                coefficients=np.array([1.0, 2.0]),
                target=3.0,
            )
        ],
        n_weights=4,
    )

    assert target_matrix.matrix.shape == (1, 4)
    np.testing.assert_array_equal(
        target_matrix.matrix.toarray(),
        np.array([[1.0, 2.0, 0.0, 0.0]]),
    )


def test_target_constraints_to_sparse_matrix_sums_duplicate_entries() -> None:
    target_matrix = target_constraints_to_sparse_matrix(
        [
            TargetReweightingConstraint(
                name="dupe",
                entity=EntityType.HOUSEHOLD,
                weight_indexes=np.array([1, 1]),
                coefficients=np.array([1.5, 2.5]),
                target=4.0,
            )
        ],
        n_weights=2,
    )

    assert target_matrix.matrix[0, 1] == 4.0


def test_target_constraints_to_sparse_matrix_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError, match="weight_indexes"):
        target_constraints_to_sparse_matrix(
            [
                TargetReweightingConstraint(
                    name="bad",
                    entity=EntityType.HOUSEHOLD,
                    weight_indexes=np.array([2]),
                    coefficients=np.array([1.0]),
                    target=1.0,
                )
            ],
            n_weights=2,
        )


def test_sparse_target_matrix_rejects_misaligned_names() -> None:
    with pytest.raises(ValueError, match="names"):
        SparseTargetMatrix(
            matrix=sparse.csr_matrix((1, 2)),
            target_vector=np.array([1.0]),
            names=(),
        )
