from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

from scipy import sparse

from microplex.core import EntityType
from microplex.targets import (
    SPARSE_TARGET_MATRIX_CERTIFICATE_SCHEMA,
    FilterOperator,
    SparseTargetMatrix,
    TargetConstraintCompilationResult,
    TargetFilter,
    TargetReweightingConstraint,
    TargetSimulationModifier,
    TargetSpec,
    assemble_clone_sparse_target_matrix,
    assert_sparse_target_matrix_certificate,
    build_sparse_target_matrix_certificate,
    compile_sparse_target_matrix,
    target_constraints_to_sparse_matrix,
)


class MatrixSimulationCompiler:
    def compile_simulation_target_constraints(
        self,
        *,
        targets,
        entity_frames,
        entity_weight_indexes,
    ) -> TargetConstraintCompilationResult:
        target = targets[0]
        return TargetConstraintCompilationResult(
            constraints=(
                TargetReweightingConstraint(
                    name=target.name,
                    entity=target.entity,
                    weight_indexes=np.array([1]),
                    coefficients=np.array([7.0]),
                    target=target.value,
                ),
            )
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


def test_compile_sparse_target_matrix_uses_simulation_compiler() -> None:
    target_matrix = compile_sparse_target_matrix(
        targets=[
            TargetSpec(
                name="snap_after_takeup",
                entity=EntityType.PERSON,
                value=14.0,
                period=2024,
                measure="snap",
                aggregation="sum",
                sim_modifiers=(TargetSimulationModifier(name="rerandomize_takeup"),),
            )
        ],
        entity_frames={EntityType.PERSON: pd.DataFrame({"snap": [0.0, 0.0]})},
        entity_weight_indexes={EntityType.PERSON: np.array([0, 1])},
        n_weights=2,
        simulation_compiler=MatrixSimulationCompiler(),
    )

    assert target_matrix.names == ("snap_after_takeup",)
    np.testing.assert_array_equal(target_matrix.target_vector, np.array([14.0]))
    np.testing.assert_array_equal(target_matrix.matrix.toarray(), np.array([[0.0, 7.0]]))
    assert target_matrix.skipped_targets == ()


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


def test_sparse_target_matrix_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="duplicate target names"):
        SparseTargetMatrix(
            matrix=sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]])),
            target_vector=np.array([1.0, 2.0]),
            names=("target", "target"),
        )


def test_sparse_target_matrix_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="target_vector"):
        SparseTargetMatrix(
            matrix=sparse.csr_matrix(np.array([[1.0, 0.0]])),
            target_vector=np.array([np.inf]),
            names=("target",),
        )

    with pytest.raises(ValueError, match="matrix coefficients"):
        SparseTargetMatrix(
            matrix=sparse.csr_matrix(np.array([[np.nan, 0.0]])),
            target_vector=np.array([1.0]),
            names=("target",),
        )


def test_assemble_clone_sparse_target_matrix_offsets_clone_columns() -> None:
    clone_0 = SparseTargetMatrix(
        matrix=sparse.csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])),
        target_vector=np.array([10.0, 20.0]),
        names=("income", "count"),
        metadata=({"family": "income"}, {"family": "demo"}),
    )
    clone_1 = SparseTargetMatrix(
        matrix=sparse.csr_matrix(np.array([[0.0, 4.0, 0.0], [5.0, 0.0, 6.0]])),
        target_vector=np.array([10.0, 20.0]),
        names=("income", "count"),
        metadata=({"family": "income"}, {"family": "demo"}),
    )

    assembled = assemble_clone_sparse_target_matrix(
        {0: clone_0, 1: clone_1},
        n_records=3,
        n_clones=2,
    )

    assert assembled.matrix.shape == (2, 6)
    np.testing.assert_array_equal(
        assembled.matrix.toarray(),
        np.array(
            [
                [1.0, 0.0, 2.0, 0.0, 4.0, 0.0],
                [0.0, 3.0, 0.0, 5.0, 0.0, 6.0],
            ]
        ),
    )
    np.testing.assert_array_equal(assembled.target_vector, np.array([10.0, 20.0]))
    assert assembled.names == ("income", "count")


def test_assemble_clone_sparse_target_matrix_rejects_missing_clone() -> None:
    clone_0 = SparseTargetMatrix(
        matrix=sparse.csr_matrix((1, 2)),
        target_vector=np.array([1.0]),
        names=("target",),
    )

    with pytest.raises(ValueError, match="missing=\\[1\\]"):
        assemble_clone_sparse_target_matrix({0: clone_0}, n_records=2, n_clones=2)


def test_assemble_clone_sparse_target_matrix_rejects_mismatched_names() -> None:
    clone_0 = SparseTargetMatrix(
        matrix=sparse.csr_matrix((1, 2)),
        target_vector=np.array([1.0]),
        names=("target",),
    )
    clone_1 = SparseTargetMatrix(
        matrix=sparse.csr_matrix((1, 2)),
        target_vector=np.array([1.0]),
        names=("other",),
    )

    with pytest.raises(ValueError, match="target names"):
        assemble_clone_sparse_target_matrix(
            {0: clone_0, 1: clone_1},
            n_records=2,
            n_clones=2,
        )


def test_assemble_clone_sparse_target_matrix_rejects_wrong_record_width() -> None:
    clone_0 = SparseTargetMatrix(
        matrix=sparse.csr_matrix((1, 2)),
        target_vector=np.array([1.0]),
        names=("target",),
    )
    clone_1 = SparseTargetMatrix(
        matrix=sparse.csr_matrix((1, 3)),
        target_vector=np.array([1.0]),
        names=("target",),
    )

    with pytest.raises(ValueError, match="expected n_records"):
        assemble_clone_sparse_target_matrix(
            {0: clone_0, 1: clone_1},
            n_records=2,
            n_clones=2,
        )


def test_sparse_target_matrix_certificate_round_trips_and_matches() -> None:
    target_matrix = SparseTargetMatrix(
        matrix=sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]])),
        target_vector=np.array([10.0, 20.0]),
        names=("income", "count"),
        metadata=({"family": "income"}, {"family": "demo"}),
        skipped_targets=(("unsupported", "missing_features:x"),),
    )

    certificate = build_sparse_target_matrix_certificate(
        target_matrix,
        extra={"baseline": "production_ecps"},
    )
    payload = certificate.to_dict()

    assert payload["schema_version"] == SPARSE_TARGET_MATRIX_CERTIFICATE_SCHEMA
    assert payload["n_targets"] == 2
    assert payload["n_weights"] == 2
    assert payload["nnz"] == 2
    assert payload["extra"] == {"baseline": "production_ecps"}
    assert_sparse_target_matrix_certificate(target_matrix, payload)
    target_matrix.assert_matches_certificate(payload)


def test_sparse_target_matrix_certificate_is_stable_across_csr_index_order() -> None:
    unsorted = sparse.csr_matrix(
        (
            np.array([2.0, 1.0]),
            np.array([1, 0]),
            np.array([0, 2]),
        ),
        shape=(1, 2),
    )
    sorted_matrix = sparse.csr_matrix(np.array([[1.0, 2.0]]))
    left = SparseTargetMatrix(
        matrix=unsorted,
        target_vector=np.array([3.0]),
        names=("target",),
    )
    right = SparseTargetMatrix(
        matrix=sorted_matrix,
        target_vector=np.array([3.0]),
        names=("target",),
    )

    assert left.certificate().to_dict() == right.certificate().to_dict()


def test_sparse_target_matrix_certificate_rejects_changed_values() -> None:
    target_matrix = SparseTargetMatrix(
        matrix=sparse.csr_matrix(np.array([[1.0, 0.0]])),
        target_vector=np.array([1.0]),
        names=("target",),
    )
    certificate = target_matrix.certificate()
    changed = SparseTargetMatrix(
        matrix=sparse.csr_matrix(np.array([[2.0, 0.0]])),
        target_vector=np.array([1.0]),
        names=("target",),
    )

    with pytest.raises(ValueError, match="matrix_values_sha256"):
        assert_sparse_target_matrix_certificate(changed, certificate)


def test_sparse_target_matrix_certificate_rejects_changed_names() -> None:
    target_matrix = SparseTargetMatrix(
        matrix=sparse.csr_matrix(np.array([[1.0, 0.0]])),
        target_vector=np.array([1.0]),
        names=("target",),
    )
    certificate = target_matrix.certificate()
    changed = SparseTargetMatrix(
        matrix=sparse.csr_matrix(np.array([[1.0, 0.0]])),
        target_vector=np.array([1.0]),
        names=("renamed",),
    )

    with pytest.raises(ValueError, match="names_sha256"):
        assert_sparse_target_matrix_certificate(changed, certificate)
