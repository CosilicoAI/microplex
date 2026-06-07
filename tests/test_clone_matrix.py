"""Sparse clone-column calibration matrix assembly tests."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from scipy import sparse

from microplex.calibration.clone_matrix import (
    CloneMatrixBlock,
    assemble_clone_csr,
    clone_column_indices,
)


class TestCloneColumnIndices:
    def test_uses_ecps_column_layout(self) -> None:
        cols = clone_column_indices(2, [0, 3], n_records=4, n_clones=3)
        np.testing.assert_array_equal(cols, np.array([8, 11]))

    def test_rejects_clone_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            clone_column_indices(3, [0], n_records=4, n_clones=3)

    def test_rejects_record_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="record_indices"):
            clone_column_indices(0, [4], n_records=4, n_clones=3)

    def test_rejects_non_integer_records(self) -> None:
        with pytest.raises(ValueError, match="integer"):
            clone_column_indices(0, [1.5], n_records=4, n_clones=3)


class TestAssembleCloneCsr:
    def test_assembles_sparse_matrix_with_clone_columns(self) -> None:
        matrix = assemble_clone_csr(
            [
                CloneMatrixBlock(
                    clone_idx=0,
                    row_indices=[0, 1],
                    record_indices=[1, 3],
                    values=[2.0, 4.0],
                ),
                CloneMatrixBlock(
                    clone_idx=2,
                    row_indices=[1, 0],
                    record_indices=[0, 3],
                    values=[5.0, 7.0],
                ),
            ],
            n_targets=2,
            n_records=4,
            n_clones=3,
        )

        assert sparse.isspmatrix_csr(matrix)
        assert matrix.shape == (2, 12)
        dense = matrix.toarray()
        assert dense[0, 1] == 2.0
        assert dense[1, 3] == 4.0
        assert dense[1, 8] == 5.0
        assert dense[0, 11] == 7.0

    def test_empty_blocks_return_empty_sparse_shape(self) -> None:
        matrix = assemble_clone_csr([], n_targets=3, n_records=5, n_clones=2)
        assert sparse.isspmatrix_csr(matrix)
        assert matrix.shape == (3, 10)
        assert matrix.nnz == 0

    def test_duplicate_entries_sum_when_converted_to_csr(self) -> None:
        matrix = assemble_clone_csr(
            [
                CloneMatrixBlock(
                    clone_idx=1,
                    row_indices=[0, 0],
                    record_indices=[2, 2],
                    values=[1.25, 2.75],
                )
            ],
            n_targets=1,
            n_records=4,
            n_clones=2,
        )
        assert matrix[0, 6] == 4.0

    def test_rejects_row_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="row_indices"):
            assemble_clone_csr(
                [
                    CloneMatrixBlock(
                        clone_idx=0,
                        row_indices=[2],
                        record_indices=[0],
                        values=[1.0],
                    )
                ],
                n_targets=2,
                n_records=4,
                n_clones=2,
            )

    def test_rejects_mismatched_part_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            assemble_clone_csr(
                [
                    CloneMatrixBlock(
                        clone_idx=0,
                        row_indices=[0],
                        record_indices=[0, 1],
                        values=[1.0],
                    )
                ],
                n_targets=2,
                n_records=4,
                n_clones=2,
            )

    def test_rejects_nonfinite_values(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            assemble_clone_csr(
                [
                    CloneMatrixBlock(
                        clone_idx=0,
                        row_indices=[0],
                        record_indices=[0],
                        values=[np.nan],
                    )
                ],
                n_targets=2,
                n_records=4,
                n_clones=2,
            )
