"""Sparse target matrix compilation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from microplex.calibration.clone_matrix import CloneMatrixBlock, assemble_clone_csr
from microplex.core import EntityType
from microplex.targets.reweighting import (
    TargetReweightingConstraint,
    compile_target_reweighting_constraints,
)
from microplex.targets.spec import TargetSpec


@dataclass(frozen=True)
class SparseTargetMatrix:
    """A compiled sparse linear target surface over one weight vector."""

    matrix: sparse.csr_matrix
    target_vector: np.ndarray
    names: tuple[str, ...]
    metadata: tuple[dict[str, Any], ...] = ()
    skipped_targets: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        matrix = self.matrix.tocsr()
        target_vector = np.asarray(self.target_vector, dtype=float)
        names = tuple(self.names)
        metadata = tuple(dict(item) for item in self.metadata)
        skipped_targets = tuple(tuple(item) for item in self.skipped_targets)

        if target_vector.ndim != 1:
            raise ValueError("target_vector must be one-dimensional.")
        if matrix.shape[0] != len(target_vector):
            raise ValueError("matrix rows must align to target_vector.")
        if len(names) != len(target_vector):
            raise ValueError("names must align to target_vector.")
        if metadata and len(metadata) != len(target_vector):
            raise ValueError("metadata must align to target_vector.")
        if not metadata:
            metadata = tuple({} for _ in names)

        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "target_vector", target_vector)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "skipped_targets", skipped_targets)

    @property
    def n_targets(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_weights(self) -> int:
        return self.matrix.shape[1]


def target_constraints_to_sparse_matrix(
    constraints: Sequence[TargetReweightingConstraint],
    *,
    n_weights: int,
    skipped_targets: Sequence[tuple[str, str]] = (),
    dtype: Any = np.float64,
) -> SparseTargetMatrix:
    """Compile target constraints to a CSR matrix and target vector.

    Args:
        constraints: Linear target constraints aligned to one shared weight
            vector.
        n_weights: Exact length of the shared weight vector. This is required
            so zero-support or trailing-zero columns cannot silently shrink the
            calibration surface.
        skipped_targets: Optional skipped-target diagnostics to carry forward.
        dtype: Numeric dtype for matrix coefficients and target values.
    """
    if n_weights < 0:
        raise ValueError(f"n_weights must be non-negative; got {n_weights}.")

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    target_values: list[float] = []
    names: list[str] = []
    metadata: list[dict[str, Any]] = []

    for row_idx, constraint in enumerate(constraints):
        indexes = np.asarray(constraint.weight_indexes, dtype=np.int64)
        coefficients = np.asarray(constraint.coefficients, dtype=dtype)
        if indexes.ndim != 1 or coefficients.ndim != 1:
            raise ValueError("constraint indexes and coefficients must be one-dimensional.")
        if len(indexes) != len(coefficients):
            raise ValueError("constraint indexes and coefficients must have the same length.")
        if indexes.size and ((indexes < 0).any() or (indexes >= n_weights).any()):
            raise ValueError(
                "constraint weight_indexes must be within [0, n_weights); "
                f"got min={indexes.min()}, max={indexes.max()}, n_weights={n_weights}."
            )
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("constraint coefficients must be finite.")
        if not np.isfinite(float(constraint.target)):
            raise ValueError("constraint target must be finite.")

        active = coefficients != 0.0
        row_parts.append(np.full(int(active.sum()), row_idx, dtype=np.int64))
        col_parts.append(indexes[active])
        value_parts.append(coefficients[active])
        target_values.append(float(constraint.target))
        names.append(constraint.name)
        metadata.append(dict(constraint.metadata))

    shape = (len(names), n_weights)
    if row_parts:
        rows = np.concatenate(row_parts)
        cols = np.concatenate(col_parts)
        values = np.concatenate(value_parts).astype(dtype, copy=False)
        matrix = sparse.coo_matrix((values, (rows, cols)), shape=shape).tocsr()
    else:
        matrix = sparse.csr_matrix(shape, dtype=dtype)

    return SparseTargetMatrix(
        matrix=matrix,
        target_vector=np.asarray(target_values, dtype=dtype),
        names=tuple(names),
        metadata=tuple(metadata),
        skipped_targets=tuple(skipped_targets),
    )


def compile_sparse_target_matrix(
    *,
    targets: list[TargetSpec],
    entity_frames: dict[EntityType, pd.DataFrame],
    entity_weight_indexes: dict[EntityType, pd.Series | np.ndarray],
    n_weights: int,
    dtype: Any = np.float64,
) -> SparseTargetMatrix:
    """Compile canonical target specs into one sparse target matrix."""
    compilation = compile_target_reweighting_constraints(
        targets=targets,
        entity_frames=entity_frames,
        entity_weight_indexes=entity_weight_indexes,
    )
    return target_constraints_to_sparse_matrix(
        compilation.constraints,
        n_weights=n_weights,
        skipped_targets=compilation.skipped_targets,
        dtype=dtype,
    )


def assemble_clone_sparse_target_matrix(
    clone_matrices: Mapping[int, SparseTargetMatrix],
    *,
    n_records: int,
    n_clones: int,
    dtype: Any = np.float64,
) -> SparseTargetMatrix:
    """Assemble per-clone target matrices into one clone-expanded surface.

    Each per-clone matrix must be compiled over exactly ``n_records`` columns
    and must expose the same target rows in the same order. All clone indices
    in ``[0, n_clones)`` must be present, which prevents a partial calibration
    surface from passing as a complete eCPS-style clone matrix.
    """
    if n_records <= 0:
        raise ValueError(f"n_records must be positive; got {n_records}.")
    if n_clones <= 0:
        raise ValueError(f"n_clones must be positive; got {n_clones}.")

    expected_clones = set(range(n_clones))
    actual_clones = set(clone_matrices)
    if actual_clones != expected_clones:
        missing = sorted(expected_clones - actual_clones)
        extra = sorted(actual_clones - expected_clones)
        raise ValueError(
            "clone_matrices must contain exactly one matrix for each clone; "
            f"missing={missing}, extra={extra}."
        )

    reference = clone_matrices[0]
    blocks: list[CloneMatrixBlock] = []
    for clone_idx in range(n_clones):
        clone_matrix = clone_matrices[clone_idx]
        _validate_clone_matrix_alignment(
            clone_idx=clone_idx,
            clone_matrix=clone_matrix,
            reference=reference,
            n_records=n_records,
        )
        coo = clone_matrix.matrix.tocoo()
        blocks.append(
            CloneMatrixBlock(
                clone_idx=clone_idx,
                row_indices=coo.row,
                record_indices=coo.col,
                values=coo.data,
            )
        )

    matrix = assemble_clone_csr(
        blocks,
        n_targets=reference.n_targets,
        n_records=n_records,
        n_clones=n_clones,
        dtype=dtype,
    )
    return SparseTargetMatrix(
        matrix=matrix,
        target_vector=reference.target_vector.astype(dtype, copy=False),
        names=reference.names,
        metadata=reference.metadata,
        skipped_targets=reference.skipped_targets,
    )


def _validate_clone_matrix_alignment(
    *,
    clone_idx: int,
    clone_matrix: SparseTargetMatrix,
    reference: SparseTargetMatrix,
    n_records: int,
) -> None:
    if clone_matrix.n_weights != n_records:
        raise ValueError(
            f"clone {clone_idx} matrix has {clone_matrix.n_weights} columns; "
            f"expected n_records={n_records}."
        )
    if clone_matrix.names != reference.names:
        raise ValueError(f"clone {clone_idx} target names do not match clone 0.")
    if not np.array_equal(clone_matrix.target_vector, reference.target_vector):
        raise ValueError(f"clone {clone_idx} target vector does not match clone 0.")
    if clone_matrix.metadata != reference.metadata:
        raise ValueError(f"clone {clone_idx} target metadata do not match clone 0.")
    if clone_matrix.skipped_targets != reference.skipped_targets:
        raise ValueError(
            f"clone {clone_idx} skipped-target diagnostics do not match clone 0."
        )
