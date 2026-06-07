"""Sparse target matrix compilation helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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

SPARSE_TARGET_MATRIX_CERTIFICATE_SCHEMA = "microplex.sparse_target_matrix.v1"


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

    def certificate(self) -> SparseTargetMatrixCertificate:
        """Build a deterministic identity certificate for this target surface."""
        return build_sparse_target_matrix_certificate(self)

    def assert_matches_certificate(
        self,
        certificate: SparseTargetMatrixCertificate | Mapping[str, Any],
    ) -> None:
        """Raise if this target surface does not match a stored certificate."""
        assert_sparse_target_matrix_certificate(self, certificate)


@dataclass(frozen=True)
class SparseTargetMatrixCertificate:
    """Deterministic identity hashes for a sparse target matrix."""

    schema_version: str
    n_targets: int
    n_weights: int
    nnz: int
    names_sha256: str
    target_vector_sha256: str
    metadata_sha256: str
    skipped_targets_sha256: str
    matrix_structure_sha256: str
    matrix_values_sha256: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "n_targets": self.n_targets,
            "n_weights": self.n_weights,
            "nnz": self.nnz,
            "names_sha256": self.names_sha256,
            "target_vector_sha256": self.target_vector_sha256,
            "metadata_sha256": self.metadata_sha256,
            "skipped_targets_sha256": self.skipped_targets_sha256,
            "matrix_structure_sha256": self.matrix_structure_sha256,
            "matrix_values_sha256": self.matrix_values_sha256,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SparseTargetMatrixCertificate:
        return cls(
            schema_version=str(payload["schema_version"]),
            n_targets=int(payload["n_targets"]),
            n_weights=int(payload["n_weights"]),
            nnz=int(payload["nnz"]),
            names_sha256=str(payload["names_sha256"]),
            target_vector_sha256=str(payload["target_vector_sha256"]),
            metadata_sha256=str(payload["metadata_sha256"]),
            skipped_targets_sha256=str(payload["skipped_targets_sha256"]),
            matrix_structure_sha256=str(payload["matrix_structure_sha256"]),
            matrix_values_sha256=str(payload["matrix_values_sha256"]),
            extra=dict(payload.get("extra", {})),
        )


def build_sparse_target_matrix_certificate(
    target_matrix: SparseTargetMatrix,
    *,
    extra: Mapping[str, Any] | None = None,
) -> SparseTargetMatrixCertificate:
    """Build a deterministic certificate for a sparse target matrix."""
    matrix = target_matrix.matrix.copy().tocsr()
    matrix.sort_indices()
    return SparseTargetMatrixCertificate(
        schema_version=SPARSE_TARGET_MATRIX_CERTIFICATE_SCHEMA,
        n_targets=target_matrix.n_targets,
        n_weights=target_matrix.n_weights,
        nnz=int(matrix.nnz),
        names_sha256=_json_sha256(target_matrix.names),
        target_vector_sha256=_array_sha256(target_matrix.target_vector, dtype="<f8"),
        metadata_sha256=_json_sha256(target_matrix.metadata),
        skipped_targets_sha256=_json_sha256(target_matrix.skipped_targets),
        matrix_structure_sha256=_matrix_structure_sha256(matrix),
        matrix_values_sha256=_array_sha256(matrix.data, dtype="<f8"),
        extra=dict(extra or {}),
    )


def assert_sparse_target_matrix_certificate(
    target_matrix: SparseTargetMatrix,
    certificate: SparseTargetMatrixCertificate | Mapping[str, Any],
) -> None:
    """Raise if a sparse target matrix does not match a stored certificate."""
    expected = (
        certificate
        if isinstance(certificate, SparseTargetMatrixCertificate)
        else SparseTargetMatrixCertificate.from_dict(certificate)
    )
    actual = build_sparse_target_matrix_certificate(target_matrix, extra=expected.extra)
    expected_payload = expected.to_dict()
    actual_payload = actual.to_dict()
    mismatches = [
        key
        for key, expected_value in expected_payload.items()
        if actual_payload.get(key) != expected_value
    ]
    if mismatches:
        raise ValueError(
            "Sparse target matrix certificate mismatch: "
            + ", ".join(sorted(mismatches))
        )


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


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _array_sha256(array: np.ndarray, *, dtype: str) -> str:
    contiguous = np.ascontiguousarray(np.asarray(array).astype(dtype, copy=False))
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _matrix_structure_sha256(matrix: sparse.csr_matrix) -> str:
    digest = hashlib.sha256()
    digest.update(str(matrix.shape).encode("utf-8"))
    digest.update(_array_bytes(matrix.indptr, dtype="<i8"))
    digest.update(_array_bytes(matrix.indices, dtype="<i8"))
    return digest.hexdigest()


def _array_bytes(array: np.ndarray, *, dtype: str) -> bytes:
    return np.ascontiguousarray(np.asarray(array).astype(dtype, copy=False)).tobytes()
