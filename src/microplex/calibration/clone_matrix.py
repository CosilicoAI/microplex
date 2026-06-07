"""Sparse clone-column calibration matrix assembly.

The eCPS-style calibration matrix uses one calibration column per
``(clone_idx, record_idx)`` pair, with the invariant:

``column = clone_idx * n_records + record_idx``.

This module is intentionally smaller than a full target compiler or
PolicyEngine simulator adapter. It provides the country-agnostic sparse
assembly primitive those stages can feed without open-coding the riskiest
indexing rule or materializing a dense target-by-clone matrix.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

__all__ = [
    "CloneMatrixBlock",
    "assemble_clone_csr",
    "clone_column_indices",
]


@dataclass(frozen=True)
class CloneMatrixBlock:
    """COO entries contributed by one clone.

    Attributes:
        clone_idx: Zero-based clone index.
        row_indices: Target-row indices for this clone's nonzero entries.
        record_indices: Base-record indices for this clone's nonzero entries.
        values: Nonzero matrix values aligned to ``row_indices`` and
            ``record_indices``.
    """

    clone_idx: int
    row_indices: Sequence[int] | np.ndarray
    record_indices: Sequence[int] | np.ndarray
    values: Sequence[float] | np.ndarray


def clone_column_indices(
    clone_idx: int,
    record_indices: Sequence[int] | np.ndarray,
    *,
    n_records: int,
    n_clones: int | None = None,
) -> np.ndarray:
    """Map base-record indices to eCPS clone-column indices.

    Args:
        clone_idx: Zero-based clone index.
        record_indices: Base-record indices within ``[0, n_records)``.
        n_records: Number of base records per clone.
        n_clones: Optional upper bound for ``clone_idx``.

    Returns:
        Integer column indices satisfying
        ``clone_idx * n_records + record_idx``.
    """
    _validate_positive("n_records", n_records)
    if n_clones is not None:
        _validate_positive("n_clones", n_clones)
    if clone_idx < 0:
        raise ValueError(f"clone_idx must be non-negative; got {clone_idx}.")
    if n_clones is not None and clone_idx >= n_clones:
        raise ValueError(
            f"clone_idx {clone_idx} is outside [0, {n_clones})."
        )

    records = _as_int_vector("record_indices", record_indices)
    if records.size and ((records < 0).any() or (records >= n_records).any()):
        raise ValueError(
            "record_indices must be within [0, n_records); "
            f"got min={records.min()}, max={records.max()}, n_records={n_records}."
        )
    return clone_idx * n_records + records


def assemble_clone_csr(
    blocks: Iterable[CloneMatrixBlock],
    *,
    n_targets: int,
    n_records: int,
    n_clones: int,
    dtype: Any = np.float64,
) -> sparse.csr_matrix:
    """Assemble clone COO blocks into a sparse CSR calibration matrix.

    Duplicate ``(row, column)`` entries are summed by SciPy's COO-to-CSR
    conversion. The function validates all index bounds before assembling.
    It never builds a dense ``(n_targets, n_records * n_clones)`` array.
    """
    if n_targets < 0:
        raise ValueError(f"n_targets must be non-negative; got {n_targets}.")
    _validate_positive("n_records", n_records)
    _validate_positive("n_clones", n_clones)

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []

    for block in blocks:
        rows = _as_int_vector("row_indices", block.row_indices)
        records = _as_int_vector("record_indices", block.record_indices)
        values = _as_float_vector("values", block.values, dtype=dtype)
        if not (len(rows) == len(records) == len(values)):
            raise ValueError(
                "CloneMatrixBlock row_indices, record_indices, and values "
                "must have the same length."
            )
        if rows.size and ((rows < 0).any() or (rows >= n_targets).any()):
            raise ValueError(
                "row_indices must be within [0, n_targets); "
                f"got min={rows.min()}, max={rows.max()}, n_targets={n_targets}."
            )
        columns = clone_column_indices(
            block.clone_idx,
            records,
            n_records=n_records,
            n_clones=n_clones,
        )
        row_parts.append(rows)
        col_parts.append(columns)
        value_parts.append(values)

    shape = (n_targets, n_records * n_clones)
    if not row_parts:
        return sparse.csr_matrix(shape, dtype=dtype)

    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    values = np.concatenate(value_parts).astype(dtype, copy=False)
    return sparse.coo_matrix((values, (rows, cols)), shape=shape).tocsr()


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value}.")


def _as_int_vector(name: str, value: Sequence[int] | np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size and not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integer indices.")
    return array.astype(np.int64, copy=False)


def _as_float_vector(
    name: str,
    value: Sequence[float] | np.ndarray,
    *,
    dtype: Any,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array
