"""Country-agnostic adapter that wraps `microcalibrate.Calibration`.

Presents the legacy `microplex.calibration.Calibrator.fit_transform`
surface on top of a gradient-descent chi-squared solver so country
packages (`microplex-us`, `microplex-uk`, etc.) share one
identity-preserving calibrator instead of duplicating the glue. Every
input record survives to the output with a non-negative weight.

`microcalibrate` is an optional upstream dependency installed via the
``microplex[calibrate]`` extra. This module raises `ImportError` at
top-level if the extra isn't installed; `microplex.calibration`'s own
``__init__.py`` imports from here inside a ``try/except`` so callers
get the adapter when the extra is present and a clean no-op otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from microcalibrate import Calibration

from microplex.calibration import LinearConstraint
from microplex.calibration.solve_policy import (
    CalibrationSolvePolicy,
    resolve_calibration_solve_policy,
)
from microplex.spec import CalibrateSpec

if TYPE_CHECKING:
    from microplex.targets import (
        SparseTargetMatrix,
        SparseTargetMatrixCertificate,
    )


@dataclass(frozen=True)
class MicrocalibrateAdapterConfig:
    """Hyperparameters for `MicrocalibrateAdapter`.

    Defaults mirror `microcalibrate.Calibration`'s own defaults
    (epochs=32, learning_rate=1e-3, noise_level=10.0) except ``device``,
    which microcalibrate auto-selects CUDA > MPS > CPU but we leave as
    None so callers keep deterministic control.
    """

    epochs: int = 32
    learning_rate: float = 1e-3
    noise_level: float = 10.0
    dropout_rate: float = 0.0
    device: str | None = None  # None = let microcalibrate auto-select
    seed: int = 42
    regularize_with_l0: bool = False
    l0_lambda: float = 5e-6
    init_mean: float = 0.999
    temperature: float = 0.5
    sparse_learning_rate: float = 0.2
    # Keep activation memory bounded at country-scale pipelines. 100k
    # records per backward step keeps per-batch autograd activation
    # under ~200 MB at k = 500 constraints (100_000 * 500 * 4 B).
    # None = full-batch, which can OOM past ~500k records.
    batch_size: int | None = 100_000


@dataclass(frozen=True)
class SparseTargetMatrixCalibrationResult:
    """Result of a policy-aware sparse target matrix calibration."""

    weights: np.ndarray
    policy: CalibrationSolvePolicy
    certificate: SparseTargetMatrixCertificate
    validation: dict[str, Any]

    def diagnostics(self) -> dict[str, Any]:
        """Stable diagnostics payload for manifests and release logs."""
        return {
            "policy": self.policy.diagnostics(),
            "certificate": self.certificate.to_dict(),
            "validation": dict(self.validation),
        }


class MicrocalibrateAdapter:
    """Drop-in replacement for `Calibrator.fit_transform` / `validate`.

    Usage:

        >>> adapter = MicrocalibrateAdapter()
        >>> result = adapter.fit_transform(
        ...     data=households_df,
        ...     weight_col="household_weight",
        ...     linear_constraints=tuple_of_LinearConstraints,
        ... )
        >>> validation = adapter.validate(result)

    The returned DataFrame is a copy of ``data`` with ``weight_col``
    updated.
    """

    def __init__(
        self,
        config: MicrocalibrateAdapterConfig | None = None,
    ) -> None:
        self.config = config or MicrocalibrateAdapterConfig()
        self._last_calibration: Calibration | None = None
        self._last_constraint_names: list[str] | None = None
        self._last_targets: np.ndarray | None = None
        self._last_performance: pd.DataFrame | None = None

    def fit_transform(
        self,
        data: pd.DataFrame,
        marginal_targets: dict[str, dict[str, float]] | None = None,
        continuous_targets: dict[str, float] | None = None,
        *,
        weight_col: str = "weight",
        linear_constraints: Sequence[LinearConstraint] = (),
    ) -> pd.DataFrame:
        """Calibrate weights via gradient-descent chi-squared.

        ``marginal_targets`` and ``continuous_targets`` are accepted for
        signature parity with the legacy `Calibrator`, but this adapter
        expects constraints to be expressed as `LinearConstraint` rows.
        Callers should compile their marginal / continuous targets into
        linear constraints before calling.
        """
        if weight_col not in data.columns:
            raise ValueError(
                f"MicrocalibrateAdapter: weight column {weight_col!r} "
                f"not found in data (columns: {list(data.columns)[:10]}...)"
            )

        n_records = len(data)
        initial_weights = data[weight_col].to_numpy(dtype=float)

        if not linear_constraints:
            # Nothing to calibrate — preserve caller expectations.
            self._last_calibration = None
            self._last_constraint_names = []
            self._last_targets = np.empty(0, dtype=float)
            self._last_performance = None
            return data.copy()

        target_names = [c.name for c in linear_constraints]
        targets = np.array([c.target for c in linear_constraints], dtype=float)

        for constraint in linear_constraints:
            if constraint.coefficients.shape != (n_records,):
                raise ValueError(
                    f"MicrocalibrateAdapter: constraint {constraint.name!r} has "
                    f"coefficients shape {constraint.coefficients.shape}, expected "
                    f"({n_records},) matching the data length."
                )

        estimate_matrix = pd.DataFrame(
            np.column_stack(
                [
                    np.asarray(c.coefficients, dtype=np.float32)
                    for c in linear_constraints
                ]
            ),
            columns=target_names,
        )

        fitted_weights = self._fit_estimate_matrix(
            weights=initial_weights,
            target_names=target_names,
            targets=targets,
            estimate_matrix=estimate_matrix,
        )

        result = data.copy()
        result[weight_col] = fitted_weights
        return result

    def fit_sparse_target_matrix(
        self,
        initial_weights: pd.Series | np.ndarray,
        target_matrix: SparseTargetMatrix,
        *,
        certificate: SparseTargetMatrixCertificate | Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        """Calibrate one certified sparse target matrix directly.

        This is the fail-closed path for release builds: compile the
        production-eCPS target surface once, persist its certificate, then pass
        that exact sparse surface to the solver. If a later run changes target
        names, targets, metadata, skipped diagnostics, matrix structure, or
        coefficients, the optional certificate check raises before fitting.
        """
        if certificate is not None:
            target_matrix.assert_matches_certificate(certificate)

        weights = np.asarray(initial_weights, dtype=float)
        if weights.ndim != 1:
            raise ValueError("initial_weights must be one-dimensional.")
        if len(weights) != target_matrix.n_weights:
            raise ValueError(
                "initial_weights length must match target_matrix.n_weights; "
                f"got {len(weights)} and {target_matrix.n_weights}."
            )

        if target_matrix.n_targets == 0:
            self._last_calibration = None
            self._last_constraint_names = []
            self._last_targets = np.empty(0, dtype=float)
            self._last_performance = None
            return weights.copy()

        # Preserve the sparse target surface as far as pandas allows. This
        # avoids rebuilding linear constraints and keeps the coefficient matrix
        # tied to the target certificate used by the release gate.
        estimate_matrix = pd.DataFrame.sparse.from_spmatrix(
            target_matrix.matrix.transpose().tocsr(),
            columns=list(target_matrix.names),
        )
        return self._fit_estimate_matrix(
            weights=weights,
            target_names=list(target_matrix.names),
            targets=target_matrix.target_vector,
            estimate_matrix=estimate_matrix,
        )

    def fit_sparse_target_matrix_with_policy(
        self,
        initial_weights: pd.Series | np.ndarray,
        target_matrix: SparseTargetMatrix,
        *,
        calibrate: CalibrateSpec,
        certificate: SparseTargetMatrixCertificate | Mapping[str, Any] | None = None,
        min_records_per_target: float | None = None,
    ) -> SparseTargetMatrixCalibrationResult:
        """Calibrate a certified sparse target matrix using a resolved spec policy.

        This is the release-build adapter path: the country pack compiles one
        sparse target matrix, persists its certificate, then asks core to
        resolve the declared ``calibrate`` section and pass that exact matrix to
        microcalibrate. Policy validation happens before fitting, so an empty
        target surface, impossible prune count, or incompatible solver/pruning
        declaration cannot silently fall through to an ad hoc solve.
        """
        policy = resolve_calibration_solve_policy(
            calibrate,
            n_records=target_matrix.n_weights,
            target_count=target_matrix.n_targets,
            min_records_per_target=min_records_per_target,
        )
        resolved_certificate = (
            target_matrix.certificate() if certificate is None else certificate
        )
        target_matrix.assert_matches_certificate(resolved_certificate)

        original_config = self.config
        self.config = replace(
            self.config,
            **policy.microcalibrate_config_overrides(),
        )
        try:
            fitted_weights = self.fit_sparse_target_matrix(
                initial_weights,
                target_matrix,
                certificate=resolved_certificate,
            )
        finally:
            self.config = original_config

        return SparseTargetMatrixCalibrationResult(
            weights=fitted_weights,
            policy=policy,
            certificate=target_matrix.certificate(),
            validation=self.validate(),
        )

    def validate(self, calibrated: pd.DataFrame | None = None) -> dict[str, Any]:
        """Return validation metrics in the shape the legacy pipeline expects.

        The legacy `Calibrator.validate` returns ``{"converged",
        "max_error", "sparsity", "linear_errors"}``. We populate the
        same keys. ``calibrated`` is accepted for interface parity but
        not read; the authoritative values come from the last
        ``calibrate()`` call.
        """
        if self._last_calibration is None:
            return {
                "converged": True,
                "max_error": 0.0,
                "sparsity": 0.0,
                "linear_errors": {},
            }

        estimates = self._last_calibration.estimate().to_numpy(dtype=float)
        targets = self._last_targets
        names = self._last_constraint_names

        rel_errors = np.where(
            np.abs(targets) > 1e-12,
            np.abs(estimates - targets) / np.abs(targets),
            np.abs(estimates - targets),
        )
        linear_errors = {
            name: {
                "target": float(target_value),
                "estimate": float(estimate_value),
                "relative_error": float(rel_error),
                "absolute_error": float(abs(estimate_value - target_value)),
            }
            for name, target_value, estimate_value, rel_error in zip(
                names, targets, estimates, rel_errors, strict=True
            )
        }

        max_error = float(rel_errors.max()) if rel_errors.size else 0.0
        weights = self._last_calibration.weights
        sparsity = float((weights == 0).sum()) / max(len(weights), 1)

        return {
            "converged": bool(max_error < 0.05),  # 5 % relative error bar
            "max_error": max_error,
            "sparsity": sparsity,
            "linear_errors": linear_errors,
        }

    def performance_history(self) -> pd.DataFrame | None:
        """Per-epoch performance log from microcalibrate, if available."""
        return self._last_performance

    def _fit_estimate_matrix(
        self,
        *,
        weights: np.ndarray,
        target_names: list[str],
        targets: np.ndarray,
        estimate_matrix: pd.DataFrame,
    ) -> np.ndarray:
        """Run microcalibrate against a precompiled estimate matrix."""
        calibrator = Calibration(
            weights=weights,
            targets=np.asarray(targets, dtype=float),
            target_names=np.array(target_names),
            estimate_matrix=estimate_matrix,
            epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            noise_level=self.config.noise_level,
            dropout_rate=self.config.dropout_rate,
            device=self.config.device,
            seed=self.config.seed,
            regularize_with_l0=self.config.regularize_with_l0,
            l0_lambda=self.config.l0_lambda,
            init_mean=self.config.init_mean,
            temperature=self.config.temperature,
            sparse_learning_rate=self.config.sparse_learning_rate,
            batch_size=self.config.batch_size,
        )

        performance_df = calibrator.calibrate()
        self._last_calibration = calibrator
        self._last_constraint_names = list(target_names)
        self._last_targets = np.asarray(targets, dtype=float)
        self._last_performance = performance_df
        return np.asarray(calibrator.weights, dtype=float).copy()


__all__ = [
    "MicrocalibrateAdapter",
    "MicrocalibrateAdapterConfig",
    "SparseTargetMatrixCalibrationResult",
]
