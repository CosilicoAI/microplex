"""
Calibration methods for adjusting sample weights to match external targets.

Implements classical survey calibration algorithms:
- IPF (Iterative Proportional Fitting) / Raking
- Chi-square distance minimization
- Entropy balancing

These methods adjust weights so weighted aggregates match known population
statistics (e.g., total income from IRS SOI, population counts from Census).

Example:
    >>> from microplex.calibration import Calibrator
    >>> calibrator = Calibrator(method="entropy")
    >>> targets = {
    ...     "state": {"CA": 400, "NY": 350, "TX": 250},
    ...     "age_group": {"0-17": 250, "18-64": 550, "65+": 200},
    ... }
>>> continuous_targets = {"income": 50_000_000}  # Total income
>>> calibrated = calibrator.fit_transform(data, targets, continuous_targets)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, minimize

from microplex.calibration.solve_policy import (  # noqa: F401
    CalibrationSolvePolicy,
    resolve_calibration_solve_policy,
)


@dataclass(frozen=True)
class LinearConstraint:
    """An explicit linear calibration row aligned to the calibrated records."""

    name: str
    coefficients: np.ndarray
    target: float

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=float)
        if coefficients.ndim != 1:
            raise ValueError("LinearConstraint.coefficients must be one-dimensional")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("LinearConstraint.coefficients must be finite")
        if not self.name:
            raise ValueError("LinearConstraint.name must be non-empty")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "target", float(self.target))


def _build_linear_constraint_system(
    data: pd.DataFrame,
    marginal_targets: dict[str, dict[str, float]],
    continuous_targets: dict[str, float] | None,
    linear_constraints: tuple[LinearConstraint, ...],
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """Build a dense linear system from built-in and explicit constraints."""
    rows: list[np.ndarray] = []
    targets: list[float] = []
    names: list[str] = []
    n_categorical = 0

    for var, var_targets in marginal_targets.items():
        for category, target in var_targets.items():
            rows.append((data[var] == category).astype(float).values)
            targets.append(float(target))
            names.append(f"{var}={category}")
            n_categorical += 1

    for var, target in (continuous_targets or {}).items():
        rows.append(data[var].values.astype(float))
        targets.append(float(target))
        names.append(var)

    for constraint in linear_constraints:
        rows.append(constraint.coefficients.astype(float))
        targets.append(float(constraint.target))
        names.append(constraint.name)

    A = np.vstack(rows) if rows else np.zeros((0, len(data)), dtype=float)
    b = np.array(targets, dtype=float)
    return A, b, names, n_categorical


def _build_sparse_constraint_system(
    data: pd.DataFrame,
    marginal_targets: dict[str, dict[str, float]],
    continuous_targets: dict[str, float] | None,
    linear_constraints: tuple[LinearConstraint, ...],
):
    """Build a CSR linear system without the dense `np.vstack` intermediate.

    Returns ``(X_csr, b, names, n_categorical)`` with exactly the same
    semantics as ``_build_linear_constraint_system`` but the matrix is
    `scipy.sparse.csr_matrix` built row-by-row, never allocating a
    dense float64 array of shape ``(n_targets, n_records)``. At v7
    scale (~1.5M records x ~4k constraints) this is the difference
    between ~24 GB dense peak and ~500 MB sparse CSR.

    Marginal-target rows are stored via row-indexed integer lookup
    (only the `n_matching_rows` entries with value 1.0 are kept).
    Continuous-target rows are dense in the record dimension but still
    built without stacking. `LinearConstraint` rows are passed through
    with `.nonzero()` filtering so explicit-zero coefficients don't
    balloon the CSR.
    """
    from scipy import sparse as _sp

    indptr: list[int] = [0]
    indices: list[np.ndarray] = []
    data_vals: list[np.ndarray] = []
    targets: list[float] = []
    names: list[str] = []
    n_categorical = 0
    n_records = len(data)

    def _append_row(col_indices: np.ndarray, values: np.ndarray) -> None:
        indices.append(col_indices.astype(np.int64, copy=False))
        data_vals.append(values.astype(np.float64, copy=False))
        indptr.append(indptr[-1] + len(col_indices))

    for var, var_targets in marginal_targets.items():
        column = data[var].to_numpy()
        for category, target in var_targets.items():
            match = np.flatnonzero(column == category)
            _append_row(match, np.ones(len(match), dtype=np.float64))
            targets.append(float(target))
            names.append(f"{var}={category}")
            n_categorical += 1

    for var, target in (continuous_targets or {}).items():
        values = data[var].to_numpy(dtype=np.float64, copy=False)
        nz = np.flatnonzero(values != 0.0)
        _append_row(nz, values[nz])
        targets.append(float(target))
        names.append(var)

    for constraint in linear_constraints:
        coef = np.asarray(constraint.coefficients, dtype=np.float64)
        nz = np.flatnonzero(coef != 0.0)
        _append_row(nz, coef[nz])
        targets.append(float(constraint.target))
        names.append(constraint.name)

    if not names:
        X_csr = _sp.csr_matrix((0, n_records), dtype=np.float64)
        b = np.zeros(0, dtype=np.float64)
        return X_csr, b, names, n_categorical

    X_csr = _sp.csr_matrix(
        (
            np.concatenate(data_vals) if data_vals else np.zeros(0),
            np.concatenate(indices) if indices else np.zeros(0, dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(names), n_records),
        dtype=np.float64,
    )
    b = np.asarray(targets, dtype=np.float64)
    return X_csr, b, names, n_categorical


def _validate_calibration_inputs(
    data: pd.DataFrame,
    marginal_targets: dict[str, dict[str, float]],
    continuous_targets: dict[str, float] | None,
    linear_constraints: tuple[LinearConstraint, ...],
) -> None:
    """Validate shared calibration inputs across calibrator backends."""
    for var in marginal_targets:
        if var not in data.columns:
            raise ValueError(f"Margin variable '{var}' not in data columns")

        data_categories = set(data[var].unique())
        target_categories = set(marginal_targets[var].keys())
        missing = data_categories - target_categories
        if missing:
            raise ValueError(
                f"Data contains categories not in targets for '{var}': {missing}"
            )

    if continuous_targets:
        for var in continuous_targets:
            if var not in data.columns:
                raise ValueError(
                    f"Continuous target variable '{var}' not in data columns"
                )

    for constraint in linear_constraints:
        if len(constraint.coefficients) != len(data):
            raise ValueError(
                "LinearConstraint.coefficients must have the same length as data"
            )


class Calibrator:
    """
    Calibrate sample weights to match external aggregate targets.

    Three calibration methods are supported:
    - IPF (Iterative Proportional Fitting): Classic raking algorithm that
      iteratively adjusts weights to match marginal totals
    - Chi-square: Minimizes chi-square distance from initial weights while
      matching targets (quadratic loss)
    - Entropy: Minimizes Kullback-Leibler divergence from initial weights
      (exponential tilting)

    Key features:
    - Supports categorical margin constraints (state, age group, etc.)
    - Supports continuous total constraints (total income, population)
    - Validation to compare weighted aggregates to targets
    - Convergence diagnostics

    Example:
        >>> from microplex.calibration import Calibrator
        >>> calibrator = Calibrator(method="ipf")
        >>> targets = {"state": {"CA": 1000, "NY": 500}}
        >>> calibrated = calibrator.fit_transform(data, targets)
    """

    def __init__(
        self,
        method: Literal["ipf", "chi2", "entropy"] = "ipf",
        tol: float = 1e-6,
        max_iter: int = 100,
        lower_bound: float = 1e-10,
        upper_bound: float | None = None,
    ):
        """
        Initialize calibrator.

        Args:
            method: Calibration method ("ipf", "chi2", or "entropy")
            tol: Convergence tolerance for iterative methods
            max_iter: Maximum number of iterations
            lower_bound: Minimum allowed weight (prevents zero weights)
            upper_bound: Maximum allowed weight (None for no limit)

        Raises:
            ValueError: If method is not one of "ipf", "chi2", "entropy"
        """
        if method not in ["ipf", "chi2", "entropy"]:
            raise ValueError(
                f"Invalid method: {method}. Must be 'ipf', 'chi2', or 'entropy'"
            )

        self.method = method
        self.tol = tol
        self.max_iter = max_iter
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # Set during fit
        self.weights_: np.ndarray | None = None
        self.is_fitted_: bool = False
        self.n_records_: int | None = None
        self.marginal_targets_: dict[str, dict[str, float]] | None = None
        self.continuous_targets_: dict[str, float] | None = None
        self.linear_constraints_: tuple[LinearConstraint, ...] = ()
        self.convergence_history_: list[dict[str, Any]] = []
        self.n_iterations_: int = 0
        self.converged_: bool = False

    def fit(
        self,
        data: pd.DataFrame,
        marginal_targets: dict[str, dict[str, float]],
        continuous_targets: dict[str, float] | None = None,
        weight_col: str = "weight",
        linear_constraints: tuple[LinearConstraint, ...] | list[LinearConstraint] | None = None,
    ) -> Self:
        """
        Fit calibration weights to match targets.

        Args:
            data: DataFrame with microdata records
            marginal_targets: Dict of categorical targets {var: {category: count}}
            continuous_targets: Dict of continuous totals {var: total}
            weight_col: Name of initial weight column in data

        Returns:
            self

        Raises:
            ValueError: If margin variable not in data or category mismatch
        """
        self.n_records_ = len(data)
        self.marginal_targets_ = marginal_targets
        self.continuous_targets_ = continuous_targets or {}
        self.linear_constraints_ = tuple(linear_constraints or ())

        # Validate inputs
        self._validate_inputs(
            data,
            marginal_targets,
            continuous_targets,
            self.linear_constraints_,
        )

        # Get initial weights
        if weight_col in data.columns:
            initial_weights = data[weight_col].values.astype(float)
        else:
            initial_weights = np.ones(len(data), dtype=float)

        # Handle zero initial weights
        initial_weights = np.maximum(initial_weights, self.lower_bound)

        # Build constraint matrices
        A, b, _, _ = _build_linear_constraint_system(
            data,
            marginal_targets,
            continuous_targets,
            self.linear_constraints_,
        )

        # Solve based on method
        if self.method == "ipf":
            weights = self._fit_ipf(data, initial_weights, marginal_targets, continuous_targets)
        elif self.method == "chi2":
            weights = self._fit_chi2(A, b, initial_weights)
        else:  # entropy
            weights = self._fit_entropy(A, b, initial_weights)

        self.weights_ = weights
        self.is_fitted_ = True

        return self

    def _validate_inputs(
        self,
        data: pd.DataFrame,
        marginal_targets: dict[str, dict[str, float]],
        continuous_targets: dict[str, float] | None,
        linear_constraints: tuple[LinearConstraint, ...],
    ):
        """Validate input data and targets."""
        _validate_calibration_inputs(
            data,
            marginal_targets,
            continuous_targets,
            linear_constraints,
        )

    def _build_categorical_constraints(
        self,
        data: pd.DataFrame,
        marginal_targets: dict[str, dict[str, float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build constraint matrix for categorical margins."""
        constraints = []
        targets = []

        for var, var_targets in marginal_targets.items():
            for category, target in var_targets.items():
                # Indicator: 1 if record matches category
                indicator = (data[var] == category).astype(float).values
                constraints.append(indicator)
                targets.append(target)

        A = np.vstack(constraints) if constraints else np.zeros((0, len(data)))
        b = np.array(targets)

        return A, b

    def _build_continuous_constraints(
        self,
        data: pd.DataFrame,
        continuous_targets: dict[str, float] | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Build constraint matrix for continuous totals."""
        if not continuous_targets:
            return None, None

        constraints = []
        targets = []

        for var, target in continuous_targets.items():
            constraints.append(data[var].values.astype(float))
            targets.append(target)

        A = np.vstack(constraints)
        b = np.array(targets)

        return A, b

    def _fit_ipf(
        self,
        data: pd.DataFrame,
        initial_weights: np.ndarray,
        marginal_targets: dict[str, dict[str, float]],
        continuous_targets: dict[str, float] | None,
    ) -> np.ndarray:
        """
        Iterative Proportional Fitting (raking).

        Classic algorithm that iteratively adjusts weights to match
        each set of marginal totals.

        Algorithm:
            1. For each margin variable:
               - Compute current weighted totals
               - Compute adjustment factor = target / current
               - Multiply weights by adjustment factor for each category
            2. Repeat until convergence

        Note: For continuous targets, IPF uses linear regression-based
        calibration within each iteration, which can disrupt categorical
        margins. For best results with continuous targets, use 'chi2' or
        'entropy' methods instead.
        """
        weights = initial_weights.copy()
        self.convergence_history_ = []

        for iteration in range(self.max_iter):
            old_weights = weights.copy()

            # Adjust for each categorical margin
            for var, var_targets in marginal_targets.items():
                for category, target in var_targets.items():
                    mask = data[var] == category
                    current_total = weights[mask].sum()

                    if current_total > self.lower_bound:
                        adjustment = target / current_total
                        weights[mask] *= adjustment

            # Adjust for continuous targets using linear calibration
            # This is a generalized raking approach for continuous variables
            if continuous_targets:
                for var, target in continuous_targets.items():
                    values = data[var].values
                    current_total = (weights * values).sum()

                    if abs(current_total) > self.lower_bound:
                        # Linear calibration: w_new = w * (1 + lambda * x)
                        # where lambda is chosen so sum(w_new * x) = target
                        # This is equivalent to: lambda = (target - current) / sum(w * x^2)
                        weighted_x2 = (weights * values ** 2).sum()
                        if weighted_x2 > self.lower_bound:
                            lam = (target - current_total) / weighted_x2
                            # Apply with damping to prevent negative weights
                            adjustment = 1 + lam * values
                            adjustment = np.maximum(adjustment, 0.1)  # Prevent negative/zero
                            weights = weights * adjustment

            # Apply bounds
            weights = np.maximum(weights, self.lower_bound)
            if self.upper_bound is not None:
                weights = np.minimum(weights, self.upper_bound)

            # Check convergence
            max_change = np.max(np.abs(weights - old_weights) / np.maximum(old_weights, self.lower_bound))

            self.convergence_history_.append({
                "iteration": iteration + 1,
                "max_error": max_change,
            })

            if max_change < self.tol:
                self.converged_ = True
                self.n_iterations_ = iteration + 1
                break

        else:
            self.converged_ = False
            self.n_iterations_ = self.max_iter

        return weights

    def _fit_chi2(
        self,
        A: np.ndarray,
        b: np.ndarray,
        initial_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Chi-square distance minimization.

        Finds weights w that minimize chi-square distance from initial weights w0:
            min sum((w - w0)^2 / w0)
        subject to:
            A @ w = b (constraints)
            w >= lower_bound

        This is a quadratic programming problem solved via Lagrange multipliers.
        """
        n = len(initial_weights)
        w0 = initial_weights

        # Lagrangian: L = sum((w - w0)^2 / w0) + lambda^T (A @ w - b)
        # Taking derivative and setting to zero:
        # 2(w - w0)/w0 + A^T @ lambda = 0
        # w = w0 * (1 - 0.5 * w0 * A^T @ lambda)
        #
        # Substituting into constraint A @ w = b:
        # A @ (w0 * (1 - 0.5 * w0 * A^T @ lambda)) = b
        #
        # We solve this iteratively using scipy.optimize

        def objective(w):
            """Chi-square distance."""
            return np.sum((w - w0) ** 2 / w0)

        def gradient(w):
            """Gradient of chi-square distance."""
            return 2 * (w - w0) / w0

        # Constraint: A @ w = b
        constraints = []
        for i in range(len(b)):
            constraints.append({
                "type": "eq",
                "fun": lambda w, i=i: A[i] @ w - b[i],
                "jac": lambda w, i=i: A[i],
            })

        # Bounds
        bounds = [(self.lower_bound, self.upper_bound) for _ in range(n)]

        # Initial guess: scale initial weights to approximately satisfy constraints
        scale = b.sum() / (A.sum(axis=0) @ w0) if (A.sum(axis=0) @ w0) > 0 else 1.0
        x0 = w0 * scale

        self.convergence_history_ = []

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            jac=gradient,
            constraints=constraints,
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        self.n_iterations_ = result.nit if hasattr(result, "nit") else self.max_iter
        self.converged_ = result.success

        self.convergence_history_.append({
            "iteration": self.n_iterations_,
            "max_error": result.fun if result.success else np.inf,
        })

        if not result.success:
            weights = self._fit_chi2_fallback(A, b, w0)
            residual_scale = np.maximum(np.abs(b), 1.0)
            fallback_error = float(np.max(np.abs((A @ weights - b) / residual_scale)))
            self.convergence_history_.append({
                "iteration": self.n_iterations_,
                "max_error": fallback_error,
                "fallback": True,
            })
            return weights

        return result.x

    def _fit_chi2_fallback(
        self,
        A: np.ndarray,
        b: np.ndarray,
        initial_weights: np.ndarray,
    ) -> np.ndarray:
        """Approximate chi-square calibration when exact equality constraints fail."""
        residual_scale = np.maximum(np.abs(b), 1.0)
        A_scaled = A / residual_scale[:, np.newaxis]
        b_scaled = b / residual_scale

        regularization = 1e-3
        A_augmented = np.vstack([A_scaled, np.eye(len(initial_weights)) * regularization])
        b_augmented = np.concatenate([b_scaled, initial_weights * regularization])

        upper_bound = self.upper_bound if self.upper_bound is not None else np.inf
        result = lsq_linear(
            A_augmented,
            b_augmented,
            bounds=(self.lower_bound, upper_bound),
            lsmr_tol="auto",
            max_iter=max(self.max_iter * 10, 200),
        )

        weights = result.x if result.success else initial_weights.copy()
        weights = np.maximum(weights, self.lower_bound)
        if self.upper_bound is not None:
            weights = np.minimum(weights, self.upper_bound)
        return weights

    def _fit_entropy(
        self,
        A: np.ndarray,
        b: np.ndarray,
        initial_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Entropy balancing (exponential tilting).

        Finds weights w that minimize Kullback-Leibler divergence from initial w0:
            min sum(w * log(w / w0))
        subject to:
            A @ w = b (constraints)
            w >= lower_bound

        This has closed-form solution: w = w0 * exp(A^T @ lambda)
        where lambda are Lagrange multipliers found by solving dual problem.
        """
        w0 = initial_weights
        n_constraints = len(b)

        if n_constraints == 0:
            self.converged_ = True
            self.n_iterations_ = 0
            self.convergence_history_ = []
            return w0.copy()

        solver_upper_bounds = self._entropy_solver_upper_bounds(A, b, w0)

        def dual_objective(lam):
            """
            Dual function: -sum(w0 * exp(A^T @ lam)) + lam^T @ b
            We minimize the negative of this (maximize dual).
            """
            tilted_weights = self._entropy_tilted_weights(A, lam, w0, solver_upper_bounds)
            return np.sum(tilted_weights) - lam @ b

        def dual_gradient(lam):
            """Gradient of dual objective."""
            tilted_weights = self._entropy_tilted_weights(A, lam, w0, solver_upper_bounds)
            return A @ tilted_weights - b

        def dual_hessian(lam):
            """Hessian of dual objective."""
            tilted_weights = self._entropy_tilted_weights(A, lam, w0, solver_upper_bounds)
            return (A * tilted_weights[np.newaxis, :]) @ A.T

        # Initial lambda (Lagrange multipliers)
        lam0 = np.zeros(n_constraints)

        self.convergence_history_ = []

        # Solve dual problem using Newton's method
        lam = lam0.copy()
        for iteration in range(self.max_iter):
            grad = dual_gradient(lam)
            hess = dual_hessian(lam)

            if not np.all(np.isfinite(grad)) or not np.all(np.isfinite(hess)):
                break

            # Check convergence
            max_error = np.max(np.abs(grad))
            self.convergence_history_.append({
                "iteration": iteration + 1,
                "max_error": max_error,
            })

            if max_error < self.tol:
                self.converged_ = True
                self.n_iterations_ = iteration + 1
                break

            # Newton step with regularization for stability
            try:
                hess_reg = hess + np.eye(n_constraints) * 1e-8
                step = np.linalg.solve(hess_reg, -grad)
            except np.linalg.LinAlgError:
                # Fallback to gradient descent
                step = -grad * 0.1

            # Line search for step size
            current_objective = dual_objective(lam)
            alpha = 1.0
            for _ in range(20):
                lam_new = lam + alpha * step
                new_objective = dual_objective(lam_new)
                if np.isfinite(new_objective) and new_objective < current_objective:
                    break
                alpha *= 0.5
            else:
                alpha = 0.01

            lam = lam + alpha * step

        else:
            self.converged_ = False
            self.n_iterations_ = self.max_iter

        if not self.converged_:
            return self._fit_entropy_fallback(A, b, w0)

        # Recover primal weights
        weights = self._entropy_tilted_weights(A, lam, w0, solver_upper_bounds)

        # Apply bounds
        weights = np.maximum(weights, self.lower_bound)
        if self.upper_bound is not None:
            weights = np.minimum(weights, self.upper_bound)

        return weights

    def _entropy_tilted_weights(
        self,
        A: np.ndarray,
        lam: np.ndarray,
        initial_weights: np.ndarray,
        upper_weights: np.ndarray,
    ) -> np.ndarray:
        """Compute entropy-balancing weights while keeping exponentials in range."""
        log_tilt = A.T @ lam

        max_log = np.log(upper_weights / initial_weights)
        min_log = np.full(len(initial_weights), np.log(np.finfo(float).tiny) + 1.0)

        if self.lower_bound > 0:
            min_log = np.maximum(min_log, np.log(self.lower_bound / initial_weights))

        safe_log_tilt = np.clip(log_tilt, min_log, max_log)
        return initial_weights * np.exp(safe_log_tilt)

    def _entropy_solver_upper_bounds(
        self,
        A: np.ndarray,
        b: np.ndarray,
        initial_weights: np.ndarray,
    ) -> np.ndarray:
        """Derive conservative per-record weight caps from positive constraints."""
        fallback_upper = max(
            float(initial_weights.sum()) * 2.0,
            float(initial_weights.max()) * 10.0,
            self.lower_bound * 10.0,
        )
        upper_bounds = np.full(len(initial_weights), fallback_upper, dtype=float)

        positive_rows = b > 0
        for row_idx in np.where(positive_rows)[0]:
            coefficients = A[row_idx]
            positive_mask = coefficients > 0
            if not positive_mask.any():
                continue
            upper_bounds[positive_mask] = np.minimum(
                upper_bounds[positive_mask],
                b[row_idx] / coefficients[positive_mask],
            )

        upper_bounds = np.maximum(
            upper_bounds,
            np.maximum(initial_weights, self.lower_bound),
        )
        if self.upper_bound is not None:
            upper_bounds = np.minimum(upper_bounds, self.upper_bound)

        return upper_bounds

    def _fit_entropy_fallback(
        self,
        A: np.ndarray,
        b: np.ndarray,
        initial_weights: np.ndarray,
    ) -> np.ndarray:
        """Approximate entropy solution for infeasible target systems."""
        w0 = initial_weights
        log_w0 = np.log(w0)
        residual_scale = np.maximum(np.abs(b), 1.0)
        upper_weights = self._entropy_solver_upper_bounds(A, b, w0)

        log_lower = np.log(self.lower_bound)
        bounds = [(log_lower, float(np.log(limit))) for limit in upper_weights]

        best_weights = w0.copy()
        best_max_error = np.inf
        best_entropy = np.inf
        best_iterations = 0

        for penalty in (1e2, 1e3, 1e4, 1e5):
            def objective(theta):
                weights = np.exp(theta)
                relative_residual = (A @ weights - b) / residual_scale
                entropy = np.sum(weights * (theta - log_w0) - weights + w0)
                return entropy + 0.5 * penalty * (relative_residual @ relative_residual)

            def gradient(theta):
                weights = np.exp(theta)
                relative_residual = (A @ weights - b) / residual_scale
                return weights * (
                    (theta - log_w0) + penalty * (A.T @ (relative_residual / residual_scale))
                )

            result = minimize(
                objective,
                log_w0.copy(),
                method="L-BFGS-B",
                jac=gradient,
                bounds=bounds,
                options={
                    "maxiter": max(self.max_iter * 5, 200),
                    "ftol": self.tol,
                },
            )

            weights = np.exp(result.x)
            relative_residual = np.abs((A @ weights - b) / residual_scale)
            max_error = float(np.max(relative_residual))
            entropy = float(np.sum(weights * (result.x - log_w0) - weights + w0))

            if (max_error, entropy) < (best_max_error, best_entropy):
                best_weights = weights
                best_max_error = max_error
                best_entropy = entropy
                best_iterations = int(getattr(result, "nit", 0))

        self.n_iterations_ = self.max_iter + best_iterations
        self.converged_ = best_max_error < self.tol
        self.convergence_history_.append({
            "iteration": self.n_iterations_,
            "max_error": best_max_error,
            "fallback": True,
        })

        best_weights = np.maximum(best_weights, self.lower_bound)
        if self.upper_bound is not None:
            best_weights = np.minimum(best_weights, self.upper_bound)

        return best_weights

    def transform(
        self,
        data: pd.DataFrame,
        weight_col: str = "weight",
    ) -> pd.DataFrame:
        """
        Apply fitted calibration weights to data.

        Args:
            data: DataFrame to reweight (must match fitted data length)
            weight_col: Name of weight column to update

        Returns:
            DataFrame with updated weights

        Raises:
            ValueError: If not fitted or data length doesn't match
        """
        if not self.is_fitted_:
            raise ValueError("Calibrator not fitted. Call fit() before transform().")

        if len(data) != self.n_records_:
            raise ValueError(
                f"Data length ({len(data)}) doesn't match fitted length ({self.n_records_})"
            )

        result = data.copy()
        result[weight_col] = self.weights_

        return result

    def fit_transform(
        self,
        data: pd.DataFrame,
        marginal_targets: dict[str, dict[str, float]],
        continuous_targets: dict[str, float] | None = None,
        weight_col: str = "weight",
        linear_constraints: tuple[LinearConstraint, ...] | list[LinearConstraint] | None = None,
    ) -> pd.DataFrame:
        """
        Fit calibration weights and apply to data in one call.

        Args:
            data: DataFrame with microdata
            marginal_targets: Dict of categorical targets
            continuous_targets: Dict of continuous totals (optional)
            weight_col: Name of weight column

        Returns:
            DataFrame with updated weights
        """
        self.fit(
            data,
            marginal_targets,
            continuous_targets,
            weight_col=weight_col,
            linear_constraints=linear_constraints,
        )
        return self.transform(data, weight_col=weight_col)

    def validate(
        self,
        data: pd.DataFrame,
        weight_col: str = "weight",
    ) -> dict[str, Any]:
        """
        Validate calibrated weights against targets.

        Computes weighted aggregates and compares to targets.

        Args:
            data: DataFrame with data (weights from fit will be used)
            weight_col: Name of weight column

        Returns:
            Dict with:
                - marginal_errors: {var: {category: (actual, target, error)}}
                - continuous_errors: {var: (actual, target, error)}
                - max_error: Maximum relative error across all targets
                - converged: Whether calibration converged
        """
        if not self.is_fitted_:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        weights = self.weights_
        marginal_errors = {}
        continuous_errors = {}
        linear_errors = {}
        all_errors = []

        # Check marginal targets
        if self.marginal_targets_:
            for var, var_targets in self.marginal_targets_.items():
                marginal_errors[var] = {}
                for category, target in var_targets.items():
                    mask = data[var] == category
                    actual = weights[mask].sum()
                    rel_error = abs(actual - target) / target if target > 0 else 0
                    marginal_errors[var][category] = {
                        "actual": actual,
                        "target": target,
                        "relative_error": rel_error,
                    }
                    all_errors.append(rel_error)

        # Check continuous targets
        if self.continuous_targets_:
            for var, target in self.continuous_targets_.items():
                actual = (weights * data[var].values).sum()
                rel_error = abs(actual - target) / abs(target) if target != 0 else 0
                continuous_errors[var] = {
                    "actual": actual,
                    "target": target,
                    "relative_error": rel_error,
                }
                all_errors.append(rel_error)

        for constraint in self.linear_constraints_:
            actual = float(weights @ constraint.coefficients)
            target = float(constraint.target)
            rel_error = abs(actual - target) / abs(target) if target != 0 else 0
            linear_errors[constraint.name] = {
                "actual": actual,
                "target": target,
                "relative_error": rel_error,
            }
            all_errors.append(rel_error)

        max_error = max(all_errors) if all_errors else 0

        return {
            "marginal_errors": marginal_errors,
            "continuous_errors": continuous_errors,
            "linear_errors": linear_errors,
            "max_error": max_error,
            "converged": self.converged_,
        }

    def get_weight_stats(self) -> dict[str, float]:
        """
        Get statistics about fitted weights.

        Returns:
            Dict with min, max, mean, cv (coefficient of variation), n_iterations
        """
        if not self.is_fitted_:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        weights = self.weights_
        mean_w = weights.mean()

        return {
            "min_weight": float(weights.min()),
            "max_weight": float(weights.max()),
            "mean_weight": float(mean_w),
            "cv": float(weights.std() / mean_w) if mean_w > 0 else 0,
            "n_iterations": self.n_iterations_,
        }

    def get_convergence_history(self) -> list[dict[str, Any]]:
        """
        Get convergence history from fitting.

        Returns:
            List of dicts with iteration number and max_error at each step
        """
        if not self.is_fitted_:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        return self.convergence_history_


class SparseCalibrator:
    """
    Sparse calibration that jointly optimizes accuracy and sparsity.

    Solves:
        min (1/2)||Aw - b||² + λ||w||₁  subject to w ≥ 0

    where:
        - A: constraint matrix (each row is a target)
        - b: target values
        - w: weights (decision variables)
        - λ: sparsity penalty (higher = more zeros)

    Records that are important for hitting targets retain positive weights.
    Redundant records are driven to zero.

    Key features:
    - Single optimization combining calibration + sparsity
    - λ controls accuracy-sparsity tradeoff
    - Optional: specify target_sparsity to auto-tune λ
    - Uses FISTA (Fast Iterative Shrinkage-Thresholding) for efficiency

    Example:
        >>> from microplex.calibration import SparseCalibrator
        >>> calibrator = SparseCalibrator(sparsity_weight=0.1)
        >>> targets = {"state": {"CA": 1000, "NY": 500}}
        >>> calibrated = calibrator.fit_transform(data, targets)
        >>> print(f"Sparsity: {calibrator.get_sparsity():.1%}")

        # Or specify target sparsity directly:
        >>> calibrator = SparseCalibrator(target_sparsity=0.8)  # 80% zeros
    """

    def __init__(
        self,
        sparsity_weight: float | None = None,
        target_sparsity: float | None = None,
        tol: float = 1e-6,
        max_iter: int = 1000,
        normalize_targets: bool = True,
    ):
        """
        Initialize sparse calibrator.

        Args:
            sparsity_weight: L1 penalty weight λ. Higher = more sparsity.
                If None, must specify target_sparsity.
            target_sparsity: Desired fraction of zero weights (0 to 1).
                If specified, λ is auto-tuned via binary search.
                Mutually exclusive with sparsity_weight.
            tol: Convergence tolerance
            max_iter: Maximum FISTA iterations
            normalize_targets: If True, normalize targets to similar scale
                before optimization (improves convergence)

        Raises:
            ValueError: If neither or both sparsity_weight and target_sparsity
                are specified
        """
        if sparsity_weight is None and target_sparsity is None:
            raise ValueError(
                "Must specify either sparsity_weight or target_sparsity"
            )
        if sparsity_weight is not None and target_sparsity is not None:
            raise ValueError(
                "Cannot specify both sparsity_weight and target_sparsity"
            )
        if target_sparsity is not None and not (0 <= target_sparsity < 1):
            raise ValueError("target_sparsity must be in [0, 1)")

        self.sparsity_weight = sparsity_weight
        self.target_sparsity = target_sparsity
        self.tol = tol
        self.max_iter = max_iter
        self.normalize_targets = normalize_targets

        # Set during fit
        self.weights_: np.ndarray | None = None
        self.is_fitted_: bool = False
        self.n_records_: int | None = None
        self.marginal_targets_: dict[str, dict[str, float]] | None = None
        self.continuous_targets_: dict[str, float] | None = None
        self.linear_constraints_: tuple[LinearConstraint, ...] = ()
        self.lambda_: float | None = None  # Actual λ used
        self.convergence_history_: list[dict[str, Any]] = []
        self.n_iterations_: int = 0
        self.calibration_error_: float = 0.0
        self.converged_: bool = False

    def fit(
        self,
        data: pd.DataFrame,
        marginal_targets: dict[str, dict[str, float]],
        continuous_targets: dict[str, float] | None = None,
        weight_col: str = "weight",
        linear_constraints: tuple[LinearConstraint, ...] | list[LinearConstraint] | None = None,
    ) -> Self:
        """
        Fit sparse calibration weights.

        Args:
            data: DataFrame with microdata records
            marginal_targets: Dict of categorical targets {var: {category: count}}
            continuous_targets: Dict of continuous totals {var: total}
            weight_col: Name of initial weight column (used for scaling)

        Returns:
            self
        """
        self.n_records_ = len(data)
        self.marginal_targets_ = marginal_targets
        self.continuous_targets_ = continuous_targets or {}
        self.linear_constraints_ = tuple(linear_constraints or ())
        _validate_calibration_inputs(
            data,
            marginal_targets,
            continuous_targets,
            self.linear_constraints_,
        )

        # Build constraint matrix A and target vector b
        A, b, self._target_names = self._build_constraints(
            data,
            marginal_targets,
            continuous_targets,
            self.linear_constraints_,
        )

        # Normalize each constraint row to have similar scale
        # This handles mixed categorical (counts) and continuous (dollars) targets
        if self.normalize_targets:
            self._row_scales = np.abs(b).copy()
            self._row_scales[self._row_scales < 1e-10] = 1.0
            # Normalize each row: (A[i] @ w) / scale[i] ≈ b[i] / scale[i] = 1
            A_norm = A / self._row_scales[:, np.newaxis]
            b_norm = b / self._row_scales  # Now all targets are ~1
        else:
            self._row_scales = np.ones(len(b))
            A_norm = A
            b_norm = b

        # Get initial weights for scaling reference
        if weight_col in data.columns:
            w0 = data[weight_col].values.astype(float)
        else:
            w0 = np.ones(len(data), dtype=float)

        # Compute step size for FISTA (1 / L where L is Lipschitz constant)
        # L = largest eigenvalue of A.T @ A = largest singular value of A squared
        # Use power iteration to estimate without forming A.T @ A explicitly
        L = self._estimate_lipschitz(A_norm)
        step_size = 1.0 / max(L, 1e-10)

        if self.target_sparsity is not None:
            # Binary search for λ that achieves target sparsity
            weights, lam = self._fit_with_target_sparsity(
                A_norm, b_norm, step_size, w0
            )
            self.lambda_ = lam
        else:
            # Use specified λ
            self.lambda_ = self.sparsity_weight
            weights = self._fista(A_norm, b_norm, self.lambda_, step_size, w0)

        # No rescaling needed - w directly satisfies A @ w ≈ b
        self.weights_ = weights
        self.is_fitted_ = True

        # Compute final calibration error
        residual = A @ weights - b
        rel_errors = np.abs(residual) / np.maximum(np.abs(b), 1e-10)
        self.calibration_error_ = np.sqrt(np.mean(rel_errors ** 2))

        return self

    def _build_constraints(
        self,
        data: pd.DataFrame,
        marginal_targets: dict[str, dict[str, float]],
        continuous_targets: dict[str, float] | None,
        linear_constraints: tuple[LinearConstraint, ...],
        use_sparse: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build constraint matrix A, target vector b, and target names.

        Uses sparse matrices by default when there are many categorical constraints.
        """
        from scipy import sparse

        n_records = len(data)
        rows = []
        cols = []
        vals = []
        targets = []
        names = []

        # Track which constraints are categorical (for cross-category selection)
        self._n_categorical_constraints = 0
        row_idx = 0

        # Categorical constraints first - build as sparse
        for var, var_targets in marginal_targets.items():
            if var not in data.columns:
                raise ValueError(f"Variable '{var}' not in data")

            for category, target in var_targets.items():
                mask = data[var] == category
                indices = np.where(mask)[0]
                rows.extend([row_idx] * len(indices))
                cols.extend(indices)
                vals.extend([1.0] * len(indices))
                targets.append(target)
                names.append(f"{var}={category}")
                self._n_categorical_constraints += 1
                row_idx += 1

        # Continuous constraints - these are dense rows
        continuous_rows = []
        if continuous_targets:
            for var, target in continuous_targets.items():
                if var not in data.columns:
                    raise ValueError(f"Variable '{var}' not in data")
                continuous_rows.append(data[var].values.astype(float))
                targets.append(target)
                names.append(var)

        linear_rows = []
        for constraint in linear_constraints:
            linear_rows.append(np.asarray(constraint.coefficients, dtype=float))
            targets.append(float(constraint.target))
            names.append(constraint.name)

        b = np.array(targets, dtype=float)

        # Build sparse matrix for categorical, then convert or stack with continuous
        n_cat = self._n_categorical_constraints
        n_cont = len(continuous_rows)
        n_linear = len(linear_rows)
        n_constraints = n_cat + n_cont + n_linear

        if n_constraints == 0:
            return np.zeros((0, n_records)), b, names

        # Decide whether to use sparse based on density
        # Categorical constraints are very sparse (1 entry per record per variable)
        # If >100 categorical constraints, sparse is much more efficient
        use_sparse = use_sparse and n_cat > 100

        if use_sparse:
            # Build sparse categorical matrix
            A_cat_sparse = sparse.csr_matrix(
                (vals, (rows, cols)), shape=(n_cat, n_records), dtype=float
            )

            if continuous_rows:
                # Stack with dense continuous rows (converted to sparse)
                A_cont = np.vstack(continuous_rows)
                A_cont_sparse = sparse.csr_matrix(A_cont)
                sparse_blocks = [A_cat_sparse, A_cont_sparse]
            else:
                sparse_blocks = [A_cat_sparse]

            if linear_rows:
                A_linear = np.vstack(linear_rows)
                A_linear_sparse = sparse.csr_matrix(A_linear)
                sparse_blocks.append(A_linear_sparse)

            A_sparse = sparse.vstack(sparse_blocks)

            # Store sparse matrix - caller must handle
            self._constraint_matrix_sparse = A_sparse
            # Return dense for compatibility, but flag that sparse is available
            self._use_sparse_internally = True
            # For now, convert to dense for backward compatibility
            # TODO: Update callers to use sparse directly
            A = A_sparse.toarray()
        else:
            # Build dense matrix (original behavior)
            constraints = []
            for var, var_targets in marginal_targets.items():
                for category in var_targets.keys():
                    indicator = (data[var] == category).astype(float).values
                    constraints.append(indicator)
            constraints.extend(continuous_rows)
            constraints.extend(linear_rows)
            A = np.vstack(constraints) if constraints else np.zeros((0, n_records))
            self._use_sparse_internally = False

        return A, b, names

    def _estimate_lipschitz(self, A: np.ndarray, n_iter: int = 20) -> float:
        """Estimate largest eigenvalue of A.T @ A using power iteration.

        This avoids forming the (n_records × n_records) matrix explicitly.
        Memory: O(n_records) instead of O(n_records²)
        """

        n_records = A.shape[1]
        # Random initial vector
        v = np.random.randn(n_records)
        v = v / np.linalg.norm(v)

        # Check if we have sparse matrix available
        if hasattr(self, '_constraint_matrix_sparse') and self._use_sparse_internally:
            A_sparse = self._constraint_matrix_sparse
            for _ in range(n_iter):
                # v = A.T @ (A @ v) without forming A.T @ A
                Av = A_sparse @ v
                v_new = A_sparse.T @ Av
                norm = np.linalg.norm(v_new)
                if norm < 1e-10:
                    return 1.0
                v = v_new / norm
            # Eigenvalue estimate
            Av = A_sparse @ v
            return np.dot(Av, Av)
        else:
            for _ in range(n_iter):
                # v = A.T @ (A @ v) without forming A.T @ A
                Av = A @ v
                v_new = A.T @ Av
                norm = np.linalg.norm(v_new)
                if norm < 1e-10:
                    return 1.0
                v = v_new / norm
            # Eigenvalue estimate
            Av = A @ v
            return np.dot(Av, Av)

    def _fista(
        self,
        A: np.ndarray,
        b: np.ndarray,
        lam: float,
        step_size: float,
        w0: np.ndarray,
    ) -> np.ndarray:
        """
        Cross-category sparse calibration.

        For overlapping constraints (e.g., state AND age), we must
        ensure the selected subset has the right JOINT distribution,
        not just the right marginals.

        Algorithm:
        1. Identify "cross-categories" (unique combinations of CATEGORICAL constraint memberships)
        2. For each cross-category, determine how many records needed
        3. Select proportionally from each cross-category
        4. Calibrate using IPF on the selected subset (including continuous)
        """
        n = len(w0)

        # Target number of records to keep
        k = max(1, int(n * np.exp(-lam)))

        self.convergence_history_ = []

        # Step 1: Identify cross-categories
        # ONLY use categorical constraints for cross-category grouping
        # Continuous constraints are handled in calibration step
        n_cat = getattr(self, '_n_categorical_constraints', len(b))
        n_constraints = len(b)

        # Create cross-category signature for each record
        # signature[j] = tuple of CATEGORICAL constraint indices that apply to record j
        from collections import defaultdict
        signatures = []
        for j in range(n):
            # Only include categorical constraints (first n_cat rows)
            sig = tuple(i for i in range(n_cat) if A[i, j] > 0)
            signatures.append(sig)

        # Group records by signature
        cross_cats = defaultdict(list)
        for j, sig in enumerate(signatures):
            cross_cats[sig].append(j)

        # Step 2: Determine selection proportion for each cross-category
        keep_fraction = k / n
        max_weight = 10.0  # Upper bound on weight per record

        # Only apply minimum coverage to categorical constraints
        min_per_constraint = np.zeros(n_constraints)
        min_per_constraint[:n_cat] = np.ceil(b[:n_cat] / max_weight)

        selected = np.zeros(n, dtype=bool)

        # For each cross-category, select proportionally
        for sig, indices in cross_cats.items():
            indices = np.array(indices)
            n_in_cat = len(indices)

            # How many to keep from this cross-category?
            n_keep = max(1, int(n_in_cat * keep_fraction))
            n_keep = min(n_keep, n_in_cat)

            # Random selection within cross-category
            np.random.shuffle(indices)
            selected[indices[:n_keep]] = True

        # Step 3: Verify minimum coverage for CATEGORICAL constraints
        for i in range(n_cat):
            row = A[i]
            in_constraint = (row > 0)
            n_selected_in = (selected & in_constraint).sum()
            min_needed = int(min_per_constraint[i])

            if n_selected_in < min_needed:
                # Need to add more from this constraint
                unselected_in = (~selected) & in_constraint
                candidates = np.where(unselected_in)[0]
                n_more = min(min_needed - n_selected_in, len(candidates))
                np.random.shuffle(candidates)
                selected[candidates[:n_more]] = True

        # Step 4: Initialize weights for selected records
        w = np.zeros(n)
        w[selected] = 1.0

        # Step 5: Calibrate using IPF-style iteration
        # First calibrate categorical, then adjust for continuous
        self.converged_ = False
        for iteration in range(self.max_iter):
            w_old = w.copy()

            # Categorical constraints: multiplicative adjustment
            for i in range(n_cat):
                row = A[i]
                current = (w * row).sum()

                if current > 1e-10:
                    adjustment = b[i] / current
                    mask = (row > 0) & selected
                    w[mask] *= adjustment

            # Continuous constraints: linear calibration (generalized raking)
            for i in range(n_cat, n_constraints):
                values = A[i]  # The actual variable values
                current = (w * values).sum()

                if abs(current) > 1e-10:
                    # Linear adjustment: w_new = w * (1 + lambda * x)
                    # where lambda is chosen so sum(w_new * x) = target
                    weighted_x2 = (w * values ** 2).sum()
                    if weighted_x2 > 1e-10:
                        lam_adj = (b[i] - current) / weighted_x2
                        # Apply with damping
                        adjustment = 1 + 0.5 * lam_adj * values  # Damped
                        adjustment = np.maximum(adjustment, 0.1)
                        w = w * adjustment
                        # Ensure zeros stay zero
                        w[~selected] = 0

            # Convergence check
            residual = A @ w - b
            change = np.linalg.norm(w - w_old) / max(np.linalg.norm(w_old), 1e-10)

            if iteration % 50 == 0 or change < self.tol:
                error = np.max(np.abs(residual) / np.maximum(np.abs(b), 1e-10))
                sparsity = (w < 1e-9).sum() / n
                self.convergence_history_.append({
                    "iteration": iteration + 1,
                    "max_error": error,
                    "change": change,
                    "sparsity": sparsity,
                    "n_selected": selected.sum(),
                })

            if change < self.tol and iteration > 5:
                self.n_iterations_ = iteration + 1
                self.converged_ = True
                break
        else:
            self.n_iterations_ = self.max_iter
            self.converged_ = False

        return w

    def _fit_with_target_sparsity(
        self,
        A: np.ndarray,
        b: np.ndarray,
        step_size: float,
        w0: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Directly compute k from target sparsity, then run IHT.

        With IHT, λ maps to k via k = n * exp(-λ).
        So for target sparsity s, we want k = n * (1 - s).
        Solving: λ = -ln(1 - s)

        Returns:
            (weights, lambda) tuple
        """
        n = len(w0)
        target = self.target_sparsity

        # Direct calculation: k = n * (1 - sparsity)
        k_target = int(n * (1 - target))
        k_target = max(1, k_target)  # At least 1 record

        # λ = -ln(k/n) = -ln(1 - sparsity)
        lam = -np.log(max(k_target / n, 1e-10))

        # Run IHT with this λ
        weights = self._fista(A, b, lam, step_size, w0)

        return weights, lam

    def transform(
        self,
        data: pd.DataFrame,
        weight_col: str = "weight",
    ) -> pd.DataFrame:
        """Apply fitted weights to data."""
        if not self.is_fitted_:
            raise ValueError("Not fitted. Call fit() first.")

        if len(data) != self.n_records_:
            raise ValueError(
                f"Data length ({len(data)}) doesn't match fitted ({self.n_records_})"
            )

        result = data.copy()
        result[weight_col] = self.weights_
        return result

    def fit_transform(
        self,
        data: pd.DataFrame,
        marginal_targets: dict[str, dict[str, float]],
        continuous_targets: dict[str, float] | None = None,
        weight_col: str = "weight",
        linear_constraints: tuple[LinearConstraint, ...] | list[LinearConstraint] | None = None,
    ) -> pd.DataFrame:
        """Fit and transform in one call."""
        self.fit(
            data,
            marginal_targets,
            continuous_targets,
            weight_col,
            linear_constraints=linear_constraints,
        )
        return self.transform(data, weight_col)

    def get_sparsity(self) -> float:
        """Get fraction of zero weights."""
        if not self.is_fitted_:
            raise ValueError("Not fitted.")
        return (self.weights_ < 1e-9).sum() / self.n_records_

    def get_n_nonzero(self) -> int:
        """Get count of non-zero weights."""
        if not self.is_fitted_:
            raise ValueError("Not fitted.")
        return (self.weights_ >= 1e-9).sum()

    def validate(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Validate calibration accuracy.

        Returns dict with per-target errors and overall metrics.
        """
        if not self.is_fitted_:
            raise ValueError("Not fitted.")

        weights = self.weights_
        results = {
            "targets": {},
            "marginal_errors": {},
            "continuous_errors": {},
            "linear_errors": {},
            "sparsity": self.get_sparsity(),
            "converged": self.converged_,
        }

        # Check marginal targets
        if self.marginal_targets_:
            for var, var_targets in self.marginal_targets_.items():
                results["marginal_errors"][var] = {}
                for category, target in var_targets.items():
                    mask = data[var] == category
                    actual = weights[mask].sum()
                    rel_error = abs(actual - target) / target if target > 0 else 0
                    info = {
                        "actual": actual,
                        "target": target,
                        "relative_error": rel_error,
                    }
                    results["marginal_errors"][var][category] = info
                    results["targets"][f"{var}={category}"] = {
                        **info,
                        "error": rel_error,
                    }

        # Check continuous targets
        if self.continuous_targets_:
            for var, target in self.continuous_targets_.items():
                actual = (weights * data[var].values).sum()
                rel_error = abs(actual - target) / abs(target) if target != 0 else 0
                info = {
                    "actual": actual,
                    "target": target,
                    "relative_error": rel_error,
                }
                results["continuous_errors"][var] = info
                results["targets"][var] = {
                    **info,
                    "error": rel_error,
                }

        for constraint in self.linear_constraints_:
            actual = float(weights @ constraint.coefficients)
            target = float(constraint.target)
            rel_error = abs(actual - target) / abs(target) if target != 0 else 0
            results["linear_errors"][constraint.name] = {
                "actual": actual,
                "target": target,
                "relative_error": rel_error,
            }

        errors = [t["error"] for t in results["targets"].values()]
        errors.extend(
            item["relative_error"] for item in results["linear_errors"].values()
        )
        results["max_error"] = max(errors) if errors else 0
        results["mean_error"] = np.mean(errors) if errors else 0
        results["rmse"] = self.calibration_error_

        return results

    def get_pareto_frontier(
        self,
        data: pd.DataFrame,
        marginal_targets: dict[str, dict[str, float]],
        continuous_targets: dict[str, float] | None = None,
        n_points: int = 20,
    ) -> pd.DataFrame:
        """
        Compute Pareto frontier of sparsity vs accuracy.

        Useful for choosing the right sparsity-accuracy tradeoff.

        Args:
            data: DataFrame with microdata
            marginal_targets: Categorical targets
            continuous_targets: Continuous targets
            n_points: Number of points on the frontier

        Returns:
            DataFrame with columns: lambda, sparsity, max_error, n_nonzero
        """
        # Build constraints once
        A, b, _ = self._build_constraints(data, marginal_targets, continuous_targets)

        # Normalize each row by its target value
        if self.normalize_targets:
            row_scales = np.abs(b).copy()
            row_scales[row_scales < 1e-10] = 1.0
            A_norm = A / row_scales[:, np.newaxis]
            b_norm = b / row_scales
        else:
            A_norm = A
            b_norm = b

        # Compute step size using power iteration (memory-efficient)
        L = self._estimate_lipschitz(A_norm)
        step_size = 1.0 / max(L, 1e-10)

        w0 = np.ones(len(data))

        # Sweep λ values
        lam_values = np.logspace(-6, 1, n_points)
        results = []

        for lam in lam_values:
            weights = self._fista(A_norm, b_norm, lam, step_size, w0)

            sparsity = (weights < 1e-9).sum() / len(weights)
            n_nonzero = (weights >= 1e-9).sum()

            # Compute error using original scale
            residual = A @ weights - b
            rel_errors = np.abs(residual) / np.maximum(np.abs(b), 1e-10)
            max_error = rel_errors.max()

            results.append({
                "lambda": lam,
                "sparsity": sparsity,
                "max_error": max_error,
                "n_nonzero": n_nonzero,
            })

            # Stop if fully sparse
            if sparsity > 0.99:
                break

        return pd.DataFrame(results)


class HardConcreteCalibrator:
    """
    L0-regularized calibration using Hard Concrete distribution.

    Uses gradient descent with differentiable L0 penalty (Hard Concrete gates)
    to jointly optimize calibration accuracy and sparsity. Unlike SparseCalibrator
    which uses deterministic cross-category selection, this approach learns which
    records are important for hitting targets.

    Based on Louizos, Welling & Kingma (2017) "Learning Sparse Neural Networks
    through L0 Regularization".

    Key features:
    - End-to-end differentiable sparsity
    - Automatic tradeoff between accuracy and sparsity
    - Handles both categorical and continuous targets uniformly
    - Learns which records matter for calibration

    Example:
        >>> from microplex.calibration import HardConcreteCalibrator
        >>> calibrator = HardConcreteCalibrator(lambda_l0=1e-5)
        >>> targets = {"state": {"CA": 1000, "NY": 500}}
        >>> calibrated = calibrator.fit_transform(data, targets)
        >>> print(f"Sparsity: {calibrator.get_sparsity():.1%}")
    """

    def __init__(
        self,
        lambda_l0: float = 1e-5,
        lambda_l2: float = 0.0,
        lr: float = 0.1,
        epochs: int = 2000,
        init_keep_prob: float = 0.99,
        loss_type: str = "relative",
        normalize_targets: bool = True,
        verbose: bool = False,
        verbose_freq: int = 200,
        device: str = "cpu",
    ):
        """
        Initialize Hard Concrete calibrator.

        Args:
            lambda_l0: L0 penalty strength. Higher = more sparsity.
                Typical range: 1e-7 to 1e-4
            lambda_l2: L2 penalty on weight magnitudes. Prevents explosion.
            lr: Learning rate for Adam optimizer.
            epochs: Number of optimization epochs.
            init_keep_prob: Initial probability of keeping each weight.
                Start high (0.99) and let optimizer prune.
            loss_type: "relative" for scale-invariant loss, "mse" for absolute.
            normalize_targets: Normalize targets to similar scale before fitting.
            verbose: Print progress during fitting.
            verbose_freq: Print every N epochs.
            device: "cpu" or "cuda" for GPU acceleration.
        """
        self.lambda_l0 = lambda_l0
        self.lambda_l2 = lambda_l2
        self.lr = lr
        self.epochs = epochs
        self.init_keep_prob = init_keep_prob
        self.loss_type = loss_type
        self.normalize_targets = normalize_targets
        self.verbose = verbose
        self.verbose_freq = verbose_freq
        self.device = device

        # Set during fit
        self.model_: Any | None = None
        self.weights_: np.ndarray | None = None
        self.is_fitted_: bool = False
        self.n_records_: int | None = None
        self.marginal_targets_: dict[str, dict[str, float]] | None = None
        self.continuous_targets_: dict[str, float] | None = None
        self.linear_constraints_: tuple[LinearConstraint, ...] = ()
        self.calibration_error_: float = 0.0
        self.max_error_: float = 0.0
        self.converged_: bool = False

    def fit(
        self,
        data: pd.DataFrame,
        marginal_targets: dict[str, dict[str, float]],
        continuous_targets: dict[str, float] | None = None,
        weight_col: str = "weight",
        linear_constraints: tuple[LinearConstraint, ...] | list[LinearConstraint] | None = None,
    ) -> Self:
        """
        Fit Hard Concrete calibration weights.

        Args:
            data: DataFrame with microdata records
            marginal_targets: Dict of categorical targets {var: {category: count}}
            continuous_targets: Dict of continuous totals {var: total}
            weight_col: Name of initial weight column (for initialization)

        Returns:
            self
        """
        try:
            from l0.calibration import SparseCalibrationWeights
        except ImportError:
            raise ImportError(
                "l0-python package required for HardConcreteCalibrator. "
                "Install with: pip install microplex[l0] or pip install l0-python. "
                "Requires Python 3.11+."
            )

        from scipy import sparse as sp

        self.n_records_ = len(data)
        self.marginal_targets_ = marginal_targets
        self.continuous_targets_ = continuous_targets or {}
        self.linear_constraints_ = tuple(linear_constraints or ())
        _validate_calibration_inputs(
            data,
            marginal_targets,
            continuous_targets,
            self.linear_constraints_,
        )

        # Build constraint matrix and targets
        A, b, target_names = self._build_constraints(
            data,
            marginal_targets,
            continuous_targets,
            self.linear_constraints_,
        )

        # Normalize targets for better optimization
        if self.normalize_targets:
            self._row_scales = np.abs(b).copy()
            self._row_scales[self._row_scales < 1e-10] = 1.0
            b_norm = b / self._row_scales
            A_norm = A / self._row_scales[:, np.newaxis]
        else:
            self._row_scales = np.ones(len(b))
            A_norm = A
            b_norm = b

        # Convert to sparse matrix
        A_sparse = sp.csr_matrix(A_norm)

        # Get initial weights
        if weight_col in data.columns:
            init_weights = data[weight_col].values.astype(float)
        else:
            init_weights = np.ones(len(data))

        # Rescale init weights so A_norm @ init_weights ≈ b_norm.
        # Without this, survey weights (e.g. CPS ~6000) produce initial
        # constraint violations of 1000x+, making gradient descent fail.
        achieved = A_norm @ init_weights
        positive = achieved > 1e-10
        if positive.any():
            scale = np.mean(b_norm[positive] / achieved[positive])
            init_weights = init_weights * scale

        # Create and fit model
        self.model_ = SparseCalibrationWeights(
            n_features=len(data),
            init_keep_prob=self.init_keep_prob,
            init_weights=init_weights,
            device=self.device,
        )

        self.model_.fit(
            M=A_sparse,
            y=b_norm,
            lambda_l0=self.lambda_l0,
            lambda_l2=self.lambda_l2,
            lr=self.lr,
            epochs=self.epochs,
            loss_type=self.loss_type,
            verbose=self.verbose,
            verbose_freq=self.verbose_freq,
        )

        # Extract final weights
        import torch
        with torch.no_grad():
            self.weights_ = self.model_.get_weights(deterministic=True).cpu().numpy()

        residual = A @ self.weights_ - b
        rel_errors = np.abs(residual) / np.maximum(np.abs(b), 1e-10)
        self.calibration_error_ = float(np.sqrt(np.mean(rel_errors ** 2)))
        self.max_error_ = float(rel_errors.max()) if len(rel_errors) else 0.0
        self.converged_ = bool(np.isfinite(self.max_error_) and self.max_error_ <= 0.05)
        self.is_fitted_ = True
        return self

    def _build_constraints(
        self,
        data: pd.DataFrame,
        marginal_targets: dict[str, dict[str, float]],
        continuous_targets: dict[str, float] | None,
        linear_constraints: tuple[LinearConstraint, ...],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build constraint matrix A, target vector b, and target names.

        Uses sparse construction for memory efficiency with many categorical targets.
        """
        from scipy import sparse

        n_records = len(data)
        rows = []
        cols = []
        vals = []
        targets = []
        names = []
        row_idx = 0

        # Categorical constraints - build as sparse
        for var, var_targets in marginal_targets.items():
            if var not in data.columns:
                raise ValueError(f"Variable '{var}' not in data")

            for category, target in var_targets.items():
                mask = data[var] == category
                indices = np.where(mask)[0]
                rows.extend([row_idx] * len(indices))
                cols.extend(indices)
                vals.extend([1.0] * len(indices))
                targets.append(target)
                names.append(f"{var}={category}")
                row_idx += 1

        # Continuous constraints - these are dense rows
        continuous_rows = []
        if continuous_targets:
            for var, target in continuous_targets.items():
                if var not in data.columns:
                    raise ValueError(f"Variable '{var}' not in data")
                continuous_rows.append(data[var].values.astype(float))
                targets.append(target)
                names.append(var)

        linear_rows = []
        for constraint in linear_constraints:
            linear_rows.append(np.asarray(constraint.coefficients, dtype=float))
            targets.append(float(constraint.target))
            names.append(constraint.name)

        b = np.array(targets, dtype=float)
        n_cat = row_idx
        n_cont = len(continuous_rows)
        n_linear = len(linear_rows)
        n_constraints = n_cat + n_cont + n_linear

        if n_constraints == 0:
            return np.zeros((0, n_records)), b, names

        # Build sparse matrix for categorical constraints
        A_cat_sparse = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(n_cat, n_records), dtype=float
        )

        if continuous_rows:
            # Stack with dense continuous rows
            A_cont = np.vstack(continuous_rows)
            sparse_blocks = [A_cat_sparse, sparse.csr_matrix(A_cont)]
        else:
            sparse_blocks = [A_cat_sparse]

        if linear_rows:
            A_linear = np.vstack(linear_rows)
            sparse_blocks.append(sparse.csr_matrix(A_linear))

        A_sparse = sparse.vstack(sparse_blocks)

        # Return as dense for backward compatibility
        # The L0 package will convert back to sparse internally
        return A_sparse.toarray(), b, names

    def transform(
        self,
        data: pd.DataFrame,
        weight_col: str = "weight",
    ) -> pd.DataFrame:
        """Apply fitted weights to data."""
        if not self.is_fitted_:
            raise ValueError("Not fitted. Call fit() first.")

        if len(data) != self.n_records_:
            raise ValueError(
                f"Data length ({len(data)}) doesn't match fitted ({self.n_records_})"
            )

        result = data.copy()
        result[weight_col] = self.weights_
        return result

    def fit_transform(
        self,
        data: pd.DataFrame,
        marginal_targets: dict[str, dict[str, float]],
        continuous_targets: dict[str, float] | None = None,
        weight_col: str = "weight",
        linear_constraints: tuple[LinearConstraint, ...] | list[LinearConstraint] | None = None,
    ) -> pd.DataFrame:
        """Fit and transform in one call."""
        self.fit(
            data,
            marginal_targets,
            continuous_targets,
            weight_col,
            linear_constraints=linear_constraints,
        )
        return self.transform(data, weight_col)

    def get_sparsity(self) -> float:
        """Get fraction of zero weights."""
        if not self.is_fitted_:
            raise ValueError("Not fitted.")
        return (self.weights_ < 1e-9).sum() / self.n_records_

    def get_n_nonzero(self) -> int:
        """Get count of non-zero weights."""
        if not self.is_fitted_:
            raise ValueError("Not fitted.")
        return int((self.weights_ >= 1e-9).sum())

    def validate(self, data: pd.DataFrame) -> dict[str, Any]:
        """Validate calibration accuracy."""
        if not self.is_fitted_:
            raise ValueError("Not fitted.")

        weights = self.weights_
        results = {
            "targets": {},
            "marginal_errors": {},
            "continuous_errors": {},
            "linear_errors": {},
            "sparsity": self.get_sparsity(),
            "converged": self.converged_,
        }

        # Check marginal targets
        if self.marginal_targets_:
            for var, var_targets in self.marginal_targets_.items():
                results["marginal_errors"][var] = {}
                for category, target in var_targets.items():
                    mask = data[var] == category
                    actual = weights[mask].sum()
                    rel_error = abs(actual - target) / target if target > 0 else 0
                    info = {
                        "actual": actual,
                        "target": target,
                        "relative_error": rel_error,
                    }
                    results["marginal_errors"][var][category] = info
                    results["targets"][f"{var}={category}"] = {
                        **info,
                        "error": rel_error,
                    }

        # Check continuous targets
        if self.continuous_targets_:
            for var, target in self.continuous_targets_.items():
                actual = (weights * data[var].values).sum()
                rel_error = abs(actual - target) / abs(target) if target != 0 else 0
                info = {
                    "actual": actual,
                    "target": target,
                    "relative_error": rel_error,
                }
                results["continuous_errors"][var] = info
                results["targets"][var] = {
                    **info,
                    "error": rel_error,
                }

        for constraint in self.linear_constraints_:
            actual = float(weights @ constraint.coefficients)
            target = float(constraint.target)
            rel_error = abs(actual - target) / abs(target) if target != 0 else 0
            results["linear_errors"][constraint.name] = {
                "actual": actual,
                "target": target,
                "relative_error": rel_error,
            }

        errors = [t["error"] for t in results["targets"].values()]
        errors.extend(
            item["relative_error"] for item in results["linear_errors"].values()
        )
        results["max_error"] = max(errors) if errors else 0
        results["mean_error"] = np.mean(errors) if errors else 0
        results["rmse"] = self.calibration_error_

        return results



# --- Optional microcalibrate adapter ---------------------------------------
# Imported when `microplex[calibrate]` extra is installed. The adapter
# pulls in torch/optuna/l0 (~1.5 GB) which microplex does not otherwise
# require, so we keep it opt-in. Callers that need identity-preserving
# gradient-descent chi-squared calibration install the extra and import
# `MicrocalibrateAdapter` / `MicrocalibrateAdapterConfig` from this
# namespace.
try:
    from microplex.calibration.microcalibrate_adapter import (  # noqa: F401
        EntityTableBundleCalibrationResult,
        EntityTableBundleMicrocalibrator,
        MicrocalibrateAdapter,
        MicrocalibrateAdapterConfig,
        SparseTargetMatrixCalibrationResult,
    )
except ImportError:  # microcalibrate not installed
    EntityTableBundleCalibrationResult = None  # type: ignore[assignment]
    EntityTableBundleMicrocalibrator = None  # type: ignore[assignment]
    MicrocalibrateAdapter = None  # type: ignore[assignment]
    MicrocalibrateAdapterConfig = None  # type: ignore[assignment]
    SparseTargetMatrixCalibrationResult = None  # type: ignore[assignment]
