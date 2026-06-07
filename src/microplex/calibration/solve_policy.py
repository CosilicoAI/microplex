"""Fail-closed calibration solve policy for spec-driven builds."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from microplex.spec import CalibrateSpec, CalibrationMethod

__all__ = [
    "CalibrationSolvePolicy",
    "resolve_calibration_solve_policy",
]


@dataclass(frozen=True)
class CalibrationSolvePolicy:
    """Resolved solver/pruning policy for one calibration target surface."""

    loss: str
    method: CalibrationMethod
    n_records: int
    target_count: int
    target_records: int | None
    regularize_with_l0: bool
    target_sparsity: float | None
    solver: str

    @property
    def effective_record_count(self) -> int:
        """Records the solve is expected to keep active."""
        return self.target_records or self.n_records

    def microcalibrate_config_overrides(self) -> dict[str, Any]:
        """Return config overrides for the microcalibrate adapter."""
        return {"regularize_with_l0": self.regularize_with_l0}

    def diagnostics(self) -> dict[str, Any]:
        """Stable diagnostics payload for manifests and calibration logs."""
        return {
            "loss": self.loss,
            "method": self.method.value,
            "solver": self.solver,
            "n_records": self.n_records,
            "target_count": self.target_count,
            "target_records": self.target_records,
            "effective_record_count": self.effective_record_count,
            "regularize_with_l0": self.regularize_with_l0,
            "target_sparsity": self.target_sparsity,
        }


def resolve_calibration_solve_policy(
    calibrate: CalibrateSpec,
    *,
    n_records: int,
    target_count: int,
    min_records_per_target: float | None = None,
) -> CalibrationSolvePolicy:
    """Resolve and validate the calibration solve policy.

    ``calibrate.method`` chooses the primary solver. ``target_records`` is the
    explicit L0/pruning knob: it may be paired with APG for "APG + L0 prune",
    and it is required when the method itself is ``l0``. IPF is kept dense so a
    stale spec cannot silently request incompatible pruning.
    """
    if n_records <= 0:
        raise ValueError(f"n_records must be positive; got {n_records}.")
    if target_count <= 0:
        raise ValueError(f"target_count must be positive; got {target_count}.")

    target_records = calibrate.target_records
    if target_records is not None and target_records > n_records:
        raise ValueError(
            "calibrate.target_records cannot exceed n_records; "
            f"got target_records={target_records}, n_records={n_records}."
        )
    if calibrate.method is CalibrationMethod.L0 and target_records is None:
        raise ValueError("calibrate.method='l0' requires target_records.")
    if calibrate.method is CalibrationMethod.IPF and target_records is not None:
        raise ValueError("calibrate.method='ipf' does not support target_records.")

    effective_records = target_records or n_records
    if min_records_per_target is not None:
        if min_records_per_target <= 0:
            raise ValueError(
                "min_records_per_target must be positive when provided; "
                f"got {min_records_per_target}."
            )
        minimum_records = math.ceil(target_count * min_records_per_target)
        if effective_records < minimum_records:
            raise ValueError(
                "calibration solve has too few records for the target surface; "
                f"effective_records={effective_records}, target_count={target_count}, "
                f"min_records_per_target={min_records_per_target}."
            )

    regularize_with_l0 = target_records is not None
    target_sparsity = (
        1.0 - (float(target_records) / float(n_records))
        if target_records is not None
        else None
    )
    solver = _solver_name(calibrate.method, regularize_with_l0=regularize_with_l0)

    return CalibrationSolvePolicy(
        loss=calibrate.loss,
        method=calibrate.method,
        n_records=n_records,
        target_count=target_count,
        target_records=target_records,
        regularize_with_l0=regularize_with_l0,
        target_sparsity=target_sparsity,
        solver=solver,
    )


def _solver_name(
    method: CalibrationMethod,
    *,
    regularize_with_l0: bool,
) -> str:
    if method is CalibrationMethod.IPF:
        return "ipf"
    if method is CalibrationMethod.L0:
        return "microcalibrate_apg_l0"
    if regularize_with_l0:
        return "microcalibrate_apg_l0_prune"
    return "microcalibrate_apg"
