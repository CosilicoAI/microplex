from __future__ import annotations

import pytest

from microplex.calibration import (
    CalibrationSolvePolicy,
    resolve_calibration_solve_policy,
)
from microplex.spec import CalibrateSpec, CalibrationMethod


def _calibrate(
    method: CalibrationMethod = CalibrationMethod.APG,
    target_records: int | None = None,
) -> CalibrateSpec:
    return CalibrateSpec(
        loss="pe_native_bucketed_huber_v1",
        method=method,
        target_records=target_records,
    )


def test_apg_without_pruning_is_dense_microcalibrate() -> None:
    policy = resolve_calibration_solve_policy(
        _calibrate(),
        n_records=100,
        target_count=10,
    )

    assert isinstance(policy, CalibrationSolvePolicy)
    assert policy.solver == "microcalibrate_apg"
    assert policy.effective_record_count == 100
    assert policy.target_sparsity is None
    assert policy.microcalibrate_config_overrides() == {
        "regularize_with_l0": False,
    }
    assert policy.diagnostics()["method"] == "apg"


def test_apg_target_records_enables_l0_prune() -> None:
    policy = resolve_calibration_solve_policy(
        _calibrate(target_records=25),
        n_records=100,
        target_count=10,
    )

    assert policy.solver == "microcalibrate_apg_l0_prune"
    assert policy.regularize_with_l0 is True
    assert policy.effective_record_count == 25
    assert policy.target_sparsity == 0.75


def test_l0_method_requires_target_records() -> None:
    with pytest.raises(ValueError, match="requires target_records"):
        resolve_calibration_solve_policy(
            _calibrate(CalibrationMethod.L0),
            n_records=100,
            target_count=10,
        )


def test_ipf_rejects_target_records() -> None:
    with pytest.raises(ValueError, match="ipf.*target_records"):
        resolve_calibration_solve_policy(
            _calibrate(CalibrationMethod.IPF, target_records=50),
            n_records=100,
            target_count=10,
        )


def test_target_records_cannot_exceed_available_records() -> None:
    with pytest.raises(ValueError, match="cannot exceed n_records"):
        resolve_calibration_solve_policy(
            _calibrate(target_records=101),
            n_records=100,
            target_count=10,
        )


def test_rejects_empty_target_surface() -> None:
    with pytest.raises(ValueError, match="target_count must be positive"):
        resolve_calibration_solve_policy(
            _calibrate(),
            n_records=100,
            target_count=0,
        )


def test_optional_records_per_target_floor_fails_closed() -> None:
    with pytest.raises(ValueError, match="too few records"):
        resolve_calibration_solve_policy(
            _calibrate(target_records=20),
            n_records=100,
            target_count=10,
            min_records_per_target=3,
        )
