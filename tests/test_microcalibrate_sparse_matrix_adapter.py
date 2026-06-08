from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from microplex.core import EntityType
from microplex.spec import CalibrateSpec, CalibrationMethod
from microplex.targets import (
    EntityTableBinding,
    EntityTableBundle,
    SparseTargetMatrix,
    TargetAggregation,
    TargetSet,
    TargetSpec,
)


@pytest.fixture(autouse=True)
def _discard_fake_adapter_module():
    yield
    sys.modules.pop("microplex.calibration.microcalibrate_adapter", None)


def _load_adapter_with_fake_calibration(monkeypatch, captured: dict):
    class FakeCalibration:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            self.weights = np.asarray(kwargs["weights"], dtype=float) + 1.0

        def calibrate(self):
            return pd.DataFrame({"loss": [0.0]})

        def estimate(self):
            estimate_matrix = captured["kwargs"]["estimate_matrix"]
            return pd.Series(
                np.asarray(estimate_matrix.to_numpy(dtype=float)).T @ self.weights
            )

    fake_microcalibrate = types.ModuleType("microcalibrate")
    fake_microcalibrate.Calibration = FakeCalibration
    monkeypatch.setitem(sys.modules, "microcalibrate", fake_microcalibrate)
    sys.modules.pop("microplex.calibration.microcalibrate_adapter", None)
    return importlib.import_module("microplex.calibration.microcalibrate_adapter")


def test_microcalibrate_adapter_fits_certified_sparse_target_matrix(
    monkeypatch,
) -> None:
    captured: dict = {}
    adapter_module = _load_adapter_with_fake_calibration(monkeypatch, captured)

    target_matrix = SparseTargetMatrix(
        matrix=sparse.csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])),
        target_vector=np.array([10.0, 20.0]),
        names=("income", "count"),
    )
    adapter = adapter_module.MicrocalibrateAdapter()

    weights = adapter.fit_sparse_target_matrix(
        np.array([1.0, 2.0, 3.0]),
        target_matrix,
        certificate=target_matrix.certificate(),
    )

    np.testing.assert_array_equal(weights, np.array([2.0, 3.0, 4.0]))
    kwargs = captured["kwargs"]
    assert kwargs["target_names"].tolist() == ["income", "count"]
    np.testing.assert_array_equal(kwargs["targets"], np.array([10.0, 20.0]))
    assert list(kwargs["estimate_matrix"].columns) == ["income", "count"]
    assert hasattr(kwargs["estimate_matrix"], "sparse")
    np.testing.assert_array_equal(
        kwargs["estimate_matrix"].sparse.to_coo().toarray(),
        target_matrix.matrix.transpose().toarray(),
    )


def test_microcalibrate_adapter_rejects_certificate_mismatch_before_fit(
    monkeypatch,
) -> None:
    captured: dict = {}
    adapter_module = _load_adapter_with_fake_calibration(monkeypatch, captured)

    target_matrix = SparseTargetMatrix(
        matrix=sparse.csr_matrix(np.array([[1.0, 0.0]])),
        target_vector=np.array([1.0]),
        names=("target",),
    )
    bad_certificate = target_matrix.certificate().to_dict()
    bad_certificate["names_sha256"] = "bad"

    adapter = adapter_module.MicrocalibrateAdapter()
    with pytest.raises(ValueError, match="Sparse target matrix certificate mismatch"):
        adapter.fit_sparse_target_matrix(
            np.array([1.0, 1.0]),
            target_matrix,
            certificate=bad_certificate,
        )

    assert "kwargs" not in captured


def test_microcalibrate_adapter_rejects_weight_length_mismatch(monkeypatch) -> None:
    captured: dict = {}
    adapter_module = _load_adapter_with_fake_calibration(monkeypatch, captured)
    target_matrix = SparseTargetMatrix(
        matrix=sparse.csr_matrix(np.array([[1.0, 0.0]])),
        target_vector=np.array([1.0]),
        names=("target",),
    )

    adapter = adapter_module.MicrocalibrateAdapter()
    with pytest.raises(ValueError, match="initial_weights length"):
        adapter.fit_sparse_target_matrix(np.array([1.0]), target_matrix)

    assert "kwargs" not in captured


def test_microcalibrate_adapter_applies_sparse_matrix_solve_policy(
    monkeypatch,
) -> None:
    captured: dict = {}
    adapter_module = _load_adapter_with_fake_calibration(monkeypatch, captured)

    target_matrix = SparseTargetMatrix(
        matrix=sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]])),
        target_vector=np.array([5.0, 8.0]),
        names=("count", "income"),
    )
    adapter = adapter_module.MicrocalibrateAdapter()

    result = adapter.fit_sparse_target_matrix_with_policy(
        np.array([1.0, 2.0]),
        target_matrix,
        calibrate=CalibrateSpec(
            loss="pe_native_bucketed_huber_v1",
            method=CalibrationMethod.APG,
            target_records=1,
        ),
        certificate=target_matrix.certificate(),
    )

    np.testing.assert_array_equal(result.weights, np.array([2.0, 3.0]))
    assert result.policy.solver == "microcalibrate_apg_l0_prune"
    assert result.policy.target_records == 1
    assert result.validation["converged"] is False
    assert result.certificate.to_dict() == target_matrix.certificate().to_dict()
    assert result.diagnostics()["policy"]["regularize_with_l0"] is True
    assert captured["kwargs"]["regularize_with_l0"] is True
    assert adapter.config.regularize_with_l0 is False


def test_microcalibrate_adapter_policy_rejects_empty_target_surface_before_fit(
    monkeypatch,
) -> None:
    captured: dict = {}
    adapter_module = _load_adapter_with_fake_calibration(monkeypatch, captured)
    target_matrix = SparseTargetMatrix(
        matrix=sparse.csr_matrix((0, 2)),
        target_vector=np.array([]),
        names=(),
    )

    adapter = adapter_module.MicrocalibrateAdapter()
    with pytest.raises(ValueError, match="target_count must be positive"):
        adapter.fit_sparse_target_matrix_with_policy(
            np.array([1.0, 2.0]),
            target_matrix,
            calibrate=CalibrateSpec(
                loss="pe_native_bucketed_huber_v1",
                method=CalibrationMethod.APG,
            ),
        )

    assert "kwargs" not in captured


def _entity_table_bundle() -> EntityTableBundle:
    households = pd.DataFrame(
        {
            "household_id": [10, 20],
            "household_weight": [1.0, 2.0],
        }
    )
    persons = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "household_id": [10, 10, 20],
            "employment_income": [2.0, 3.0, 4.0],
        }
    )
    return EntityTableBundle(
        weight_entity=EntityType.HOUSEHOLD,
        weight_column="household_weight",
        bindings={
            EntityType.HOUSEHOLD: EntityTableBinding(
                frame=households,
                id_column="household_id",
            ),
            EntityType.PERSON: EntityTableBinding(
                frame=persons,
                id_column="person_id",
                weight_link_column="household_id",
                synced_weight_column="person_weight",
            ),
        },
    )


def _income_target_set() -> TargetSet:
    return TargetSet(
        [
            TargetSpec(
                name="employment_income_sum",
                entity=EntityType.PERSON,
                value=20.0,
                period=2024,
                measure="employment_income",
                aggregation=TargetAggregation.SUM,
            )
        ]
    )


def _income_target_with_missing_feature_set() -> TargetSet:
    return TargetSet(
        [
            TargetSpec(
                name="employment_income_sum",
                entity=EntityType.PERSON,
                value=20.0,
                period=2024,
                measure="employment_income",
                aggregation=TargetAggregation.SUM,
            ),
            TargetSpec(
                name="missing_feature_sum",
                entity=EntityType.PERSON,
                value=1.0,
                period=2024,
                measure="not_present",
                aggregation=TargetAggregation.SUM,
            ),
        ]
    )


def test_entity_table_bundle_microcalibrator_compiles_and_syncs_weights(
    monkeypatch,
) -> None:
    captured: dict = {}
    adapter_module = _load_adapter_with_fake_calibration(monkeypatch, captured)

    calibrator = adapter_module.EntityTableBundleMicrocalibrator()
    result = calibrator.calibrate_bundle(
        _entity_table_bundle(),
        target_set=_income_target_set(),
        calibrate=CalibrateSpec(
            loss="pe_native_bucketed_huber_v1",
            method=CalibrationMethod.APG,
        ),
    )

    household_weights = result.bundle.table_for(EntityType.HOUSEHOLD)[
        "household_weight"
    ]
    person_weights = result.bundle.table_for(EntityType.PERSON)["person_weight"]

    np.testing.assert_array_equal(household_weights.to_numpy(), np.array([2.0, 3.0]))
    np.testing.assert_array_equal(person_weights.to_numpy(), np.array([2.0, 2.0, 3.0]))
    assert result.target_matrix.names == ("employment_income_sum",)
    np.testing.assert_array_equal(
        result.target_matrix.matrix.toarray(),
        np.array([[5.0, 4.0]]),
    )
    assert result.diagnostics()["weight_entity"] == "household"
    assert result.diagnostics()["policy"]["solver"] == "microcalibrate_apg"
    assert result.diagnostics()["skipped_targets"] == []
    assert captured["kwargs"]["target_names"].tolist() == ["employment_income_sum"]


def test_entity_table_bundle_microcalibrator_rejects_stale_certificate_before_fit(
    monkeypatch,
) -> None:
    captured: dict = {}
    adapter_module = _load_adapter_with_fake_calibration(monkeypatch, captured)

    bundle = _entity_table_bundle()
    target_set = _income_target_set()
    calibrator = adapter_module.EntityTableBundleMicrocalibrator()
    first = calibrator.calibrate_bundle(
        bundle,
        target_set=target_set,
        calibrate=CalibrateSpec(
            loss="pe_native_bucketed_huber_v1",
            method=CalibrationMethod.APG,
        ),
    )
    stale_certificate = first.target_matrix.certificate().to_dict()
    stale_certificate["target_vector_sha256"] = "stale"
    captured.clear()

    with pytest.raises(ValueError, match="Sparse target matrix certificate mismatch"):
        calibrator.calibrate_bundle(
            bundle,
            target_set=target_set,
            calibrate=CalibrateSpec(
                loss="pe_native_bucketed_huber_v1",
                method=CalibrationMethod.APG,
            ),
            certificate=stale_certificate,
        )

    assert "kwargs" not in captured


def test_entity_table_bundle_microcalibrator_rejects_skipped_targets_before_fit(
    monkeypatch,
) -> None:
    captured: dict = {}
    adapter_module = _load_adapter_with_fake_calibration(monkeypatch, captured)

    calibrator = adapter_module.EntityTableBundleMicrocalibrator()
    with pytest.raises(ValueError) as excinfo:
        calibrator.calibrate_bundle(
            _entity_table_bundle(),
            target_set=_income_target_with_missing_feature_set(),
            calibrate=CalibrateSpec(
                loss="pe_native_bucketed_huber_v1",
                method=CalibrationMethod.APG,
            ),
        )

    message = str(excinfo.value)
    assert "Sparse target compilation skipped target(s)" in message
    assert "missing_feature_sum" in message
    assert "missing_features:not_present" in message
    assert "kwargs" not in captured
