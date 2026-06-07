from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from microplex.targets import SparseTargetMatrix


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


def test_microcalibrate_adapter_fits_certified_sparse_target_matrix(monkeypatch) -> None:
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
