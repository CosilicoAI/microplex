"""Tests for the spec TransformEngine (microplex.spec_transforms).

Covers: fractional split sums back to the source, split fraction values,
expression splits, derive evaluation, sequential application (derive sees a
split output), and error cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microplex.spec import DeriveTransform, SplitTransform, TransformSpec
from microplex.spec_transforms import TransformEngine


def _frame(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "social_security": rng.uniform(0, 30000, n),
            "employment_income": rng.uniform(0, 80000, n),
            "interest_income": rng.uniform(0, 5000, n),
        }
    )


def _split_rule() -> TransformSpec:
    return TransformSpec(
        split=SplitTransform(
            source="social_security",
            into={
                "ss_retirement": 0.7,
                "ss_disability": 0.2,
                "ss_survivors": 0.1,
            },
        )
    )


class TestSplit:
    def test_split_sums_back_to_source(self) -> None:
        frame = _frame()
        engine = TransformEngine()
        out = engine.apply(frame, [_split_rule()])
        recombined = out["ss_retirement"] + out["ss_disability"] + out["ss_survivors"]
        np.testing.assert_allclose(
            recombined.to_numpy(), frame["social_security"].to_numpy(), atol=1e-9
        )

    def test_split_fraction_values(self) -> None:
        frame = _frame()
        out = TransformEngine().apply(frame, [_split_rule()])
        np.testing.assert_allclose(
            out["ss_retirement"].to_numpy(),
            frame["social_security"].to_numpy() * 0.7,
        )
        np.testing.assert_allclose(
            out["ss_disability"].to_numpy(),
            frame["social_security"].to_numpy() * 0.2,
        )

    def test_split_preserves_source_column(self) -> None:
        frame = _frame()
        out = TransformEngine().apply(frame, [_split_rule()])
        # Source column is retained (split adds pieces, doesn't drop source).
        np.testing.assert_array_equal(
            out["social_security"].to_numpy(),
            frame["social_security"].to_numpy(),
        )

    def test_expression_split(self) -> None:
        frame = _frame()
        rule = TransformSpec(
            split=SplitTransform(
                source="social_security",
                into={
                    "ss_half": "social_security * 0.5",
                    "ss_rest": "social_security - social_security * 0.5",
                },
            )
        )
        # Expression splits are not fractional -> no automatic sum-back check,
        # but these happen to sum back.
        out = TransformEngine().apply(frame, [rule])
        recombined = out["ss_half"] + out["ss_rest"]
        np.testing.assert_allclose(
            recombined.to_numpy(), frame["social_security"].to_numpy()
        )

    def test_results_recorded(self) -> None:
        engine = TransformEngine()
        engine.apply(_frame(), [_split_rule()])
        assert len(engine.results) == 1
        result = engine.results[0]
        assert result.kind == "split"
        assert result.source == "social_security"
        assert result.outputs == ["ss_retirement", "ss_disability", "ss_survivors"]

    def test_missing_source_raises(self) -> None:
        frame = _frame().drop(columns=["social_security"])
        with pytest.raises(ValueError, match="source column 'social_security' not"):
            TransformEngine().apply(frame, [_split_rule()])

    def test_output_collision_raises(self) -> None:
        frame = _frame()
        frame["ss_retirement"] = 1.0  # collides with a split output
        with pytest.raises(ValueError, match="already exist"):
            TransformEngine().apply(frame, [_split_rule()])


class TestDerive:
    def test_derive_sum_expression(self) -> None:
        frame = _frame()
        rule = TransformSpec(
            derive=DeriveTransform(
                target="total_income",
                expr="employment_income + interest_income",
            )
        )
        out = TransformEngine().apply(frame, [rule])
        np.testing.assert_allclose(
            out["total_income"].to_numpy(),
            (frame["employment_income"] + frame["interest_income"]).to_numpy(),
        )

    def test_derive_can_overwrite(self) -> None:
        frame = _frame()
        rule = TransformSpec(
            derive=DeriveTransform(target="interest_income", expr="interest_income * 2")
        )
        out = TransformEngine().apply(frame, [rule])
        np.testing.assert_allclose(
            out["interest_income"].to_numpy(),
            frame["interest_income"].to_numpy() * 2,
        )

    def test_bad_expression_raises(self) -> None:
        frame = _frame()
        rule = TransformSpec(
            derive=DeriveTransform(target="x", expr="nonexistent_col + 1")
        )
        with pytest.raises(ValueError, match="failed to evaluate"):
            TransformEngine().apply(frame, [rule])


class TestSequential:
    def test_derive_sees_split_output(self) -> None:
        frame = _frame()
        transforms = [
            _split_rule(),
            TransformSpec(
                derive=DeriveTransform(
                    target="ss_non_retirement",
                    expr="ss_disability + ss_survivors",
                )
            ),
        ]
        out = TransformEngine().apply(frame, transforms)
        expected = frame["social_security"] * 0.3
        np.testing.assert_allclose(
            out["ss_non_retirement"].to_numpy(), expected.to_numpy(), atol=1e-9
        )

    def test_empty_transforms_is_identity(self) -> None:
        frame = _frame()
        out = TransformEngine().apply(frame, [])
        pd.testing.assert_frame_equal(out, frame)

    def test_apply_does_not_mutate_input(self) -> None:
        frame = _frame()
        before = frame.copy()
        TransformEngine().apply(frame, [_split_rule()])
        pd.testing.assert_frame_equal(frame, before)
