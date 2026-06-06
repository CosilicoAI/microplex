"""Tests for ImputationRunner (microplex.imputation) on synthetic frames.

Covers: spine_first ordering, condition_on resolution, imputing requested
vars, passthrough preservation vs. synthesize-overwrite, chaining wired
(assert via the fitted Imputer's predictors_), regime-gated signed support,
donor-missing handling, and the full-graph run() including 'both'.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd
import pytest

try:
    version("microimpute")
except PackageNotFoundError:
    pytest.skip(
        "microimpute is required for spec-driven imputation tests",
        allow_module_level=True,
    )

from microplex.imputation import (
    SPINE_FIRST_KEYWORDS,
    ImputationRunner,
    order_variables,
    spine_first_order,
)
from microplex.spec import ImputationOrder, ImputationStep

DEMOGRAPHIC_COLS = ["age", "is_male"]


def _donor(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A donor frame with demographics, income vars, and a weight column."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n).astype(float)
    return pd.DataFrame(
        {
            "age": age,
            "is_male": rng.integers(0, 2, n).astype(float),
            # Income correlates with age so chaining is meaningful.
            "employment_income": (age * 1000 + rng.normal(0, 5000, n)).clip(min=0),
            "capital_gains": rng.normal(2000, 5000, n).clip(min=0),
            "interest_income": rng.normal(1000, 2000, n).clip(min=0),
            "household_weight": rng.uniform(1000, 5000, n),
        }
    )


def _recipient(n: int = 120, seed: int = 1) -> pd.DataFrame:
    """A stripped recipient half: demographics + id only."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "tax_unit_id": np.arange(n),
            "age": rng.integers(18, 80, n).astype(float),
            "is_male": rng.integers(0, 2, n).astype(float),
        }
    )


def _runner(**kwargs) -> ImputationRunner:
    defaults = dict(
        column_groups={"demographics": DEMOGRAPHIC_COLS},
        weight_column="household_weight",
        seed=0,
    )
    defaults.update(kwargs)
    return ImputationRunner(**defaults)


# ---------------------------------------------------------------------------
# Ordering heuristic
# ---------------------------------------------------------------------------


class TestSpineFirstOrder:
    def test_income_vars_come_first(self) -> None:
        vars_in = ["cdcc_relevant_expenses", "employment_income", "ssi_flag"]
        ordered = spine_first_order(vars_in)
        # employment_income (spine) precedes the non-income vars.
        assert ordered[0] == "employment_income"
        assert set(ordered) == set(vars_in)

    def test_stable_within_tiers(self) -> None:
        vars_in = [
            "wage_income",
            "deduction_a",
            "self_employment_income",
            "deduction_b",
        ]
        ordered = spine_first_order(vars_in)
        assert ordered == [
            "wage_income",
            "self_employment_income",
            "deduction_a",
            "deduction_b",
        ]

    def test_custom_keywords(self) -> None:
        vars_in = ["foo_widget", "bar_baz"]
        ordered = spine_first_order(vars_in, keywords=["widget"])
        assert ordered == ["foo_widget", "bar_baz"]

    def test_no_spine_vars_preserves_order(self) -> None:
        vars_in = ["alpha", "beta", "gamma"]
        assert spine_first_order(vars_in) == vars_in

    def test_order_variables_as_declared(self) -> None:
        vars_in = ["deduction", "employment_income"]
        assert order_variables(vars_in, ImputationOrder.AS_DECLARED) == vars_in
        # spine_first would move income first:
        assert order_variables(vars_in, ImputationOrder.SPINE_FIRST) == [
            "employment_income",
            "deduction",
        ]

    def test_default_keywords_nonempty(self) -> None:
        assert "income" in SPINE_FIRST_KEYWORDS


# ---------------------------------------------------------------------------
# condition_on resolution
# ---------------------------------------------------------------------------


class TestConditionOn:
    def test_default_is_demographics(self) -> None:
        runner = _runner()
        step = ImputationStep(onto="synthetic", **{"from": "puf"}, vars=["x"])
        assert runner.resolve_condition_on(step) == DEMOGRAPHIC_COLS

    def test_explicit_group_plus_literal(self) -> None:
        runner = _runner()
        step = ImputationStep(
            onto="cps_keep",
            **{"from": "puf"},
            vars=["capital_gains"],
            condition_on=["demographics", "employment_income"],
        )
        assert runner.resolve_condition_on(step) == [
            *DEMOGRAPHIC_COLS,
            "employment_income",
        ]

    def test_literal_only(self) -> None:
        runner = _runner()
        step = ImputationStep(
            onto="x",
            **{"from": "puf"},
            vars=["y"],
            condition_on=["age", "employment_income"],
        )
        assert runner.resolve_condition_on(step) == ["age", "employment_income"]

    def test_dedup(self) -> None:
        runner = _runner()
        step = ImputationStep(
            onto="x",
            **{"from": "puf"},
            vars=["y"],
            condition_on=["demographics", "age"],
        )
        # 'age' is already in demographics -> not duplicated.
        assert runner.resolve_condition_on(step) == DEMOGRAPHIC_COLS

    def test_unmapped_demographics_raises(self) -> None:
        runner = ImputationRunner()  # no column_groups
        step = ImputationStep(onto="x", **{"from": "puf"}, vars=["y"])
        with pytest.raises(ValueError, match="no demographic columns"):
            runner.resolve_condition_on(step)

    def test_demographic_columns_arg_takes_precedence(self) -> None:
        runner = ImputationRunner(demographic_columns=["a", "b"])
        assert runner.demographic_columns == ["a", "b"]


# ---------------------------------------------------------------------------
# Running a single step
# ---------------------------------------------------------------------------


class TestRunStep:
    def test_default_factory_uses_canonical_regime_imputer(self) -> None:
        from microimpute import Imputer

        imputer = _runner()._make_imputer()
        assert isinstance(imputer, Imputer)
        assert imputer.signregime is True

    def test_default_imputer_preserves_signed_support_gap(self) -> None:
        """Loss-bearing targets must not leak into unsupported sign gaps.

        This guards the release default against regressing to bare QRF. The donor
        has a negative tail and a positive tail with no observations in
        (-20, 20); the canonical regime gate must expose the fitted sign regime
        and keep synthetic predictions out of that unsupported gap.
        """
        rng = np.random.default_rng(123)
        n = 800
        negative_x = rng.normal(-2, 0.2, n // 2)
        positive_x = rng.normal(2, 0.2, n // 2)
        donor = pd.DataFrame(
            {
                "x": np.concatenate([negative_x, positive_x]),
                "lossy_income": np.concatenate(
                    [
                        rng.uniform(-100, -20, n // 2),
                        rng.uniform(20, 100, n // 2),
                    ]
                ),
                "household_weight": np.ones(n),
            }
        )
        target = pd.DataFrame({"x": np.linspace(-2.5, 2.5, 400)})

        runner = ImputationRunner(
            column_groups={"demographics": ["x"]},
            weight_column="household_weight",
            seed=0,
        )
        step = ImputationStep(
            onto="synthetic",
            **{"from": "puf"},
            vars=["lossy_income"],
            synthesize=True,
        )
        new_target, result = runner.run_step(step, donor=donor, target=target)
        predictions = new_target["lossy_income"].to_numpy()

        assert result.regimes["lossy_income"] == "SIGN_ONLY"
        assert not ((predictions > -20) & (predictions < 20)).any()
        negative_share = (predictions < 0).mean()
        assert 0.35 < negative_share < 0.65

    def test_imputes_requested_vars(self) -> None:
        runner = _runner()
        step = ImputationStep(
            onto="synthetic",
            **{"from": "puf"},
            vars=["employment_income", "capital_gains"],
            synthesize=True,
        )
        new_target, result = runner.run_step(step, donor=_donor(), target=_recipient())
        assert "employment_income" in new_target.columns
        assert "capital_gains" in new_target.columns
        assert new_target["employment_income"].notna().all()
        assert set(result.imputed) == {"employment_income", "capital_gains"}
        assert set(result.regimes) == {"employment_income", "capital_gains"}
        # Demographics + id preserved.
        for col in ["tax_unit_id", "age", "is_male"]:
            assert col in new_target.columns

    def test_chaining_wired_via_predictors(self) -> None:
        """A later var's fitted predictors must include an earlier-imputed var."""
        runner = _runner()
        step = ImputationStep(
            onto="synthetic",
            **{"from": "puf"},
            # spine_first will order employment_income (income) before
            # capital_gains... but both are income keywords. Use explicit
            # as_declared order to make the chain deterministic for the assert.
            vars=["employment_income", "capital_gains"],
            order=ImputationOrder.AS_DECLARED,
            synthesize=True,
        )
        _, result = runner.run_step(step, donor=_donor(), target=_recipient())
        # employment_income is imputed first; capital_gains chains on it.
        assert "capital_gains" in result.predictors
        assert "employment_income" in result.predictors["capital_gains"], (
            f"chain not wired: capital_gains predictors = "
            f"{result.predictors['capital_gains']}"
        )
        # And the first var conditions only on the demographics (no income yet).
        assert result.predictors["employment_income"] == DEMOGRAPHIC_COLS

    def test_spine_first_orders_income_before_dependent(self) -> None:
        """spine_first chains a numeric non-income dependent var on income.

        microimpute only chains *numeric* targets (the regime-gated per-variable
        bundles surfaced by predictors_); boolean/categorical targets route
        through a non-chaining auxiliary path. So the dependent var here is a
        continuous amount with no income keyword in its name, which spine_first
        therefore orders *after* employment_income.
        """
        runner = _runner()
        step = ImputationStep(
            onto="synthetic",
            **{"from": "puf"},
            vars=["medical_expense", "employment_income"],
            order=ImputationOrder.SPINE_FIRST,
            synthesize=True,
        )
        donor = _donor()
        # Continuous amount correlated with income (so chaining is sensible).
        donor["medical_expense"] = (
            donor["employment_income"] * 0.05
            + np.random.default_rng(5).normal(0, 500, len(donor))
        ).clip(lower=0)
        _, result = runner.run_step(step, donor=donor, target=_recipient())
        # employment_income imputed first; medical_expense conditions on it.
        assert "medical_expense" in result.predictors
        assert "employment_income" in result.predictors["medical_expense"]

    def test_passthrough_preserves_existing_column(self) -> None:
        runner = _runner()
        target = _recipient()
        # Target already has employment_income (real values).
        sentinel = np.full(len(target), 12345.0)
        target["employment_income"] = sentinel
        step = ImputationStep(
            onto="cps_keep",
            **{"from": "puf"},
            vars=["employment_income", "capital_gains"],
            condition_on=["demographics", "employment_income"],
            synthesize=False,  # default
        )
        new_target, result = runner.run_step(step, donor=_donor(), target=target)
        # employment_income untouched (passthrough), capital_gains imputed.
        np.testing.assert_array_equal(
            new_target["employment_income"].to_numpy(), sentinel
        )
        assert "employment_income" in result.skipped_passthrough
        assert "capital_gains" in result.imputed
        # capital_gains conditioned on the real employment_income.
        assert "employment_income" in result.predictors["capital_gains"]

    def test_synthesize_overwrites_existing_column(self) -> None:
        runner = _runner()
        target = _recipient()
        target["capital_gains"] = np.full(len(target), 999.0)
        step = ImputationStep(
            onto="synthetic",
            **{"from": "puf"},
            vars=["capital_gains"],
            synthesize=True,
        )
        new_target, result = runner.run_step(step, donor=_donor(), target=target)
        assert "capital_gains" in result.imputed
        # Overwritten -> not all equal to the sentinel.
        assert not (new_target["capital_gains"].to_numpy() == 999.0).all()

    def test_donor_missing_var_skipped(self) -> None:
        runner = _runner()
        step = ImputationStep(
            onto="synthetic",
            **{"from": "puf"},
            vars=["employment_income", "not_in_donor"],
            synthesize=True,
        )
        new_target, result = runner.run_step(step, donor=_donor(), target=_recipient())
        assert "not_in_donor" in result.skipped_missing_in_donor
        assert "not_in_donor" not in new_target.columns
        assert "employment_income" in result.imputed

    def test_weighted_fit_when_weight_present(self) -> None:
        """Smoke: a weight column in the donor doesn't break the fit."""
        runner = _runner()
        step = ImputationStep(
            onto="synthetic",
            **{"from": "puf"},
            vars=["employment_income"],
            synthesize=True,
        )
        new_target, result = runner.run_step(step, donor=_donor(), target=_recipient())
        assert result.imputed == ["employment_income"]
        assert new_target["employment_income"].notna().all()

    def test_missing_predictor_in_donor_raises(self) -> None:
        runner = _runner()
        donor = _donor().drop(columns=["age"])
        step = ImputationStep(
            onto="synthetic", **{"from": "puf"}, vars=["capital_gains"]
        )
        with pytest.raises(ValueError, match="donor is missing predictor"):
            runner.run_step(step, donor=donor, target=_recipient())

    def test_missing_predictor_in_target_raises(self) -> None:
        runner = _runner()
        target = _recipient().drop(columns=["age"])
        step = ImputationStep(
            onto="synthetic", **{"from": "puf"}, vars=["capital_gains"]
        )
        with pytest.raises(ValueError, match="target half is missing predictor"):
            runner.run_step(step, donor=_donor(), target=target)


# ---------------------------------------------------------------------------
# Running the full graph
# ---------------------------------------------------------------------------


class TestRunGraph:
    def test_both_applies_to_all_halves(self) -> None:
        runner = _runner()
        halves = {
            "cps_keep": _recipient(80, seed=2),
            "synthetic": _recipient(80, seed=3),
        }
        step = ImputationStep(
            onto="both",
            **{"from": "scf"},
            vars=["employment_income"],
            synthesize=True,
        )
        new_halves, results = runner.run(
            [step], halves=halves, donors={"scf": _donor()}
        )
        # Both halves got the column.
        for name in ("cps_keep", "synthetic"):
            assert "employment_income" in new_halves[name].columns
            assert new_halves[name]["employment_income"].notna().all()
        # One result per (step, half).
        assert {r.onto for r in results} == {"cps_keep", "synthetic"}

    def test_sequential_steps_on_one_half(self) -> None:
        runner = _runner()
        halves = {"synthetic": _recipient(100)}
        steps = [
            ImputationStep(
                onto="synthetic",
                **{"from": "puf"},
                vars=["employment_income"],
                synthesize=True,
            ),
            # Second step conditions on the just-imputed employment_income.
            ImputationStep(
                onto="synthetic",
                **{"from": "puf"},
                vars=["capital_gains"],
                condition_on=["demographics", "employment_income"],
                synthesize=True,
            ),
        ]
        new_halves, results = runner.run(steps, halves=halves, donors={"puf": _donor()})
        synth = new_halves["synthetic"]
        assert "employment_income" in synth.columns
        assert "capital_gains" in synth.columns
        # The second step's capital_gains chained on employment_income.
        cg_result = next(r for r in results if "capital_gains" in r.imputed)
        assert "employment_income" in cg_result.predictors["capital_gains"]

    def test_unknown_donor_raises(self) -> None:
        runner = _runner()
        step = ImputationStep(onto="synthetic", **{"from": "ghost"}, vars=["x"])
        with pytest.raises(KeyError, match="unknown donor"):
            runner.run(
                [step],
                halves={"synthetic": _recipient()},
                donors={"puf": _donor()},
            )

    def test_unknown_half_raises(self) -> None:
        runner = _runner()
        step = ImputationStep(onto="ghost_half", **{"from": "puf"}, vars=["x"])
        with pytest.raises(KeyError, match="unknown half"):
            runner.run(
                [step],
                halves={"synthetic": _recipient()},
                donors={"puf": _donor()},
            )
