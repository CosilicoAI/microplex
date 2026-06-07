"""End-to-end smoke test for run_spec (microplex.run) on tiny synthetic sources.

Runs the full wired sequence (sources -> base imputation -> spine ->
half imputation -> transforms) on small in-memory frames and asserts the output
frame has the expected columns and structure, and that the not-yet-wired stages
are reported as pending.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microplex.core import EntityType
from microplex.run import (
    PENDING_STAGES,
    SpecCalibrationResult,
    resolve_sources,
    run_spec,
)
from microplex.spec import load_spec_dict
from microplex.targets import (
    TargetAggregation,
    TargetProvider,
    TargetQuery,
    TargetSet,
    TargetSpec,
    apply_target_query,
)

DEMOGRAPHIC_COLS = ["age", "is_male", "tax_unit_is_joint"]
US_SPINE_KEYWORDS = (
    "employment_income",
    "taxable_interest_income",
)


def _cps(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """A CPS-like spine source: demographics, id, income, social_security, weight."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n).astype(float)
    return pd.DataFrame(
        {
            "tax_unit_id": np.arange(n),
            "age": age,
            "is_male": rng.integers(0, 2, n).astype(float),
            "tax_unit_is_joint": rng.integers(0, 2, n).astype(float),
            "employment_income": (age * 800 + rng.normal(0, 4000, n)).clip(min=0),
            "social_security": np.where(age > 62, rng.uniform(8000, 25000, n), 0.0),
            "household_weight": rng.uniform(1000, 5000, n),
        }
    )


def _puf(n: int = 400, seed: int = 1) -> pd.DataFrame:
    """A PUF-like donor: demographics + tax vars + weight."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, n).astype(float)
    emp = (age * 1200 + rng.normal(0, 8000, n)).clip(min=0)
    net_worth = age * 6000 + emp * 1.5 + rng.normal(0, 60000, n)
    return pd.DataFrame(
        {
            "age": age,
            "is_male": rng.integers(0, 2, n).astype(float),
            "tax_unit_is_joint": rng.integers(0, 2, n).astype(float),
            "employment_income": emp,
            "long_term_capital_gains": (emp * 0.05 + rng.normal(0, 2000, n)).clip(
                min=0
            ),
            "taxable_interest_income": rng.uniform(0, 4000, n),
            "net_worth": net_worth,
            "household_weight": rng.uniform(500, 8000, n),
        }
    )


def _scf(n: int = 300, seed: int = 2) -> pd.DataFrame:
    """An SCF-like donor: demographics + net_worth + weight."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, n).astype(float)
    return pd.DataFrame(
        {
            "age": age,
            "is_male": rng.integers(0, 2, n).astype(float),
            "tax_unit_is_joint": rng.integers(0, 2, n).astype(float),
            "net_worth": (age * 5000 + rng.normal(0, 50000, n)),
            "household_weight": rng.uniform(800, 6000, n),
        }
    )


def _spec_dict() -> dict:
    return {
        "meta": {"country": "us", "model_year": 2024},
        "sources": {
            "cps": {"dataset": "cps_2024", "role": "spine"},
            "puf": {"dataset": "puf_2024", "role": "donor"},
            "scf": {"dataset": "scf_2022", "role": "donor"},
        },
        "spine": {
            "base": "cps",
            "method": "clone",
            "clone": {"seed": 0},
            "halves": [
                {"name": "cps_keep", "keep": "all"},
                {
                    "name": "synthetic_puf",
                    "strip_to": ["demographics", "tax_unit_id", "net_worth"],
                },
            ],
        },
        "imputation": [
            # Source-level imputation runs before the spine is split, so both
            # halves can retain net_worth as a post-split predictor.
            {
                "at": "base",
                "onto": "base",
                "from": "scf",
                "vars": ["net_worth"],
            },
            # Synthesize the full PUF tax block onto the stripped half.
            {
                "onto": "synthetic_puf",
                "from": "puf",
                "vars": [
                    "employment_income",
                    "long_term_capital_gains",
                    "taxable_interest_income",
                ],
                "condition_on": ["demographics", "net_worth"],
                "order": "spine_first",
                "synthesize": True,
            },
            # On the kept half, impute a PUF-only var conditioned on real income.
            {
                "onto": "cps_keep",
                "from": "puf",
                "vars": ["long_term_capital_gains"],
                "condition_on": ["demographics", "employment_income"],
            },
        ],
        "transforms": [
            {
                "split": {
                    "source": "social_security",
                    "into": {
                        "ss_retirement": 0.8,
                        "ss_survivors": 0.2,
                    },
                }
            },
            {
                "derive": {
                    "target": "total_market_income",
                    "expr": "employment_income + taxable_interest_income",
                }
            },
        ],
        "targets": {
            "arch": {
                "country": "us",
                "model_year": 2024,
                "target_profile": "pe_native_broad",
                "calibration_target_profile": "pe_native_broad_source_backed",
            }
        },
        "calibrate": {"loss": "pe_native_bucketed_huber_v1", "method": "apg"},
    }


def _sources() -> dict[str, pd.DataFrame]:
    return {"cps": _cps(), "puf": _puf(), "scf": _scf()}


def _run_spec(*args, **kwargs):
    kwargs.setdefault("spine_keywords", US_SPINE_KEYWORDS)
    return run_spec(*args, **kwargs)


class RecordingTargetProvider:
    def __init__(self, target_set: TargetSet) -> None:
        self.target_set = target_set
        self.queries: list[TargetQuery | None] = []

    def load_target_set(self, query: TargetQuery | None = None) -> TargetSet:
        self.queries.append(query)
        return apply_target_query(self.target_set, query)


class RecordingCalibrator:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def calibrate(
        self,
        frame: pd.DataFrame,
        *,
        target_set: TargetSet,
        calibrate,
        weight_column: str | None,
    ) -> SpecCalibrationResult:
        self.calls.append(
            {
                "target_names": tuple(target.name for target in target_set.targets),
                "loss": calibrate.loss,
                "method": calibrate.method.value,
                "weight_column": weight_column,
            }
        )
        calibrated = frame.copy()
        calibrated["household_weight"] = calibrated["household_weight"] + 1.0
        return SpecCalibrationResult(
            frame=calibrated,
            diagnostics={"targets": len(target_set.targets)},
        )


def _target_set() -> TargetSet:
    return TargetSet(
        [
            TargetSpec(
                name="employment_income_total",
                entity=EntityType.TAX_UNIT,
                value=1_000_000.0,
                period=2024,
                measure="employment_income",
                aggregation=TargetAggregation.SUM,
            )
        ]
    )


class TestRunSpec:
    def test_spine_first_requires_explicit_pack_keywords(self) -> None:
        spec = load_spec_dict(_spec_dict())
        with pytest.raises(ValueError, match="requires explicit spine_keywords"):
            run_spec(
                spec,
                _sources(),
                demographic_columns=DEMOGRAPHIC_COLS,
            )

    def test_spine_first_rejects_empty_pack_keywords(self) -> None:
        spec = load_spec_dict(_spec_dict())
        with pytest.raises(ValueError, match="empty spine_keywords"):
            run_spec(
                spec,
                _sources(),
                demographic_columns=DEMOGRAPHIC_COLS,
                spine_keywords=(),
            )

    def test_end_to_end_runs(self) -> None:
        spec = load_spec_dict(_spec_dict())
        result = _run_spec(
            spec,
            _sources(),
            demographic_columns=DEMOGRAPHIC_COLS,
        )
        assert isinstance(result.frame, pd.DataFrame)
        base = _cps()
        assert len(result.frame) == len(base)
        label = result.spine.half_label_column
        assert result.frame[label].value_counts().to_dict() == {
            "cps_keep": len(base) // 2,
            "synthetic_puf": len(base) // 2,
        }
        synthetic = result.frame[result.frame[label] == "synthetic_puf"]
        assert (synthetic["household_weight"] == 0).all()

    def test_output_has_expected_columns(self) -> None:
        spec = load_spec_dict(_spec_dict())
        result = _run_spec(spec, _sources(), demographic_columns=DEMOGRAPHIC_COLS)
        cols = set(result.frame.columns)
        expected = {
            # demographics + id (carried through)
            "tax_unit_id",
            "age",
            "is_male",
            "tax_unit_is_joint",
            # imputed
            "employment_income",
            "long_term_capital_gains",
            "taxable_interest_income",
            "net_worth",
            # transform outputs
            "ss_retirement",
            "ss_survivors",
            "total_market_income",
            # spine label
            result.spine.half_label_column,
        }
        missing = expected - cols
        assert not missing, f"output missing columns: {missing}"

    def test_both_halves_have_net_worth(self) -> None:
        spec = load_spec_dict(_spec_dict())
        result = _run_spec(spec, _sources(), demographic_columns=DEMOGRAPHIC_COLS)
        frame = result.frame
        label = result.spine.half_label_column
        for half in ("cps_keep", "synthetic_puf"):
            sub = frame[frame[label] == half]
            assert sub["net_worth"].notna().all(), f"{half} missing net_worth"

    def test_base_phase_imputation_runs_before_spine_clone(self) -> None:
        spec = load_spec_dict(_spec_dict())
        result = _run_spec(spec, _sources(), demographic_columns=DEMOGRAPHIC_COLS)
        assert result.base["net_worth"].notna().all()
        for half_name, half in result.spine.halves.items():
            assert half["net_worth"].notna().all(), f"{half_name} missing net_worth"

    def test_synthetic_half_income_synthesized(self) -> None:
        spec = load_spec_dict(_spec_dict())
        result = _run_spec(spec, _sources(), demographic_columns=DEMOGRAPHIC_COLS)
        frame = result.frame
        label = result.spine.half_label_column
        synth = frame[frame[label] == "synthetic_puf"]
        # The stripped half had no employment_income; it must now be filled.
        assert synth["employment_income"].notna().all()

    def test_kept_half_preserves_real_employment_income(self) -> None:
        """cps_keep step does NOT synthesize employment_income (passthrough)."""
        spec = load_spec_dict(_spec_dict())
        sources = _sources()
        result = _run_spec(spec, sources, demographic_columns=DEMOGRAPHIC_COLS)
        frame = result.frame
        label = result.spine.half_label_column
        kept = frame[frame[label] == "cps_keep"]
        # Real CPS employment_income values are non-null and finite.
        assert kept["employment_income"].notna().all()
        # The kept half got long_term_capital_gains imputed (it had none).
        assert kept["long_term_capital_gains"].notna().all()

    def test_split_sums_back_in_output(self) -> None:
        spec = load_spec_dict(_spec_dict())
        result = _run_spec(spec, _sources(), demographic_columns=DEMOGRAPHIC_COLS)
        frame = result.frame
        recombined = frame["ss_retirement"] + frame["ss_survivors"]
        np.testing.assert_allclose(
            recombined.to_numpy(), frame["social_security"].to_numpy(), atol=1e-6
        )

    def test_pending_stages_reported(self) -> None:
        spec = load_spec_dict(_spec_dict())
        result = _run_spec(spec, _sources(), demographic_columns=DEMOGRAPHIC_COLS)
        assert result.pending_stages == PENDING_STAGES
        assert result.target_set is None
        assert "calibrate" in result.pending_stages
        assert "export" in result.pending_stages

    def test_target_provider_loads_declared_target_surface(self) -> None:
        spec = load_spec_dict(_spec_dict())
        provider = RecordingTargetProvider(_target_set())

        assert isinstance(provider, TargetProvider)
        result = _run_spec(
            spec,
            _sources(),
            demographic_columns=DEMOGRAPHIC_COLS,
            target_provider=provider,
        )

        assert result.target_set is not None
        assert [target.name for target in result.target_set.targets] == [
            "employment_income_total"
        ]
        assert result.pending_stages == ("calibrate", "export")
        assert len(provider.queries) == 1
        query = provider.queries[0]
        assert query is not None
        assert query.period == 2024
        assert query.provider_filters == {
            "source": "arch",
            "country": "us",
            "model_year": 2024,
            "target_profile": "pe_native_broad",
            "calibration_target_profile": "pe_native_broad_source_backed",
        }

    def test_calibrator_runs_after_declared_targets_are_loaded(self) -> None:
        spec = load_spec_dict(_spec_dict())
        provider = RecordingTargetProvider(_target_set())
        calibrator = RecordingCalibrator()

        result = _run_spec(
            spec,
            _sources(),
            demographic_columns=DEMOGRAPHIC_COLS,
            target_provider=provider,
            calibrator=calibrator,
        )

        assert result.pending_stages == ("export",)
        assert result.calibration_result is not None
        assert result.calibration_result.diagnostics == {"targets": 1}
        assert calibrator.calls == [
            {
                "target_names": ("employment_income_total",),
                "loss": "pe_native_bucketed_huber_v1",
                "method": "apg",
                "weight_column": "household_weight",
            }
        ]
        uncalibrated = _run_spec(
            spec,
            _sources(),
            demographic_columns=DEMOGRAPHIC_COLS,
            target_provider=provider,
        )
        pd.testing.assert_series_equal(
            result.frame["household_weight"],
            uncalibrated.frame["household_weight"] + 1.0,
            check_names=False,
        )

    def test_calibrator_requires_loaded_declared_target_surface(self) -> None:
        spec = load_spec_dict(_spec_dict())
        calibrator = RecordingCalibrator()

        with pytest.raises(ValueError, match="requires a loaded target_set"):
            _run_spec(
                spec,
                _sources(),
                demographic_columns=DEMOGRAPHIC_COLS,
                calibrator=calibrator,
            )

    def test_imputation_results_recorded(self) -> None:
        spec = load_spec_dict(_spec_dict())
        result = _run_spec(spec, _sources(), demographic_columns=DEMOGRAPHIC_COLS)
        # 1 (base) + 1 (synthetic) + 1 (cps_keep) = 3 step-results.
        assert len(result.imputation_results) == 3
        ontos = [r.onto for r in result.imputation_results]
        assert ontos == ["cps", "synthetic_puf", "cps_keep"]


class TestResolveSources:
    def test_missing_source_raises(self) -> None:
        spec = load_spec_dict(_spec_dict())
        with pytest.raises(KeyError, match="missing frames"):
            resolve_sources(spec, {"cps": _cps(), "puf": _puf()})  # no scf

    def test_returns_declared_sources_only(self) -> None:
        spec = load_spec_dict(_spec_dict())
        extra = _sources()
        extra["unused"] = _scf()
        resolved = resolve_sources(spec, extra)
        assert set(resolved) == {"cps", "puf", "scf"}
