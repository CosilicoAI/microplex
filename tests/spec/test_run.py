"""End-to-end smoke test for run_spec (microplex.run) on tiny synthetic sources.

Runs the full wired sequence (sources -> base imputation -> spine ->
half imputation -> transforms) on small in-memory frames and asserts the output
frame has the expected columns and structure, and that the not-yet-wired stages
are reported as pending.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microplex.core import EntityType
from microplex.core.sources import (
    EntityObservation,
    ObservationFrame,
    Shareability,
    SourceDescriptor,
    StaticSourceProvider,
    TimeStructure,
)
from microplex.data_sources.source_impute import SourceImputeBlock, SourceImputeManifest
from microplex.run import (
    PENDING_STAGES,
    SpecCalibrationResult,
    resolve_sources,
    run_source_impute_stage,
    run_spec,
)
from microplex.source_registry import SourceRegistry
from microplex.spec import load_spec_dict
from microplex.targets import (
    EntityTableBinding,
    EntityTableBundle,
    MaterializedSimulationTargetCompiler,
    TargetAggregation,
    TargetProvider,
    TargetQuery,
    TargetSet,
    TargetSimulationModifier,
    TargetSpec,
    apply_target_query,
)

DEMOGRAPHIC_COLS = ["age", "is_male", "tax_unit_is_joint"]
US_SPINE_KEYWORDS = (
    "employment_income",
    "taxable_interest_income",
)
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _discard_fake_microcalibrate_adapter():
    yield
    sys.modules.pop("microplex.calibration.microcalibrate_adapter", None)


def _install_fake_microcalibrate(monkeypatch, captured: dict) -> None:
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


def _with_block_geoids(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["block_geoid"] = [f"06001020100{i % 10}{i % 10:03d}" for i in range(len(out))]
    return out


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
                "manifest": "manifests/arch_targets.json",
                "target_profile": "pe_native_broad",
                "calibration_target_profile": "pe_native_broad_source_backed",
            }
        },
        "calibrate": {"loss": "pe_native_bucketed_huber_v1", "method": "apg"},
    }


def _sources() -> dict[str, pd.DataFrame]:
    return {"cps": _cps(), "puf": _puf(), "scf": _scf()}


def _sources_with_block_geography() -> dict[str, pd.DataFrame]:
    sources = _sources()
    sources["cps"] = _with_block_geoids(sources["cps"])
    sources["scf"] = sources["scf"].assign(
        source_impute_asset=lambda frame: frame["net_worth"] * 0.5
    )
    return sources


def _source_impute_spec_dict() -> dict:
    data = _spec_dict()
    data["spine"]["halves"][1]["strip_to"] = [
        "demographics",
        "tax_unit_id",
        "net_worth",
        "block_geoid",
    ]
    data.setdefault("variables", {})["source_impute_asset"] = {
        "mp_spec": {
            "method": "impute source asset from SCF after geography",
            "operation": {
                "kind": "impute",
                "source": "scf",
                "imputation_step": "scf_source_impute",
            },
        }
    }
    return data


def _source_impute_manifest(
    *,
    target_variables: tuple[str, ...] = ("source_impute_asset",),
) -> SourceImputeManifest:
    return SourceImputeManifest(
        blocks={
            "scf": SourceImputeBlock(
                name="scf",
                survey_name="scf",
                default_year=2022,
                dataset_id=None,
                archetype="wealth",
                dataset_loader=None,
                raw_loader=None,
                required_monthcode=None,
                annualized_variables=(),
                household_count_variables=(),
                household_variables=(),
                person_variables=("age", "source_impute_asset"),
                target_variables=target_variables,
                predictors=("age",),
            )
        }
    )


def _sources_with_sipp_block_geography() -> dict[str, pd.DataFrame]:
    sources = _sources_with_block_geography()
    rng = np.random.default_rng(12)
    sources["sipp"] = pd.DataFrame(
        {
            "age": rng.integers(18, 80, 220).astype(float),
            "tip_income": rng.uniform(0, 2500, 220),
            "bank_account_assets": rng.uniform(0, 50_000, 220),
        }
    )
    return sources


def _sipp_source_impute_spec_dict() -> dict:
    data = _source_impute_spec_dict()
    data["sources"]["sipp"] = {"dataset": "sipp_2023", "role": "donor"}
    data["variables"]["tip_income"] = {
        "mp_spec": {
            "method": "impute tips from SIPP after geography",
            "operation": {
                "kind": "impute",
                "source": "sipp",
                "imputation_step": "sipp_source_impute",
            },
        }
    }
    data["variables"]["bank_account_assets"] = {
        "mp_spec": {
            "method": "impute account assets from SIPP after geography",
            "operation": {
                "kind": "impute",
                "source": "sipp",
                "imputation_step": "sipp_source_impute",
            },
        }
    }
    return data


def _sipp_source_impute_manifest() -> SourceImputeManifest:
    return SourceImputeManifest(
        blocks={
            "sipp_tips": SourceImputeBlock(
                name="sipp_tips",
                survey_name="sipp",
                default_year=2023,
                dataset_id=None,
                archetype="income",
                dataset_loader=None,
                raw_loader=None,
                required_monthcode=None,
                annualized_variables=(),
                household_count_variables=(),
                household_variables=(),
                person_variables=("age", "tip_income"),
                target_variables=("tip_income",),
                predictors=("age",),
            ),
            "sipp_assets": SourceImputeBlock(
                name="sipp_assets",
                survey_name="sipp",
                default_year=2023,
                dataset_id=None,
                archetype="wealth",
                dataset_loader=None,
                raw_loader=None,
                required_monthcode=None,
                annualized_variables=(),
                household_count_variables=(),
                household_variables=(),
                person_variables=("age", "bank_account_assets"),
                target_variables=("bank_account_assets",),
                predictors=("age",),
            ),
        }
    )


class RecordingImputer:
    def __init__(self, calls: list[dict]) -> None:
        self.calls = calls
        self.fit_kwargs: dict | None = None
        self.regimes_: dict[str, str] = {}

    def fit(self, **kwargs):
        self.fit_kwargs = kwargs
        self.calls.append(kwargs)
        self.regimes_ = {variable: "TEST" for variable in kwargs["imputed_variables"]}
        return self

    def predict(self, target: pd.DataFrame) -> pd.DataFrame:
        assert self.fit_kwargs is not None
        return pd.DataFrame(
            {
                variable: np.arange(len(target), dtype=float) + 1000.0
                for variable in self.fit_kwargs["imputed_variables"]
            },
            index=target.index,
        )


def _install_recording_imputer(monkeypatch) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setattr(
        "microplex.imputation.ImputationRunner._make_imputer",
        lambda self: RecordingImputer(calls),
    )
    return calls


def _registry_provider(dataset: str, frame: pd.DataFrame) -> StaticSourceProvider:
    table = frame.copy()
    table.insert(0, "source_record_id", np.arange(len(table)))
    source = SourceDescriptor(
        name=dataset,
        shareability=Shareability.PUBLIC,
        time_structure=TimeStructure.CROSS_SECTION,
        observations=(
            EntityObservation(
                entity=EntityType.TAX_UNIT,
                key_column="source_record_id",
                variable_names=tuple(
                    column for column in table.columns if column != "source_record_id"
                ),
                weight_column="household_weight"
                if "household_weight" in table.columns
                else None,
            ),
        ),
    )
    return StaticSourceProvider(
        ObservationFrame(
            source=source,
            tables={EntityType.TAX_UNIT: table},
        )
    )


def _source_registry() -> SourceRegistry:
    return (
        SourceRegistry()
        .register(
            "cps_2024",
            _registry_provider("cps_2024", _cps()),
            default_entity=EntityType.TAX_UNIT,
        )
        .register(
            "puf_2024",
            _registry_provider("puf_2024", _puf()),
            default_entity=EntityType.TAX_UNIT,
        )
        .register(
            "scf_2022",
            _registry_provider("scf_2022", _scf()),
            default_entity=EntityType.TAX_UNIT,
        )
    )


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


def _target_set_with_missing_feature() -> TargetSet:
    target_set = _target_set()
    target_set.add(
        TargetSpec(
            name="missing_feature_total",
            entity=EntityType.TAX_UNIT,
            value=1.0,
            period=2024,
            measure="not_present",
            aggregation=TargetAggregation.SUM,
        )
    )
    return target_set


def _person_income_target_set() -> TargetSet:
    return TargetSet(
        [
            TargetSpec(
                name="person_employment_income_total",
                entity=EntityType.PERSON,
                value=20.0,
                period=2024,
                measure="employment_income",
                aggregation=TargetAggregation.SUM,
            )
        ]
    )


def _simulated_person_income_target_set() -> TargetSet:
    return TargetSet(
        [
            TargetSpec(
                name="simulated_person_income_total",
                entity=EntityType.PERSON,
                value=25.0,
                period=2024,
                measure="simulated_income",
                aggregation=TargetAggregation.SUM,
                sim_modifiers=(
                    TargetSimulationModifier(
                        name="materialize_policyengine",
                        parameters={"model": "policyengine-us"},
                    ),
                ),
            )
        ]
    )


def _multi_entity_bundle() -> EntityTableBundle:
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
    tax_units = pd.DataFrame(
        {
            "tax_unit_id": [100, 200],
            "household_id": [10, 20],
            "employment_income": [5.0, 4.0],
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
            EntityType.TAX_UNIT: EntityTableBinding(
                frame=tax_units,
                id_column="tax_unit_id",
                weight_link_column="household_id",
                synced_weight_column="tax_unit_weight",
            ),
        },
    )


def _arch_consumer_fact_row(
    concept: str,
    *,
    aggregation: str = "count",
    period: int = 2024,
    value: float = 1.0,
) -> dict:
    return {
        "aggregation": {"method": aggregation},
        "geography": {"id": "0100000US", "level": "country"},
        "observed_measure": {
            "source_concept": concept,
            "source_name": "irs_soi",
            "source_table": "fixture",
            "unit": "count" if aggregation == "count" else "usd",
        },
        "period": {"type": "tax_year", "value": period},
        "schema_version": "arch.consumer_fact.v1",
        "source": {"source_name": "irs_soi", "source_table": "fixture"},
        "universe_constraints": {"constraints": []},
        "value": value,
    }


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

    def test_end_to_end_runs_from_source_registry(self) -> None:
        spec = load_spec_dict(_spec_dict())

        result = _run_spec(
            spec,
            _source_registry(),
            demographic_columns=DEMOGRAPHIC_COLS,
        )

        assert isinstance(result.frame, pd.DataFrame)
        assert len(result.frame) == len(_cps())
        assert result.frame["employment_income"].notna().all()

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

    def test__given_arch_consumer_fact_paths__then_declared_target_surface_loads(
        self, tmp_path: Path
    ) -> None:
        facts = tmp_path / "consumer_facts.jsonl"
        facts.write_text(
            json.dumps(
                _arch_consumer_fact_row(
                    "irs_soi.individual_income_tax_returns",
                    value=123.0,
                ),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        spec = load_spec_dict(_spec_dict())

        result = _run_spec(
            spec,
            _sources(),
            demographic_columns=DEMOGRAPHIC_COLS,
            arch_consumer_fact_paths=[facts],
            arch_target_manifest_base=ROOT / "packs/us",
        )

        assert result.target_set is not None
        assert result.pending_stages == ("calibrate", "export")
        assert len(result.target_set.targets) == 1
        target = result.target_set.targets[0]
        assert target.entity is EntityType.TAX_UNIT
        assert target.measure == "tax_unit_count"
        assert target.value == 123.0

    def test__given_target_provider_and_arch_paths__then_run_spec_rejects_ambiguity(
        self, tmp_path: Path
    ) -> None:
        facts = tmp_path / "consumer_facts.jsonl"
        facts.write_text(
            json.dumps(_arch_consumer_fact_row("irs_soi.individual_income_tax_returns"))
            + "\n",
            encoding="utf-8",
        )
        spec = load_spec_dict(_spec_dict())

        with pytest.raises(
            ValueError,
            match="target_provider or arch_consumer_fact_paths",
        ):
            _run_spec(
                spec,
                _sources(),
                demographic_columns=DEMOGRAPHIC_COLS,
                target_provider=RecordingTargetProvider(_target_set()),
                arch_consumer_fact_paths=[facts],
                arch_target_manifest_base=ROOT / "packs/us",
            )

    def test__given_relative_arch_manifest_without_base__then_run_spec_fails_closed(
        self, tmp_path: Path
    ) -> None:
        facts = tmp_path / "consumer_facts.jsonl"
        facts.write_text(
            json.dumps(_arch_consumer_fact_row("irs_soi.individual_income_tax_returns"))
            + "\n",
            encoding="utf-8",
        )
        spec = load_spec_dict(_spec_dict())

        with pytest.raises(ValueError, match="arch_target_manifest_base"):
            _run_spec(
                spec,
                _sources(),
                demographic_columns=DEMOGRAPHIC_COLS,
                arch_consumer_fact_paths=[facts],
            )

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

    def test_generic_entity_table_calibration_runs_after_declared_targets_loaded(
        self,
        monkeypatch,
    ) -> None:
        captured: dict = {}
        _install_fake_microcalibrate(monkeypatch, captured)
        spec = load_spec_dict(_spec_dict())
        provider = RecordingTargetProvider(_target_set())

        result = _run_spec(
            spec,
            _sources(),
            demographic_columns=DEMOGRAPHIC_COLS,
            target_provider=provider,
            calibration_entity=EntityType.TAX_UNIT,
            calibration_id_column="tax_unit_id",
        )

        assert result.pending_stages == ("export",)
        assert result.calibration_result is not None
        assert (
            result.entity_table_bundle is result.calibration_result.entity_table_bundle
        )
        assert result.entity_table_bundle is not None
        assert result.calibration_result.diagnostics["weight_entity"] == "tax_unit"
        assert result.calibration_result.diagnostics["skipped_targets"] == []
        assert captured["kwargs"]["target_names"].tolist() == [
            "employment_income_total"
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

    def test_prepared_entity_table_bundle_calibration_defaults_to_weight_entity(
        self,
        monkeypatch,
    ) -> None:
        captured: dict = {}
        _install_fake_microcalibrate(monkeypatch, captured)
        spec = load_spec_dict(_spec_dict())
        provider = RecordingTargetProvider(_person_income_target_set())

        result = _run_spec(
            spec,
            _sources(),
            demographic_columns=DEMOGRAPHIC_COLS,
            target_provider=provider,
            calibration_entity_bundle=_multi_entity_bundle(),
        )

        assert result.pending_stages == ("export",)
        assert result.entity_table_bundle is not None
        assert result.calibration_result is not None
        assert (
            result.calibration_result.entity_table_bundle is result.entity_table_bundle
        )
        assert result.calibration_result.diagnostics["weight_entity"] == "household"
        assert result.frame["household_weight"].tolist() == [2.0, 3.0]
        assert captured["kwargs"]["target_names"].tolist() == [
            "person_employment_income_total"
        ]

    def test_prepared_entity_table_bundle_calibration_can_return_linked_entity(
        self,
        monkeypatch,
    ) -> None:
        captured: dict = {}
        _install_fake_microcalibrate(monkeypatch, captured)
        spec = load_spec_dict(_spec_dict())
        provider = RecordingTargetProvider(_person_income_target_set())

        result = _run_spec(
            spec,
            _sources(),
            demographic_columns=DEMOGRAPHIC_COLS,
            target_provider=provider,
            calibration_entity_bundle=_multi_entity_bundle(),
            calibration_entity=EntityType.PERSON,
        )

        assert result.frame["person_id"].tolist() == [1, 2, 3]
        assert result.frame["person_weight"].tolist() == [2.0, 2.0, 3.0]
        assert result.entity_table_bundle is not None
        household_weights = result.entity_table_bundle.table_for(EntityType.HOUSEHOLD)[
            "household_weight"
        ]
        assert household_weights.tolist() == [2.0, 3.0]
        np.testing.assert_array_equal(
            captured["kwargs"]["estimate_matrix"].to_numpy(),
            np.array([[5.0], [4.0]]),
        )

    def test_prepared_entity_table_bundle_routes_simulation_compiler(
        self,
        monkeypatch,
    ) -> None:
        captured: dict = {}
        _install_fake_microcalibrate(monkeypatch, captured)
        spec = load_spec_dict(_spec_dict())
        provider = RecordingTargetProvider(_simulated_person_income_target_set())

        class RecordingSimulationMaterializer:
            def __init__(self) -> None:
                self.calls: list[tuple[str, tuple[EntityType, ...]]] = []

            def materialize_simulation_features(
                self,
                *,
                targets,
                entity_frames,
                modifiers,
            ):
                assert tuple(modifier.name for modifier in modifiers) == (
                    "materialize_policyengine",
                )
                self.calls.append(
                    (
                        targets[0].name,
                        tuple(sorted(entity_frames, key=lambda entity: entity.value)),
                    )
                )
                person_frame = entity_frames[EntityType.PERSON]
                return {
                    EntityType.PERSON: pd.DataFrame(
                        {
                            "simulated_income": np.arange(
                                len(person_frame),
                                dtype=float,
                            )
                            + 10.0
                        }
                    )
                }

        materializer = RecordingSimulationMaterializer()

        result = _run_spec(
            spec,
            _sources(),
            demographic_columns=DEMOGRAPHIC_COLS,
            target_provider=provider,
            calibration_entity_bundle=_multi_entity_bundle(),
            calibration_entity=EntityType.PERSON,
            simulation_compiler=MaterializedSimulationTargetCompiler(materializer),
        )

        assert result.pending_stages == ("export",)
        assert materializer.calls == [
            (
                "simulated_person_income_total",
                (EntityType.HOUSEHOLD, EntityType.PERSON, EntityType.TAX_UNIT),
            )
        ]
        assert captured["kwargs"]["target_names"].tolist() == [
            "simulated_person_income_total"
        ]

    def test_prepared_entity_table_bundle_rejects_unused_id_column(self) -> None:
        spec = load_spec_dict(_spec_dict())
        provider = RecordingTargetProvider(_person_income_target_set())

        with pytest.raises(ValueError, match="does not use calibration_id_column"):
            _run_spec(
                spec,
                _sources(),
                demographic_columns=DEMOGRAPHIC_COLS,
                target_provider=provider,
                calibration_entity_bundle=_multi_entity_bundle(),
                calibration_id_column="tax_unit_id",
            )

    def test_generic_entity_table_calibration_requires_stable_id_column(
        self,
    ) -> None:
        spec = load_spec_dict(_spec_dict())
        provider = RecordingTargetProvider(_target_set())

        with pytest.raises(ValueError, match="requires calibration_id_column"):
            _run_spec(
                spec,
                _sources(),
                demographic_columns=DEMOGRAPHIC_COLS,
                target_provider=provider,
                calibration_entity=EntityType.TAX_UNIT,
            )

    def test_generic_entity_table_calibration_requires_loaded_declared_targets(
        self,
    ) -> None:
        spec = load_spec_dict(_spec_dict())

        with pytest.raises(ValueError, match="requires a loaded target_set"):
            _run_spec(
                spec,
                _sources(),
                demographic_columns=DEMOGRAPHIC_COLS,
                calibration_entity=EntityType.TAX_UNIT,
                calibration_id_column="tax_unit_id",
            )

    def test_generic_entity_table_calibration_rejects_skipped_targets_before_fit(
        self,
        monkeypatch,
    ) -> None:
        captured: dict = {}
        _install_fake_microcalibrate(monkeypatch, captured)
        spec = load_spec_dict(_spec_dict())
        provider = RecordingTargetProvider(_target_set_with_missing_feature())

        with pytest.raises(ValueError) as excinfo:
            _run_spec(
                spec,
                _sources(),
                demographic_columns=DEMOGRAPHIC_COLS,
                target_provider=provider,
                calibration_entity=EntityType.TAX_UNIT,
                calibration_id_column="tax_unit_id",
            )

        message = str(excinfo.value)
        assert "Sparse target compilation skipped target(s)" in message
        assert "missing_feature_total" in message
        assert "missing_features:not_present" in message
        assert "kwargs" not in captured

    def test_generic_entity_table_calibration_rejects_duplicate_record_ids(
        self,
    ) -> None:
        spec = load_spec_dict(_spec_dict())
        provider = RecordingTargetProvider(_target_set())

        with pytest.raises(ValueError, match="contains duplicate ids"):
            _run_spec(
                spec,
                _sources(),
                demographic_columns=DEMOGRAPHIC_COLS,
                target_provider=provider,
                calibration_entity=EntityType.TAX_UNIT,
                calibration_id_column="is_male",
            )

    def test_generic_entity_table_calibration_rejects_legacy_calibrator_combo(
        self,
    ) -> None:
        spec = load_spec_dict(_spec_dict())

        with pytest.raises(ValueError, match="either a legacy calibrator"):
            _run_spec(
                spec,
                _sources(),
                demographic_columns=DEMOGRAPHIC_COLS,
                calibrator=RecordingCalibrator(),
                calibration_entity=EntityType.TAX_UNIT,
                calibration_id_column="tax_unit_id",
            )

    def test_imputation_results_recorded(self) -> None:
        spec = load_spec_dict(_spec_dict())
        result = _run_spec(spec, _sources(), demographic_columns=DEMOGRAPHIC_COLS)
        # 1 (base) + 1 (synthetic) + 1 (cps_keep) = 3 step-results.
        assert len(result.imputation_results) == 3
        ontos = [r.onto for r in result.imputation_results]
        assert ontos == ["cps", "synthetic_puf", "cps_keep"]

    def test_source_impute_manifest_requires_post_geography_halves(
        self,
        monkeypatch,
    ) -> None:
        _install_recording_imputer(monkeypatch)
        data = _source_impute_spec_dict()
        data["spine"]["halves"][1]["strip_to"].remove("block_geoid")
        spec = load_spec_dict(data)

        with pytest.raises(ValueError, match="post-geography halves"):
            _run_spec(
                spec,
                _sources_with_block_geography(),
                demographic_columns=DEMOGRAPHIC_COLS,
                source_impute_manifest=_source_impute_manifest(),
            )

    def test_source_impute_manifest_rejects_null_block_geoids(
        self,
        monkeypatch,
    ) -> None:
        _install_recording_imputer(monkeypatch)
        spec = load_spec_dict(_source_impute_spec_dict())
        sources = _sources_with_block_geography()
        sources["cps"].loc[0, "block_geoid"] = None

        with pytest.raises(ValueError, match="non-null block_geoid"):
            _run_spec(
                spec,
                sources,
                demographic_columns=DEMOGRAPHIC_COLS,
                source_impute_manifest=_source_impute_manifest(),
            )

    def test_source_impute_manifest_rejects_malformed_block_geoids(
        self,
        monkeypatch,
    ) -> None:
        _install_recording_imputer(monkeypatch)
        spec = load_spec_dict(_source_impute_spec_dict())
        sources = _sources_with_block_geography()
        sources["cps"].loc[0, "block_geoid"] = "12345"

        with pytest.raises(ValueError, match="15-character block_geoid"):
            _run_spec(
                spec,
                sources,
                demographic_columns=DEMOGRAPHIC_COLS,
                source_impute_manifest=_source_impute_manifest(),
            )

    def test_source_impute_manifest_runs_after_spine_geography(
        self,
        monkeypatch,
    ) -> None:
        calls = _install_recording_imputer(monkeypatch)
        spec = load_spec_dict(_source_impute_spec_dict())

        result = _run_spec(
            spec,
            _sources_with_block_geography(),
            demographic_columns=DEMOGRAPHIC_COLS,
            source_impute_manifest=_source_impute_manifest(),
        )

        frame = result.frame
        assert frame["block_geoid"].notna().all()
        assert frame["source_impute_asset"].notna().all()
        source_impute_results = [
            result
            for result in result.imputation_results
            if result.imputed == ["source_impute_asset"]
        ]
        assert [result.onto for result in source_impute_results] == [
            "cps_keep",
            "synthetic_puf",
        ]
        source_impute_calls = [
            call
            for call in calls
            if call["imputed_variables"] == ["source_impute_asset"]
        ]
        assert len(source_impute_calls) == 2
        assert all(call["predictors"] == ["age"] for call in source_impute_calls)

    def test_run_source_impute_stage_resolves_donors_from_registry(
        self,
        monkeypatch,
    ) -> None:
        calls = _install_recording_imputer(monkeypatch)
        data = _source_impute_spec_dict()
        data["transforms"] = []
        spec = load_spec_dict(data)
        sources = _sources_with_block_geography()
        base_result = _run_spec(
            spec,
            sources,
            demographic_columns=DEMOGRAPHIC_COLS,
        )
        registry = SourceRegistry().register(
            "scf_2022",
            _registry_provider("scf_2022", sources["scf"]),
            default_entity=EntityType.TAX_UNIT,
        )

        stage = run_source_impute_stage(
            base_result,
            spec,
            registry,
            source_impute_manifest=_source_impute_manifest(),
            demographic_columns=DEMOGRAPHIC_COLS,
        )

        assert set(stage.sources) == {"scf"}
        assert stage.run_result.frame["source_impute_asset"].notna().all()
        assert [result.onto for result in stage.imputation_results] == [
            "cps_keep",
            "synthetic_puf",
        ]
        source_impute_calls = [
            call
            for call in calls
            if call["imputed_variables"] == ["source_impute_asset"]
        ]
        assert len(source_impute_calls) == 2
        assert all(call["predictors"] == ["age"] for call in source_impute_calls)

    def test_run_source_impute_stage_rejects_post_transform_result(
        self,
        monkeypatch,
    ) -> None:
        calls = _install_recording_imputer(monkeypatch)
        spec = load_spec_dict(_source_impute_spec_dict())
        transformed_result = _run_spec(
            spec,
            _sources_with_block_geography(),
            demographic_columns=DEMOGRAPHIC_COLS,
        )

        with pytest.raises(ValueError, match="pre-transform RunResult"):
            run_source_impute_stage(
                transformed_result,
                spec,
                _sources_with_block_geography(),
                source_impute_manifest=_source_impute_manifest(),
                demographic_columns=DEMOGRAPHIC_COLS,
            )

        assert not any(
            call["imputed_variables"] == ["source_impute_asset"] for call in calls
        )

    def test_run_source_impute_stage_rejects_post_transform_overwrite(
        self,
        monkeypatch,
    ) -> None:
        calls = _install_recording_imputer(monkeypatch)
        data = _source_impute_spec_dict()
        data["transforms"] = [
            {"derive": {"target": "employment_income", "expr": "employment_income + 1"}}
        ]
        spec = load_spec_dict(data)
        transformed_result = _run_spec(
            spec,
            _sources_with_block_geography(),
            demographic_columns=DEMOGRAPHIC_COLS,
        )

        with pytest.raises(ValueError, match="frame values match its halves"):
            run_source_impute_stage(
                transformed_result,
                spec,
                _sources_with_block_geography(),
                source_impute_manifest=_source_impute_manifest(),
                demographic_columns=DEMOGRAPHIC_COLS,
            )

        assert not any(
            call["imputed_variables"] == ["source_impute_asset"] for call in calls
        )

    def test_source_impute_step_filter_applies_after_block_filter(
        self,
        monkeypatch,
    ) -> None:
        calls = _install_recording_imputer(monkeypatch)
        spec = load_spec_dict(_source_impute_spec_dict())

        result = _run_spec(
            spec,
            _sources_with_block_geography(),
            demographic_columns=DEMOGRAPHIC_COLS,
            source_impute_manifest=_source_impute_manifest(),
            source_impute_blocks=("scf",),
            source_impute_imputation_steps=(),
        )

        assert "source_impute_asset" not in result.frame.columns
        assert not any(
            call["imputed_variables"] == ["source_impute_asset"] for call in calls
        )

    def test_source_impute_block_filter_keeps_shared_survey_blocks_separate(
        self,
        monkeypatch,
    ) -> None:
        calls = _install_recording_imputer(monkeypatch)
        spec = load_spec_dict(_sipp_source_impute_spec_dict())

        result = _run_spec(
            spec,
            _sources_with_sipp_block_geography(),
            demographic_columns=DEMOGRAPHIC_COLS,
            source_impute_manifest=_sipp_source_impute_manifest(),
            source_impute_blocks=("sipp_tips",),
            source_impute_imputation_steps=("sipp_source_impute",),
        )

        assert result.frame["tip_income"].notna().all()
        assert "bank_account_assets" not in result.frame.columns
        sipp_tip_calls = [
            call for call in calls if call["imputed_variables"] == ["tip_income"]
        ]
        assert len(sipp_tip_calls) == 2
        assert all(call["predictors"] == ["age"] for call in sipp_tip_calls)
        assert not any(
            call["imputed_variables"] == ["bank_account_assets"] for call in calls
        )

    def test_source_impute_step_filter_keeps_strict_manifest_target_validation(
        self,
        monkeypatch,
    ) -> None:
        calls = _install_recording_imputer(monkeypatch)
        spec = load_spec_dict(_source_impute_spec_dict())

        with pytest.raises(ValueError, match="not a manifest target"):
            _run_spec(
                spec,
                _sources_with_block_geography(),
                demographic_columns=DEMOGRAPHIC_COLS,
                source_impute_manifest=_source_impute_manifest(
                    target_variables=("different_asset",)
                ),
                source_impute_blocks=("scf",),
                source_impute_imputation_steps=("scf_source_impute",),
            )

        assert not any(
            call["imputed_variables"] == ["source_impute_asset"] for call in calls
        )

    def test_runtime_variable_operation_handlers_run_after_transforms(self) -> None:
        class TakeupHandler:
            def apply_variable_operations(
                self,
                frame: pd.DataFrame,
                *,
                variables,
                spec,
                operation_kind,
            ) -> pd.DataFrame:
                assert operation_kind.value == "rerandomize_takeup"
                assert spec.meta.country == "us"
                out = frame.copy()
                for variable_name in variables:
                    out[variable_name] = out["total_market_income"].gt(0)
                return out

        data = _spec_dict()
        data["variables"] = {
            "takes_up_snap_if_eligible": {
                "entity": "person",
                "mp_spec": {
                    "method": "fixture takeup",
                    "operation": {
                        "kind": "rerandomize_takeup",
                        "handler": "policyengine_us_takeup",
                    },
                },
            }
        }
        spec = load_spec_dict(data)

        result = _run_spec(
            spec,
            _sources(),
            demographic_columns=DEMOGRAPHIC_COLS,
            variable_operation_handlers={
                "policyengine_us_takeup": TakeupHandler(),
            },
        )

        assert "takes_up_snap_if_eligible" in result.frame.columns
        assert result.frame["takes_up_snap_if_eligible"].dtype == bool
        assert [batch.variables for batch in result.variable_operation_results] == [
            ("takes_up_snap_if_eligible",)
        ]

    def test_runtime_variable_operation_handlers_fail_closed_when_missing(self) -> None:
        data = _spec_dict()
        data["variables"] = {
            "income_tax": {
                "entity": "tax_unit",
                "mp_spec": {
                    "method": "PolicyEngine formula",
                    "operation": {
                        "kind": "materialize_policyengine",
                        "handler": "policyengine_us",
                    },
                },
            }
        }
        spec = load_spec_dict(data)

        with pytest.raises(
            ValueError,
            match="No runtime variable operation handler registered",
        ):
            _run_spec(
                spec,
                _sources(),
                demographic_columns=DEMOGRAPHIC_COLS,
                variable_operation_handlers={},
            )

    def test_runtime_variable_operation_handlers_must_materialize_columns(self) -> None:
        class EmptyHandler:
            def apply_variable_operations(
                self,
                frame: pd.DataFrame,
                *,
                variables,
                spec,
                operation_kind,
            ) -> pd.DataFrame:
                _ = (variables, spec, operation_kind)
                return frame.copy()

        data = _spec_dict()
        data["variables"] = {
            "income_tax": {
                "entity": "tax_unit",
                "mp_spec": {
                    "method": "PolicyEngine formula",
                    "operation": {
                        "kind": "materialize_policyengine",
                        "handler": "policyengine_us",
                    },
                },
            }
        }
        spec = load_spec_dict(data)

        with pytest.raises(ValueError, match="did not materialize"):
            _run_spec(
                spec,
                _sources(),
                demographic_columns=DEMOGRAPHIC_COLS,
                variable_operation_handlers={"policyengine_us": EmptyHandler()},
            )


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
