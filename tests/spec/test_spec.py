"""Tests for the Microplex spec DSL (microplex.spec).

Covers: loading a valid example spec from YAML, the parsed structure, and
rejection of a range of malformed specs with clear, field-pathed errors.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from microplex.spec import (
    BASE_TOKEN,
    BOTH_TOKEN,
    CalibrationMethod,
    ImputationOrder,
    ImputationPhase,
    MicroplexSpec,
    SourceRole,
    SpecError,
    SpineMethod,
    TransformKind,
    load_spec,
    load_spec_dict,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _valid_spec_dict() -> dict:
    """A minimal but complete valid spec as a plain dict."""
    return {
        "meta": {"country": "us", "model_year": 2024},
        "sources": {
            "cps": {"dataset": "cps_2024", "role": "spine"},
            "puf": {"dataset": "puf_2024", "role": "donor"},
        },
        "spine": {
            "base": "cps",
            "method": "clone",
            "clone": {"seed": 0},
            "halves": [
                {"name": "cps_keep", "keep": "all"},
                {"name": "synthetic_puf", "strip_to": ["demographics", "tax_unit_id"]},
            ],
        },
        "imputation": [
            {
                "onto": "synthetic_puf",
                "from": "puf",
                "vars": ["employment_income", "capital_gains"],
                "synthesize": True,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Loading a valid spec
# ---------------------------------------------------------------------------


class TestLoadValid:
    def test_loads_yaml_fixture(self) -> None:
        spec = load_spec(FIXTURES / "us_2024.yaml")
        assert isinstance(spec, MicroplexSpec)
        assert spec.meta.country == "us"
        assert spec.meta.model_year == 2024
        assert spec.meta.policyengine_model == "policyengine-us"

    def test_fixture_sources_and_roles(self) -> None:
        spec = load_spec(FIXTURES / "us_2024.yaml")
        assert set(spec.sources) == {"cps_asec", "puf", "scf"}
        assert spec.sources["cps_asec"].role is SourceRole.SPINE
        assert spec.sources["puf"].role is SourceRole.DONOR
        assert spec.spine_source == "cps_asec"
        assert spec.donor_sources == ("puf", "scf")

    def test_fixture_spine_halves(self) -> None:
        spec = load_spec(FIXTURES / "us_2024.yaml")
        assert spec.spine.half_names == ("cps_keep", "synthetic_puf")
        assert spec.spine.method is SpineMethod.CLONE
        assert spec.spine.passthrough_half.name == "cps_keep"
        assert spec.spine.synthetic_half.name == "synthetic_puf"
        assert spec.spine.synthetic_half.strip_to == ["demographics", "tax_unit_id"]
        assert spec.spine.clone.seed == 0

    def test_fixture_imputation_steps(self) -> None:
        spec = load_spec(FIXTURES / "us_2024.yaml")
        assert len(spec.imputation) == 3
        first = spec.imputation[0]
        assert first.onto == "synthetic_puf"
        # `from` is reserved; exposed as `from_`.
        assert first.from_ == "puf"
        assert first.vars == [
            "employment_income",
            "long_term_capital_gains",
            "taxable_interest_income",
        ]
        assert first.order is ImputationOrder.SPINE_FIRST
        assert first.at is ImputationPhase.HALVES
        assert first.synthesize is True

        second = spec.imputation[1]
        assert second.condition_on == ["demographics", "employment_income"]
        assert second.synthesize is False  # default passthrough

        third = spec.imputation[2]
        assert third.onto == BOTH_TOKEN
        assert third.at is ImputationPhase.HALVES
        assert third.targets_both is True

    def test_fixture_transforms(self) -> None:
        spec = load_spec(FIXTURES / "us_2024.yaml")
        assert len(spec.transforms) == 2
        split_rule = spec.transforms[0]
        assert split_rule.kind is TransformKind.SPLIT
        assert split_rule.split is not None
        assert split_rule.split.source == "social_security"
        assert split_rule.split.is_fractional is True
        derive_rule = spec.transforms[1]
        assert derive_rule.kind is TransformKind.DERIVE
        assert derive_rule.derive is not None
        assert derive_rule.derive.target == "total_income"

    def test_fixture_targets_and_calibrate(self) -> None:
        spec = load_spec(FIXTURES / "us_2024.yaml")
        assert spec.targets is not None
        assert spec.targets.arch.country == "us"
        assert spec.targets.arch.model_year == 2024
        assert spec.targets.arch.target_profile == "pe_native_broad"
        assert (
            spec.targets.arch.resolved_calibration_target_profile
            == "pe_native_broad_source_backed"
        )
        assert spec.calibrate is not None
        assert spec.calibrate.loss == "pe_native_bucketed_huber_v1"
        assert spec.calibrate.method is CalibrationMethod.APG
        assert spec.calibrate.target_records == 100000

    def test_load_spec_dict_roundtrip(self) -> None:
        spec = load_spec_dict(_valid_spec_dict())
        assert spec.meta.country == "us"
        assert len(spec.imputation) == 1
        assert spec.imputation[0].at is ImputationPhase.HALVES

    def test_base_phase_imputation_can_target_base_alias(self) -> None:
        data = _valid_spec_dict()
        data["imputation"].insert(
            0,
            {
                "at": "base",
                "onto": BASE_TOKEN,
                "from": "puf",
                "vars": ["net_worth"],
            },
        )
        spec = load_spec_dict(data)
        first = spec.imputation[0]
        assert first.at is ImputationPhase.BASE
        assert first.onto == BASE_TOKEN

    def test_base_phase_imputation_can_target_spine_source_name(self) -> None:
        data = _valid_spec_dict()
        data["imputation"].insert(
            0,
            {
                "at": "base",
                "onto": "cps",
                "from": "puf",
                "vars": ["net_worth"],
            },
        )
        spec = load_spec_dict(data)
        first = spec.imputation[0]
        assert first.at is ImputationPhase.BASE
        assert first.onto == "cps"

    def test_minimal_spec_without_optional_sections(self) -> None:
        data = _valid_spec_dict()
        # imputation/transforms/targets/calibrate are all optional.
        del data["imputation"]
        spec = load_spec_dict(data)
        assert spec.imputation == []
        assert spec.transforms == []
        assert spec.targets is None
        assert spec.calibrate is None

    def test_country_is_lowercased(self) -> None:
        data = _valid_spec_dict()
        data["meta"]["country"] = "US"
        spec = load_spec_dict(data)
        assert spec.meta.country == "us"


# ---------------------------------------------------------------------------
# Rejecting malformed specs
# ---------------------------------------------------------------------------


class TestRejectMalformed:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SpecError, match="not found"):
            load_spec(tmp_path / "nope.yaml")

    def test_not_yaml_mapping(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("- just\n- a\n- list\n")
        with pytest.raises(SpecError, match="mapping at the top level"):
            load_spec(bad)

    def test_invalid_yaml_syntax(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("meta: {country: us\n  model_year]: oops\n")
        with pytest.raises(SpecError, match="not valid YAML"):
            load_spec(bad)

    def test_unknown_top_level_field_rejected(self) -> None:
        data = _valid_spec_dict()
        data["bogus_section"] = {}
        with pytest.raises(SpecError, match="bogus_section"):
            load_spec_dict(data)

    def test_unknown_source_field_rejected(self) -> None:
        data = _valid_spec_dict()
        data["sources"]["cps"]["typo"] = 1
        with pytest.raises(SpecError, match="typo"):
            load_spec_dict(data)

    def test_two_spine_sources_rejected(self) -> None:
        data = _valid_spec_dict()
        data["sources"]["puf"]["role"] = "spine"
        with pytest.raises(
            SpecError, match="exactly one source must have role 'spine'"
        ):
            load_spec_dict(data)

    def test_no_spine_source_rejected(self) -> None:
        data = _valid_spec_dict()
        data["sources"]["cps"]["role"] = "donor"
        with pytest.raises(SpecError, match="role 'spine'"):
            load_spec_dict(data)

    def test_spine_base_not_a_source(self) -> None:
        data = _valid_spec_dict()
        data["spine"]["base"] = "does_not_exist"
        with pytest.raises(SpecError, match="is not a declared source"):
            load_spec_dict(data)

    def test_spine_base_wrong_role(self) -> None:
        # Make cps a donor and puf the spine source, but keep base=cps.
        data = _valid_spec_dict()
        data["sources"]["cps"]["role"] = "donor"
        data["sources"]["puf"]["role"] = "spine"
        # base still points at cps (a donor) -> error.
        with pytest.raises(SpecError, match="must have role 'spine'|does not match"):
            load_spec_dict(data)

    def test_half_with_both_keep_and_strip(self) -> None:
        data = _valid_spec_dict()
        data["spine"]["halves"][0] = {
            "name": "cps_keep",
            "keep": "all",
            "strip_to": ["demographics"],
        }
        with pytest.raises(SpecError, match="exactly one of 'keep' or 'strip_to'"):
            load_spec_dict(data)

    def test_half_with_neither_keep_nor_strip(self) -> None:
        data = _valid_spec_dict()
        data["spine"]["halves"][0] = {"name": "cps_keep"}
        with pytest.raises(SpecError, match="exactly one of 'keep' or 'strip_to'"):
            load_spec_dict(data)

    def test_keep_token_other_than_all_rejected(self) -> None:
        data = _valid_spec_dict()
        data["spine"]["halves"][0] = {"name": "cps_keep", "keep": "most"}
        with pytest.raises(SpecError, match="only keep='all'"):
            load_spec_dict(data)

    def test_two_passthrough_halves_rejected(self) -> None:
        data = _valid_spec_dict()
        data["spine"]["halves"] = [
            {"name": "a", "keep": "all"},
            {"name": "b", "keep": "all"},
        ]
        with pytest.raises(SpecError, match="exactly one passthrough"):
            load_spec_dict(data)

    def test_duplicate_half_names_rejected(self) -> None:
        data = _valid_spec_dict()
        data["spine"]["halves"] = [
            {"name": "dup", "keep": "all"},
            {"name": "dup", "strip_to": ["demographics"]},
        ]
        with pytest.raises(SpecError, match="distinct names"):
            load_spec_dict(data)

    def test_wrong_number_of_halves_rejected(self) -> None:
        data = _valid_spec_dict()
        data["spine"]["halves"] = [{"name": "only", "keep": "all"}]
        with pytest.raises(SpecError, match="halves"):
            load_spec_dict(data)

    def test_old_partition_split_field_rejected(self) -> None:
        data = _valid_spec_dict()
        del data["spine"]["clone"]
        data["spine"]["split"] = {"fraction": 0.5, "seed": 0}
        with pytest.raises(SpecError, match="split"):
            load_spec_dict(data)

    def test_unsupported_spine_method_rejected(self) -> None:
        data = _valid_spec_dict()
        data["spine"]["method"] = "split"
        with pytest.raises(SpecError, match="method"):
            load_spec_dict(data)

    def test_imputation_onto_unknown_half(self) -> None:
        data = _valid_spec_dict()
        data["imputation"][0]["onto"] = "ghost_half"
        with pytest.raises(SpecError, match="is not a declared half or 'both'"):
            load_spec_dict(data)

    def test_base_phase_rejects_half_target(self) -> None:
        data = _valid_spec_dict()
        data["imputation"][0]["at"] = "base"
        data["imputation"][0]["onto"] = "synthetic_puf"
        with pytest.raises(SpecError, match="at 'base' must target 'base'"):
            load_spec_dict(data)

    def test_halves_phase_rejects_base_target(self) -> None:
        data = _valid_spec_dict()
        data["imputation"][0]["onto"] = BASE_TOKEN
        with pytest.raises(SpecError, match="is not a declared half or 'both'"):
            load_spec_dict(data)

    def test_imputation_from_unknown_source(self) -> None:
        data = _valid_spec_dict()
        data["imputation"][0]["from"] = "ghost_source"
        with pytest.raises(SpecError, match="is not a declared source"):
            load_spec_dict(data)

    def test_imputation_empty_vars_rejected(self) -> None:
        data = _valid_spec_dict()
        data["imputation"][0]["vars"] = []
        with pytest.raises(SpecError, match="vars"):
            load_spec_dict(data)

    def test_imputation_duplicate_vars_rejected(self) -> None:
        data = _valid_spec_dict()
        data["imputation"][0]["vars"] = ["x", "x"]
        with pytest.raises(SpecError, match="duplicates"):
            load_spec_dict(data)

    def test_split_transform_fractions_must_sum_to_one(self) -> None:
        data = _valid_spec_dict()
        data["transforms"] = [{"split": {"source": "ss", "into": {"a": 0.5, "b": 0.2}}}]
        with pytest.raises(SpecError, match="must sum to"):
            load_spec_dict(data)

    def test_split_transform_output_collides_with_source(self) -> None:
        data = _valid_spec_dict()
        data["transforms"] = [{"split": {"source": "ss", "into": {"ss": 1.0}}}]
        with pytest.raises(SpecError, match="collides with the source"):
            load_spec_dict(data)

    def test_transform_with_both_split_and_derive(self) -> None:
        data = _valid_spec_dict()
        data["transforms"] = [
            {
                "split": {"source": "ss", "into": {"a": 1.0}},
                "derive": {"target": "t", "expr": "a + b"},
            }
        ]
        with pytest.raises(SpecError, match="exactly one of 'split' or 'derive'"):
            load_spec_dict(data)

    def test_transform_with_neither_split_nor_derive(self) -> None:
        data = _valid_spec_dict()
        data["transforms"] = [{}]
        with pytest.raises(SpecError, match="exactly one of 'split' or 'derive'"):
            load_spec_dict(data)

    def test_unknown_calibration_method_rejected(self) -> None:
        data = _valid_spec_dict()
        data["calibrate"] = {"loss": "x", "method": "magic"}
        with pytest.raises(SpecError, match="method"):
            load_spec_dict(data)
