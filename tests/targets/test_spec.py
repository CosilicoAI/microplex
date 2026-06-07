"""Tests for the canonical target specification primitives."""

from microplex.core import EntityType
from microplex.targets import (
    FilterOperator,
    TargetAggregation,
    TargetFilter,
    TargetSet,
    TargetSimulationModifier,
    TargetSpec,
)


class TestTargetFilter:
    def test_operator_normalizes_from_string(self):
        target_filter = TargetFilter(feature="snap", operator=">", value=0)

        assert target_filter.operator is FilterOperator.GT


class TestTargetSpec:
    def test_entity_and_aggregation_normalize_from_strings(self):
        target = TargetSpec(
            name="snap_recipients",
            entity="spm_unit",
            value=100.0,
            period=2024,
            aggregation="count",
            filters=(TargetFilter(feature="snap", operator=">", value=0),),
        )

        assert target.entity is EntityType.SPM_UNIT
        assert target.aggregation is TargetAggregation.COUNT

    def test_count_target_rejects_measure(self):
        try:
            TargetSpec(
                name="bad_target",
                entity=EntityType.HOUSEHOLD,
                value=1.0,
                period=2024,
                measure="income",
                aggregation=TargetAggregation.COUNT,
            )
        except ValueError as exc:
            assert "Count targets" in str(exc)
        else:
            raise AssertionError("Expected ValueError for count target with measure")

    def test_required_features_deduplicates_and_preserves_order(self):
        target = TargetSpec(
            name="california_snap",
            entity=EntityType.HOUSEHOLD,
            value=1_000.0,
            period=2024,
            measure="snap",
            aggregation=TargetAggregation.SUM,
            filters=(
                TargetFilter(feature="state_fips", operator="==", value="06"),
                TargetFilter(feature="snap", operator=">", value=0),
            ),
        )

        assert target.required_features == ("snap", "state_fips")

    def test_sim_modifiers_normalize_from_dicts(self):
        target = TargetSpec(
            name="snap_after_takeup",
            entity=EntityType.SPM_UNIT,
            value=1_000.0,
            period=2024,
            measure="snap",
            aggregation=TargetAggregation.SUM,
            sim_modifiers=(
                {
                    "name": "rerandomize_takeup",
                    "parameters": {"program": "snap", "seed": 7},
                },
            ),
        )

        assert target.requires_simulation is True
        assert target.sim_modifier_names == ("rerandomize_takeup",)
        assert target.sim_modifiers == (
            TargetSimulationModifier(
                name="rerandomize_takeup",
                parameters={"program": "snap", "seed": 7},
            ),
        )

    def test_sim_modifiers_reject_duplicate_names(self):
        try:
            TargetSpec(
                name="bad_takeup",
                entity=EntityType.SPM_UNIT,
                value=1_000.0,
                period=2024,
                measure="snap",
                aggregation=TargetAggregation.SUM,
                sim_modifiers=(
                    TargetSimulationModifier("rerandomize_takeup"),
                    TargetSimulationModifier(
                        "rerandomize_takeup", {"program": "medicaid"}
                    ),
                ),
            )
        except ValueError as exc:
            assert "distinct names" in str(exc)
        else:
            raise AssertionError("Expected ValueError for duplicate sim modifiers")


class TestTargetSet:
    def test_collection_helpers(self):
        targets = TargetSet(
            targets=[
                TargetSpec(
                    name="households",
                    entity=EntityType.HOUSEHOLD,
                    value=10.0,
                    period=2024,
                    aggregation=TargetAggregation.COUNT,
                ),
                TargetSpec(
                    name="people",
                    entity=EntityType.PERSON,
                    value=20.0,
                    period=2025,
                    aggregation=TargetAggregation.COUNT,
                ),
            ]
        )

        assert len(targets.for_entity(EntityType.HOUSEHOLD)) == 1
        assert len(targets.for_period(2025)) == 1
        assert targets.required_features() == ()
