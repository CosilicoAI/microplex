"""Tests for SpineBuilder (microplex.spine) on synthetic frames.

Covers: disjoint + total-cover split, stripped columns dropped on the
synthetic half while demographics + ids are kept, passthrough half keeps
everything, correct half labels, determinism, and error cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microplex.spec import HalfSpec, SpineSpec, SplitSpec
from microplex.spine import DEFAULT_HALF_LABEL_COLUMN, SpineBuilder

DEMOGRAPHIC_COLS = ["age", "is_male", "tax_unit_is_joint"]
INCOME_COLS = ["employment_income", "capital_gains", "interest_income"]


def _synthetic_base(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """A small CPS-like frame with demographics, an id, and income columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "tax_unit_id": np.arange(n),
            "age": rng.integers(18, 80, n),
            "is_male": rng.integers(0, 2, n),
            "tax_unit_is_joint": rng.integers(0, 2, n),
            "employment_income": rng.normal(50000, 20000, n).clip(min=0),
            "capital_gains": rng.normal(2000, 5000, n).clip(min=0),
            "interest_income": rng.normal(1000, 2000, n).clip(min=0),
            "household_weight": rng.uniform(1000, 5000, n),
        }
    )


def _spine_spec(fraction: float = 0.5, seed: int = 0) -> SpineSpec:
    """A spine: first half keeps all; second stripped to demographics + id."""
    return SpineSpec(
        base="cps",
        split=SplitSpec(fraction=fraction, seed=seed),
        halves=[
            HalfSpec(name="cps_keep", keep="all"),
            HalfSpec(name="synthetic", strip_to=["demographics", "tax_unit_id"]),
        ],
    )


def _builder(spine: SpineSpec | None = None) -> SpineBuilder:
    return SpineBuilder(
        spine or _spine_spec(),
        column_groups={"demographics": DEMOGRAPHIC_COLS},
    )


class TestSplit:
    def test_total_cover(self) -> None:
        base = _synthetic_base(200)
        result = _builder().build(base)
        assert len(result.frame) == len(base)

    def test_halves_are_disjoint_and_partition_ids(self) -> None:
        base = _synthetic_base(200)
        result = _builder().build(base)
        keep_ids = set(result.halves["cps_keep"]["tax_unit_id"])
        synth_ids = set(result.halves["synthetic"]["tax_unit_id"])
        # Disjoint.
        assert keep_ids.isdisjoint(synth_ids)
        # Together cover every base id exactly once.
        assert keep_ids | synth_ids == set(base["tax_unit_id"])
        assert len(keep_ids) + len(synth_ids) == len(base)

    def test_fraction_assigns_first_half(self) -> None:
        base = _synthetic_base(100)
        spine = _spine_spec(fraction=0.3, seed=1)
        result = SpineBuilder(
            spine, column_groups={"demographics": DEMOGRAPHIC_COLS}
        ).build(base)
        # 30% to the first-declared half (cps_keep).
        assert len(result.halves["cps_keep"]) == 30
        assert len(result.halves["synthetic"]) == 70

    def test_determinism_same_seed(self) -> None:
        base = _synthetic_base(200)
        r1 = _builder().build(base)
        r2 = _builder().build(base)
        ids1 = set(r1.halves["cps_keep"]["tax_unit_id"])
        ids2 = set(r2.halves["cps_keep"]["tax_unit_id"])
        assert ids1 == ids2

    def test_different_seed_different_split(self) -> None:
        base = _synthetic_base(200)
        r_a = SpineBuilder(
            _spine_spec(seed=0), column_groups={"demographics": DEMOGRAPHIC_COLS}
        ).build(base)
        r_b = SpineBuilder(
            _spine_spec(seed=999), column_groups={"demographics": DEMOGRAPHIC_COLS}
        ).build(base)
        ids_a = set(r_a.halves["cps_keep"]["tax_unit_id"])
        ids_b = set(r_b.halves["cps_keep"]["tax_unit_id"])
        assert ids_a != ids_b

    def test_neither_half_empty_for_extreme_fraction(self) -> None:
        base = _synthetic_base(50)
        # fraction must be in (0,1); a tiny fraction still yields >=1 row each.
        spine = _spine_spec(fraction=0.001, seed=3)
        result = SpineBuilder(
            spine, column_groups={"demographics": DEMOGRAPHIC_COLS}
        ).build(base)
        assert len(result.halves["cps_keep"]) >= 1
        assert len(result.halves["synthetic"]) >= 1


class TestColumns:
    def test_passthrough_half_keeps_all_columns(self) -> None:
        base = _synthetic_base()
        result = _builder().build(base)
        keep = result.halves["cps_keep"]
        for col in base.columns:
            assert col in keep.columns, f"passthrough dropped {col}"

    def test_stripped_half_drops_income_columns(self) -> None:
        base = _synthetic_base()
        result = _builder().build(base)
        synth = result.halves["synthetic"]
        for col in INCOME_COLS:
            assert col not in synth.columns, f"stripped half still has {col}"
        # household_weight is also not in strip_to -> dropped.
        assert "household_weight" not in synth.columns

    def test_stripped_half_keeps_demographics_and_id(self) -> None:
        base = _synthetic_base()
        result = _builder().build(base)
        synth = result.halves["synthetic"]
        for col in DEMOGRAPHIC_COLS + ["tax_unit_id"]:
            assert col in synth.columns, f"stripped half is missing {col}"

    def test_resolve_columns_expands_demographics_group(self) -> None:
        builder = _builder()
        cols = builder.resolve_columns(builder.spine.synthetic_half)
        assert cols == [*DEMOGRAPHIC_COLS, "tax_unit_id"]

    def test_resolve_columns_dedupes(self) -> None:
        spine = SpineSpec(
            base="cps",
            split=SplitSpec(fraction=0.5),
            halves=[
                HalfSpec(name="keep", keep="all"),
                # 'age' appears both via the group and literally.
                HalfSpec(name="synthetic", strip_to=["demographics", "age"]),
            ],
        )
        builder = SpineBuilder(spine, column_groups={"demographics": DEMOGRAPHIC_COLS})
        cols = builder.resolve_columns(spine.synthetic_half)
        assert cols == DEMOGRAPHIC_COLS  # 'age' not duplicated
        assert cols.count("age") == 1


class TestLabels:
    def test_label_column_present_and_correct(self) -> None:
        base = _synthetic_base()
        result = _builder().build(base)
        assert result.half_label_column == DEFAULT_HALF_LABEL_COLUMN
        labels = set(result.frame[DEFAULT_HALF_LABEL_COLUMN].unique())
        assert labels == {"cps_keep", "synthetic"}

    def test_label_values_match_half_membership(self) -> None:
        base = _synthetic_base()
        result = _builder().build(base)
        frame = result.frame
        keep_rows = frame[frame[DEFAULT_HALF_LABEL_COLUMN] == "cps_keep"]
        synth_rows = frame[frame[DEFAULT_HALF_LABEL_COLUMN] == "synthetic"]
        # Synthetic rows have NaN income (column absent -> stacked as NaN).
        assert "employment_income" in keep_rows.columns
        assert keep_rows["employment_income"].notna().all()
        assert synth_rows["employment_income"].isna().all()

    def test_custom_label_column(self) -> None:
        base = _synthetic_base()
        builder = SpineBuilder(
            _spine_spec(),
            column_groups={"demographics": DEMOGRAPHIC_COLS},
            half_label_column="origin",
        )
        result = builder.build(base)
        assert "origin" in result.frame.columns
        assert result.half_label_column == "origin"


class TestErrors:
    def test_label_column_collision(self) -> None:
        base = _synthetic_base()
        base[DEFAULT_HALF_LABEL_COLUMN] = "x"
        with pytest.raises(ValueError, match="already exists"):
            _builder().build(base)

    def test_missing_strip_column_raises(self) -> None:
        base = _synthetic_base().drop(columns=["age"])
        with pytest.raises(ValueError, match="missing from the base frame"):
            _builder().build(base)

    def test_unmapped_group_token_raises(self) -> None:
        base = _synthetic_base()
        # No column_groups provided -> demographics token unresolved.
        builder = SpineBuilder(_spine_spec())
        with pytest.raises(ValueError, match="no column_groups mapping"):
            builder.build(base)

    def test_resolve_columns_on_passthrough_raises(self) -> None:
        builder = _builder()
        with pytest.raises(ValueError, match="passthrough"):
            builder.resolve_columns(builder.spine.passthrough_half)
