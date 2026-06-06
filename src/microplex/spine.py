"""Build the spine: clone a base survey frame into passthrough and synthetic halves.

This is the eCPS ``puf_clone`` pattern generalized (see
``docs/spec-driven-rebuild.md`` §4). The reference implementation
(``policyengine_us_data/calibration/puf_impute.py:puf_clone_dataset``) doubles
CPS by hand: one half keeps CPS values, the other half has its tax variables
synthesized from demographic predictors. :class:`SpineBuilder` does the same
thing generically, driven by the ``spine:`` section of a
:class:`microplex.spec.MicroplexSpec`.

Crucially (the correctness anchor), the synthetic half keeps **only** the
declared columns (demographics + entity ids, plus zeroed weights) -- its income
variables are dropped so the imputation runner can synthesize the full
distribution from scratch, rather than inheriting the survey's tail-deficient
income distribution. There is no country logic here: what "demographics" means
is supplied by the caller as a column group.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from microplex.spec import DEMOGRAPHICS_TOKEN, HalfSpec, SpineSpec

__all__ = [
    "SpineBuildResult",
    "SpineBuilder",
    "DEFAULT_HALF_LABEL_COLUMN",
]

#: Column added to every output row recording which half it came from.
DEFAULT_HALF_LABEL_COLUMN = "_spine_half"


@dataclass(frozen=True)
class SpineBuildResult:
    """The output of :meth:`SpineBuilder.build`.

    Attributes:
        frame: The concatenated spine (both halves stacked), with the
            half-label column appended. Row order is the passthrough half
            followed by the stripped half; the index is reset.
        half_label_column: Name of the label column in ``frame``.
        halves: Mapping from half name to that half's sub-frame (views with
            their original-but-reset index). Convenience for callers that want
            the halves separately; ``frame`` is the canonical stacked output.
    """

    frame: pd.DataFrame
    half_label_column: str
    halves: Mapping[str, pd.DataFrame]


class SpineBuilder:
    """Clone a base frame into a passthrough half and a synthetic half.

    The passthrough half keeps all base columns. The synthetic half is a cloned
    copy of every base row, stripped to declared columns, with numeric id
    columns offset and weight columns set to zero. This mirrors PE-US-data's
    PUF clone shape: original records first, synthetic PUF copies second.

    Args:
        spine: The validated :class:`~microplex.spec.SpineSpec`.
        column_groups: Mapping from group token (e.g. ``"demographics"``) to the
            concrete column names it expands to. Required if any half's
            ``strip_to`` references a group token. Entries in ``strip_to`` that
            are not group tokens are treated as literal column names.
        id_columns: ID columns whose synthetic-half values should be offset so
            clone entities cannot collide with original entities. Defaults to
            all base columns ending in ``"_id"``.
        weight_columns: Weight columns to retain on the synthetic half and set
            to ``clone_weight_value``. Defaults to all base columns ending in
            ``"_weight"``.
        clone_weight_value: Initial value for synthetic-half weights.
        half_label_column: Name of the column recording the half label.

    Notes:
        This class is country-agnostic. It never inspects column *meaning* — it
        only clones rows and selects/drops columns by name. The demographic
        column set is injected by the caller, not hard-coded.
    """

    def __init__(
        self,
        spine: SpineSpec,
        *,
        column_groups: Mapping[str, Sequence[str]] | None = None,
        id_columns: Sequence[str] | None = None,
        weight_columns: Sequence[str] | None = None,
        clone_weight_value: float = 0.0,
        half_label_column: str = DEFAULT_HALF_LABEL_COLUMN,
    ) -> None:
        self.spine = spine
        self.column_groups = {
            token: list(cols) for token, cols in (column_groups or {}).items()
        }
        self.id_columns = tuple(id_columns) if id_columns is not None else None
        self.weight_columns = (
            tuple(weight_columns) if weight_columns is not None else None
        )
        self.clone_weight_value = clone_weight_value
        self.half_label_column = half_label_column

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, base: pd.DataFrame) -> SpineBuildResult:
        """Clone ``base`` into the declared passthrough and synthetic halves.

        Args:
            base: The base survey frame (one row per spine entity).

        Returns:
            A :class:`SpineBuildResult` with the stacked frame and per-half
            sub-frames.

        Raises:
            ValueError: if the label column collides with an existing column,
                if a stripped column is missing from ``base``, or if a group
                token used in ``strip_to`` has no mapping.
        """
        if self.half_label_column in base.columns:
            raise ValueError(
                f"half label column '{self.half_label_column}' already exists in "
                "the base frame; pass a different half_label_column."
            )
        normalized = base.reset_index(drop=True)
        passthrough_half = self.spine.passthrough_half
        synthetic_half = self.spine.synthetic_half

        halves: dict[str, pd.DataFrame] = {}
        halves[passthrough_half.name] = self._materialize_half(
            normalized, passthrough_half, synthetic=False
        )
        halves[synthetic_half.name] = self._materialize_half(
            normalized, synthetic_half, synthetic=True
        )

        stacked = pd.concat(
            [halves[passthrough_half.name], halves[synthetic_half.name]],
            axis=0,
            ignore_index=True,
        )

        return SpineBuildResult(
            frame=stacked,
            half_label_column=self.half_label_column,
            halves=halves,
        )

    def resolve_columns(self, half: HalfSpec) -> list[str]:
        """Resolve a stripped half's ``strip_to`` list into concrete columns.

        Expands the ``demographics`` (and any other configured) group token,
        de-duplicates while preserving order, and leaves literal column names
        untouched. Raises if a group token has no mapping.
        """
        if half.strip_to is None:
            raise ValueError(
                f"half '{half.name}' is passthrough; it has no strip_to columns."
            )
        resolved: list[str] = []
        for entry in half.strip_to:
            if entry in self.column_groups:
                resolved.extend(self.column_groups[entry])
            elif entry == DEMOGRAPHICS_TOKEN:
                raise ValueError(
                    f"half '{half.name}' strip_to references the "
                    f"'{DEMOGRAPHICS_TOKEN}' group but no column_groups mapping "
                    "was provided for it."
                )
            else:
                resolved.append(entry)
        # De-duplicate, preserving first-seen order.
        seen: set[str] = set()
        unique: list[str] = []
        for col in resolved:
            if col not in seen:
                seen.add(col)
                unique.append(col)
        return unique

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _materialize_half(
        self,
        rows: pd.DataFrame,
        half: HalfSpec,
        *,
        synthetic: bool,
    ) -> pd.DataFrame:
        """Select/drop columns for one half and append the label column."""
        if half.is_passthrough:
            out = rows.copy()
        else:
            keep = self._columns_for_synthetic_half(rows, half)
            missing = [c for c in keep if c not in rows.columns]
            if missing:
                raise ValueError(
                    f"half '{half.name}' strip_to requires columns missing from "
                    f"the base frame: {missing}."
                )
            out = rows.loc[:, keep].copy()
        out = out.reset_index(drop=True)
        if synthetic:
            self._offset_id_columns(out, rows)
            for column in self._weight_columns(rows):
                if column in out.columns:
                    out[column] = self.clone_weight_value
        out[self.half_label_column] = half.name
        return out

    def _columns_for_synthetic_half(
        self,
        base: pd.DataFrame,
        half: HalfSpec,
    ) -> list[str]:
        """Columns retained on the synthetic half before imputation."""
        columns = [
            *self.resolve_columns(half),
            *self._id_columns(base),
            *self._weight_columns(base),
        ]
        unique = dict.fromkeys(columns)
        return list(unique)

    def _id_columns(self, base: pd.DataFrame) -> tuple[str, ...]:
        """Resolve id columns, defaulting to columns ending in '_id'."""
        if self.id_columns is not None:
            return self.id_columns
        return tuple(column for column in base.columns if column.endswith("_id"))

    def _weight_columns(self, base: pd.DataFrame) -> tuple[str, ...]:
        """Resolve weight columns, defaulting to columns ending in '_weight'."""
        if self.weight_columns is not None:
            return self.weight_columns
        return tuple(column for column in base.columns if column.endswith("_weight"))

    def _offset_id_columns(
        self,
        clone: pd.DataFrame,
        base: pd.DataFrame,
    ) -> None:
        """Offset synthetic-half id columns so original/clone ids are disjoint."""
        for column in self._id_columns(base):
            if column not in clone.columns:
                continue
            if not pd.api.types.is_numeric_dtype(clone[column]):
                raise ValueError(
                    f"id column '{column}' must be numeric to offset clone ids."
                )
            max_value = base[column].max()
            offset = 0 if pd.isna(max_value) else int(max_value) + 1
            clone[column] = clone[column] + offset
