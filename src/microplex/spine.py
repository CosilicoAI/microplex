"""Build the spine: split a base survey frame into two disjoint halves, one
kept whole and one stripped to its declared columns for synthesis.

This is the eCPS ``puf_clone`` pattern generalized (see
``docs/spec-driven-rebuild.md`` §4). The reference implementation
(``policyengine_us_data/calibration/puf_impute.py:puf_clone_dataset``) doubles
CPS by hand: one half keeps CPS values, the other half has its tax variables
synthesized from demographic predictors. :class:`SpineBuilder` does the same
thing generically, driven by the ``spine:`` section of a
:class:`microplex.spec.MicroplexSpec`.

Crucially (the correctness anchor), the stripped half keeps **only** the
declared columns (demographics + the entity id) — its income variables are
dropped so the imputation runner can synthesize the full distribution from
scratch, rather than inheriting the survey's (different) income tail. There is
no country logic here: what "demographics" means is supplied by the caller as a
column group.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
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
    """Split a base frame into a passthrough half and a stripped half.

    The split is deterministic (seeded), disjoint, and covers every base row.
    The passthrough half keeps all base columns; the stripped half keeps only
    the columns named in its ``strip_to`` list (resolving the ``demographics``
    group token via ``column_groups``). A label column records each row's half.

    Args:
        spine: The validated :class:`~microplex.spec.SpineSpec`.
        column_groups: Mapping from group token (e.g. ``"demographics"``) to the
            concrete column names it expands to. Required if any half's
            ``strip_to`` references a group token. Entries in ``strip_to`` that
            are not group tokens are treated as literal column names.
        half_label_column: Name of the column recording the half label.

    Notes:
        This class is country-agnostic. It never inspects column *meaning* — it
        only partitions rows and selects/drops columns by name. The demographic
        column set is injected by the caller, not hard-coded.
    """

    def __init__(
        self,
        spine: SpineSpec,
        *,
        column_groups: Mapping[str, Sequence[str]] | None = None,
        half_label_column: str = DEFAULT_HALF_LABEL_COLUMN,
    ) -> None:
        self.spine = spine
        self.column_groups = {
            token: list(cols) for token, cols in (column_groups or {}).items()
        }
        self.half_label_column = half_label_column

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, base: pd.DataFrame) -> SpineBuildResult:
        """Split ``base`` into the declared halves.

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

        first_mask = self._split_mask(len(base))

        # The passthrough/stripped roles are independent of split position: the
        # spec marks one half keep:all and the other strip_to:. We assign the
        # FIRST half (per the spec's declared order) to first_mask, the SECOND
        # to its complement, so the split fraction maps to the first-declared
        # half exactly as written.
        first_half, second_half = self.spine.halves

        halves: dict[str, pd.DataFrame] = {}
        halves[first_half.name] = self._materialize_half(
            base.loc[first_mask], first_half
        )
        halves[second_half.name] = self._materialize_half(
            base.loc[~first_mask], second_half
        )

        stacked = pd.concat(
            [halves[first_half.name], halves[second_half.name]],
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

    def _split_mask(self, n: int) -> np.ndarray:
        """Return a boolean mask selecting the FIRST half's rows.

        Deterministic given the spec's seed; the two halves are disjoint and
        partition all ``n`` rows. The first half receives ``round(fraction*n)``
        rows.
        """
        rng = np.random.default_rng(self.spine.split.seed)
        permutation = rng.permutation(n)
        n_first = int(round(self.spine.split.fraction * n))
        # Guard the degenerate ends so neither half is empty for non-trivial n.
        if n >= 2:
            n_first = max(1, min(n - 1, n_first))
        first_positions = permutation[:n_first]
        mask = np.zeros(n, dtype=bool)
        mask[first_positions] = True
        return mask

    def _materialize_half(self, rows: pd.DataFrame, half: HalfSpec) -> pd.DataFrame:
        """Select/drop columns for one half and append the label column."""
        if half.is_passthrough:
            out = rows.copy()
        else:
            keep = self.resolve_columns(half)
            missing = [c for c in keep if c not in rows.columns]
            if missing:
                raise ValueError(
                    f"half '{half.name}' strip_to requires columns missing from "
                    f"the base frame: {missing}."
                )
            out = rows.loc[:, keep].copy()
        out = out.reset_index(drop=True)
        out[self.half_label_column] = half.name
        return out
