"""Apply the spec's declared deterministic transforms (split / derive).

This is the :class:`TransformEngine` stage of the spec-driven engine (see
``docs/spec-driven-rebuild.md`` §2 stage 5 and the migration map: "SS-split etc.
become declared ``transforms:`` consumed by ``microplex.TransformEngine``").
The engine applies the rules in :class:`~microplex.spec.TransformSpec` order:

- **split:** partition one source column into named pieces. With fractional
  weights, ``out_i = source * fraction_i`` and the pieces sum back to the
  source exactly. With expression weights, each piece is a pandas-eval
  expression over the frame.
- **derive:** evaluate a pandas-eval expression over existing columns and write
  the result to a new (or overwritten) column.

There is no country logic here; the rules name columns and fractions/expressions
declared in the pack's spec.

Module name note: this is intentionally ``spec_transforms`` rather than
``transforms`` because :mod:`microplex.transforms` already houses the unrelated
numeric/array variable transformers (``LogTransform``, ``Standardizer``, etc.)
used for neural-network training. The blueprint's "TransformEngine" is a
distinct, spec-driven concern, so it lives in its own module to avoid
clobbering that public surface.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from microplex.spec import DeriveTransform, SplitTransform, TransformSpec

logger = logging.getLogger(__name__)

__all__ = [
    "TransformResult",
    "TransformEngine",
]


@dataclass
class TransformResult:
    """Record of what one transform did, for inspection/testing.

    Attributes:
        kind: ``"split"`` or ``"derive"``.
        source: The source column (split) or ``None`` (derive).
        outputs: The output column names written.
    """

    kind: str
    outputs: list[str]
    source: str | None = None


@dataclass
class TransformEngine:
    """Apply declared split/derive transforms to a frame deterministically.

    Args:
        strict_fraction_sum: When ``True`` (default), a fractional split must
            partition the source — the engine asserts the outputs sum back to
            the source column (within ``atol``). Set ``False`` to allow partial
            splits.
        atol: Absolute tolerance for the sum-back check.
    """

    strict_fraction_sum: bool = True
    atol: float = 1e-6
    _results: list[TransformResult] = field(default_factory=list, init=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self, frame: pd.DataFrame, transforms: Sequence[TransformSpec]
    ) -> pd.DataFrame:
        """Apply all transforms in order, returning a new frame.

        Args:
            frame: The input frame.
            transforms: The ordered transform rules.

        Returns:
            A copy of ``frame`` with the transforms applied. Later transforms
            see columns written by earlier ones (e.g. a derive can reference a
            split output).
        """
        out = frame.copy()
        self._results = []
        for index, rule in enumerate(transforms):
            if rule.split is not None:
                out = self._apply_split(out, rule.split, index)
            elif rule.derive is not None:
                out = self._apply_derive(out, rule.derive, index)
            else:  # pragma: no cover - schema guarantees exactly one is set
                raise ValueError(
                    f"transform[{index}] has neither split nor derive set."
                )
        return out

    @property
    def results(self) -> list[TransformResult]:
        """Per-transform results from the most recent :meth:`apply` call."""
        return list(self._results)

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def _apply_split(
        self, frame: pd.DataFrame, split: SplitTransform, index: int
    ) -> pd.DataFrame:
        if split.source not in frame.columns:
            raise ValueError(
                f"transform[{index}] split: source column '{split.source}' not "
                "in frame."
            )
        collisions = [
            name
            for name in split.into
            if name in frame.columns and name != split.source
        ]
        if collisions:
            raise ValueError(
                f"transform[{index}] split on '{split.source}': output "
                f"column(s) {collisions} already exist in the frame."
            )

        source = frame[split.source]
        out = frame.copy()

        if split.is_fractional:
            for name, fraction in split.into.items():
                out[name] = source * float(fraction)
            if self.strict_fraction_sum:
                self._assert_sums_back(out, split)
        else:
            for name, expr in split.into.items():
                if isinstance(expr, (int, float)):
                    out[name] = source * float(expr)
                else:
                    out[name] = self._eval(out, expr, index, context=f"split '{name}'")

        self._results.append(
            TransformResult(
                kind="split",
                source=split.source,
                outputs=list(split.into.keys()),
            )
        )
        logger.info(
            "transform[%d] split '%s' -> %s",
            index,
            split.source,
            list(split.into.keys()),
        )
        return out

    def _assert_sums_back(self, frame: pd.DataFrame, split: SplitTransform) -> None:
        """Assert the split outputs sum back to the source within tolerance."""
        recombined = sum(frame[name] for name in split.into)
        diff = (recombined - frame[split.source]).abs()
        max_diff = float(diff.max()) if len(diff) else 0.0
        if max_diff > self.atol:
            raise ValueError(
                f"split on '{split.source}' does not sum back to the source "
                f"(max abs diff {max_diff:.3g} > atol {self.atol:.3g})."
            )

    # ------------------------------------------------------------------
    # Derive
    # ------------------------------------------------------------------

    def _apply_derive(
        self, frame: pd.DataFrame, derive: DeriveTransform, index: int
    ) -> pd.DataFrame:
        out = frame.copy()
        out[derive.target] = self._eval(
            out, derive.expr, index, context=f"derive '{derive.target}'"
        )
        self._results.append(TransformResult(kind="derive", outputs=[derive.target]))
        logger.info(
            "transform[%d] derive '%s' = %s",
            index,
            derive.target,
            derive.expr,
        )
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _eval(
        self, frame: pd.DataFrame, expr: str, index: int, *, context: str
    ) -> pd.Series:
        """Evaluate a pandas expression against the frame.

        Uses the pandas (non-Python) engine and ``local_dict={}`` so the
        expression can only reference frame columns, not arbitrary globals —
        keeping declared transforms deterministic and side-effect free.
        """
        try:
            result = frame.eval(expr, engine="python", local_dict={}, global_dict={})
        except Exception as exc:  # noqa: BLE001 - re-raise with field context
            raise ValueError(
                f"transform[{index}] {context}: failed to evaluate "
                f"expression {expr!r}: {exc}"
            ) from exc
        if not isinstance(result, pd.Series):
            # Scalar (e.g. "0") broadcasts to the frame length.
            result = pd.Series(result, index=frame.index)
        return result
