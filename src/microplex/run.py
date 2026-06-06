"""The thin spec-driven pipeline: sequence the generic engine stages.

``run_spec(spec, sources)`` runs the country-agnostic stages of the
spec-driven engine (see ``docs/spec-driven-rebuild.md`` §2) over a validated
:class:`~microplex.spec.MicroplexSpec`:

1. **Sources** — resolve the spec's declared sources to loaded frames.
2. **Spine** (:class:`~microplex.spine.SpineBuilder`) — clone the base into a
   passthrough half and a stripped synthetic half.
3. **Imputation** (:class:`~microplex.imputation.ImputationRunner`) — synthesize
   the declared variable graph onto the halves via canonical microimpute.
4. **Transforms** (:class:`~microplex.spec_transforms.TransformEngine`) — apply
   declared split/derive rules to the stacked frame.

Targets, calibration, and export are **not yet wired** here — they are marked
as explicit ``TODO`` stages (see :data:`PENDING_STAGES`) and the blueprint's
build order (§6 steps 3, 6). ``run_spec`` returns the post-transform frame; a
later phase will fetch the Arch target set, reweight to it, and export the
PolicyEngine dataset.

Source resolution contract: ``run_spec`` takes an already-loaded
``{source_name: DataFrame}`` mapping. Wiring the full provider-backed
``SourceRegistry`` (which loads + harmonizes datasets named in
``sources[*].dataset``) is a separate seam — see :func:`resolve_sources`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from microplex.imputation import (
    SPINE_FIRST_KEYWORDS,
    ImputationRunner,
    ImputationStepResult,
)
from microplex.spec import MicroplexSpec
from microplex.spec_transforms import TransformEngine
from microplex.spine import SpineBuilder, SpineBuildResult

logger = logging.getLogger(__name__)

__all__ = [
    "PENDING_STAGES",
    "RunResult",
    "resolve_sources",
    "run_spec",
]

#: Stages declared by the spec/blueprint that are not yet wired into
#: :func:`run_spec`. Each is a clear TODO, not a stub that fabricates output.
PENDING_STAGES: tuple[str, ...] = (
    "targets",  # ArchTargetProvider: fetch + roll up the Arch target set.
    "calibrate",  # Calibrator: reweight to targets via the declared loss/method.
    "export",  # Exporter: write the PolicyEngine dataset.
)


@dataclass
class RunResult:
    """The output of :func:`run_spec`.

    Attributes:
        frame: The final post-transform stacked frame (both spine halves).
        spine: The :class:`~microplex.spine.SpineBuildResult` from stage 2.
        halves: The per-half frames after imputation (before stacking for
            transforms), keyed by half name.
        imputation_results: Per-(step, half) imputation outcomes.
        pending_stages: Stages declared but not yet run (see
            :data:`PENDING_STAGES`).
    """

    frame: pd.DataFrame
    spine: SpineBuildResult
    halves: dict[str, pd.DataFrame]
    imputation_results: list[ImputationStepResult] = field(default_factory=list)
    pending_stages: tuple[str, ...] = PENDING_STAGES


def resolve_sources(
    spec: MicroplexSpec,
    sources: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Validate and return the loaded source frames for a spec.

    This is the seam where a full provider-backed ``SourceRegistry`` would
    load + harmonize the datasets named in ``spec.sources[*].dataset``. For now
    the caller supplies already-loaded frames keyed by the spec's *source
    names*; this function only checks that every declared source is present.

    Args:
        spec: The validated spec.
        sources: Mapping of source name -> loaded frame.

    Returns:
        A plain dict copy of ``sources`` restricted to the declared names.

    Raises:
        KeyError: if a declared source has no frame.
    """
    missing = [name for name in spec.sources if name not in sources]
    if missing:
        raise KeyError(
            f"missing frames for declared sources: {missing}. Provide a frame "
            "for every source named in the spec (keyed by source name)."
        )
    return {name: sources[name] for name in spec.sources}


def run_spec(
    spec: MicroplexSpec,
    sources: Mapping[str, pd.DataFrame],
    *,
    column_groups: Mapping[str, Sequence[str]] | None = None,
    demographic_columns: Sequence[str] | None = None,
    weight_column: str | None = "household_weight",
    spine_keywords: Sequence[str] = SPINE_FIRST_KEYWORDS,
    imputer_factory=None,
    seed: int = 0,
) -> RunResult:
    """Run the wired stages of the spec-driven engine end-to-end.

    Args:
        spec: A validated :class:`~microplex.spec.MicroplexSpec`.
        sources: Mapping of source name -> loaded frame (see
            :func:`resolve_sources`).
        column_groups: Group-token -> columns mapping (e.g. ``demographics``),
            shared by the spine builder and imputation runner.
        demographic_columns: Convenience for the ``demographics`` group; takes
            precedence over ``column_groups['demographics']`` when set.
        weight_column: Sampling-weight column for weighted imputation fits.
        spine_keywords: Keyword list for the spine-first ordering heuristic.
        imputer_factory: Optional callable returning a fresh imputer per step
            (defaults to canonical regime-aware ``microimpute.Imputer``).
        seed: Seed forwarded to the default imputer.

    Returns:
        A :class:`RunResult`. ``frame`` is the post-transform stacked frame;
        ``pending_stages`` lists the not-yet-wired targets/calibrate/export
        stages.

    Raises:
        KeyError: if a declared source has no frame.
        ValueError: on spine/imputation/transform validation failures.
    """
    resolved_groups: dict[str, list[str]] = {
        token: list(cols) for token, cols in (column_groups or {}).items()
    }
    if demographic_columns is not None:
        resolved_groups["demographics"] = list(demographic_columns)

    # Stage 1: sources.
    frames = resolve_sources(spec, sources)
    base = frames[spec.spine_source]
    donors = {name: frames[name] for name in spec.sources}
    logger.info(
        "run_spec: %d sources, spine base '%s' (%d rows)",
        len(frames),
        spec.spine_source,
        len(base),
    )

    # Stage 2: spine.
    spine_builder = SpineBuilder(spec.spine, column_groups=resolved_groups)
    spine_result = spine_builder.build(base)

    # Stage 3: imputation.
    runner = ImputationRunner(
        column_groups=resolved_groups,
        weight_column=weight_column,
        spine_keywords=spine_keywords,
        imputer_factory=imputer_factory,
        seed=seed,
    )
    halves, imputation_results = runner.run(
        spec.imputation,
        halves=spine_result.halves,
        donors=donors,
    )

    # Re-stack the (imputed) halves in the spine's declared order so the
    # transform stage sees the full frame, with the half-label column intact.
    stacked = _restack(halves, spec, spine_result.half_label_column)

    # Stage 4: transforms.
    transform_engine = TransformEngine()
    final_frame = transform_engine.apply(stacked, spec.transforms)

    # Stages 5+ (targets / calibrate / export): not yet wired. See
    # PENDING_STAGES and docs/spec-driven-rebuild.md §6. We deliberately do not
    # fabricate weights or a calibrated dataset here.
    if spec.targets is not None or spec.calibrate is not None:
        logger.info(
            "run_spec: targets/calibrate declared but not yet wired; "
            "returning the post-transform frame. Pending: %s",
            PENDING_STAGES,
        )

    return RunResult(
        frame=final_frame,
        spine=spine_result,
        halves=halves,
        imputation_results=imputation_results,
        pending_stages=PENDING_STAGES,
    )


def _restack(
    halves: Mapping[str, pd.DataFrame],
    spec: MicroplexSpec,
    half_label_column: str,
) -> pd.DataFrame:
    """Concatenate the imputed halves in the spine's declared order.

    Ensures the half-label column is present on each half (the imputation
    runner preserves it from the spine builder's output) before stacking.
    """
    ordered = [halves[name] for name in spec.spine.half_names if name in halves]
    # A half could in principle be absent (e.g. a future spec variant); stack
    # whatever halves we have, in declared order, then any extras.
    extras = [
        frame for name, frame in halves.items() if name not in spec.spine.half_names
    ]
    return pd.concat(ordered + extras, axis=0, ignore_index=True)
