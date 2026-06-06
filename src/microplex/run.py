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
5. **Targets** (:class:`~microplex.targets.TargetProvider`) — when a provider is
   supplied, load the spec-declared target surface and attach it to the result.

Calibration and export are **not yet wired** here — they are marked as explicit
``TODO`` stages (see :data:`PENDING_STAGES`) and the blueprint's build order (§6
steps 3, 6). ``run_spec`` returns the post-transform frame plus any loaded
target set; a later phase will reweight to it and export the PolicyEngine
dataset.

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
from microplex.targets.provider import TargetProvider, TargetQuery
from microplex.targets.spec import TargetSet

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
        target_set: The spec-declared target set when a target provider was
            supplied; otherwise ``None`` and ``targets`` remains pending.
        pending_stages: Stages declared but not yet run (see
            :data:`PENDING_STAGES`).
    """

    frame: pd.DataFrame
    spine: SpineBuildResult
    halves: dict[str, pd.DataFrame]
    imputation_results: list[ImputationStepResult] = field(default_factory=list)
    target_set: TargetSet | None = None
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
    target_provider: TargetProvider | None = None,
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
        target_provider: Optional provider used to load the spec-declared target
            surface. When omitted, targets remain an explicit pending stage.
        seed: Seed forwarded to the default imputer.

    Returns:
        A :class:`RunResult`. ``frame`` is the post-transform stacked frame;
        ``target_set`` is populated only when ``target_provider`` is supplied;
        ``pending_stages`` lists the not-yet-wired stages.

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

    target_set: TargetSet | None = None
    pending_stages = list(PENDING_STAGES)

    # Stage 5: targets. A provider-backed load is the first non-faked seam for
    # the clean scoring/calibration surface. Calibration/export still remain
    # explicit TODOs; we deliberately do not fabricate weights or a dataset.
    if spec.targets is not None and target_provider is not None:
        target_query = _target_query_from_spec(spec)
        target_set = target_provider.load_target_set(target_query)
        pending_stages.remove("targets")
        logger.info(
            "run_spec: loaded %d targets for profile '%s' "
            "(calibration profile '%s')",
            len(target_set.targets),
            spec.targets.arch.target_profile,
            spec.targets.arch.resolved_calibration_target_profile,
        )
    elif spec.targets is not None:
        logger.info(
            "run_spec: targets declared but no target_provider was supplied; "
            "returning the post-transform frame. Pending: %s",
            tuple(pending_stages),
        )

    return RunResult(
        frame=final_frame,
        spine=spine_result,
        halves=halves,
        imputation_results=imputation_results,
        target_set=target_set,
        pending_stages=tuple(pending_stages),
    )


def _target_query_from_spec(spec: MicroplexSpec) -> TargetQuery:
    """Build the provider query for the spec-declared Arch target surface."""
    if spec.targets is None:
        raise ValueError("cannot build a target query for a spec without targets")
    arch = spec.targets.arch
    provider_filters = {
        "source": "arch",
        "country": arch.country,
        "model_year": arch.model_year,
    }
    if arch.target_profile is not None:
        provider_filters["target_profile"] = arch.target_profile
    if arch.resolved_calibration_target_profile is not None:
        provider_filters["calibration_target_profile"] = (
            arch.resolved_calibration_target_profile
        )
    return TargetQuery(
        period=arch.model_year,
        provider_filters=provider_filters,
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
