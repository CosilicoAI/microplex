"""The thin spec-driven pipeline: sequence the generic engine stages.

``run_spec(spec, sources)`` runs the country-agnostic stages of the
spec-driven engine (see ``docs/spec-driven-rebuild.md`` §2) over a validated
:class:`~microplex.spec.MicroplexSpec`:

1. **Sources** — resolve the spec's declared sources to loaded frames.
2. **Base imputation** (:class:`~microplex.imputation.ImputationRunner`) — run
   any ``at: base`` steps before the spine split, so source-level predictors are
   present on both halves.
3. **Spine** (:class:`~microplex.spine.SpineBuilder`) — split the enriched base
   into passthrough and stripped synthetic halves.
4. **Half imputation** (:class:`~microplex.imputation.ImputationRunner`) —
   synthesize the ``at: halves`` variable graph onto the halves via canonical
   microimpute.
5. **Transforms** (:class:`~microplex.spec_transforms.TransformEngine`) — apply
   declared split/derive rules to the stacked frame.
6. **Targets** (:class:`~microplex.targets.TargetProvider`) — when a provider is
   supplied, load the spec-declared target surface and attach it to the result.
7. **Calibration** — when both a target provider and a generic entity
   calibration binding are supplied, compile the post-transform entity table
   into the certified sparse target matrix path and reweight with
   ``microcalibrate``. A legacy :class:`SpecCalibrator` protocol remains
   available as a compatibility fallback.

Export is **not yet wired** here — it is marked as an explicit ``TODO`` stage
(see :data:`PENDING_STAGES`) and the blueprint's build order (§6 step 6).
``run_spec`` returns the post-transform or calibrated frame plus any loaded
target set, entity-table bundle, and calibration diagnostics; a later phase
will export the PolicyEngine dataset.

Source resolution contract: ``run_spec`` takes an already-loaded
``{source_name: DataFrame}`` mapping. Wiring the full provider-backed
``SourceRegistry`` (which loads + harmonizes datasets named in
``sources[*].dataset``) is a separate seam — see :func:`resolve_sources`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from microplex.core import EntityType
from microplex.imputation import (
    ImputationRunner,
    ImputationStepResult,
)
from microplex.source_registry import SourceRegistry
from microplex.spec import (
    CalibrateSpec,
    ImputationOrder,
    ImputationPhase,
    ImputationStep,
    MicroplexSpec,
)
from microplex.spec_transforms import TransformEngine
from microplex.spine import SpineBuilder, SpineBuildResult
from microplex.targets.bundles import EntityTableBinding, EntityTableBundle
from microplex.targets.provider import TargetProvider, TargetQuery
from microplex.targets.spec import TargetSet

logger = logging.getLogger(__name__)

__all__ = [
    "PENDING_STAGES",
    "RunResult",
    "SpecCalibrationResult",
    "SpecCalibrator",
    "resolve_sources",
    "run_spec",
]

#: Stages declared by the spec/blueprint that are not yet wired into
#: :func:`run_spec`. Each is a clear TODO, not a stub that fabricates output.
PENDING_STAGES: tuple[str, ...] = (
    "targets",  # ArchTargetProvider: fetch + roll up the Arch target set.
    "calibrate",  # Reweight only with explicit target/calibration bindings.
    "export",  # Exporter: write the PolicyEngine dataset.
)


@dataclass(frozen=True)
class SpecCalibrationResult:
    """The output of a spec-runner calibration stage."""

    frame: pd.DataFrame
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    entity_table_bundle: EntityTableBundle | None = None


@runtime_checkable
class SpecCalibrator(Protocol):
    """Compatibility protocol for pre-entity-table calibration adapters."""

    def calibrate(
        self,
        frame: pd.DataFrame,
        *,
        target_set: TargetSet,
        calibrate: CalibrateSpec,
        weight_column: str | None,
    ) -> SpecCalibrationResult | pd.DataFrame:
        """Return a calibrated frame for the loaded target surface."""


@dataclass
class RunResult:
    """The output of :func:`run_spec`.

    Attributes:
        frame: The final post-transform stacked frame (both spine halves).
        spine: The :class:`~microplex.spine.SpineBuildResult` from stage 2.
        base: The spine-source frame after all ``at: base`` imputation steps
            and before the split.
        halves: The per-half frames after imputation (before stacking for
            transforms), keyed by half name.
        imputation_results: Per-(step, half) imputation outcomes.
        target_set: The spec-declared target set when a target provider was
            supplied; otherwise ``None`` and ``targets`` remains pending.
        calibration_result: The calibration output when a calibrator was
            supplied and run; otherwise ``None`` and ``calibrate`` remains
            pending.
        entity_table_bundle: The calibrated entity tables when the generic
            entity-table calibration path ran; otherwise ``None``.
        pending_stages: Stages declared but not yet run (see
            :data:`PENDING_STAGES`).
    """

    frame: pd.DataFrame
    spine: SpineBuildResult
    base: pd.DataFrame
    halves: dict[str, pd.DataFrame]
    imputation_results: list[ImputationStepResult] = field(default_factory=list)
    target_set: TargetSet | None = None
    calibration_result: SpecCalibrationResult | None = None
    entity_table_bundle: EntityTableBundle | None = None
    pending_stages: tuple[str, ...] = PENDING_STAGES


def resolve_sources(
    spec: MicroplexSpec,
    sources: Mapping[str, pd.DataFrame] | SourceRegistry,
) -> dict[str, pd.DataFrame]:
    """Validate and return the loaded source frames for a spec.

    ``sources`` can be either an already-loaded mapping keyed by source name
    or a provider-backed :class:`~microplex.source_registry.SourceRegistry`
    keyed by the spec's ``sources[*].dataset`` ids.

    Args:
        spec: The validated spec.
        sources: Mapping of source name -> loaded frame.

    Returns:
        A plain dict copy of ``sources`` restricted to the declared names.

    Raises:
        KeyError: if a declared source has no frame.
    """
    if isinstance(sources, SourceRegistry):
        return sources.resolve_sources(spec)

    missing = [name for name in spec.sources if name not in sources]
    if missing:
        raise KeyError(
            f"missing frames for declared sources: {missing}. Provide a frame "
            "for every source named in the spec (keyed by source name)."
        )
    return {name: sources[name] for name in spec.sources}


def run_spec(
    spec: MicroplexSpec,
    sources: Mapping[str, pd.DataFrame] | SourceRegistry,
    *,
    column_groups: Mapping[str, Sequence[str]] | None = None,
    demographic_columns: Sequence[str] | None = None,
    weight_column: str | None = "household_weight",
    spine_keywords: Sequence[str] | None = None,
    target_provider: TargetProvider | None = None,
    calibrator: SpecCalibrator | None = None,
    calibration_entity: EntityType | str | None = None,
    calibration_id_column: str | None = None,
    simulation_compiler: Any | None = None,
    calibration_certificate: Mapping[str, Any] | None = None,
    calibration_min_records_per_target: float | None = None,
    allow_skipped_calibration_targets: bool = False,
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
        spine_keywords: Pack-specific keyword list for the spine-first ordering
            heuristic. Required when the spec declares any ``order:
            spine_first`` step, so country packs do not accidentally rely on
            broad generic substrings that can mis-tier variables.
        target_provider: Optional provider used to load the spec-declared target
            surface. When omitted, targets remain an explicit pending stage.
        calibrator: Optional country-specific calibrator used to reweight the
            post-transform frame to the loaded target surface. Requires both
            ``spec.targets`` and ``target_provider`` so calibration never runs
            against an implicit or freshly recomputed target surface.
        calibration_entity: Optional entity label for generic entity-table
            calibration of the post-transform frame. When set, ``run_spec``
            builds a one-table :class:`EntityTableBundle` from ``frame`` and
            calibrates it through ``EntityTableBundleMicrocalibrator``.
        calibration_id_column: Record id column for the post-transform frame
            when ``calibration_entity`` is set.
        simulation_compiler: Optional simulator-aware target compiler for
            targets that declare simulation modifiers.
        calibration_certificate: Optional sparse target matrix certificate to
            assert before fitting.
        calibration_min_records_per_target: Optional fail-closed floor passed
            into the calibration solve policy.
        allow_skipped_calibration_targets: Explicit opt-in for partial target
            surfaces. Defaults to ``False`` so skipped target rows fail before
            fitting.
        seed: Seed forwarded to ``microimpute.Imputer``.

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
    base = frames[spec.spine_source].copy()
    donors = {name: frames[name] for name in spec.sources}
    logger.info(
        "run_spec: %d sources, spine base '%s' (%d rows)",
        len(frames),
        spec.spine_source,
        len(base),
    )

    # Stage 2: base imputation. These source-level steps run before the spine
    # split, and the updated base is what the spine builder partitions. Keep the
    # donor mapping in sync so later steps can use the enriched spine source as
    # a donor.
    resolved_spine_keywords = _resolve_spine_keywords(spec, spine_keywords)
    runner = ImputationRunner(
        column_groups=resolved_groups,
        weight_column=weight_column,
        spine_keywords=resolved_spine_keywords,
        seed=seed,
    )
    base_steps = [step for step in spec.imputation if step.at is ImputationPhase.BASE]
    half_steps = [step for step in spec.imputation if step.at is ImputationPhase.HALVES]
    base, base_imputation_results = _run_base_imputation_steps(
        base_steps,
        base=base,
        donors=donors,
        runner=runner,
        spine_source=spec.spine_source,
    )
    donors[spec.spine_source] = base

    # Stage 3: spine.
    spine_builder = SpineBuilder(spec.spine, column_groups=resolved_groups)
    spine_result = spine_builder.build(base)

    # Stage 4: half imputation.
    halves, half_imputation_results = runner.run(
        half_steps,
        halves=spine_result.halves,
        donors=donors,
    )
    imputation_results = [*base_imputation_results, *half_imputation_results]

    # Re-stack the (imputed) halves in the spine's declared order so the
    # transform stage sees the full frame, with the half-label column intact.
    stacked = _restack(halves, spec, spine_result.half_label_column)

    # Stage 4: transforms.
    transform_engine = TransformEngine()
    final_frame = transform_engine.apply(stacked, spec.transforms)

    target_set: TargetSet | None = None
    calibration_result: SpecCalibrationResult | None = None
    pending_stages = list(PENDING_STAGES)

    # Stage 5: targets. A provider-backed load is the first non-faked seam for
    # the clean scoring/calibration surface. Calibration/export still remain
    # explicit TODOs; we deliberately do not fabricate weights or a dataset.
    if spec.targets is not None and target_provider is not None:
        target_query = _target_query_from_spec(spec)
        target_set = target_provider.load_target_set(target_query)
        pending_stages.remove("targets")
        logger.info(
            "run_spec: loaded %d targets for profile '%s' (calibration profile '%s')",
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

    # Stage 6: calibration. This seam is intentionally strict: calibrating
    # without a loaded TargetSet would recreate the stale eCPS-surface failure
    # mode that the release gates now forbid.
    if calibrator is not None and calibration_entity is not None:
        raise ValueError(
            "pass either a legacy calibrator or calibration_entity, not both"
        )

    entity_table_bundle: EntityTableBundle | None = None
    if calibration_entity is not None:
        if spec.calibrate is None:
            raise ValueError(
                "calibration_entity was supplied but the spec has no "
                "'calibrate' section"
            )
        if target_set is None:
            raise ValueError(
                "calibration_entity requires a loaded target_set; supply "
                "target_provider for the spec-declared target surface"
            )
        bundle_result = _calibrate_entity_frame(
            final_frame,
            entity=calibration_entity,
            id_column=calibration_id_column,
            weight_column=weight_column,
            target_set=target_set,
            calibrate=spec.calibrate,
            simulation_compiler=simulation_compiler,
            certificate=calibration_certificate,
            min_records_per_target=calibration_min_records_per_target,
            allow_skipped_targets=allow_skipped_calibration_targets,
        )
        entity_type = (
            calibration_entity
            if isinstance(calibration_entity, EntityType)
            else EntityType(calibration_entity)
        )
        entity_table_bundle = bundle_result.bundle
        final_frame = bundle_result.bundle.table_for(entity_type)
        calibration_result = SpecCalibrationResult(
            frame=final_frame,
            diagnostics=bundle_result.diagnostics(),
            entity_table_bundle=bundle_result.bundle,
        )
        pending_stages.remove("calibrate")
        logger.info(
            "run_spec: calibrated entity table '%s' with %d targets using '%s'/%s",
            entity_type.value,
            len(target_set.targets),
            spec.calibrate.loss,
            spec.calibrate.method.value,
        )
    elif calibrator is not None:
        if spec.calibrate is None:
            raise ValueError(
                "calibrator was supplied but the spec has no 'calibrate' section"
            )
        if target_set is None:
            raise ValueError(
                "calibrator requires a loaded target_set; supply target_provider "
                "for the spec-declared target surface"
            )
        raw_calibration_result = calibrator.calibrate(
            final_frame,
            target_set=target_set,
            calibrate=spec.calibrate,
            weight_column=weight_column,
        )
        if isinstance(raw_calibration_result, pd.DataFrame):
            calibration_result = SpecCalibrationResult(frame=raw_calibration_result)
        else:
            calibration_result = raw_calibration_result
        final_frame = calibration_result.frame
        pending_stages.remove("calibrate")
        logger.info(
            "run_spec: calibrated frame with %d targets using '%s'/%s",
            len(target_set.targets),
            spec.calibrate.loss,
            spec.calibrate.method.value,
        )

    return RunResult(
        frame=final_frame,
        spine=spine_result,
        base=base,
        halves=halves,
        imputation_results=imputation_results,
        target_set=target_set,
        calibration_result=calibration_result,
        entity_table_bundle=entity_table_bundle,
        pending_stages=tuple(pending_stages),
    )


def _calibrate_entity_frame(
    frame: pd.DataFrame,
    *,
    entity: EntityType | str,
    id_column: str | None,
    weight_column: str | None,
    target_set: TargetSet,
    calibrate: CalibrateSpec,
    simulation_compiler: Any | None,
    certificate: Mapping[str, Any] | None,
    min_records_per_target: float | None,
    allow_skipped_targets: bool,
):
    """Calibrate one post-transform entity frame through the generic bundle path."""
    if id_column is None:
        raise ValueError(
            "calibration_entity requires calibration_id_column so weights can "
            "be synced and audited by stable record id"
        )
    if id_column not in frame.columns:
        raise ValueError(
            f"calibration_id_column {id_column!r} is not present in the frame"
        )
    if frame[id_column].isna().any():
        raise ValueError(f"calibration_id_column {id_column!r} contains null ids")
    if not frame[id_column].is_unique:
        raise ValueError(f"calibration_id_column {id_column!r} contains duplicate ids")
    if weight_column is None:
        raise ValueError("calibration_entity requires a weight_column")
    if weight_column not in frame.columns:
        raise ValueError(f"weight_column {weight_column!r} is not present in the frame")

    try:
        from microplex.calibration.microcalibrate_adapter import (
            EntityTableBundleMicrocalibrator,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "generic entity-table calibration requires the microcalibrate extra"
        ) from exc

    entity_type = entity if isinstance(entity, EntityType) else EntityType(entity)
    bundle = EntityTableBundle(
        weight_entity=entity_type,
        weight_column=weight_column,
        bindings={
            entity_type: EntityTableBinding(
                frame=frame,
                id_column=id_column,
            )
        },
    )
    bundle_calibrator = EntityTableBundleMicrocalibrator(
        simulation_compiler=simulation_compiler,
        min_records_per_target=min_records_per_target,
        allow_skipped_targets=allow_skipped_targets,
    )
    return bundle_calibrator.calibrate_bundle(
        bundle,
        target_set=target_set,
        calibrate=calibrate,
        certificate=certificate,
    )


def _run_base_imputation_steps(
    steps: Sequence[ImputationStep],
    *,
    base: pd.DataFrame,
    donors: Mapping[str, pd.DataFrame],
    runner: ImputationRunner,
    spine_source: str,
) -> tuple[pd.DataFrame, list[ImputationStepResult]]:
    """Run ``at: base`` imputation steps before the spine split.

    The base frame mutates across steps so later source-level imputations can
    condition on earlier ones. Results record the concrete spine source name,
    even when a spec uses the declarative ``onto: base`` alias.
    """
    working = base.copy()
    results: list[ImputationStepResult] = []
    for step in steps:
        if step.from_ not in donors:
            raise KeyError(f"imputation step references unknown donor '{step.from_}'.")
        working, result = runner.run_step(
            step,
            donor=donors[step.from_],
            target=working,
        )
        result.onto = spine_source
        results.append(result)
    return working, results


def _resolve_spine_keywords(
    spec: MicroplexSpec,
    spine_keywords: Sequence[str] | None,
) -> tuple[str, ...]:
    """Return explicit pack keywords required by ``order: spine_first``.

    ``spine_first`` controls the microimpute chain order and therefore the
    statistical meaning of a spec. The low-level imputation module keeps a
    generic default for direct use, but full spec runs must fail closed unless
    the pack supplies its own reviewed variable-name markers.
    """
    if any(step.order is ImputationOrder.SPINE_FIRST for step in spec.imputation):
        if spine_keywords is None:
            raise ValueError(
                "run_spec requires explicit spine_keywords when the spec declares "
                "order: spine_first; pass pack-specific receipt/income markers "
                "instead of relying on generic substring defaults"
            )
        if len(spine_keywords) == 0:
            raise ValueError(
                "run_spec received an empty spine_keywords list for order: spine_first"
            )
    return tuple(spine_keywords or ())


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
