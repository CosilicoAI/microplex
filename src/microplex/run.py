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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from microplex.core import EntityType
from microplex.data_sources.source_impute import (
    SourceImputeManifest,
    compile_source_impute_steps_from_manifest,
)
from microplex.imputation import (
    ImputationRunner,
    ImputationStepResult,
)
from microplex.runtime_operations import (
    RuntimeVariableOperationHandler,
    RuntimeVariableOperationResult,
    apply_runtime_variable_operations,
)
from microplex.source_registry import SourceRegistry
from microplex.spec import (
    BOTH_TOKEN,
    CalibrateSpec,
    ImputationOrder,
    ImputationPhase,
    ImputationStep,
    MicroplexSpec,
    VariableOperationKind,
)
from microplex.spec_transforms import TransformEngine
from microplex.spine import SpineBuilder, SpineBuildResult
from microplex.targets.arch_manifest import arch_target_provider_from_consumer_facts
from microplex.targets.bundles import EntityTableBinding, EntityTableBundle
from microplex.targets.provider import TargetProvider, TargetQuery
from microplex.targets.spec import TargetSet

logger = logging.getLogger(__name__)

__all__ = [
    "PENDING_STAGES",
    "RunResult",
    "SpecCalibrationResult",
    "SpecCalibrator",
    "SourceImputeStageResult",
    "resolve_sources",
    "run_source_impute_stage",
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
        variable_operation_results: Runtime variable-operation batches that
            materialized post-transform columns before target loading or
            calibration.
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
    variable_operation_results: tuple[RuntimeVariableOperationResult, ...] = ()
    entity_table_bundle: EntityTableBundle | None = None
    pending_stages: tuple[str, ...] = PENDING_STAGES


@dataclass(frozen=True)
class SourceImputeStageResult:
    """Output of a post-geography source-imputation stage."""

    run_result: RunResult
    sources: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    imputation_results: list[ImputationStepResult] = field(default_factory=list)


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


def run_source_impute_stage(
    run_result: RunResult,
    source_impute_spec: MicroplexSpec,
    sources: Mapping[str, pd.DataFrame] | SourceRegistry,
    *,
    source_impute_manifest: SourceImputeManifest | str | Path,
    source_impute_blocks: Sequence[str] | None = None,
    source_impute_imputation_steps: Sequence[str] | None = None,
    column_groups: Mapping[str, Sequence[str]] | None = None,
    demographic_columns: Sequence[str] | None = None,
    weight_column: str | None = "household_weight",
    spine_keywords: Sequence[str] | None = None,
    seed: int = 0,
) -> SourceImputeStageResult:
    """Run manifest-backed source imputation on an already-geocoded result.

    This stage is the explicit post-geography seam used by US first-principles
    builds: callers run the support spine, assign atomic geography, and only
    then execute SCF/SIPP/ACS source imputations over the resolved support
    universe. It shares the same manifest compilation and block/step filtering
    semantics as ``run_spec(..., source_impute_manifest=...)``.
    """
    full_source_manifest = _load_source_impute_manifest(source_impute_manifest)
    source_manifest = _filter_source_impute_manifest(
        full_source_manifest,
        blocks=source_impute_blocks,
    )
    source_steps = _source_impute_steps(
        source_impute_spec,
        source_manifest,
        full_manifest=full_source_manifest,
        imputation_steps=source_impute_imputation_steps,
        manifest_was_filtered=source_impute_blocks is not None,
    )
    if not source_steps:
        return SourceImputeStageResult(run_result=run_result)

    _require_source_impute_stage_boundary(run_result)
    _require_post_geography_halves(run_result.halves)
    source_names = tuple(dict.fromkeys(step.from_ for step in source_steps))
    donors = _resolve_selected_sources(source_impute_spec, sources, source_names)
    resolved_groups = _resolved_column_groups(
        column_groups=column_groups,
        demographic_columns=demographic_columns,
    )
    runner = ImputationRunner(
        column_groups=resolved_groups,
        weight_column=weight_column,
        spine_keywords=tuple(spine_keywords or ()),
        seed=seed,
    )
    halves, results = runner.run(
        source_steps,
        halves=run_result.halves,
        donors=donors,
    )
    updated_run_result = replace(
        run_result,
        halves=halves,
        frame=_restack_run_result_halves(halves, run_result),
        imputation_results=[*run_result.imputation_results, *results],
    )
    return SourceImputeStageResult(
        run_result=updated_run_result,
        sources=donors,
        imputation_results=results,
    )


def _resolved_column_groups(
    *,
    column_groups: Mapping[str, Sequence[str]] | None,
    demographic_columns: Sequence[str] | None,
) -> dict[str, list[str]]:
    resolved = {
        group: list(columns) for group, columns in (column_groups or {}).items()
    }
    if demographic_columns is not None:
        resolved["demographics"] = list(demographic_columns)
    return resolved


def _resolve_selected_sources(
    spec: MicroplexSpec,
    sources: Mapping[str, pd.DataFrame] | SourceRegistry,
    source_names: Sequence[str],
) -> dict[str, pd.DataFrame]:
    if isinstance(sources, SourceRegistry):
        return {
            source_name: sources.resolve_source(spec, source_name)
            for source_name in source_names
        }

    missing = [
        source_name for source_name in source_names if source_name not in sources
    ]
    if missing:
        raise KeyError(
            "missing frames for source-impute donor source(s): "
            f"{missing}. Provide frames keyed by source name."
        )
    return {source_name: sources[source_name] for source_name in source_names}


def _restack_run_result_halves(
    halves: Mapping[str, pd.DataFrame],
    run_result: RunResult,
) -> pd.DataFrame:
    ordered = [halves[name] for name in run_result.spine.halves if name in halves]
    extras = [
        frame for name, frame in halves.items() if name not in run_result.spine.halves
    ]
    return pd.concat(ordered + extras, axis=0, ignore_index=True)


def _require_source_impute_stage_boundary(run_result: RunResult) -> None:
    """Reject post-transform results before they can lose transform outputs."""
    restacked = _restack_run_result_halves(run_result.halves, run_result)
    half_column_names = set(restacked.columns)
    frame_column_names = set(run_result.frame.columns)
    if frame_column_names != half_column_names:
        extra = sorted(frame_column_names - half_column_names)
        missing = sorted(half_column_names - frame_column_names)
        raise ValueError(
            "run_source_impute_stage requires a pre-transform RunResult whose "
            "frame columns match its halves; call it after geography assignment "
            "and before transforms. "
            f"Extra frame columns: {extra}; missing frame columns: {missing}."
        )
    frame_rows = len(run_result.frame)
    half_rows = len(restacked)
    if frame_rows != half_rows:
        raise ValueError(
            "run_source_impute_stage requires a pre-transform RunResult whose "
            f"frame row count matches its halves; frame has {frame_rows} rows "
            f"but halves sum to {half_rows} rows."
        )
    frame = run_result.frame.reset_index(drop=True)
    restacked = restacked.loc[:, list(run_result.frame.columns)].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(frame, restacked, check_dtype=False)
    except AssertionError as exc:
        raise ValueError(
            "run_source_impute_stage requires a pre-transform RunResult whose "
            "frame values match its halves; call it after geography assignment "
            "and before transforms."
        ) from exc


def run_spec(
    spec: MicroplexSpec,
    sources: Mapping[str, pd.DataFrame] | SourceRegistry,
    *,
    column_groups: Mapping[str, Sequence[str]] | None = None,
    demographic_columns: Sequence[str] | None = None,
    weight_column: str | None = "household_weight",
    spine_keywords: Sequence[str] | None = None,
    target_provider: TargetProvider | None = None,
    arch_consumer_fact_paths: str | Path | Sequence[str | Path] | None = None,
    arch_reference_consumer_fact_paths: str | Path | Sequence[str | Path] | None = None,
    arch_target_manifest_base: str | Path | None = None,
    calibrator: SpecCalibrator | None = None,
    calibration_entity: EntityType | str | None = None,
    calibration_id_column: str | None = None,
    simulation_compiler: Any | None = None,
    calibration_certificate: Mapping[str, Any] | None = None,
    calibration_min_records_per_target: float | None = None,
    allow_skipped_calibration_targets: bool = False,
    variable_operation_handlers: (
        Mapping[str, RuntimeVariableOperationHandler] | None
    ) = None,
    source_impute_manifest: SourceImputeManifest | str | Path | None = None,
    source_impute_blocks: Sequence[str] | None = None,
    source_impute_imputation_steps: Sequence[str] | None = None,
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
        arch_consumer_fact_paths: Optional Arch consumer-fact JSONL files or
            artifact directories used to construct an ``ArchTargetProvider``
            from ``spec.targets.arch.manifest``. Mutually exclusive with
            ``target_provider``.
        arch_reference_consumer_fact_paths: Optional Arch consumer-fact JSONL
            files or artifact directories used only as derivation reference
            records. Requires ``arch_consumer_fact_paths``.
        arch_target_manifest_base: Base directory for a relative
            ``spec.targets.arch.manifest`` path when constructing an Arch target
            provider from JSONL artifacts.
        calibrator: Optional country-specific calibrator used to reweight the
            post-transform frame to the loaded target surface. Requires both
            ``spec.targets`` and either ``target_provider`` or
            ``arch_consumer_fact_paths`` so calibration never runs against an
            implicit target surface.
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
        variable_operation_handlers: Optional runtime handlers keyed by
            ``variables[*].mp_spec.operation.handler`` (or operation kind when
            no handler is set). When supplied, ``materialize_policyengine`` and
            ``rerandomize_takeup`` operations run after transforms and before
            target loading/calibration.
        source_impute_manifest: Optional source-impute block manifest. When
            supplied, executable ``variables[*].mp_spec.operation`` rows backed
            by the manifest run after the spine/half imputation stage and before
            transforms. The support halves must already have ``block_geoid``.
        source_impute_blocks: Optional manifest block names to execute. This
            narrows the manifest before compiling operations.
        source_impute_imputation_steps: Optional operation step names to execute
            after the optional block filter is applied.
        seed: Seed forwarded to ``microimpute.Imputer``.

    Returns:
        A :class:`RunResult`. ``frame`` is the post-transform stacked frame;
        ``target_set`` is populated only when ``target_provider`` is supplied;
        ``pending_stages`` lists the not-yet-wired stages.

    Raises:
        KeyError: if a declared source has no frame.
        ValueError: on spine/imputation/transform validation failures.
    """
    _validate_target_provider_inputs(
        target_provider=target_provider,
        arch_consumer_fact_paths=arch_consumer_fact_paths,
        arch_reference_consumer_fact_paths=arch_reference_consumer_fact_paths,
    )
    resolved_groups = _resolved_column_groups(
        column_groups=column_groups,
        demographic_columns=demographic_columns,
    )

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
    stage_result = RunResult(
        frame=_restack(halves, spec, spine_result.half_label_column),
        spine=spine_result,
        base=base,
        halves=halves,
        imputation_results=[*base_imputation_results, *half_imputation_results],
    )
    if source_impute_manifest is not None:
        source_stage = run_source_impute_stage(
            stage_result,
            spec,
            donors,
            source_impute_manifest=source_impute_manifest,
            source_impute_blocks=source_impute_blocks,
            source_impute_imputation_steps=source_impute_imputation_steps,
            column_groups=resolved_groups,
            weight_column=weight_column,
            spine_keywords=resolved_spine_keywords,
            seed=seed,
        )
        stage_result = source_stage.run_result
        if source_stage.imputation_results:
            logger.info(
                "run_spec: ran %d post-geography source-impute result(s)",
                len(source_stage.imputation_results),
            )
    halves = stage_result.halves
    imputation_results = stage_result.imputation_results

    # Re-stack the (imputed) halves in the spine's declared order so the
    # transform stage sees the full frame, with the half-label column intact.
    stacked = stage_result.frame

    # Stage 4: transforms.
    transform_engine = TransformEngine()
    final_frame = transform_engine.apply(stacked, spec.transforms)
    variable_operation_results: tuple[RuntimeVariableOperationResult, ...] = ()
    if variable_operation_handlers is not None:
        final_frame, variable_operation_results = apply_runtime_variable_operations(
            final_frame,
            spec=spec,
            handlers=variable_operation_handlers,
        )
        if variable_operation_results:
            logger.info(
                "run_spec: materialized %d runtime variable-operation batch(es)",
                len(variable_operation_results),
            )

    target_set: TargetSet | None = None
    calibration_result: SpecCalibrationResult | None = None
    pending_stages = list(PENDING_STAGES)

    # Stage 5: targets. A provider-backed load is the first non-faked seam for
    # the clean scoring/calibration surface. Calibration/export still remain
    # explicit TODOs; we deliberately do not fabricate weights or a dataset.
    resolved_target_provider = _target_provider_from_inputs(
        spec,
        target_provider=target_provider,
        arch_consumer_fact_paths=arch_consumer_fact_paths,
        arch_reference_consumer_fact_paths=arch_reference_consumer_fact_paths,
        arch_target_manifest_base=arch_target_manifest_base,
    )
    if spec.targets is not None and resolved_target_provider is not None:
        target_query = _target_query_from_spec(spec)
        target_set = resolved_target_provider.load_target_set(target_query)
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
        variable_operation_results=variable_operation_results,
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


def _load_source_impute_manifest(
    manifest: SourceImputeManifest | str | Path,
) -> SourceImputeManifest:
    return (
        manifest
        if isinstance(manifest, SourceImputeManifest)
        else SourceImputeManifest.from_path(manifest)
    )


def _filter_source_impute_manifest(
    manifest: SourceImputeManifest,
    *,
    blocks: Sequence[str] | None,
) -> SourceImputeManifest:
    if blocks is None:
        return manifest
    return SourceImputeManifest(
        blocks={block_name: manifest.block(block_name) for block_name in blocks}
    )


def _source_impute_steps(
    spec: MicroplexSpec,
    manifest: SourceImputeManifest,
    *,
    full_manifest: SourceImputeManifest,
    imputation_steps: Sequence[str] | None,
    manifest_was_filtered: bool,
) -> list[ImputationStep]:
    if manifest_was_filtered:
        return _compile_selected_block_source_impute_steps(
            spec,
            selected_manifest=manifest,
            full_manifest=full_manifest,
            imputation_steps=imputation_steps,
        )
    return compile_source_impute_steps_from_manifest(
        spec,
        manifest,
        imputation_steps=imputation_steps,
    )


def _compile_selected_block_source_impute_steps(
    spec: MicroplexSpec,
    *,
    selected_manifest: SourceImputeManifest,
    full_manifest: SourceImputeManifest,
    imputation_steps: Sequence[str] | None,
) -> list[ImputationStep]:
    requested = set(imputation_steps) if imputation_steps is not None else None
    if requested == set():
        return []
    selected_surveys = {
        block.survey_name for block in selected_manifest.blocks.values()
    }
    grouped_variables: dict[tuple[str, str], list[str]] = {}
    grouped_blocks: dict[tuple[str, str], Any] = {}
    unresolved: list[str] = []

    for variable_name, variable in spec.variables.items():
        operation = variable.mp_spec.operation if variable.mp_spec else None
        if operation is None or operation.kind is not VariableOperationKind.IMPUTE:
            continue
        if operation.imputation_step is None or operation.source is None:
            continue
        if requested is not None and operation.imputation_step not in requested:
            continue

        selected_blocks = [
            block
            for block in selected_manifest.blocks.values()
            if operation.source == block.survey_name
            and variable_name in block.target_variables
        ]
        if len(selected_blocks) > 1:
            block_names = [block.name for block in selected_blocks]
            raise ValueError(
                "source-impute variable operation is ambiguous across selected "
                f"manifest blocks: {variable_name} appears in {block_names}"
            )
        if selected_blocks:
            block = selected_blocks[0]
            key = (operation.imputation_step, block.name)
            grouped_variables.setdefault(key, []).append(variable_name)
            grouped_blocks[key] = block
            continue

        if operation.source not in selected_surveys:
            continue
        full_blocks = [
            block
            for block in full_manifest.blocks.values()
            if operation.source == block.survey_name
            and variable_name in block.target_variables
        ]
        if full_blocks:
            continue
        unresolved.append(
            f"{variable_name} ({operation.imputation_step} from "
            f"{operation.source}: not a manifest target)"
        )

    if unresolved:
        raise ValueError(
            "source-impute variable operations are not backed by manifest target "
            f"variables: {unresolved}"
        )

    return [
        ImputationStep(
            onto=BOTH_TOKEN,
            **{"from": block.survey_name},
            vars=variables,
            condition_on=list(block.predictors),
            at=ImputationPhase.HALVES,
            order=ImputationOrder.AS_DECLARED,
        )
        for key, variables in grouped_variables.items()
        for block in [grouped_blocks[key]]
    ]


def _require_post_geography_halves(halves: Mapping[str, pd.DataFrame]) -> None:
    for half_name, half in halves.items():
        if "block_geoid" not in half.columns:
            raise ValueError(
                "source-impute execution requires post-geography halves with "
                f"block_geoid; half {half_name!r} is missing block_geoid"
            )
        block_geoid = half["block_geoid"]
        if block_geoid.isna().any():
            raise ValueError(
                "source-impute execution requires non-null block_geoid values; "
                f"half {half_name!r} contains null block_geoid"
            )
        if not block_geoid.astype("string").str.len().eq(15).all():
            raise ValueError(
                "source-impute execution requires 15-character block_geoid values; "
                f"half {half_name!r} has malformed block_geoid"
            )


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


def _validate_target_provider_inputs(
    *,
    target_provider: TargetProvider | None,
    arch_consumer_fact_paths: str | Path | Sequence[str | Path] | None,
    arch_reference_consumer_fact_paths: str | Path | Sequence[str | Path] | None,
) -> None:
    if target_provider is not None and arch_consumer_fact_paths is not None:
        raise ValueError(
            "pass either target_provider or arch_consumer_fact_paths, not both"
        )
    if (
        arch_reference_consumer_fact_paths is not None
        and arch_consumer_fact_paths is None
    ):
        raise ValueError(
            "arch_reference_consumer_fact_paths requires arch_consumer_fact_paths"
        )


def _target_provider_from_inputs(
    spec: MicroplexSpec,
    *,
    target_provider: TargetProvider | None,
    arch_consumer_fact_paths: str | Path | Sequence[str | Path] | None,
    arch_reference_consumer_fact_paths: str | Path | Sequence[str | Path] | None,
    arch_target_manifest_base: str | Path | None,
) -> TargetProvider | None:
    if arch_consumer_fact_paths is None:
        return target_provider
    if spec.targets is None:
        raise ValueError(
            "arch_consumer_fact_paths were supplied but the spec has no targets section"
        )
    return arch_target_provider_from_consumer_facts(
        _resolve_arch_target_manifest_path(
            spec.targets.arch.manifest,
            base=arch_target_manifest_base,
        ),
        arch_consumer_fact_paths,
        target_year=spec.targets.arch.model_year,
        reference_paths=arch_reference_consumer_fact_paths,
    )


def _resolve_arch_target_manifest_path(
    manifest: str | None,
    *,
    base: str | Path | None,
) -> Path:
    if manifest is None:
        raise ValueError(
            "arch_consumer_fact_paths require targets.arch.manifest in the spec"
        )
    path = Path(manifest)
    if path.is_absolute():
        return path
    if base is None:
        raise ValueError(
            "arch_target_manifest_base is required to resolve relative "
            f"targets.arch.manifest {manifest!r}"
        )
    return Path(base) / path


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
