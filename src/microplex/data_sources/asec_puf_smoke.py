"""ASEC+PUF support-spine smoke run for the first real-data engine slice."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd

from microplex.data_sources.census_blocks import CensusBlockCrosswalkProvider
from microplex.data_sources.puf import SHARED_VARS
from microplex.data_sources.source_impute import SourceImputeManifest
from microplex.data_sources.us_registry import (
    create_us_asec_puf_source_registry,
    register_us_declared_source_impute_blocks,
)
from microplex.geography import (
    LowestAvailableAtomicGeographyAssigner,
    LowestAvailableGeographyAssignmentPlan,
    normalize_string_code,
    normalize_us_state_fips,
)
from microplex.imputation import ImputationStepResult
from microplex.run import RunResult, run_source_impute_stage, run_spec
from microplex.source_registry import SourceRegistry
from microplex.spec import (
    MicroplexSpec,
    VariableOperationKind,
    load_spec,
    load_spec_dict,
)
from microplex.stage_manifest import (
    StageManifest,
    build_stage_manifest,
    write_stage_manifest,
)

DEFAULT_DEMOGRAPHIC_COLUMNS: tuple[str, ...] = (
    "age",
    "filing_status",
    "ctc_qualifying_children",
)

DEFAULT_GEOGRAPHY_CONSTRAINT_COLUMNS: tuple[str, ...] = (
    "state_fips",
    "cbsa",
)
DEFAULT_SOURCE_IMPUTE_STEPS: tuple[str, ...] = (
    "scf_source_impute",
    "sipp_source_impute",
    "acs_source_impute",
)


@dataclass(frozen=True)
class AsecPufSupportSpineSmokeResult:
    """Result of the provider-backed ASEC+PUF support-spine smoke run."""

    spec: MicroplexSpec
    sources: Mapping[str, pd.DataFrame]
    run_result: RunResult
    diagnostics: Mapping[str, Any]
    source_impute_sources: Mapping[str, pd.DataFrame] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        """Return serializable smoke diagnostics."""
        return dict(self.diagnostics)


def write_asec_puf_support_spine_stage_artifacts(
    result: AsecPufSupportSpineSmokeResult,
    output_dir: Path,
) -> StageManifest:
    """Write a resumable stage checkpoint for an ASEC+PUF support-spine run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / "support_frame.parquet"
    diagnostics_path = output_dir / "diagnostics.json"
    manifest_path = output_dir / "stage_manifest.json"

    result.run_result.frame.to_parquet(frame_path, index=False)
    diagnostics_path.write_text(
        json.dumps(result.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = build_stage_manifest(
        stage_id="us_asec_puf_support_spine",
        root=output_dir,
        artifacts={
            "support_frame": frame_path.name,
            "diagnostics": diagnostics_path.name,
        },
        seeds={"support_partition": int(result.spec.spine.partition_seed)},
        parameters={
            "demographic_columns": list(result.diagnostics["demographic_columns"]),
            "geography_constraint_columns": list(
                result.diagnostics["geography_constraint_columns"]
            ),
        },
        metadata={
            "country": result.diagnostics["country"],
            "model_year": result.diagnostics["model_year"],
            "output_rows": result.diagnostics["output_rows"],
            "half_counts": result.diagnostics["half_counts"],
            "block_geography_assigned": result.diagnostics["block_geography"][
                "assigned"
            ],
            "source_imputation_enabled": result.diagnostics["source_imputation"][
                "enabled"
            ],
            "pending_stages": result.diagnostics["pending_stages"],
        },
    )
    write_stage_manifest(manifest_path, manifest)
    return manifest


def build_asec_puf_support_spine_spec(
    *,
    asec_year: int = 2025,
    calendar_year: int = 2024,
    puf_year: int = 2024,
    seed: int = 20260529,
    geography_constraint_columns: Sequence[str] = DEFAULT_GEOGRAPHY_CONSTRAINT_COLUMNS,
) -> MicroplexSpec:
    """Build the minimal ASEC+PUF source and support-spine spec."""
    return load_spec_dict(
        {
            "meta": {"country": "us", "model_year": calendar_year},
            "sources": {
                "cps_asec": {
                    "dataset": f"cps_asec_{asec_year}_calendar_{calendar_year}",
                    "role": "spine",
                    "entity": "tax_unit",
                },
                "puf": {
                    "dataset": f"puf_{puf_year}",
                    "role": "donor",
                    "entity": "tax_unit",
                },
            },
            "spine": {
                "base": "cps_asec",
                "method": "support_spine",
                "support": {"seed": seed},
                "halves": [
                    {"name": "cps_keep", "keep": "all"},
                    {
                        "name": "synthetic_puf",
                        "strip_to": [
                            "demographics",
                            *dict.fromkeys(geography_constraint_columns),
                        ],
                    },
                ],
            },
            "imputation": [],
        }
    )


def run_asec_puf_support_spine_smoke(
    *,
    registry: SourceRegistry | None = None,
    cache_dir: Path | None = None,
    cps_cache_dir: Path | None = None,
    puf_cache_dir: Path | None = None,
    puf_path: Path | None = None,
    puf_demographics_path: Path | None = None,
    download_cps: bool = True,
    asec_year: int = 2025,
    calendar_year: int = 2024,
    puf_year: int = 2024,
    seed: int = 20260529,
    max_cps_rows: int | None = None,
    max_puf_rows: int | None = None,
    block_crosswalk_path: Path | None = None,
    max_block_crosswalk_rows: int | None = None,
    source_impute_spec_path: Path | None = None,
    source_impute_manifest_path: Path | None = None,
    source_impute_storage_dir: Path | None = None,
    source_impute_blocks: Sequence[str] | None = None,
    source_impute_imputation_steps: Sequence[str] | None = DEFAULT_SOURCE_IMPUTE_STEPS,
    max_source_impute_rows: int | None = None,
    demographic_columns: Sequence[str] = DEFAULT_DEMOGRAPHIC_COLUMNS,
    geography_constraint_columns: Sequence[str] = DEFAULT_GEOGRAPHY_CONSTRAINT_COLUMNS,
) -> AsecPufSupportSpineSmokeResult:
    """Load ASEC+PUF providers and run the support-spine stage."""
    source_impute_spec: MicroplexSpec | None = None
    selected_source_impute_blocks = (
        tuple(source_impute_blocks) if source_impute_blocks is not None else None
    )
    effective_demographic_columns = tuple(demographic_columns)
    if source_impute_manifest_path is not None:
        source_impute_spec = load_spec(
            source_impute_spec_path or Path("packs/us/specs/us-2024.yaml")
        )
        source_impute_manifest = SourceImputeManifest.from_path(
            source_impute_manifest_path
        )
        source_impute_predictors = _source_impute_predictor_columns(
            source_impute_spec,
            source_impute_manifest,
            blocks=selected_source_impute_blocks,
            imputation_steps=source_impute_imputation_steps,
        )
        effective_demographic_columns = tuple(
            dict.fromkeys((*demographic_columns, *source_impute_predictors))
        )

    spec = build_asec_puf_support_spine_spec(
        asec_year=asec_year,
        calendar_year=calendar_year,
        puf_year=puf_year,
        seed=seed,
        geography_constraint_columns=geography_constraint_columns,
    )
    source_registry = registry or create_us_asec_puf_source_registry(
        asec_year=asec_year,
        calendar_year=calendar_year,
        puf_year=puf_year,
        cache_dir=cache_dir,
        cps_cache_dir=cps_cache_dir,
        puf_cache_dir=puf_cache_dir,
        puf_path=puf_path,
        puf_demographics_path=puf_demographics_path,
        download_cps=download_cps,
    )
    sources = source_registry.resolve_sources(spec)
    sources = {
        "cps_asec": _cap_rows(sources["cps_asec"], max_cps_rows, "max_cps_rows"),
        "puf": _cap_rows(sources["puf"], max_puf_rows, "max_puf_rows"),
    }
    _validate_source_surface(
        sources,
        effective_demographic_columns,
        geography_constraint_columns,
    )

    run_result = run_spec(
        spec,
        sources,
        demographic_columns=effective_demographic_columns,
        spine_keywords=(),
    )
    if block_crosswalk_path is not None:
        run_result = _assign_block_geography(
            run_result,
            block_crosswalk_path=block_crosswalk_path,
            max_block_crosswalk_rows=max_block_crosswalk_rows,
            seed=seed,
        )
    source_impute_sources: Mapping[str, pd.DataFrame] = {}
    source_impute_results: list[ImputationStepResult] = []
    if source_impute_manifest_path is not None:
        if "block_geoid" not in run_result.frame.columns:
            raise ValueError(
                "source-impute execution requires block geography assignment; "
                "supply block_crosswalk_path first"
            )
        assert source_impute_spec is not None
        register_us_declared_source_impute_blocks(
            source_registry,
            spec=source_impute_spec,
            manifest_path=source_impute_manifest_path,
            storage_dir=source_impute_storage_dir,
            max_rows=max_source_impute_rows,
            blocks=selected_source_impute_blocks,
        )
        source_impute_stage = run_source_impute_stage(
            run_result,
            source_impute_spec,
            source_registry,
            source_impute_manifest=source_impute_manifest_path,
            source_impute_blocks=selected_source_impute_blocks,
            source_impute_imputation_steps=source_impute_imputation_steps,
            demographic_columns=effective_demographic_columns,
            seed=seed,
        )
        run_result = source_impute_stage.run_result
        source_impute_sources = source_impute_stage.sources
        source_impute_results = source_impute_stage.imputation_results
    diagnostics = _diagnostics(
        spec=spec,
        sources=sources,
        run_result=run_result,
        source_impute_sources=source_impute_sources,
        source_impute_results=source_impute_results,
        demographic_columns=effective_demographic_columns,
        geography_constraint_columns=geography_constraint_columns,
    )
    _validate_run_result(diagnostics, sources=sources, run_result=run_result)
    return AsecPufSupportSpineSmokeResult(
        spec=spec,
        sources=sources,
        run_result=run_result,
        diagnostics=diagnostics,
        source_impute_sources=source_impute_sources,
    )


def _cap_rows(frame: pd.DataFrame, max_rows: int | None, name: str) -> pd.DataFrame:
    if max_rows is None:
        return frame.copy()
    if max_rows < 2:
        raise ValueError(f"{name} must be at least 2 when supplied")
    return frame.head(max_rows).copy()


def _source_impute_predictor_columns(
    spec: MicroplexSpec,
    manifest: SourceImputeManifest,
    *,
    blocks: Sequence[str] | None,
    imputation_steps: Sequence[str] | None,
) -> tuple[str, ...]:
    """Return selected source-impute predictors that must survive the split."""
    requested_steps = set(imputation_steps) if imputation_steps is not None else None
    if requested_steps == set():
        return ()

    selected_blocks = (
        tuple(manifest.block(block_name) for block_name in blocks)
        if blocks is not None
        else tuple(manifest.blocks.values())
    )
    selected_surveys = {block.survey_name for block in selected_blocks}
    predictors: list[str] = []
    unresolved: list[str] = []

    for variable_name, variable in spec.variables.items():
        operation = variable.mp_spec.operation if variable.mp_spec else None
        if operation is None or operation.kind is not VariableOperationKind.IMPUTE:
            continue
        if operation.imputation_step is None or operation.source is None:
            continue
        if (
            requested_steps is not None
            and operation.imputation_step not in requested_steps
        ):
            continue

        matching_blocks = [
            block
            for block in selected_blocks
            if operation.source == block.survey_name
            and variable_name in block.target_variables
        ]
        if len(matching_blocks) > 1:
            block_names = [block.name for block in matching_blocks]
            raise ValueError(
                "source-impute variable operation is ambiguous across selected "
                f"manifest blocks: {variable_name} appears in {block_names}"
            )
        if matching_blocks:
            predictors.extend(matching_blocks[0].predictors)
            continue

        if operation.source not in selected_surveys:
            if blocks is not None or requested_steps is None:
                continue
            unresolved.append(
                f"{variable_name} ({operation.imputation_step} from "
                f"{operation.source}: no manifest block)"
            )
            continue
        full_matches = [
            block
            for block in manifest.blocks.values()
            if operation.source == block.survey_name
            and variable_name in block.target_variables
        ]
        if blocks is not None and full_matches:
            continue
        unresolved.append(
            f"{variable_name} ({operation.imputation_step} from "
            f"{operation.source}: not a selected manifest target)"
        )

    if unresolved:
        raise ValueError(
            "source-impute variable operations are not backed by selected manifest "
            f"target variables: {unresolved}"
        )
    return tuple(dict.fromkeys(predictors))


def _validate_source_surface(
    sources: Mapping[str, pd.DataFrame],
    demographic_columns: Sequence[str],
    geography_constraint_columns: Sequence[str],
) -> None:
    for source_name in ("cps_asec", "puf"):
        if source_name not in sources:
            raise ValueError(f"missing loaded source frame: {source_name}")
        if sources[source_name].empty:
            raise ValueError(f"source frame {source_name} is empty")

    missing_shared = {
        name: sorted(set(SHARED_VARS) - set(frame.columns))
        for name, frame in sources.items()
    }
    missing = {name: cols for name, cols in missing_shared.items() if cols}
    if missing:
        raise ValueError(f"ASEC+PUF shared-variable surface is incomplete: {missing}")

    missing_demographics = sorted(set(demographic_columns) - set(sources["cps_asec"]))
    if missing_demographics:
        raise ValueError(
            "CPS ASEC tax-unit spine is missing demographic columns: "
            f"{missing_demographics}"
        )

    _validate_geography_constraint_values(
        sources["cps_asec"],
        geography_constraint_columns,
        surface_name="CPS ASEC tax-unit spine",
    )


def _diagnostics(
    *,
    spec: MicroplexSpec,
    sources: Mapping[str, pd.DataFrame],
    run_result: RunResult,
    source_impute_sources: Mapping[str, pd.DataFrame],
    source_impute_results: Sequence[ImputationStepResult],
    demographic_columns: Sequence[str],
    geography_constraint_columns: Sequence[str],
) -> dict[str, Any]:
    label = run_result.spine.half_label_column
    frame = run_result.frame
    half_counts = {
        str(name): int(count)
        for name, count in frame[label].value_counts(sort=False).to_dict().items()
    }
    synthetic = frame.loc[frame[label] == "synthetic_puf"]
    keep = frame.loc[frame[label] == "cps_keep"]
    return {
        "country": spec.meta.country,
        "model_year": spec.meta.model_year,
        "support_partition_seed": spec.spine.partition_seed,
        "source_rows": {name: int(len(frame)) for name, frame in sources.items()},
        "source_weight_sums": {
            "cps_asec_household_weight": _sum_if_present(
                sources["cps_asec"], "household_weight"
            ),
            "puf_weight": _sum_if_present(sources["puf"], "weight"),
        },
        "output_rows": int(len(frame)),
        "half_counts": half_counts,
        "demographic_columns": list(demographic_columns),
        "geography_constraint_columns": list(geography_constraint_columns),
        "block_geography": _block_geography_diagnostics(frame),
        "source_imputation": _source_imputation_diagnostics(
            source_impute_sources,
            source_impute_results,
        ),
        "shared_variables": list(SHARED_VARS),
        "shared_missing": {
            name: sorted(set(SHARED_VARS) - set(source_frame.columns))
            for name, source_frame in sources.items()
        },
        "cps_keep_household_weight_sum": _sum_if_present(keep, "household_weight"),
        "synthetic_puf_household_weight_sum": _sum_if_present(
            synthetic, "household_weight"
        ),
        "pending_stages": list(run_result.pending_stages),
    }


def _sum_if_present(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    return float(frame[column].sum())


def _source_imputation_diagnostics(
    source_impute_sources: Mapping[str, pd.DataFrame],
    source_impute_results: Sequence[ImputationStepResult],
) -> dict[str, Any]:
    return {
        "enabled": bool(source_impute_results),
        "source_rows": {
            name: int(len(frame)) for name, frame in source_impute_sources.items()
        },
        "results": [
            {
                "onto": result.onto,
                "donor": result.donor,
                "imputed": list(result.imputed),
                "skipped_passthrough": list(result.skipped_passthrough),
                "skipped_missing_in_donor": list(result.skipped_missing_in_donor),
            }
            for result in source_impute_results
        ],
    }


def _validate_geography_constraint_values(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    surface_name: str,
) -> None:
    if not columns:
        return
    missing_columns = sorted(set(columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"{surface_name} is missing geography constraint columns: {missing_columns}"
        )
    has_constraint = frame.loc[:, list(columns)].notna().any(axis=1)
    if not has_constraint.all():
        missing_count = int((~has_constraint).sum())
        raise ValueError(
            f"{surface_name} has {missing_count} rows with no geography "
            f"constraint values across {list(columns)}"
        )


def _validate_run_result(
    diagnostics: Mapping[str, Any],
    *,
    sources: Mapping[str, pd.DataFrame],
    run_result: RunResult,
) -> None:
    if diagnostics["output_rows"] != len(sources["cps_asec"]):
        raise ValueError(
            "support-spine output row count must match the CPS support universe"
        )
    half_counts = diagnostics["half_counts"]
    if set(half_counts) != {"cps_keep", "synthetic_puf"}:
        raise ValueError(f"unexpected support-spine halves: {half_counts}")
    if sum(half_counts.values()) != len(sources["cps_asec"]):
        raise ValueError("support-spine half counts do not sum to CPS row count")
    frame = run_result.frame
    label = run_result.spine.half_label_column
    synthetic = frame.loc[frame[label] == "synthetic_puf"]
    if (
        "household_weight" in synthetic
        and not (synthetic["household_weight"] == 0).all()
    ):
        raise ValueError("synthetic_puf household weights must initialize to zero")
    _validate_geography_constraint_values(
        synthetic,
        tuple(diagnostics["geography_constraint_columns"]),
        surface_name="synthetic_puf support-spine half",
    )
    if diagnostics["block_geography"]["assigned"]:
        if frame["block_geoid"].isna().any():
            raise ValueError("assigned block_geoid contains missing values")
        if not frame["block_geoid"].astype("string").str.len().eq(15).all():
            raise ValueError("assigned block_geoid values must be 15 characters")
    source_imputation = diagnostics["source_imputation"]
    if source_imputation["enabled"]:
        for result in source_imputation["results"]:
            for variable in result["imputed"]:
                if variable not in frame.columns:
                    raise ValueError(
                        f"source-imputed variable {variable!r} missing from output"
                    )
                if frame[variable].isna().any():
                    raise ValueError(
                        f"source-imputed variable {variable!r} contains missing values"
                    )


def _assign_block_geography(
    run_result: RunResult,
    *,
    block_crosswalk_path: Path,
    max_block_crosswalk_rows: int | None,
    seed: int,
) -> RunResult:
    frame = run_result.frame
    if "state_fips" not in frame.columns:
        raise ValueError("block assignment requires state_fips in the support frame")
    states = frame["state_fips"].dropna().map(normalize_us_state_fips).drop_duplicates()
    if states.empty:
        raise ValueError("block assignment requires at least one state_fips value")
    provider = CensusBlockCrosswalkProvider(
        path=block_crosswalk_path,
        state_fips=tuple(states),
        max_rows=max_block_crosswalk_rows,
    )
    crosswalk = provider.load_crosswalk()
    partition_columns = tuple(
        column
        for column in ("tract_geoid", "county_fips", "cbsa", "state_fips")
        if column in frame.columns and column in crosswalk.data.columns
    )
    if not partition_columns:
        raise ValueError(
            "No shared geography columns between support frame and block crosswalk"
        )
    assigner = LowestAvailableAtomicGeographyAssigner(
        crosswalk=crosswalk,
        plan=LowestAvailableGeographyAssignmentPlan(
            partition_columns=partition_columns,
            atomic_id_column="block_geoid",
            geography_columns=("county_fips", "tract_geoid"),
            partition_normalizers={
                "cbsa": normalize_string_code,
                "state_fips": normalize_us_state_fips,
            },
        ),
    )
    assigned_frame = assigner.assign(frame, random_state=seed)
    label = run_result.spine.half_label_column
    halves = {
        name: assigned_frame.loc[assigned_frame[label] == name].copy()
        for name in run_result.halves
    }
    return replace(run_result, frame=assigned_frame, halves=halves)


def _block_geography_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    if "block_geoid" not in frame.columns:
        return {"assigned": False}
    partition_column = "_geography_partition_column"
    return {
        "assigned": True,
        "assigned_rows": int(frame["block_geoid"].notna().sum()),
        "unique_block_geoids": int(frame["block_geoid"].nunique(dropna=True)),
        "columns": [
            column
            for column in ("block_geoid", "state_fips", "county_fips", "tract_geoid")
            if column in frame.columns
        ],
        "partition_counts": (
            {
                str(name): int(count)
                for name, count in frame[partition_column]
                .value_counts(sort=False)
                .to_dict()
                .items()
            }
            if partition_column in frame.columns
            else {}
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the provider-backed US ASEC+PUF support-spine smoke check."
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--cps-cache-dir", type=Path, default=None)
    parser.add_argument("--puf-cache-dir", type=Path, default=None)
    parser.add_argument("--puf-path", type=Path, default=None)
    parser.add_argument("--puf-demographics-path", type=Path, default=None)
    parser.add_argument("--block-crosswalk-path", type=Path, default=None)
    parser.add_argument("--source-impute-spec", type=Path, default=None)
    parser.add_argument("--source-impute-manifest", type=Path, default=None)
    parser.add_argument("--source-impute-storage-dir", type=Path, default=None)
    parser.add_argument("--source-impute-block", action="append", default=None)
    parser.add_argument("--max-cps-rows", type=int, default=None)
    parser.add_argument("--max-puf-rows", type=int, default=None)
    parser.add_argument("--max-block-crosswalk-rows", type=int, default=None)
    parser.add_argument("--max-source-impute-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--no-download-cps", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the ASEC+PUF support-spine smoke run."""
    args = _build_parser().parse_args(argv)
    result = run_asec_puf_support_spine_smoke(
        cache_dir=args.cache_dir,
        cps_cache_dir=args.cps_cache_dir,
        puf_cache_dir=args.puf_cache_dir,
        puf_path=args.puf_path,
        puf_demographics_path=args.puf_demographics_path,
        download_cps=not args.no_download_cps,
        max_cps_rows=args.max_cps_rows,
        max_puf_rows=args.max_puf_rows,
        block_crosswalk_path=args.block_crosswalk_path,
        max_block_crosswalk_rows=args.max_block_crosswalk_rows,
        source_impute_spec_path=args.source_impute_spec,
        source_impute_manifest_path=args.source_impute_manifest,
        source_impute_storage_dir=args.source_impute_storage_dir,
        source_impute_blocks=tuple(args.source_impute_block)
        if args.source_impute_block
        else None,
        max_source_impute_rows=args.max_source_impute_rows,
        seed=args.seed,
    )
    if args.output_dir is not None:
        write_asec_puf_support_spine_stage_artifacts(result, args.output_dir)
    payload = result.to_json_dict()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
