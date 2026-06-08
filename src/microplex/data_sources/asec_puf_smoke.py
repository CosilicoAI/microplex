"""ASEC+PUF support-spine smoke run for the first real-data engine slice."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from microplex.data_sources.puf import SHARED_VARS
from microplex.data_sources.us_registry import create_us_asec_puf_source_registry
from microplex.run import RunResult, run_spec
from microplex.source_registry import SourceRegistry
from microplex.spec import MicroplexSpec, load_spec_dict

DEFAULT_DEMOGRAPHIC_COLUMNS: tuple[str, ...] = (
    "age",
    "filing_status",
    "ctc_qualifying_children",
)


@dataclass(frozen=True)
class AsecPufSupportSpineSmokeResult:
    """Result of the provider-backed ASEC+PUF support-spine smoke run."""

    spec: MicroplexSpec
    sources: Mapping[str, pd.DataFrame]
    run_result: RunResult
    diagnostics: Mapping[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        """Return serializable smoke diagnostics."""
        return dict(self.diagnostics)


def build_asec_puf_support_spine_spec(
    *,
    asec_year: int = 2025,
    calendar_year: int = 2024,
    puf_year: int = 2024,
    seed: int = 20260529,
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
                    {"name": "synthetic_puf", "strip_to": ["demographics"]},
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
    demographic_columns: Sequence[str] = DEFAULT_DEMOGRAPHIC_COLUMNS,
) -> AsecPufSupportSpineSmokeResult:
    """Load ASEC+PUF providers and run the support-spine stage."""
    spec = build_asec_puf_support_spine_spec(
        asec_year=asec_year,
        calendar_year=calendar_year,
        puf_year=puf_year,
        seed=seed,
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
    _validate_source_surface(sources, demographic_columns)

    run_result = run_spec(
        spec,
        sources,
        demographic_columns=demographic_columns,
        spine_keywords=(),
    )
    diagnostics = _diagnostics(
        spec=spec,
        sources=sources,
        run_result=run_result,
        demographic_columns=demographic_columns,
    )
    _validate_run_result(diagnostics, sources=sources, run_result=run_result)
    return AsecPufSupportSpineSmokeResult(
        spec=spec,
        sources=sources,
        run_result=run_result,
        diagnostics=diagnostics,
    )


def _cap_rows(frame: pd.DataFrame, max_rows: int | None, name: str) -> pd.DataFrame:
    if max_rows is None:
        return frame.copy()
    if max_rows < 2:
        raise ValueError(f"{name} must be at least 2 when supplied")
    return frame.head(max_rows).copy()


def _validate_source_surface(
    sources: Mapping[str, pd.DataFrame],
    demographic_columns: Sequence[str],
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


def _diagnostics(
    *,
    spec: MicroplexSpec,
    sources: Mapping[str, pd.DataFrame],
    run_result: RunResult,
    demographic_columns: Sequence[str],
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
    if "household_weight" in synthetic and not (synthetic["household_weight"] == 0).all():
        raise ValueError("synthetic_puf household weights must initialize to zero")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the provider-backed US ASEC+PUF support-spine smoke check."
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--cps-cache-dir", type=Path, default=None)
    parser.add_argument("--puf-cache-dir", type=Path, default=None)
    parser.add_argument("--puf-path", type=Path, default=None)
    parser.add_argument("--puf-demographics-path", type=Path, default=None)
    parser.add_argument("--max-cps-rows", type=int, default=None)
    parser.add_argument("--max-puf-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--no-download-cps", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
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
        seed=args.seed,
    )
    payload = result.to_json_dict()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
