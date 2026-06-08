"""US source-provider registry helpers for content-only country specs."""

from __future__ import annotations

from pathlib import Path

from microplex.core import EntityType
from microplex.data_sources.cps import CPSAsecSourceProvider
from microplex.data_sources.puf import PUFSourceProvider
from microplex.data_sources.source_impute import (
    ManifestSourceImputeProvider,
    SourceImputeBlock,
    SourceImputeManifest,
    validate_source_impute_block_supported,
)
from microplex.source_registry import SourceRegistry


def create_us_asec_puf_source_registry(
    *,
    asec_year: int = 2025,
    calendar_year: int = 2024,
    puf_year: int = 2024,
    cache_dir: Path | None = None,
    cps_cache_dir: Path | None = None,
    puf_cache_dir: Path | None = None,
    puf_path: Path | None = None,
    puf_demographics_path: Path | None = None,
    download_cps: bool = True,
) -> SourceRegistry:
    """Create the first-principles ASEC+PUF source registry.

    The registry intentionally covers only the first critical data step:
    ASEC/CPS as the support spine and PUF as the first tax donor. SCF/SIPP/ACS
    providers should be registered in later slices after the ASEC+PUF flow is
    exercised on real data.
    """
    cps_dataset = f"cps_asec_{asec_year}_calendar_{calendar_year}"
    puf_dataset = f"puf_{puf_year}"
    return (
        SourceRegistry()
        .register(
            cps_dataset,
            CPSAsecSourceProvider(
                asec_year=asec_year,
                calendar_year=calendar_year,
                cache_dir=cps_cache_dir or cache_dir,
                download=download_cps,
            ),
            default_entity=EntityType.TAX_UNIT,
        )
        .register(
            puf_dataset,
            PUFSourceProvider(
                target_year=puf_year,
                cache_dir=puf_cache_dir or cache_dir,
                puf_path=puf_path,
                demographics_path=puf_demographics_path,
            ),
            default_entity=EntityType.TAX_UNIT,
        )
    )


def register_us_source_impute_blocks(
    registry: SourceRegistry,
    *,
    manifest_path: str | Path,
    storage_dir: str | Path | None = None,
    max_rows: int | None = None,
    blocks: tuple[str, ...] = ("scf",),
) -> SourceRegistry:
    """Register manifest-backed US source-imputation donor providers.

    The content package declares source-imputation blocks in JSON; this helper
    keeps executable loading in Microplex while preserving a Python-free pack.
    Dataset ids follow the existing spec convention, e.g. ``scf_2022``.
    """
    manifest = SourceImputeManifest.from_path(manifest_path)
    source_blocks = [manifest.block(block_name) for block_name in blocks]
    for block in source_blocks:
        validate_source_impute_block_supported(block)

    datasets: list[tuple[str, SourceImputeBlock]] = []
    seen_dataset_ids: set[str] = set()
    for block in source_blocks:
        dataset_id = f"{block.survey_name}_{block.default_year}"
        if dataset_id in seen_dataset_ids:
            raise ValueError(
                f"Duplicate source-impute dataset requested: {dataset_id!r}"
            )
        seen_dataset_ids.add(dataset_id)
        try:
            registry.provider_for(dataset_id)
        except KeyError:
            pass
        else:
            raise ValueError(
                f"SourceRegistry already has a provider for {dataset_id!r}"
            )
        datasets.append((dataset_id, block))

    for dataset_id, block in datasets:
        registry.register(
            dataset_id,
            ManifestSourceImputeProvider(
                manifest_path=manifest_path,
                block_name=block.name,
                storage_dir=storage_dir,
                max_rows=max_rows,
                source_name=block.survey_name,
            ),
            default_entity=EntityType.PERSON,
        )
    return registry
