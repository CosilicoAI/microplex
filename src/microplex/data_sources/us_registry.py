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
from microplex.spec import MicroplexSpec


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
    Dataset ids follow the manifest ``dataset_id`` when present, otherwise
    the existing spec convention, e.g. ``scf_2022``.
    """
    manifest = SourceImputeManifest.from_path(manifest_path)
    source_blocks = [manifest.block(block_name) for block_name in blocks]
    for block in source_blocks:
        validate_source_impute_block_supported(block)

    datasets: dict[str, list[SourceImputeBlock]] = {}
    seen_dataset_ids: set[str] = set()
    seen_block_names: set[str] = set()
    for block in source_blocks:
        if block.name in seen_block_names:
            raise ValueError(f"Duplicate source-impute block requested: {block.name!r}")
        seen_block_names.add(block.name)
        dataset_id = block.dataset_id or f"{block.survey_name}_{block.default_year}"
        if dataset_id not in seen_dataset_ids:
            seen_dataset_ids.add(dataset_id)
            try:
                registry.provider_for(dataset_id)
            except KeyError:
                pass
            else:
                raise ValueError(
                    f"SourceRegistry already has a provider for {dataset_id!r}"
                )
        datasets.setdefault(dataset_id, []).append(block)

    for dataset_id, blocks_for_dataset in datasets.items():
        first_block = blocks_for_dataset[0]
        registry.register(
            dataset_id,
            ManifestSourceImputeProvider(
                manifest_path=manifest_path,
                block_name=first_block.name,
                block_names=tuple(block.name for block in blocks_for_dataset),
                storage_dir=storage_dir,
                max_rows=max_rows,
                source_name=first_block.survey_name,
            ),
            default_entity=EntityType.PERSON,
        )
    return registry


def register_us_declared_source_impute_blocks(
    registry: SourceRegistry,
    *,
    spec: MicroplexSpec,
    manifest_path: str | Path,
    storage_dir: str | Path | None = None,
    max_rows: int | None = None,
    blocks: tuple[str, ...] | None = None,
) -> SourceRegistry:
    """Register source-impute donor providers declared by a US content spec.

    ``run_spec`` resolves every source named by the spec before executing its
    stage graph. The US content spec declares SCF/SIPP/ACS as donor sources, but
    the executable loaders live in Microplex. This helper bridges those pieces:
    it reads the content-pack source-impute manifest, selects the blocks whose
    ``survey_name`` appears in ``spec.sources``, asserts that their dataset id
    matches the spec declaration, and delegates to
    :func:`register_us_source_impute_blocks`.

    Args:
        registry: Registry to extend.
        spec: Validated US content spec.
        manifest_path: Path to ``pe_source_impute_blocks.json``.
        storage_dir: Optional directory containing source-impute input files.
        max_rows: Optional row cap for smoke-scale providers.
        blocks: Optional explicit manifest block filter. When supplied, every
            requested block must correspond to a source declared by ``spec``.

    Returns:
        The same ``registry`` instance, extended in place.

    Raises:
        ValueError: if a requested block is absent from the spec sources or its
            manifest dataset id disagrees with the spec declaration.
    """
    manifest = SourceImputeManifest.from_path(manifest_path)
    selected_block_names = tuple(manifest.blocks) if blocks is None else blocks
    declared_block_names: list[str] = []
    missing_sources: list[str] = []
    dataset_mismatches: list[str] = []

    for block_name in selected_block_names:
        block = manifest.block(block_name)
        source_spec = spec.sources.get(block.survey_name)
        if source_spec is None:
            if blocks is not None:
                missing_sources.append(f"{block_name}:{block.survey_name}")
            continue
        expected_dataset = (
            block.dataset_id or f"{block.survey_name}_{block.default_year}"
        )
        if source_spec.dataset != expected_dataset:
            dataset_mismatches.append(
                f"{block_name}:{block.survey_name} manifest={expected_dataset} "
                f"spec={source_spec.dataset}"
            )
            continue
        declared_block_names.append(block_name)

    if missing_sources:
        raise ValueError(
            "source-impute block(s) are not declared as spec sources: "
            f"{missing_sources}"
        )
    if dataset_mismatches:
        raise ValueError(
            "source-impute block dataset id does not match spec source dataset: "
            f"{dataset_mismatches}"
        )
    if not declared_block_names:
        return registry

    return register_us_source_impute_blocks(
        registry,
        manifest_path=manifest_path,
        storage_dir=storage_dir,
        max_rows=max_rows,
        blocks=tuple(declared_block_names),
    )
