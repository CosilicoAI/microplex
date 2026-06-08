"""US source-provider registry helpers for content-only country specs."""

from __future__ import annotations

from pathlib import Path

from microplex.core import EntityType
from microplex.data_sources.cps import CPSAsecSourceProvider
from microplex.data_sources.puf import PUFSourceProvider
from microplex.source_registry import SourceRegistry


def create_us_asec_puf_source_registry(
    *,
    asec_year: int = 2025,
    calendar_year: int = 2024,
    puf_year: int = 2024,
    cache_dir: Path | None = None,
    cps_cache_dir: Path | None = None,
    puf_cache_dir: Path | None = None,
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
            ),
            default_entity=EntityType.TAX_UNIT,
        )
    )
