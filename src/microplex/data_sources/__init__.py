"""
Data sources for microplex.

This module provides loaders for various microdata sources:
- CPS ASEC (Census Bureau's primary income/poverty survey)
- PSID (Panel Study of Income Dynamics - longitudinal household survey)
- PUF (Public Use File - tax return data)
- CPS to PolicyEngine variable mappings with legal references
- Data transformation utilities
"""

from microplex.data_sources.census_blocks import (
    CensusBlockCrosswalkProvider,
    load_census_block_crosswalk,
    prepare_census_block_crosswalk,
)
from microplex.data_sources.cps import (
    HOUSEHOLD_VARIABLES,
    PERSON_VARIABLES,
    CPSAsecSourceProvider,
    CPSDataset,
    download_cps_asec,
    get_available_years,
)
from microplex.data_sources.cps import (
    load_cps_asec as load_cps_asec_polars,
)
from microplex.data_sources.cps_mappings import (
    CoverageGap,
    CoverageLevel,
    VariableMapping,
    coverage_summary,
    get_all_mappings,
    get_mapping_metadata,
    map_age,
    map_agi_proxy,
    map_ctc_qualifying_children,
    map_earned_income,
    map_filing_status,
    map_household_size,
    map_is_blind,
    map_is_dependent,
)
from microplex.data_sources.cps_transform import (
    TransformedDataset,
    transform_cps_to_policyengine,
)
from microplex.data_sources.psid import (
    PSID_TO_MICROPLEX_VARS,
    PSIDDataset,
    calibrate_divorce_rates,
    calibrate_marriage_rates,
    create_psid_fusion_source,
    extract_transition_rates,
    get_age_specific_rates,
    load_psid_panel,
)
from microplex.data_sources.puf import (
    PUF_EXCLUSIVE_VARS,
    PUF_VARIABLE_MAP,
    SHARED_VARS,
    UPRATING_FACTORS,
    PUFSourceProvider,
    download_puf,
    expand_to_persons,
    load_puf,
    map_puf_variables,
    uprate_puf,
)
from microplex.data_sources.source_impute import (
    ManifestSourceImputeProvider,
    SourceImputeBlock,
    SourceImputeManifest,
    compile_source_impute_steps_from_manifest,
    load_source_impute_block_table,
    validate_source_impute_block_supported,
)
from microplex.data_sources.us_registry import (
    create_us_asec_puf_source_registry,
    register_us_source_impute_blocks,
)

__all__ = [
    # CPS loading
    "CPSDataset",
    "CPSAsecSourceProvider",
    "download_cps_asec",
    "load_cps_asec_polars",
    "get_available_years",
    "PERSON_VARIABLES",
    "HOUSEHOLD_VARIABLES",
    # Mappings
    "CoverageLevel",
    "CoverageGap",
    "VariableMapping",
    "map_age",
    "map_earned_income",
    "map_filing_status",
    "map_is_blind",
    "map_is_dependent",
    "map_ctc_qualifying_children",
    "map_agi_proxy",
    "map_household_size",
    "get_mapping_metadata",
    "get_all_mappings",
    "coverage_summary",
    "CensusBlockCrosswalkProvider",
    "load_census_block_crosswalk",
    "prepare_census_block_crosswalk",
    # Transform
    "TransformedDataset",
    "transform_cps_to_policyengine",
    # PUF loading
    "PUFSourceProvider",
    "load_puf",
    "download_puf",
    "map_puf_variables",
    "uprate_puf",
    "expand_to_persons",
    "PUF_VARIABLE_MAP",
    "UPRATING_FACTORS",
    "PUF_EXCLUSIVE_VARS",
    "SHARED_VARS",
    # Source-impute manifests
    "ManifestSourceImputeProvider",
    "SourceImputeBlock",
    "SourceImputeManifest",
    "compile_source_impute_steps_from_manifest",
    "load_source_impute_block_table",
    "validate_source_impute_block_supported",
    "create_us_asec_puf_source_registry",
    "register_us_source_impute_blocks",
    # PSID loading
    "PSIDDataset",
    "load_psid_panel",
    "extract_transition_rates",
    "get_age_specific_rates",
    "calibrate_marriage_rates",
    "calibrate_divorce_rates",
    "create_psid_fusion_source",
    "PSID_TO_MICROPLEX_VARS",
]
