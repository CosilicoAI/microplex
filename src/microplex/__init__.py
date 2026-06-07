"""
microplex: Microdata synthesis and reweighting using normalizing flows.

A country-agnostic library for creating rich, calibrated microdata through:
- Conditional synthesis (demographics → outcomes)
- Reweighting to population targets
- Zero-inflated distributions (common in economic/health data)
- Joint correlations between variables
- Hierarchical structures (households, firms, etc.)
- Longitudinal / panel synthesis with trajectory models

Country-specific primitives (CPS, PUF, SOI, SSA life tables, census GEOIDs,
PolicyEngine-US parity helpers) live in country-pack packages such as
`microplex-us` and are not re-exported here.

Example:
    >>> from microplex import Synthesizer
    >>> synth = Synthesizer(
    ...     target_vars=["income", "expenditure"],
    ...     condition_vars=["age", "education"],
    ... )
    >>> synth.fit(training_data)
    >>> synthetic = synth.generate(new_demographics)
"""

from importlib import import_module

__version__ = "0.2.0"

_LAZY_EXPORTS = {
    "Synthesizer": "microplex.synthesizer",
    "HierarchicalSynthesizer": "microplex.hierarchical",
    "HouseholdSchema": "microplex.hierarchical",
    "PreservedTaxUnitTables": "microplex.tax_units",
    "build_preserved_tax_unit_tables": "microplex.tax_units",
    "Reweighter": "microplex.reweighting",
    "Calibrator": "microplex.calibration",
    "SparseCalibrator": "microplex.calibration",
    "HardConcreteCalibrator": "microplex.calibration",
    "StatMatchSynthesizer": "microplex.statmatch_backend",
    "create_synthesizer": "microplex.statmatch_backend",
    "HAS_STATMATCH": "microplex.statmatch_backend",
    "ZeroInflatedTransform": "microplex.transforms",
    "LogTransform": "microplex.transforms",
    "Standardizer": "microplex.transforms",
    "VariableTransformer": "microplex.transforms",
    "MultiVariableTransformer": "microplex.transforms",
    "ConditionalMAF": "microplex.flows",
    "MADE": "microplex.flows",
    "AffineCouplingLayer": "microplex.flows",
    "BinaryModel": "microplex.discrete",
    "CategoricalModel": "microplex.discrete",
    "DiscreteModelCollection": "microplex.discrete",
    "Mortality": "microplex.transitions",
    "DisabilityOnset": "microplex.transitions",
    "DisabilityRecovery": "microplex.transitions",
    "DisabilityTransitionModel": "microplex.transitions",
    "MarriageTransition": "microplex.transitions",
    "DivorceTransition": "microplex.transitions",
    "EntityType": "microplex.core",
    "FilingStatus": "microplex.core",
    "RecordType": "microplex.core",
    "Entity": "microplex.core",
    "Person": "microplex.core",
    "TaxUnit": "microplex.core",
    "Household": "microplex.core",
    "Family": "microplex.core",
    "SPMUnit": "microplex.core",
    "Record": "microplex.core",
    "DataType": "microplex.core",
    "VariableRole": "microplex.core",
    "LegalReference": "microplex.core",
    "Variable": "microplex.core",
    "VariableRegistry": "microplex.core",
    "PeriodType": "microplex.core",
    "Period": "microplex.core",
    "ResolutionLevel": "microplex.core",
    "ResolutionConfig": "microplex.core",
    "HardConcreteGate": "microplex.core",
    "compress_dataset": "microplex.core",
    "for_browser": "microplex.core",
    "for_api": "microplex.core",
    "for_research": "microplex.core",
    "MaskedMAF": "microplex.fusion",
    "FusionConfig": "microplex.fusion",
    "FusionResult": "microplex.fusion",
    "FusionSynthesizer": "microplex.fusion",
    "synthesize_from_surveys": "microplex.fusion",
    "PopulationDGP": "microplex.dgp",
    "Survey": "microplex.dgp",
    "EvalResult": "microplex.dgp",
    "run_multi_source_benchmark": "microplex.dgp",
}


def __getattr__(name: str):
    """Lazily import heavyweight public exports on first access."""
    if name == "DefaultSparseCalibrator":
        module = import_module("microplex.calibration")
        value = module.SparseCalibrator
    elif module_name := _LAZY_EXPORTS.get(name):
        module = import_module(module_name)
        value = getattr(module, name)
    else:
        raise AttributeError(f"module 'microplex' has no attribute {name!r}")
    globals()[name] = value
    return value

__all__ = [
    # Core synthesis
    "Synthesizer",
    "HierarchicalSynthesizer",
    "HouseholdSchema",
    "PreservedTaxUnitTables",
    "build_preserved_tax_unit_tables",
    # Calibration
    "Reweighter",
    "Calibrator",
    "SparseCalibrator",
    "HardConcreteCalibrator",
    "DefaultSparseCalibrator",
    # Statistical matching (optional backend)
    "StatMatchSynthesizer",
    "create_synthesizer",
    "HAS_STATMATCH",
    # Transforms
    "ZeroInflatedTransform",
    "LogTransform",
    "Standardizer",
    "VariableTransformer",
    "MultiVariableTransformer",
    # Flows
    "ConditionalMAF",
    "MADE",
    "AffineCouplingLayer",
    # Discrete
    "BinaryModel",
    "CategoricalModel",
    "DiscreteModelCollection",
    # Transitions
    "Mortality",
    "DisabilityOnset",
    "DisabilityRecovery",
    "DisabilityTransitionModel",
    "MarriageTransition",
    "DivorceTransition",
    # Core entities
    "EntityType",
    "FilingStatus",
    "RecordType",
    "Entity",
    "Person",
    "TaxUnit",
    "Household",
    "Family",
    "SPMUnit",
    "Record",
    # Variables
    "DataType",
    "VariableRole",
    "LegalReference",
    "Variable",
    "VariableRegistry",
    # Periods
    "PeriodType",
    "Period",
    # Resolution
    "ResolutionLevel",
    "ResolutionConfig",
    "HardConcreteGate",
    "compress_dataset",
    "for_browser",
    "for_api",
    "for_research",
    # Fusion (multi-survey synthesis)
    "MaskedMAF",
    "FusionConfig",
    "FusionResult",
    "FusionSynthesizer",
    "synthesize_from_surveys",
    # Multi-source DGP
    "PopulationDGP",
    "Survey",
    "EvalResult",
    "run_multi_source_benchmark",
]
