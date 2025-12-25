"""
microplex: Microdata synthesis and reweighting using normalizing flows.

A library for creating rich, calibrated microdata through:
- Conditional synthesis (demographics → outcomes)
- Reweighting to population targets
- Zero-inflated distributions (common in economic/health data)
- Joint correlations between variables
- Hierarchical structures (households, firms, etc.)

Example:
    >>> from microplex import Synthesizer
    >>> synth = Synthesizer(
    ...     target_vars=["income", "expenditure"],
    ...     condition_vars=["age", "education"],
    ... )
    >>> synth.fit(training_data)
    >>> synthetic = synth.generate(new_demographics)
"""

from microplex.synthesizer import Synthesizer
from microplex.transforms import (
    ZeroInflatedTransform,
    LogTransform,
    Standardizer,
    VariableTransformer,
    MultiVariableTransformer,
)
from microplex.flows import ConditionalMAF, MADE, AffineCouplingLayer
from microplex.discrete import (
    BinaryModel,
    CategoricalModel,
    DiscreteModelCollection,
)

__version__ = "0.1.0"

__all__ = [
    # Main class
    "Synthesizer",
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
]
