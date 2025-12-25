# microplex

**Microdata synthesis and reweighting using normalizing flows.**

`microplex` creates rich, calibrated microdata by combining synthesis and reweighting.

## Key Features

- **Conditional synthesis**: Generate target variables given demographics
- **Sparse reweighting**: L0/L1 optimization to match population targets
- **Multi-source fusion**: Combine CPS, ACS, admin data into one population
- **Zero-inflation handling**: Built-in support for variables with many zeros
- **Scalable**: Synthesize billions of records, reweight to any geography

## Installation

```bash
pip install microplex
```

## Quick Example

```python
from microplex import Synthesizer

# Initialize
synth = Synthesizer(
    target_vars=["income", "expenditure"],
    condition_vars=["age", "education", "region"],
)

# Fit on training data
synth.fit(training_data, weight_col="weight")

# Generate for new demographics
synthetic = synth.generate(new_demographics)
```

## Use Cases

| Use Case | Description |
|----------|-------------|
| Survey enhancement | Impute income variables from tax data onto census |
| Small area estimation | Reweight synthetic population to county/tract targets |
| Privacy synthesis | Generate synthetic data for public release |
| Data fusion | Combine variables from CPS, ACS, SIPP, admin data |

## The microplex Workflow

```
                         ┌─────────────────────────────────────┐
                         │           DATA SOURCES              │
                         ├─────────┬─────────┬─────────────────┤
                         │   CPS   │   ACS   │   Admin Data    │
                         │ income  │  geo    │   validation    │
                         │  tax    │ housing │    targets      │
                         └────┬────┴────┬────┴────────┬────────┘
                              │         │             │
                              ▼         ▼             │
                    ┌─────────────────────────┐       │
                    │    CONDITIONAL MAF      │       │
                    │  P(targets | context)   │       │
                    │                         │       │
                    │  • Zero-inflation       │       │
                    │  • Joint correlations   │       │
                    │  • Hierarchical         │       │
                    └───────────┬─────────────┘       │
                                │                     │
                                ▼                     │
                    ┌─────────────────────────┐       │
                    │    SYNTHESIZE           │       │
                    │    BILLIONS OF          │       │
                    │    HOUSEHOLDS           │       │
                    └───────────┬─────────────┘       │
                                │                     │
                                ▼                     ▼
                    ┌─────────────────────────────────────────┐
                    │         SPARSE REWEIGHTING              │
                    │                                         │
                    │   min ||w||₀  s.t.  Σ wᵢxᵢ = targets   │
                    │                                         │
                    │   • Match population margins            │
                    │   • Any geography (state/county/tract)  │
                    │   • Minimal record subset               │
                    └───────────────────┬─────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │         CALIBRATED MICRODATA            │
                    │                                         │
                    │   Rich population with:                 │
                    │   • All variables from all sources      │
                    │   • Matches official statistics         │
                    │   • Any geographic granularity          │
                    └─────────────────────────────────────────┘
```

## Comparison to Alternatives

| Feature | microplex | CT-GAN | TVAE | synthpop |
|---------|:---------:|:------:|:----:|:--------:|
| Conditional generation | ✅ | ❌ | ❌ | ❌ |
| Zero-inflation | ✅ | ❌ | ❌ | ⚠️ |
| Sparse reweighting | ✅ | ❌ | ❌ | ❌ |
| Multi-source fusion | ✅ | ❌ | ❌ | ⚠️ |
| Exact likelihood | ✅ | ❌ | ❌ | N/A |
| Stable training | ✅ | ⚠️ | ✅ | ✅ |

## Contents

```{tableofcontents}
```
