# TabPFN Benchmark Comparison

**Date:** December 26, 2024
**TabPFN Version:** 0.1.11
**microplex Version:** 0.1.0

## Executive Summary

TabPFN (Prior-Data Fitted Networks) is a transformer-based approach for tabular prediction that uses prior-data fitting - pre-training on synthetic data from a prior distribution. This benchmark compares TabPFN-adapted generation against microplex for economic microdata synthesis.

### Key Findings

| Metric | microplex | TabPFN (best) | Winner |
|--------|-----------|---------------|--------|
| **Marginal Fidelity (KS)** | 0.0766 | 0.0716 | TabPFN+ZI (7% better) |
| **Correlation Preservation** | 0.0907 | 0.1451 | microplex (37% better) |
| **Zero-Inflation Error** | 0.0444 | 0.0324 | TabPFN+ZI (27% better) |
| **Generation Speed** | 0.01s | 1.86s | microplex (186x faster) |

**Verdict:** Mixed results. TabPFN with zero-inflation handling slightly edges microplex on marginal fidelity and zero handling, but microplex has significantly better correlation preservation and is dramatically faster at generation.

## Methodology

### TabPFN Adaptation for Generation

TabPFN is designed for classification on small datasets (<1000 rows). We adapted it for conditional generation through:

1. **Discretization**: Continuous targets are binned into 10 quantile-based categories
2. **Classification**: TabPFN classifier predicts bin probabilities
3. **Sampling**: Values are sampled uniformly within predicted bins
4. **Sequential Prediction**: Each target is predicted conditioned on demographics + previously predicted targets

### Two Approaches Tested

1. **TabPFN Sequential**: Direct discretization and sequential prediction
2. **TabPFN + Zero Inflation**: Two-stage approach similar to microplex:
   - Stage 1: Binary classifier for P(positive | features)
   - Stage 2: Binned regression for P(value | positive, features)

### Limitations of TabPFN for Generation

- **Small data only**: Works best on <1000 rows (we subsampled 5000 -> 1000)
- **Classification only**: v0.1.11 has no native regression (newer gated versions do)
- **Sequential breaks correlations**: Like QRF, sequential prediction can break joint distribution
- **Binning loses precision**: Discretization into 10 bins limits distribution fidelity
- **Slow generation**: In-context learning requires full transformer pass per prediction

## Detailed Results

### Summary Table

| Method | Mean KS | Corr Error | Zero Error | Train (s) | Gen (s) | N Train |
|--------|---------|------------|------------|-----------|---------|---------|
| tabpfn_sequential | 0.3052 | 0.1114 | 0.2297 | 0.6 | 1.6 | 1000 |
| tabpfn_zero_inflation | **0.0716** | 0.1451 | **0.0324** | 0.0 | 1.9 | 1000 |
| microplex | 0.0766 | **0.0907** | 0.0444 | 1.1 | **0.01** | 5000 |

### Analysis by Metric

#### Marginal Fidelity (KS Statistic)

- **TabPFN+ZI**: 0.0716 (best)
- **microplex**: 0.0766 (7% worse)
- **TabPFN Sequential**: 0.3052 (very poor)

The two-stage TabPFN approach matches marginal distributions well when using separate models for zero/non-zero. The sequential approach without zero handling performs poorly due to mode collapse in bins.

#### Correlation Preservation

- **microplex**: 0.0907 (best)
- **TabPFN Sequential**: 0.1114 (23% worse)
- **TabPFN+ZI**: 0.1451 (60% worse)

microplex's joint distribution modeling via normalizing flows provides significantly better correlation preservation. TabPFN's sequential nature (predict A, then B|A, then C|A,B) inherently breaks correlation structure.

#### Zero-Inflation Handling

- **TabPFN+ZI**: 0.0324 (best)
- **microplex**: 0.0444 (37% worse)
- **TabPFN Sequential**: 0.2297 (very poor)

The explicit two-stage modeling in TabPFN+ZI handles zeros well. The sequential approach without this fails completely at preserving zero fractions.

#### Generation Speed

- **microplex**: 0.01s (best)
- **TabPFN Sequential**: 1.6s (160x slower)
- **TabPFN+ZI**: 1.9s (190x slower)

microplex's normalizing flow is a single forward pass. TabPFN requires transformer inference for every sample, making it dramatically slower for generation.

## Per-Variable Analysis

| Variable | TabPFN Seq KS | TabPFN+ZI KS | microplex KS |
|----------|---------------|--------------|--------------|
| income | 0.181 | 0.068 | **0.062** |
| assets | 0.413 | **0.074** | 0.082 |
| debt | 0.396 | **0.071** | 0.078 |
| savings | 0.231 | 0.074 | **0.085** |

TabPFN+ZI performs best on zero-inflated variables (assets, debt), while microplex is more consistent across all variables.

## Conclusions

### When to Use TabPFN

- Very small datasets (<500 rows) where microplex may overfit
- When marginal distribution matching is the only priority
- When generation speed is not critical
- As a baseline comparison method

### When to Use microplex

- Larger datasets (1000+ rows)
- When joint distribution / correlation preservation matters
- When real-time generation is needed (microsimulation)
- When you need to scale to many samples

### Recommendations

1. **For PolicyEngine/Cosilico production**: Continue using microplex
   - Better correlation preservation is critical for policy simulation
   - Generation speed matters for interactive applications
   - Scales to full CPS/ACS datasets

2. **For benchmarking**: TabPFN+ZI provides a useful comparison point
   - Shows ceiling for marginal fidelity with limited training data
   - Validates that two-stage zero handling is effective

3. **Future work**: When TabPFN v2.5+ becomes publicly available without gating:
   - Re-run with native regression support
   - Test with ensemble methods
   - Compare computational efficiency improvements

## Reproducibility

```bash
cd /Users/maxghenis/CosilicoAI/micro
source .venv/bin/activate
pip install tabpfn==0.1.11  # Must use v0.1.11 (later versions are gated)
python benchmarks/run_tabpfn_benchmark.py
```

Results are deterministic with random seed = 42.

## Visualizations

- `tabpfn_comparison.png`: Summary metrics comparison
- `tabpfn_distributions_*.png`: Per-method distribution histograms
- `tabpfn_zero_inflation.png`: Zero-fraction preservation analysis
- `tabpfn_per_variable_ks.png`: Per-variable KS statistics

---

**Generated:** December 26, 2024
**Location:** /Users/maxghenis/CosilicoAI/micro/benchmarks/results/
