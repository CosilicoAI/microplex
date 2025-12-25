"""
Benchmark comparisons between microplex and other synthesis methods.

Compares:
- microplex (normalizing flows)
- CT-GAN (Conditional Tabular GAN from SDV)
- TVAE (Tabular VAE from SDV)
- GaussianCopula (from SDV)
- CART (sequential CART, similar to synthpop)

Metrics:
- Marginal fidelity (KS statistic)
- Joint fidelity (correlation preservation)
- Zero-fraction accuracy
- Training time
- Generation time
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    method: str
    dataset: str

    # Marginal fidelity
    ks_stats: Dict[str, float]
    mean_ks: float

    # Joint fidelity
    correlation_error: float

    # Zero handling
    zero_fraction_error: Dict[str, float]
    mean_zero_error: float

    # Timing
    train_time: float
    generate_time: float

    # Metadata
    n_train: int
    n_generate: int


def compute_marginal_fidelity(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: List[str],
) -> Tuple[Dict[str, float], float]:
    """Compute KS statistics for marginal distributions."""
    ks_stats = {}
    for col in columns:
        stat, _ = stats.ks_2samp(real[col], synthetic[col])
        ks_stats[col] = stat
    return ks_stats, np.mean(list(ks_stats.values()))


def compute_correlation_fidelity(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: List[str],
) -> float:
    """Compute correlation matrix preservation error."""
    real_corr = real[columns].corr().values
    synth_corr = synthetic[columns].corr().values

    # Frobenius norm of difference
    error = np.sqrt(np.sum((real_corr - synth_corr) ** 2))
    # Normalize by matrix size
    error /= len(columns)

    return error


def compute_zero_fidelity(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: List[str],
) -> Tuple[Dict[str, float], float]:
    """Compute zero-fraction preservation error."""
    errors = {}
    for col in columns:
        real_frac = (real[col] == 0).mean()
        synth_frac = (synthetic[col] == 0).mean()
        errors[col] = abs(real_frac - synth_frac)
    return errors, np.mean(list(errors.values()))


class MicroBenchmark:
    """Benchmark wrapper for micro synthesizer."""

    def __init__(self, target_vars: List[str], condition_vars: List[str], **kwargs):
        self.target_vars = target_vars
        self.condition_vars = condition_vars
        self.kwargs = kwargs
        self.model = None

    def fit(self, data: pd.DataFrame, **fit_kwargs):
        from microplex import Synthesizer

        self.model = Synthesizer(
            target_vars=self.target_vars,
            condition_vars=self.condition_vars,
            **self.kwargs,
        )
        self.model.fit(data, verbose=False, **fit_kwargs)

    def generate(self, conditions: pd.DataFrame) -> pd.DataFrame:
        return self.model.generate(conditions)


class CTGANBenchmark:
    """Benchmark wrapper for CT-GAN (from SDV)."""

    def __init__(self, target_vars: List[str], condition_vars: List[str], **kwargs):
        self.target_vars = target_vars
        self.condition_vars = condition_vars
        self.all_vars = condition_vars + target_vars
        self.model = None

    def fit(self, data: pd.DataFrame, epochs: int = 100, **kwargs):
        try:
            from sdv.single_table import CTGANSynthesizer
            from sdv.metadata import SingleTableMetadata
        except ImportError:
            raise ImportError("Install sdv: pip install sdv")

        # SDV requires metadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data[self.all_vars])

        self.model = CTGANSynthesizer(metadata, epochs=epochs, verbose=False)
        self.model.fit(data[self.all_vars])

    def generate(self, conditions: pd.DataFrame) -> pd.DataFrame:
        # CT-GAN doesn't do conditional generation natively,
        # so we generate and filter/match
        n = len(conditions)
        synthetic = self.model.sample(num_rows=n * 10)

        # Simple nearest-neighbor matching on condition vars
        result_rows = []
        for _, cond_row in conditions.iterrows():
            # Find closest match in synthetic data
            dists = np.sum([
                (synthetic[var] - cond_row[var]) ** 2
                for var in self.condition_vars
            ], axis=0)
            best_idx = np.argmin(dists)
            result_rows.append(synthetic.iloc[best_idx])

        result = pd.DataFrame(result_rows).reset_index(drop=True)

        # Preserve original conditions
        for var in self.condition_vars:
            result[var] = conditions[var].values

        return result


class TVAEBenchmark:
    """Benchmark wrapper for TVAE (from SDV)."""

    def __init__(self, target_vars: List[str], condition_vars: List[str], **kwargs):
        self.target_vars = target_vars
        self.condition_vars = condition_vars
        self.all_vars = condition_vars + target_vars
        self.model = None

    def fit(self, data: pd.DataFrame, epochs: int = 100, **kwargs):
        try:
            from sdv.single_table import TVAESynthesizer
            from sdv.metadata import SingleTableMetadata
        except ImportError:
            raise ImportError("Install sdv: pip install sdv")

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data[self.all_vars])

        self.model = TVAESynthesizer(metadata, epochs=epochs)
        self.model.fit(data[self.all_vars])

    def generate(self, conditions: pd.DataFrame) -> pd.DataFrame:
        n = len(conditions)
        synthetic = self.model.sample(num_rows=n * 10)

        result_rows = []
        for _, cond_row in conditions.iterrows():
            dists = np.sum([
                (synthetic[var] - cond_row[var]) ** 2
                for var in self.condition_vars
            ], axis=0)
            best_idx = np.argmin(dists)
            result_rows.append(synthetic.iloc[best_idx])

        result = pd.DataFrame(result_rows).reset_index(drop=True)
        for var in self.condition_vars:
            result[var] = conditions[var].values

        return result


class GaussianCopulaBenchmark:
    """Benchmark wrapper for Gaussian Copula (from SDV)."""

    def __init__(self, target_vars: List[str], condition_vars: List[str], **kwargs):
        self.target_vars = target_vars
        self.condition_vars = condition_vars
        self.all_vars = condition_vars + target_vars
        self.model = None

    def fit(self, data: pd.DataFrame, **kwargs):
        try:
            from sdv.single_table import GaussianCopulaSynthesizer
            from sdv.metadata import SingleTableMetadata
        except ImportError:
            raise ImportError("Install sdv: pip install sdv")

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data[self.all_vars])

        self.model = GaussianCopulaSynthesizer(metadata)
        self.model.fit(data[self.all_vars])

    def generate(self, conditions: pd.DataFrame) -> pd.DataFrame:
        n = len(conditions)
        synthetic = self.model.sample(num_rows=n * 10)

        result_rows = []
        for _, cond_row in conditions.iterrows():
            dists = np.sum([
                (synthetic[var] - cond_row[var]) ** 2
                for var in self.condition_vars
            ], axis=0)
            best_idx = np.argmin(dists)
            result_rows.append(synthetic.iloc[best_idx])

        result = pd.DataFrame(result_rows).reset_index(drop=True)
        for var in self.condition_vars:
            result[var] = conditions[var].values

        return result


def run_benchmark(
    train_data: pd.DataFrame,
    test_conditions: pd.DataFrame,
    target_vars: List[str],
    condition_vars: List[str],
    methods: Optional[List[str]] = None,
    epochs: int = 100,
) -> List[BenchmarkResult]:
    """
    Run benchmarks for multiple methods.

    Args:
        train_data: Training data with all variables
        test_conditions: Test conditions to generate for
        target_vars: Variables to synthesize
        condition_vars: Variables to condition on
        methods: List of methods to benchmark (default: all)
        epochs: Training epochs

    Returns:
        List of BenchmarkResult for each method
    """
    if methods is None:
        methods = ["micro", "ctgan", "tvae", "copula"]

    benchmarks = {
        "micro": MicroBenchmark,
        "ctgan": CTGANBenchmark,
        "tvae": TVAEBenchmark,
        "copula": GaussianCopulaBenchmark,
    }

    results = []

    for method_name in methods:
        print(f"\nBenchmarking {method_name}...")

        benchmark_cls = benchmarks[method_name]
        benchmark = benchmark_cls(target_vars, condition_vars)

        # Training
        start = time.time()
        try:
            benchmark.fit(train_data, epochs=epochs)
        except Exception as e:
            print(f"  {method_name} training failed: {e}")
            continue
        train_time = time.time() - start

        # Generation
        start = time.time()
        try:
            synthetic = benchmark.generate(test_conditions)
        except Exception as e:
            print(f"  {method_name} generation failed: {e}")
            continue
        generate_time = time.time() - start

        # Compute metrics
        ks_stats, mean_ks = compute_marginal_fidelity(
            train_data, synthetic, target_vars
        )
        corr_error = compute_correlation_fidelity(
            train_data, synthetic, target_vars
        )
        zero_errors, mean_zero = compute_zero_fidelity(
            train_data, synthetic, target_vars
        )

        result = BenchmarkResult(
            method=method_name,
            dataset="benchmark",
            ks_stats=ks_stats,
            mean_ks=mean_ks,
            correlation_error=corr_error,
            zero_fraction_error=zero_errors,
            mean_zero_error=mean_zero,
            train_time=train_time,
            generate_time=generate_time,
            n_train=len(train_data),
            n_generate=len(test_conditions),
        )
        results.append(result)

        print(f"  KS stat: {mean_ks:.4f}")
        print(f"  Corr error: {corr_error:.4f}")
        print(f"  Zero error: {mean_zero:.4f}")
        print(f"  Train time: {train_time:.1f}s")

    return results


def results_to_dataframe(results: List[BenchmarkResult]) -> pd.DataFrame:
    """Convert benchmark results to DataFrame for analysis."""
    rows = []
    for r in results:
        rows.append({
            "method": r.method,
            "mean_ks": r.mean_ks,
            "correlation_error": r.correlation_error,
            "mean_zero_error": r.mean_zero_error,
            "train_time": r.train_time,
            "generate_time": r.generate_time,
        })
    return pd.DataFrame(rows)
