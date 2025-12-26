"""
Comprehensive performance benchmark: microplex vs PolicyEngine's QRF approach.

This benchmark compares:
1. Training time
2. Generation speed (samples per second)
3. Memory usage during training and inference
4. Statistical fidelity (marginal, correlation, zero-inflation)

Tested at different scales:
- Record counts: 1K, 10K, 100K, 1M
- Variable counts: 5, 10, 20

PolicyEngine currently uses Sequential Quantile Random Forests (QRF) for
microdata enhancement. microplex uses Masked Autoregressive Flows (MAF).
"""

import gc
import json
import os
import sys
import time
import tracemalloc
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

warnings.filterwarnings("ignore")

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 12


@dataclass
class PerformanceResult:
    """Complete benchmark result for a single configuration."""

    method: str
    n_records: int
    n_target_vars: int
    n_condition_vars: int

    # Timing
    train_time_seconds: float
    generate_time_seconds: float
    samples_per_second: float

    # Memory (in MB)
    train_peak_memory_mb: float
    generate_peak_memory_mb: float

    # Statistical fidelity
    mean_ks_statistic: float
    correlation_error: float
    mean_zero_error: float

    # Per-variable KS stats
    ks_per_variable: Dict[str, float]

    # Metadata
    timestamp: str
    success: bool
    error_message: Optional[str] = None


class SequentialQRF:
    """
    Sequential Quantile Random Forest - PolicyEngine's current approach.

    Uses two-stage modeling:
    1. Binary classifier for P(positive | features)
    2. Quantile regression for P(value | positive, features)

    Variables are predicted sequentially, with each subsequent variable
    conditioned on previously predicted variables.
    """

    def __init__(
        self,
        target_vars: List[str],
        condition_vars: List[str],
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42,
    ):
        self.target_vars = target_vars
        self.condition_vars = condition_vars
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.models = {}

    def fit(self, data: pd.DataFrame, verbose: bool = False):
        """Fit two-stage QRF models for each target variable."""
        from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
        )

        features_so_far = self.condition_vars.copy()

        for i, target in enumerate(self.target_vars):
            if verbose and (i + 1) % 5 == 0:
                print(f"  Training QRF {i+1}/{len(self.target_vars)}: {target}")

            X = data[features_so_far].values
            y = data[target].values

            # Stage 1: Binary classifier for P(positive)
            zero_threshold = 1e-6
            is_positive = (y > zero_threshold).astype(int)
            zero_frac = (is_positive == 0).mean()

            classifier = None
            if 0.01 < zero_frac < 0.99:
                classifier = HistGradientBoostingClassifier(
                    max_iter=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state + i,
                    verbose=0,
                )
                classifier.fit(X, is_positive)

            # Stage 2: Quantile regression for positive values
            regressor = None
            if is_positive.sum() > 10:
                X_pos = X[is_positive == 1]
                y_pos = y[is_positive == 1]

                regressor = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=0.5,
                    max_iter=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state + i,
                    verbose=0,
                )
                regressor.fit(X_pos, y_pos)

            self.models[target] = {
                "classifier": classifier,
                "regressor": regressor,
                "features": features_so_far.copy(),
                "zero_frac": zero_frac,
            }

            features_so_far.append(target)

    def generate(self, conditions: pd.DataFrame, seed: int = None) -> pd.DataFrame:
        """Generate synthetic data using sequential two-stage prediction."""
        if seed is not None:
            np.random.seed(seed)

        n_samples = len(conditions)
        result = conditions.copy()

        for target in self.target_vars:
            model_info = self.models[target]
            classifier = model_info["classifier"]
            regressor = model_info["regressor"]
            features = model_info["features"]
            zero_frac = model_info["zero_frac"]

            X = result[features].values

            # Stage 1: Predict which samples are positive
            if classifier is not None:
                is_positive_proba = classifier.predict_proba(X)[:, 1]
                is_positive = np.random.random(n_samples) < is_positive_proba
            else:
                is_positive = np.random.random(n_samples) > zero_frac

            # Stage 2: Predict values for positive samples
            predictions = np.zeros(n_samples)

            if regressor is not None and is_positive.sum() > 0:
                X_pos = X[is_positive]
                pred_pos = regressor.predict(X_pos)

                # Add noise for variability
                noise_scale = np.std(pred_pos) * 0.3 if len(pred_pos) > 1 else 1.0
                pred_pos += np.random.normal(0, noise_scale, len(pred_pos))
                pred_pos = np.maximum(pred_pos, 0)

                predictions[is_positive] = pred_pos

            result[target] = predictions

        return result


def generate_test_data(
    n_samples: int,
    n_target_vars: int,
    n_condition_vars: int = 3,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Generate realistic economic microdata for benchmarking.

    Creates correlated variables with zero-inflation patterns
    typical of household survey data (CPS, ACS).
    """
    np.random.seed(seed)

    # Generate condition variables (demographics)
    condition_data = {}
    condition_vars = []

    for i in range(n_condition_vars):
        if i == 0:
            # Age-like continuous
            condition_data[f"cond_{i}"] = np.random.normal(45, 15, n_samples).clip(18, 90)
        elif i == 1:
            # Education-like categorical
            condition_data[f"cond_{i}"] = np.random.choice(
                [1, 2, 3, 4], n_samples, p=[0.1, 0.3, 0.35, 0.25]
            )
        else:
            # Region-like or other demographics
            condition_data[f"cond_{i}"] = np.random.choice(
                list(range(1, 5)), n_samples
            )
        condition_vars.append(f"cond_{i}")

    # Generate target variables (economic outcomes)
    target_data = {}
    target_vars = []

    # Base effect from conditions
    age_effect = condition_data["cond_0"] if n_condition_vars > 0 else 45
    edu_effect = condition_data.get("cond_1", 2)

    for i in range(n_target_vars):
        var_name = f"target_{i}"
        target_vars.append(var_name)

        if i == 0:
            # Income-like: log-normal, always positive
            base = 30000 + edu_effect * 15000 + (age_effect - 18) * 800
            target_data[var_name] = np.maximum(
                0, np.random.lognormal(np.log(base) - 0.5, 0.7, n_samples)
            )
        elif i % 3 == 1:
            # Asset-like: zero-inflated (~40% zeros)
            has_value = np.random.random(n_samples) > 0.4
            base = target_data["target_0"] * 2 + edu_effect * 10000
            values = np.maximum(
                0, np.random.lognormal(np.log(base) - 1.0, 1.2, n_samples)
            )
            target_data[var_name] = np.where(has_value, values, 0)
        elif i % 3 == 2:
            # Debt-like: zero-inflated (~50% zeros)
            has_value = np.random.random(n_samples) > 0.5
            base = target_data["target_0"] * 0.5 + edu_effect * 5000
            values = np.maximum(
                0, np.random.lognormal(np.log(base) - 1.5, 1.0, n_samples)
            )
            target_data[var_name] = np.where(has_value, values, 0)
        else:
            # Savings-like: can be negative, correlated with others
            target_data[var_name] = (
                0.1 * target_data["target_0"]
                + 0.05 * target_data.get(f"target_{i-2}", 0)
                - 0.1 * target_data.get(f"target_{i-1}", 0)
                + np.random.normal(0, 5000, n_samples)
            )

    # Combine into DataFrame
    data = pd.DataFrame({**condition_data, **target_data})

    return data, target_vars, condition_vars


def measure_memory(func, *args, **kwargs) -> Tuple[Any, float]:
    """
    Measure peak memory usage of a function in MB.

    Returns:
        (function_result, peak_memory_mb)
    """
    gc.collect()
    tracemalloc.start()

    try:
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        peak_mb = peak / 1024 / 1024
    finally:
        tracemalloc.stop()

    return result, peak_mb


def compute_fidelity_metrics(
    train_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_vars: List[str],
) -> Tuple[float, Dict[str, float], float, float]:
    """
    Compute statistical fidelity metrics.

    Returns:
        (mean_ks, ks_per_var, correlation_error, mean_zero_error)
    """
    # Marginal fidelity: KS statistic per variable
    ks_stats = {}
    for var in target_vars:
        stat, _ = stats.ks_2samp(train_data[var], synthetic_data[var])
        ks_stats[var] = float(stat)
    mean_ks = np.mean(list(ks_stats.values()))

    # Correlation preservation
    real_corr = train_data[target_vars].corr().values
    synth_corr = synthetic_data[target_vars].corr().values
    corr_error = float(np.sqrt(np.sum((real_corr - synth_corr) ** 2)) / len(target_vars))

    # Zero-fraction preservation
    zero_errors = []
    for var in target_vars:
        real_zero = (train_data[var] == 0).mean()
        synth_zero = (synthetic_data[var] == 0).mean()
        if real_zero > 0.01:  # Only count if variable has zeros
            zero_errors.append(abs(real_zero - synth_zero))
    mean_zero_error = float(np.mean(zero_errors)) if zero_errors else 0.0

    return mean_ks, ks_stats, corr_error, mean_zero_error


def benchmark_method(
    method_name: str,
    model_class,
    train_data: pd.DataFrame,
    test_conditions: pd.DataFrame,
    target_vars: List[str],
    condition_vars: List[str],
    epochs: int = 100,
    verbose: bool = True,
) -> PerformanceResult:
    """
    Run complete benchmark for a single method.

    Measures training time, generation time, memory usage, and fidelity.
    """
    timestamp = datetime.now().isoformat()
    n_records = len(train_data)
    n_test = len(test_conditions)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {method_name.upper()}")
        print(f"Records: {n_records:,}, Target vars: {len(target_vars)}, Condition vars: {len(condition_vars)}")
        print(f"{'='*60}")

    try:
        # Initialize model
        if method_name == "microplex":
            from microplex import Synthesizer
            model = Synthesizer(
                target_vars=target_vars,
                condition_vars=condition_vars,
            )
        else:
            model = model_class(
                target_vars=target_vars,
                condition_vars=condition_vars,
            )

        # Training with memory measurement
        if verbose:
            print("Training...")

        gc.collect()
        tracemalloc.start()
        train_start = time.time()

        if method_name == "microplex":
            model.fit(train_data, epochs=epochs, verbose=False)
        else:
            model.fit(train_data, verbose=False)

        train_time = time.time() - train_start
        _, train_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        train_peak_mb = train_peak / 1024 / 1024

        if verbose:
            print(f"  Training time: {train_time:.2f}s")
            print(f"  Training peak memory: {train_peak_mb:.1f} MB")

        # Generation with memory measurement
        if verbose:
            print("Generating synthetic data...")

        gc.collect()
        tracemalloc.start()
        gen_start = time.time()

        synthetic = model.generate(test_conditions)

        gen_time = time.time() - gen_start
        _, gen_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        gen_peak_mb = gen_peak / 1024 / 1024

        samples_per_second = n_test / gen_time if gen_time > 0 else float('inf')

        if verbose:
            print(f"  Generation time: {gen_time:.4f}s")
            print(f"  Samples/second: {samples_per_second:,.0f}")
            print(f"  Generation peak memory: {gen_peak_mb:.1f} MB")

        # Compute fidelity metrics
        if verbose:
            print("Computing fidelity metrics...")

        mean_ks, ks_per_var, corr_error, mean_zero_error = compute_fidelity_metrics(
            train_data, synthetic, target_vars
        )

        if verbose:
            print(f"  Mean KS statistic: {mean_ks:.4f}")
            print(f"  Correlation error: {corr_error:.4f}")
            print(f"  Zero-fraction error: {mean_zero_error:.4f}")

        return PerformanceResult(
            method=method_name,
            n_records=n_records,
            n_target_vars=len(target_vars),
            n_condition_vars=len(condition_vars),
            train_time_seconds=train_time,
            generate_time_seconds=gen_time,
            samples_per_second=samples_per_second,
            train_peak_memory_mb=train_peak_mb,
            generate_peak_memory_mb=gen_peak_mb,
            mean_ks_statistic=mean_ks,
            correlation_error=corr_error,
            mean_zero_error=mean_zero_error,
            ks_per_variable=ks_per_var,
            timestamp=timestamp,
            success=True,
        )

    except Exception as e:
        if verbose:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

        return PerformanceResult(
            method=method_name,
            n_records=n_records,
            n_target_vars=len(target_vars),
            n_condition_vars=len(condition_vars),
            train_time_seconds=0,
            generate_time_seconds=0,
            samples_per_second=0,
            train_peak_memory_mb=0,
            generate_peak_memory_mb=0,
            mean_ks_statistic=1.0,
            correlation_error=1.0,
            mean_zero_error=1.0,
            ks_per_variable={},
            timestamp=timestamp,
            success=False,
            error_message=str(e),
        )


def run_scale_benchmark(
    record_counts: List[int] = [1000, 10000, 100000],
    variable_counts: List[int] = [5, 10, 20],
    epochs: int = 50,
    verbose: bool = True,
) -> List[PerformanceResult]:
    """
    Run benchmarks at different scales.

    Args:
        record_counts: List of training record counts to test
        variable_counts: List of target variable counts to test
        epochs: Training epochs for microplex
        verbose: Print progress

    Returns:
        List of PerformanceResult objects
    """
    results = []
    total_configs = len(record_counts) * len(variable_counts) * 2  # 2 methods
    config_num = 0

    for n_records in record_counts:
        for n_vars in variable_counts:
            # Skip very large configurations that would take too long
            if n_records >= 1000000 and n_vars >= 20:
                if verbose:
                    print(f"\nSkipping {n_records:,} x {n_vars} vars (too large)")
                continue

            config_num += 1
            if verbose:
                print(f"\n{'#'*70}")
                print(f"Configuration {config_num}/{total_configs}: {n_records:,} records, {n_vars} variables")
                print(f"{'#'*70}")

            # Generate data
            if verbose:
                print(f"Generating test data...")

            data, target_vars, condition_vars = generate_test_data(
                n_samples=n_records + 2000,  # Extra for test set
                n_target_vars=n_vars,
                n_condition_vars=3,
                seed=42,
            )

            train_data = data.iloc[:n_records]
            test_conditions = data.iloc[n_records:][condition_vars]

            # Benchmark QRF (PolicyEngine's approach)
            result_qrf = benchmark_method(
                method_name="policyengine_qrf",
                model_class=SequentialQRF,
                train_data=train_data,
                test_conditions=test_conditions,
                target_vars=target_vars,
                condition_vars=condition_vars,
                epochs=epochs,
                verbose=verbose,
            )
            results.append(result_qrf)

            # Benchmark microplex
            result_microplex = benchmark_method(
                method_name="microplex",
                model_class=None,  # Will be imported internally
                train_data=train_data,
                test_conditions=test_conditions,
                target_vars=target_vars,
                condition_vars=condition_vars,
                epochs=epochs,
                verbose=verbose,
            )
            results.append(result_microplex)

            # Clear memory between configurations
            gc.collect()

    return results


def create_visualizations(
    results: List[PerformanceResult],
    output_dir: Path,
):
    """Create comprehensive benchmark visualizations."""

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame([asdict(r) for r in results if r.success])

    if len(df) == 0:
        print("No successful results to visualize")
        return

    # 1. Training Time vs Scale
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Training time by record count
    ax = axes[0, 0]
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        grouped = method_df.groupby('n_records')['train_time_seconds'].mean()
        ax.plot(grouped.index, grouped.values, 'o-', label=method, linewidth=2, markersize=8)
    ax.set_xlabel('Number of Records')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time vs Dataset Size')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Generation speed by record count
    ax = axes[0, 1]
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        grouped = method_df.groupby('n_records')['samples_per_second'].mean()
        ax.plot(grouped.index, grouped.values, 'o-', label=method, linewidth=2, markersize=8)
    ax.set_xlabel('Number of Records (Training)')
    ax.set_ylabel('Samples per Second')
    ax.set_title('Generation Speed vs Training Size')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Training memory by record count
    ax = axes[1, 0]
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        grouped = method_df.groupby('n_records')['train_peak_memory_mb'].mean()
        ax.plot(grouped.index, grouped.values, 'o-', label=method, linewidth=2, markersize=8)
    ax.set_xlabel('Number of Records')
    ax.set_ylabel('Peak Memory (MB)')
    ax.set_title('Training Memory vs Dataset Size')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mean KS by record count
    ax = axes[1, 1]
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        grouped = method_df.groupby('n_records')['mean_ks_statistic'].mean()
        ax.plot(grouped.index, grouped.values, 'o-', label=method, linewidth=2, markersize=8)
    ax.set_xlabel('Number of Records')
    ax.set_ylabel('Mean KS Statistic (lower is better)')
    ax.set_title('Statistical Fidelity vs Dataset Size')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'scale_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved scale comparison to {output_dir / 'scale_comparison.png'}")
    plt.close()

    # 2. Variable count comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Training time by variable count
    ax = axes[0, 0]
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        grouped = method_df.groupby('n_target_vars')['train_time_seconds'].mean()
        ax.bar(
            grouped.index + (0.2 if method == 'microplex' else -0.2),
            grouped.values,
            width=0.35,
            label=method,
            alpha=0.8,
        )
    ax.set_xlabel('Number of Target Variables')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time vs Number of Variables')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Generation speed by variable count
    ax = axes[0, 1]
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        grouped = method_df.groupby('n_target_vars')['samples_per_second'].mean()
        ax.bar(
            grouped.index + (0.2 if method == 'microplex' else -0.2),
            grouped.values,
            width=0.35,
            label=method,
            alpha=0.8,
        )
    ax.set_xlabel('Number of Target Variables')
    ax.set_ylabel('Samples per Second')
    ax.set_title('Generation Speed vs Number of Variables')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Correlation error by variable count
    ax = axes[1, 0]
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        grouped = method_df.groupby('n_target_vars')['correlation_error'].mean()
        ax.bar(
            grouped.index + (0.2 if method == 'microplex' else -0.2),
            grouped.values,
            width=0.35,
            label=method,
            alpha=0.8,
        )
    ax.set_xlabel('Number of Target Variables')
    ax.set_ylabel('Correlation Error (lower is better)')
    ax.set_title('Correlation Preservation vs Number of Variables')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Zero error by variable count
    ax = axes[1, 1]
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        grouped = method_df.groupby('n_target_vars')['mean_zero_error'].mean()
        ax.bar(
            grouped.index + (0.2 if method == 'microplex' else -0.2),
            grouped.values,
            width=0.35,
            label=method,
            alpha=0.8,
        )
    ax.set_xlabel('Number of Target Variables')
    ax.set_ylabel('Zero-Fraction Error (lower is better)')
    ax.set_title('Zero-Inflation Handling vs Number of Variables')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'variable_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved variable comparison to {output_dir / 'variable_comparison.png'}")
    plt.close()

    # 3. Summary comparison at 10K records
    df_10k = df[df['n_records'] == 10000]
    if len(df_10k) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        metrics = [
            ('train_time_seconds', 'Training Time (s)', True),
            ('generate_time_seconds', 'Generation Time (s)', True),
            ('train_peak_memory_mb', 'Training Memory (MB)', True),
            ('mean_ks_statistic', 'Mean KS (lower=better)', True),
            ('correlation_error', 'Correlation Error', True),
            ('mean_zero_error', 'Zero-Fraction Error', True),
        ]

        for idx, (metric, label, lower_better) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]

            microplex_val = df_10k[df_10k['method'] == 'microplex'][metric].mean()
            qrf_val = df_10k[df_10k['method'] == 'policyengine_qrf'][metric].mean()

            bars = ax.bar(['microplex', 'PolicyEngine QRF'], [microplex_val, qrf_val],
                         color=['#2ecc71', '#3498db'], alpha=0.8)
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for bar, val in zip(bars, [microplex_val, qrf_val]):
                height = bar.get_height()
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)

        plt.suptitle('Performance Comparison at 10K Records', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'summary_10k.png', dpi=300, bbox_inches='tight')
        print(f"Saved 10K summary to {output_dir / 'summary_10k.png'}")
        plt.close()

    # 4. Speedup/improvement ratios
    fig, ax = plt.subplots(figsize=(12, 6))

    ratios = []
    labels = []

    for n_records in df['n_records'].unique():
        df_subset = df[df['n_records'] == n_records]
        if len(df_subset) < 2:
            continue

        microplex = df_subset[df_subset['method'] == 'microplex'].iloc[0] if len(df_subset[df_subset['method'] == 'microplex']) > 0 else None
        qrf = df_subset[df_subset['method'] == 'policyengine_qrf'].iloc[0] if len(df_subset[df_subset['method'] == 'policyengine_qrf']) > 0 else None

        if microplex is not None and qrf is not None:
            # Training speedup
            train_speedup = qrf['train_time_seconds'] / microplex['train_time_seconds'] if microplex['train_time_seconds'] > 0 else 1
            gen_speedup = microplex['samples_per_second'] / qrf['samples_per_second'] if qrf['samples_per_second'] > 0 else 1
            ks_improvement = qrf['mean_ks_statistic'] / microplex['mean_ks_statistic'] if microplex['mean_ks_statistic'] > 0 else 1

            ratios.append({
                'n_records': n_records,
                'Training Speedup': train_speedup,
                'Generation Speedup': gen_speedup,
                'Fidelity Improvement': ks_improvement,
            })

    if ratios:
        ratio_df = pd.DataFrame(ratios)
        x = np.arange(len(ratio_df))
        width = 0.25

        ax.bar(x - width, ratio_df['Training Speedup'], width, label='Training Speedup', color='#e74c3c')
        ax.bar(x, ratio_df['Generation Speedup'], width, label='Generation Speedup', color='#2ecc71')
        ax.bar(x + width, ratio_df['Fidelity Improvement'], width, label='Fidelity Improvement', color='#3498db')

        ax.set_xlabel('Number of Records')
        ax.set_ylabel('Improvement Ratio (microplex / QRF)')
        ax.set_title('microplex vs PolicyEngine QRF: Improvement Ratios\n(>1 means microplex is better)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(r):,}' for r in ratio_df['n_records']])
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / 'improvement_ratios.png', dpi=300, bbox_inches='tight')
        print(f"Saved improvement ratios to {output_dir / 'improvement_ratios.png'}")
        plt.close()


def generate_report(
    results: List[PerformanceResult],
    output_dir: Path,
) -> str:
    """Generate markdown benchmark report."""

    df = pd.DataFrame([asdict(r) for r in results])
    successful = df[df['success'] == True]

    report = f"""# PolicyEngine vs microplex Performance Benchmark

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This benchmark compares microplex (Masked Autoregressive Flows) against PolicyEngine's
current approach (Sequential Quantile Random Forests) for synthetic microdata generation.

### Key Findings

"""

    # Compute aggregate statistics
    if len(successful) > 0:
        microplex = successful[successful['method'] == 'microplex']
        qrf = successful[successful['method'] == 'policyengine_qrf']

        if len(microplex) > 0 and len(qrf) > 0:
            avg_train_speedup = qrf['train_time_seconds'].mean() / microplex['train_time_seconds'].mean()
            avg_gen_speedup = microplex['samples_per_second'].mean() / qrf['samples_per_second'].mean()
            avg_ks_improvement = qrf['mean_ks_statistic'].mean() / microplex['mean_ks_statistic'].mean()
            avg_memory_ratio = qrf['train_peak_memory_mb'].mean() / microplex['train_peak_memory_mb'].mean()

            report += f"""| Metric | microplex | PolicyEngine QRF | Winner |
|--------|-----------|------------------|--------|
| Avg Training Time | {microplex['train_time_seconds'].mean():.2f}s | {qrf['train_time_seconds'].mean():.2f}s | {'microplex' if avg_train_speedup > 1 else 'QRF'} ({avg_train_speedup:.1f}x) |
| Avg Generation Speed | {microplex['samples_per_second'].mean():,.0f}/s | {qrf['samples_per_second'].mean():,.0f}/s | {'microplex' if avg_gen_speedup > 1 else 'QRF'} ({avg_gen_speedup:.1f}x) |
| Avg Training Memory | {microplex['train_peak_memory_mb'].mean():.1f}MB | {qrf['train_peak_memory_mb'].mean():.1f}MB | {'microplex' if avg_memory_ratio > 1 else 'QRF'} |
| Avg KS Statistic | {microplex['mean_ks_statistic'].mean():.4f} | {qrf['mean_ks_statistic'].mean():.4f} | {'microplex' if avg_ks_improvement > 1 else 'QRF'} ({avg_ks_improvement:.1f}x better) |
| Avg Correlation Error | {microplex['correlation_error'].mean():.4f} | {qrf['correlation_error'].mean():.4f} | {'microplex' if microplex['correlation_error'].mean() < qrf['correlation_error'].mean() else 'QRF'} |
| Avg Zero-Fraction Error | {microplex['mean_zero_error'].mean():.4f} | {qrf['mean_zero_error'].mean():.4f} | {'microplex' if microplex['mean_zero_error'].mean() < qrf['mean_zero_error'].mean() else 'QRF'} |

"""

    report += """## Benchmark Configurations

### Scale Testing
- **Record counts:** 1K, 10K, 100K (optionally 1M)
- **Variable counts:** 5, 10, 20 target variables
- **Condition variables:** 3 (age, education, region)

### Methods Compared

**microplex (Masked Autoregressive Flows)**
- Joint distribution modeling via normalizing flows
- Two-stage zero-inflation handling
- GPU-accelerated training

**PolicyEngine QRF (Sequential Quantile Random Forests)**
- Sequential prediction: each variable conditioned on previously predicted
- Two-stage: binary classifier + quantile regression
- Uses scikit-learn's HistGradientBoostingRegressor

## Detailed Results

"""

    # Results by scale
    for n_records in sorted(df['n_records'].unique()):
        df_subset = df[df['n_records'] == n_records]
        report += f"""### {n_records:,} Records

| Method | Variables | Train Time | Gen Speed | Memory | KS Stat | Corr Err | Zero Err |
|--------|-----------|------------|-----------|--------|---------|----------|----------|
"""
        for _, row in df_subset.iterrows():
            status = "OK" if row['success'] else "FAILED"
            if row['success']:
                report += f"| {row['method']} | {row['n_target_vars']} | {row['train_time_seconds']:.2f}s | {row['samples_per_second']:,.0f}/s | {row['train_peak_memory_mb']:.1f}MB | {row['mean_ks_statistic']:.4f} | {row['correlation_error']:.4f} | {row['mean_zero_error']:.4f} |\n"
            else:
                report += f"| {row['method']} | {row['n_target_vars']} | FAILED | - | - | - | - | - |\n"

        report += "\n"

    report += """## Visualizations

The following visualizations are available in the `benchmarks/results/` directory:

1. **scale_comparison.png** - Training time, generation speed, memory, and fidelity vs dataset size
2. **variable_comparison.png** - Performance vs number of target variables
3. **summary_10k.png** - Direct comparison at 10K records
4. **improvement_ratios.png** - microplex improvement over QRF

## Interpretation Guide

### KS Statistic (Kolmogorov-Smirnov)
- Measures how well marginal distributions are preserved
- Range: 0 (perfect) to 1 (completely different)
- **Lower is better**

### Correlation Error
- Frobenius norm of correlation matrix difference
- Measures joint distribution preservation
- **Lower is better**

### Zero-Fraction Error
- Absolute difference in proportion of zeros
- Critical for zero-inflated economic variables
- **Lower is better**

### Samples per Second
- Generation throughput
- **Higher is better**

## Recommendations

Based on these benchmarks:

"""

    if len(microplex) > 0 and len(qrf) > 0:
        if avg_ks_improvement > 1.5:
            report += f"""1. **Use microplex for production** - {avg_ks_improvement:.1f}x better statistical fidelity
"""
        if avg_gen_speedup > 1.5:
            report += f"""2. **microplex for high-throughput** - {avg_gen_speedup:.1f}x faster generation
"""
        if avg_train_speedup > 1.2:
            report += f"""3. **microplex trains faster** - {avg_train_speedup:.1f}x speedup on average
"""

    report += """
## Reproducibility

```bash
cd /Users/maxghenis/CosilicoAI/micro
source .venv/bin/activate
python benchmarks/compare_policyengine.py
```

Results are reproducible with seed=42.

---
*Generated by microplex benchmark suite*
"""

    return report


def main():
    """Run the complete PolicyEngine comparison benchmark."""

    print("="*80)
    print("MICROPLEX vs POLICYENGINE PERFORMANCE BENCHMARK")
    print("="*80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration
    record_counts = [1000, 10000, 100000]
    variable_counts = [5, 10, 20]
    epochs = 50

    # Create output directory
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Run benchmarks
    print(f"\nRunning benchmarks...")
    print(f"  Record counts: {record_counts}")
    print(f"  Variable counts: {variable_counts}")
    print(f"  Epochs: {epochs}")

    results = run_scale_benchmark(
        record_counts=record_counts,
        variable_counts=variable_counts,
        epochs=epochs,
        verbose=True,
    )

    # Save raw results as JSON
    results_json = [asdict(r) for r in results]
    with open(output_dir / "policyengine_comparison.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved raw results to {output_dir / 'policyengine_comparison.json'}")

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results, output_dir)

    # Generate report
    print("\nGenerating report...")
    report = generate_report(results, output_dir)

    report_path = output_dir / "policyengine_comparison.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report to {report_path}")

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"\nSuccessful runs: {len(successful)}")
    print(f"Failed runs: {len(failed)}")

    if successful:
        microplex_results = [r for r in successful if r.method == 'microplex']
        qrf_results = [r for r in successful if r.method == 'policyengine_qrf']

        if microplex_results and qrf_results:
            avg_train_microplex = np.mean([r.train_time_seconds for r in microplex_results])
            avg_train_qrf = np.mean([r.train_time_seconds for r in qrf_results])
            avg_ks_microplex = np.mean([r.mean_ks_statistic for r in microplex_results])
            avg_ks_qrf = np.mean([r.mean_ks_statistic for r in qrf_results])

            print(f"\nKey metrics (averages):")
            print(f"  microplex training: {avg_train_microplex:.2f}s")
            print(f"  QRF training: {avg_train_qrf:.2f}s")
            print(f"  Training speedup: {avg_train_qrf/avg_train_microplex:.1f}x")
            print(f"  microplex KS: {avg_ks_microplex:.4f}")
            print(f"  QRF KS: {avg_ks_qrf:.4f}")
            print(f"  Fidelity improvement: {avg_ks_qrf/avg_ks_microplex:.1f}x")

    print(f"\nResults saved to: {output_dir}")
    print("  - policyengine_comparison.json")
    print("  - policyengine_comparison.md")
    print("  - scale_comparison.png")
    print("  - variable_comparison.png")
    print("  - summary_10k.png")
    print("  - improvement_ratios.png")


if __name__ == "__main__":
    main()
