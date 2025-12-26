"""
Run TabPFN benchmark and generate results report.

Compares TabPFN (Prior-Data Fitted Networks) against microplex for
conditional microdata generation.

TabPFN is a transformer-based approach pre-trained on synthetic data
from a prior distribution. It uses in-context learning and doesn't
require traditional training/fine-tuning.

Key limitations of TabPFN:
- Works best on small datasets (<1000 rows, <100 features)
- Designed for prediction, not generation (adaptation required)
- Sequential generation may break joint distribution consistency
"""

import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from compare_tabpfn import (
    TabPFNBenchmarkResult,
    SequentialTabPFN,
    TabPFNWithZeroInflation,
    benchmark_tabpfn_vs_microplex,
    results_to_dataframe,
)
from run_benchmarks import generate_realistic_microdata

warnings.filterwarnings("ignore")

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def create_visualizations(
    results: list,
    real_data: pd.DataFrame,
    synthetic_data: dict,
    output_dir: Path,
):
    """Create benchmark visualizations."""

    # 1. Summary metrics bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = [r.method for r in results]
    colors = ["#2ecc71" if "microplex" in m else "#3498db" if "zero" in m else "#e74c3c" for m in methods]

    # KS statistic (lower is better)
    axes[0, 0].bar(methods, [r.mean_ks for r in results], color=colors)
    axes[0, 0].set_ylabel("Mean KS Statistic")
    axes[0, 0].set_title("Marginal Distribution Fidelity (lower is better)")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # Correlation error (lower is better)
    axes[0, 1].bar(methods, [r.correlation_error for r in results], color=colors)
    axes[0, 1].set_ylabel("Correlation Matrix Error")
    axes[0, 1].set_title("Joint Distribution Fidelity (lower is better)")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Zero-fraction error (lower is better)
    axes[1, 0].bar(methods, [r.mean_zero_error for r in results], color=colors)
    axes[1, 0].set_ylabel("Mean Zero-Fraction Error")
    axes[1, 0].set_title("Zero-Inflation Handling (lower is better)")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Generation time (lower is better)
    axes[1, 1].bar(methods, [r.generate_time for r in results], color=colors)
    axes[1, 1].set_ylabel("Generation Time (s)")
    axes[1, 1].set_title("Generation Speed (lower is better)")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "tabpfn_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'tabpfn_comparison.png'}")
    plt.close()

    # 2. Distribution comparison
    target_vars = ["income", "assets", "debt", "savings"]

    for method_name, synthetic in synthetic_data.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Distribution Comparison: {method_name.upper()}", fontsize=16)

        for idx, var in enumerate(target_vars):
            ax = axes[idx // 2, idx % 2]

            ax.hist(
                real_data[var],
                bins=50,
                alpha=0.5,
                label="Real",
                density=True,
                color="blue",
            )
            ax.hist(
                synthetic[var],
                bins=50,
                alpha=0.5,
                label="Synthetic",
                density=True,
                color="red",
            )

            ks_stat = next(r for r in results if r.method == method_name).ks_stats[var]
            ax.text(
                0.95,
                0.95,
                f"KS: {ks_stat:.4f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            ax.set_xlabel(var.capitalize())
            ax.set_ylabel("Density")
            ax.legend()
            ax.set_xlim(0, np.percentile(real_data[var], 95))

        plt.tight_layout()
        plt.savefig(
            output_dir / f"tabpfn_distributions_{method_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Saved: {output_dir / f'tabpfn_distributions_{method_name}.png'}")
        plt.close()

    # 3. Zero-inflation comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    zero_vars = ["assets", "debt"]
    width = 0.15
    x = np.arange(len(zero_vars))

    real_zeros = [(real_data[var] == 0).mean() for var in zero_vars]

    for i, method_name in enumerate(methods):
        synthetic = synthetic_data[method_name]
        synth_zeros = [(synthetic[var] == 0).mean() for var in zero_vars]
        axes[0].bar(x + i * width, synth_zeros, width, label=method_name, alpha=0.8)

    axes[0].bar(x + len(methods) * width, real_zeros, width, label="Real", alpha=0.8)
    axes[0].set_ylabel("Zero Fraction")
    axes[0].set_title("Zero-Inflation Preservation")
    axes[0].set_xticks(x + width * len(methods) / 2)
    axes[0].set_xticklabels([v.capitalize() for v in zero_vars])
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    for i, method_name in enumerate(methods):
        result = next(r for r in results if r.method == method_name)
        errors = [result.zero_fraction_error[var] for var in zero_vars]
        axes[1].bar(x + i * width, errors, width, label=method_name, alpha=0.8)

    axes[1].set_ylabel("Absolute Error")
    axes[1].set_title("Zero-Fraction Error")
    axes[1].set_xticks(x + width * (len(methods) - 1) / 2)
    axes[1].set_xticklabels([v.capitalize() for v in zero_vars])
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "tabpfn_zero_inflation.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'tabpfn_zero_inflation.png'}")
    plt.close()

    # 4. Per-variable KS statistics
    fig, ax = plt.subplots(figsize=(12, 6))

    width = 0.2
    x = np.arange(len(target_vars))

    for i, method_name in enumerate(methods):
        result = next(r for r in results if r.method == method_name)
        ks_values = [result.ks_stats[var] for var in target_vars]
        ax.bar(x + i * width, ks_values, width, label=method_name, alpha=0.8)

    ax.set_ylabel("KS Statistic")
    ax.set_title("Per-Variable Marginal Fidelity")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([v.capitalize() for v in target_vars])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "tabpfn_per_variable_ks.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'tabpfn_per_variable_ks.png'}")
    plt.close()


def save_results(results: list, output_dir: Path):
    """Save results as CSV and markdown."""

    df = results_to_dataframe(results)

    # CSV
    csv_path = output_dir / "tabpfn_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Markdown table
    md_path = output_dir / "tabpfn_results.md"
    with open(md_path, "w") as f:
        f.write("# TabPFN Benchmark Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"Saved: {md_path}")

    return df


def main():
    """Run TabPFN benchmark."""

    print("=" * 60)
    print("TABPFN BENCHMARK SUITE")
    print("=" * 60)

    # Configuration
    # TabPFN works best on small datasets, so we use smaller sizes
    n_train = 5000  # Full training data for microplex
    n_test = 1000
    epochs = 50
    max_tabpfn_samples = 1000  # TabPFN subsampled to this

    target_vars = ["income", "assets", "debt", "savings"]
    condition_vars = ["age", "education", "region"]

    # Create output directory
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Generate data
    print(f"\nGenerating realistic microdata...")
    print(f"  Training samples: {n_train}")
    print(f"  Test samples: {n_test}")
    print(f"  TabPFN will use: {max_tabpfn_samples} samples (limitation)")

    full_data = generate_realistic_microdata(n_train + n_test, seed=42)
    train_data = full_data.iloc[:n_train].copy()
    test_data = full_data.iloc[n_train:].copy()
    test_conditions = test_data[condition_vars].copy()

    # Print data statistics
    print("\nData statistics:")
    print(f"  Income: ${train_data['income'].mean():,.0f} +/- ${train_data['income'].std():,.0f}")
    print(f"  Assets: ${train_data['assets'].mean():,.0f} (zero-fraction: {(train_data['assets'] == 0).mean():.1%})")
    print(f"  Debt: ${train_data['debt'].mean():,.0f} (zero-fraction: {(train_data['debt'] == 0).mean():.1%})")
    print(f"  Savings: ${train_data['savings'].mean():,.0f}")

    # Run benchmark
    results, synthetic_data = benchmark_tabpfn_vs_microplex(
        train_data,
        test_conditions,
        target_vars,
        condition_vars,
        epochs=epochs,
        max_train_samples=max_tabpfn_samples,
    )

    if not results:
        print("\nERROR: All benchmarks failed!")
        return

    # Create visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    create_visualizations(results, train_data, synthetic_data, output_dir)

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    df = save_results(results, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    best_ks = min(results, key=lambda r: r.mean_ks)
    print(f"\nBest marginal fidelity: {best_ks.method} (KS: {best_ks.mean_ks:.4f})")

    best_corr = min(results, key=lambda r: r.correlation_error)
    print(f"Best correlation preservation: {best_corr.method} (error: {best_corr.correlation_error:.4f})")

    best_zero = min(results, key=lambda r: r.mean_zero_error)
    print(f"Best zero-inflation handling: {best_zero.method} (error: {best_zero.mean_zero_error:.4f})")

    fastest_gen = min(results, key=lambda r: r.generate_time)
    print(f"Fastest generation: {fastest_gen.method} ({fastest_gen.generate_time:.1f}s)")

    # Compare microplex vs best TabPFN
    microplex_result = next((r for r in results if r.method == "microplex"), None)
    tabpfn_results = [r for r in results if "tabpfn" in r.method]

    if microplex_result and tabpfn_results:
        best_tabpfn = min(tabpfn_results, key=lambda r: r.mean_ks)

        print(f"\n--- microplex vs {best_tabpfn.method} ---")

        ks_ratio = microplex_result.mean_ks / best_tabpfn.mean_ks if best_tabpfn.mean_ks > 0 else float('inf')
        print(f"KS ratio: {ks_ratio:.2f}x ({'microplex better' if ks_ratio < 1 else 'TabPFN better'})")

        corr_ratio = microplex_result.correlation_error / best_tabpfn.correlation_error if best_tabpfn.correlation_error > 0 else float('inf')
        print(f"Corr error ratio: {corr_ratio:.2f}x ({'microplex better' if corr_ratio < 1 else 'TabPFN better'})")

        zero_ratio = microplex_result.mean_zero_error / best_tabpfn.mean_zero_error if best_tabpfn.mean_zero_error > 0 else float('inf')
        print(f"Zero error ratio: {zero_ratio:.2f}x ({'microplex better' if zero_ratio < 1 else 'TabPFN better'})")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - tabpfn_results.csv")
    print(f"  - tabpfn_results.md")
    print(f"  - tabpfn_comparison.png")
    print(f"  - tabpfn_distributions_*.png")
    print(f"  - tabpfn_zero_inflation.png")
    print(f"  - tabpfn_per_variable_ks.png")

    return results, synthetic_data


if __name__ == "__main__":
    main()
