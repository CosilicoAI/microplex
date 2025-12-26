"""
CPS ASEC Hierarchical Synthesis Demo

Demonstrates using microplex's HierarchicalSynthesizer on real CPS ASEC data:
1. Load CPS data using load_cps_for_synthesis()
2. Fit HierarchicalSynthesizer on the data
3. Generate synthetic households
4. Evaluate quality using benchmark metrics (dCor, energy distance, etc.)
5. Compare synthetic distributions to original

Outputs saved to examples/results/
"""

import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add paths for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

from microplex import HierarchicalSynthesizer, HouseholdSchema
from microplex.data import load_cps_for_synthesis, create_sample_data

warnings.filterwarnings("ignore")

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)


def setup_schema_for_cps():
    """Create a schema tailored for CPS ASEC data."""
    return HouseholdSchema(
        hh_vars=["n_persons", "n_adults", "n_children", "state_fips", "tenure"],
        person_vars=["age", "sex", "income", "employment_status", "education"],
        person_condition_vars=[
            "n_persons",
            "n_adults",
            "n_children",
            "state_fips",
            "tenure",
            "person_number",
            "is_first_adult",
            "is_child_slot",
        ],
        derived_vars={
            "hh_income": "sum:income",
            "n_workers": "count:employment_status==1",
        },
        hh_id_col="household_id",
        person_id_col="person_id",
    )


def compute_benchmark_metrics(
    original_persons: pd.DataFrame,
    synthetic_persons: pd.DataFrame,
    original_hh: pd.DataFrame,
    synthetic_hh: pd.DataFrame,
    person_vars: list,
    hh_vars: list,
) -> dict:
    """Compute comprehensive benchmark metrics."""
    from scipy import stats
    from scipy.spatial.distance import cdist
    from sklearn.preprocessing import StandardScaler

    metrics = {}

    # --- Person-level metrics ---
    print("\nComputing person-level metrics...")

    # 1. Marginal fidelity (KS statistics)
    ks_stats = {}
    for var in person_vars:
        if var in original_persons.columns and var in synthetic_persons.columns:
            orig_vals = original_persons[var].dropna().values
            synth_vals = synthetic_persons[var].dropna().values
            if len(orig_vals) > 0 and len(synth_vals) > 0:
                ks_stat, _ = stats.ks_2samp(orig_vals, synth_vals)
                ks_stats[var] = ks_stat

    metrics["person_ks_stats"] = ks_stats
    metrics["person_mean_ks"] = np.mean(list(ks_stats.values())) if ks_stats else 0

    # 2. Variance ratios (dispersion check)
    var_ratios = {}
    for var in person_vars:
        if var in original_persons.columns and var in synthetic_persons.columns:
            orig_var = np.var(original_persons[var].dropna())
            synth_var = np.var(synthetic_persons[var].dropna())
            if orig_var > 0:
                var_ratios[var] = synth_var / orig_var

    metrics["person_variance_ratios"] = var_ratios

    # 3. Zero-inflation accuracy (for income)
    if "income" in original_persons.columns:
        orig_zero_frac = (original_persons["income"] == 0).mean()
        synth_zero_frac = (synthetic_persons["income"] == 0).mean()
        metrics["income_zero_fraction_original"] = orig_zero_frac
        metrics["income_zero_fraction_synthetic"] = synth_zero_frac
        metrics["income_zero_fraction_error"] = abs(synth_zero_frac - orig_zero_frac)

    # 4. Distance correlation (dCor) - captures nonlinear relationships
    print("Computing distance correlation (dCor)...")
    dcor_results = compute_dcor_comparison(
        original_persons, synthetic_persons, person_vars
    )
    metrics["dcor"] = dcor_results

    # 5. Energy distance (multivariate)
    print("Computing energy distance...")
    numeric_vars = [v for v in person_vars if v in original_persons.columns]
    if len(numeric_vars) >= 2:
        orig_subset = original_persons[numeric_vars].dropna()
        synth_subset = synthetic_persons[numeric_vars].dropna()

        # Sample for computational efficiency
        n_sample = min(1000, len(orig_subset), len(synth_subset))
        if n_sample > 100:
            orig_sample = orig_subset.sample(n=n_sample, random_state=42)
            synth_sample = synth_subset.sample(n=n_sample, random_state=42)

            # Normalize
            scaler = StandardScaler()
            orig_norm = scaler.fit_transform(orig_sample)
            synth_norm = scaler.transform(synth_sample)

            energy_dist = compute_energy_distance(orig_norm, synth_norm)
            metrics["energy_distance"] = energy_dist

    # --- Household-level metrics ---
    print("\nComputing household-level metrics...")

    hh_ks_stats = {}
    for var in hh_vars:
        if var in original_hh.columns and var in synthetic_hh.columns:
            orig_vals = original_hh[var].dropna().values
            synth_vals = synthetic_hh[var].dropna().values
            if len(orig_vals) > 0 and len(synth_vals) > 0:
                ks_stat, _ = stats.ks_2samp(orig_vals, synth_vals)
                hh_ks_stats[var] = ks_stat

    metrics["hh_ks_stats"] = hh_ks_stats
    metrics["hh_mean_ks"] = np.mean(list(hh_ks_stats.values())) if hh_ks_stats else 0

    # Household size distribution
    if "n_persons" in original_hh.columns:
        orig_size_dist = original_hh["n_persons"].value_counts(normalize=True).sort_index()
        synth_size_dist = synthetic_hh["n_persons"].value_counts(normalize=True).sort_index()
        metrics["hh_size_dist_original"] = orig_size_dist.to_dict()
        metrics["hh_size_dist_synthetic"] = synth_size_dist.to_dict()

    return metrics


def distance_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Distance Correlation between two variables."""
    n = len(X)
    if n < 2:
        return 0.0

    X = np.asarray(X).reshape(-1)
    Y = np.asarray(Y).reshape(-1)

    a = np.abs(X[:, None] - X[None, :])
    b = np.abs(Y[:, None] - Y[None, :])

    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()

    dcov2 = (A * B).sum() / (n * n)
    dvar_x = (A * A).sum() / (n * n)
    dvar_y = (B * B).sum() / (n * n)

    if dvar_x <= 0 or dvar_y <= 0:
        return 0.0

    dcor = np.sqrt(dcov2 / np.sqrt(dvar_x * dvar_y))
    return float(np.clip(dcor, 0, 1))


def compute_dcor_comparison(
    original: pd.DataFrame, synthetic: pd.DataFrame, variables: list
) -> dict:
    """Compare distance correlation structure between original and synthetic."""
    # Sample for efficiency
    n_sample = min(500, len(original), len(synthetic))
    orig_sample = original.sample(n=n_sample, random_state=42)
    synth_sample = synthetic.sample(n=n_sample, random_state=42)

    available_vars = [v for v in variables if v in orig_sample.columns and v in synth_sample.columns]

    dcor_errors = {}
    orig_dcors = {}
    synth_dcors = {}

    for i, var1 in enumerate(available_vars):
        for j, var2 in enumerate(available_vars):
            if i < j:
                pair = f"{var1}_vs_{var2}"
                orig_dcor = distance_correlation(
                    orig_sample[var1].values, orig_sample[var2].values
                )
                synth_dcor = distance_correlation(
                    synth_sample[var1].values, synth_sample[var2].values
                )
                orig_dcors[pair] = orig_dcor
                synth_dcors[pair] = synth_dcor
                dcor_errors[pair] = abs(orig_dcor - synth_dcor)

    return {
        "original_dcors": orig_dcors,
        "synthetic_dcors": synth_dcors,
        "dcor_errors": dcor_errors,
        "mean_dcor_error": np.mean(list(dcor_errors.values())) if dcor_errors else 0,
    }


def compute_energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute energy distance between two distributions."""
    from scipy.spatial.distance import cdist

    n_x = len(X)
    n_y = len(Y)

    dist_XX = cdist(X, X, metric="euclidean")
    dist_YY = cdist(Y, Y, metric="euclidean")
    dist_XY = cdist(X, Y, metric="euclidean")

    term1 = 2 * np.mean(dist_XY)
    term2 = (np.sum(dist_XX) - np.trace(dist_XX)) / (n_x * (n_x - 1))
    term3 = (np.sum(dist_YY) - np.trace(dist_YY)) / (n_y * (n_y - 1))

    return float(term1 - term2 - term3)


def create_comparison_visualizations(
    original_persons: pd.DataFrame,
    synthetic_persons: pd.DataFrame,
    original_hh: pd.DataFrame,
    synthetic_hh: pd.DataFrame,
    metrics: dict,
    output_dir: Path,
):
    """Create comprehensive comparison visualizations."""
    print("\nGenerating visualizations...")

    # 1. Person-level variable distributions
    person_vars = ["age", "income", "education", "employment_status"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Person-Level Variable Distributions: Original vs Synthetic", fontsize=14)

    for idx, var in enumerate(person_vars):
        ax = axes[idx // 2, idx % 2]

        if var in original_persons.columns and var in synthetic_persons.columns:
            orig_vals = original_persons[var].dropna()
            synth_vals = synthetic_persons[var].dropna()

            if var == "income":
                # Log scale for income
                orig_positive = orig_vals[orig_vals > 0]
                synth_positive = synth_vals[synth_vals > 0]
                bins = np.logspace(
                    np.log10(max(1, orig_positive.min())),
                    np.log10(orig_positive.max()),
                    50,
                )
                ax.hist(orig_positive, bins=bins, alpha=0.5, label="Original", density=True)
                ax.hist(synth_positive, bins=bins, alpha=0.5, label="Synthetic", density=True)
                ax.set_xscale("log")
            else:
                ax.hist(orig_vals, bins=30, alpha=0.5, label="Original", density=True)
                ax.hist(synth_vals, bins=30, alpha=0.5, label="Synthetic", density=True)

            ks_stat = metrics["person_ks_stats"].get(var, 0)
            ax.text(
                0.95,
                0.95,
                f"KS: {ks_stat:.4f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        ax.set_xlabel(var.replace("_", " ").title())
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "person_distributions.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: person_distributions.png")
    plt.close()

    # 2. Household size distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Household Composition Comparison", fontsize=14)

    # Size distribution
    ax = axes[0]
    if "n_persons" in original_hh.columns:
        orig_sizes = original_hh["n_persons"].value_counts(normalize=True).sort_index()
        synth_sizes = synthetic_hh["n_persons"].value_counts(normalize=True).sort_index()

        all_sizes = sorted(set(orig_sizes.index) | set(synth_sizes.index))
        x = np.arange(len(all_sizes))
        width = 0.35

        orig_vals = [orig_sizes.get(s, 0) for s in all_sizes]
        synth_vals = [synth_sizes.get(s, 0) for s in all_sizes]

        ax.bar(x - width / 2, orig_vals, width, label="Original", alpha=0.8)
        ax.bar(x + width / 2, synth_vals, width, label="Synthetic", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(all_sizes)
        ax.set_xlabel("Household Size")
        ax.set_ylabel("Proportion")
        ax.set_title("Household Size Distribution")
        ax.legend()

    # Household income distribution
    ax = axes[1]
    if "hh_income" in synthetic_hh.columns:
        # Compute original HH income
        orig_hh_income = (
            original_persons.groupby("household_id")["income"].sum().reset_index()
        )
        synth_hh_income = synthetic_hh["hh_income"].dropna()

        orig_positive = orig_hh_income["income"][orig_hh_income["income"] > 0]
        synth_positive = synth_hh_income[synth_hh_income > 0]

        if len(orig_positive) > 0 and len(synth_positive) > 0:
            bins = np.logspace(
                np.log10(max(1, orig_positive.min())),
                np.log10(orig_positive.max()),
                50,
            )
            ax.hist(orig_positive, bins=bins, alpha=0.5, label="Original", density=True)
            ax.hist(synth_positive, bins=bins, alpha=0.5, label="Synthetic", density=True)
            ax.set_xscale("log")
            ax.set_xlabel("Household Income ($)")
            ax.set_ylabel("Density")
            ax.set_title("Household Income Distribution")
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "household_distributions.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: household_distributions.png")
    plt.close()

    # 3. Distance correlation comparison
    if "dcor" in metrics and metrics["dcor"]["dcor_errors"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Distance Correlation (dCor) Comparison", fontsize=14)

        dcor_data = metrics["dcor"]
        pairs = list(dcor_data["dcor_errors"].keys())[:10]  # Top 10 pairs

        # Original vs Synthetic dCor
        ax = axes[0]
        orig_vals = [dcor_data["original_dcors"][p] for p in pairs]
        synth_vals = [dcor_data["synthetic_dcors"][p] for p in pairs]
        x = np.arange(len(pairs))
        width = 0.35

        ax.bar(x - width / 2, orig_vals, width, label="Original", alpha=0.8)
        ax.bar(x + width / 2, synth_vals, width, label="Synthetic", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("_vs_", "\nvs\n") for p in pairs], fontsize=8)
        ax.set_ylabel("Distance Correlation")
        ax.set_title("dCor by Variable Pair")
        ax.legend()

        # dCor error
        ax = axes[1]
        errors = [dcor_data["dcor_errors"][p] for p in pairs]
        colors = ["green" if e < 0.05 else "orange" if e < 0.1 else "red" for e in errors]
        ax.bar(x, errors, color=colors, alpha=0.8)
        ax.axhline(y=0.05, color="green", linestyle="--", alpha=0.5, label="Good (<0.05)")
        ax.axhline(y=0.1, color="orange", linestyle="--", alpha=0.5, label="Acceptable (<0.1)")
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("_vs_", "\nvs\n") for p in pairs], fontsize=8)
        ax.set_ylabel("Absolute Error")
        ax.set_title("dCor Preservation Error")
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "dcor_comparison.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: dcor_comparison.png")
        plt.close()

    # 4. Age-Income relationship (key conditional relationship)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Age-Income Conditional Relationship", fontsize=14)

    for idx, (df, title) in enumerate(
        [(original_persons, "Original"), (synthetic_persons, "Synthetic")]
    ):
        ax = axes[idx]
        if "age" in df.columns and "income" in df.columns:
            sample = df.sample(n=min(2000, len(df)), random_state=42)
            positive = sample[sample["income"] > 0]

            ax.scatter(
                positive["age"],
                positive["income"],
                alpha=0.3,
                s=5,
            )
            ax.set_xlabel("Age")
            ax.set_ylabel("Income ($)")
            ax.set_title(f"{title} Data")
            ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_dir / "age_income_relationship.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: age_income_relationship.png")
    plt.close()

    # 5. Summary metrics chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Quality Metrics Summary", fontsize=14)

    # KS statistics
    ax = axes[0, 0]
    ks_data = metrics["person_ks_stats"]
    if ks_data:
        vars_list = list(ks_data.keys())
        values = list(ks_data.values())
        colors = ["green" if v < 0.1 else "orange" if v < 0.2 else "red" for v in values]
        ax.bar(vars_list, values, color=colors, alpha=0.8)
        ax.axhline(y=0.1, color="green", linestyle="--", alpha=0.5)
        ax.set_ylabel("KS Statistic")
        ax.set_title("Marginal Fidelity (KS Test)")
        ax.tick_params(axis="x", rotation=45)

    # Variance ratios
    ax = axes[0, 1]
    var_data = metrics.get("person_variance_ratios", {})
    if var_data:
        vars_list = list(var_data.keys())
        values = list(var_data.values())
        colors = ["green" if 0.8 <= v <= 1.2 else "orange" if 0.5 <= v <= 2 else "red" for v in values]
        ax.bar(vars_list, values, color=colors, alpha=0.8)
        ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)
        ax.axhline(y=0.8, color="gray", linestyle=":", alpha=0.3)
        ax.axhline(y=1.2, color="gray", linestyle=":", alpha=0.3)
        ax.set_ylabel("Variance Ratio (Synth / Orig)")
        ax.set_title("Dispersion Preservation")
        ax.tick_params(axis="x", rotation=45)

    # Zero-inflation
    ax = axes[1, 0]
    if "income_zero_fraction_original" in metrics:
        labels = ["Original", "Synthetic"]
        values = [
            metrics["income_zero_fraction_original"],
            metrics["income_zero_fraction_synthetic"],
        ]
        colors = ["steelblue", "coral"]
        ax.bar(labels, values, color=colors, alpha=0.8)
        ax.set_ylabel("Zero Fraction")
        ax.set_title(f"Income Zero-Inflation (Error: {metrics['income_zero_fraction_error']:.4f})")

    # Key metrics summary
    ax = axes[1, 1]
    summary_data = {
        "Mean KS": metrics.get("person_mean_ks", 0),
        "Mean dCor Error": metrics.get("dcor", {}).get("mean_dcor_error", 0),
        "Energy Distance": metrics.get("energy_distance", 0),
    }
    ax.bar(list(summary_data.keys()), list(summary_data.values()), color="steelblue", alpha=0.8)
    ax.set_ylabel("Metric Value")
    ax.set_title("Key Quality Metrics (Lower is Better)")
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_summary.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: metrics_summary.png")
    plt.close()


def save_metrics_report(metrics: dict, output_dir: Path):
    """Save metrics as markdown report."""
    report_path = output_dir / "cps_synthesis_report.md"

    with open(report_path, "w") as f:
        f.write("# CPS ASEC Hierarchical Synthesis Report\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"- **Mean KS Statistic (Person):** {metrics.get('person_mean_ks', 'N/A'):.4f}\n")
        f.write(f"- **Mean KS Statistic (Household):** {metrics.get('hh_mean_ks', 'N/A'):.4f}\n")
        f.write(f"- **Mean dCor Error:** {metrics.get('dcor', {}).get('mean_dcor_error', 'N/A'):.4f}\n")
        f.write(f"- **Energy Distance:** {metrics.get('energy_distance', 'N/A'):.4f}\n\n")

        f.write("## Person-Level Marginal Fidelity (KS Statistics)\n\n")
        f.write("| Variable | KS Statistic | Status |\n")
        f.write("|----------|--------------|--------|\n")
        for var, ks in metrics.get("person_ks_stats", {}).items():
            status = "Good" if ks < 0.1 else "Acceptable" if ks < 0.2 else "Poor"
            f.write(f"| {var} | {ks:.4f} | {status} |\n")

        f.write("\n## Variance Ratios (Dispersion Check)\n\n")
        f.write("| Variable | Ratio | Status |\n")
        f.write("|----------|-------|--------|\n")
        for var, ratio in metrics.get("person_variance_ratios", {}).items():
            status = "Good" if 0.8 <= ratio <= 1.2 else "Acceptable" if 0.5 <= ratio <= 2 else "Poor"
            f.write(f"| {var} | {ratio:.4f} | {status} |\n")

        f.write("\n## Zero-Inflation Accuracy\n\n")
        if "income_zero_fraction_original" in metrics:
            f.write(f"- Original zero fraction: {metrics['income_zero_fraction_original']:.4f}\n")
            f.write(f"- Synthetic zero fraction: {metrics['income_zero_fraction_synthetic']:.4f}\n")
            f.write(f"- Absolute error: {metrics['income_zero_fraction_error']:.4f}\n")

        f.write("\n## Distance Correlation (dCor) - Nonlinear Relationships\n\n")
        dcor_data = metrics.get("dcor", {})
        if dcor_data.get("dcor_errors"):
            f.write(f"**Mean dCor Error:** {dcor_data['mean_dcor_error']:.4f}\n\n")
            f.write("| Variable Pair | Original | Synthetic | Error |\n")
            f.write("|---------------|----------|-----------|-------|\n")
            for pair in list(dcor_data["dcor_errors"].keys())[:10]:
                orig = dcor_data["original_dcors"][pair]
                synth = dcor_data["synthetic_dcors"][pair]
                error = dcor_data["dcor_errors"][pair]
                f.write(f"| {pair} | {orig:.4f} | {synth:.4f} | {error:.4f} |\n")

        f.write("\n## Household Size Distribution\n\n")
        if "hh_size_dist_original" in metrics:
            f.write("| Size | Original | Synthetic |\n")
            f.write("|------|----------|----------|\n")
            orig_dist = metrics["hh_size_dist_original"]
            synth_dist = metrics["hh_size_dist_synthetic"]
            all_sizes = sorted(set(orig_dist.keys()) | set(synth_dist.keys()))
            for size in all_sizes:
                orig = orig_dist.get(size, 0)
                synth = synth_dist.get(size, 0)
                f.write(f"| {size} | {orig:.4f} | {synth:.4f} |\n")

        f.write("\n## Visualizations\n\n")
        f.write("- `person_distributions.png` - Person variable distributions\n")
        f.write("- `household_distributions.png` - Household composition\n")
        f.write("- `dcor_comparison.png` - Distance correlation analysis\n")
        f.write("- `age_income_relationship.png` - Conditional relationship\n")
        f.write("- `metrics_summary.png` - Quality metrics overview\n")

        f.write("\n## Interpretation Guide\n\n")
        f.write("**KS Statistic:**\n")
        f.write("- < 0.1: Excellent marginal match\n")
        f.write("- 0.1-0.2: Acceptable\n")
        f.write("- > 0.2: Poor match\n\n")

        f.write("**Variance Ratio:**\n")
        f.write("- 0.8-1.2: Good dispersion preservation\n")
        f.write("- < 0.8: Under-dispersed (mode collapse risk)\n")
        f.write("- > 1.2: Over-dispersed\n\n")

        f.write("**dCor Error:**\n")
        f.write("- < 0.05: Excellent relationship preservation\n")
        f.write("- 0.05-0.1: Good\n")
        f.write("- > 0.1: Nonlinear relationships may not be captured\n\n")

        f.write("**Energy Distance:**\n")
        f.write("- 0 = identical distributions\n")
        f.write("- Lower is better\n")

    print(f"\nSaved report: {report_path}")


def main():
    """Run CPS ASEC hierarchical synthesis demo."""

    print("=" * 80)
    print("CPS ASEC HIERARCHICAL SYNTHESIS DEMO")
    print("=" * 80)

    # Configuration
    n_synthetic_households = 10000
    epochs = 50
    sample_fraction = 0.2  # Use 20% of CPS data for faster demo

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Step 1: Load CPS data
    print("\n" + "=" * 80)
    print("STEP 1: Loading CPS ASEC Data")
    print("=" * 80)

    try:
        households, persons = load_cps_for_synthesis(
            sample_fraction=sample_fraction, random_state=42
        )
        print(f"\nLoaded CPS ASEC data:")
        print(f"  Households: {len(households):,}")
        print(f"  Persons: {len(persons):,}")
        print(f"  Avg HH size: {len(persons) / len(households):.2f}")

        # Show data summary
        print("\nHousehold variables:")
        for col in households.columns[:10]:
            print(f"  {col}: {households[col].dtype}")

        print("\nPerson variables:")
        for col in persons.columns[:10]:
            print(f"  {col}: {persons[col].dtype}")

    except FileNotFoundError as e:
        print(f"\nWARNING: CPS data not found. Using synthetic sample data instead.")
        print("To download real CPS data, run: python scripts/download_cps_asec.py\n")

        households, persons = create_sample_data(n_households=5000, seed=42)
        print(f"Generated sample data:")
        print(f"  Households: {len(households):,}")
        print(f"  Persons: {len(persons):,}")

    # Step 2: Set up and fit the hierarchical synthesizer
    print("\n" + "=" * 80)
    print("STEP 2: Training HierarchicalSynthesizer")
    print("=" * 80)

    schema = setup_schema_for_cps()
    print(f"\nSchema configuration:")
    print(f"  HH vars: {schema.hh_vars}")
    print(f"  Person vars: {schema.person_vars}")
    print(f"  Derived vars: {list(schema.derived_vars.keys())}")

    synth = HierarchicalSynthesizer(
        schema=schema,
        hh_flow_kwargs={"n_layers": 4, "hidden_dim": 64},
        person_flow_kwargs={"n_layers": 6, "hidden_dim": 128},
        random_state=42,
    )

    print(f"\nTraining with {epochs} epochs...")
    start_time = time.time()

    synth.fit(
        households,
        persons,
        hh_weight_col="hh_weight" if "hh_weight" in households.columns else None,
        epochs=epochs,
        verbose=True,
    )

    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.1f} seconds")

    # Step 3: Generate synthetic households
    print("\n" + "=" * 80)
    print(f"STEP 3: Generating {n_synthetic_households:,} Synthetic Households")
    print("=" * 80)

    start_time = time.time()
    synthetic_hh, synthetic_persons = synth.generate(
        n_households=n_synthetic_households, verbose=True
    )
    generate_time = time.time() - start_time

    print(f"\nGeneration completed in {generate_time:.1f} seconds")
    print(f"  Synthetic households: {len(synthetic_hh):,}")
    print(f"  Synthetic persons: {len(synthetic_persons):,}")
    print(f"  Avg HH size: {len(synthetic_persons) / len(synthetic_hh):.2f}")

    # Step 4: Evaluate quality
    print("\n" + "=" * 80)
    print("STEP 4: Evaluating Quality")
    print("=" * 80)

    person_vars = ["age", "sex", "income", "employment_status", "education"]
    hh_vars = ["n_persons", "n_adults", "n_children", "state_fips", "tenure"]

    metrics = compute_benchmark_metrics(
        persons,
        synthetic_persons,
        households,
        synthetic_hh,
        person_vars,
        hh_vars,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("QUALITY METRICS SUMMARY")
    print("=" * 80)

    print(f"\nPerson-Level Marginal Fidelity (KS Statistics):")
    for var, ks in metrics.get("person_ks_stats", {}).items():
        status = "Good" if ks < 0.1 else "Fair" if ks < 0.2 else "Poor"
        print(f"  {var}: {ks:.4f} [{status}]")
    print(f"  Mean KS: {metrics.get('person_mean_ks', 0):.4f}")

    print(f"\nVariance Ratios (should be close to 1.0):")
    for var, ratio in metrics.get("person_variance_ratios", {}).items():
        status = "Good" if 0.8 <= ratio <= 1.2 else "Fair" if 0.5 <= ratio <= 2 else "Poor"
        print(f"  {var}: {ratio:.4f} [{status}]")

    if "income_zero_fraction_error" in metrics:
        print(f"\nZero-Inflation (Income):")
        print(f"  Original zero fraction: {metrics['income_zero_fraction_original']:.4f}")
        print(f"  Synthetic zero fraction: {metrics['income_zero_fraction_synthetic']:.4f}")
        print(f"  Error: {metrics['income_zero_fraction_error']:.4f}")

    dcor_data = metrics.get("dcor", {})
    if dcor_data:
        print(f"\nDistance Correlation (captures nonlinear relationships):")
        print(f"  Mean dCor error: {dcor_data.get('mean_dcor_error', 0):.4f}")

    if "energy_distance" in metrics:
        print(f"\nEnergy Distance (multivariate): {metrics['energy_distance']:.4f}")

    print(f"\nHousehold-Level Marginal Fidelity:")
    for var, ks in metrics.get("hh_ks_stats", {}).items():
        status = "Good" if ks < 0.1 else "Fair" if ks < 0.2 else "Poor"
        print(f"  {var}: {ks:.4f} [{status}]")
    print(f"  Mean KS: {metrics.get('hh_mean_ks', 0):.4f}")

    # Step 5: Create visualizations and save report
    print("\n" + "=" * 80)
    print("STEP 5: Generating Visualizations and Report")
    print("=" * 80)

    create_comparison_visualizations(
        persons,
        synthetic_persons,
        households,
        synthetic_hh,
        metrics,
        output_dir,
    )

    save_metrics_report(metrics, output_dir)

    # Save synthetic data
    synthetic_hh.to_parquet(output_dir / "synthetic_households.parquet")
    synthetic_persons.to_parquet(output_dir / "synthetic_persons.parquet")
    print(f"\nSaved synthetic data to {output_dir}")

    # Final summary
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - cps_synthesis_report.md: Full quality report")
    print(f"  - person_distributions.png: Person variable comparisons")
    print(f"  - household_distributions.png: Household composition")
    print(f"  - dcor_comparison.png: Distance correlation analysis")
    print(f"  - age_income_relationship.png: Conditional relationship")
    print(f"  - metrics_summary.png: Quality metrics overview")
    print(f"  - synthetic_households.parquet: Generated households")
    print(f"  - synthetic_persons.parquet: Generated persons")

    print(f"\nKey Results:")
    print(f"  Training time: {train_time:.1f}s")
    print(f"  Generation time: {generate_time:.1f}s")
    print(f"  Mean KS (person): {metrics.get('person_mean_ks', 0):.4f}")
    print(f"  Mean dCor error: {dcor_data.get('mean_dcor_error', 0):.4f}")

    return metrics


if __name__ == "__main__":
    main()
