"""
Benchmark comparing microplex against TabPFN (Prior-Data Fitted Networks).

TabPFN is a transformer-based approach for tabular prediction that uses
prior-data fitting - pre-training on synthetic data from a prior distribution.

Key differences from microplex:
- TabPFN: Pre-trained transformer, in-context learning, no fine-tuning needed
- microplex: Trained normalizing flow for joint distribution modeling

TabPFN is designed for classification on small datasets (<1000 rows,
<100 features). We adapt it for conditional generation by:
1. Discretizing continuous targets into bins
2. Using TabPFN classifier to predict bin probabilities
3. Sampling from predicted bins and adding noise for continuity
4. Sequential prediction for multiple targets

Limitations:
- TabPFN v0.1.11 only supports classification (no regression)
- Works best on small datasets (designed for <1000 rows)
- Maximum 100 features, 10 classes
- Designed for prediction, not generation (adaptation required)

This benchmark compares:
- Marginal fidelity (KS statistic)
- Correlation preservation (Frobenius norm)
- Zero-inflation error
- Generation speed
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TabPFNBenchmarkResult:
    """Results from TabPFN benchmark."""

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


class SequentialTabPFN:
    """
    Sequential TabPFN implementation for conditional generation.

    Adapts TabPFN (designed for classification) for regression/generation by:
    1. Discretizing continuous targets into bins
    2. Using TabPFN classifier to predict bin probabilities
    3. Sampling from predicted bins with uniform noise within bins
    4. Sequential prediction: predict var1, then var2|var1, etc.

    Similar to Sequential QRF but using TabPFN instead of random forests.
    """

    def __init__(
        self,
        target_vars: List[str],
        condition_vars: List[str],
        n_bins: int = 10,
        n_ensemble_configurations: int = 3,
        random_state: int = 42,
    ):
        self.target_vars = target_vars
        self.condition_vars = condition_vars
        self.n_bins = n_bins
        self.n_ensemble_configurations = n_ensemble_configurations
        self.random_state = random_state

        # One model per target variable
        self.models = {}
        self.bin_edges = {}

    def fit(self, data: pd.DataFrame, verbose: bool = False):
        """
        Fit sequential TabPFN models.

        For each target variable:
        1. Discretize into bins
        2. Train a TabPFN classifier
        """
        from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

        features_so_far = self.condition_vars.copy()

        for i, target in enumerate(self.target_vars):
            if verbose:
                print(
                    f"  Fitting TabPFN {i+1}/{len(self.target_vars)}: {target} | {features_so_far}"
                )

            # Features = conditions + previously predicted targets
            X = data[features_so_far].values
            y = data[target].values

            # Discretize target into bins for classification
            # Use quantile-based binning to ensure balanced classes
            # Handle zero-inflated data specially
            non_zero_mask = y > 0
            if non_zero_mask.sum() > self.n_bins:
                # Create bins for non-zero values
                non_zero_vals = y[non_zero_mask]
                try:
                    _, bin_edges = pd.qcut(
                        non_zero_vals,
                        q=self.n_bins - 1,
                        retbins=True,
                        duplicates='drop'
                    )
                    # Add 0 as first edge
                    bin_edges = np.concatenate([[0], bin_edges])
                except ValueError:
                    # Fallback to uniform bins
                    bin_edges = np.linspace(0, np.max(y), self.n_bins + 1)
            else:
                bin_edges = np.linspace(0, np.max(y) + 1, self.n_bins + 1)

            self.bin_edges[target] = bin_edges

            # Assign bin labels
            y_binned = np.digitize(y, bin_edges[1:])  # Returns 0 to n_bins-1
            y_binned = np.clip(y_binned, 0, len(bin_edges) - 2)

            # Train classifier
            model = TabPFNClassifier(
                device='cpu',
                N_ensemble_configurations=self.n_ensemble_configurations,
                seed=self.random_state + i,
            )

            model.fit(X, y_binned)

            self.models[target] = {
                "model": model,
                "features": features_so_far.copy(),
                "bin_edges": bin_edges,
            }

            # Add this target to feature set for next variable
            features_so_far.append(target)

    def generate(self, conditions: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic data using sequential TabPFN prediction.

        For each sample:
        1. Predict bin probabilities from TabPFN
        2. Sample a bin from the distribution
        3. Add uniform noise within the bin
        """
        np.random.seed(self.random_state)
        n_samples = len(conditions)
        result = conditions.copy()

        for target in self.target_vars:
            model_info = self.models[target]
            model = model_info["model"]
            features = model_info["features"]
            bin_edges = model_info["bin_edges"]

            # Get features (includes previously predicted targets)
            X = result[features].values

            # Get bin probabilities
            probs = model.predict_proba(X)

            # Sample bins and generate values
            predictions = np.zeros(n_samples)
            for j in range(n_samples):
                # Sample a bin from the probability distribution
                bin_idx = np.random.choice(len(probs[j]), p=probs[j])

                # Generate value uniformly within the bin
                lower = bin_edges[bin_idx] if bin_idx < len(bin_edges) else bin_edges[-2]
                upper = bin_edges[bin_idx + 1] if bin_idx + 1 < len(bin_edges) else bin_edges[-1]

                predictions[j] = np.random.uniform(lower, upper)

            # Ensure non-negative for economic variables
            predictions = np.maximum(predictions, 0)

            result[target] = predictions

        return result


class TabPFNWithZeroInflation:
    """
    Enhanced TabPFN with two-stage zero-inflation handling.

    For each target variable:
    1. Train TabPFN classifier for P(positive | features)
    2. Train TabPFN classifier (via binning) for P(value | positive, features)

    This is a fairer comparison to microplex's two-stage approach.
    """

    def __init__(
        self,
        target_vars: List[str],
        condition_vars: List[str],
        n_bins: int = 10,
        n_ensemble_configurations: int = 3,
        zero_threshold: float = 1e-6,
        random_state: int = 42,
    ):
        self.target_vars = target_vars
        self.condition_vars = condition_vars
        self.n_bins = n_bins
        self.n_ensemble_configurations = n_ensemble_configurations
        self.zero_threshold = zero_threshold
        self.random_state = random_state

        self.models = {}

    def fit(self, data: pd.DataFrame, verbose: bool = False):
        """Fit two-stage TabPFN models for each target variable."""
        from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

        features_so_far = self.condition_vars.copy()

        for i, target in enumerate(self.target_vars):
            if verbose:
                print(
                    f"  Fitting TabPFN+ZI {i+1}/{len(self.target_vars)}: {target} | {features_so_far}"
                )

            X = data[features_so_far].values
            y = data[target].values

            # Stage 1: Binary classifier for P(positive)
            is_positive = (y > self.zero_threshold).astype(int)
            zero_frac = (is_positive == 0).mean()

            classifier = None
            if 0.01 < zero_frac < 0.99:  # Only train if non-trivial
                classifier = TabPFNClassifier(
                    device='cpu',
                    N_ensemble_configurations=self.n_ensemble_configurations,
                    seed=self.random_state + i,
                )
                classifier.fit(X, is_positive)

            # Stage 2: Classifier for positive values (via binning)
            regressor = None
            bin_edges = None
            if is_positive.sum() > 10:  # Need enough positive samples
                X_pos = X[is_positive == 1]
                y_pos = y[is_positive == 1]

                # Create bins for positive values
                try:
                    _, bin_edges = pd.qcut(
                        y_pos,
                        q=self.n_bins,
                        retbins=True,
                        duplicates='drop'
                    )
                except ValueError:
                    bin_edges = np.linspace(np.min(y_pos), np.max(y_pos), self.n_bins + 1)

                y_binned = np.digitize(y_pos, bin_edges[1:])
                y_binned = np.clip(y_binned, 0, len(bin_edges) - 2)

                regressor = TabPFNClassifier(
                    device='cpu',
                    N_ensemble_configurations=self.n_ensemble_configurations,
                    seed=self.random_state + i + 100,
                )
                regressor.fit(X_pos, y_binned)

            self.models[target] = {
                "classifier": classifier,
                "regressor": regressor,
                "features": features_so_far.copy(),
                "zero_frac": zero_frac,
                "bin_edges": bin_edges,
            }

            features_so_far.append(target)

    def generate(self, conditions: pd.DataFrame) -> pd.DataFrame:
        """Generate using two-stage process."""
        np.random.seed(self.random_state)
        n_samples = len(conditions)
        result = conditions.copy()

        for target in self.target_vars:
            model_info = self.models[target]
            classifier = model_info["classifier"]
            regressor = model_info["regressor"]
            features = model_info["features"]
            zero_frac = model_info["zero_frac"]
            bin_edges = model_info["bin_edges"]

            X = result[features].values

            # Stage 1: Predict which samples are positive
            if classifier is not None:
                is_positive_proba = classifier.predict_proba(X)[:, 1]
                is_positive = np.random.random(n_samples) < is_positive_proba
            else:
                # Fallback to observed zero fraction
                is_positive = np.random.random(n_samples) > zero_frac

            # Stage 2: Predict values for positive samples
            predictions = np.zeros(n_samples)

            if regressor is not None and is_positive.sum() > 0 and bin_edges is not None:
                X_pos = X[is_positive]

                # Get bin probabilities
                probs = regressor.predict_proba(X_pos)

                # Sample values
                n_pos = is_positive.sum()
                pred_pos = np.zeros(n_pos)
                for j in range(n_pos):
                    # Sample a bin
                    bin_idx = np.random.choice(len(probs[j]), p=probs[j])

                    # Generate value within bin
                    lower = bin_edges[bin_idx] if bin_idx < len(bin_edges) else bin_edges[-2]
                    upper = bin_edges[bin_idx + 1] if bin_idx + 1 < len(bin_edges) else bin_edges[-1]

                    pred_pos[j] = np.random.uniform(lower, upper)

                pred_pos = np.maximum(pred_pos, 0)
                predictions[is_positive] = pred_pos

            result[target] = predictions

        return result


def benchmark_tabpfn_vs_microplex(
    train_data: pd.DataFrame,
    test_conditions: pd.DataFrame,
    target_vars: List[str],
    condition_vars: List[str],
    epochs: int = 100,
    max_train_samples: int = 1000,
) -> Tuple[List[TabPFNBenchmarkResult], Dict[str, pd.DataFrame]]:
    """
    Run benchmark: TabPFN vs TabPFN+ZI vs microplex.

    Args:
        train_data: Training data with all variables
        test_conditions: Test conditions to generate for
        target_vars: Variables to synthesize
        condition_vars: Variables to condition on
        epochs: Training epochs for microplex
        max_train_samples: Maximum samples for TabPFN (it works best on <1000)

    Returns:
        results: List of benchmark results
        synthetic_data: Dict mapping method name to synthetic data
    """
    results = []
    synthetic_data = {}

    # Subsample training data for TabPFN if needed
    if len(train_data) > max_train_samples:
        print(f"Note: Subsampling training data from {len(train_data)} to {max_train_samples} for TabPFN")
        train_tabpfn = train_data.sample(n=max_train_samples, random_state=42)
    else:
        train_tabpfn = train_data

    methods = {
        "tabpfn_sequential": (SequentialTabPFN, train_tabpfn),
        "tabpfn_zero_inflation": (TabPFNWithZeroInflation, train_tabpfn),
    }

    # Benchmark TabPFN methods
    for method_name, (method_cls, train_subset) in methods.items():
        print(f"\n{'='*60}")
        print(f"Benchmarking: {method_name.upper()}")
        print(f"{'='*60}")

        model = method_cls(target_vars, condition_vars)

        # Training
        print("Training (fitting in-context)...")
        start = time.time()
        try:
            model.fit(train_subset, verbose=True)
        except Exception as e:
            print(f"ERROR: {method_name} training failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        train_time = time.time() - start
        print(f"Training time: {train_time:.1f}s")

        # Generation
        print("Generating synthetic data...")
        start = time.time()
        try:
            synthetic = model.generate(test_conditions)
        except Exception as e:
            print(f"ERROR: {method_name} generation failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        generate_time = time.time() - start
        print(f"Generation time: {generate_time:.1f}s")

        # Compute metrics (compare against full training data)
        print("Computing metrics...")

        # Marginal fidelity
        ks_stats = {}
        for var in target_vars:
            stat, _ = stats.ks_2samp(train_data[var], synthetic[var])
            ks_stats[var] = stat
        mean_ks = np.mean(list(ks_stats.values()))

        # Joint fidelity
        real_corr = train_data[target_vars].corr().values
        synth_corr = synthetic[target_vars].corr().values
        corr_error = np.sqrt(np.sum((real_corr - synth_corr) ** 2)) / len(target_vars)

        # Zero fidelity
        zero_errors = {}
        for var in target_vars:
            real_zero = (train_data[var] == 0).mean()
            synth_zero = (synthetic[var] == 0).mean()
            zero_errors[var] = abs(real_zero - synth_zero)
        mean_zero_error = np.mean(list(zero_errors.values()))

        result = TabPFNBenchmarkResult(
            method=method_name,
            dataset="economic_microdata",
            ks_stats=ks_stats,
            mean_ks=mean_ks,
            correlation_error=corr_error,
            zero_fraction_error=zero_errors,
            mean_zero_error=mean_zero_error,
            train_time=train_time,
            generate_time=generate_time,
            n_train=len(train_subset),
            n_generate=len(test_conditions),
        )

        results.append(result)
        synthetic_data[method_name] = synthetic

        # Print summary
        print(f"\nResults:")
        print(f"  Mean KS: {mean_ks:.4f}")
        print(f"  Correlation error: {corr_error:.4f}")
        print(f"  Zero-fraction error: {mean_zero_error:.4f}")

    # Benchmark microplex for comparison
    print(f"\n{'='*60}")
    print("Benchmarking: MICROPLEX (for comparison)")
    print(f"{'='*60}")

    try:
        from microplex import Synthesizer

        model = Synthesizer(target_vars=target_vars, condition_vars=condition_vars)

        print("Training...")
        start = time.time()
        model.fit(train_data, epochs=epochs, verbose=False)
        train_time = time.time() - start
        print(f"Training time: {train_time:.1f}s")

        print("Generating...")
        start = time.time()
        synthetic = model.generate(test_conditions)
        generate_time = time.time() - start
        print(f"Generation time: {generate_time:.1f}s")

        # Metrics
        ks_stats = {}
        for var in target_vars:
            stat, _ = stats.ks_2samp(train_data[var], synthetic[var])
            ks_stats[var] = stat
        mean_ks = np.mean(list(ks_stats.values()))

        real_corr = train_data[target_vars].corr().values
        synth_corr = synthetic[target_vars].corr().values
        corr_error = np.sqrt(np.sum((real_corr - synth_corr) ** 2)) / len(target_vars)

        zero_errors = {}
        for var in target_vars:
            real_zero = (train_data[var] == 0).mean()
            synth_zero = (synthetic[var] == 0).mean()
            zero_errors[var] = abs(real_zero - synth_zero)
        mean_zero_error = np.mean(list(zero_errors.values()))

        result = TabPFNBenchmarkResult(
            method="microplex",
            dataset="economic_microdata",
            ks_stats=ks_stats,
            mean_ks=mean_ks,
            correlation_error=corr_error,
            zero_fraction_error=zero_errors,
            mean_zero_error=mean_zero_error,
            train_time=train_time,
            generate_time=generate_time,
            n_train=len(train_data),
            n_generate=len(test_conditions),
        )

        results.append(result)
        synthetic_data["microplex"] = synthetic

        print(f"\nResults:")
        print(f"  Mean KS: {mean_ks:.4f}")
        print(f"  Correlation error: {corr_error:.4f}")
        print(f"  Zero-fraction error: {mean_zero_error:.4f}")

    except Exception as e:
        print(f"ERROR: microplex benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    return results, synthetic_data


def results_to_dataframe(results: List[TabPFNBenchmarkResult]) -> pd.DataFrame:
    """Convert benchmark results to DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "Method": r.method,
            "Mean KS": r.mean_ks,
            "Corr Error": r.correlation_error,
            "Zero Error": r.mean_zero_error,
            "Train (s)": r.train_time,
            "Gen (s)": r.generate_time,
            "N Train": r.n_train,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick test with small data (TabPFN works best on small datasets)
    from run_benchmarks import generate_realistic_microdata

    print("=" * 60)
    print("TABPFN BENCHMARK")
    print("=" * 60)

    print("\nGenerating test data...")
    # Use smaller dataset for TabPFN
    data = generate_realistic_microdata(n_samples=2000, seed=42)

    train = data.iloc[:1500]
    test = data.iloc[1500:]

    target_vars = ["income", "assets", "debt", "savings"]
    condition_vars = ["age", "education", "region"]

    print(f"Training samples: {len(train)}")
    print(f"Test samples: {len(test)}")

    results, synth = benchmark_tabpfn_vs_microplex(
        train,
        test[condition_vars],
        target_vars,
        condition_vars,
        epochs=50,
        max_train_samples=1000,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    df = results_to_dataframe(results)
    print(df.to_string(index=False))

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    if len(results) >= 2:
        microplex_result = next((r for r in results if r.method == "microplex"), None)
        tabpfn_results = [r for r in results if "tabpfn" in r.method]

        if microplex_result and tabpfn_results:
            best_tabpfn = min(tabpfn_results, key=lambda r: r.mean_ks)

            print(f"\nmicroplex vs {best_tabpfn.method}:")
            ks_ratio = microplex_result.mean_ks / best_tabpfn.mean_ks if best_tabpfn.mean_ks > 0 else float('inf')
            print(f"  KS ratio: {ks_ratio:.2f}x ({'microplex better' if ks_ratio < 1 else 'TabPFN better'})")

            corr_ratio = microplex_result.correlation_error / best_tabpfn.correlation_error if best_tabpfn.correlation_error > 0 else float('inf')
            print(f"  Corr error ratio: {corr_ratio:.2f}x ({'microplex better' if corr_ratio < 1 else 'TabPFN better'})")

            zero_ratio = microplex_result.mean_zero_error / best_tabpfn.mean_zero_error if best_tabpfn.mean_zero_error > 0 else float('inf')
            print(f"  Zero error ratio: {zero_ratio:.2f}x ({'microplex better' if zero_ratio < 1 else 'TabPFN better'})")
