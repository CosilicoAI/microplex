"""
Tests for the Synthesizer class.

TDD tests that verify the core synthesis workflow:
1. Initialize with target and condition variables
2. Fit on training data
3. Generate synthetic data for new conditions
4. Save and load models
"""

import pytest
import numpy as np
import pandas as pd
import torch


class TestSynthesizerInit:
    """Test Synthesizer initialization."""

    def test_basic_initialization(self):
        """Should initialize with target and condition variables."""
        from microplex import Synthesizer

        synth = Synthesizer(
            target_vars=["income", "expenditure"],
            condition_vars=["age", "education"],
        )

        assert synth.target_vars == ["income", "expenditure"]
        assert synth.condition_vars == ["age", "education"]
        assert not synth.is_fitted_

    def test_with_discrete_vars(self):
        """Should accept discrete target variables."""
        from microplex import Synthesizer

        synth = Synthesizer(
            target_vars=["income"],
            condition_vars=["age"],
            discrete_vars=["employed"],
        )

        assert synth.discrete_vars == ["employed"]


class TestSynthesizerFit:
    """Test Synthesizer training."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 1000

        age = np.random.randint(18, 80, n)
        education = np.random.choice([1, 2, 3, 4], n)

        # Income depends on age and education
        base = np.random.lognormal(10, 1, n)
        income = base * (1 + 0.01 * (age - 18)) * (1 + 0.2 * education)
        income[np.random.random(n) < 0.1] = 0  # 10% have zero income

        # Expenditure depends on income
        expenditure = income * np.random.uniform(0.5, 0.9, n)
        expenditure[income == 0] = 0

        return pd.DataFrame({
            "age": age,
            "education": education,
            "income": income,
            "expenditure": expenditure,
            "weight": np.ones(n),
        })

    def test_fit_completes(self, sample_data):
        """Fit should complete without errors."""
        from microplex import Synthesizer

        synth = Synthesizer(
            target_vars=["income", "expenditure"],
            condition_vars=["age", "education"],
        )

        synth.fit(sample_data, epochs=10)

        assert synth.is_fitted_

    def test_fit_learns_transforms(self, sample_data):
        """Fit should learn data transforms."""
        from microplex import Synthesizer

        synth = Synthesizer(
            target_vars=["income", "expenditure"],
            condition_vars=["age", "education"],
        )

        synth.fit(sample_data, epochs=10)

        assert synth.transformer_ is not None
        assert "income" in synth.transformer_.transformers_

    def test_fit_trains_flow(self, sample_data):
        """Fit should train the normalizing flow."""
        from microplex import Synthesizer

        synth = Synthesizer(
            target_vars=["income", "expenditure"],
            condition_vars=["age", "education"],
        )

        synth.fit(sample_data, epochs=50)

        assert synth.flow_model_ is not None
        assert synth.training_history_[-1] < synth.training_history_[0]


class TestSynthesizerGenerate:
    """Test synthetic data generation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 1000

        age = np.random.randint(18, 80, n)
        education = np.random.choice([1, 2, 3, 4], n)
        base = np.random.lognormal(10, 1, n)
        income = base * (1 + 0.01 * (age - 18)) * (1 + 0.2 * education)
        income[np.random.random(n) < 0.1] = 0
        expenditure = income * np.random.uniform(0.5, 0.9, n)
        expenditure[income == 0] = 0

        return pd.DataFrame({
            "age": age,
            "education": education,
            "income": income,
            "expenditure": expenditure,
            "weight": np.ones(n),
        })

    @pytest.fixture
    def fitted_synth(self, sample_data):
        """Return a fitted synthesizer."""
        from microplex import Synthesizer

        synth = Synthesizer(
            target_vars=["income", "expenditure"],
            condition_vars=["age", "education"],
        )
        synth.fit(sample_data, epochs=50, verbose=False)
        return synth

    @pytest.fixture
    def test_conditions(self):
        """Create test conditions for generation."""
        np.random.seed(123)
        n = 100

        return pd.DataFrame({
            "age": np.random.randint(18, 80, n),
            "education": np.random.choice([1, 2, 3, 4], n),
        })

    def test_generate_returns_dataframe(self, fitted_synth, test_conditions):
        """Generate should return a DataFrame."""
        result = fitted_synth.generate(test_conditions)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(test_conditions)

    def test_generate_includes_all_variables(self, fitted_synth, test_conditions):
        """Generated data should include all variables."""
        result = fitted_synth.generate(test_conditions)

        assert "age" in result.columns
        assert "education" in result.columns
        assert "income" in result.columns
        assert "expenditure" in result.columns

    def test_generate_preserves_conditions(self, fitted_synth, test_conditions):
        """Condition variables should be preserved exactly."""
        result = fitted_synth.generate(test_conditions)

        pd.testing.assert_series_equal(
            result["age"], test_conditions["age"], check_names=False
        )
        pd.testing.assert_series_equal(
            result["education"], test_conditions["education"], check_names=False
        )

    def test_generate_produces_non_negative(self, fitted_synth, test_conditions):
        """Generated values should be non-negative."""
        result = fitted_synth.generate(test_conditions)

        assert (result["income"] >= 0).all()
        assert (result["expenditure"] >= 0).all()

    def test_generate_is_stochastic(self, fitted_synth, test_conditions):
        """Multiple generations should differ."""
        result1 = fitted_synth.generate(test_conditions)
        result2 = fitted_synth.generate(test_conditions)

        assert not np.allclose(result1["income"].values, result2["income"].values)

    def test_generate_with_seed_is_reproducible(self, fitted_synth, test_conditions):
        """Generation with same seed should be reproducible."""
        result1 = fitted_synth.generate(test_conditions, seed=42)
        result2 = fitted_synth.generate(test_conditions, seed=42)

        np.testing.assert_array_equal(
            result1["income"].values, result2["income"].values
        )


class TestSaveLoad:
    """Test model serialization."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n = 500

        return pd.DataFrame({
            "age": np.random.randint(18, 80, n),
            "education": np.random.choice([1, 2, 3, 4], n),
            "income": np.random.lognormal(10, 1, n),
            "weight": np.ones(n),
        })

    def test_save_and_load(self, sample_data, tmp_path):
        """Should save and load model correctly."""
        from microplex import Synthesizer

        synth = Synthesizer(
            target_vars=["income"],
            condition_vars=["age", "education"],
        )
        synth.fit(sample_data, epochs=20, verbose=False)

        # Save
        save_path = tmp_path / "model.pt"
        synth.save(save_path)

        # Load
        loaded = Synthesizer.load(save_path)

        # Should generate same results with same seed
        conditions = sample_data[["age", "education"]].head(10)
        result1 = synth.generate(conditions, seed=42)
        result2 = loaded.generate(conditions, seed=42)

        np.testing.assert_array_almost_equal(
            result1["income"].values, result2["income"].values
        )
