"""
Synthesizer: Main class for conditional microdata synthesis.

Uses normalizing flows to learn the joint distribution of target
variables conditioned on context variables.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .transforms import MultiVariableTransformer
from .flows import ConditionalMAF
from .discrete import BinaryModel, DiscreteModelCollection


@dataclass
class SynthesizerConfig:
    """Configuration for Synthesizer."""

    target_vars: List[str]
    condition_vars: List[str]
    discrete_vars: Optional[List[str]] = None

    # Model architecture
    n_layers: int = 6
    hidden_dim: int = 64

    # Variable handling
    zero_inflated: bool = True
    log_transform: bool = True


class Synthesizer:
    """
    Conditional microdata synthesizer using normalizing flows.

    Learns P(target_vars | condition_vars) from training data,
    then generates synthetic target variables for new observations.

    Key features:
    - Handles zero-inflated variables (common in economic data)
    - Preserves joint correlations between target variables
    - Supports sample weights for survey data
    - Reproducible generation with seed parameter

    Example:
        >>> synth = Synthesizer(
        ...     target_vars=["income", "expenditure"],
        ...     condition_vars=["age", "education", "region"],
        ... )
        >>> synth.fit(training_data, weight_col="weight")
        >>> synthetic = synth.generate(new_demographics)
    """

    def __init__(
        self,
        target_vars: List[str],
        condition_vars: List[str],
        discrete_vars: Optional[List[str]] = None,
        n_layers: int = 6,
        hidden_dim: int = 64,
        zero_inflated: bool = True,
        log_transform: bool = True,
    ):
        """
        Initialize synthesizer.

        Args:
            target_vars: Variables to synthesize (continuous)
            condition_vars: Variables to condition on (preserved in output)
            discrete_vars: Additional discrete target variables (optional)
            n_layers: Number of layers in normalizing flow
            hidden_dim: Hidden layer size
            zero_inflated: Whether target vars have many zeros
            log_transform: Whether to log-transform positive values
        """
        self.target_vars = target_vars
        self.condition_vars = condition_vars
        self.discrete_vars = discrete_vars or []
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.zero_inflated = zero_inflated
        self.log_transform = log_transform

        # Will be set during fit
        self.transformer_: Optional[MultiVariableTransformer] = None
        self.flow_model_: Optional[ConditionalMAF] = None
        self.zero_indicators_: Optional[nn.ModuleDict] = None
        self.discrete_model_: Optional[DiscreteModelCollection] = None
        self.is_fitted_: bool = False
        self.training_history_: List[float] = []

    def fit(
        self,
        data: pd.DataFrame,
        weight_col: Optional[str] = "weight",
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        verbose: bool = True,
    ) -> "Synthesizer":
        """
        Fit synthesizer on training data.

        Uses a two-stage approach:
        1. Binary models predict P(positive | context) for each variable
        2. Normalizing flow learns P(value | context) for positive cases

        Args:
            data: DataFrame with target and condition variables
            weight_col: Name of weight column (None if unweighted)
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Optimizer learning rate
            verbose: Whether to print progress

        Returns:
            self
        """
        # Prepare data dict for transforms
        data_dict = {col: data[col].values for col in data.columns}

        # Fit transforms on target variables
        self.transformer_ = MultiVariableTransformer(
            self.target_vars,
            zero_inflated=self.zero_inflated,
            log_transform=self.log_transform,
        )
        self.transformer_.fit(data_dict, weight_col=weight_col or "weight")

        # Transform target variables
        transformed = self.transformer_.transform(data_dict)

        # Prepare tensors
        n_context = len(self.condition_vars)
        n_targets = len(self.target_vars)

        # Context tensor
        context_np = np.column_stack([
            data[var].values for var in self.condition_vars
        ])
        context = torch.tensor(context_np, dtype=torch.float32)

        # Target tensor (replace NaN with 0 for training)
        targets_list = []
        for var in self.target_vars:
            vals = transformed[var].copy()
            vals = np.nan_to_num(vals, nan=0.0)
            targets_list.append(vals)
        targets_np = np.column_stack(targets_list)
        targets = torch.tensor(targets_np, dtype=torch.float32)

        # Positive mask for each variable
        positive_mask = torch.ones_like(targets)
        for i, var in enumerate(self.target_vars):
            is_positive = (data[var].values > 0).astype(np.float32)
            positive_mask[:, i] = torch.tensor(is_positive)

        # Weights
        if weight_col and weight_col in data.columns:
            weights = torch.tensor(data[weight_col].values, dtype=torch.float32)
        else:
            weights = torch.ones(len(data), dtype=torch.float32)

        # Create normalizing flow
        self.flow_model_ = ConditionalMAF(
            n_features=n_targets,
            n_context=n_context,
            n_layers=self.n_layers,
            hidden_dim=self.hidden_dim,
        )

        # Train flow on positive observations
        self._train_flow(
            targets, context, weights, positive_mask,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=verbose,
        )

        # Train zero indicators
        self._train_zero_indicators(
            data, context,
            epochs=epochs // 2,
            learning_rate=learning_rate,
        )

        # Train discrete models if specified
        if self.discrete_vars:
            self._train_discrete(
                data, context,
                epochs=epochs // 2,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )

        self.is_fitted_ = True
        return self

    def _train_flow(
        self,
        targets: torch.Tensor,
        context: torch.Tensor,
        weights: torch.Tensor,
        positive_mask: torch.Tensor,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        verbose: bool,
    ):
        """Train the normalizing flow model."""
        optimizer = torch.optim.Adam(
            self.flow_model_.parameters(), lr=learning_rate
        )

        # Only train on rows where all target vars are positive
        all_positive = positive_mask.all(dim=1)

        if all_positive.sum() < 10:
            train_targets = targets
            train_context = context
            train_weights = weights
        else:
            train_targets = targets[all_positive]
            train_context = context[all_positive]
            train_weights = weights[all_positive]

        dataset = TensorDataset(train_targets, train_context, train_weights)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.training_history_ = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_targets, batch_context, batch_weights in loader:
                optimizer.zero_grad()

                log_prob = self.flow_model_.log_prob(batch_targets, batch_context)
                loss = -(log_prob * batch_weights).sum() / batch_weights.sum()

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.training_history_.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    def _train_zero_indicators(
        self,
        data: pd.DataFrame,
        context: torch.Tensor,
        epochs: int,
        learning_rate: float,
    ):
        """Train binary models for zero/non-zero indicators."""
        self.zero_indicators_ = nn.ModuleDict()

        for var in self.target_vars:
            model = BinaryModel(
                n_context=len(self.condition_vars),
                hidden_dim=self.hidden_dim // 2,
            )

            target = torch.tensor(
                (data[var].values > 0).astype(np.float32)
            ).unsqueeze(-1)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            for _ in range(epochs):
                optimizer.zero_grad()
                pred = model(context)
                loss = nn.functional.binary_cross_entropy(pred, target)
                loss.backward()
                optimizer.step()

            self.zero_indicators_[var] = model

    def _train_discrete(
        self,
        data: pd.DataFrame,
        context: torch.Tensor,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ):
        """Train discrete variable models."""
        binary_vars = []
        categorical_vars = {}

        for var in self.discrete_vars:
            unique_vals = data[var].nunique()
            if unique_vals == 2:
                binary_vars.append(var)
            else:
                categorical_vars[var] = unique_vals

        self.discrete_model_ = DiscreteModelCollection(
            n_context=len(self.condition_vars),
            binary_vars=binary_vars,
            categorical_vars=categorical_vars,
            hidden_dim=self.hidden_dim // 2,
        )

        targets = {
            var: torch.tensor(data[var].values, dtype=torch.long)
            for var in self.discrete_vars
        }

        optimizer = torch.optim.Adam(
            self.discrete_model_.parameters(), lr=learning_rate
        )

        for _ in range(epochs):
            optimizer.zero_grad()
            log_prob = self.discrete_model_.log_prob(context, targets)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()

    def generate(
        self,
        conditions: pd.DataFrame,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic target variables for given conditions.

        Two-stage generation:
        1. Sample zero/non-zero indicators for each target variable
        2. For non-zero cases, sample from flow and inverse transform

        Args:
            conditions: DataFrame with condition variables
            seed: Random seed for reproducibility

        Returns:
            DataFrame with conditions + synthetic target variables
        """
        if not self.is_fitted_:
            raise ValueError("Synthesizer not fitted. Call fit() first.")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Prepare context tensor
        context_np = np.column_stack([
            conditions[var].values for var in self.condition_vars
        ])
        context = torch.tensor(context_np, dtype=torch.float32)

        # Sample from flow
        with torch.no_grad():
            samples = self.flow_model_.sample(context)

        samples_np = samples.numpy()

        # Create dict with transformed values
        transformed_dict = {
            var: samples_np[:, i]
            for i, var in enumerate(self.target_vars)
        }

        # Inverse transform to original scale
        original_dict = self.transformer_.inverse_transform(transformed_dict)

        # Apply zero indicators
        with torch.no_grad():
            for var in self.target_vars:
                if self.zero_indicators_ and var in self.zero_indicators_:
                    prob_positive = self.zero_indicators_[var](context).squeeze(-1)
                    is_positive = torch.bernoulli(prob_positive).numpy()
                    original_dict[var] = np.where(
                        is_positive > 0.5,
                        original_dict[var],
                        0.0
                    )

        # Ensure non-negative values
        for var in self.target_vars:
            original_dict[var] = np.maximum(original_dict[var], 0)

        # Sample discrete variables
        if self.discrete_model_ is not None:
            with torch.no_grad():
                discrete_samples = self.discrete_model_.sample(context)
            for var in self.discrete_vars:
                original_dict[var] = discrete_samples[var].numpy().flatten()

        # Build result DataFrame
        result = conditions.copy()
        for var in self.target_vars:
            result[var] = original_dict[var]
        if self.discrete_model_ is not None:
            for var in self.discrete_vars:
                result[var] = original_dict[var]

        return result

    def save(self, path: Union[str, Path]) -> None:
        """Save fitted model to disk."""
        if not self.is_fitted_:
            raise ValueError("Synthesizer not fitted. Call fit() first.")

        state = {
            "target_vars": self.target_vars,
            "condition_vars": self.condition_vars,
            "discrete_vars": self.discrete_vars,
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
            "zero_inflated": self.zero_inflated,
            "log_transform": self.log_transform,
            "transformer": self.transformer_,
            "flow_state_dict": self.flow_model_.state_dict(),
            "zero_indicators_state_dict": (
                self.zero_indicators_.state_dict()
                if self.zero_indicators_ else None
            ),
            "discrete_state_dict": (
                self.discrete_model_.state_dict()
                if self.discrete_model_ else None
            ),
            "training_history": self.training_history_,
        }

        torch.save(state, Path(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Synthesizer":
        """Load fitted model from disk."""
        state = torch.load(Path(path), weights_only=False)

        synth = cls(
            target_vars=state["target_vars"],
            condition_vars=state["condition_vars"],
            discrete_vars=state["discrete_vars"],
            n_layers=state["n_layers"],
            hidden_dim=state["hidden_dim"],
            zero_inflated=state["zero_inflated"],
            log_transform=state["log_transform"],
        )

        synth.transformer_ = state["transformer"]
        synth.training_history_ = state["training_history"]

        # Reconstruct flow
        n_targets = len(state["target_vars"])
        n_context = len(state["condition_vars"])

        synth.flow_model_ = ConditionalMAF(
            n_features=n_targets,
            n_context=n_context,
            n_layers=state["n_layers"],
            hidden_dim=state["hidden_dim"],
        )
        synth.flow_model_.load_state_dict(state["flow_state_dict"])

        # Reconstruct zero indicators
        if state["zero_indicators_state_dict"]:
            synth.zero_indicators_ = nn.ModuleDict()
            for var in state["target_vars"]:
                synth.zero_indicators_[var] = BinaryModel(
                    n_context=n_context,
                    hidden_dim=state["hidden_dim"] // 2,
                )
            synth.zero_indicators_.load_state_dict(state["zero_indicators_state_dict"])

        # Reconstruct discrete model
        if state["discrete_state_dict"]:
            synth.discrete_model_ = DiscreteModelCollection(
                n_context=n_context,
                binary_vars=state["discrete_vars"],
                categorical_vars={},
                hidden_dim=state["hidden_dim"] // 2,
            )
            synth.discrete_model_.load_state_dict(state["discrete_state_dict"])

        synth.is_fitted_ = True
        return synth
