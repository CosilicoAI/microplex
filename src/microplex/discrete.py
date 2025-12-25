"""
Models for discrete/categorical variables.

Handles binary variables (yes/no indicators) and categorical
variables (multi-class) separately from continuous variables.
"""

import torch
import torch.nn as nn


class BinaryModel(nn.Module):
    """
    Model for binary variables (0/1).

    Examples: has_income, is_employed, owns_home
    """

    def __init__(self, n_context: int, hidden_dim: int = 32):
        """
        Initialize binary variable model.

        Args:
            n_context: Number of conditioning features
            hidden_dim: Size of hidden layer
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_context, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Predict probability of 1 given context.

        Args:
            context: Conditioning features [batch, n_context]

        Returns:
            Probability of 1 [batch, 1]
        """
        return self.network(context)


class CategoricalModel(nn.Module):
    """
    Model for categorical variables (multiple classes).

    Example: education_level, region, industry
    """

    def __init__(self, n_context: int, n_categories: int, hidden_dim: int = 32):
        """
        Initialize categorical variable model.

        Args:
            n_context: Number of conditioning features
            n_categories: Number of categories
            hidden_dim: Size of hidden layer
        """
        super().__init__()
        self.n_categories = n_categories

        self.network = nn.Sequential(
            nn.Linear(n_context, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_categories),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Predict category probabilities given context.

        Args:
            context: Conditioning features [batch, n_context]

        Returns:
            Category probabilities [batch, n_categories]
        """
        logits = self.network(context)
        return torch.softmax(logits, dim=-1)


class DiscreteModelCollection(nn.Module):
    """
    Collection of discrete variable models.

    Manages multiple binary and categorical models for different variables.
    """

    def __init__(
        self,
        n_context: int,
        binary_vars: list,
        categorical_vars: dict,
        hidden_dim: int = 32,
    ):
        """
        Initialize collection of discrete models.

        Args:
            n_context: Number of context features
            binary_vars: List of binary variable names
            categorical_vars: Dict of {var_name: n_categories}
            hidden_dim: Size of hidden layers
        """
        super().__init__()
        self.binary_vars = binary_vars
        self.categorical_vars = categorical_vars

        # Binary models
        self.binary_models = nn.ModuleDict({
            var: BinaryModel(n_context, hidden_dim)
            for var in binary_vars
        })

        # Categorical models
        self.categorical_models = nn.ModuleDict({
            var: CategoricalModel(n_context, n_cats, hidden_dim)
            for var, n_cats in categorical_vars.items()
        })

    def forward(self, context: torch.Tensor) -> dict:
        """
        Predict probabilities for all discrete variables.

        Args:
            context: Conditioning features

        Returns:
            Dict of {var_name: probabilities}
        """
        result = {}

        for var in self.binary_vars:
            result[var] = self.binary_models[var](context)

        for var in self.categorical_vars:
            result[var] = self.categorical_models[var](context)

        return result

    def sample(self, context: torch.Tensor) -> dict:
        """
        Sample all discrete variables.

        Args:
            context: Conditioning features

        Returns:
            Dict of {var_name: samples}
        """
        probs = self.forward(context)
        result = {}

        for var in self.binary_vars:
            result[var] = torch.bernoulli(probs[var]).long()

        for var in self.categorical_vars:
            result[var] = torch.multinomial(probs[var], num_samples=1).squeeze(-1)

        return result

    def log_prob(self, context: torch.Tensor, targets: dict) -> torch.Tensor:
        """
        Compute log probability of discrete variables.

        Args:
            context: Conditioning features
            targets: Dict of {var_name: values}

        Returns:
            Total log probability [batch]
        """
        probs = self.forward(context)
        total_log_prob = 0.0

        for var in self.binary_vars:
            p = probs[var]
            y = targets[var].float()
            log_p = y * torch.log(p + 1e-8) + (1 - y) * torch.log(1 - p + 1e-8)
            total_log_prob = total_log_prob + log_p.sum(dim=-1)

        for var in self.categorical_vars:
            p = probs[var]
            y = targets[var].long()
            log_p = torch.log(p.gather(1, y.unsqueeze(-1)) + 1e-8)
            total_log_prob = total_log_prob + log_p.squeeze(-1)

        return total_log_prob
