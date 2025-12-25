"""
Normalizing flow models for conditional generation.

Implements Conditional Masked Autoregressive Flow (MAF) for learning
the joint distribution of tax variables conditioned on demographics.
"""

from typing import Tuple
import torch
import torch.nn as nn
import numpy as np


class MADE(nn.Module):
    """
    Masked Autoencoder for Distribution Estimation (MADE).

    Implements autoregressive property: output[i] only depends on input[:i].
    Used as the conditioner network in MAF.
    """

    def __init__(
        self,
        n_features: int,
        n_context: int,
        hidden_dim: int,
        n_hidden: int = 2,
    ):
        """
        Initialize MADE network.

        Args:
            n_features: Number of input/output features
            n_context: Number of context/conditioning features
            hidden_dim: Size of hidden layers
            n_hidden: Number of hidden layers
        """
        super().__init__()
        self.n_features = n_features
        self.n_context = n_context
        self.hidden_dim = hidden_dim

        # Create masks for autoregressive property
        self._create_masks(n_hidden)

        # Input layer: takes concatenated [x, context]
        self.input_layer = nn.Linear(n_features + n_context, hidden_dim)

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden)
        ])

        # Output layer: predicts mu and log_scale for each feature
        self.output_layer = nn.Linear(hidden_dim, n_features * 2)

        self.activation = nn.ReLU()

    def _create_masks(self, n_hidden: int):
        """Create masks enforcing autoregressive property."""
        # Assign each hidden unit to a degree (which inputs it can see)
        rng = np.random.RandomState(42)

        # Input degrees: features have degrees 0 to n_features-1
        # Context features can connect to all (degree -1)
        input_degrees = np.concatenate([
            np.arange(self.n_features),
            np.full(self.n_context, -1)  # Context connects to all
        ])

        # Hidden degrees: uniform random assignment
        hidden_degrees = []
        for _ in range(n_hidden + 1):  # +1 for output layer
            if self.n_features > 1:
                degrees = rng.randint(0, self.n_features - 1, self.hidden_dim)
            else:
                # For single feature, all hidden units have degree 0
                degrees = np.zeros(self.hidden_dim, dtype=np.int64)
            hidden_degrees.append(degrees)

        # Output degrees: each output i needs degree < i
        output_degrees = np.arange(self.n_features)

        # Create masks
        # Input -> hidden1: hidden can see input if hidden_degree >= input_degree
        self.register_buffer(
            "input_mask",
            torch.tensor(
                hidden_degrees[0][:, None] >= input_degrees[None, :],
                dtype=torch.float32,
            ),
        )

        # Hidden -> hidden masks
        self.hidden_masks = []
        for i in range(n_hidden):
            mask = torch.tensor(
                hidden_degrees[i + 1][:, None] >= hidden_degrees[i][None, :],
                dtype=torch.float32,
            )
            self.register_buffer(f"hidden_mask_{i}", mask)
            self.hidden_masks.append(mask)

        # Hidden -> output: output i can see hidden if output_degree > hidden_degree
        # (strictly greater for autoregressive: output[i] depends on input[:i])
        self.register_buffer(
            "output_mask",
            torch.tensor(
                output_degrees[:, None] > hidden_degrees[-1][None, :],
                dtype=torch.float32,
            ),
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MADE.

        Args:
            x: Input features [batch, n_features]
            context: Context features [batch, n_context]

        Returns:
            mu: Mean parameters [batch, n_features]
            log_scale: Log scale parameters [batch, n_features]
        """
        # Concatenate input and context
        h = torch.cat([x, context], dim=-1)

        # Masked input layer
        h = self.activation(
            nn.functional.linear(h, self.input_mask * self.input_layer.weight,
                                 self.input_layer.bias)
        )

        # Masked hidden layers
        for i, layer in enumerate(self.hidden_layers):
            mask = getattr(self, f"hidden_mask_{i}")
            h = self.activation(
                nn.functional.linear(h, mask * layer.weight, layer.bias)
            )

        # Masked output layer
        out = nn.functional.linear(
            h, self.output_mask.repeat(2, 1) * self.output_layer.weight,
            self.output_layer.bias
        )

        # Split into mu and log_scale
        mu, log_scale = out.chunk(2, dim=-1)

        # Clamp log_scale for stability
        log_scale = torch.clamp(log_scale, min=-5, max=3)

        return mu, log_scale


class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer using MADE as the conditioner.

    Transform: z = (x - mu(x, context)) / exp(log_scale(x, context))
    This is invertible and the Jacobian is easy to compute.
    """

    def __init__(
        self,
        n_features: int,
        n_context: int,
        hidden_dim: int,
    ):
        """
        Initialize affine coupling layer.

        Args:
            n_features: Number of features to transform
            n_context: Number of context/conditioning features
            hidden_dim: Size of hidden layers in MADE
        """
        super().__init__()
        self.n_features = n_features
        self.made = MADE(n_features, n_context, hidden_dim)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation: x -> z.

        Args:
            x: Input [batch, n_features]
            context: Context [batch, n_context]

        Returns:
            z: Transformed output
            log_det: Log determinant of Jacobian
        """
        mu, log_scale = self.made(x, context)

        # Affine transform
        z = (x - mu) * torch.exp(-log_scale)

        # Log det Jacobian = -sum(log_scale)
        log_det = -log_scale.sum(dim=-1)

        return z, log_det

    def inverse(
        self, z: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        Inverse transformation: z -> x.

        Must be done autoregressively since mu, log_scale depend on x.

        Args:
            z: Latent space input
            context: Context features

        Returns:
            x: Reconstructed input
        """
        batch_size = z.shape[0]
        x = torch.zeros_like(z)

        for i in range(self.n_features):
            mu, log_scale = self.made(x, context)
            x[:, i] = z[:, i] * torch.exp(log_scale[:, i]) + mu[:, i]

        return x


class ConditionalMAF(nn.Module):
    """
    Conditional Masked Autoregressive Flow.

    Stacks multiple affine coupling layers with permutations between them
    to model complex distributions.
    """

    def __init__(
        self,
        n_features: int,
        n_context: int,
        n_layers: int = 4,
        hidden_dim: int = 64,
    ):
        """
        Initialize conditional MAF.

        Args:
            n_features: Number of features to model
            n_context: Number of context/conditioning features
            n_layers: Number of flow layers
            hidden_dim: Size of hidden layers
        """
        super().__init__()
        self.n_features = n_features
        self.n_context = n_context

        # Stack of affine coupling layers
        self.layers = nn.ModuleList([
            AffineCouplingLayer(n_features, n_context, hidden_dim)
            for _ in range(n_layers)
        ])

        # Permutations between layers (reverse order alternating)
        self.permutations = []
        for i in range(n_layers):
            if i % 2 == 0:
                perm = torch.arange(n_features)
            else:
                perm = torch.arange(n_features - 1, -1, -1)
            self.register_buffer(f"perm_{i}", perm)
            self.permutations.append(perm)

        # Base distribution (standard normal)
        self.register_buffer(
            "base_mean", torch.zeros(n_features)
        )
        self.register_buffer(
            "base_std", torch.ones(n_features)
        )

    def log_prob(
        self, x: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probability of x given context.

        Args:
            x: Data [batch, n_features]
            context: Context [batch, n_context]

        Returns:
            Log probability [batch]
        """
        z = x
        total_log_det = 0.0

        for i, layer in enumerate(self.layers):
            # Apply permutation
            perm = getattr(self, f"perm_{i}")
            z = z[:, perm]

            # Apply affine coupling
            z, log_det = layer(z, context)
            total_log_det = total_log_det + log_det

        # Log prob under base distribution
        base_log_prob = -0.5 * (
            self.n_features * np.log(2 * np.pi) +
            (z ** 2).sum(dim=-1)
        )

        return base_log_prob + total_log_det

    def sample(self, context: torch.Tensor) -> torch.Tensor:
        """
        Sample from the flow given context.

        Args:
            context: Context [batch, n_context]

        Returns:
            Samples [batch, n_features]
        """
        batch_size = context.shape[0]

        # Sample from base distribution
        z = torch.randn(batch_size, self.n_features, device=context.device)

        # Inverse transform through layers (in reverse order)
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]

            # Inverse affine coupling
            z = layer.inverse(z, context)

            # Inverse permutation
            perm = getattr(self, f"perm_{i}")
            inv_perm = torch.argsort(perm)
            z = z[:, inv_perm]

        return z
