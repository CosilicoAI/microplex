"""
Data transformations for microdata synthesis.

Handles common patterns in survey/administrative data:
- Zero-inflated variables (many observations are exactly 0)
- Heavy-tailed distributions (log transform)
- Standardization (for neural network training)
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch


class ZeroInflatedTransform:
    """Handle zero-inflated variables by splitting into indicator and values.

    Common in economic data where many people have $0 for a given category
    (e.g., capital gains, medical expenditures, business income).
    """

    def split(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into zero indicator and positive values.

        Args:
            x: Array of values (may contain zeros)

        Returns:
            indicator: Binary array (1 if positive, 0 if zero)
            positive_values: Array of only the positive values
        """
        indicator = (x > 0).astype(np.float64)
        positive_values = x[x > 0]
        return indicator, positive_values

    def combine(
        self, indicator: np.ndarray, positive_values: np.ndarray
    ) -> np.ndarray:
        """
        Recombine indicator and positive values.

        Args:
            indicator: Binary array (1 if should be positive)
            positive_values: Array of positive values to fill in

        Returns:
            Combined array with zeros and positive values
        """
        result = np.zeros_like(indicator, dtype=np.float64)
        positive_mask = indicator > 0.5
        result[positive_mask] = positive_values
        return result


class LogTransform:
    """Log transformation for heavy-tailed distributions.

    Most income/expenditure variables are approximately log-normal,
    so log-transforming before modeling improves results.
    """

    def __init__(self, offset: float = 0.0):
        """
        Initialize log transform.

        Args:
            offset: Value to add before log (for handling zeros/small values)
        """
        self.offset = offset

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply log transform: log(x + offset)."""
        return np.log(x + self.offset)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        """Inverse log transform: exp(y) - offset."""
        return np.exp(y) - self.offset


class Standardizer:
    """Standardize data to zero mean and unit variance.

    Supports sample weights for proper handling of survey data.
    """

    def __init__(self):
        self.mean_: Optional[float] = None
        self.std_: Optional[float] = None

    def fit(self, x: np.ndarray, weights: Optional[np.ndarray] = None) -> "Standardizer":
        """
        Compute (weighted) mean and standard deviation.

        Args:
            x: Data array
            weights: Sample weights (optional)

        Returns:
            self
        """
        if weights is None:
            weights = np.ones_like(x)

        # Normalize weights
        w = weights / weights.sum()

        # Weighted mean
        self.mean_ = np.sum(w * x)

        # Weighted variance
        variance = np.sum(w * (x - self.mean_) ** 2)
        self.std_ = np.sqrt(variance)

        # Avoid division by zero
        if self.std_ < 1e-8:
            self.std_ = 1.0

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Standardize: (x - mean) / std."""
        if self.mean_ is None:
            raise ValueError("Standardizer not fitted. Call fit() first.")
        return (x - self.mean_) / self.std_

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Inverse standardize: y * std + mean."""
        if self.mean_ is None:
            raise ValueError("Standardizer not fitted. Call fit() first.")
        return y * self.std_ + self.mean_


class VariableTransformer:
    """
    Complete transformation pipeline for a single variable.

    Combines zero-inflation handling, log transform, and standardization.
    Designed for variables that are:
    - Often zero (zero_inflated=True)
    - Heavy-tailed when positive (log_transform=True)
    - Need standardization for neural network training
    """

    def __init__(
        self,
        zero_inflated: bool = True,
        log_transform: bool = True,
        standardize: bool = True,
    ):
        """
        Initialize transformer.

        Args:
            zero_inflated: Whether to handle zeros separately
            log_transform: Whether to apply log transform to positive values
            standardize: Whether to standardize the result
        """
        self.zero_inflated = zero_inflated
        self.log_transform = log_transform
        self.standardize = standardize

        self._zero_transform = ZeroInflatedTransform() if zero_inflated else None
        self._log_transform = LogTransform(offset=1.0) if log_transform else None
        self._standardizer = Standardizer() if standardize else None

        self._is_fitted = False

    def fit(
        self, x: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> "VariableTransformer":
        """
        Fit the transformer on training data.

        Args:
            x: Data array
            weights: Sample weights

        Returns:
            self
        """
        if weights is None:
            weights = np.ones_like(x, dtype=np.float64)

        # Get positive values for fitting standardizer
        if self.zero_inflated:
            indicator, positive_values = self._zero_transform.split(x)
            positive_weights = weights[x > 0]
        else:
            positive_values = x
            positive_weights = weights

        # Apply log transform
        if self.log_transform and len(positive_values) > 0:
            transformed = self._log_transform.forward(positive_values)
        else:
            transformed = positive_values

        # Fit standardizer
        if self.standardize and len(transformed) > 0:
            self._standardizer.fit(transformed, positive_weights)

        self._is_fitted = True
        return self

    def transform(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Transform data.

        For zero-inflated variables, returns NaN at zero positions
        (to distinguish from transformed values that equal 0).

        Args:
            x: Data array or tensor

        Returns:
            Transformed data
        """
        if not self._is_fitted:
            raise ValueError("Transformer not fitted. Call fit() first.")

        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x_np = x.numpy()
        else:
            x_np = x

        # Use NaN for zeros to distinguish from transformed values that equal 0
        result = np.full_like(x_np, np.nan, dtype=np.float64)
        positive_mask = x_np > 0

        if positive_mask.any():
            positive_values = x_np[positive_mask].astype(np.float64)

            # Log transform
            if self.log_transform:
                positive_values = self._log_transform.forward(positive_values)

            # Standardize
            if self.standardize:
                positive_values = self._standardizer.transform(positive_values)

            result[positive_mask] = positive_values

        if is_tensor:
            return torch.tensor(result, dtype=torch.float32)
        return result

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse transform data.

        Args:
            y: Transformed data (NaN indicates original zeros)

        Returns:
            Original scale data
        """
        if not self._is_fitted:
            raise ValueError("Transformer not fitted. Call fit() first.")

        result = np.zeros_like(y, dtype=np.float64)

        # Positive mask: non-NaN values were originally positive
        positive_mask = ~np.isnan(y)

        if positive_mask.any():
            positive_values = y[positive_mask].copy()

            # Inverse standardize
            if self.standardize:
                positive_values = self._standardizer.inverse_transform(positive_values)

            # Inverse log transform
            if self.log_transform:
                positive_values = self._log_transform.inverse(positive_values)

            result[positive_mask] = positive_values

        return result


class MultiVariableTransformer:
    """Transform multiple variables at once.

    Fits separate transformers for each variable, enabling
    independent handling of different distribution types.
    """

    def __init__(
        self,
        var_names: list,
        zero_inflated: bool = True,
        log_transform: bool = True,
    ):
        """
        Initialize multi-variable transformer.

        Args:
            var_names: List of variable names to transform
            zero_inflated: Whether variables are zero-inflated
            log_transform: Whether to apply log transform
        """
        self.var_names = var_names
        self.zero_inflated = zero_inflated
        self.log_transform = log_transform
        self.transformers_: Dict[str, VariableTransformer] = {}

    def fit(
        self, data: Dict[str, np.ndarray], weight_col: str = "weight"
    ) -> "MultiVariableTransformer":
        """
        Fit transformers for all variables.

        Args:
            data: Dict with variable names as keys, arrays as values
            weight_col: Name of weight column (optional)

        Returns:
            self
        """
        weights = data.get(weight_col, None)

        for var_name in self.var_names:
            transformer = VariableTransformer(
                zero_inflated=self.zero_inflated,
                log_transform=self.log_transform,
                standardize=True,
            )
            transformer.fit(data[var_name], weights)
            self.transformers_[var_name] = transformer

        return self

    def transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Transform all variables.

        Args:
            data: Dict with variable arrays

        Returns:
            Dict with transformed arrays
        """
        result = {}
        for var_name in self.var_names:
            result[var_name] = self.transformers_[var_name].transform(data[var_name])
        return result

    def inverse_transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Inverse transform all variables.

        Args:
            data: Dict with transformed arrays

        Returns:
            Dict with original scale arrays
        """
        result = {}
        for var_name in self.var_names:
            result[var_name] = self.transformers_[var_name].inverse_transform(
                data[var_name]
            )
        return result
