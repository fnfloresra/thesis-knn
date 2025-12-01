"""
Data utilities for Multivariate Time Series (MTS)

This module provides utility functions for generating sample MTS data
and introducing missing values for testing imputation methods.
"""

import numpy as np
from typing import Tuple, Optional

# Small constant to avoid division by zero
EPSILON = 1e-10


def generate_sample_mts(
    n_timesteps: int = 100,
    n_features: int = 3,
    noise_level: float = 0.1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a sample multivariate time series with correlated features.

    Parameters
    ----------
    n_timesteps : int, default=100
        Number of time steps in the series.
    n_features : int, default=3
        Number of features (variables) in the time series.
    noise_level : float, default=0.1
        Standard deviation of noise to add.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    data : np.ndarray
        Generated MTS data of shape (n_timesteps, n_features).

    Examples
    --------
    >>> data = generate_sample_mts(n_timesteps=50, n_features=4, seed=42)
    >>> data.shape
    (50, 4)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate time axis
    t = np.linspace(0, 4 * np.pi, n_timesteps)

    # Generate base signals with different frequencies
    data = np.zeros((n_timesteps, n_features))

    for i in range(n_features):
        # Create correlated signals with different phases and frequencies
        freq = 1 + i * 0.3
        phase = i * np.pi / n_features
        data[:, i] = np.sin(freq * t + phase) + 0.5 * np.cos(2 * freq * t)

        # Add some trend
        data[:, i] += 0.01 * t * (i + 1)

        # Add noise
        data[:, i] += noise_level * np.random.randn(n_timesteps)

    return data


def introduce_missing_values(
    data: np.ndarray,
    missing_rate: float = 0.1,
    missing_type: str = "random",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Introduce missing values into a dataset.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_timesteps, n_features).
    missing_rate : float, default=0.1
        Proportion of values to make missing (0 to 1).
    missing_type : str, default='random'
        Type of missing pattern:
        - 'random': Missing completely at random (MCAR)
        - 'temporal': Missing in temporal blocks
        - 'feature': Missing entire features for some time periods
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    data_missing : np.ndarray
        Data with missing values (NaN).
    mask : np.ndarray
        Boolean mask where True indicates original values (False = missing).

    Examples
    --------
    >>> data = generate_sample_mts(n_timesteps=50, n_features=3, seed=42)
    >>> data_missing, mask = introduce_missing_values(data, missing_rate=0.2, seed=42)
    >>> np.sum(~mask) / mask.size  # Approximately 0.2
    """
    if seed is not None:
        np.random.seed(seed)

    data = np.asarray(data, dtype=np.float64)
    n_timesteps, n_features = data.shape
    data_missing = data.copy()

    if missing_type == "random":
        # Missing completely at random
        mask = np.random.random(data.shape) > missing_rate

    elif missing_type == "temporal":
        # Missing in temporal blocks
        mask = np.ones(data.shape, dtype=bool)
        n_blocks = max(1, int(n_timesteps * missing_rate / 5))

        for _ in range(n_blocks):
            start = np.random.randint(0, n_timesteps - 5)
            length = np.random.randint(3, min(10, n_timesteps - start))
            feature = np.random.randint(0, n_features)
            mask[start : start + length, feature] = False

    elif missing_type == "feature":
        # Missing entire features for some time periods
        mask = np.ones(data.shape, dtype=bool)
        n_periods = max(1, int(n_timesteps * missing_rate / 10))

        for _ in range(n_periods):
            start = np.random.randint(0, n_timesteps - 10)
            length = np.random.randint(5, min(20, n_timesteps - start))
            mask[start : start + length, :] = False

    else:
        raise ValueError(f"Unknown missing_type: {missing_type}")

    # Apply mask
    data_missing[~mask] = np.nan

    return data_missing, mask


def calculate_imputation_error(
    original: np.ndarray,
    imputed: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """
    Calculate imputation error metrics.

    Parameters
    ----------
    original : np.ndarray
        Original complete data.
    imputed : np.ndarray
        Imputed data.
    mask : np.ndarray
        Boolean mask where False indicates missing values.

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'rmse': Root Mean Squared Error on imputed values
        - 'mae': Mean Absolute Error on imputed values
        - 'mape': Mean Absolute Percentage Error on imputed values
    """
    # Get only the values that were imputed
    missing_mask = ~mask
    original_missing = original[missing_mask]
    imputed_missing = imputed[missing_mask]

    # Calculate metrics
    errors = original_missing - imputed_missing

    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))

    # MAPE (avoid division by zero using EPSILON constant)
    nonzero_mask = np.abs(original_missing) > EPSILON
    if np.any(nonzero_mask):
        mape = np.mean(
            np.abs(errors[nonzero_mask] / original_missing[nonzero_mask])
        ) * 100
    else:
        mape = np.nan

    return {"rmse": rmse, "mae": mae, "mape": mape}


def normalize_data(
    data: np.ndarray, method: str = "minmax"
) -> Tuple[np.ndarray, dict]:
    """
    Normalize the data.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_timesteps, n_features).
    method : str, default='minmax'
        Normalization method: 'minmax' or 'zscore'.

    Returns
    -------
    normalized_data : np.ndarray
        Normalized data.
    params : dict
        Parameters used for normalization (for inverse transform).
    """
    data = np.asarray(data, dtype=np.float64)

    if method == "minmax":
        min_vals = np.nanmin(data, axis=0)
        max_vals = np.nanmax(data, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero

        normalized_data = (data - min_vals) / range_vals
        params = {"method": "minmax", "min": min_vals, "max": max_vals}

    elif method == "zscore":
        mean_vals = np.nanmean(data, axis=0)
        std_vals = np.nanstd(data, axis=0)
        std_vals[std_vals == 0] = 1  # Avoid division by zero

        normalized_data = (data - mean_vals) / std_vals
        params = {"method": "zscore", "mean": mean_vals, "std": std_vals}

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized_data, params


def denormalize_data(data: np.ndarray, params: dict) -> np.ndarray:
    """
    Denormalize the data using saved parameters.

    Parameters
    ----------
    data : np.ndarray
        Normalized data.
    params : dict
        Parameters from normalize_data.

    Returns
    -------
    denormalized_data : np.ndarray
        Denormalized data.
    """
    data = np.asarray(data, dtype=np.float64)

    if params["method"] == "minmax":
        range_vals = params["max"] - params["min"]
        denormalized_data = data * range_vals + params["min"]

    elif params["method"] == "zscore":
        denormalized_data = data * params["std"] + params["mean"]

    else:
        raise ValueError(f"Unknown normalization method: {params['method']}")

    return denormalized_data
