"""
Tests for KNN Time Series Imputer
"""

import numpy as np
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knn_imputer import KNNTimeSeriesImputer
from data_utils import (
    generate_sample_mts,
    introduce_missing_values,
    calculate_imputation_error,
    normalize_data,
    denormalize_data,
)


class TestKNNTimeSeriesImputer:
    """Test cases for KNNTimeSeriesImputer class."""

    def test_initialization(self):
        """Test that imputer initializes with correct parameters."""
        imputer = KNNTimeSeriesImputer(n_neighbors=3, window_size=5)
        assert imputer.n_neighbors == 3
        assert imputer.window_size == 5
        assert imputer.weights == "distance"
        assert imputer.metric == "euclidean"

    def test_fit(self):
        """Test that fit method works correctly."""
        imputer = KNNTimeSeriesImputer()
        data = np.array([[1, 2], [3, 4], [5, 6]])
        imputer.fit(data)
        assert imputer._fitted is True
        assert imputer._training_data is not None

    def test_transform_without_fit_raises_error(self):
        """Test that transform without fit raises ValueError."""
        imputer = KNNTimeSeriesImputer()
        data = np.array([[1, 2], [np.nan, 4], [5, 6]])
        with pytest.raises(ValueError):
            imputer.transform(data)

    def test_fit_transform(self):
        """Test fit_transform method."""
        imputer = KNNTimeSeriesImputer(n_neighbors=2)
        data = np.array(
            [[1.0, 2.0], [np.nan, 3.0], [3.0, 4.0], [4.0, np.nan], [5.0, 6.0]]
        )
        result = imputer.fit_transform(data)

        # Check that no NaN values remain
        assert not np.any(np.isnan(result))

        # Check shape is preserved
        assert result.shape == data.shape

    def test_imputation_accuracy(self):
        """Test that imputation produces reasonable results."""
        # Create simple data with known pattern
        np.random.seed(42)
        data = generate_sample_mts(n_timesteps=50, n_features=3, seed=42)

        # Introduce missing values
        data_missing, mask = introduce_missing_values(data, missing_rate=0.1, seed=42)

        # Impute
        imputer = KNNTimeSeriesImputer(n_neighbors=5, window_size=3)
        imputed = imputer.fit_transform(data_missing)

        # Calculate error
        metrics = calculate_imputation_error(data, imputed, mask)

        # RMSE should be reasonable (less than data range)
        data_range = np.max(data) - np.min(data)
        assert metrics["rmse"] < data_range

    def test_no_missing_values(self):
        """Test behavior when there are no missing values."""
        imputer = KNNTimeSeriesImputer()
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = imputer.fit_transform(data)

        np.testing.assert_array_almost_equal(result, data)

    def test_get_params(self):
        """Test get_params method."""
        imputer = KNNTimeSeriesImputer(n_neighbors=7, window_size=9)
        params = imputer.get_params()

        assert params["n_neighbors"] == 7
        assert params["window_size"] == 9

    def test_set_params(self):
        """Test set_params method."""
        imputer = KNNTimeSeriesImputer()
        imputer.set_params(n_neighbors=10, window_size=7)

        assert imputer.n_neighbors == 10
        assert imputer.window_size == 7


class TestDataUtils:
    """Test cases for data utility functions."""

    def test_generate_sample_mts_shape(self):
        """Test that generated data has correct shape."""
        data = generate_sample_mts(n_timesteps=100, n_features=5)
        assert data.shape == (100, 5)

    def test_generate_sample_mts_reproducibility(self):
        """Test that seed produces reproducible results."""
        data1 = generate_sample_mts(seed=42)
        data2 = generate_sample_mts(seed=42)
        np.testing.assert_array_equal(data1, data2)

    def test_introduce_missing_random(self):
        """Test random missing value introduction."""
        data = generate_sample_mts(n_timesteps=100, n_features=3, seed=42)
        data_missing, mask = introduce_missing_values(
            data, missing_rate=0.2, missing_type="random", seed=42
        )

        # Check that NaN values were introduced
        assert np.any(np.isnan(data_missing))

        # Check that mask correctly identifies missing values
        assert np.all(np.isnan(data_missing) == ~mask)

    def test_introduce_missing_temporal(self):
        """Test temporal block missing value introduction."""
        data = generate_sample_mts(n_timesteps=100, n_features=3, seed=42)
        data_missing, mask = introduce_missing_values(
            data, missing_rate=0.2, missing_type="temporal", seed=42
        )

        assert np.any(np.isnan(data_missing))

    def test_introduce_missing_feature(self):
        """Test feature-wise missing value introduction."""
        data = generate_sample_mts(n_timesteps=100, n_features=3, seed=42)
        data_missing, mask = introduce_missing_values(
            data, missing_rate=0.2, missing_type="feature", seed=42
        )

        assert np.any(np.isnan(data_missing))

    def test_normalize_minmax(self):
        """Test min-max normalization."""
        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        normalized, params = normalize_data(data, method="minmax")

        # Check that normalized values are between 0 and 1
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)

        # Check that params are correct
        assert params["method"] == "minmax"

    def test_normalize_zscore(self):
        """Test z-score normalization."""
        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        normalized, params = normalize_data(data, method="zscore")

        # Check that mean is approximately 0
        np.testing.assert_array_almost_equal(
            np.mean(normalized, axis=0), [0, 0], decimal=5
        )

        # Check that std is approximately 1
        np.testing.assert_array_almost_equal(
            np.std(normalized, axis=0), [1, 1], decimal=5
        )

    def test_denormalize_minmax(self):
        """Test denormalization for min-max."""
        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        normalized, params = normalize_data(data, method="minmax")
        denormalized = denormalize_data(normalized, params)

        np.testing.assert_array_almost_equal(denormalized, data)

    def test_denormalize_zscore(self):
        """Test denormalization for z-score."""
        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        normalized, params = normalize_data(data, method="zscore")
        denormalized = denormalize_data(normalized, params)

        np.testing.assert_array_almost_equal(denormalized, data)

    def test_calculate_imputation_error(self):
        """Test imputation error calculation."""
        original = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        imputed = np.array([[1.1, 2.0], [3.0, 4.2], [5.0, 6.0]])
        mask = np.array([[False, True], [True, False], [True, True]])

        metrics = calculate_imputation_error(original, imputed, mask)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "mape" in metrics
        assert metrics["rmse"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
