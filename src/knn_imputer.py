"""
KNN Imputer for Multivariate Time Series Data

This module implements a K-Nearest Neighbors approach for imputing
missing values in multivariate time series data.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Tuple

# Small constant to avoid division by zero
EPSILON = 1e-10


class KNNTimeSeriesImputer:
    """
    K-Nearest Neighbors imputer for Multivariate Time Series (MTS).

    This class implements KNN-based imputation that considers temporal
    relationships in time series data by using a sliding window approach.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of nearest neighbors to use for imputation.
    window_size : int, default=3
        Size of the temporal window to consider around each time point.
    weights : str, default='distance'
        Weight function used in prediction. Possible values:
        - 'uniform': All neighbors have equal weight.
        - 'distance': Weight points by the inverse of their distance.
    metric : str, default='euclidean'
        Distance metric for finding nearest neighbors.

    Attributes
    ----------
    n_neighbors : int
        Number of neighbors used for imputation.
    window_size : int
        Temporal window size.
    weights : str
        Weight function type.
    metric : str
        Distance metric used.

    Examples
    --------
    >>> import numpy as np
    >>> from knn_imputer import KNNTimeSeriesImputer
    >>> # Create sample data with missing values
    >>> data = np.array([[1, 2], [np.nan, 3], [3, 4], [4, np.nan], [5, 6]])
    >>> imputer = KNNTimeSeriesImputer(n_neighbors=2)
    >>> imputed_data = imputer.fit_transform(data)
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        window_size: int = 3,
        weights: str = "distance",
        metric: str = "euclidean",
    ):
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.weights = weights
        self.metric = metric
        self._fitted = False
        self._training_data = None

    def fit(self, X: np.ndarray) -> "KNNTimeSeriesImputer":
        """
        Fit the imputer using the provided data.

        Parameters
        ----------
        X : np.ndarray
            Input multivariate time series data of shape (n_timesteps, n_features).

        Returns
        -------
        self : KNNTimeSeriesImputer
            Returns self.
        """
        X = np.asarray(X, dtype=np.float64)
        self._training_data = X.copy()
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Impute missing values in the input data using KNN.

        Parameters
        ----------
        X : np.ndarray
            Input multivariate time series data of shape (n_timesteps, n_features)
            with missing values represented as NaN.

        Returns
        -------
        X_imputed : np.ndarray
            Data with missing values imputed.
        """
        if not self._fitted:
            raise ValueError("Imputer has not been fitted. Call 'fit' first.")

        X = np.asarray(X, dtype=np.float64)
        X_imputed = X.copy()

        # Find positions of missing values
        missing_mask = np.isnan(X_imputed)

        if not np.any(missing_mask):
            return X_imputed

        # Get indices of missing values
        missing_indices = np.argwhere(missing_mask)

        # Impute each missing value
        for idx in missing_indices:
            time_idx, feature_idx = idx
            imputed_value = self._impute_single_value(
                X_imputed, time_idx, feature_idx
            )
            X_imputed[time_idx, feature_idx] = imputed_value

        return X_imputed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the imputer and transform the data in one step.

        Parameters
        ----------
        X : np.ndarray
            Input multivariate time series data of shape (n_timesteps, n_features)
            with missing values represented as NaN.

        Returns
        -------
        X_imputed : np.ndarray
            Data with missing values imputed.
        """
        return self.fit(X).transform(X)

    def _impute_single_value(
        self, X: np.ndarray, time_idx: int, feature_idx: int
    ) -> float:
        """
        Impute a single missing value using KNN with temporal windowing.

        Parameters
        ----------
        X : np.ndarray
            The data array with some values already imputed.
        time_idx : int
            Time index of the missing value.
        feature_idx : int
            Feature index of the missing value.

        Returns
        -------
        imputed_value : float
            The imputed value.
        """
        n_timesteps = X.shape[0]

        # Define the temporal window
        half_window = self.window_size // 2
        start_idx = max(0, time_idx - half_window)
        end_idx = min(n_timesteps, time_idx + half_window + 1)

        # Extract features from the window for similarity computation
        window_features = self._extract_window_features(X, time_idx, feature_idx)

        # Find complete rows (no NaN in the target feature) for training
        candidate_indices = []
        candidate_features = []

        for t in range(n_timesteps):
            if t == time_idx:
                continue
            if not np.isnan(X[t, feature_idx]):
                candidate_feature = self._extract_window_features(X, t, feature_idx)
                # Only include candidates without NaN in their features
                if candidate_feature is not None and not np.any(np.isnan(candidate_feature)):
                    candidate_indices.append(t)
                    candidate_features.append(candidate_feature)

        if len(candidate_indices) == 0:
            # Fallback: use column mean if no candidates available
            col_values = X[:, feature_idx]
            valid_values = col_values[~np.isnan(col_values)]
            if len(valid_values) > 0:
                return np.mean(valid_values)
            return 0.0

        candidate_features = np.array(candidate_features)
        candidate_indices = np.array(candidate_indices)

        # Determine number of neighbors (can't exceed available candidates)
        effective_k = min(self.n_neighbors, len(candidate_indices))

        # Use sklearn's NearestNeighbors for efficient neighbor search
        if window_features is not None:
            # Handle case where window_features might have NaN
            if np.any(np.isnan(window_features)):
                # Use mean imputation for query features temporarily
                if len(candidate_features) > 0:
                    feature_means = np.nanmean(candidate_features, axis=0)
                    # Replace NaN values in window_features with corresponding means
                    for i in range(len(window_features)):
                        if np.isnan(window_features[i]):
                            window_features[i] = feature_means[i] if not np.isnan(feature_means[i]) else 0.0
                else:
                    window_features = np.nan_to_num(window_features, nan=0.0)

            nn = NearestNeighbors(n_neighbors=effective_k, metric=self.metric)
            nn.fit(candidate_features)
            distances, indices = nn.kneighbors(window_features.reshape(1, -1))
            distances = distances.flatten()
            indices = indices.flatten()
        else:
            # Fallback to random selection if no valid window features
            indices = np.random.choice(len(candidate_indices), size=effective_k, replace=False)
            distances = np.ones(effective_k)

        # Get the values from nearest neighbors
        neighbor_values = X[candidate_indices[indices], feature_idx]

        # Compute weighted average
        if self.weights == "distance":
            # Avoid division by zero using EPSILON constant
            weights = 1.0 / (distances + EPSILON)
            weights /= weights.sum()
            imputed_value = np.average(neighbor_values, weights=weights)
        else:
            # Uniform weights
            imputed_value = np.mean(neighbor_values)

        return imputed_value

    def _extract_window_features(
        self, X: np.ndarray, time_idx: int, exclude_feature: int
    ) -> Optional[np.ndarray]:
        """
        Extract features from a temporal window around the given time index.

        Parameters
        ----------
        X : np.ndarray
            The data array.
        time_idx : int
            Center time index for the window.
        exclude_feature : int
            Feature index to exclude (the one being imputed).

        Returns
        -------
        features : np.ndarray or None
            Flattened feature vector from the window, or None if invalid.
        """
        n_timesteps, n_features = X.shape
        half_window = self.window_size // 2

        # Collect features from the window
        features = []
        for offset in range(-half_window, half_window + 1):
            t = time_idx + offset
            if 0 <= t < n_timesteps:
                for f in range(n_features):
                    if f != exclude_feature:
                        features.append(X[t, f])
            else:
                # Pad with zeros for boundary cases
                for f in range(n_features):
                    if f != exclude_feature:
                        features.append(0.0)

        return np.array(features) if features else None

    def get_params(self) -> dict:
        """
        Get parameters of the imputer.

        Returns
        -------
        params : dict
            Dictionary of parameters.
        """
        return {
            "n_neighbors": self.n_neighbors,
            "window_size": self.window_size,
            "weights": self.weights,
            "metric": self.metric,
        }

    def set_params(self, **params) -> "KNNTimeSeriesImputer":
        """
        Set parameters of the imputer.

        Parameters
        ----------
        **params : dict
            Parameters to set.

        Returns
        -------
        self : KNNTimeSeriesImputer
            Returns self.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
