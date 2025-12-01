#!/usr/bin/env python3
"""
Example script demonstrating KNN imputation for Multivariate Time Series.

This script shows how to:
1. Generate sample MTS data
2. Introduce missing values
3. Apply KNN imputation
4. Evaluate imputation quality
5. Visualize results
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from knn_imputer import KNNTimeSeriesImputer
from data_utils import (
    generate_sample_mts,
    introduce_missing_values,
    calculate_imputation_error,
    normalize_data,
    denormalize_data,
)


def main():
    """Run the KNN imputation example."""
    print("=" * 60)
    print("KNN Imputation for Multivariate Time Series - Example")
    print("=" * 60)

    # Generate sample data
    print("\n1. Generating sample MTS data...")
    n_timesteps = 100
    n_features = 4
    data = generate_sample_mts(
        n_timesteps=n_timesteps, n_features=n_features, noise_level=0.1, seed=42
    )
    print(f"   Generated data shape: {data.shape}")
    print(f"   Data range: [{data.min():.3f}, {data.max():.3f}]")

    # Normalize data
    print("\n2. Normalizing data...")
    data_normalized, norm_params = normalize_data(data, method="minmax")
    print(f"   Normalized data range: [{data_normalized.min():.3f}, {data_normalized.max():.3f}]")

    # Introduce missing values
    print("\n3. Introducing missing values...")
    missing_rates = [0.1, 0.2, 0.3]

    for missing_rate in missing_rates:
        print(f"\n   --- Missing rate: {missing_rate * 100:.0f}% ---")

        data_missing, mask = introduce_missing_values(
            data_normalized, missing_rate=missing_rate, missing_type="random", seed=42
        )
        n_missing = np.sum(~mask)
        print(f"   Actual missing values: {n_missing} ({n_missing / mask.size * 100:.1f}%)")

        # Apply KNN imputation
        print("\n4. Applying KNN imputation...")
        imputer = KNNTimeSeriesImputer(n_neighbors=5, window_size=3, weights="distance")
        data_imputed = imputer.fit_transform(data_missing)
        print("   Imputation complete!")

        # Calculate error metrics
        print("\n5. Evaluating imputation quality...")
        metrics = calculate_imputation_error(data_normalized, data_imputed, mask)
        print(f"   RMSE: {metrics['rmse']:.4f}")
        print(f"   MAE:  {metrics['mae']:.4f}")
        if not np.isnan(metrics["mape"]):
            print(f"   MAPE: {metrics['mape']:.2f}%")

    # Compare different KNN configurations
    print("\n" + "=" * 60)
    print("Comparing different KNN configurations")
    print("=" * 60)

    data_missing, mask = introduce_missing_values(
        data_normalized, missing_rate=0.15, missing_type="random", seed=123
    )

    configurations = [
        {"n_neighbors": 3, "window_size": 1},
        {"n_neighbors": 5, "window_size": 3},
        {"n_neighbors": 7, "window_size": 5},
        {"n_neighbors": 10, "window_size": 7},
    ]

    print("\nConfiguration Results:")
    print("-" * 50)
    print(f"{'K':<5} {'Window':<8} {'RMSE':<10} {'MAE':<10}")
    print("-" * 50)

    for config in configurations:
        imputer = KNNTimeSeriesImputer(**config)
        imputed = imputer.fit_transform(data_missing)
        metrics = calculate_imputation_error(data_normalized, imputed, mask)
        print(
            f"{config['n_neighbors']:<5} {config['window_size']:<8} "
            f"{metrics['rmse']:<10.4f} {metrics['mae']:<10.4f}"
        )

    # Denormalize and show final result
    print("\n" + "=" * 60)
    print("Final Result (Best Configuration)")
    print("=" * 60)

    best_imputer = KNNTimeSeriesImputer(n_neighbors=5, window_size=3)
    data_imputed_normalized = best_imputer.fit_transform(data_missing)
    data_imputed_final = denormalize_data(data_imputed_normalized, norm_params)

    print(f"\nOriginal data shape: {data.shape}")
    print(f"Imputed data shape:  {data_imputed_final.shape}")
    print(f"Original range:      [{data.min():.3f}, {data.max():.3f}]")
    print(f"Imputed range:       [{data_imputed_final.min():.3f}, {data_imputed_final.max():.3f}]")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
