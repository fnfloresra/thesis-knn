# KNN Imputation for Multivariate Time Series (MTS)

This repository implements a K-Nearest Neighbors (KNN) approach for imputing missing values in Multivariate Time Series data, developed as part of a thesis project.

## Overview

Missing data is a common challenge in time series analysis. This project provides a specialized KNN imputation method that considers temporal relationships in the data through a sliding window approach, making it particularly suitable for multivariate time series.

## Features

- **KNN-based imputation** with configurable number of neighbors
- **Temporal windowing** to capture local patterns in time series
- **Multiple weight schemes** (uniform, distance-based)
- **Support for different missing patterns** (random, temporal blocks, feature-wise)
- **Data normalization utilities** (min-max, z-score)
- **Comprehensive evaluation metrics** (RMSE, MAE, MAPE)

## Installation

```bash
# Clone the repository
git clone https://github.com/fnfloresra/thesis-knn.git
cd thesis-knn

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from src.knn_imputer import KNNTimeSeriesImputer
from src.data_utils import generate_sample_mts, introduce_missing_values

# Generate sample data
data = generate_sample_mts(n_timesteps=100, n_features=4, seed=42)

# Introduce missing values (10% missing)
data_missing, mask = introduce_missing_values(data, missing_rate=0.1, seed=42)

# Create and apply KNN imputer
imputer = KNNTimeSeriesImputer(n_neighbors=5, window_size=3)
data_imputed = imputer.fit_transform(data_missing)

print("Original shape:", data.shape)
print("Imputed shape:", data_imputed.shape)
print("Missing values remaining:", np.sum(np.isnan(data_imputed)))
```

## Usage

### KNNTimeSeriesImputer

The main class for KNN-based imputation:

```python
from src.knn_imputer import KNNTimeSeriesImputer

imputer = KNNTimeSeriesImputer(
    n_neighbors=5,      # Number of nearest neighbors
    window_size=3,      # Temporal window size
    weights='distance', # Weight function ('uniform' or 'distance')
    metric='euclidean'  # Distance metric
)

# Fit and transform
imputed_data = imputer.fit_transform(data_with_missing)

# Or separately
imputer.fit(data_with_missing)
imputed_data = imputer.transform(data_with_missing)
```

### Data Utilities

Generate sample data and introduce missing values:

```python
from src.data_utils import (
    generate_sample_mts,
    introduce_missing_values,
    calculate_imputation_error,
    normalize_data,
    denormalize_data
)

# Generate sample MTS data
data = generate_sample_mts(
    n_timesteps=100,
    n_features=3,
    noise_level=0.1,
    seed=42
)

# Introduce missing values with different patterns
data_missing, mask = introduce_missing_values(
    data,
    missing_rate=0.2,
    missing_type='random',  # Options: 'random', 'temporal', 'feature'
    seed=42
)

# Normalize data
normalized, params = normalize_data(data, method='minmax')

# Calculate imputation errors
metrics = calculate_imputation_error(original, imputed, mask)
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Running the Example

```bash
python example.py
```

## Project Structure

```
thesis-knn/
├── README.md
├── requirements.txt
├── example.py
├── src/
│   ├── __init__.py
│   ├── knn_imputer.py    # KNN imputation implementation
│   └── data_utils.py     # Data utilities
└── tests/
    ├── __init__.py
    └── test_knn_imputer.py
```

## Algorithm Description

The KNN imputation algorithm works as follows:

1. For each missing value at position (time_idx, feature_idx):
   - Extract features from a temporal window around the missing value
   - Find K nearest neighbors from complete observations
   - Compute the imputed value as a weighted average of neighbor values

2. The temporal window captures local patterns by including features from adjacent time steps

3. Distance-based weighting gives more influence to closer neighbors

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| n_neighbors | 5 | Number of nearest neighbors to use |
| window_size | 3 | Size of temporal window around each point |
| weights | 'distance' | Weight function ('uniform' or 'distance') |
| metric | 'euclidean' | Distance metric for neighbor search |

## License

This project is part of academic research. Please contact the author for usage permissions.

## Author

Francisco Flores
