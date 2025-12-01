"""
Individual Site KNN Imputation Script
Simple approach for imputing each site independently

Author: Environmental Monitoring AI System
Date: November 28, 2025
Version: 1.0
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import warnings
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


def evaluate_imputation(
    df: pd.DataFrame,
    feature_cols: list,
    n_neighbors: int = 5,
    mask_rate: float = 0.1,
    random_seed: int = 42
) -> dict:
    """
    Evaluate imputation quality by masking some known values.
    
    Args:
        df: Original DataFrame
        feature_cols: List of feature columns to evaluate
        n_neighbors: Number of neighbors for KNN
        mask_rate: Proportion of complete values to mask for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with MAE, RMSE, and R2 metrics
    """
    np.random.seed(random_seed)
    
    # Find rows with complete data
    complete_mask = df[feature_cols].notna().all(axis=1)
    complete_indices = df[complete_mask].index
    
    if len(complete_indices) == 0:
        logger.warning("  No complete rows found for evaluation")
        return {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'masked_values': 0}
    
    # Select rows to mask
    n_mask = max(1, int(len(complete_indices) * mask_rate))
    mask_indices = np.random.choice(complete_indices, size=n_mask, replace=False)
    
    # Create masked dataset
    df_masked = df.copy()
    ground_truth = []
    
    for idx in mask_indices:
        # Mask 30% of features in each selected row
        n_features_to_mask = max(1, int(len(feature_cols) * 0.3))
        cols_to_mask = np.random.choice(feature_cols, size=n_features_to_mask, replace=False)
        
        for col in cols_to_mask:
            ground_truth.append({
                'value': df_masked.loc[idx, col],
                'index': idx,
                'column': col
            })
            df_masked.loc[idx, col] = np.nan
    
    # Perform imputation on masked data
    imputer = KNNImputer(n_neighbors=n_neighbors, metric='nan_euclidean')
    df_imputed = df_masked.copy()
    df_imputed[feature_cols] = imputer.fit_transform(df_masked[feature_cols])
    
    # Calculate metrics
    y_true = []
    y_pred = []
    
    for item in ground_truth:
        y_true.append(item['value'])
        y_pred.append(df_imputed.loc[item['index'], item['column']])
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    try:
        r2 = r2_score(y_true, y_pred)
    except ValueError:
        r2 = np.nan
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'masked_values': len(ground_truth)
    }


def impute_single_site(
    file_path: str,
    n_neighbors: int = 5,
    output_suffix: str = '_individual',
    evaluate: bool = True
) -> tuple:
    """
    Perform KNN imputation on a single site.

    Args:
        file_path: Path to CSV file
        n_neighbors: Number of neighbors for KNN
        output_suffix: Suffix for output filename
        evaluate: Whether to evaluate imputation quality

    Returns:
        Tuple of (imputed DataFrame, metrics dictionary)
    """
    logger.info(f"Processing: {file_path}")

    # Load data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise

    # Identify feature columns (exclude Date column)
    feature_cols = [col for col in df.columns if col not in ['Date', 'week']]

    # Check missing data
    missing_before = df[feature_cols].isnull().sum().sum()
    total_values = df[feature_cols].size
    missing_pct = (missing_before / total_values) * 100

    logger.info(f"  Missing: {missing_before} values ({missing_pct:.2f}%)")

    # Evaluate imputation quality if requested
    metrics = None
    if evaluate:
        logger.info("  Evaluating imputation quality...")
        metrics = evaluate_imputation(df, feature_cols, n_neighbors)
        logger.info(f"  MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R2']:.4f}")

    # Perform imputation
    imputer = KNNImputer(n_neighbors=n_neighbors, metric='nan_euclidean')
    df_imputed = df.copy()
    df_imputed[feature_cols] = imputer.fit_transform(df[feature_cols])

    # Verify no missing values remain
    missing_after = df_imputed[feature_cols].isnull().sum().sum()
    logger.info(f"  After imputation: {missing_after} missing values")

    # Save result
    output_file = file_path.replace('.csv', f'{output_suffix}.csv')
    df_imputed.to_csv(output_file, index=False)
    logger.info(f"  ✓ Saved: {output_file}\n")

    return df_imputed, metrics


def main():
    """Main execution function."""

    start_time = datetime.now()

    logger.info("="*70)
    logger.info("INDIVIDUAL SITE KNN IMPUTATION")
    logger.info("="*70)
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Configuration
    file_paths = [
        'scaled_reindexed_station_279_weekly_limited (1).csv',
        'scaled_reindexed_station_280_weekly_limited (1).csv',
        'scaled_reindexed_station_281_weekly_limited (1).csv',
        'scaled_reindexed_station_282_weekly_limited (1).csv',
        'scaled_reindexed_station_283_weekly_limited (1).csv'
    ]

    n_neighbors = 5

    # Process each site independently
    results = {}
    all_metrics = {}
    
    for file_path in file_paths:
        try:
            df_imputed, metrics = impute_single_site(
                file_path=file_path,
                n_neighbors=n_neighbors,
                output_suffix='_individual',
                evaluate=True
            )
            # Extract station number correctly (e.g., '279' from 'station_279')
            site_id = file_path.split('_')[3]
            results[site_id] = df_imputed
            all_metrics[site_id] = metrics
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Display metrics summary
    logger.info("="*70)
    logger.info("EVALUATION METRICS SUMMARY")
    logger.info("="*70)
    
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.index.name = 'Station'
    logger.info("\n" + metrics_df.to_string())
    
    # Calculate average metrics
    if len(all_metrics) > 0:
        avg_mae = metrics_df['MAE'].mean()
        avg_rmse = metrics_df['RMSE'].mean()
        avg_r2 = metrics_df['R2'].mean()
        logger.info("\nAverage Metrics:")
        logger.info(f"  MAE:  {avg_mae:.4f}")
        logger.info(f"  RMSE: {avg_rmse:.4f}")
        logger.info(f"  R²:   {avg_r2:.4f}")
    
    # Save metrics to CSV
    metrics_output = 'individual_imputation_metrics.csv'
    metrics_df.to_csv(metrics_output)
    logger.info(f"\n✓ Metrics saved to: {metrics_output}")

    logger.info("\n" + "="*70)
    logger.info("✓ ALL SITES PROCESSED")
    logger.info("="*70)
    logger.info(f"Total sites: {len(results)}")
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    return results, all_metrics


if __name__ == "__main__":
    results = main()
