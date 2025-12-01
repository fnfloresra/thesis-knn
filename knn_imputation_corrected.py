"""
CORRECTED Production-Ready KNN Imputation Script with MAE, RMSE, and R² Metrics
Fixed Evaluation Logic - Proper Cross-Validation Approach

CRITICAL FIX:
The previous evaluation was flawed because it compared known values with themselves.
This corrected version properly evaluates by masking known values BEFORE imputation.

Author: Environmental Monitoring AI System
Date: November 28, 2025
Version: 2.0 (CORRECTED)
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple
import logging
import warnings
from datetime import datetime
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class KNNImputationMetrics:
    """Handle all metric calculations."""

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate MAE, RMSE, and R²."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0

        return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'n_samples': len(y_true)}

    @staticmethod
    def print_metrics_table(metrics_dict: Dict, title: str = "Metrics"):
        """Print formatted metrics table."""
        logger.info(f"\n{title}")
        logger.info("-" * 90)
        logger.info(f"{'Variable':<45} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
        logger.info("-" * 90)

        for var, m in metrics_dict.items():
            var_short = var[:44] if len(var) > 44 else var
            logger.info(
                f"{var_short:<45} "
                f"{m['MAE']:<12.6f} "
                f"{m['RMSE']:<12.6f} "
                f"{m['R2']:<12.6f}"
            )

    @staticmethod
    def aggregate_metrics(metrics_dict: Dict) -> Dict[str, float]:
        """Calculate average metrics."""
        if not metrics_dict:
            return {}

        mae_values = [m['MAE'] for m in metrics_dict.values()]
        rmse_values = [m['RMSE'] for m in metrics_dict.values()]
        r2_values = [m['R2'] for m in metrics_dict.values()]

        return {
            'Avg_MAE': np.mean(mae_values),
            'Std_MAE': np.std(mae_values),
            'Avg_RMSE': np.mean(rmse_values),
            'Std_RMSE': np.std(rmse_values),
            'Avg_R2': np.mean(r2_values),
            'Std_R2': np.std(r2_values),
            'Min_R2': np.min(r2_values),
            'Max_R2': np.max(r2_values)
        }


class CorrectedKNNImputer:
    """KNN Imputation with CORRECTED evaluation methodology."""

    def __init__(self, file_paths: List[str], n_neighbors: int = 10):
        """Initialize imputer."""
        self.file_paths = file_paths
        self.n_neighbors = n_neighbors

    def _extract_site_id(self, file_path: str) -> str:
        """Extract site ID from filename."""
        match = re.search(r'station[_\-]?(\d+)', file_path)
        return match.group(1) if match else file_path.split('/')[-1].split('.')[0]

    def proper_cross_validation_evaluation(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str],
        mask_ratio: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """
        CORRECTED Evaluation: Mask known values BEFORE imputation.

        This approach:
        1. Creates a copy of data with KNOWN values artificially masked
        2. Runs KNN imputation to fill ALL missing values (real + artificial)
        3. Compares imputed artificial values with original known values
        4. This realistically tests KNN's ability to predict values

        Args:
            df: DataFrame with original data
            feature_cols: Feature columns to evaluate
            mask_ratio: Proportion of known values to mask
            random_state: For reproducibility

        Returns:
            Dictionary of metrics per variable
        """
        logger.info("\nCORRECTED Evaluation: Cross-Validation with Artificial Masking")
        logger.info("-" * 90)
        logger.info("Methodology: Mask 20% of KNOWN values before imputation")
        logger.info("Purpose: Test if KNN can predict values it hasn't seen")
        logger.info("-" * 90)

        metrics = {}
        np.random.seed(random_state)

        for col in feature_cols:
            # Get all non-missing values
            known_mask = df[col].notna()
            known_indices = df[known_mask].index.tolist()

            if len(known_indices) < 10:
                logger.debug(f"  Skipping {col}: insufficient data ({len(known_indices)} values)")
                continue

            # Randomly select some to mask (simulate as missing)
            n_mask = max(5, int(len(known_indices) * mask_ratio))
            n_mask = min(n_mask, len(known_indices))  # Ensure we don't exceed available data
            mask_indices = np.random.choice(known_indices, size=n_mask, replace=False)

            # Create copy with artificially masked values
            df_masked = df.copy()
            df_masked.loc[mask_indices, col] = np.nan  # Artificially mask these

            # Store original values we masked
            y_true = df.loc[mask_indices, col].values

            # NOW impute with masked values (use all features for better accuracy)
            imputer = KNNImputer(n_neighbors=self.n_neighbors, metric='nan_euclidean')
            imputed_data = imputer.fit_transform(df_masked[feature_cols])

            # Get imputed values for our masked indices from the specific column
            col_idx = feature_cols.index(col)
            y_pred = imputed_data[mask_indices, col_idx]

            # Calculate metrics
            metrics[col] = KNNImputationMetrics.calculate_metrics(y_true, y_pred)

        return metrics

    def run_individual_site(self, file_path: str) -> Tuple[pd.DataFrame, Dict]:
        """Process individual site."""
        site_id = self._extract_site_id(file_path)
        logger.info(f"\nIndividual Site: {site_id}")
        logger.info("-" * 90)

        # Load data
        df = pd.read_csv(file_path)
        # Exclude date/time columns
        exclude_cols = ['week', 'Week', 'Date', 'DATE', 'date']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if not feature_cols:
            logger.error(f"  No feature columns found in {file_path}")
            return pd.DataFrame(), {}

        missing_pct = (df[feature_cols].isnull().sum().sum() / (df.shape[0] * len(feature_cols))) * 100
        logger.info(f"  Data: {len(df)} records, {len(feature_cols)} variables, {missing_pct:.1f}% missing")

        # Perform cross-validation evaluation BEFORE imputation
        cv_metrics = self.proper_cross_validation_evaluation(df, feature_cols)

        # Now impute the real data for output
        imputer = KNNImputer(n_neighbors=self.n_neighbors, metric='nan_euclidean')
        df_imputed = df.copy()
        df_imputed[feature_cols] = imputer.fit_transform(df[feature_cols])

        # Report metrics
        agg = KNNImputationMetrics.aggregate_metrics(cv_metrics)
        logger.info(
            f"  Metrics: MAE={agg['Avg_MAE']:.6f}±{agg['Std_MAE']:.6f}, "
            f"RMSE={agg['Avg_RMSE']:.6f}±{agg['Std_RMSE']:.6f}, "
            f"R²={agg['Avg_R2']:.4f}±{agg['Std_R2']:.4f}"
        )

        return df_imputed, cv_metrics

    def run(self, output_dir: str = ".") -> Dict:
        """Run for all sites."""
        logger.info("\n" + "="*90)
        logger.info("CORRECTED KNN IMPUTATION WITH PROPER CROSS-VALIDATION EVALUATION")
        logger.info("="*90)
        logger.info(f"Configuration: n_neighbors={self.n_neighbors}")
        logger.info(f"Evaluation Method: Artificial masking of 20% known values")

        all_results = {
            'imputed_data': {},
            'metrics': {},
            'aggregate_metrics': {}
        }

        for file_path in self.file_paths:
            try:
                site_id = self._extract_site_id(file_path)
                df_imputed, metrics = self.run_individual_site(file_path)
                
                if df_imputed.empty:
                    logger.warning(f"  Skipping site {site_id} due to errors")
                    continue
            except FileNotFoundError:
                logger.error(f"  File not found: {file_path}")
                continue
            except Exception as e:
                logger.error(f"  Error processing {file_path}: {e}")
                continue

            all_results['imputed_data'][site_id] = df_imputed
            all_results['metrics'][site_id] = metrics
            all_results['aggregate_metrics'][site_id] = (
                KNNImputationMetrics.aggregate_metrics(metrics)
            )

            # Save imputed data
            output_file = f"{output_dir}/imputed_station_{site_id}_corrected_knn.csv"
            df_imputed.to_csv(output_file, index=False)
            logger.info(f"  Saved: {output_file}")

        return all_results


def main():
    """Main execution."""
    start_time = datetime.now()

    logger.info("\n╔" + "="*88 + "╗")
    logger.info("║" + " "*88 + "║")
    logger.info("║  CORRECTED KNN IMPUTATION - PROPER EVALUATION METHODOLOGY".center(88) + "║")
    logger.info("║" + " "*88 + "║")
    logger.info("╚" + "="*88 + "╝")

    file_paths = [
        'scaled_reindexed_station_279_weekly_limited (1).csv',
        'scaled_reindexed_station_280_weekly_limited (1).csv',
        'scaled_reindexed_station_281_weekly_limited (1).csv',
        'scaled_reindexed_station_282_weekly_limited (1).csv',
        'scaled_reindexed_station_283_weekly_limited (1).csv'
    ]

    imputer = CorrectedKNNImputer(file_paths, n_neighbors=5)
    results = imputer.run()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("\n" + "="*90)
    logger.info("✓ EXECUTION COMPLETE")
    logger.info("="*90)
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info(f"Sites processed: {len(results['imputed_data'])}")

    # Summary
    logger.info("\n" + "="*90)
    logger.info("SUMMARY OF RESULTS")
    logger.info("="*90)

    for site, agg in results['aggregate_metrics'].items():
        logger.info(f"\nSite {site}:")
        logger.info(f"  MAE:  {agg['Avg_MAE']:.6f} ± {agg['Std_MAE']:.6f}")
        logger.info(f"  RMSE: {agg['Avg_RMSE']:.6f} ± {agg['Std_RMSE']:.6f}")
        logger.info(f"  R²:   {agg['Avg_R2']:.4f} ± {agg['Std_R2']:.4f}")
        logger.info(f"         (min: {agg['Min_R2']:.4f}, max: {agg['Max_R2']:.4f})")

    return results


if __name__ == "__main__":
    results = main()
