"""
Multisite KNN Imputation Pipeline for Water Quality Data
Author: AI Assistant
Date: November 21, 2025

This script implements multisite transfer learning using KNN imputation
for water quality monitoring data across multiple stations.

Required libraries:
pip install pandas numpy scikit-learn matplotlib seaborn

Usage:
python multisite_knn_pipeline.py
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
import re
from functools import partial
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input CSV files (modify paths as needed)
# Training stations (used to build the model)
TRAINING_FILES = [
    'scaled_reindexed_station_279_weekly_limited (1).csv',
    #'scaled_reindexed_station_280_weekly_limited (1).csv',
    #'scaled_reindexed_station_281_weekly_limited (1).csv',
    #'scaled_reindexed_station_282_weekly_limited (1).csv'
]

# Test station (used for evaluation only)
TEST_FILES = [
    'scaled_reindexed_station_283_weekly_limited (1).csv'
]
# Output directory
OUTPUT_DIR = './imputed_results'

# Water quality feature columns (adjust if your CSV has different column names)
FEATURE_COLS = [
    'Potencial de Hidrógeno (in situ)',
    'Sólidos Suspendidos Totales',
    'Conductividad Eléctrica',
    'Temperatura de la muestra',
    'Caudal',
    'Oxígeno disuelto (en campo)',
    'Turbiedad',
    'Cianuro Total',
    'Plomo total',
    'Cobre total',
    'Arsénico total',
    'Zinc total',
    'Cadmio total',
    'Mercurio total'
]

# KNN Hyperparameters
N_NEIGHBORS = 5
WEIGHTS = 'distance'
METRIC = 'nan_euclidean'

# Imputation strategy: 'pool' or 'transfer'
STRATEGY = 'pool'

# Evaluation parameters
ARTIFICIAL_MISSING_RATE = 0.2  # 20% of data for test station
RANDOM_SEED = 42


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Output directory created: {output_dir}")


def load_stations(file_paths):
    """Load all station CSV files"""
    stations = {}
    station_ids = []
    
    print("\n" + "="*80)
    print("LOADING STATION DATA")
    print("="*80)
    
    # Validate files exist
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"⚠ Warning: File not found: {file_path}")
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue
        try:
            # Extract station ID from filename using regex
            match = re.search(r'station_(\d+)', file_path)
            if match:
                station_id = int(match.group(1))
            else:
                # Fallback or error if pattern not found
                print(f"⚠ Could not extract station ID from {file_path}, skipping.")
                continue

            station_ids.append(station_id)
            
            # Load CSV
            df = pd.read_csv(file_path)
            
            # Replace empty strings with NaN
            df.replace('', np.nan, inplace=True)
            
            # Add station identifier
            df['station_id'] = station_id
            
            stations[station_id] = df
            
            print(f"✓ Station {station_id}: {df.shape[0]} rows × {df.shape[1]} columns")
            
        except Exception as e:
            print(f"✗ Error loading {file_path}: {e}")
    
    return stations, station_ids


def analyze_missingness(df, station_id):
    """Analyze missing data patterns per station"""
    missing_stats = pd.DataFrame({
        'feature': df.columns,
        'missing_count': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df)) * 100
    })
    
    return missing_stats


def print_missingness_summary(stations, feature_cols):
    """Print comprehensive missingness summary"""
    print("\n" + "="*80)
    print("MISSINGNESS ANALYSIS")
    print("="*80)
    
    for station_id, df in stations.items():
        missing_stats = analyze_missingness(df, station_id)
        missing_features = missing_stats[
            (missing_stats['missing_pct'] > 0) & 
            (missing_stats['feature'].isin(feature_cols))
        ].sort_values('missing_pct', ascending=False)
        
        if len(missing_features) > 0:
            print(f"\nStation {station_id}:")
            print(f"  Total missing features: {len(missing_features)}")
            print(f"  Average missing: {missing_features['missing_pct'].mean():.2f}%")
            print(f"  Top 3 features with missing data:")
            for _, row in missing_features.head(3).iterrows():
                print(f"    - {row['feature']}: {row['missing_pct']:.2f}%")
        else:
            print(f"\nStation {station_id}: No missing data ✓")


def categorize_stations(stations, feature_cols):
    """Categorize stations by completeness for transfer learning"""
    completeness = {}
    
    for station_id, df in stations.items():
        missing_stats = analyze_missingness(df, station_id)
        feature_missing = missing_stats[missing_stats['feature'].isin(feature_cols)]
        avg_missing = feature_missing['missing_pct'].mean()
        completeness[station_id] = 100 - avg_missing
    
    # Sort by completeness
    sorted_stations = sorted(completeness.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*80)
    print("STATION COMPLETENESS RANKING")
    print("="*80)
    for station_id, comp_pct in sorted_stations:
        print(f"Station {station_id}: {comp_pct:.2f}% complete")
    
    # Designate source and target stations
    source_stations = [s[0] for s in sorted_stations[:2]]
    target_stations = [s[0] for s in sorted_stations[2:]]
    
    print(f"\n→ Source stations (most complete): {source_stations}")
    print(f"→ Target stations (need transfer): {target_stations}")
    
    return source_stations, target_stations


def add_temporal_features(df):
    """Extract temporal patterns for better KNN matching"""
    df = df.copy()
    added_cols = []
    
    if 'week' in df.columns:
        df['week'] = pd.to_datetime(df['week'], errors='coerce')
        df['year'] = df['week'].dt.year
        df['month'] = df['week'].dt.month
        df['season'] = df['month'].apply(lambda x: (x % 12 + 3) // 3 if pd.notna(x) else np.nan)
        
        added_cols = ['year', 'month', 'season']
        
        # Ensure they are numeric
        for col in added_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df, added_cols


# ============================================================================
# IMPUTATION STRATEGIES
# ============================================================================

def pool_then_impute(stations, feature_cols, n_neighbors=5, weights='distance', training_only=False, training_stations=None):
    """
    Strategy 1: Pool all station data, then apply KNN imputation
    Best when stations share similar characteristics
    If training_only=True, only trains on training_stations
    """
    print("\n" + "="*80)
    print("POOL-THEN-IMPUTE STRATEGY")
    print("="*80)
    
    # Determine which stations to use for training
    if training_only and training_stations is not None:
        train_data = {sid: stations[sid] for sid in training_stations if sid in stations}
        combined_df = pd.concat(train_data.values(), axis=0, ignore_index=True)
        print(f"Training on stations: {list(train_data.keys())}")
    else:
        combined_df = pd.concat(stations.values(), axis=0, ignore_index=True)
    
    print(f"Combined dataset shape: {combined_df.shape}")
    
    # Separate features for imputation
    X_pool = combined_df[feature_cols].values
    
    print(f"Applying KNN Imputation (n_neighbors={n_neighbors}, weights={weights})...")
    
    # Apply KNN Imputation
    imputer = KNNImputer(
        n_neighbors=n_neighbors,
        weights=weights,
        metric='nan_euclidean'
    )
    
    # Fit the imputer on the combined training data
    imputer.fit(X_pool)
    
    # Now transform ALL stations (including test if present)
    all_stations_list = []
    for station_id in stations.keys():
        df_copy = stations[station_id].copy()
        df_copy['station_id'] = station_id
        all_stations_list.append(df_copy)
    
    all_combined_df = pd.concat(all_stations_list, axis=0, ignore_index=True)
    X_all = all_combined_df[feature_cols].values
    X_imputed = imputer.transform(X_all)
    
    # Reconstruct dataframe with imputed values
    all_combined_df[feature_cols] = X_imputed
    
    # Split back into individual stations, preserving original indices
    imputed_stations = {}
    for station_id in stations.keys():
        mask = all_combined_df['station_id'] == station_id
        imputed_station = all_combined_df[mask].copy()
        
        # Reset index to match the original station dataframe
        original_df = stations[station_id]
        imputed_station.index = original_df.index
        
        imputed_stations[station_id] = imputed_station
        print(f"✓ Station {station_id} imputed: {imputed_stations[station_id].shape}")
    
    return imputed_stations


def source_to_target_transfer(stations, source_ids, target_ids, feature_cols,
                               n_neighbors=5, weights='distance'):
    """
    Strategy 2: Train KNN on source stations, apply to target stations
    Best when stations have different characteristics
    """
    print("\n" + "="*80)
    print("SOURCE-TO-TARGET TRANSFER STRATEGY")
    print("="*80)
    
    # Combine source stations for training
    source_data = pd.concat([stations[sid] for sid in source_ids], axis=0)
    X_source = source_data[feature_cols].values
    
    print(f"Training on source stations {source_ids}...")
    print(f"Source data shape: {X_source.shape}")
    
    # Fit KNN on source data
    imputer = KNNImputer(
        n_neighbors=n_neighbors,
        weights=weights,
        metric='nan_euclidean'
    )
    imputer.fit(X_source)
    
    # Apply to each target station
    imputed_stations = {}
    
    print(f"\nApplying to target stations {target_ids}...")
    for target_id in target_ids:
        target_df = stations[target_id].copy()
        X_target = target_df[feature_cols].values
        X_imputed = imputer.transform(X_target)
        target_df[feature_cols] = X_imputed
        imputed_stations[target_id] = target_df
        print(f"✓ Station {target_id} imputed")
    
    # Source stations: self-impute
    print(f"\nSelf-imputing source stations {source_ids}...")
    for source_id in source_ids:
        source_df = stations[source_id].copy()
        X_source_self = source_df[feature_cols].values
        X_imputed = imputer.transform(X_source_self)
        source_df[feature_cols] = X_imputed
        imputed_stations[source_id] = source_df
        print(f"✓ Station {source_id} imputed")
    
    return imputed_stations


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def create_masked_dataset(stations, feature_cols, mask_rate=0.1, random_seed=42):
    """
    Create a copy of the dataset with artificial missing values 
    to serve as a test set for evaluation.
    """
    np.random.seed(random_seed)
    masked_stations = {}
    ground_truth = [] # List of (station_id, index, column, original_value)
    
    total_masked = 0
    
    for station_id, df in stations.items():
        # Work on a copy
        masked_df = df.copy()
        
        # Identify rows that are fully complete for the feature columns
        # We only want to mask values where we know the truth
        complete_rows_mask = masked_df[feature_cols].notna().all(axis=1)
        complete_indices = masked_df[complete_rows_mask].index
        
        if len(complete_indices) == 0:
            print(f"⚠ Station {station_id}: No complete rows found to use for evaluation.")
            masked_stations[station_id] = masked_df
            continue
            
        # Determine how many rows to touch
        n_rows_to_mask = max(1, int(len(complete_indices) * mask_rate))
        
        # Select random rows
        rows_to_mask = np.random.choice(complete_indices, size=n_rows_to_mask, replace=False)
        
        # For each selected row, mask a random subset of features (e.g., 30%)
        features_per_row = max(1, int(len(feature_cols) * 0.3))
        
        for idx in rows_to_mask:
            # Pick features to mask
            cols_to_mask = np.random.choice(feature_cols, size=features_per_row, replace=False)
            
            for col in cols_to_mask:
                original_val = masked_df.loc[idx, col]
                
                # Store ground truth
                ground_truth.append({
                    'station_id': station_id,
                    'index': idx,
                    'column': col,
                    'original_value': original_val
                })
                
                # Mask the value
                masked_df.loc[idx, col] = np.nan
                total_masked += 1
                
        masked_stations[station_id] = masked_df
        
    print(f"Created masked dataset with {total_masked} artificial missing values across {len(stations)} stations.")
    return masked_stations, pd.DataFrame(ground_truth)


def calculate_metrics(ground_truth_df, imputed_stations):
    """
    Compare imputed values against ground truth.
    """
    y_true = []
    y_pred = []
    
    # Group by station for per-station metrics if needed, 
    # but here we'll do global and per-station
    
    results = []
    
    # Iterate by station to calculate per-station metrics
    for station_id in imputed_stations.keys():
        station_truth = ground_truth_df[ground_truth_df['station_id'] == station_id]
        
        if len(station_truth) == 0:
            continue
            
        station_imputed = imputed_stations[station_id]
        
        s_true = []
        s_pred = []
        
        for _, row in station_truth.iterrows():
            idx = row['index']
            col = row['column']
            val_true = row['original_value']
            
            # Get imputed value
            try:
                val_pred = station_imputed.loc[idx, col]
                s_true.append(val_true)
                s_pred.append(val_pred)
            except KeyError:
                print(f"⚠ Index {idx} not found in imputed data for station {station_id}")
        
        if not s_true:
            continue
            
        s_true = np.array(s_true)
        s_pred = np.array(s_pred)
        
        rmse = np.sqrt(mean_squared_error(s_true, s_pred))
        mae = mean_absolute_error(s_true, s_pred)
        try:
            r2 = r2_score(s_true, s_pred)
        except (ValueError, ZeroDivisionError):
            # R² undefined for constant predictions or insufficient variance
            r2 = np.nan
            
        results.append({
            'Station': station_id,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Masked_Values': len(s_true)
        })
        
        y_true.extend(s_true)
        y_pred.extend(s_pred)
        
    results_df = pd.DataFrame(results)
    
    # Calculate global metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    print("\n" + "-"*80)
    print("OVERALL STATISTICS (Global Evaluation):")
    if len(y_true) > 0:
        global_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        global_mae = mean_absolute_error(y_true, y_pred)
        global_r2 = r2_score(y_true, y_pred)
        
        print(f"  Average RMSE: {global_rmse:.4f}")
        print(f"  Average MAE:  {global_mae:.4f}")
        print(f"  Average R²:   {global_r2:.4f}")
    else:
        print("  No evaluation data available.")
        
    return results_df


def run_evaluation_pipeline_test_only(training_stations, test_stations, feature_cols, 
                                      strategy_func, strategy_kwargs, 
                                      artificial_missing_rate=0.2, random_seed=42):
    """
    Evaluation pipeline for separate training and test sets:
    1. Mask TEST station data only
    2. Train strategy on TRAINING stations
    3. Apply to TEST station
    4. Compare
    """
    print("\n" + "="*80)
    print("TEST STATION EVALUATION")
    print("="*80)
    print(f"Training stations: {list(training_stations.keys())}")
    print(f"Test stations: {list(test_stations.keys())}")
    print(f"Artificial masking on test station: {artificial_missing_rate*100:.0f}% of complete rows")
    
    # 1. Create masked dataset for TEST stations only
    masked_test_stations, ground_truth_df = create_masked_dataset(
        test_stations, feature_cols, artificial_missing_rate, random_seed
    )
    
    if len(ground_truth_df) == 0:
        print("✗ No ground truth data generated. Cannot evaluate.")
        return pd.DataFrame()
    
    # 2. Combine training stations with masked test stations
    all_stations_for_imputation = {**training_stations, **masked_test_stations}
    
    # 3. Run the imputation strategy
    print("\nRunning imputation (training on training stations, applying to test station)...")
    imputed_all_stations = strategy_func(all_stations_for_imputation, feature_cols, **strategy_kwargs)
    
    # 4. Extract only the test station results
    imputed_test_stations = {sid: imputed_all_stations[sid] for sid in test_stations.keys()}
    
    # 5. Calculate metrics
    results_df = calculate_metrics(ground_truth_df, imputed_test_stations)
    
    print("\nTest Station Results:")
    print(results_df.to_string(index=False))
    
    return results_df


def run_evaluation_pipeline(stations, feature_cols, strategy_func, strategy_kwargs, 
                           artificial_missing_rate=0.1, random_seed=42):
    """
    Full evaluation pipeline:
    1. Mask data
    2. Run strategy
    3. Compare
    """
    print("\n" + "="*80)
    print("IMPUTATION EVALUATION")
    print("="*80)
    print(f"Testing with artificial masking: {artificial_missing_rate*100:.0f}% of complete rows")
    
    # 1. Create masked dataset
    masked_stations, ground_truth_df = create_masked_dataset(
        stations, feature_cols, artificial_missing_rate, random_seed
    )
    
    if len(ground_truth_df) == 0:
        print("✗ No ground truth data generated. Cannot evaluate.")
        return pd.DataFrame()
    
    # 2. Run the imputation strategy on the MASKED data
    print("\nRunning imputation on masked dataset...")
    imputed_masked_stations = strategy_func(masked_stations, feature_cols, **strategy_kwargs)
    
    # 3. Calculate metrics
    results_df = calculate_metrics(ground_truth_df, imputed_masked_stations)
    
    print("\nPer-Station Results:")
    print(results_df.to_string(index=False))
    
    return results_df


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_imputed_data(imputed_stations, output_dir, strategy_name):
    """Export imputed datasets"""
    print("\n" + "="*80)
    print("EXPORTING IMPUTED DATA")
    print("="*80)
    
    for station_id, df in imputed_stations.items():
        output_path = os.path.join(
            output_dir, 
            f"imputed_station_{station_id}_{strategy_name}.csv"
        )
        df.to_csv(output_path, index=False)
        print(f"✓ Exported: {output_path}")


def export_evaluation_results(results_df, output_dir, strategy_name):
    """Export evaluation metrics"""
    output_path = os.path.join(output_dir, f"evaluation_metrics_{strategy_name}.csv")
    results_df.to_csv(output_path, index=False)
    print(f"✓ Evaluation metrics saved: {output_path}")


def export_summary_report(stations, imputed_stations, results_df, 
                         output_dir, strategy_name, params):
    """Export comprehensive summary report"""
    output_path = os.path.join(output_dir, f"summary_report_{strategy_name}.txt")
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MULTISITE KNN IMPUTATION SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Strategy: {strategy_name.upper()}\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("PARAMETERS:\n")
        f.write(f"  n_neighbors: {params['n_neighbors']}\n")
        f.write(f"  weights: {params['weights']}\n")
        f.write(f"  metric: {params['metric']}\n\n")
        
        f.write("STATIONS PROCESSED:\n")
        for station_id in stations.keys():
            f.write(f"  Station {station_id}: {stations[station_id].shape[0]} rows\n")
        
        f.write("\nEVALUATION RESULTS:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("OVERALL PERFORMANCE:\n")
        f.write(f"  Average RMSE: {results_df['RMSE'].mean():.4f}\n")
        f.write(f"  Average MAE:  {results_df['MAE'].mean():.4f}\n")
        f.write(f"  Average R²:   {results_df['R²'].mean():.4f}\n")
    
    print(f"✓ Summary report saved: {output_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline execution"""
    print("\n" + "="*80)
    print("MULTISITE KNN IMPUTATION PIPELINE")
    print("="*80)
    print(f"Strategy: {STRATEGY.upper()}")
    print(f"Random seed: {RANDOM_SEED}")
    
    # Create output directory
    create_output_dir(OUTPUT_DIR)
    
    # Step 1: Load data - separate training and test
    training_stations, training_ids = load_stations(TRAINING_FILES)
    test_stations, test_ids = load_stations(TEST_FILES)
    
    if len(training_stations) == 0:
        print("\n✗ Error: No training stations loaded. Check file paths.")
        return
    
    if len(test_stations) == 0:
        print("\n✗ Error: No test stations loaded. Check file paths.")
        return
    
    # Combine for feature engineering and initial analysis
    all_stations = {**training_stations, **test_stations}
    station_ids = training_ids + test_ids
    
    # Step 2: Add temporal features
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    # Track all available features for imputation
    all_feature_cols = FEATURE_COLS.copy()
    temporal_cols_added = False
    
    # Add temporal features to training stations
    for station_id in training_stations.keys():
        training_stations[station_id], new_cols = add_temporal_features(training_stations[station_id])
        if new_cols and not temporal_cols_added:
            all_feature_cols.extend(new_cols)
            temporal_cols_added = True
            print(f"✓ Added temporal features to usage list: {new_cols}")
        print(f"✓ Training Station {station_id}: Temporal features added")
    
    # Add temporal features to test stations
    for station_id in test_stations.keys():
        test_stations[station_id], new_cols = add_temporal_features(test_stations[station_id])
        print(f"✓ Test Station {station_id}: Temporal features added")
    
    # Step 3: Analyze missingness
    print("\n--- Training Stations ---")
    print_missingness_summary(training_stations, all_feature_cols)
    print("\n--- Test Station ---")
    print_missingness_summary(test_stations, all_feature_cols)
    
    # Step 4: Run imputation (Final Production Run)
    print("\n" + "="*80)
    print("RUNNING FINAL IMPUTATION")
    print("="*80)
    
    if STRATEGY == 'pool':
        # Train on training stations, apply to all
        strategy_kwargs = {
            'n_neighbors': N_NEIGHBORS, 
            'weights': WEIGHTS,
            'training_only': True,
            'training_stations': list(training_stations.keys())
        }
        
        # Combine for imputation
        all_stations_combined = {**training_stations, **test_stations}
        
        imputed_stations = pool_then_impute(
            all_stations_combined, 
            all_feature_cols, 
            n_neighbors=N_NEIGHBORS,
            weights=WEIGHTS,
            training_only=True,
            training_stations=list(training_stations.keys())
        )
        
        # Define strategy function for evaluation
        strategy_func = pool_then_impute
        
    elif STRATEGY == 'transfer':
        # Use training stations as source
        all_stations_combined = {**training_stations, **test_stations}
        source_ids = list(training_stations.keys())
        target_ids = list(test_stations.keys())
        
        # We need to wrap the transfer function to match the signature expected by evaluation
        # or just pass the kwargs. 
        # But wait, source_to_target_transfer needs source_ids/target_ids which are derived from stations.
        # If we pass masked stations, we should re-derive them or pass them in.
        # For simplicity in evaluation, we'll let the wrapper re-derive or we just pass the function.
        # However, categorize_stations relies on missingness. Masking changes missingness.
        # Ideally, we keep the SAME source/target designation for evaluation to be fair.
        
        # Training stations are sources, test station is target
        strategy_kwargs = {
            'source_ids': source_ids,
            'target_ids': target_ids,
            'n_neighbors': N_NEIGHBORS, 
            'weights': WEIGHTS
        }
        
        imputed_stations = source_to_target_transfer(
            all_stations_combined,
            source_ids,
            target_ids,
            all_feature_cols,
            n_neighbors=N_NEIGHBORS,
            weights=WEIGHTS
        )
        
        # Define strategy function for evaluation
        strategy_func = source_to_target_transfer
    else:
        print(f"\n✗ Error: Unknown strategy '{STRATEGY}'. Use 'pool' or 'transfer'.")
        return
    
    # Step 5: Evaluate on test station only
    results_df = run_evaluation_pipeline_test_only(
        training_stations,
        test_stations,
        all_feature_cols,
        strategy_func,
        strategy_kwargs,
        artificial_missing_rate=ARTIFICIAL_MISSING_RATE,
        random_seed=RANDOM_SEED
    )
    
    # Step 6: Export results
    export_imputed_data(imputed_stations, OUTPUT_DIR, STRATEGY)
    export_evaluation_results(results_df, OUTPUT_DIR, STRATEGY)
    
    params = {
        'n_neighbors': N_NEIGHBORS,
        'weights': WEIGHTS,
        'metric': METRIC
    }
    export_summary_report(
        all_stations_combined, 
        imputed_stations, 
        results_df, 
        OUTPUT_DIR, 
        STRATEGY,
        params
    )
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nTraining Stations: {list(training_stations.keys())}")
    print(f"Test Station: {list(test_stations.keys())}")
    print(f"Artificial Missing Rate (Test): {ARTIFICIAL_MISSING_RATE*100:.0f}%")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - {len(imputed_stations)} imputed CSV files")
    print(f"  - 1 evaluation metrics file (test station only)")
    print(f"  - 1 summary report\n")


if __name__ == "__main__":
    main()
