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
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input CSV files (modify paths as needed)
INPUT_FILES = [
    'scaled_reindexed_station_279_weekly_limited (1).csv',
    'scaled_reindexed_station_280_weekly_limited (1).csv',
    'scaled_reindexed_station_281_weekly_limited (1).csv',
    'scaled_reindexed_station_282_weekly_limited (1).csv',
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
N_NEIGHBORS = 7
WEIGHTS = 'distance'
METRIC = 'nan_euclidean'

# Imputation strategy: 'pool' or 'transfer'
STRATEGY = 'pool'

# Evaluation parameters
ARTIFICIAL_MISSING_RATE = 0.1
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
    
    for file_path in file_paths:
        try:
            # Extract station ID from filename
            station_id = int(file_path.split('_')[-3])
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
    
    if 'week' in df.columns:
        df['week'] = pd.to_datetime(df['week'], errors='coerce')
        df['year'] = df['week'].dt.year
        df['month'] = df['week'].dt.month
        df['season'] = df['month'].apply(lambda x: (x % 12 + 3) // 3 if pd.notna(x) else np.nan)
    
    return df


# ============================================================================
# IMPUTATION STRATEGIES
# ============================================================================

def pool_then_impute(stations, feature_cols, n_neighbors=5, weights='distance'):
    """
    Strategy 1: Pool all station data, then apply KNN imputation
    Best when stations share similar characteristics
    """
    print("\n" + "="*80)
    print("POOL-THEN-IMPUTE STRATEGY")
    print("="*80)
    
    # Combine all stations
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
    
    X_imputed = imputer.fit_transform(X_pool)
    
    # Reconstruct dataframe with imputed values
    combined_df[feature_cols] = X_imputed
    
    # Split back into individual stations, preserving original indices
    imputed_stations = {}
    for station_id in stations.keys():
        mask = combined_df['station_id'] == station_id
        imputed_station = combined_df[mask].copy()
        
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

def evaluate_imputation(original_df, imputed_df, feature_cols, 
                        artificial_missing_rate=0.1, random_seed=42):
    """
    Evaluate imputation quality by creating artificial missingness
    """
    np.random.seed(random_seed)
    
    # Get rows with complete data from original
    complete_mask = original_df[feature_cols].notna().all(axis=1)
    df_complete = original_df[complete_mask].copy()
    
    if len(df_complete) < 10:
        return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'n_samples': 0}
    
    # Create artificial missing data (sample from complete rows)
    n_test = max(10, int(len(df_complete) * artificial_missing_rate))
    test_indices = df_complete.sample(n=min(n_test, len(df_complete)), random_state=random_seed).index
    
    # Extract values using the same indices from both dataframes
    X_true = original_df.loc[test_indices, feature_cols].values
    X_imputed = imputed_df.loc[test_indices, feature_cols].values
    
    # Flatten for overall metrics
    X_true_flat = X_true.flatten()
    X_imputed_flat = X_imputed.flatten()
    
    # Remove any remaining NaN values
    valid_mask = ~(np.isnan(X_true_flat) | np.isnan(X_imputed_flat))
    X_true_flat = X_true_flat[valid_mask]
    X_imputed_flat = X_imputed_flat[valid_mask]
    
    if len(X_true_flat) == 0:
        return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'n_samples': 0}
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(X_true_flat, X_imputed_flat))
    mae = mean_absolute_error(X_true_flat, X_imputed_flat)
    
    # R2 calculation with error handling
    try:
        if X_true_flat.std() > 1e-10:
            r2 = r2_score(X_true_flat, X_imputed_flat)
        else:
            r2 = np.nan
    except:
        r2 = np.nan
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'n_samples': len(X_true_flat)
    }


def evaluate_all_stations(stations, imputed_stations, feature_cols, 
                          artificial_missing_rate=0.1, random_seed=42):
    """Evaluate imputation for all stations"""
    print("\n" + "="*80)
    print("IMPUTATION EVALUATION")
    print("="*80)
    
    results = []
    
    for station_id in stations.keys():
        metrics = evaluate_imputation(
            stations[station_id],
            imputed_stations[station_id],
            feature_cols,
            artificial_missing_rate,
            random_seed
        )
        
        results.append({
            'Station': station_id,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'R²': metrics['R2'],
            'Samples': metrics['n_samples']
        })
        
        print(f"\nStation {station_id}:")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAE:  {metrics['MAE']:.4f}")
        print(f"  R²:   {metrics['R2']:.4f}")
        print(f"  Test samples: {metrics['n_samples']}")
    
    results_df = pd.DataFrame(results)
    
    # Overall statistics
    print("\n" + "-"*80)
    print("OVERALL STATISTICS:")
    print(f"  Average RMSE: {results_df['RMSE'].mean():.4f}")
    print(f"  Average MAE:  {results_df['MAE'].mean():.4f}")
    print(f"  Average R²:   {results_df['R²'].mean():.4f}")
    
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
    
    # Step 1: Load data
    stations, station_ids = load_stations(INPUT_FILES)
    
    if len(stations) == 0:
        print("\n✗ Error: No stations loaded. Check file paths.")
        return
    
    # Step 2: Add temporal features
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    for station_id in stations.keys():
        stations[station_id] = add_temporal_features(stations[station_id])
        print(f"✓ Station {station_id}: Temporal features added")
    
    # Step 3: Analyze missingness
    print_missingness_summary(stations, FEATURE_COLS)
    
    # Step 4: Run imputation
    if STRATEGY == 'pool':
        imputed_stations = pool_then_impute(
            stations, 
            FEATURE_COLS, 
            n_neighbors=N_NEIGHBORS,
            weights=WEIGHTS
        )
    elif STRATEGY == 'transfer':
        source_ids, target_ids = categorize_stations(stations, FEATURE_COLS)
        imputed_stations = source_to_target_transfer(
            stations,
            source_ids,
            target_ids,
            FEATURE_COLS,
            n_neighbors=N_NEIGHBORS,
            weights=WEIGHTS
        )
    else:
        print(f"\n✗ Error: Unknown strategy '{STRATEGY}'. Use 'pool' or 'transfer'.")
        return
    
    # Step 5: Evaluate
    results_df = evaluate_all_stations(
        stations,
        imputed_stations,
        FEATURE_COLS,
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
        stations, 
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
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - {len(imputed_stations)} imputed CSV files")
    print(f"  - 1 evaluation metrics file")
    print(f"  - 1 summary report\n")


if __name__ == "__main__":
    main()
