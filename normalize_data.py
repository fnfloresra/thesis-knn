"""
Data Normalization Script
Normalizes CSV data to a range from 1 to 5

Usage:
python normalize_data.py
"""

import pandas as pd
import numpy as np
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input files
INPUT_FILES = [
    'site_1_weekly_top20_filled_zeros.csv',
    'site_2_weekly_top20_filled_zeros.csv',
    'site_3_weekly_top20_filled_zeros.csv',
    'site_4_weekly_top20_filled_zeros.csv',
    'site_5_weekly_top20_filled_zeros.csv'
]

# Output directory
OUTPUT_DIR = './normalized_data'

# Normalization range
MIN_VALUE = 0
MAX_VALUE = 1

# Columns to exclude from normalization (e.g., date columns, identifiers)
EXCLUDE_COLUMNS = ['Date']


# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================

def normalize_to_range(series, min_val=1, max_val=5):
    """
    Normalize a series to a specified range [min_val, max_val]
    
    Formula: normalized = min_val + (x - x_min) * (max_val - min_val) / (x_max - x_min)
    
    Parameters:
    -----------
    series : pd.Series
        The data to normalize
    min_val : float
        Minimum value of the target range
    max_val : float
        Maximum value of the target range
    
    Returns:
    --------
    pd.Series : Normalized series
    """
    # Get the min and max of the series (excluding NaN)
    series_min = series.min()
    series_max = series.max()
    
    # If all values are the same, return the middle of the range
    if series_max == series_min:
        return pd.Series([np.mean([min_val, max_val])] * len(series), index=series.index)
    
    # Apply min-max normalization to the specified range
    normalized = min_val + (series - series_min) * (max_val - min_val) / (series_max - series_min)
    
    return normalized


def normalize_dataframe(df, exclude_cols=None, min_val=1, max_val=5):
    """
    Normalize all numeric columns in a dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    exclude_cols : list
        List of column names to exclude from normalization
    min_val : float
        Minimum value of the target range
    max_val : float
        Maximum value of the target range
    
    Returns:
    --------
    pd.DataFrame : Normalized dataframe
    dict : Dictionary with normalization statistics for each column
    """
    if exclude_cols is None:
        exclude_cols = []
    
    df_normalized = df.copy()
    normalization_stats = {}
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove excluded columns
    cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"\nNormalizing {len(cols_to_normalize)} columns to range [{min_val}, {max_val}]...")
    
    for col in cols_to_normalize:
        original_min = df[col].min()
        original_max = df[col].max()
        
        # Normalize the column
        df_normalized[col] = normalize_to_range(df[col], min_val, max_val)
        
        # Store statistics
        normalization_stats[col] = {
            'original_min': original_min,
            'original_max': original_max,
            'normalized_min': df_normalized[col].min(),
            'normalized_max': df_normalized[col].max(),
            'original_mean': df[col].mean(),
            'normalized_mean': df_normalized[col].mean()
        }
        
        print(f"  ✓ {col}: [{original_min:.6f}, {original_max:.6f}] → [{min_val}, {max_val}]")
    
    return df_normalized, normalization_stats


def save_normalization_report(stats, output_file='normalization_report.txt'):
    """
    Save normalization statistics to a text file
    
    Parameters:
    -----------
    stats : dict
        Dictionary with normalization statistics
    output_file : str
        Output file path
    """
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("NORMALIZATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total columns normalized: {len(stats)}\n\n")
        
        for col, stat in stats.items():
            f.write(f"\nColumn: {col}\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Original Range: [{stat['original_min']:.6f}, {stat['original_max']:.6f}]\n")
            f.write(f"  Original Mean:  {stat['original_mean']:.6f}\n")
            f.write(f"  Normalized Range: [{stat['normalized_min']:.6f}, {stat['normalized_max']:.6f}]\n")
            f.write(f"  Normalized Mean:  {stat['normalized_mean']:.6f}\n")
    
    print(f"\n✓ Normalization report saved: {output_file}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("DATA NORMALIZATION PIPELINE")
    print("=" * 80)
    print(f"Input files: {len(INPUT_FILES)} CSV files")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target range: [{MIN_VALUE}, {MAX_VALUE}]")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Output directory created/verified: {OUTPUT_DIR}")
    
    # Process each file
    processed_files = []
    
    for input_file in INPUT_FILES:
        print("\n" + "=" * 80)
        print(f"PROCESSING: {input_file}")
        print("=" * 80)
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"✗ Error: Input file '{input_file}' not found. Skipping...")
            continue
        
        # Load data
        print("\n" + "-" * 80)
        print("LOADING DATA")
        print("-" * 80)
        
        try:
            df = pd.read_csv(input_file)
            print(f"✓ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        except Exception as e:
            print(f"✗ Error loading file: {e}")
            continue
        
        # Normalize data
        print("\n" + "-" * 80)
        print("NORMALIZING DATA")
        print("-" * 80)
        
        df_normalized, stats = normalize_dataframe(
            df, 
            exclude_cols=EXCLUDE_COLUMNS,
            min_val=MIN_VALUE,
            max_val=MAX_VALUE
        )
        
        # Save normalized data
        print("\n" + "-" * 80)
        print("SAVING RESULTS")
        print("-" * 80)
        
        # Generate output filename
        base_name = os.path.basename(input_file).replace('_filled_zeros', '_normalized')
        output_file = os.path.join(OUTPUT_DIR, base_name)
        
        try:
            df_normalized.to_csv(output_file, index=False)
            print(f"✓ Normalized data saved: {output_file}")
            processed_files.append(output_file)
        except Exception as e:
            print(f"✗ Error saving file: {e}")
            continue
        
        # Save normalization report
        report_file = output_file.replace('.csv', '_report.txt')
        save_normalization_report(stats, report_file)
    
    # Final summary
    print("\n" + "=" * 80)
    print("NORMALIZATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nProcessed {len(processed_files)} files:")
    for f in processed_files:
        print(f"  ✓ {f}")
    print()


if __name__ == "__main__":
    main()
