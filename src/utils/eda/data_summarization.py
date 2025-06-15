from pathlib import Path
import pandas as pd
import sys
sys.path.append(str(Path(__file__).parent.parent)) # Add current directory to path for import
from data_loader import load_data
import numpy as np


def calculate_loss_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Loss Ratio (TotalClaims / TotalPremium) for each row
    and handles potential division by zero or infinite results.

    Args:
        df (pd.DataFrame): The input DataFrame which must contain
                           'TotalClaims' and 'TotalPremium' columns.

    Returns:
        pd.DataFrame: The DataFrame with a new 'LossRatio' column added.
                      Returns the original DataFrame if required columns are missing.
    """
    if 'TotalClaims' not in df.columns or 'TotalPremium' not in df.columns:
        print("Error: 'TotalClaims' or 'TotalPremium' column missing for Loss Ratio calculation.")
        return df.copy() # Return a copy to avoid unintended modifications

    # Create a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()

    # Calculate Loss Ratio, handling division by zero/inf by setting to NaN, then fill with 0
    # A small epsilon is added to TotalPremium to avoid exact division by zero,
    # then np.inf results are replaced by np.nan
    df_copy['LossRatio'] = df_copy['TotalClaims'] / (df_copy['TotalPremium'] + np.finfo(float).eps)
    df_copy['LossRatio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df_copy['LossRatio'].fillna(0, inplace=True) # Replace NaNs (from division by zero or missing values) with 0

    print("Calculated 'LossRatio' column.")
    return df_copy

def get_overall_loss_ratio(df: pd.DataFrame) -> dict:
    """
    Calculates the overall Total Premium, Total Claims, and Loss Ratio for the entire portfolio.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary containing 'TotalPremium', 'TotalClaims', and 'OverallLossRatio'.
              Returns None if required columns are missing.
    """
    if 'TotalClaims' not in df.columns or 'TotalPremium' not in df.columns:
        print("Error: 'TotalClaims' or 'TotalPremium' column missing for overall loss ratio calculation.")
        return None

    total_claims = df['TotalClaims'].sum()
    total_premium = df['TotalPremium'].sum()

    overall_loss_ratio = total_claims / total_premium if total_premium != 0 else 0

    return {
        'TotalPremium': total_premium,
        'TotalClaims': total_claims,
        'OverallLossRatio': overall_loss_ratio
    }

if __name__ == "__main__":
    # Example Usage for data_summarization.py
    print("--- Testing data_summarization.py ---")

    # Create a dummy DataFrame for demonstration
    # dummy_data = {
    #     'PolicyID': [1, 2, 3, 4, 5],
    #     'TotalPremium': [1000, 1200, 800, 0, 900], # Added a zero premium case
    #     'TotalClaims': [500, 0, 1500, 100, np.nan], # Added a NaN claim case
    #     'Province': ['Gauteng', 'KZN', 'Western Cape', 'Gauteng', 'Limpopo']
    # }
    # df_dummy = pd.DataFrame(dummy_data)
    # print("\nOriginal Dummy DataFrame:")
    # print(df_dummy)

    # Load the data
    processed_output_dir = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "processed_insurance_data.csv"
    
    # Raw data
    data_file_path = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "temp_extracted_data" / "MachineLearningRating_v3.txt"

    # Load the data using our data_loader module
    df = load_data(processed_output_dir)

    # Test calculate_loss_ratio
    df_with_lr = calculate_loss_ratio(df)
    print("\nDataFrame after calculating LossRatio:")
    print(df_with_lr)

    # Test get_overall_loss_ratio
    overall_metrics = get_overall_loss_ratio(df)
    if overall_metrics:
        print("\nOverall Portfolio Metrics:")
        for key, value in overall_metrics.items():
            print(f"- {key}: {value:.2f}")

    # Test with missing columns
    df_missing_cols = pd.DataFrame({'A': [1], 'B': [2]})
    print("\nTesting with missing columns:")
    df_test_lr = calculate_loss_ratio(df_missing_cols)
    overall_metrics_missing = get_overall_loss_ratio(df_missing_cols)
    print(f"DataFrame unchanged for missing cols: {df_test_lr.equals(df_missing_cols)}")
    print(f"Overall metrics for missing cols: {overall_metrics_missing}")

