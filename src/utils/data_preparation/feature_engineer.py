import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to sys.path to access data_loader
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Assuming data_loader.py is in src/utils/
try:
    from src.utils.data_loader import load_data
except ImportError:
    print("Warning: Could not import load_data from src.utils.data_loader. "
          "Ensure src/utils/data_loader.py exists and path is correct.")
    # Define a minimal load_data function for standalone testing if import fails
    def load_data(file_path, delimiter=',', file_type='csv'):
        print(f"Using fallback load_data for {file_path}")
        if file_type == 'csv':
            return pd.read_csv(file_path, delimiter=delimiter, low_memory=False)
        elif file_type == 'txt':
            return pd.read_csv(file_path, delimiter=delimiter, low_memory=False)
        else:
            raise ValueError("Unsupported file type for fallback loader.")


def create_time_features(df: pd.DataFrame, date_col: str = 'TransactionMonth') -> pd.DataFrame:
    """
    Creates temporal features from a datetime column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        date_col (str): The name of the datetime column.
                        It attempts to parse 'YYYY-MM-DD' or 'YYYYMM' formats.

    Returns:
        pd.DataFrame: DataFrame with new time-based features. Returns an empty DataFrame
                      with original columns if all dates are invalid after conversion.
    """
    if date_col not in df.columns:
        print(f"Warning: Date column '{date_col}' not found. Skipping time feature creation.")
        return df.copy()
    
    df_copy = df.copy()
    
    print(f"DEBUG: Initial '{date_col}' dtype: {df_copy[date_col].dtype}")
    print(f"DEBUG: Initial '{date_col}' head:\n{df_copy[date_col].head()}")

    # Check if already datetime, if not, attempt conversion
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
        print(f"Warning: '{date_col}' is not datetime type. Attempting conversion for feature engineering.")
        
        # Convert to string first if it's a number, to allow consistent parsing attempts
        if pd.api.types.is_numeric_dtype(df_copy[date_col]):
            df_copy[date_col] = df_copy[date_col].astype(str)
            print(f"DEBUG: Converted '{date_col}' to string dtype: {df_copy[date_col].dtype}")

        # Attempt 1: Infer format (handles YYYY-MM-DD, YYYY/MM/DD, etc.)
        converted_dates_1 = pd.to_datetime(df_copy[date_col], errors='coerce', infer_datetime_format=True)
        print(f"DEBUG: NaNs after inferring format: {converted_dates_1.isnull().sum()} / {len(df_copy)}")

        # Attempt 2: Explicitly try 'YYYYMM' format if first attempt failed for most values
        # This handles cases where TransactionMonth might be an integer/string like '201503'
        if converted_dates_1.isnull().sum() > 0.5 * len(df_copy): # If more than 50% failed
            print(f"  Attempting to parse '{date_col}' as 'YYYYMM' format explicitly...")
            converted_dates_2 = pd.to_datetime(df_copy[date_col], format='%Y%m', errors='coerce')
            print(f"DEBUG: NaNs after YYYYMM format: {converted_dates_2.isnull().sum()} / {len(df_copy)}")
            # Use the result from the second attempt if it's better (fewer NaNs)
            if converted_dates_2.isnull().sum() < converted_dates_1.isnull().sum():
                df_copy[date_col] = converted_dates_2
                print(f"  Using 'YYYYMM' parsing result for '{date_col}'.")
            else:
                df_copy[date_col] = converted_dates_1
                print(f"  Sticking with inferred parsing result for '{date_col}'.")
        else:
            df_copy[date_col] = converted_dates_1
            print(f"  Using inferred parsing result for '{date_col}'.")

    # Now, drop rows where date conversion truly failed, leaving NaT
    initial_rows = df_copy.shape[0]
    df_copy.dropna(subset=[date_col], inplace=True) # Modified: Removed inplace=True, replaced by direct assignment (next line)
    # df_copy = df_copy.dropna(subset=[date_col]) # This is the preferred way in Pandas 3.0+

    if df_copy.shape[0] < initial_rows:
        print(f"Dropped {initial_rows - df_copy.shape[0]} rows due to invalid dates in '{date_col}'.")
        if df_copy.empty:
            print(f"CRITICAL: DataFrame became empty after dropping NaNs in '{date_col}'. No time features will be created.")
            # Return an empty DataFrame with the original columns to maintain schema for subsequent steps
            return pd.DataFrame(columns=df.columns) 

    if not df_copy.empty:
        df_copy['Month'] = df_copy[date_col].dt.month
        df_copy['Year'] = df_copy[date_col].dt.year
        df_copy['DayOfWeek'] = df_copy[date_col].dt.dayofweek # Monday=0, Sunday=6
        df_copy['DayOfYear'] = df_copy[date_col].dt.dayofyear
        df_copy['WeekOfYear'] = df_copy[date_col].dt.isocalendar().week.astype(int)
        df_copy['Quarter'] = df_copy[date_col].dt.quarter
        print(f"Created time-based features from '{date_col}'.")
    else:
        # This branch is now reached if df_copy becomes empty after dropna
        print(f"DataFrame is empty after processing '{date_col}'. No time features created.")
    return df_copy

def create_risk_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates risk-related ratio features (e.g., Claim-to-Premium ratio).

    Args:
        df (pd.DataFrame): The input DataFrame with 'TotalPremium' and 'TotalClaims'.

    Returns:
        pd.DataFrame: DataFrame with new risk ratio features.
    """
    df_copy = df.copy()
    
    if 'TotalPremium' in df_copy.columns and 'TotalClaims' in df_copy.columns:
        # Convert to numeric and fill NaNs before calculation to avoid errors
        df_copy['TotalPremium'] = pd.to_numeric(df_copy['TotalPremium'], errors='coerce').fillna(0)
        df_copy['TotalClaims'] = pd.to_numeric(df_copy['TotalClaims'], errors='coerce').fillna(0)

        # Avoid division by zero, add a small epsilon to denominator
        df_copy['ClaimPremiumRatio'] = df_copy['TotalClaims'] / (df_copy['TotalPremium'] + np.finfo(float).eps)
        # Cap very high ratios that might be outliers or represent zero premium policies with claims
        df_copy['ClaimPremiumRatio'].replace([np.inf, -np.inf], np.nan, inplace=True) # Modified: `replace` is okay with inplace, but safer to assign
        df_copy['ClaimPremiumRatio'] = df_copy['ClaimPremiumRatio'].fillna(0) # Modified: No inplace=True
        print("Created 'ClaimPremiumRatio' feature.")
    else:
        print("Warning: 'TotalPremium' or 'TotalClaims' not found. Skipping 'ClaimPremiumRatio' creation.")
        
    return df_copy

def create_vehicle_age_feature(df: pd.DataFrame, current_year: int = 2025) -> pd.DataFrame:
    """
    Calculates vehicle age based on 'RegistrationYear'.

    Args:
        df (pd.DataFrame): The input DataFrame with 'RegistrationYear'.
        current_year (int): The current reference year (default to 2025 as per project context).

    Returns:
        pd.DataFrame: DataFrame with new 'VehicleAge' feature.
    """
    df_copy = df.copy()
    if 'RegistrationYear' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['RegistrationYear']):
        df_copy['VehicleAge'] = current_year - df_copy['RegistrationYear']
        # Handle cases where RegistrationYear is future or very old (negative age or extreme ages)
        df_copy['VehicleAge'] = df_copy['VehicleAge'].clip(lower=0) # Age cannot be negative
        df_copy['VehicleAge'] = df_copy['VehicleAge'].fillna(df_copy['VehicleAge'].median()) # Modified: No inplace=True
        print("Created 'VehicleAge' feature.")
    else:
        print("Warning: 'RegistrationYear' not found or not numerical. Skipping 'VehicleAge' creation.")
    return df_copy

# Example usage (for standalone testing)
if __name__ == "__main__":
    print("--- Testing FeatureEngineer with Actual Data ---")

    # Define path to the actual processed data file
    processed_data_path = project_root / "data" / "processed" / "processed_insurance_data.csv"
    raw_data_path = project_root / "data" / "raw" / "temp_extracted_data" / "MachineLearningRating_v3.txt"
    
    # --- IMPORTANT: The script now *expects* the processed data to exist. ---
    # If the file does not exist, it will raise a FileNotFoundError.
    if not processed_data_path.is_file():
        raise FileNotFoundError(f"Processed data file not found at: {processed_data_path}. "
                                "Please ensure you have run previous data processing steps to create this file.")
    
    # Load the actual processed data using the specified comma delimiter
    df = load_data(raw_data_path, delimiter='|', file_type='txt')

    if not df.empty:
        print(f"Original 'TransactionMonth' dtype: {df['TransactionMonth'].dtype}")
        print(f"Original 'TransactionMonth' head:\n{df['TransactionMonth'].head()}")
        
        print("\nOriginal DataFrame Head (before feature engineering):")
        print(df.head())

        # Apply feature engineering functions sequentially
        df_fe = create_time_features(df.copy(), 'TransactionMonth')
        
        # Check if df_fe became empty after time feature creation
        if df_fe.empty:
            print("\nDEBUG: df_fe is empty after create_time_features. Skipping further FE steps.")
        else:
            df_fe = create_risk_ratio_features(df_fe.copy()) 
            df_fe = create_vehicle_age_feature(df_fe.copy())

        print("\nDataFrame Head after Feature Engineering:")
        print(df_fe.head())
        print("\nDataFrame Info after Feature Engineering:")
        df_fe.info()
        print("\nColumns after Feature Engineering:")
        print(df_fe.columns.tolist())
    else:
        print("DataFrame is empty after loading. Skipping feature engineering testing.")

    print("\nFeatureEngineer demonstrations complete.")
