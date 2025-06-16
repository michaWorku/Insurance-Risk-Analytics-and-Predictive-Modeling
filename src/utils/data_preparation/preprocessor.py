import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to sys.path to access data_loader if running standalone
# This assumes data_loader is at src/utils/data_loader.py
# If data_loader is elsewhere, adjust the path
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


class DataPreprocessor:
    """
    A modular class for comprehensive data preprocessing steps including:
    - Initial data loading and cleaning (handling missing values, duplicates).
    - Flexible encoding of categorical features (OneHotEncoder, LabelEncoder).
    - Various scaling methods for numerical features (StandardScaler, MinMaxScaler, Log Transform).
    - Imputation strategies within the preprocessing pipeline.
    """

    def __init__(self, numerical_cols: list, categorical_cols: list):
        """
        Initializes the DataPreprocessor with lists of numerical and categorical columns.

        Args:
            numerical_cols (list): List of column names to be treated as numerical.
            categorical_cols (list): List of column names to be treated as categorical.
        """
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.column_transformer = None # Will be built dynamically based on method calls
        self.label_encoders = {} # Store LabelEncoders if used
        self.fitted_preprocessor_pipeline = None


    @staticmethod
    def load_and_clean_data(file_path: Path, delimiter: str = ',', file_type = 'csv') -> pd.DataFrame:
        """
        Loads data from a CSV/TXT file, removes duplicates, and drops rows with
        missing values in critical columns.

        Args:
            file_path (Path): The path to the data file.
            delimiter (str): The delimiter used in the data file (e.g., ',', '|').
            file_type (str, optional): The type of the file (e.g., 'csv', 'txt').
                                   If None, infers from file extension.

        Returns:
            pd.DataFrame: The loaded and initially cleaned DataFrame.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found at: {file_path}")

        df = load_data(file_path, delimiter=delimiter, file_type=file_type)

        initial_rows = df.shape[0]
        # Remove duplicate rows
        df.drop_duplicates(inplace=True)
        print(f"Removed {initial_rows - df.shape[0]} duplicate rows.")

        # For initial cleaning, we can drop rows where 'TotalPremium' or 'TotalClaims' are NaN
        # (Assuming these are critical for any analysis, and not 0 for no claim)
        # However, for this preprocessor, we'll let SimpleImputer handle NaNs in defined columns.
        # So, the "cleaning" here is mainly duplicates.
        
        # Ensure critical financial columns are numeric, filling NaNs with 0 if they represent absence of value
        for col in ['TotalPremium', 'TotalClaims']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) # Fill financial NaNs with 0
            else:
                print(f"Warning: Critical column '{col}' not found during initial cleaning.")

        print(f"Data loaded and duplicates removed. Current shape: {df.shape}")
        return df

    def apply_encoding(self, df: pd.DataFrame, encoder_type: str = 'onehot') -> pd.DataFrame:
        """
        Applies encoding to categorical columns. Supports 'onehot' and 'label' encoding.

        Args:
            df (pd.DataFrame): The input DataFrame.
            encoder_type (str): Type of encoder to use ('onehot' or 'label').

        Returns:
            pd.DataFrame: DataFrame with encoded categorical features.
        """
        df_encoded = df.copy()
        if not self.categorical_cols:
            print("No categorical columns specified for encoding.")
            return df_encoded

        if encoder_type == 'onehot':
            print("Applying One-Hot Encoding to categorical features...")
            # For one-hot encoding, we build a ColumnTransformer sub-pipeline
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            # Use ColumnTransformer for one-hot encoding to handle multiple columns efficiently
            onehot_preprocessor = ColumnTransformer(
                transformers=[
                    ('cat_onehot', categorical_pipeline, self.categorical_cols)
                ],
                remainder='passthrough'
            )
            
            # Fit and transform
            transformed_array = onehot_preprocessor.fit_transform(df_encoded)
            
            # Get feature names after one-hot encoding
            ohe_feature_names = onehot_preprocessor.named_transformers_['cat_onehot']['onehot'].get_feature_names_out(self.categorical_cols)
            
            # Reconstruct DataFrame with new column names
            # Need to get names of passthrough columns too
            passthrough_cols = [col for col in df_encoded.columns if col not in self.categorical_cols]
            
            # Order of columns from ColumnTransformer: transformed, then passthrough
            df_encoded = pd.DataFrame(transformed_array, columns=list(ohe_feature_names) + passthrough_cols, index=df_encoded.index)
            print("One-Hot Encoding complete.")

        elif encoder_type == 'label':
            print("Applying Label Encoding to categorical features...")
            for col in self.categorical_cols:
                if col in df_encoded.columns:
                    # Handle potential NaNs before LabelEncoding, as LabelEncoder doesn't support them
                    df_encoded[col] = df_encoded[col].fillna('Missing').astype(str) # Impute with 'Missing' string
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col])
                    self.label_encoders[col] = le # Store encoder for inverse_transform if needed
                else:
                    print(f"Warning: Categorical column '{col}' not found for Label Encoding.")
            print("Label Encoding complete.")
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}. Choose 'onehot' or 'label'.")

        return df_encoded

    def apply_scaling(self, df: pd.DataFrame, scaler_type: str = 'standard') -> pd.DataFrame:
        """
        Applies scaling/transformation to numerical columns. Supports 'standard', 'minmax', 'log', 'none'.

        Args:
            df (pd.DataFrame): The input DataFrame.
            scaler_type (str): Type of scaler/transformer to use ('standard', 'minmax', 'log', 'none').

        Returns:
            pd.DataFrame: DataFrame with scaled/transformed numerical features.
        """
        df_scaled = df.copy()
        if not self.numerical_cols:
            print("No numerical columns specified for scaling.")
            return df_scaled

        # Filter numerical columns to only include those present in the current DataFrame
        cols_to_scale = [col for col in self.numerical_cols if col in df_scaled.columns]
        if not cols_to_scale:
            print("No existing numerical columns to scale.")
            return df_scaled

        if scaler_type == 'log':
            print("Applying Log Transformation to numerical features...")
            for col in cols_to_scale:
                # Add a small constant to avoid log(0) or log(negative)
                df_scaled[col] = np.log1p(df_scaled[col].fillna(0)) # log1p handles x=0 gracefully, fill NaNs with 0
                # Re-check for inf/-inf after log transform (e.g. if original was negative and log applied)
                df_scaled[col].replace([np.inf, -np.inf], np.nan, inplace=True)
                df_scaled[col].fillna(df_scaled[col].mean(), inplace=True) # Impute any new NaNs from log transform
            print("Log Transformation complete.")
            # If log is applied, subsequent scaling might be redundant or undesirable,
            # but we allow it as a sequence. For simplicity, we just apply log and return.
            return df_scaled # Log transform usually replaces scaling, so return here

        # For Standard and MinMaxScaler, we use a ColumnTransformer for robustness
        numerical_pipeline_steps = []
        # Impute numerical NaNs before scaling
        numerical_pipeline_steps.append(('imputer', SimpleImputer(strategy='mean'))) # Default numerical imputation

        if scaler_type == 'standard':
            print("Applying Standard Scaling to numerical features...")
            numerical_pipeline_steps.append(('scaler', StandardScaler()))
        elif scaler_type == 'minmax':
            print("Applying Min-Max Scaling to numerical features...")
            numerical_pipeline_steps.append(('scaler', MinMaxScaler()))
        elif scaler_type == 'none':
            print("Skipping numerical scaling.")
            return df_scaled # No scaling needed, return original
        else:
            raise ValueError(f"Unsupported scaler_type: {scaler_type}. Choose 'standard', 'minmax', or 'log'.")

        numerical_transformer = Pipeline(numerical_pipeline_steps)

        # Create ColumnTransformer to apply numerical pipeline only to specified columns
        scaling_preprocessor = ColumnTransformer(
            transformers=[
                ('num_scaler', numerical_transformer, cols_to_scale)
            ],
            remainder='passthrough' # Keep other columns as is
        )
        
        # Fit and transform
        transformed_array = scaling_preprocessor.fit_transform(df_scaled)

        # Reconstruct DataFrame with original column names (for transformed numerical) and passthrough
        transformed_columns = list(scaling_preprocessor.named_transformers_['num_scaler'].named_steps['scaler'].get_feature_names_out(cols_to_scale))
        passthrough_cols = [col for col in df_scaled.columns if col not in cols_to_scale]

        df_scaled = pd.DataFrame(transformed_array, columns=transformed_columns + passthrough_cols, index=df_scaled.index)
        print(f"{scaler_type} Scaling complete.")

        return df_scaled

    def preprocess(self, df: pd.DataFrame, encoder_type: str = 'onehot', scaler_type: str = 'standard') -> pd.DataFrame:
        """
        Orchestrates the full preprocessing pipeline:
        1. Applies encoding to categorical features.
        2. Applies scaling/transformation to numerical features.

        Args:
            df (pd.DataFrame): The input DataFrame.
            encoder_type (str): Type of encoder for categorical features ('onehot' or 'label').
            scaler_type (str): Type of scaler for numerical features ('standard', 'minmax', 'log', 'none').

        Returns:
            pd.DataFrame: The fully preprocessed DataFrame.
        """
        print(f"\n--- Starting Full Preprocessing (Encoder: {encoder_type}, Scaler: {scaler_type}) ---")

        # Step 1: Apply encoding
        df_temp = self.apply_encoding(df.copy(), encoder_type=encoder_type)
        
        # Step 2: Apply scaling/transformation
        # Ensure only the numerical columns that are still present after encoding are passed
        # This is crucial if encoding adds/removes columns or changes types.
        current_numerical_cols = [col for col in self.numerical_cols if col in df_temp.columns and pd.api.types.is_numeric_dtype(df_temp[col])]
        
        # Temporarily update self.numerical_cols for the scaling method to use the current set
        original_numerical_cols = self.numerical_cols
        self.numerical_cols = current_numerical_cols
        df_processed = self.apply_scaling(df_temp, scaler_type=scaler_type)
        self.numerical_cols = original_numerical_cols # Restore original list

        print("--- Full Preprocessing Complete ---")
        return df_processed


# Example usage (for standalone testing and demonstration)
if __name__ == "__main__":
    print("--- Testing DataPreprocessor with Actual Data ---")

    # Define path to the actual processed data file
    processed_data_path = project_root / "data" / "processed" / "processed_insurance_data.csv"
    
    # Load and initially clean the data (removing duplicates and handling critical NaNs)
    df = DataPreprocessor.load_and_clean_data(processed_data_path, delimiter=',')

    if not df.empty:
        # Pre-convert TransactionMonth to datetime if it's needed for other feature engineering
        # This step is outside the preprocessor's methods but part of general data prep flow.
        if 'TransactionMonth' in df.columns:
            df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], format='%Y%m', errors='coerce')
        
        # Define numerical and categorical columns for preprocessing
        # These are features we intend to use in a model
        numerical_cols_for_model = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate', 'RegistrationYear', 'Cylinders']
        categorical_cols_for_model = ['Gender', 'Province', 'VehicleType', 'IsVATRegistered']

        # Ensure all columns exist in the DataFrame before passing to Preprocessor
        numerical_cols_for_model = [col for col in numerical_cols_for_model if col in df.columns]
        categorical_cols_for_model = [col for col in categorical_cols_for_model if col in df.columns]

        # --- Demonstration 1: One-Hot Encoding + Standard Scaling ---
        print("\n=== Demo 1: One-Hot Encoding + Standard Scaling ===")
        preprocessor_ohe_standard = DataPreprocessor(
            numerical_cols=numerical_cols_for_model,
            categorical_cols=categorical_cols_for_model
        )
        df_processed_ohe_standard = preprocessor_ohe_standard.preprocess(df.copy(), encoder_type='onehot', scaler_type='standard')
        print("\nProcessed DataFrame (One-Hot + Standard) Head:")
        print(df_processed_ohe_standard.head())
        print("\nProcessed DataFrame (One-Hot + Standard) Info:")
        df_processed_ohe_standard.info()

        # --- Demonstration 2: Label Encoding + Min-Max Scaling ---
        # Note: Label Encoding is usually for ordinal features or target variables.
        # For nominal features like 'Gender' or 'Province', One-Hot is generally preferred
        # to avoid implying an arbitrary order. This demo is for functional completeness.
        print("\n=== Demo 2: Label Encoding + Min-Max Scaling ===")
        preprocessor_le_minmax = DataPreprocessor(
            numerical_cols=numerical_cols_for_model,
            categorical_cols=categorical_cols_for_model
        )
        # Apply label encoding first as a separate step if not using ColumnTransformer for it
        df_encoded_le = preprocessor_le_minmax.apply_encoding(df.copy(), encoder_type='label')
        df_processed_le_minmax = preprocessor_le_minmax.apply_scaling(df_encoded_le, scaler_type='minmax')

        print("\nProcessed DataFrame (Label + Min-Max) Head:")
        print(df_processed_le_minmax.head())
        print("\nProcessed DataFrame (Label + Min-Max) Info:")
        df_processed_le_minmax.info()

        # --- Demonstration 3: Log Transformation + No further scaling ---
        print("\n=== Demo 3: Log Transformation + No further scaling ===")
        preprocessor_log = DataPreprocessor(
            numerical_cols=numerical_cols_for_model,
            categorical_cols=categorical_cols_for_model # Categorical will still be one-hot encoded by default
        )
        df_processed_log = preprocessor_log.preprocess(df.copy(), encoder_type='onehot', scaler_type='log')
        print("\nProcessed DataFrame (One-Hot + Log) Head:")
        print(df_processed_log.head())
        print("\nProcessed DataFrame (One-Hot + Log) Info:")
        df_processed_log.info()

    else:
        print("DataFrame is empty after loading and initial cleaning. Skipping preprocessor demonstrations.")

    print("\nDataPreprocessor demonstrations complete.")
