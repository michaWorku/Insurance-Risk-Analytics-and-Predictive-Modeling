import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import warnings # Import warnings for suppressing specific messages
from pathlib import Path # Required for the @staticmethod load_and_clean_data and save_dataframe

class DataPreprocessor:
    """
    A class to preprocess DataFrame for machine learning.
    It handles numerical imputation, scaling, and categorical encoding.
    """
    def __init__(self, numerical_cols=None, categorical_cols=None):
        self.numerical_cols = numerical_cols if numerical_cols is not None else []
        self.categorical_cols = categorical_cols if categorical_cols is not None else []
        self.imputer = None
        self.scaler = None
        self.encoders = {}
        self.fitted_numerical_cols_for_imputer = [] # NEW: To store the exact columns imputer was fitted on

    def fit_imputer(self, df: pd.DataFrame):
        """Fits the imputer on numerical columns."""
        # Ensure only columns present in df and numerical are selected for fitting
        cols_to_fit = [col for col in self.numerical_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if cols_to_fit and not df[cols_to_fit].empty:
            self.imputer = SimpleImputer(strategy='mean')
            self.imputer.fit(df[cols_to_fit])
            self.fitted_numerical_cols_for_imputer = cols_to_fit # Store the actual columns used for fitting
            print(f"Imputer fitted on numerical columns: {self.fitted_numerical_cols_for_imputer}")
        else:
            print("No numerical columns specified or DataFrame is empty for imputer fitting. Skipping fitting.")
            self.imputer = None # Ensure imputer is reset if not fitted
            self.fitted_numerical_cols_for_imputer = [] # Reset fitted columns

    def apply_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies imputation to numerical columns."""
        df_imputed = df.copy()
        
        # Use the exact columns the imputer was fitted on
        if self.imputer and self.fitted_numerical_cols_for_imputer:
            # Check if all fitted columns are actually present in the current df_imputed
            missing_cols_for_transform = [col for col in self.fitted_numerical_cols_for_imputer if col not in df_imputed.columns]
            if missing_cols_for_transform:
                print(f"Error: Missing columns in DataFrame for imputation transform: {missing_cols_for_transform}. Cannot apply imputation.")
                return df_imputed # Return original df, cannot proceed

            if not df_imputed[self.fitted_numerical_cols_for_imputer].empty:
                # IMPORTANT: .values ensures that we pass a NumPy array, not a DataFrame
                # This avoids potential pandas indexing quirks that lead to 'Columns must be same length as key'
                imputed_data = self.imputer.transform(df_imputed[self.fitted_numerical_cols_for_imputer].values)
                df_imputed[self.fitted_numerical_cols_for_imputer] = imputed_data
                print("Numerical imputation applied.")
            else:
                print("Numerical columns for imputation are empty. Skipping imputation.")
        else:
            print("Imputer not fitted or no numerical columns to impute. Skipping imputation.")
        return df_imputed

    def fit_scaler(self, df: pd.DataFrame, scaler_type='standard'):
        """Fits the scaler on numerical columns."""
        if self.numerical_cols and not df[self.numerical_cols].empty:
            if scaler_type == 'standard':
                self.scaler = StandardScaler()
            # Add other scaler types here if needed (e.g., MinMaxScaler)
            
            if self.scaler:
                self.scaler.fit(df[self.numerical_cols])
                print(f"{scaler_type.capitalize()} scaler fitted on numerical columns: {self.numerical_cols}")
            else:
                print(f"Scaler type '{scaler_type}' not recognized. Skipping scaler fitting.")
        else:
            print("No numerical columns specified or DataFrame is empty for scaler fitting.")

    def apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies scaling to numerical columns."""
        df_scaled = df.copy()
        if self.scaler and self.numerical_cols and not df_scaled[self.numerical_cols].empty:
            df_scaled[self.numerical_cols] = self.scaler.transform(df_scaled[self.numerical_cols])
            print("Numerical scaling applied.")
        elif self.numerical_cols and df_scaled[self.numerical_cols].empty:
            print("Warning: Numerical columns for scaling are empty. Skipping scaling.")
        else:
            print("Scaler not fitted or no numerical columns specified. Skipping scaling.")
        return df_scaled
        
    def apply_encoding(self, df: pd.DataFrame, encoder_type='label') -> pd.DataFrame:
        """Applies encoding to categorical columns."""
        df_encoded = df.copy()
        cols_to_encode = [col for col in self.categorical_cols if col in df_encoded.columns]

        if not cols_to_encode:
            print("No categorical columns specified or found for encoding. Skipping encoding.")
            return df_encoded

        if encoder_type == 'label':
            print("Applying Label Encoding to categorical features...")
            for col in cols_to_encode:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = le
            print("Label Encoding applied.")
        elif encoder_type == 'onehot':
            print("Applying One-Hot Encoding to categorical features...")
            df_encoded[cols_to_encode] = df_encoded[cols_to_encode].astype(str)
            df_encoded = pd.get_dummies(df_encoded, columns=cols_to_encode, dummy_na=False)
            print("One-Hot Encoding applied.")
        else:
            print(f"Encoder type '{encoder_type}' not recognized. Skipping encoding.")
        
        return df_encoded


    def preprocess(self, df: pd.DataFrame, encoder_type='label', scaler_type='standard') -> pd.DataFrame:
        """
        Performs a full preprocessing pipeline: imputation, encoding, and scaling.
        This method assumes initial data preparation (type conversions, dropping infinities)
        has already been done via `apply_initial_data_preparation`.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
            encoder_type (str): Type of encoder ('label' or 'onehot').
            scaler_type (str): Type of scaler ('standard').

        Returns:
            pd.DataFrame: The fully preprocessed DataFrame.
        """
        if df.empty:
            print("Input DataFrame is empty. Skipping preprocessing pipeline.")
            return df

        print(f"\n--- Starting Full Preprocessing (Encoder: {encoder_type}, Scaler: {scaler_type}) ---")

        # Step 1: Apply encoding to categorical features
        df_temp = self.apply_encoding(df.copy(), encoder_type=encoder_type)
        if df_temp.empty:
            print("Warning: DataFrame became empty after encoding. Skipping scaling.")
            return df_temp

        # Step 2: Impute numerical features BEFORE scaling
        for col in self.numerical_cols:
            if col in df_temp.columns:
                df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce') # Ensure numeric after encoding/before imputation
        
        if self.numerical_cols:
            nans_in_numerical = df_temp[self.numerical_cols].isnull().any().any()
            if nans_in_numerical:
                self.fit_imputer(df_temp) # Fit on the current df_temp state
                df_temp = self.apply_imputation(df_temp)
            else:
                print("No NaNs in specified numerical columns, skipping imputation.")
        else:
            print("No numerical columns specified for imputation.")


        # Step 3: Apply scaling
        for col in self.numerical_cols:
            if col in df_temp.columns:
                df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce') # Re-ensure numeric after imputation/before scaling

        if self.numerical_cols and not df_temp[self.numerical_cols].empty:
            # Check if there's any variance to scale. If all values are constant, StandardScaler will warn/error.
            if df_temp[self.numerical_cols].nunique().sum() > len(self.numerical_cols):
                self.fit_scaler(df_temp, scaler_type=scaler_type)
                df_processed = self.apply_scaling(df_temp)
            else:
                print("Numerical columns have no variance for scaling. Skipping scaling.")
                df_processed = df_temp
        elif self.numerical_cols:
            print("Numerical columns are empty or not specified for scaling. Skipping scaling.")
            df_processed = df_temp
        else:
            print("No numerical columns specified for scaling. Skipping scaling.")
            df_processed = df_temp

        print("Full preprocessing pipeline completed.")
        return df_processed
    
    @staticmethod
    def load_and_clean_data(file_path: Path, delimiter: str = ',', file_type: str = 'csv') -> pd.DataFrame:
        """
        Loads data from a specified file path and performs initial cleaning.
        Includes handling duplicate rows and filling critical NaNs in TotalPremium/TotalClaims.
        """
        if not file_path.is_file():
            print(f"Error: File not found at {file_path}")
            return pd.DataFrame()

        try:
            if file_type == 'csv':
                df = pd.read_csv(file_path, sep=delimiter)
            elif file_type == 'txt':
                try:
                    df = pd.read_csv(file_path, sep=delimiter, engine='python')
                except pd.errors.ParserError:
                    try:
                        df = pd.read_csv(file_path, sep='\t', engine='python')
                    except pd.errors.ParserError:
                        df = pd.read_csv(file_path, sep=r'\s+', engine='python')
            else:
                print(f"Error: Unsupported file type '{file_type}'. Only 'csv' and 'txt' are supported.")
                return pd.DataFrame()

            print(f"Successfully loaded data from {file_path}. Initial shape: {df.shape}")

            initial_rows = df.shape[0]
            df.drop_duplicates(inplace=True)
            rows_after_dedup = df.shape[0]
            if initial_rows > rows_after_dedup:
                print(f"Removed {initial_rows - rows_after_dedup} duplicate rows.")

            for col in ['TotalPremium', 'TotalClaims']:
                if col in df.columns:
                    # Convert to numeric, coercing errors to NaN, then fill NaNs with 0
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    # Explicitly replace any remaining np.inf or -np.inf with 0 for these critical columns
                    df[col].replace([np.inf, -np.inf], 0, inplace=True) 
                    if df[col].isnull().any(): # Check again after filling with 0, in case of all NaN column
                        print(f"Warning: After filling with 0, '{col}' still has {df[col].isnull().sum()} NaNs. This indicates column might be entirely non-numeric.")
                else:
                    print(f"Warning: Critical column '{col}' not found for initial NaN handling.")

            return df
        except Exception as e:
            print(f"Error loading or initially cleaning data from {file_path}: {e}")
            return pd.DataFrame()

    @staticmethod
    def save_dataframe(df: pd.DataFrame, file_path: Path, index=False):
        """Saves a DataFrame to a specified CSV file path."""
        try:
            df.to_csv(file_path, index=index)
            print(f"DataFrame successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving DataFrame to {file_path}: {e}")

    @staticmethod
    def reduce_high_cardinality_features(df: pd.DataFrame, 
                                        categorical_cols_to_reduce: list = None,
                                        postal_code_col: str = None,
                                        max_unique_categories_for_ohe: int = 50,
                                        other_threshold_ratio: float = 0.01):
        """
        Reduces cardinality of specified categorical features.
        """
        df_reduced = df.copy()

        if categorical_cols_to_reduce is None:
            categorical_cols_to_reduce = []
        
        print("\nApplying cardinality reduction...")

        if postal_code_col and postal_code_col in df_reduced.columns:
            if df_reduced[postal_code_col].dtype == 'object' or pd.api.types.is_numeric_dtype(df_reduced[postal_code_col]):
                df_reduced[postal_code_col] = df_reduced[postal_code_col].astype(str).str[:3]
                df_reduced[postal_code_col] = df_reduced[postal_code_col].astype('category')
                print(f"  Truncated '{postal_code_col}' to first 3 digits and converted to category.")
            else:
                print(f"  Skipping truncation for '{postal_code_col}': not a string or numeric type.")

        for col in categorical_cols_to_reduce:
            if col in df_reduced.columns and (pd.api.types.is_categorical_dtype(df_reduced[col]) or df_reduced[col].dtype == 'object'):
                temp_series = df_reduced[col].astype(str)
                value_counts = temp_series.value_counts(normalize=True)
                
                rare_categories = value_counts[value_counts < other_threshold_ratio].index
                
                if not rare_categories.empty:
                    df_reduced[col] = temp_series.replace(rare_categories, 'Other_Category').astype('category')
                    print(f"  Grouped {len(rare_categories)} rare categories in '{col}' into 'Other_Category'. New unique count: {df_reduced[col].nunique()}")
                else:
                    print(f"  No rare categories to group in '{col}'.")
            elif col in df_reduced.columns:
                print(f"  Skipping grouping for '{col}': not a categorical or object type.")
            else:
                print(f"  Column '{col}' not found for cardinality reduction.")

        print("Cardinality reduction complete.")
        return df_reduced

    @staticmethod
    def apply_initial_data_preparation(df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs initial data preparation steps including:
        1. Correcting data types for known numerical and boolean-like columns.
           This step also explicitly DROPS rows containing np.inf or -np.inf in these columns.
        2. Converts remaining 'object' columns to 'category' dtype.
        
        Note: This method no longer performs general dropping of rows with NaNs.
              NaNs (from missing values or coercion from non-numeric strings) will be handled
              by the imputer in the `preprocess` method.
              'TransactionMonth' conversion is skipped as it will be dropped later.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
            
        Returns:
            pd.DataFrame: The DataFrame after initial data preparation.
                          Returns an empty DataFrame if it becomes empty after dropping infinities.
        """
        df_prepared = df.copy()

        if df_prepared.empty:
            print("Input DataFrame is empty for initial data preparation. Returning empty DataFrame.")
            return df_prepared

        print("\n--- Performing Initial Data Preparation (Type Conversions, Dropping Infinities, No General NaN Dropping or Date Conversion) ---")

        # Skip TransactionMonth date conversion as requested.
        if 'TransactionMonth' in df_prepared.columns:
            print("  Skipping 'TransactionMonth' date conversion as requested. It will retain its original dtype.")

        cols_to_force_numeric = [
            'RegistrationYear', 'Cylinders', 'Cubiccapacity', 'Kilowatts',
            'NumberOfDoors', 'CustomValueEstimate', 'CapitalOutstanding',
            'SumInsured', 'CalculatedPremiumPerTerm', 'Mmcode',
            'NumberOfVehiclesInFleet', 'TermFrequency', 'ExcessSelected',
            'ClaimPremiumRatio', 'VehicleAge'
        ]
        
        # 1. Correcting data types for known numerical columns (including coercing errors to NaN)
        print("\n  Explicitly Converting Object/Mixed Dtypes to Numerical (Comprehensive) ---")
        for col in cols_to_force_numeric:
            if col in df_prepared.columns:
                # Special handling for RegistrationYear if it has 'MM/YYYY' format strings
                if col == 'RegistrationYear' and df_prepared[col].dtype == 'object':
                    print(f"    Attempting to parse '{col}' (object dtype) for 'MM/YYYY' format to extract year...")
                    try:
                        df_prepared[col] = pd.to_datetime(df_prepared[col], format='%m/%Y', errors='coerce').dt.year
                        df_prepared[col] = pd.to_numeric(df_prepared[col], errors='coerce') # Ensure purely numeric year, convert remaining errors
                    except Exception as e:
                        print(f"    Warning: Specific MM/YYYY parsing for '{col}' failed: {e}. Falling back to general numeric conversion.")
                        df_prepared[col] = pd.to_numeric(df_prepared[col], errors='coerce')
                else:
                    # General numeric conversion. 'errors='coerce' will turn non-numeric values
                    # (including string "inf", "-inf") to NaN.
                    df_prepared[col] = pd.to_numeric(df_prepared[col], errors='coerce')
                
                print(f"    Converted '{col}' to numeric. Current dtype: {df_prepared[col].dtype}. NaNs: {df_prepared[col].isnull().sum()}.")
            else:
                print(f"    Column '{col}' not found for comprehensive numeric conversion.")

        # --- NEW: Drop rows containing actual np.inf or -np.inf in numerical columns ---
        # This step is done *after* pd.to_numeric conversions have had a chance to run.
        # It targets any `np.inf` or `-np.inf` values that might still exist (not converted to NaN by coerce).
        
        initial_rows_before_inf_drop = df_prepared.shape[0]
        # Get numerical columns from the DataFrame *after* attempted conversions
        numeric_cols_in_df = df_prepared.select_dtypes(include=np.number).columns
        
        if not numeric_cols_in_df.empty:
            # Create a boolean mask for rows containing any infinity in any numeric column
            rows_with_infinity_mask = df_prepared[numeric_cols_in_df].isin([np.inf, -np.inf]).any(axis=1)
            
            if rows_with_infinity_mask.any():
                df_prepared = df_prepared[~rows_with_infinity_mask].copy()
                dropped_count = initial_rows_before_inf_drop - df_prepared.shape[0]
                print(f"  Dropped {dropped_count} rows containing infinite numerical values after type conversions.")
                if df_prepared.empty:
                    print("Warning: DataFrame became empty after dropping infinite values. Subsequent steps may not function correctly with an empty DataFrame.")
                    return df_prepared # Return early if empty
            else:
                print("  No infinite values detected in numerical columns after type conversions. No rows dropped for infinities.")
        else:
            print("  No numerical columns to check for infinities. Skipping infinity drop.")
        # --- END NEW: Drop rows containing actual np.inf or -np.inf ---

        # Explicitly convert boolean-like object columns to int (0/1)
        bool_cols_to_int = ['IsVATRegistered', 'AlarmImmobiliser', 'TrackingDevice', 'NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder']
        for col in bool_cols_to_int:
            if col in df_prepared.columns and df_prepared[col].dtype == 'object':
                df_prepared[col] = df_prepared[col].astype(str).str.lower().map({'true': 1, 'false': 0}).fillna(-1).astype(int) 
                print(f"    Converted '{col}' (bool-like object) to int. Current dtype: {df_prepared[col].dtype}. NaNs: {df_prepared[col].isnull().sum()}")
            else:
                print(f"    Column '{col}' not found or already numeric/boolean.")

        print("\n  DataFrame Info after comprehensive explicit type conversion (numerical/boolean) and dropping infinities:")
        df_prepared.info()

        # 2. Convert all remaining 'object' columns to 'category' dtype
        print("\n  Converting remaining 'object' columns to 'category' dtype ---")
        object_cols_after_numeric_conversion = df_prepared.select_dtypes(include=['object']).columns.tolist()
        for col in object_cols_after_numeric_conversion:
            if col not in ['UnderwrittenCoverID', 'PolicyID']: # 'mmcode' should already be handled as numeric
                df_prepared[col] = df_prepared[col].astype('category')
                print(f"    Converted '{col}' to category. Current dtype: {df_prepared[col].dtype}")
            else:
                print(f"    Skipping '{col}' (ID) from category conversion.")
        
        print("\n  DataFrame Info after converting objects to category:")
        df_prepared.info()

        print("\nInitial data preparation completed.")
        return df_prepared

