import pandas as pd
import numpy as np
from pathlib import Path
import os


def perform_initial_data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs initial data preprocessing steps on the insurance DataFrame.
    This includes:
    1. Converting 'TransactionMonth' to datetime objects.
    2. Identifying and reporting missing values.
    3. Correcting data types for known categorical and numerical columns.

    Args:
        df (pd.DataFrame): The raw insurance claims DataFrame.

    Returns:
        pd.DataFrame: The DataFrame after initial preprocessing.
                      Returns the original DataFrame if it's empty.
    """
    if df.empty:
        print("Input DataFrame is empty. Skipping preprocessing.")
        return df.copy() # Return a copy to prevent modifying original if desired

    print("\n--- Performing Initial Data Preprocessing ---")

    # 1. Convert 'TransactionMonth' to datetime
    if 'TransactionMonth' in df.columns:
        print("Attempting to convert 'TransactionMonth' to datetime...")
        # Try specific format first if known, then infer
        try:
            # Assuming 'YYYYMM' or 'YYYY-MM' is a common format, or pandas can infer
            df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
            print("Converted 'TransactionMonth' to datetime objects.")
        except Exception as e:
            print(f"Error converting 'TransactionMonth' to datetime: {e}. Column will remain as is.")
        
        # Handle cases where conversion might result in NaT (Not a Time)
        if df['TransactionMonth'].isnull().any():
            invalid_dates_count = df['TransactionMonth'].isnull().sum()
            print(f"Warning: 'TransactionMonth' has {invalid_dates_count} invalid date entries after conversion (converted to NaT).")
    else:
        print("Warning: 'TransactionMonth' column not found for datetime conversion.")

    # 2. Identify and report missing values
    print("\n--- Checking for Missing Values ---")
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
    if not missing_values.empty:
        print("Missing values by column:")
        print(missing_values)
        print(f"\nTotal missing values across DataFrame: {df.isnull().sum().sum()}")
    else:
        print("No missing values found in the DataFrame.")

    # 3. Correcting data types for known columns
    print("\n--- Correcting Data Types ---")
    
    # Define columns that should be categorical
    # This list should be updated based on your actual data schema
    categorical_cols = [
        'IsVATRegistered', 'Citizenship', 'LegalType', 'Title', 'Language', 
        'Bank', 'AccountType', 'MaritalStatus', 'Gender', 'Country', 
        'Province', 'PostalCode', 'MainCrestazone', 'SubCrestazone', 
        'ItemType', 'VehicleType', 'Make', 'Model', 'Bodytype', 
        'AlarmImmobiliser', 'Tracking Device', 'NewVehicle', 
        'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder', 
        'CoverCategory', 'CoverType', 'CoverGroup', 'Section', 'Product', 
        'StatutoryClass', 'StatutoryRiskType'
    ]
    
    # Define columns that should be numerical (and might be objects/strings)
    # This list should be updated based on your actual data schema
    numerical_cols = [
        'Mmcode', 'RegistrationYear', 'Cylinders', 'Cubiccapacity', 
        'Kilowatts', 'NumberOfDoors', 'CustomValueEstimate', 
        'CapitalOutstanding', 'NumberOfVehiclesInFleet', 
        'SumInsured', 'TermFrequency', 'CalculatedPremiumPerTerm', 
        'ExcessSelected', 'TotalPremium', 'TotalClaims'
    ]

    for col in categorical_cols:
        if col in df.columns:
            # Convert to 'category' dtype for efficiency and proper handling in some libraries
            df[col] = df[col].astype('category')
            print(f"Converted '{col}' to category.") # Uncomment for verbose output
        else:
            print(f"Warning: Categorical column '{col}' not found.") # Uncomment for verbose output

    for col in numerical_cols:
        if col in df.columns:
            # Attempt to convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"Converted '{col}' to numeric.") # Uncomment for verbose output
        else:
            print(f"Warning: Numerical column '{col}' not found.") # Uncomment for verbose output
            
    print("\nInitial data types after correction attempt:")
    print(df.dtypes.value_counts())
    
    print("\nInitial data preprocessing complete.")
    return df

def save_processed_data(df: pd.DataFrame, output_file_path: Path):
    """
    Saves the processed DataFrame to a specified CSV file path.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        output_file_path (Path): The full path including filename where the
                                 DataFrame should be saved.
    """
    try:
        # Ensure the parent directory exists
        os.makedirs(output_file_path.parent, exist_ok=True)
        df.to_csv(output_file_path, index=False)
        print(f"\n✅ Successfully saved processed data to: {output_file_path}")
    except Exception as e:
        print(f"❌ Error saving processed data to {output_file_path}: {e}")


if __name__ == "__main__":
    # --- Example Usage for data_preprocessing.py ---
    # This block runs only when data_preprocessing.py is executed directly.
    
    # We need to import load_data from our data_loader module
    # Adjust path if data_loader.py is not in the same directory (e.g., ../src)
    import sys
    sys.path.append(str(Path(__file__).parent)) # Add current directory to path for import
    from data_loader import load_data

    # Define the path to your raw data file (e.g., Car_Insurance_Claim.csv)
    # Using the path specified in your query for the dummy data source
    data_file_path = Path(__file__).parent.parent.parent / "data" / "raw" / "temp_extracted_data" / "MachineLearningRating_v3.txt"
    
    # Define the output path for the processed data
    # This will save the processed data into project_root/data/processed/
    processed_output_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    processed_output_file_name = "processed_insurance_data.csv"
    processed_output_path = processed_output_dir / processed_output_file_name

    # Create a dummy CSV file for demonstration if the actual one is not present
    # Ensure the parent directories for the dummy data exist
    if not data_file_path.is_file():
        print(f"Creating a dummy 'MachineLearningRating_v3.txt' for demonstration at: {data_file_path}")
        os.makedirs(data_file_path.parent, exist_ok=True)
        dummy_content = """PolicyID,TransactionMonth,TotalPremium,TotalClaims,Gender,Province,RegistrationYear,CustomValueEstimate
1,202301,1000,500,Male,Gauteng,2015,25000
2,202301,1200,0,Female,KZN,2018,30000
3,202302,800,1500,Male,Western Cape,2010,18000
4,202302,1500,100,Female,Gauteng,2022,40000
5,202303,900,NaN,Male,Limpopo,2019,22000
6,202303,1100,20000,Female,Eastern Cape,2017,2800000
7,202304,950,700,Male,Free State,2016,26000
8,202304,1300,0,Female,North West,2020,35000
9,202305,750,50,Male,Mpumalanga,2014,16000
10,202305,1050,NaN,Female,Northern Cape,2021,29000
"""
        with open(data_file_path, 'w') as f:
            f.write(dummy_content)
        print("Dummy data created. Remember to replace it with your actual data!")

    # Load the data using our data_loader module
    raw_df = load_data(data_file_path, delimiter="|", file_type='txt')

    if not raw_df.empty:
        # Perform initial preprocessing on the loaded DataFrame
        processed_df = perform_initial_data_preprocessing(raw_df.copy()) # Use .copy() to avoid SettingWithCopyWarning

        print("\n--- Processed Data Summary ---")
        print("Processed Data Info:")
        processed_df.info()
        print("\nFirst 5 rows of processed data:")
        print(processed_df.head())
        print("\nMissing values after preprocessing:")
        print(processed_df.isnull().sum()[processed_df.isnull().sum() > 0])
        
        # Save the processed data to the 'data/processed' directory
        save_processed_data(processed_df, processed_output_path)

        # Clean up dummy file after demonstration
        # Note: This will remove MachineLearningRating_v3.txt and its parent temp_extracted_data if empty
        # if data_file_path.is_file():
        #     data_file_path.unlink()
        #     print(f"Cleaned up dummy input file: {data_file_path.name}")
        #     if not os.listdir(data_file_path.parent):
        #         data_file_path.parent.rmdir()
        #         print(f"Removed empty dummy input directory: {data_file_path.parent}")

