import pandas as pd
from pathlib import Path

def load_data(file_path: Path, delimiter: str = ',', file_type: str = 'csv') -> pd.DataFrame:
    """
    Loads the data from a specified file path.

    Args:
        file_path (Path): The path to the data file.
        delimiter (str): The delimiter to use for parsing (e.g., ',', '|', '\t').
        file_type (str): The type of file ('csv' or 'txt').

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded data.
                      Returns an empty DataFrame if the file does not exist or
                      an error occurs during loading.
    """
    if not file_path.is_file():
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame() # Return empty DataFrame on file not found

    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path, sep=delimiter)
        elif file_type == 'txt':
            # For TXT, try with specified delimiter, then fallbacks if needed
            try:
                df = pd.read_csv(file_path, sep=delimiter, engine='python')
            except pd.errors.ParserError:
                try:
                    df = pd.read_csv(file_path, sep='\t', engine='python')
                except pd.errors.ParserError:
                    df = pd.read_csv(file_path, sep=r'\s+', engine='python') # Fallback to whitespace
        else:
            print(f"Error: Unsupported file type '{file_type}'. Only 'csv' and 'txt' are supported.")
            return pd.DataFrame()

        print(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame() # Return empty DataFrame on loading error

# if __name__ == "__main__":
#     # Example usage when run directly:
#     current_script_dir = Path(__file__).parent
#     data_file_path = current_script_dir.parent / "data" / "raw" / "Car_Insurance_Claim.csv"
    
#     if not data_file_path.is_file():
#         print(f"Attempting alternative path for data file: {Path.cwd() / 'data' / 'raw' / 'Car_Insurance_Claim.csv'}")
#         data_file_path = Path.cwd() / "data" / "raw" / "Car_Insurance_Claim.csv"
        
#     df = load_data(data_file_path) # Now uses default delimiter ',' and file_type 'csv'
#     if not df.empty:
#         print("\nFirst 5 rows of the loaded data:")
#         print(df.head())
#         print("\nData Info:")
#         df.info()

