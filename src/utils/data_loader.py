# src/data_loader.py

import pandas as pd
from pathlib import Path
import os # Import os for os.makedirs and os.listdir in example usage

def load_data(file_path: Path, delimiter: str = None, file_type: str = None) -> pd.DataFrame:
    """
    Loads data from a specified file path into a pandas DataFrame.
    It can infer file type and delimiter or accept them as explicit arguments.

    Args:
        file_path (Path): The path to the data file.
        delimiter (str, optional): The delimiter to use when reading the file
                                   (e.g., ',', '|', '\t', '\s+'). If None,
                                   the function attempts to infer.
        file_type (str, optional): The type of the file (e.g., 'csv', 'txt').
                                   If None, infers from file extension.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded data.
                      Returns an empty DataFrame if the file does not exist,
                      is empty, unsupported, or an error occurs during loading.
    """
    if not file_path.is_file():
        print(f"Error: File not found at '{file_path}'")
        return pd.DataFrame()

    # Determine file type
    if file_type:
        file_type = file_type.lower().strip()
    else:
        file_type = file_path.suffix.lower().strip().lstrip('.') # Remove leading dot

    if file_type not in ['csv', 'txt']:
        print(f"Error: Unsupported file type '{file_type}'. Only 'csv' and 'txt' are supported by this loader.")
        return pd.DataFrame()

    try:
        # If delimiter is explicitly provided, use it
        if delimiter:
            df = pd.read_csv(file_path, sep=delimiter, engine='python')
            print(f"Successfully loaded data from '{file_path.name}' using specified delimiter '{delimiter}'. Shape: {df.shape}")
        else:
            # Delimiter inference logic
            if file_type == 'csv':
                # For CSV, default to comma, no complex inference unless specified
                df = pd.read_csv(file_path)
                print(f"Successfully loaded CSV data from '{file_path.name}' (inferred comma delimiter). Shape: {df.shape}")
            elif file_type == 'txt':
                # For TXT, try pipe, then comma, then tab, then whitespace
                print(f"Attempting to infer delimiter for TXT file '{file_path.name}'...")
                try:
                    df = pd.read_csv(file_path, sep='|', engine='python')
                    print(f"  Inferred pipe ('|') delimiter.")
                except pd.errors.ParserError:
                    try:
                        df = pd.read_csv(file_path, sep=',', engine='python')
                        print(f"  Inferred comma (',') delimiter.")
                    except pd.errors.ParserError:
                        try:
                            df = pd.read_csv(file_path, sep='\t', engine='python')
                            print(f"  Inferred tab ('\\t') delimiter.")
                        except pd.errors.ParserError:
                            df = pd.read_csv(file_path, sep=r'\s+', engine='python') # Fallback to whitespace
                            print(f"  Inferred whitespace ('\\s+') delimiter.")
                print(f"Successfully loaded data from '{file_path.name}' with inferred delimiter. Shape: {df.shape}")

        if df.empty:
            print(f"Warning: File '{file_path.name}' loaded successfully but resulted in an empty DataFrame.")
        
        return df

    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path.name}' is empty.")
        return pd.DataFrame()
    except pd.errors.ParserError as pe:
        print(f"Error parsing data from '{file_path.name}'. Check delimiter or file format: {pe}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred loading data from '{file_path.name}': {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # --- Example Usage for data_loader.py ---
    # This block runs only when data_loader.py is executed directly.

    current_dir = Path.cwd()
    data_raw_dir = current_dir / "data" / "raw"
    os.makedirs(data_raw_dir, exist_ok=True)

    print("\n--- Testing load_data function ---")

    # Scenario 1: CSV file with default comma delimiter
    csv_file_example = data_raw_dir / "example_data.csv"
    dummy_csv_content = """col1,col2,col3
1,A,202301
2,B,202302
3,C,202303
"""
    with open(csv_file_example, 'w') as f:
        f.write(dummy_csv_content)
    print(f"\n--- Loading {csv_file_example.name} (default CSV, no explicit args) ---")
    df_csv_default = load_data(csv_file_example)
    if not df_csv_default.empty:
        print(df_csv_default.head())

    # Scenario 2: TXT file with pipe delimiter (inferred)
    pipe_txt_file_example = data_raw_dir / "example_data_pipe.txt"
    dummy_pipe_content = """col1|col2|col3
1|A_pipe|202301
2|B_pipe|202302
3|C_pipe|202303
"""
    with open(pipe_txt_file_example, 'w') as f:
        f.write(dummy_pipe_content)
    print(f"\n--- Loading {pipe_txt_file_example.name} (TXT, pipe inferred) ---")
    df_txt = data_raw_dir /  "temp_extracted_data" / "MachineLearningRating_v3.txt"
    df_pipe_inferred = load_data(df_txt, delimiter="|", file_type='txt')
    if not df_pipe_inferred.empty:
        print(df_pipe_inferred.head())

    # Scenario 3: TXT file with tab delimiter (explicitly specified)
    tab_txt_file_example = data_raw_dir / "example_data_tab.txt"
    dummy_tab_content = """col1\tcol2\tcol3
1\tA_tab\t202301
2\tB_tab\t202302
3\tC_tab\t202303
"""
    with open(tab_txt_file_example, 'w') as f:
        f.write(dummy_tab_content)
    print(f"\n--- Loading {tab_txt_file_example.name} (TXT, tab explicit) ---")
    df_tab_explicit = load_data(tab_txt_file_example, delimiter='\t')
    if not df_tab_explicit.empty:
        print(df_tab_explicit.head())
        
    # Scenario 4: Non-existent file
    non_existent_file = data_raw_dir / "non_existent.csv"
    print(f"\n--- Loading {non_existent_file.name} (non-existent) ---")
    df_non_existent = load_data(non_existent_file)
    print(f"DataFrame empty? {df_non_existent.empty}")

    # Scenario 5: Empty file
    empty_file = data_raw_dir / "empty.csv"
    open(empty_file, 'w').close() # Create an empty file
    print(f"\n--- Loading {empty_file.name} (empty) ---")
    df_empty = load_data(empty_file)
    print(f"DataFrame empty? {df_empty.empty}")

    # Scenario 6: Unsupported file type
    unsupported_file = data_raw_dir / "image.jpg"
    open(unsupported_file, 'w').close() # Create dummy file
    print(f"\n--- Loading {unsupported_file.name} (unsupported type) ---")
    df_unsupported = load_data(unsupported_file)
    print(f"DataFrame empty? {df_unsupported.empty}")

    # Clean up dummy files and directories
    print("\n--- Cleaning up dummy files ---")
    for f in [csv_file_example, pipe_txt_file_example, tab_txt_file_example, non_existent_file, empty_file, unsupported_file]:
        if f.is_file():
            f.unlink()
            print(f"Removed: {f.name}")
    if not os.listdir(data_raw_dir):
        data_raw_dir.rmdir()
        print(f"Removed empty directory: {data_raw_dir.name}")
