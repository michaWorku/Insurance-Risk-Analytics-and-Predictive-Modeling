import os
import zipfile
from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path

# Define an abstract class for Data Ingestor
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """
        Abstract method to ingest data from a given file.
        Implementations should handle specific file types and error conditions.
        """
        pass


# Implement a concrete class for ZIP Ingestion
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: Path, extract_to_dir: Path = None) -> pd.DataFrame:
        """
        Extracts a .zip file and loads the first found .csv or .txt file into a pandas DataFrame.

        Args:
            file_path (Path): The path to the .zip file.
            extract_to_dir (Path, optional): The directory where contents will be extracted.
                                            If None, defaults to a 'temp_extracted_data'
                                            subdirectory within the zip file's parent directory.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the loaded data.

        Raises:
            ValueError: If the provided file is not a .zip file, or if multiple
                        supported data files are found without clear instruction.
            FileNotFoundError: If the .zip file does not exist or no supported
                               CSV/TXT file is found inside the zip.
            IOError: If there's an error during zip extraction or file reading.
        """
        if not file_path.suffix.lower() == ".zip":
            raise ValueError(f"The provided file '{file_path.name}' is not a .zip file.")

        if not file_path.is_file():
            raise FileNotFoundError(f"Zip file not found at: {file_path}")

        # Determine the extraction directory
        if extract_to_dir is None:
            extract_to_dir = file_path.parent / "temp_extracted_data"
            print(f"No 'extract_to_dir' specified. Extracting to temporary directory: {extract_to_dir}")
        
        # Ensure the extraction directory exists
        os.makedirs(extract_to_dir, exist_ok=True)

        try:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(extract_to_dir)
            print(f"Successfully extracted '{file_path.name}' to '{extract_to_dir}'.")
        except zipfile.BadZipFile:
            raise IOError(f"Error: '{file_path.name}' is a bad or corrupted zip file.")
        except Exception as e:
            raise IOError(f"Error during zip extraction of '{file_path.name}': {e}")

        # Find the extracted CSV or TXT file
        valid_extensions = (".csv", ".txt")
        # Use rglob for recursive search in case files are in subfolders within the zip
        extracted_files = [f for f in extract_to_dir.rglob('*') if f.is_file() and f.suffix.lower() in valid_extensions]

        if not extracted_files:
            raise FileNotFoundError(f"No CSV or TXT file found in the extracted data from '{file_path.name}'.")

        file_to_load = None
        csv_files = [f for f in extracted_files if f.suffix.lower() == ".csv"]
        txt_files = [f for f in extracted_files if f.suffix.lower() == ".txt"]

        if len(csv_files) == 1:
            file_to_load = csv_files[0]
        elif len(txt_files) == 1 and not csv_files: # Only TXT and exactly one
            file_to_load = txt_files[0]
        elif len(csv_files) > 1:
            raise ValueError(f"Multiple CSV files found in zip: {[f.name for f in csv_files]}. "
                             "Please ensure only one relevant CSV or TXT file is present or refine selection logic.")
        elif len(txt_files) > 1:
             raise ValueError(f"Multiple TXT files found in zip: {[f.name for f in txt_files]}. "
                              "Please ensure only one relevant CSV or TXT file is present or refine selection logic.")
        elif csv_files and txt_files:
            # If both CSV and TXT are present, default to CSV
            print(f"Warning: Both CSV and TXT files found in '{file_path.name}'. Prioritizing CSV: {csv_files[0].name}")
            file_to_load = csv_files[0]
        else: # Fallback: if somehow multiple types but none of above single-file conditions met, pick first.
              # This should ideally be covered by the specific checks above.
            file_to_load = extracted_files[0]
            print(f"Warning: Picking the first found data file: {file_to_load.name}")

        if file_to_load is None: # Should be caught by FileNotFoundError earlier, but good defensive programming
            raise FileNotFoundError("Could not determine a single CSV or TXT file to load from zip.")

        print(f"Attempting to load data from: {file_to_load.name}")
        df = pd.DataFrame() # Initialize empty DataFrame

        try:
            if file_to_load.suffix.lower() == ".csv":
                df = pd.read_csv(file_to_load)
            elif file_to_load.suffix.lower() == ".txt":
                # Try common delimiters for TXT files
                try:
                    df = pd.read_csv(file_to_load, sep=',', engine='python')
                except pd.errors.ParserError:
                    try:
                        df = pd.read_csv(file_to_load, sep='\t', engine='python')
                    except pd.errors.ParserError:
                        # Fallback for space separated or other delimiters. '\s+' handles multiple spaces.
                        # Using header=None and inferring might be needed for very unstructured text.
                        df = pd.read_csv(file_to_load, sep=r'\s+', engine='python')
        except Exception as e:
            raise IOError(f"Error reading data file '{file_to_load.name}' from zip: {e}")

        return df


# Implement a Factory to create DataIngestors
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """
        Returns the appropriate DataIngestor based on file extension.

        Args:
            file_extension (str): The file extension (e.g., ".zip", ".csv").

        Returns:
            DataIngestor: An instance of a concrete DataIngestor.

        Raises:
            ValueError: If no ingestor is available for the given file extension.
        """
        if file_extension.lower() == ".zip":
            return ZipDataIngestor()
        # Add more ingestor types here as needed, e.g., for direct CSV, Excel, Parquet
        # elif file_extension.lower() == ".csv":
        #     return CsvDataIngestor()
        # elif file_extension.lower() == ".xlsx":
        #     return ExcelDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}")


# Example usage:
if __name__ == "__main__":
    # --- IMPORTANT: Adjust this file_path to your actual zip file location ---
    # Example: If your zip is in project_root/data/raw/
    # Get current working directory (usually project root if run from makefile/terminal)
    current_dir = Path.cwd()
    # Assuming the zip file is in 'data/raw/' relative to the current working directory
    # Replace 'MachineLearningRating_v3.zip' with your actual zip file name
    zip_file_name = "MachineLearningRating_v3.zip" 
    zip_file_path = current_dir / "data" / "raw" / zip_file_name

    # Define where to extract the files
    # Option 1: Extract to data/raw/temp_extracted_data (default if not specified for ingest method)
    # Option 2: Extract directly into data/raw/ (might clutter raw folder if many zips)
    # Option 3: Extract to data/processed/ (if you consider extraction a processing step)
    extraction_target_dir = current_dir / "data" / "raw/temp_extracted_data" # Example: extract to processed folder

    # Ensure the target directory exists for extraction
    os.makedirs(extraction_target_dir, exist_ok=True)

    try:
        # Determine the file extension
        file_extension = zip_file_path.suffix

        # Get the appropriate DataIngestor
        data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

        # Ingest the data and load it into a DataFrame
        # Pass the extraction_target_dir to the ingest method for explicit control
        df = data_ingestor.ingest(zip_file_path, extract_to_dir=extraction_target_dir)

        # Now df contains the DataFrame from the extracted CSV/TXT
        print(f"\nSuccessfully ingested data from '{zip_file_name}'. First 5 rows:")
        print(df.head())
        print("\nDataFrame Info:")
        df.info()

        # You might want to clean up the temporary extraction directory here if it was used
        # (e.g., if extract_to_dir was the default temp_extracted_data)
        # import shutil
        # if "temp_extracted_data" in str(extraction_target_dir): # Or based on a flag from ingest method
        #     shutil.rmtree(extraction_target_dir)
        #     print(f"Cleaned up temporary extraction directory: {extraction_target_dir}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the zip file path is correct and the file exists.")
    except ValueError as e:
        print(f"Error: {e}")
        print("Check the file extension or the contents of the zip file (e.g., multiple CSV/TXT files).")
    except IOError as e:
        print(f"Error during file operation: {e}")
        print("There might be an issue with file permissions, disk space, or the zip file integrity.")
    except Exception as e:
        print(f"An unexpected error occurred during data ingestion: {e}")

