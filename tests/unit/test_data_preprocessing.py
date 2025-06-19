import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to system path to allow imports from src.data_loader
# Assuming tests/unit is at project_root/tests/unit
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
print("path:", str(Path(__file__).parent.parent.parent / 'src'))

from src.utils.data_preprocessing import perform_initial_data_preprocessing, save_processed_data

@pytest.fixture
def sample_raw_df():
    """Fixture for a sample raw DataFrame."""
    data = {
        'UnderwrittenCoverID': [1, 2, 3, 4, 5],
        'PolicyID': [1001, 1002, 1003, 1004, 1005],
        'TransactionMonth': [202301, 202302, '202303', 'InvalidDate', 202305],
        'IsVATRegistered': [True, False, True, False, True],
        'TotalPremium': ['1000', 1200, 800, 'abc', 900], # Mixed type, invalid num
        'TotalClaims': [500, 0, 1500, 100, np.nan], # NaN value
        'PostalCode': ['1234', '5678', '9101', '1234', '1313'],
        'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
        'CustomValueEstimate': [150000, 500000, None, 450000, 200000], # None value
        'VehicleIntroDate': ['2015-01-01', '2018-03-15', '2010-06-01', '2022-02-20', '2019-05-10']
    }
    return pd.DataFrame(data)

def test_perform_initial_data_preprocessing_basic(sample_raw_df):
    """Test basic functionality of preprocessing."""
    df_processed = perform_initial_data_preprocessing(sample_raw_df.copy())

    # Check TransactionMonth conversion
    assert pd.api.types.is_datetime64_any_dtype(df_processed['TransactionMonth'])
    assert df_processed['TransactionMonth'].iloc[0] == pd.Timestamp('2023-01-01')
    assert pd.isna(df_processed['TransactionMonth'].iloc[3]) # 'InvalidDate' should be NaT

    # Check numerical conversion and NaNs
    assert pd.api.types.is_numeric_dtype(df_processed['TotalPremium'])
    assert pd.isna(df_processed['TotalPremium'].iloc[3]) # 'abc' should be NaN
    assert pd.isna(df_processed['TotalClaims'].iloc[4]) # Original NaN should remain

    # Check categorical conversion
    assert pd.api.types.is_categorical_dtype(df_processed['Gender'])
    assert pd.api.types.is_categorical_dtype(df_processed['PostalCode'])

    # Check None conversion to NaN for numerical
    assert pd.isna(df_processed['CustomValueEstimate'].iloc[2])

def test_perform_initial_data_preprocessing_empty_df():
    """Test handling of an empty DataFrame."""
    df_empty = pd.DataFrame()
    df_processed = perform_initial_data_preprocessing(df_empty)
    assert df_processed.empty

def test_perform_initial_data_preprocessing_missing_cols(capsys):
    """Test handling when expected columns are missing."""
    df_partial = pd.DataFrame({
        'col_A': [1, 2, 3],
        'col_B': ['X', 'Y', 'Z']
    })
    df_processed = perform_initial_data_preprocessing(df_partial)
    
    # Check that it runs without error and prints warnings
    captured = capsys.readouterr()
    assert "Warning: 'TransactionMonth' column not found." in captured.out
    assert "Warning: Column 'TotalPremium' not found." in captured.out
    assert "Warning: Column 'TotalClaims' not found." in captured.out

def test_save_processed_data(tmp_path):
    """Test saving processed data to a CSV file."""
    output_path = tmp_path / "processed_data.csv"
    data_to_save = pd.DataFrame({'col1': [1, 2], 'col2': ['A', 'B']})
    
    save_processed_data(data_to_save, output_path)
    
    assert output_path.is_file()
    loaded_df = pd.read_csv(output_path, sep='|') # Assuming pipe-separated as per data_preprocessing
    pd.testing.assert_frame_equal(loaded_df, data_to_save)

