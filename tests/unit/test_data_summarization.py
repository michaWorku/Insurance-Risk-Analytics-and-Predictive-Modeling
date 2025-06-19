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

from src.utils.eda.data_summarization import calculate_loss_ratio, get_overall_loss_ratio

@pytest.fixture
def sample_df():
    """Fixture for a sample DataFrame with premium and claims data."""
    data = {
        'PolicyID': [1, 2, 3, 4, 5, 6],
        'TotalPremium': [1000, 2000, 500, 1500, 0, 1000],
        'TotalClaims': [500, 0, 1000, 200, 0, np.nan], # NaN to test handling
        'OtherCol': ['A', 'B', 'C', 'D', 'E', 'F']
    }
    return pd.DataFrame(data)

def test_calculate_loss_ratio_basic(sample_df):
    """Test calculate_loss_ratio with valid data."""
    df_with_lr = calculate_loss_ratio(sample_df.copy())
    
    expected_lr = pd.Series([0.5, 0.0, 2.0, 200/1500, 0.0, 0.0]) # 1000/0 is inf, then 0.2, 0 for NaN
    # Handle the np.finfo(float).eps in denominator for 0 premium and NaN claims
    expected_lr_calc = [
        500/1000, # 0.5
        0/2000,   # 0.0
        1000/500, # 2.0
        200/1500, # 0.1333...
        0/(0 + np.finfo(float).eps), # Should become 0 due to fillna
        0/(1000 + np.finfo(float).eps) # NaN claim becomes 0, so 0/1000 becomes 0
    ]
    # Small tolerance for float comparisons
    assert 'LossRatio' in df_with_lr.columns
    for i, val in enumerate(expected_lr_calc):
        if pd.isna(sample_df['TotalClaims'].iloc[i]): # Claims was NaN
            assert df_with_lr['LossRatio'].iloc[i] == 0.0
        elif sample_df['TotalPremium'].iloc[i] == 0: # Premium was 0
            assert df_with_lr['LossRatio'].iloc[i] == 0.0 # From fillna(0) after inf
        else:
            assert np.isclose(df_with_lr['LossRatio'].iloc[i], val)


def test_calculate_loss_ratio_missing_columns(sample_df, capsys):
    """Test calculate_loss_ratio with missing required columns."""
    df_missing_premium = sample_df.drop(columns=['TotalPremium'])
    df_result = calculate_loss_ratio(df_missing_premium.copy())
    assert 'LossRatio' not in df_result.columns
    captured = capsys.readouterr()
    assert "Error: 'TotalClaims' or 'TotalPremium' column missing" in captured.out
    
    df_missing_claims = sample_df.drop(columns=['TotalClaims'])
    df_result = calculate_loss_ratio(df_missing_claims.copy())
    assert 'LossRatio' not in df_result.columns

def test_calculate_loss_ratio_empty_df():
    """Test calculate_loss_ratio with an empty DataFrame."""
    df_empty = pd.DataFrame({'TotalPremium': [], 'TotalClaims': []})
    df_with_lr = calculate_loss_ratio(df_empty)
    assert df_with_lr.empty
    assert 'LossRatio' in df_with_lr.columns # Column should still be added but empty

def test_get_overall_loss_ratio_basic(sample_df):
    """Test get_overall_loss_ratio with valid data."""
    # Ensure NaNs are filled for sum calculation in input
    sample_df_cleaned = sample_df.copy()
    sample_df_cleaned['TotalClaims'] = sample_df_cleaned['TotalClaims'].fillna(0)

    overall_metrics = get_overall_loss_ratio(sample_df_cleaned)
    assert overall_metrics is not None
    assert overall_metrics['TotalPremium'] == 1000 + 2000 + 500 + 1500 + 0 + 1000 # 6000
    assert overall_metrics['TotalClaims'] == 500 + 0 + 1000 + 200 + 0 + 0 # 1700 (NaN becomes 0)
    assert np.isclose(overall_metrics['OverallLossRatio'], 1700 / 6000)

def test_get_overall_loss_ratio_zero_premium():
    """Test get_overall_loss_ratio with zero total premium."""
    df_zero_premium = pd.DataFrame({
        'PolicyID': [1, 2],
        'TotalPremium': [0, 0],
        'TotalClaims': [100, 50]
    })
    overall_metrics = get_overall_loss_ratio(df_zero_premium)
    assert overall_metrics is not None
    assert overall_metrics['TotalPremium'] == 0
    assert overall_metrics['TotalClaims'] == 150
    assert overall_metrics['OverallLossRatio'] == 0 # Should be 0 if total premium is 0

def test_get_overall_loss_ratio_missing_columns(sample_df, capsys):
    """Test get_overall_loss_ratio with missing required columns."""
    df_missing_premium = sample_df.drop(columns=['TotalPremium'])
    overall_metrics = get_overall_loss_ratio(df_missing_premium)
    assert overall_metrics is None
    captured = capsys.readouterr()
    assert "Error: 'TotalClaims' or 'TotalPremium' column missing" in captured.out
