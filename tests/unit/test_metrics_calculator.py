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

from src.utils.hypothesis_testing.metrics_calculator import calculate_claim_frequency, calculate_margin, calculate_claim_severity

@pytest.fixture
def sample_df_metrics():
    """Fixture for a sample DataFrame to test metrics calculations."""
    data = {
        'PolicyID': [1, 2, 3, 4, 5, 6],
        'TotalPremium': [1000, 1200, 800, 1500, 900, 1100],
        'TotalClaims': [500, 0, 1500, 100, np.nan, 2000] # NaN for no claim, 0 for no claim
    }
    return pd.DataFrame(data)

def test_calculate_claim_frequency_basic(sample_df_metrics):
    """Test basic functionality of calculate_claim_frequency."""
    df_freq = calculate_claim_frequency(sample_df_metrics.copy())
    
    expected_has_claim = pd.Series([1, 0, 1, 1, 0, 1]) # np.nan TotalClaims treated as 0
    pd.testing.assert_series_equal(df_freq['HasClaim'], expected_has_claim, check_names=False)

def test_calculate_claim_frequency_empty_df():
    """Test calculate_claim_frequency with an empty DataFrame."""
    df_empty = pd.DataFrame({'PolicyID': [], 'TotalClaims': []})
    df_freq = calculate_claim_frequency(df_empty)
    assert df_freq.empty
    assert 'HasClaim' in df_freq.columns

def test_calculate_claim_frequency_missing_columns(capsys):
    """Test calculate_claim_frequency with missing required columns."""
    df_no_claims = pd.DataFrame({'PolicyID': [1, 2]})
    df_result = calculate_claim_frequency(df_no_claims.copy())
    assert 'HasClaim' not in df_result.columns
    captured = capsys.readouterr()
    assert "Error: 'TotalClaims' or 'PolicyID' column missing" in captured.out

def test_calculate_margin_basic(sample_df_metrics):
    """Test basic functionality of calculate_margin."""
    df_margin = calculate_margin(sample_df_metrics.copy())
    
    # Expected Margin: 1000-500=500, 1200-0=1200, 800-1500=-700, 1500-100=1400, 900-0=900, 1100-2000=-900
    expected_margin = pd.Series([500.0, 1200.0, -700.0, 1400.0, 900.0, -900.0])
    pd.testing.assert_series_equal(df_margin['Margin'], expected_margin, check_names=False)

def test_calculate_margin_empty_df():
    """Test calculate_margin with an empty DataFrame."""
    df_empty = pd.DataFrame({'TotalPremium': [], 'TotalClaims': []})
    df_margin = calculate_margin(df_empty)
    assert df_margin.empty
    assert 'Margin' in df_margin.columns

def test_calculate_margin_missing_columns(capsys):
    """Test calculate_margin with missing required columns."""
    df_no_premium = pd.DataFrame({'TotalClaims': [100, 50]})
    df_result = calculate_margin(df_no_premium.copy())
    assert 'Margin' not in df_result.columns
    captured = capsys.readouterr()
    assert "Error: 'TotalPremium' or 'TotalClaims' column missing" in captured.out

def test_calculate_claim_severity_basic(sample_df_metrics):
    """Test basic functionality of calculate_claim_severity."""
    # This function returns a filtered DataFrame, not a new column
    claims_only_df = calculate_claim_severity(sample_df_metrics.copy())
    
    # Expected: PolicyID 1, 3, 4, 6 (with TotalClaims 500, 1500, 100, 2000)
    expected_data = {
        'PolicyID': [1, 3, 4, 6],
        'TotalPremium': [1000, 800, 1500, 1100],
        'TotalClaims': [500, 1500, 100, 2000],
        'OtherCol': ['A', 'C', 'D', 'F']
    }
    expected_df = pd.DataFrame(expected_data)
    # Check only filtered rows and columns (excluding index)
    pd.testing.assert_frame_equal(claims_only_df.reset_index(drop=True), expected_df.reset_index(drop=True))

def test_calculate_claim_severity_no_claims():
    """Test calculate_claim_severity when no claims occurred."""
    df_no_claims = pd.DataFrame({
        'PolicyID': [1, 2],
        'TotalPremium': [1000, 1200],
        'TotalClaims': [0, 0]
    })
    result_df = calculate_claim_severity(df_no_claims.copy())
    assert result_df.empty

def test_calculate_claim_severity_missing_claims_col(capsys):
    """Test calculate_claim_severity with missing TotalClaims column."""
    df_no_claims_col = pd.DataFrame({'PolicyID': [1, 2], 'TotalPremium': [100, 200]})
    result_df = calculate_claim_severity(df_no_claims_col.copy())
    assert result_df.empty # It will be empty as no claims are filtered
    captured = capsys.readouterr()
    assert "Error: 'TotalClaims' column missing" in captured.out
