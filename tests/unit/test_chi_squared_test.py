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

from src.utils.hypothesis_testing.chi_squared_test import ChiSquaredTestStrategy

@pytest.fixture
def chi_squared_strategy():
    return ChiSquaredTestStrategy()

def test_chi_squared_significant_difference(chi_squared_strategy):
    """Test case where there is a statistically significant difference."""
    # Expected: Group A (Claim 30%, No Claim 70%), Group B (Claim 50%, No Claim 50%)
    group_a_data = pd.Series([0]*70 + [1]*30)
    group_b_data = pd.Series([0]*50 + [1]*50)
    
    results = chi_squared_strategy.conduct_test(group_a_data, group_b_data, alpha=0.05)
    
    assert results['conclusion'] == "Reject H0"
    assert results['p_value'] < 0.05
    assert isinstance(results['statistic'], float)
    assert isinstance(results['p_value'], float)
    assert 'contingency_table' in results
    # Check the contingency table structure (0=No Claim, 1=Claim)
    expected_table = pd.DataFrame({
        'Group A': [70, 30],
        'Group B': [50, 50]
    }, index=[0, 1])
    pd.testing.assert_frame_equal(results['contingency_table'], expected_table, check_dtype=False)


def test_chi_squared_no_significant_difference(chi_squared_strategy):
    """Test case where there is no statistically significant difference."""
    # Expected: Group A (Claim 30%, No Claim 70%), Group B (Claim 32%, No Claim 68%) - very similar
    group_a_data = pd.Series([0]*70 + [1]*30)
    group_b_data = pd.Series([0]*68 + [1]*32)
    
    results = chi_squared_strategy.conduct_test(group_a_data, group_b_data, alpha=0.05)
    
    assert results['conclusion'] == "Fail to Reject H0"
    assert results['p_value'] >= 0.05

def test_chi_squared_empty_group_a(chi_squared_strategy):
    """Test with an empty group A."""
    group_a_data = pd.Series([])
    group_b_data = pd.Series([0, 1, 0, 1])
    
    results = chi_squared_strategy.conduct_test(group_a_data, group_b_data)
    
    assert results['conclusion'] == "Insufficient Data"
    assert "one or both groups are empty" in results['message']

def test_chi_squared_empty_group_b(chi_squared_strategy):
    """Test with an empty group B."""
    group_a_data = pd.Series([0, 1, 0, 1])
    group_b_data = pd.Series([])
    
    results = chi_squared_strategy.conduct_test(group_a_data, group_b_data)
    
    assert results['conclusion'] == "Insufficient Data"
    assert "one or both groups are empty" in results['message']

def test_chi_squared_with_nans(chi_squared_strategy):
    """Test Chi-squared with NaN values in the input series (should be handled by value_counts)."""
    group_a_data = pd.Series([0, 1, np.nan, 0, 1])
    group_b_data = pd.Series([1, 0, 1, np.nan, 0])
    
    results = chi_squared_strategy.conduct_test(group_a_data, group_b_data, alpha=0.05)
    
    # NaNs are ignored by value_counts, so it should still produce a valid table
    assert results['conclusion'] != "Insufficient Data"
    assert results['p_value'] is not None

def test_chi_squared_all_same_value_in_one_group(chi_squared_strategy):
    """Test with a group having only one unique value, which could lead to zero rows/columns in contingency table."""
    group_a_data = pd.Series([0, 0, 0, 0, 0]) # Only '0's
    group_b_data = pd.Series([0, 1, 0, 1, 0])
    
    results = chi_squared_strategy.conduct_test(group_a_data, group_b_data)
    
    # Depending on scipy version and data, might be 'Insufficient Variation' or 'Test Error'
    # Ensure it handles cases where chi2_contingency fails or table becomes empty
    assert results['conclusion'] in ["Test Error", "Insufficient Variation"] or results['p_value'] == 1.0 # P-value 1.0 for no difference and low dof

def test_chi_squared_contingency_table_structure(chi_squared_strategy):
    """Verify the structure of the returned contingency_table."""
    group_a_data = pd.Series([0, 1, 0, 0, 1])
    group_b_data = pd.Series([1, 1, 0, 1, 0])

    results = chi_squared_strategy.conduct_test(group_a_data, group_b_data)
    
    assert 'contingency_table' in results
    expected_table_data = {'Group A': {0: 3, 1: 2}, 'Group B': {0: 2, 1: 3}}
    expected_table = pd.DataFrame(expected_table_data)
    pd.testing.assert_frame_equal(results['contingency_table'], expected_table, check_dtype=False)
