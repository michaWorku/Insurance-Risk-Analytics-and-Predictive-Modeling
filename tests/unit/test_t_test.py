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

from src.utils.hypothesis_testing.t_test import TTestStrategy

@pytest.fixture
def t_test_strategy():
    return TTestStrategy()

def test_t_test_significant_difference(t_test_strategy):
    """Test case where there is a statistically significant difference."""
    np.random.seed(42)
    group_a = pd.Series(np.random.normal(loc=10, scale=2, size=50))
    group_b = pd.Series(np.random.normal(loc=12, scale=2, size=50))
    
    results = t_test_strategy.conduct_test(group_a, group_b, alpha=0.05)
    
    assert results['conclusion'] == "Reject H0"
    assert results['p_value'] < 0.05
    assert isinstance(results['statistic'], float)
    assert isinstance(results['p_value'], float)
    assert 'Group A (Mean=' in results['message']
    assert 'Group B (Mean=' in results['message']

def test_t_test_no_significant_difference(t_test_strategy):
    """Test case where there is no statistically significant difference."""
    np.random.seed(42)
    group_a = pd.Series(np.random.normal(loc=10, scale=2, size=50))
    group_b = pd.Series(np.random.normal(loc=10.2, scale=2, size=50))
    
    results = t_test_strategy.conduct_test(group_a, group_b, alpha=0.05)
    
    assert results['conclusion'] == "Fail to Reject H0"
    assert results['p_value'] >= 0.05
    assert isinstance(results['statistic'], float)
    assert isinstance(results['p_value'], float)

def test_t_test_empty_group_a(t_test_strategy):
    """Test with an empty group A."""
    group_a = pd.Series([])
    group_b = pd.Series(np.random.normal(loc=10, scale=2, size=50))
    
    results = t_test_strategy.conduct_test(group_a, group_b)
    
    assert results['conclusion'] == "Insufficient Data"
    assert "one or both groups are empty" in results['message']

def test_t_test_empty_group_b(t_test_strategy):
    """Test with an empty group B."""
    group_a = pd.Series(np.random.normal(loc=10, scale=2, size=50))
    group_b = pd.Series([])
    
    results = t_test_strategy.conduct_test(group_a, group_b)
    
    assert results['conclusion'] == "Insufficient Data"
    assert "one or both groups are empty" in results['message']

def test_t_test_with_nans(t_test_strategy):
    """Test t-test with NaN values in the input series."""
    np.random.seed(42)
    group_a = pd.Series([10, 11, np.nan, 12, 10])
    group_b = pd.Series([15, 14, 16, np.nan, 15])
    
    results = t_test_strategy.conduct_test(group_a, group_b, alpha=0.05)
    
    assert results['conclusion'] == "Reject H0" # Should still reject if sufficient non-NaN data
    assert results['p_value'] < 0.05
    assert results['group_a_n'] == 4 # NaNs should be dropped
    assert results['group_b_n'] == 4

def test_t_test_all_nans_in_one_group(t_test_strategy):
    """Test t-test when one group becomes empty after NaN removal."""
    group_a = pd.Series([np.nan, np.nan, np.nan])
    group_b = pd.Series([1, 2, 3])
    
    results = t_test_strategy.conduct_test(group_a, group_b)
    assert results['conclusion'] == "Insufficient Data (after NaN removal)"
    assert "one or both groups became empty after removing NaN values" in results['message']

def test_t_test_equal_var_false(t_test_strategy):
    """Test Welch's t-test (equal_var=False)."""
    np.random.seed(42)
    group_a = pd.Series(np.random.normal(loc=10, scale=1, size=50))
    group_b = pd.Series(np.random.normal(loc=12, scale=5, size=50)) # Different variance
    
    results = t_test_strategy.conduct_test(group_a, group_b, alpha=0.05, equal_var=False)
    
    assert results['conclusion'] == "Reject H0"
    assert results['p_value'] < 0.05
    assert "Welch's T-test" in results['message']
