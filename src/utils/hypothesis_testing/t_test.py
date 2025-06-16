import pandas as pd
from scipy import stats

from pathlib import Path
import sys
project_root = Path.cwd()
print(f"Adding project root to sys.path: {project_root}")
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.hypothesis_testing.base_strategy import HypothesisTestingStrategy

class TTestStrategy(HypothesisTestingStrategy):
    """
    Implements the Independent Samples T-test for comparing the means of a
    numerical variable between two independent groups.
    """
    def conduct_test(self, group_a_data: pd.Series, group_b_data: pd.Series, alpha: float = 0.05, **kwargs) -> dict:
        """
        Conducts an independent samples t-test between two groups.

        Parameters:
        group_a_data (pd.Series): Numerical data for Group A.
        group_b_data (pd.Series): Numerical data for Group B.
        alpha (float): Significance level (default 0.05).
        kwargs:
            equal_var (bool): If True (default), performs a standard independent 2-sample test
                              that assumes equal population variances. If False, performs Welch's t-test,
                              which does not assume equal population variance.

        Returns:
        dict: Test results including statistic, p-value, conclusion, and message.
        """
        equal_var = kwargs.get('equal_var', True) # Default to True (standard t-test)

        if group_a_data.empty or group_b_data.empty:
            return {
                'statistic': None,
                'p_value': None,
                'conclusion': 'Insufficient Data',
                'message': 'Cannot perform T-test: one or both groups are empty.'
            }
        
        # Drop NaN values as t-test cannot handle them
        group_a_data_clean = group_a_data.dropna()
        group_b_data_clean = group_b_data.dropna()

        if group_a_data_clean.empty or group_b_data_clean.empty:
            return {
                'statistic': None,
                'p_value': None,
                'conclusion': 'Insufficient Data (after NaN removal)',
                'message': 'Cannot perform T-test: one or both groups became empty after removing NaN values.'
            }
            
        try:
            t_statistic, p_value = stats.ttest_ind(group_a_data_clean, group_b_data_clean, equal_var=equal_var)
        except Exception as e:
            return {
                'statistic': None,
                'p_value': None,
                'conclusion': 'Test Error',
                'message': f'Error during t-test calculation: {e}. Check data for non-numeric values.'
            }


        conclusion = "Fail to Reject H0"
        if p_value < alpha:
            conclusion = "Reject H0"

        test_type = "Independent Samples T-test (Equal Variances Assumed)" if equal_var else "Welch's T-test (Unequal Variances Not Assumed)"
        message = (f"{test_type} (α={alpha}): Statistic={t_statistic:.4f}, P-value={p_value:.4f}. "
                   f"Conclusion: {conclusion} (p-value {'>=' if p_value >= alpha else '<'} {alpha}).\n"
                   f"  Group A (Mean={group_a_data_clean.mean():.2f}, N={len(group_a_data_clean)}).\n"
                   f"  Group B (Mean={group_b_data_clean.mean():.2f}, N={len(group_b_data_clean)}).")

        return {
            'statistic': t_statistic,
            'p_value': p_value,
            'conclusion': conclusion,
            'message': message,
            'group_a_mean': group_a_data_clean.mean(),
            'group_b_mean': group_b_data_clean.mean(),
            'group_a_n': len(group_a_data_clean),
            'group_b_n': len(group_b_data_clean)
        }

