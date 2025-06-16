import pandas as pd
from scipy.stats import chi2_contingency

from pathlib import Path
import sys
project_root = Path.cwd()
print(f"Adding project root to sys.path: {project_root}")
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.hypothesis_testing.base_strategy import HypothesisTestingStrategy

class ChiSquaredTestStrategy(HypothesisTestingStrategy):
    """
    Implements the Chi-squared test for assessing independence between two categorical variables.
    Typically used for comparing proportions (e.g., Claim Frequency) between groups.
    """
    def conduct_test(self, group_a_data: pd.Series, group_b_data: pd.Series, alpha: float = 0.05, **kwargs) -> dict:
        """
        Conducts a Chi-squared test on the contingency table formed by the two groups' data.

        Parameters:
        group_a_data (pd.Series): Categorical data for Group A.
        group_b_data (pd.Series): Categorical data for Group B.
        alpha (float): Significance level (default 0.05).

        Returns:
        dict: Test results including statistic, p-value, conclusion, and message.
        """
        if group_a_data.empty or group_b_data.empty:
            return {
                'statistic': None,
                'p_value': None,
                'conclusion': 'Insufficient Data',
                'message': 'Cannot perform Chi-squared test: one or both groups are empty.'
            }

        # Combine the data and create a contingency table
        # We assume group_a_data and group_b_data represent the same binary outcome (e.g., Claimed/Not Claimed)
        # for two different groups.
        # So, we need to create a contingency table from the counts of outcomes in each group.
        
        # To simplify, assume group_a_data and group_b_data are Series of the outcome (e.g., 0/1 for No Claim/Claim)
        # and we want to compare counts of 0s and 1s between the groups.
        
        # A more robust way to build the contingency table for A/B testing on a binary outcome:
        # data = pd.DataFrame({'outcome': pd.concat([group_a_data, group_b_data]),
        #                      'group': ['A']*len(group_a_data) + ['B']*len(group_b_data)})
        # contingency_table = pd.crosstab(data['group'], data['outcome'])
        # Or even simpler, if group_a_data and group_b_data are directly counts of (success, failure):
        # e.g., group_a_data = pd.Series([success_A, failure_A])
        #       group_b_data = pd.Series([success_B, failure_B])
        
        # For A/B testing on Claim Frequency (Claimed vs. Not Claimed), a common input would be:
        # group_a_data = df[df['Group'] == 'A']['HasClaim']
        # group_b_data = df[df['Group'] == 'B']['HasClaim']
        # Where 'HasClaim' is binary (0 or 1)
        
        # Let's assume group_a_data and group_b_data are binary series (e.g., 0/1)
        # We need counts of each category (0 and 1) for each group
        counts_A = group_a_data.value_counts().reindex([0, 1], fill_value=0)
        counts_B = group_b_data.value_counts().reindex([0, 1], fill_value=0)
        
        contingency_table = pd.DataFrame({
            'Group A': counts_A,
            'Group B': counts_B
        })
        
        if contingency_table.empty or contingency_table.sum().sum() == 0:
             return {
                'statistic': None,
                'p_value': None,
                'conclusion': 'Insufficient Data',
                'message': 'Contingency table is empty or all counts are zero.'
            }
        
        # Check for columns with zero sum after reindexing, which can cause issues
        # Remove rows/columns from contingency table if their sum is 0 to avoid errors in chi2_contingency
        contingency_table = contingency_table.loc[(contingency_table != 0).any(axis=1), (contingency_table != 0).any(axis=0)]
        
        if contingency_table.empty:
            return {
                'statistic': None,
                'p_value': None,
                'conclusion': 'Insufficient Variation',
                'message': 'Contingency table became empty after removing zero-sum rows/columns. No variation to test.'
            }
        
        try:
            chi2, p, dof, ex = chi2_contingency(contingency_table)
        except ValueError as e:
             return {
                'statistic': None,
                'p_value': None,
                'conclusion': 'Test Error',
                'message': f'Error during chi2_contingency calculation: {e}. Check contingency table for zero rows/columns or insufficient data.'
            }

        conclusion = "Fail to Reject H0"
        if p < alpha:
            conclusion = "Reject H0"

        message = (f"Chi-squared test (α={alpha}): Statistic={chi2:.4f}, P-value={p:.4f}. "
                   f"Conclusion: {conclusion} (p-value {'>=' if p >= alpha else '<'} {alpha}).")

        return {
            'statistic': chi2,
            'p_value': p,
            'conclusion': conclusion,
            'message': message,
            'contingency_table': contingency_table # Include for transparency
        }

