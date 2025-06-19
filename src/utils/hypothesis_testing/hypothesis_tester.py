import pandas as pd
import numpy as np

from pathlib import Path
import sys
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.hypothesis_testing.base_strategy import HypothesisTestingStrategy

class HypothesisTester:
    """
    Context class for executing various hypothesis tests using a strategy pattern.
    Allows switching between different statistical test implementations.
    """
    def __init__(self, strategy: HypothesisTestingStrategy):
        """
        Initializes the HypothesisTester with a specific testing strategy.

        Parameters:
        strategy (HypothesisTestingStrategy): An instance of a concrete HypothesisTestingStrategy.
        """
        if not isinstance(strategy, HypothesisTestingStrategy):
            raise TypeError("Provided strategy must be an instance of HypothesisTestingStrategy.")
        self._strategy = strategy

    def set_strategy(self, strategy: HypothesisTestingStrategy):
        """
        Sets a new strategy for the HypothesisTester.

        Parameters:
        strategy (HypothesisTestingStrategy): The new strategy to be used for testing.
        """
        if not isinstance(strategy, HypothesisTestingStrategy):
            raise TypeError("Provided strategy must be an instance of HypothesisTestingStrategy.")
        self._strategy = strategy

    def execute_test(self, group_a_data: pd.Series, group_b_data: pd.Series, alpha: float = 0.05, test_name: str = "Hypothesis Test", **kwargs) -> dict:
        """
        Executes the hypothesis test using the current strategy.

        Parameters:
        group_a_data (pd.Series): Data for Group A.
        group_b_data (pd.Series): Data for Group B.
        alpha (float): Significance level for the test (default 0.05).
        test_name (str): A descriptive name for the test being performed.
        kwargs: Additional parameters to pass to the strategy's conduct_test method.

        Returns:
        dict: The results of the statistical test.
        """
        print(f"\n--- Conducting {test_name} ---")
        if group_a_data.empty and group_b_data.empty:
            print("Warning: Both Group A and Group B data are empty. Skipping test.")
            return {
                'statistic': None, 'p_value': None, 'conclusion': 'No Data',
                'message': f'Skipped {test_name}: Both groups are empty.'
            }
            
        # Call the strategy's conduct_test method
        results = self._strategy.conduct_test(group_a_data, group_b_data, alpha=alpha, **kwargs)
        print(results['message']) # Print the summary message from the strategy
        return results


# if __name__ == "__main__":
#     # Example usage for HypothesisTester
#     print("--- Testing HypothesisTester with different strategies ---")

#     # Create dummy data for demonstration
#     np.random.seed(42)
#     data = {
#         'Group': ['A'] * 50 + ['B'] * 50,
#         'Numerical_Metric': np.concatenate([np.random.normal(loc=10, scale=2, size=50),
#                                             np.random.normal(loc=11, scale=2.5, size=50)]),
#         'Binary_Outcome': np.concatenate([np.random.choice([0, 1], size=50, p=[0.7, 0.3]),
#                                           np.random.choice([0, 1], size=50, p=[0.5, 0.5])])
#     }
#     df_dummy = pd.DataFrame(data)

#     group_a_num = df_dummy[df_dummy['Group'] == 'A']['Numerical_Metric']
#     group_b_num = df_dummy[df_dummy['Group'] == 'B']['Numerical_Metric']

#     group_a_binary = df_dummy[df_dummy['Group'] == 'A']['Binary_Outcome']
#     group_b_binary = df_dummy[df_dummy['Group'] == 'B']['Binary_Outcome']

#     # Test with T-test strategy
#     from t_test import TTestStrategy
#     t_test_strategy = TTestStrategy()
#     tester_t = HypothesisTester(t_test_strategy)
    
#     tester_t.execute_test(group_a_num, group_b_num, test_name="Numerical Metric (A vs B) - T-test")

#     # Test with Chi-squared strategy
#     from chi_squared_test import ChiSquaredTestStrategy
#     chi_test_strategy = ChiSquaredTestStrategy()
#     tester_chi = HypothesisTester(chi_test_strategy)
    
#     tester_chi.execute_test(group_a_binary, group_b_binary, test_name="Binary Outcome (A vs B) - Chi-squared Test")

#     # Test case: Insufficient data
#     print("\n--- Testing with Insufficient Data ---")
#     tester_t.execute_test(pd.Series([]), group_b_num, test_name="T-test with empty Group A")
#     tester_chi.execute_test(group_a_binary, pd.Series([]), test_name="Chi-squared with empty Group B")

