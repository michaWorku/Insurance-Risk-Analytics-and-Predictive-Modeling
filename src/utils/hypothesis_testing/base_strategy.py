from abc import ABC, abstractmethod
import pandas as pd

class HypothesisTestingStrategy(ABC):
    """
    Abstract Base Class for various hypothesis testing strategies.
    Defines the interface for conducting statistical tests.
    """
    @abstractmethod
    def conduct_test(self, group_a_data: pd.Series, group_b_data: pd.Series, **kwargs) -> dict:
        """
        Conducts a specific statistical test between two groups of data.

        Parameters:
        group_a_data (pd.Series): Data for Group A (control group).
        group_b_data (pd.Series): Data for Group B (test group).
        kwargs: Additional parameters specific to the test (e.g., 'alpha' for significance level).

        Returns:
        dict: A dictionary containing test results, typically including:
              - 'statistic': The calculated test statistic.
              - 'p_value': The p-value of the test.
              - 'conclusion': A string indicating whether to 'Reject H0' or 'Fail to Reject H0'.
              - 'message': A descriptive message about the test and its result.
        """
        pass

