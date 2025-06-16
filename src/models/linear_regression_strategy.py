import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from pathlib import Path
import sys
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.base_model_strategy import BaseModelStrategy


class LinearRegressionStrategy(BaseModelStrategy):
    """
    Concrete strategy for Linear Regression model.
    """
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()
        self._name = "Linear Regression"

    @property
    def name(self) -> str:
        return self._name

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the Linear Regression model.

        Parameters:
        X (pd.DataFrame): Training features.
        y (pd.Series): Target variable for training.
        """
        if X.empty or y.empty:
            print("Warning: Training data (X or y) is empty for Linear Regression. Skipping training.")
            return

        print(f"Training {self.name} model...")
        self.model.fit(X, y)
        print(f"{self.name} training complete.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the trained Linear Regression model.

        Parameters:
        X (pd.DataFrame): Features for prediction.

        Returns:
        np.ndarray: Array of predictions.
        """
        if self.model is None:
            raise RuntimeError(f"{self.name} model not trained. Call train() first.")
        if X.empty:
            print("Warning: Prediction data (X) is empty for Linear Regression. Returning empty array.")
            return np.array([])
            
        return self.model.predict(X)

    def get_model(self):
        """
        Returns the trained Linear Regression model object.
        """
        return self.model

