import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from pathlib import Path
import sys
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.base_model_strategy import BaseModelStrategy


class RandomForestStrategy(BaseModelStrategy):
    """
    Concrete strategy for Random Forest Regressor model.
    """
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        super().__init__()
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self._name = "Random Forest Regressor"

    @property
    def name(self) -> str:
        return self._name

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the Random Forest Regressor model.

        Parameters:
        X (pd.DataFrame): Training features.
        y (pd.Series): Target variable for training.
        """
        if X.empty or y.empty:
            print("Warning: Training data (X or y) is empty for Random Forest. Skipping training.")
            return

        print(f"Training {self.name} model...")
        self.model.fit(X, y)
        print(f"{self.name} training complete.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the trained Random Forest Regressor model.

        Parameters:
        X (pd.DataFrame): Features for prediction.

        Returns:
        np.ndarray: Array of predictions.
        """
        if self.model is None:
            raise RuntimeError(f"{self.name} model not trained. Call train() first.")
        if X.empty:
            print("Warning: Prediction data (X) is empty for Random Forest. Returning empty array.")
            return np.array([])

        return self.model.predict(X)

    def get_model(self):
        """
        Returns the trained Random Forest Regressor model object.
        """
        return self.model

