import pandas as pd
import numpy as np
from typing import Dict, Any

from pathlib import Path
import sys
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.base_model_strategy import BaseModelStrategy


class ModelTrainer:
    """
    Context class for training and predicting with various machine learning models
    using a strategy pattern.
    """
    def __init__(self, strategy: BaseModelStrategy):
        """
        Initializes the ModelTrainer with a specific modeling strategy.

        Parameters:
        strategy (BaseModelStrategy): An instance of a concrete BaseModelStrategy.
        """
        if not isinstance(strategy, BaseModelStrategy):
            raise TypeError("Provided strategy must be an instance of BaseModelStrategy.")
        self._strategy = strategy
        self.trained_model = None

    def set_strategy(self, strategy: BaseModelStrategy):
        """
        Sets a new modeling strategy for the ModelTrainer.

        Parameters:
        strategy (BaseModelStrategy): The new strategy to be used for modeling.
        """
        if not isinstance(strategy, BaseModelStrategy):
            raise TypeError("Provided strategy must be an instance of BaseModelStrategy.")
        self._strategy = strategy
        self.trained_model = None # Reset trained model when strategy changes

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains the model using the current strategy.

        Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        """
        print(f"--- Training Model: {self._strategy.name} ---")
        if X_train.empty or y_train.empty:
            print("Warning: Training data is empty. Skipping training.")
            return

        self._strategy.train(X_train, y_train)
        self.trained_model = self._strategy.get_model()
        print(f"Model '{self._strategy.name}' trained.")

    def predict_model(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the trained model of the current strategy.

        Parameters:
        X_test (pd.DataFrame): Features for prediction.

        Returns:
        np.ndarray: Array of predictions.
        """
        if self.trained_model is None:
            raise RuntimeError(f"Model '{self._strategy.name}' not trained. Call train_model() first.")
        if X_test.empty:
            print("Warning: Test data is empty. Returning empty predictions.")
            return np.array([])

        print(f"--- Generating Predictions with: {self._strategy.name} ---")
        predictions = self._strategy.predict(X_test)
        print(f"Predictions generated for {len(predictions)} samples.")
        return predictions

    def get_current_model_object(self) -> Any:
        """
        Returns the raw trained model object from the current strategy.
        """
        return self.trained_model

    def get_strategy_name(self) -> str:
        """
        Returns the name of the current modeling strategy.
        """
        return self._strategy.name

# Example usage (for standalone testing)
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from src.models.linear_regression_strategy import LinearRegressionStrategy
    from src.models.random_forest_strategy import RandomForestStrategy
    from src.models.xgboost_strategy import XGBoostStrategy
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    print("--- Testing ModelTrainer with different strategies ---")

    # 1. Create dummy data
    np.random.seed(42)
    X_data = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
    y_data = X_data['feature_0'] * 2 + X_data['feature_1'] * 3 - X_data['feature_2'] * 0.5 + np.random.normal(0, 0.1, 100)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

    # 2. Test Linear Regression
    lr_strategy = LinearRegressionStrategy()
    trainer = ModelTrainer(lr_strategy)
    trainer.train_model(X_train, y_train)
    lr_predictions = trainer.predict_model(X_test)
    print(f"Linear Regression RMSE: {np.sqrt(mean_squared_error(y_test, lr_predictions)):.4f}")

    # 3. Test Random Forest
    rf_strategy = RandomForestStrategy()
    trainer.set_strategy(rf_strategy)
    trainer.train_model(X_train, y_train)
    rf_predictions = trainer.predict_model(X_test)
    print(f"Random Forest RMSE: {np.sqrt(mean_squared_error(y_test, rf_predictions)):.4f}")

    # 4. Test XGBoost Regressor
    xgb_reg_strategy = XGBoostStrategy(objective='reg:squarederror')
    trainer.set_strategy(xgb_reg_strategy)
    trainer.train_model(X_train, y_train)
    xgb_reg_predictions = trainer.predict_model(X_test)
    print(f"XGBoost Regressor RMSE: {np.sqrt(mean_squared_error(y_test, xgb_reg_predictions)):.4f}")

    # 5. Test XGBoost Classifier (with dummy binary target)
    y_binary = (y_data > y_data.median()).astype(int)
    _, _, y_train_b, y_test_b = train_test_split(X_data, y_binary, test_size=0.3, random_state=42)

    xgb_clf_strategy = XGBoostStrategy(objective='binary:logistic')
    trainer.set_strategy(xgb_clf_strategy)
    trainer.train_model(X_train, y_train_b)
    xgb_clf_predictions_proba = trainer.predict_model(X_test) # This returns probabilities
    # For evaluation, you'd usually convert to binary predictions at a threshold
    print(f"XGBoost Classifier predicted probabilities (first 5): {xgb_clf_predictions_proba[:5]}")

    print("\nModelTrainer testing complete.")
