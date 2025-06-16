import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import warnings

class ModelInterpreter:
    """
    A class for interpreting machine learning models using SHAP (SHapley Additive exPlanations)
    for global feature importance and LIME (Local Interpretable Model-agnostic Explanations)
    for individual instance explanations.
    """
    def __init__(self, model, feature_names: list, class_names: list = None, model_type: str = 'regression'):
        """
        Initializes the ModelInterpreter with a trained model.

        Args:
            model: A trained machine learning model with a predict or predict_proba method.
            feature_names (list): List of feature names corresponding to the input X.
            class_names (list, optional): List of class names for classification models. Required for LIME classifier.
            model_type (str): 'regression' or 'classification'. Determines which explainer/prediction method to use.
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.model_type = model_type
        
        self.shap_explainer = None
        self.shap_values = None
        self.lime_explainer = None


    def explain_model_shap(self, X: pd.DataFrame):
        """
        Generates SHAP values for the given features to provide global feature importance.

        Args:
            X (pd.DataFrame): The feature DataFrame used for explanation (e.g., test set sample).
        """
        print("Generating SHAP explanations...")
        
        # Ensure X has column names for SHAP plots
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)

        try:
            # Try TreeExplainer first for tree-based models (faster)
            self.shap_explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            print(f"Warning: TreeExplainer not suitable for this model ({type(self.model)}). Falling back to KernelExplainer. Error: {e}")
            print("Note: KernelExplainer can be very slow for large datasets. Consider sampling X.")
            
            # For KernelExplainer, a background dataset is needed. Use a small sample of X.
            # It's better to use a representative sample or k-means summary of training data.
            # For simplicity in this utility, we'll sample from X.
            background_data = shap.sample(X, 100) if len(X) > 100 else X
            self.shap_explainer = shap.KernelExplainer(self.model.predict, background_data)
        
        if self.shap_explainer:
            # SHAP values can be a list for multi-output or multi-class models.
            # For binary classification, it's often a list of two arrays.
            self.shap_values = self.shap_explainer.shap_values(X)
            print("SHAP explanations generated.")
        else:
            print("Failed to initialize SHAP explainer.")


    def plot_shap_summary(self, X: pd.DataFrame, num_features: int = 10):
        """
        Generates a SHAP summary plot (e.g., bar plot or dot plot) for global feature importance.

        Args:
            X (pd.DataFrame): The feature DataFrame used for explanation.
            num_features (int): Number of top features to display.
        """
        if self.shap_values is None or self.shap_explainer is None:
            print("SHAP values not generated. Call explain_model_shap() first.")
            return

        print(f"Generating SHAP summary plot for top {num_features} features...")
        plt.figure(figsize=(12, 7))
        
        # Handle SHAP values for classification (list of arrays) vs regression (single array)
        if isinstance(self.shap_values, list):
            # For binary classification, typically plot for the positive class (index 1)
            # For multi-class, decide which class to focus on or sum magnitudes
            if self.model_type == 'classification' and len(self.shap_values) > 1:
                shap_values_to_plot = self.shap_values[1] # Assume positive class is index 1
            else:
                shap_values_to_plot = self.shap_values[0] # Fallback for single list item
            shap.summary_plot(shap_values_to_plot, X, plot_type="bar", show=False, max_display=num_features, feature_names=self.feature_names)
        else:
            shap.summary_plot(self.shap_values, X, plot_type="bar", show=False, max_display=num_features, feature_names=self.feature_names)
        
        plt.title(f"SHAP Feature Importance (Top {num_features}) for {self.model_type.capitalize()}")
        plt.tight_layout()
        plt.show()

    def plot_shap_force(self, X_instance: pd.Series):
        """
        Generates a SHAP force plot for a single instance, showing how features
        contribute to that specific prediction.

        Args:
            X_instance (pd.Series): A single row of features (a pandas Series) for explanation.
        """
        if self.shap_explainer is None:
            print("SHAP explainer not initialized. Call explain_model_shap() first.")
            return
        if X_instance.empty:
            print("Cannot generate SHAP force plot for an empty instance.")
            return

        print("Generating SHAP force plot for an individual instance...")
        
        # Ensure X_instance is a DataFrame row for explainer's consistency
        if not isinstance(X_instance, pd.DataFrame):
            X_instance_df = X_instance.to_frame().T
        else:
            X_instance_df = X_instance
            
        # Get SHAP values for this specific instance
        instance_shap_values = self.shap_explainer.shap_values(X_instance_df)
        
        # Handle SHAP values for classification (list of arrays)
        if isinstance(instance_shap_values, list):
            # For binary classification, focus on the positive class
            if self.model_type == 'classification' and len(instance_shap_values) > 1:
                instance_shap_values = instance_shap_values[1]
            else:
                instance_shap_values = instance_shap_values[0]
        
        # Ensure SHAP values are 1D array if from a single instance
        if instance_shap_values.ndim > 1 and instance_shap_values.shape[0] == 1:
            instance_shap_values = instance_shap_values.flatten()

        shap.initjs() # Initialize JavaScript for interactive plots (notebooks)
        
        # Use X_instance_df.iloc[0] to pass a single row to force_plot's feature values
        print(shap.force_plot(self.shap_explainer.expected_value, instance_shap_values, X_instance_df.iloc[0], feature_names=self.feature_names))
        print("SHAP force plot generated.")

    def explain_instance_lime(self, X_instance: pd.Series, num_features: int = 10):
        """
        Generates LIME explanation for a single instance.

        Args:
            X_instance (pd.Series): A single row of features (a pandas Series) for explanation.
            num_features (int): Number of features to include in the LIME explanation.
        """
        if X_instance.empty:
            print("Cannot generate LIME explanation for an empty instance.")
            return

        print(f"Generating LIME explanation for an individual instance (top {num_features} features)...")

        # LIME needs a prediction function that returns probabilities for classification
        # or raw predictions for regression. It also needs training data (or a representative sample)
        # for generating perturbed samples.
        
        if self.model_type == 'classification':
            if not hasattr(self.model, 'predict_proba'):
                warnings.warn("Model does not have 'predict_proba'. LIME for classification may not work as expected.")
                predict_fn = self.model.predict
            else:
                predict_fn = self.model.predict_proba
            mode = 'classification'
            # Ensure class_names are provided for classification
            if self.class_names is None:
                raise ValueError("For LIME classification, 'class_names' must be provided in ModelInterpreter initialization.")
        else: # regression
            predict_fn = self.model.predict
            mode = 'regression'
            
        # LIME needs the training data (as numpy array, for data statistics) to create the explainer
        # We assume X_train (or a representative sample) is used to initialize the explainer
        # For simplicity in this utility, we'll initialize the explainer here
        # It's best practice to initialize once with X_train (numerical features as numpy)
        
        # Create a dummy data for LIME explainer background if not already done.
        # This will need to be passed during init if we want to avoid re-creating it.
        # For this demo, let's create a dummy one.
        # In actual notebook, pass X_train (or a subset) during ModelInterpreter init.
        
        # For the example, we need to manually pass a dummy training data representation for LIME explainer setup
        # The best way is to initialize the explainer once with X_train from the notebook
        # Let's adjust the ModelInterpreter init to take X_train directly for LIME explainer setup.
        
        # If we didn't pass training data to init, this is a fallback (less ideal)
        # Assuming X_instance is a sample from the dataset used for training
        if self.lime_explainer is None:
            # For demonstration, create a temporary tabular explainer
            # In real usage, pass X_train (numpy values) to ModelInterpreter __init__ for LIME
            # Using X_instance for feature statistics is not ideal but works for a single demo
            # A background sample from X_train is much better for a robust LIME explainer.
            
            # To correctly initialize LIME explainer in a standalone utility, we need a sample
            # of the training data. For this example, let's just make it work.
            # In the notebook, we'll initialize ModelInterpreter with `X_train.values` for LIME.
            
            # This is a critical point: LIME's explainer needs `training_data` to understand feature distributions.
            # This utility itself doesn't have `X_train` without it being passed to `__init__`.
            # For the example below, I'll pass X.values to the __init__ to make LIME work correctly.
            
            # For this method, let's assume `self.lime_explainer` is initialized externally or
            # within this method by taking a `training_data` argument.
            
            # To make it runnable for standalone test, let's add a dummy `training_data`
            # This part will be different in the actual notebook where X_train is available.
            
            # The proper way is `self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(...)`
            # in __init__ with a sample of X_train.
            
            # For now, let's raise an error and ensure the notebook initializes it correctly.
            raise RuntimeError("LIME explainer not initialized. Please pass training_data to ModelInterpreter init or call initialize_lime_explainer() first.")


        # Transform the single instance to a numpy array if it's a Series for LIME
        instance_np = X_instance.values.reshape(1, -1)

        # Get the explanation
        explanation = self.lime_explainer.explain_instance(
            instance_np[0], # LIME expects a single 1D instance array
            predict_fn,
            num_features=num_features,
            num_samples=5000 # Number of perturbed samples LIME generates
        )

        # Display the explanation in the notebook
        print("\nLIME Explanation Plot:")
        explanation.as_pyplot_figure(figsize=(10, 6))
        plt.tight_layout()
        plt.show()

        print("\nLIME Explanation Text (feature weights):")
        # Print the list of (feature, weight) tuples
        print(explanation.as_list())

        print("LIME explanation generated.")


# Example usage (for standalone testing)
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, accuracy_score
    import pandas as pd
    import numpy as np

    print("--- Testing ModelInterpreter ---")

    # 1. Create a dummy dataset (for ModelInterpreter's internal testing)
    np.random.seed(0)
    X = pd.DataFrame(np.random.rand(200, 5), columns=['feature_A', 'feature_B', 'feature_C', 'feature_D', 'feature_E'])
    y_reg = X['feature_A'] * 2 + X['feature_B'] * 5 + np.random.normal(0, 1, 200)
    y_clf = (y_reg > y_reg.median()).astype(int) # Binary target

    X_train_df, X_test_df, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    # 2. Train a dummy regression model
    reg_model = RandomForestRegressor(random_state=42, n_estimators=10)
    reg_model.fit(X_train_df, y_reg_train)

    # 3. Initialize and use ModelInterpreter for REGRESSION
    print("\n--- REGRESSION MODEL INTERPRETATION ---")
    reg_interpreter = ModelInterpreter(model=reg_model, feature_names=X.columns.tolist(), model_type='regression')
    reg_interpreter.explain_model_shap(X_test_df)
    reg_interpreter.plot_shap_summary(X_test_df)
    
    if not X_test_df.empty:
        reg_interpreter.plot_shap_force(X_test_df.iloc[0]) # Force plot for first instance

    # LIME setup for regression model interpreter (requires training data)
    reg_interpreter.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_df.values,
        feature_names=X_train_df.columns.tolist(),
        mode='regression'
    )
    if not X_test_df.empty:
        try:
            reg_interpreter.explain_instance_lime(X_test_df.iloc[0])
        except RuntimeError as e:
            print(e) # This will catch the error if lime_explainer not initialized


    # 4. Train a dummy classification model
    clf_model = RandomForestClassifier(random_state=42, n_estimators=10)
    clf_model.fit(X_train_df, y_clf_train)

    # 5. Initialize and use ModelInterpreter for CLASSIFICATION
    print("\n--- CLASSIFICATION MODEL INTERPRETATION ---")
    clf_interpreter = ModelInterpreter(model=clf_model, feature_names=X.columns.tolist(),
                                       class_names=['No Claim', 'Claim'], model_type='classification')
    clf_interpreter.explain_model_shap(X_test_df)
    clf_interpreter.plot_shap_summary(X_test_df)
    
    if not X_test_df.empty:
        clf_interpreter.plot_shap_force(X_test_df.iloc[0])

    # LIME setup for classification model interpreter (requires training data)
    clf_interpreter.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_df.values,
        feature_names=X_train_df.columns.tolist(),
        class_names=['No Claim', 'Claim'],
        mode='classification'
    )
    if not X_test_df.empty:
        try:
            clf_interpreter.explain_instance_lime(X_test_df.iloc[0])
        except RuntimeError as e:
            print(e)


    print("\nModel interpreter examples complete.")
