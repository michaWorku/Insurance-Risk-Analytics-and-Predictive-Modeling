Usage
=====

This project is primarily driven through Jupyter notebooks, which orchestrate the execution of modular Python scripts located in the `src/` directory. For a comprehensive understanding and reproduction of the analysis, follow the notebooks in sequential order.

Running the Notebooks
--------------------

Navigate to the `notebooks/` directory and open the notebooks in the specified order:

1.  **Exploratory Data Analysis (EDA):**
    Open `EDA.ipynb`. This notebook performs initial data loading, quality checks, descriptive statistics, and visualization to understand the dataset's characteristics and uncover initial patterns.
    **Command:**
    .. code-block:: bash

        jupyter notebook notebooks/EDA.ipynb

2.  **A/B Hypothesis Testing:**
    Open `Hypothesis_Testing.ipynb`. This notebook conducts statistical hypothesis tests to validate assumptions about risk drivers and profit margins across different segments. It provides data-backed evidence for segmentation strategies.
    **Command:**
    .. code-block:: bash

        jupyter notebook notebooks/Hypothesis_Testing.ipynb

3.  **Predictive Modeling for Risk and Premium Optimization:**
    Open `Predictive_Modeling.ipynb`. This notebook covers the development, training, evaluation, and interpretation of machine learning models for predicting claim severity, claim probability, and optimizing premiums.
    **Command:**
    .. code-block:: bash

        jupyter notebook notebooks/Predictive_Modeling.ipynb

**Important Note:** Ensure that `EDA.ipynb` is run **before** `Hypothesis_Testing.ipynb` and `Predictive_Modeling.ipynb`, as it generates and saves the processed data required by subsequent notebooks.

Exploring the Source Code
-------------------------
For a deeper understanding of the implemented methodologies, explore the `src/` directory. It contains modularized Python code for:

* **`src/data_loader.py`**: Handles raw data loading and initial format detection.
* **`src/utils/data_preparation/`**: Contains scripts for data preprocessing, cleaning, and feature engineering.
* **`src/utils/hypothesis_testing/`**: Implements statistical test strategies (T-test, Chi-squared) and metrics calculation.
* **`src/utils/models/`**: Defines strategies for various machine learning models, model training, evaluation, and interpretability (SHAP, LIME).

