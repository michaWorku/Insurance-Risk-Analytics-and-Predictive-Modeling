Project Structure
=================

The project follows a modular and well-organized structure to promote maintainability, reusability, and collaboration.

.. code-block:: text

    .
    ├── .vscode/                 # VSCode specific settings
    ├── .github/                 # GitHub specific configurations (e.g., Workflows)
    │   └── workflows/
    │       └── unittests.yml    # CI/CD workflow for tests and linting
    ├── .gitignore               # Specifies intentionally untracked files to ignore
    ├── requirements.txt         # Python dependencies
    ├── pyproject.toml           # Modern Python packaging configuration (PEP 517/621)
    ├── README.md                # Project overview, installation, usage
    ├── Makefile                 # Common development tasks (setup, test, lint, clean)
    ├── .env                     # Environment variables (e.g., API keys - kept out of Git)
    ├── src/                     # Core source code for the project
    │   ├── __init__.py
    │   ├── data_loader.py        # Handles raw data loading and initial format detection
    │   ├── utils/                # Utility functions, helper classes
    │   │   ├── __init__.py
    │   │   ├── data_preparation/ # Scripts for data preprocessing, cleaning, and feature engineering
    │   │   │   ├── __init__.py
    │   │   │   ├── feature_engineer.py
    │   │   │   └── preprocessor.py
    │   │   ├── hypothesis_testing/ # Implements statistical test strategies and metrics calculation
    │   │   │   ├── __init__.py
    │   │   │   ├── chi_squared_test.py
    │   │   │   ├── hypothesis_tester.py
    │   │   │   ├── metrics_calculator.py
    │   │   │   └── t_test.py
    │   │   └── models/           # Defines strategies for various machine learning models, training, evaluation, and interpretability
    │   │       ├── __init__.py
    │   │       ├── decision_tree_strategy.py
    │   │       ├── linear_regression_strategy.py
    │   │       ├── model_evaluator.py
    │   │       ├── model_interpreter.py
    │   │       ├── model_trainer.py
    │   │       ├── random_forest_strategy.py
    │   │       └── xgboost_strategy.py
    ├── tests/                   # Test suite (unit, integration)
    │   ├── unit/                # Unit tests for individual components
    │   │   ├── test_chi_squared_test.py
    │   │   ├── test_data_loader.py
    │   │   ├── test_data_preprocessing.py
    │   │   ├── test_data_summarization.py
    │   │   ├── test_metrics_calculator.py
    │   │   ├── test_model_evaluator.py
    │   │   └── test_t_test.py
    │   └── integration/         # Integration tests for combined components
    ├── notebooks/               # Jupyter notebooks for experimentation, EDA, prototyping
    │   ├── EDA.ipynb
    │   ├── Hypothesis_Testing.ipynb
    │   └── Predictive_Modeling.ipynb
    ├── scripts/                 # Standalone utility scripts (e.g., data processing, deployment)
    ├── docs/                    # Project documentation (Sphinx generated)
    │   ├── build/               # Generated HTML/PDF output
    │   └── source/              # reStructuredText source files
    ├── data/                    # Data storage (raw, processed)
    │   ├── raw/                 # Original, immutable raw data
    │   └── processed/           # Cleaned, transformed, or feature-engineered data
    ├── config/                  # Configuration files
    └── examples/                # Example usage of the project components

