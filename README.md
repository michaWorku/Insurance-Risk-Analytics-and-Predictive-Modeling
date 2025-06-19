# **Insurance-Risk-Analytics-And-Predictive-Modeling**

## **Project Description**

This project focuses on an end-to-end insurance risk analytics and predictive modeling solution for **AlphaCare Insurance Solutions (ACIS)** in South Africa. As a marketing analytics engineer, the primary objective is to transform traditional pricing models into a dynamic, risk-based system by leveraging data-driven insights.

The project encompasses three key phases:

1. **Exploratory Data Analysis (EDA):** A thorough investigation into ACIS's historical car insurance claim data to understand data quality, distributions, temporal trends, and segment-wise profitability. This phase lays the foundation for all subsequent analytical steps.
2. **A/B Hypothesis Testing:** Statistical validation of key business assumptions about risk drivers and profit margins across different segments (e.g., provinces, zip codes, demographics). This phase provides data-backed evidence for refining segmentation strategies.
3. **Predictive Modeling for Risk and Premium Optimization:** Development and evaluation of machine learning models to:
    - **Predict Claim Severity:** Forecast the financial cost of a claim if it occurs.
    - **Predict Claim Probability:** Determine the likelihood of a policy resulting in a claim.
    - **Direct Premium Prediction:** Optimize the calculation of appropriate premiums.

By integrating advanced analytics with business objectives, this project aims to identify low-risk customer segments, optimize marketing strategies, reduce premiums where appropriate, attract new clients, and ultimately enhance ACIS's profitability and competitive advantage.

## **Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## **Installation**

### **Prerequisites**

- Python 3.8+ (recommended)
- Git

### **Steps**

1. **Clone the repository:**
    
    ```
    git clone https://github.com/michaWorku/Insurance-Risk-Analytics-and-Predictive-Modeling.git 
    cd Insurance-Risk-Analytics-and-Predictive-Modeling
    
    ```
    
    If you created the project in the current directory:
    
    ```
    # Already in the project root
    
    ```
    
2. **Create and activate a virtual environment:**
    
    ```
    python3 -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    
    ```
    
3. **Install dependencies:**
    
    ```
    pip install -r requirements.txt
    
    ```
    

## **Usage**

This project is primarily driven through Jupyter notebooks located in the `notebooks/` directory, which execute the modular Python scripts in `src/`. Follow the notebooks in sequential order for a comprehensive understanding and reproduction of the analysis:

1. **Run EDA Notebook:**
    - Navigate to the `notebooks/` directory.
    - Open `EDA.ipynb` to perform exploratory data analysis, data quality checks, and initial summarization.
    - Execute all cells in the notebook. This will also preprocess and save the data to `data/processed/`.
    
    ```
    # From project root:
    # jupyter notebook notebooks/EDA.ipynb
    
    ```
    
2. **Run Hypothesis Testing Notebook:**
    - Ensure `EDA.ipynb` has been run first to generate the processed data.
    - Open `Hypothesis_Testing.ipynb` to conduct A/B hypothesis tests on key risk drivers and profit margins.
    - Execute all cells.
    
    ```
    # From project root:
    # jupyter notebook notebooks/Hypothesis_Testing.ipynb
    
    ```
    
3. **Run Predictive Modeling Notebook:**
    - Ensure `EDA.ipynb` has been run first.
    - Open `Predictive_Modeling.ipynb` to build, train, evaluate, and interpret machine learning models for claim severity, claim probability, and premium prediction.
    - Execute all cells.
    
    ```
    # From project root:
    # jupyter notebook notebooks/Predictive_Modeling.ipynb
    
    ```
    

For deeper insights, explore the `src/` directory which contains the modularized Python code for data loading, preprocessing, analysis strategies, and model implementations.

## **Project Structure**

```
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
│   ├── core/                # Core logic/components
│   ├── models/              # Data models, ORM definitions, ML models
│   ├── utils/              # Utility functions, helper classes
│   └── services/            # Business logic, service layer
├── tests/                   # Test suite (unit, integration)
│   ├── unit/                # Unit tests for individual components
│   └── integration/         # Integration tests for combined components
├── notebooks/               # Jupyter notebooks for experimentation, EDA, prototyping
├── scripts/                 # Standalone utility scripts (e.g., data processing, deployment)
├── docs/                    # Project documentation (e.g., Sphinx docs)
├── data/                    # Data storage (raw, processed)
│   ├── raw/                 # Original, immutable raw data
│   └── processed/           # Cleaned, transformed, or feature-engineered data
├── config/                  # Configuration files
└── examples/                # Example usage of the project components

```

## **Contributing**

Guidelines for contributing to the project.

## **License**

This project is licensed under the [MIT License](https://gemini.google.com/app/LICENSE). (Create a LICENSE file if you want to use MIT)
