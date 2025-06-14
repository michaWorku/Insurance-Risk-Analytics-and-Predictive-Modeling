# Insurance Risk Analytics and Predictive Modeling

## Project Description

This project focuses on analyzing and modeling historical car insurance data to support **risk segmentation**, **claim prediction**, and **premium optimization**. It leverages data engineering, statistical analysis, and machine learning techniques to help identify low-risk customer segments and improve pricing strategies for insurance providers.

Key features include:

* Exploratory data analysis and risk profiling by geography, vehicle type, and customer demographics
* A/B hypothesis testing for statistically validating risk drivers
* Machine learning models to predict claim severity and premium amounts
* Model interpretability using SHAP and LIME to support transparent decision-making
* Reproducible pipeline using Git, GitHub Actions (CI/CD), and Data Version Control (DVC)


## Table of Contents

* [Installation](#installation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Contributing](#contributing)
* [License](#license)


## Installation

### Prerequisites

* Python 3.8+
* Git

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/michaWorku/Insurance-Risk-Analytics-and-Predictive-Modeling.git
   cd Insurance-Risk-Analytics-and-Predictive-Modeling
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```


## ▶️ Usage

To run the main analytics or modeling workflows:

```bash
# Example: Running a data preparation or model training script
python src/main.py
```

To launch exploratory notebooks or visualizations:

```bash
jupyter notebook notebooks/
```


## 🗂️ Project Structure

```bash
├── .vscode/                 # VSCode specific settings
├── .github/                 # GitHub Workflows (CI/CD)
│   └── workflows/
│       └── unittests.yml    # CI pipeline: linting, testing
├── .gitignore               # Files/folders to ignore in Git
├── requirements.txt         # Python dependency list
├── pyproject.toml           # Python packaging and formatting
├── README.md                # This file
├── Makefile                 # Dev task automation (optional)
├── .env                     # Local environment config (ignored by Git)
├── src/                     # Main source code
│   ├── core/                # Core logic modules
│   ├── models/              # ML/statistical models
│   ├── utils/               # Helper functions
│   └── services/            # Feature and pipeline services
├── tests/                   # Unit & integration tests
│   ├── unit/
│   └── integration/
├── notebooks/               # EDA, hypothesis testing, modeling notebooks
├── scripts/                 # Automation scripts (data download, preprocessing, etc.)
├── docs/                    # Documentation and reference materials
├── data/                    # DVC-tracked data
│   ├── raw/                 # Original source data
│   └── processed/           # Cleaned and engineered datasets
├── config/                  # YAML or JSON configuration files
└── examples/                # Example usages and demos
```


## 🤝 Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request with clear documentation and commits.


## 📄 License

This project is licensed under the [MIT License](LICENSE).
