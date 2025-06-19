Installation
============

This section outlines the steps to set up the project environment and install all necessary dependencies.

Prerequisites
-------------
Before you begin, ensure you have the following installed on your system:

* **Python 3.8+** (recommended)
* **Git**

Steps
-----

1.  **Clone the repository:**

    If you are cloning from a remote repository:

    .. code-block:: bash

        git clone https://github.com/michaWorku/Insurance-Risk-Analytics-and-Predictive-Modeling.git
        cd Insurance-Risk-Analytics-and-Predictive-Modeling

    If you already have the project files in your current directory:

    .. code-block:: bash

        # You are already in the project root directory. No cloning needed.

2.  **Create and activate a virtual environment (recommended):**

    It is highly recommended to use a virtual environment to manage project dependencies isolation.

    .. code-block:: bash

        python3 -m venv venv
        source venv/bin/activate # On macOS/Linux
        # For Windows: .\venv\Scripts\activate

3.  **Install dependencies:**

    All required Python packages are listed in `requirements.txt`. Install them using pip:

    .. code-block:: bash

        pip install -r requirements.txt

Your project environment is now set up! You can proceed to the :doc:`usage` section to start exploring the notebooks.
