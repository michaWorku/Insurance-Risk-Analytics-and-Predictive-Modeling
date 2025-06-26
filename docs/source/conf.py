
import os
import sys
# Add the project root to the Python path so Sphinx can find your modules
sys.path.insert(0, os.path.abspath('../../src')) # Adjust path as necessary from docs/source/

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Insurance Risk Analytics & Predictive Modeling'
copyright = '2025, Mikias Worku' # Update this
author = 'Mikias Worku' # Update this
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # To automatically document Python code
    'sphinx.ext.napoleon', # To support NumPy and Google style docstrings
    'sphinx.ext.viewcode', # To link to source code
    'sphinx.ext.todo',     # To include todo notes
    'sphinx.ext.coverage', # To check documentation coverage
    'sphinx.ext.mathjax',  # For math rendering
]

# Configure Napoleon to parse docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Todo extension configuration
todo_include_todos = True

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Use a modern Sphinx theme
html_theme = 'sphinx_rtd_theme' # Or 'furo', 'alabaster'
html_static_path = ['_static']

# Set the URL for the external GitHub repository (optional but recommended for viewcode)
html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "michaworku", # Replace with your GitHub username
    "github_repo": "Insurance-Risk-Analytics-and-Predictive-Modeling", # Replace with your repository name
    "github_version": "main", # Or your default branch name (e.g., "master")
    "conf_py_path": "/docs/source/", # Path to conf.py relative to repo root
}

# Add any custom CSS (optional)
# html_css_files = [
#     'custom.css',
# ]
