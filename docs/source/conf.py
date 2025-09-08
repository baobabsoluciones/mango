# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../mango/"))
sys.path.insert(0, os.path.abspath("../../mango_time_series/"))
sys.path.insert(0, os.path.abspath("../../mango_autoencoder/"))
sys.path.insert(0, os.path.abspath("../../mango_calendar/"))
sys.path.insert(0, os.path.abspath("../../mango_dashboard/"))
sys.path.insert(0, os.path.abspath("../../mango_genetic/"))


project = "mango"
copyright = "2023, baobab soluciones"
author = "baobab soluciones"
# release = "0.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.intersphinx", "sphinxcontrib.bibtex"]

templates_path = ["templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "shibuya"
html_static_path = ["static"]

# autodoc_mock_imports = ["mango"]
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "inherited-members": True}

from mango import mango

version = mango.__version__
release = mango.__version__


# Options for bibtex
bibtex_bibfiles = ["./refs.bib"]
bibtex_default_style = "plain"

autodoc_mock_imports = [
    "bs4",
    "certifi",
    "charset_normalizer",
    "click",
    "et_xmlfile",
    "fastjsonschema",
    "google",
    "google.cloud",
    "google.cloud.storage",
    "holidays",
    "idna",
    "jinja2",
    "lightgbm",
    "numpy",
    "openpyxl",
    "pandas",
    "Pillow",
    "plotly",
    "polars",
    "pyarrow",
    "pycountry",
    "pydantic",
    "pydantic_core",
    "pyomo",
    "python_dateutil",
    "pytups",
    "pytz",
    "requests",
    "sklearn",
    "scipy",
    "shap",
    "six",
    "statsforecast",
    "statsmodels",
    "streamlit",
    "streamlit_date_picker",
    "tensorflow",
    "tensorflow.keras",
    "keras",
    "tensorflow_io_gcs_filesystem",
    "tqdm",
    "unidecode",
    "urllib3",
    "xgboost",
    "XlsxWriter",
    "matplotlib",
]
