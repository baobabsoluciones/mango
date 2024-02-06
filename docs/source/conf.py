# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))


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

import mango

version = mango.__version__
release = mango.__version__


# Options for bibtex
bibtex_bibfiles = ["./refs.bib"]
bibtex_default_style = "plain"
