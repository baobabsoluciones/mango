Mango
------

.. image:: https://img.shields.io/pypi/v/mango?label=version&logo=python&logoColor=white&style=for-the-badge&color=E58164
   :alt: PyPI
   :target: https://pypi.org/project/mango/
.. image:: https://img.shields.io/pypi/l/mango?color=blue&style=for-the-badge
  :alt: PyPI - License
  :target: https://github.com/baobabsoluciones/mango/blob/master/LICENSE
.. image:: https://img.shields.io/github/actions/workflow/status/baobabsoluciones/mango/build_docs.yml?label=docs&logo=github&style=for-the-badge
   :alt: GitHub Workflow Status
   :target: https://github.com/baobabsoluciones/mango/actions
.. image:: https://img.shields.io/codecov/c/gh/baobabsoluciones/mango?flag=unit-tests&label=coverage&logo=codecov&logoColor=white&style=for-the-badge&token=0KKRF3J95L
    :alt: Codecov
    :target: https://app.codecov.io/gh/baobabsoluciones/mango

This repository contains multiple libraries developed by the team at Baobab Soluciones: **mango**, **mango_autoencoder**, **mango_calendar**, **mango_dashboard**, **mango_genetic**, and **mango_time_series**. These libraries are the result of the team's experience, understanding, and knowledge of the Python language and its ecosystem.

The functions are divided in different modules so even though everything is imported the dependencies can be installed only for the modules that are needed.

Installation
===========

Core Library
-----------

The core mango library is modular and you can install additional features as needed:

**Command Line Interface:**
   pip install mango[cli]

**Data Processing:**
   pip install mango[data]

**Dashboard Creation:**
   pip install mango[dashboard]

**Google Cloud Integration:**
   pip install mango[gcloud]

**Optimization Models:**
   pip install mango[models]

**Plotting Capabilities:**
   pip install mango[plot]

**SHAP Analysis:**
   pip install mango[shap]

**Multiple dependencies:**
   pip install mango[data,plot,dashboard]

Individual Libraries
------------------

**Mango Autoencoder:**
   pip install mango-autoencoder

**Mango Calendar:**
   pip install mango-calendar

**Mango Dashboard:**
   pip install mango-dashboard

**Mango Genetic:**
   pip install mango-genetic

**Mango Time Series:**
   pip install mango-time-series

**Complete installation:**
   pip install mango[data,plot,dashboard,models,shap]
   pip install mango-autoencoder mango-calendar mango-dashboard mango-genetic mango-time-series

Core Library Modules
==================

**Core Functionality**
- mango.core: Base classes and utilities
- mango.cli: Command line interface tools (optional)

**Data Processing**
- mango.data: Data manipulation and processing tools
- mango.validators: Data validation utilities
- mango.transformers: Data transformation pipelines

**Visualization**
- mango.plots: Interactive plotting with plotly
- mango.dashboards: Streamlit-based dashboard creation

**Optimization**
- mango.models: Pyomo-based optimization models
- mango.solvers: Solver interfaces and utilities

**Cloud Integration**
- mango.gcloud: Google Cloud Storage integration

**Analysis**
- mango.shap: SHAP value analysis and visualization

Library Features
==============

Mango Autoencoder
----------------

- Neural Network Architecture: Encoder-decoder for dimensionality reduction
- Anomaly Detection: Built-in capabilities with threshold-based methods
- Data Processing: Sequence handling, normalization, and batch processing

Mango Calendar
-------------

- Calendar Operations: Date manipulation, holiday detection, business day logic
- Time Utilities: Date range operations, working day calculations, time zone handling

Mango Dashboard
--------------

- Interactive Dashboards: Streamlit-based creation with file explorer
- Data Visualization: Interactive charts, real-time updates, customizable layouts
- File Management: Browser, data upload, processing, and export capabilities

Mango Genetic
------------

- Genetic Algorithms: Individual and population management
- Optimization Framework: Abstract problem interface with multiple encoding types
- Advanced Operators: Tournament selection, blend crossover, adaptive mutation

Mango Time Series
----------------

- High Performance: Support for pandas and polars DataFrames with pyarrow
- Statistical Analysis: Advanced modeling, decomposition, anomaly detection
- Time Operations: Date handling, resampling, missing data imputation
- Integration: Seamless integration with mango[data] features

Development Installation
======================

For development installation of any library:

   git clone https://github.com/baobabsoluciones/mango.git
   cd mango/mango_time_series  # or mango_autoencoder, mango_calendar, etc.
   pip install -e .

Or using uv (recommended):

   uv venv
   uv sync
   uv run pip install -e .

Contributing
===========

We welcome contributions! Please see our `contributing guidelines <https://github.com/baobabsoluciones/mango/blob/master/CONTRIBUTING.rst>`_ for more details.

Discussion and Development
=========================

We encourage open discussion and collaboration on development via GitHub issues. If you have ideas, suggestions, or encounter any issues, please feel free to open an issue on our `GitHub repository <https://github.com/baobabsoluciones/mango/issues>`_.

License
=======

This project is licensed under the MIT License - see the `LICENSE <https://github.com/baobabsoluciones/mango/blob/master/LICENSE>`_ file for details.
