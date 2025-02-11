Mango repository
--------------------

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


This repository contains two libraries developed by the team at Baobab Soluciones: **mango** and **mango_time_series**. These libraries are the result of the team's experience, understanding, and knowledge of the Python language and its ecosystem.

Libraries Overview
==================

Mango Core Library
~~~~~~~~~~~~~~~~

Core Dependencies
^^^^^^^^^^^^^^^

- Python >= 3.8
- numpy >= 1.24.4
- certifi >= 2023.7.22
- charset-normalizer >= 3.3.0
- et-xmlfile >= 1.1.0
- fastjsonschema >= 2.18.1
- idna >= 3.4
- openpyxl >= 3.1.2
- pydantic >= 2.4.2
- python-dateutil >= 2.8.2
- pytups >= 0.86.2
- pytz >= 2023.3
- requests >= 2.31.0
- six >= 1.16.0
- tqdm >= 4.66.1
- urllib3 >= 2.0.7
- XlsxWriter >= 3.1.9

Optional Dependencies
------------------

The library is modular, and you can install additional features as needed:

1. Command Line Interface:

   .. code-block:: bash

       pip install mango[cli]  # Adds click >= 8.1.7

2. Data Processing:

   .. code-block:: bash

       pip install mango[data]  # Adds pandas, holidays, pycountry, unidecode, tabulate

3. Dashboard Creation:

   .. code-block:: bash

       pip install mango[dashboard]  # Adds streamlit, pandas, plotly

4. Google Cloud Integration:

   .. code-block:: bash

       pip install mango[gcloud]  # Adds google-cloud-storage

5. Optimization Models:

   .. code-block:: bash

       pip install mango[models]  # Adds pyomo

6. Plotting Capabilities:

   .. code-block:: bash

       pip install mango[plot]  # Adds beautifulsoup4, pandas, plotly

7. SHAP Analysis:

   .. code-block:: bash

       pip install mango[shap]  # Adds shap with plotting capabilities

You can also install multiple optional dependencies at once:

.. code-block:: bash

    pip install mango[data,plot,dashboard]

Main Modules
^^^^^^^^^^

1. **Core Functionality**
   - ``mango.core``: Base classes and utilities
   - ``mango.cli``: Command line interface tools (optional)

2. **Data Processing**
   - ``mango.data``: Data manipulation and processing tools
   - ``mango.validators``: Data validation utilities
   - ``mango.transformers``: Data transformation pipelines

3. **Visualization**
   - ``mango.plots``: Interactive plotting with plotly
   - ``mango.dashboards``: Streamlit-based dashboard creation

4. **Optimization**
   - ``mango.models``: Pyomo-based optimization models
   - ``mango.solvers``: Solver interfaces and utilities

5. **Cloud Integration**
   - ``mango.gcloud``: Google Cloud Storage integration

6. **Analysis**
   - ``mango.shap``: SHAP value analysis and visualization

Mango Time Series Library
~~~~~~~~~~~~~~~~~~~~~~~

Core Dependencies
^^^^^^^^^^^^^^^

- Python >= 3.8
- pandas >= 2.0.3
- numpy >= 1.24.4
- polars >= 1.8.2
- scipy >= 1.10.1
- statsmodels >= 0.14.1
- mango[data] >= 0.2.0
- pyarrow >= 17.0.0

Main Features
^^^^^^^^^^

1. **High Performance Data Processing**
   - Support for both pandas and polars DataFrames
   - Efficient time series operations with pyarrow
   - Scalable data transformations

2. **Statistical Analysis**
   - Advanced statistical modeling with statsmodels
   - Time series decomposition
   - Anomaly detection
   - Seasonality analysis

3. **Time Series Operations**
   - Date and time handling
   - Resampling and rolling windows
   - Missing data imputation
   - Frequency conversion

4. **Integration with Mango Core**
   - Seamless integration with mango[data] features
   - Extended data validation
   - Enhanced plotting capabilities
   - Consistent API design

Installation
^^^^^^^^^^

Basic installation with all core dependencies:

.. code-block:: bash

    pip install mango-time-series

For development installation:

.. code-block:: bash

    git clone https://github.com/baobabsoluciones/mango.git
    cd mango/mango_time_series
    pip install -e .

Contributing
============

We welcome contributions! Please see our `contributing guidelines <https://github.com/baobabsoluciones/mango/blob/master/CONTRIBUTING.rst>`_ for more details.

Discussion and Development
=========================

We encourage open discussion and collaboration on development via GitHub issues. If you have ideas, suggestions, or encounter any issues, please feel free to open an issue on our `GitHub repository <https://github.com/baobabsoluciones/mango/issues>`_.

License
=======

This project is licensed under the MIT License - see the `LICENSE <https://github.com/baobabsoluciones/mango/blob/master/LICENSE>`_ file for details.
