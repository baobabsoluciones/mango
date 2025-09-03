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


This repository contains multiple libraries developed by the team at Baobab Soluciones: **mango**, **mango_autoencoder**, **mango_calendar**, **mango_dashboard**, **mango_genetic**, and **mango_time_series**. These libraries are the result of the team's experience, understanding, and knowledge of the Python language and its ecosystem.

Libraries Overview
==================

Mango Core Library
~~~~~~~~~~~~~~~~

Optional Dependencies
------------------

The library is modular, and you can install additional features as needed:

1. Command Line Interface:

   .. code-block:: bash

       pip install mango[cli]

2. Data Processing:

   .. code-block:: bash

       pip install mango[data]

3. Dashboard Creation:

   .. code-block:: bash

       pip install mango[dashboard]

4. Google Cloud Integration:

   .. code-block:: bash

       pip install mango[gcloud]

5. Optimization Models:

   .. code-block:: bash

       pip install mango[models]

6. Plotting Capabilities:

   .. code-block:: bash

       pip install mango[plot]

7. SHAP Analysis:

   .. code-block:: bash

       pip install mango[shap]

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

Mango Autoencoder Library
~~~~~~~~~~~~~~~~~~~~~~~~

Main Features
^^^^^^^^^^

1. **Neural Network Architecture**
   - Encoder-decoder architecture for dimensionality reduction
   - Configurable network layers and activation functions
   - Support for various input data types

2. **Anomaly Detection**
   - Built-in anomaly detection capabilities
   - Threshold-based detection methods
   - Performance metrics and evaluation

3. **Data Processing**
   - Sequence handling and preprocessing
   - Data normalization and scaling
   - Batch processing support

Installation
^^^^^^^^^^

.. code-block:: bash

    pip install mango-autoencoder

Mango Calendar Library
~~~~~~~~~~~~~~~~~~~~

Main Features
^^^^^^^^^^

1. **Calendar Operations**
   - Date manipulation and calculations
   - Holiday detection and business day logic
   - Calendar feature extraction

2. **Time Utilities**
   - Date range operations
   - Working day calculations
   - Time zone handling

Installation
^^^^^^^^^^

.. code-block:: bash

    pip install mango-calendar

Mango Dashboard Library
~~~~~~~~~~~~~~~~~~~~~

Main Features
^^^^^^^^^^

1. **Interactive Dashboards**
   - Streamlit-based dashboard creation
   - File explorer and visualization tools
   - Time series analysis dashboards

2. **Data Visualization**
   - Interactive charts and plots
   - Real-time data updates
   - Customizable dashboard layouts

3. **File Management**
   - File browser and explorer
   - Data upload and processing
   - Export capabilities

Installation
^^^^^^^^^^

.. code-block:: bash

    pip install mango-dashboard

Mango Genetic Library
~~~~~~~~~~~~~~~~~~~

Main Features
^^^^^^^^^^

1. **Genetic Algorithms**
   - Individual and population management
   - Selection, crossover, and mutation operators
   - Configurable genetic parameters

2. **Optimization Framework**
   - Abstract problem interface
   - Multiple encoding types (real, binary, integer, categorical)
   - Fitness evaluation and optimization

3. **Advanced Operators**
   - Tournament and rank-based selection
   - Blend, split, and gaussian crossover
   - Adaptive mutation control

Installation
^^^^^^^^^^

.. code-block:: bash

    pip install mango-genetic

Mango Time Series Library
~~~~~~~~~~~~~~~~~~~~~~~

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

For development installation of any library:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/baobabsoluciones/mango.git
    
    # Navigate to specific library
    cd mango/mango_time_series  # or mango_autoencoder, mango_calendar, etc.
    
    # Install in development mode
    pip install -e .
    
    # Or using uv (recommended)
    uv venv
    uv sync
    uv run pip install -e .

Complete Installation
^^^^^^^^^^^^^^^^^^

To install all mango libraries:

.. code-block:: bash

    # Core mango library with all optional dependencies
    pip install mango[data,plot,dashboard,models,shap]
    
    # Individual libraries
    pip install mango-autoencoder
    pip install mango-calendar
    pip install mango-dashboard
    pip install mango-genetic
    pip install mango-time-series

Contributing
============

We welcome contributions! Please see our `contributing guidelines <https://github.com/baobabsoluciones/mango/blob/master/CONTRIBUTING.rst>`_ for more details.

Discussion and Development
=========================

We encourage open discussion and collaboration on development via GitHub issues. If you have ideas, suggestions, or encounter any issues, please feel free to open an issue on our `GitHub repository <https://github.com/baobabsoluciones/mango/issues>`_.

License
=======

This project is licensed under the MIT License - see the `LICENSE <https://github.com/baobabsoluciones/mango/blob/master/LICENSE>`_ file for details.
