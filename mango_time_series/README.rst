Mango Time Series
=================

A comprehensive Python library for time series analysis, forecasting, and feature engineering built on top of the Mango framework.

Overview
--------

Mango Time Series provides specialized tools for temporal data analysis, including exploratory data analysis, validation techniques, and utility functions for time series processing. It is designed to work seamlessly with the broader Mango ecosystem.

Features
--------

**Exploratory Analysis**
- Comprehensive time series data exploration tools
- Statistical analysis and visualization capabilities
- Data quality assessment and profiling

**Validation**
- Time series data validation techniques
- Cross-validation methods for temporal data
- Model validation and performance assessment

**Utilities**
- Data preprocessing and transformation tools
- Date and time manipulation functions
- Integration with pandas and other data science libraries

**Data Management**
- Efficient handling of large time series datasets
- Support for multiple data formats
- Memory-optimized processing

Installation
------------

**Using uv:**

.. code-block:: bash

   uv add mango-time-series

**Using pip:**

.. code-block:: bash

   pip install mango-time-series

Dependencies
------------

- Python >= 3.10
- pandas >= 2.0.0
- numpy >= 1.24.0
- mango[data] >= 0.3.0

Quick Start
-----------

.. code-block:: python

   from mango_time_series import TimeSeriesAnalyzer
   import pandas as pd

   # Load time series data
   data = pd.read_csv('your_time_series_data.csv')
   
   # Initialize analyzer
   analyzer = TimeSeriesAnalyzer(data)
   
   # Perform exploratory analysis
   analysis_results = analyzer.explore()
   
   # Generate validation report
   validation_report = analyzer.validate()

Documentation
-------------

For detailed documentation, visit the `Mango Documentation <https://baobabsoluciones.github.io/mango/>`_.

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.


Support
-------

For questions, issues, or contributions, please contact:

- Email: mango@baobabsoluciones.es
- Create an issue on the repository

---

Made with ❤️ by `baobab soluciones <https://baobabsoluciones.es/>`_