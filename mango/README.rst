Mango
------

A comprehensive Python library providing essential tools for data processing, analysis, and machine learning workflows.

Overview
--------

Mango is the core library that provides fundamental functionality for data manipulation, processing, and analysis. It serves as the foundation for the broader mango ecosystem of specialized libraries, offering a robust and flexible platform for data science and machine learning applications.

Main Features
-------------

**Data Processing**
- File handling for multiple formats (CSV, Excel, JSON)
- Data imputation and cleaning utilities
- Date and time manipulation functions
- Object processing and validation tools

**Data Management**
- Table operations with pytups integration
- Efficient data structures and tools
- Data validation and quality checks
- Flexible data transformation capabilities

**External Integrations**
- AEMET weather data client
- ArcGIS geospatial services
- REST API client utilities
- Cloud storage integration (Google Cloud)

**Machine Learning**
- Neural network implementations
- Model evaluation and benchmarking
- Optimization algorithms and benchmarks

**Utilities**
- Comprehensive logging system
- Configuration management
- Exception handling and validation
- Spatial and mathematical utilities

Quick Start
-----------

.. code-block:: python

   import mango
   from mango.processing import DataProcessor
   from mango.clients import AEMETClient
   from mango.table import Table

   # Initialize data processor
   processor = DataProcessor()
   
   # Load and process data
   data = processor.load_csv('data.csv')
   cleaned_data = processor.clean_data(data)
   
   # Use external services
   weather_client = AEMETClient()
   weather_data = weather_client.get_weather_data()

Installation
------------

**Using uv:**

.. code-block:: bash

   uv add mango

**Using pip:**

.. code-block:: bash

   pip install mango

Dependencies
------------

Core dependencies include numpy and other essentials. Optional dependencies are available for specific functionality:

- **data**: Pandas/Polars, scikit-learn, Pyomo and related tooling for data processing and ML features
- **dev**: Development and testing tools (combine with ``data`` to run the full test suite)

Documentation
-------------

Full documentation is available at: https://baobabsoluciones.github.io/mango/

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