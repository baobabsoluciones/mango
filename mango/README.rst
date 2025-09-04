Mango
------

A comprehensive Python library providing essential tools for data processing, analysis, and machine learning workflows.

Overview
--------

Mango is the core library that provides fundamental functionality for data manipulation, processing, and analysis. It serves as the foundation for the broader mango ecosystem of specialized libraries.

Main Features
------------

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
- SHAP analysis tools
- Model evaluation and benchmarking
- Optimization algorithms and benchmarks

**Utilities**
- Comprehensive logging system
- Configuration management
- Exception handling and validation
- Spatial and mathematical utilities

Installation
------------

**Using uv (recommended):**

.. code-block:: bash

   git clone https://github.com/baobabsoluciones/mango.git
   cd mango
   uv venv
   uv sync
   uv run pip install -e .

**Using pip:**

.. code-block:: bash

   pip install mango

Dependencies
------------

Core dependencies include numpy, pandas, and other essential data science libraries. Optional dependencies are available for specific functionality:

- **models**: Pyomo for optimization
- **shap**: SHAP analysis with lightgbm and xgboost
- **dev**: Development and testing tools

Documentation
-------------

Full documentation is available at: https://mango.readthedocs.io/

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.