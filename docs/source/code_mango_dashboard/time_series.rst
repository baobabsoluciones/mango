Time Series Dashboard
======================

The Time Series Dashboard provides a comprehensive web-based interface for time series analysis, visualization, and forecasting. It offers advanced analytical capabilities with an intuitive user interface for exploring temporal data patterns and generating predictions.

.. note::
    This dashboard is currently under active development. Some features may be experimental or subject to change. If you encounter any issues or have suggestions, please open an issue in the `GitHub repository <https://github.com/baobabsoluciones/mango>`_.

Features
--------

- **Multi-language Support**: Available in English and Spanish
- **Data Upload**: Support for CSV and Excel file formats with custom parsing options
- **Interactive Visualizations**: Multiple plot types including original series, STL decomposition, lag analysis, and seasonality analysis
- **Automated Forecasting**: Integration with StatsForecast for time series predictions
- **Cross-validation**: Built-in model validation and error analysis
- **Model Comparison**: Compare multiple forecasting models with performance metrics
- **Export Capabilities**: Download results and generated code templates
- **Experimental Features**: Advanced analysis tools (toggleable)

Main Application
----------------

Time Series App
~~~~~~~~~~~~~~~

.. autofunction:: mango_dashboard.time_series.dashboards.time_series_app.interface_visualization

Utility Components
------------------

Data Loading and Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: mango_dashboard.time_series.dashboards.time_series_utils.data_loader
   :members:

.. automodule:: mango_dashboard.time_series.dashboards.time_series_utils.data_processing
   :members:

File Management
~~~~~~~~~~~~~~~

.. automodule:: mango_dashboard.time_series.dashboards.time_series_utils.file_uploader
   :members:

User Interface Components
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: mango_dashboard.time_series.dashboards.time_series_utils.ui_components
   :members:

Configuration
~~~~~~~~~~~~~

.. automodule:: mango_dashboard.time_series.dashboards.time_series_utils.constants
   :members:

Usage
-----

The Time Series Dashboard can be launched using Streamlit:

.. code-block:: bash

    streamlit run mango_dashboard/time_series/dashboards/time_series_app.py

Environment Variables
---------------------

The dashboard can be configured using the following environment variables:

- ``TS_DASHBOARD_PROJECT_NAME``: Set the project name displayed in the dashboard
- ``TS_DASHBOARD_EXPERIMENTAL_FEATURES``: Enable experimental features (true/false)