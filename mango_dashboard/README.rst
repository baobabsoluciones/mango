Mango Dashboard
===============

Mango Dashboard is an interactive web application built with Streamlit that provides tools for exploring and visualizing data files. It includes two main modules: **File Explorer** and **Time Series Dashboard**.

Features
--------

File Explorer
~~~~~~~~~~~~~

- **Interactive file explorer**: Navigate through folders and files visually
- **Multiple format visualization**: Supports CSV, Excel, JSON, HTML, images (PNG, JPG, JPEG) and Markdown
- **Data editing**: Edit and save CSV, Excel and JSON files directly from the interface
- **Flexible configuration**: Customize layout with multiple rows and columns
- **GCP support**: Integration with Google Cloud Storage for remote files
- **Directory tree**: Hierarchical visualization of folder structure

Time Series Dashboard
~~~~~~~~~~~~~~~~~~~~~

- **Time series analysis**: Specialized tools for temporal data
- **Data loading**: Interface for uploading and processing time series files
- **Visualizations**: Interactive charts for temporal analysis
- **Forecast templates**: Predefined templates for forecasting models

Installation
------------

**Using uv:**

.. code-block:: bash

   uv add mango-dashboard

**Using pip:**

.. code-block:: bash

   pip install mango-dashboard

Usage
-----

File Explorer
~~~~~~~~~~~~~

Launch the File Explorer dashboard using the CLI:

**Using uv (development)**

.. code-block:: bash

   uv run mango-dashboard-fe dashboard file_explorer --path "path/to/your/folder"

**After installation**

.. code-block:: bash

   mango-dashboard-fe dashboard file_explorer --path "path/to/your/folder"


Available parameters:

- ``--path`` or ``-p``: Path of the folder to explore (default: current directory)
- ``--editable`` or ``-e``: Enable dashboard editing (0=no, 1=yes, -1=default)
- ``--config_path`` or ``-c``: Path to the JSON configuration file
- ``--gcp_credentials_path``: Path to Google Cloud credentials (GCP only)

Usage examples:

**Local folder**

.. code-block:: bash

   mango-dashboard-fe dashboard file_explorer --path "./data"

**Existing configuration**

.. code-block:: bash

   mango-dashboard-fe dashboard file_explorer --path "./data" --config_path "./config.json" --editable 0

**Google Cloud Storage**

.. code-block:: bash

   mango-dashboard-fe dashboard file_explorer --path "gs://my-bucket/data" --gcp_credentials_path "./credentials.json"


Time Series Dashboard
~~~~~~~~~~~~~~~~~~~~~

Launch the time series dashboard:

**Using uv (development)**

.. code-block:: bash

   uv run mango-dashboard-ts dashboard time_series

**After installation**

.. code-block:: bash
   
   mango-dashboard-ts dashboard time_series

Optional parameters:

- ``--project_name``: Dashboard project name (or env `TS_DASHBOARD_PROJECT_NAME`)
- ``--logo_url``: Logo URL (https:// or file:///)
- ``--experimental_features``: Enable experimental features (0 or 1)

Configuration
-------------

JSON Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~

The File Explorer uses a JSON configuration file to customize the interface:

.. code-block:: json

   {
       "title": "My Custom Dashboard",
       "header": "Data Explorer",
       "icon": ":chart_with_upwards_trend:",
       "layout": "wide",
       "dir_path": "/path/to/my/data",
       "n_rows": 2,
       "n_cols_1": 1,
       "n_cols_2": 2,
       "editable": true,
       "dict_layout": {
           "file_1_1": "/path/to/file1.csv",
           "file_2_1": "/path/to/file2.html",
           "file_2_2": "/path/to/file3.xlsx"
       }
   }

Configuration Parameters:

- **title**: Application title
- **header**: Main header
- **icon**: Page icon (emoji or icon code)
- **layout**: Page layout ("wide" or "centered")
- **dir_path**: Default directory path
- **n_rows**: Number of rows in the layout
- **n_cols_X**: Number of columns in row X
- **editable**: Whether the dashboard is editable
- **dict_layout**: Mapping of files to specific positions

Supported File Formats
----------------------

Visualization
~~~~~~~~~~~~~

- **CSV**: Editable tables with pandas
- **Excel (.xlsx)**: Multiple sheets with tabs
- **JSON**: Table visualization or raw JSON
- **HTML**: Embedded Plotly charts
- **Images**: PNG, JPG, JPEG with size controls
- **Markdown**: Rendered as HTML

Editing
~~~~~~~

- **CSV**: Data editor with pandas
- **Excel**: Sheet editor
- **JSON**: Table editor or raw JSON

Advanced Features
-----------------

Google Cloud Storage Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Support for ``gs://`` paths
- Authentication with JSON credential files
- Remote bucket and object exploration

Visualization Customization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Width and height control for images and HTML charts
- Flexible layout with multiple rows and columns
- Persistent configuration in JSON files

Editing Features
~~~~~~~~~~~~~~~~

- Inline editing of tabular data
- Automatic change saving
- Support for multiple output formats

Project Structure
-----------------

::

   mango_dashboard/
   ├── mango_dashboard/
   │   ├── file_explorer/
   │   │   ├── cli/
   │   │   │   └── dashboard.py          # CLI for File Explorer
   │   │   └── dashboards/
   │   │       ├── file_explorer_app.py  # Main application
   │   │       └── file_explorer_handlers.py  # File handlers
   │   └── time_series/
   │       ├── cli/
   │       │   └── dashboard.py          # CLI for Time Series
   │       └── dashboards/
   │           └── time_series_app.py    # Time series application
   ├── README.rst
   └── pyproject.toml


Contributing
------------

1. Fork the project
2. Create a feature branch (``git checkout -b feature/AmazingFeature``)
3. Commit your changes (``git commit -m 'Add some AmazingFeature'``)
4. Push to the branch (``git push origin feature/AmazingFeature``)
5. Open a Pull Request

License
-------

This project is under the MIT License. See the ``LICENSE`` file for more details.


Support
-------

For questions, issues, or contributions, please contact:

- Email: mango@baobabsoluciones.es
- Create an issue on the repository

---

Made with ❤️ by `baobab soluciones <https://baobabsoluciones.es/>`_