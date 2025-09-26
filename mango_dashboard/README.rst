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

Basic command:

.. code-block:: bash

   # From the mango project root
   cd C:\Users\NataliaGorrin\Desktop\Proyectos_baobab\mango

   # Using the CLI
   python -m mango_dashboard.file_explorer.cli.dashboard file_explorer --path "path/to/your/folder"

   # Or running directly
   streamlit run mango_dashboard/mango_dashboard/file_explorer/dashboards/file_explorer_app.py -- --path "path/to/your/folder"

Available parameters:

- ``--path`` or ``-p``: Path of the folder to explore (default: current directory)
- ``--editable`` or ``-e``: Enable dashboard editing (0=no, 1=yes, -1=default)
- ``--config_path`` or ``-c``: Path to the JSON configuration file
- ``--gcp_credentials_path``: Path to Google Cloud credentials (GCP only)

Usage examples:

**Explore a local folder:**

.. code-block:: bash

   python -m mango_dashboard.file_explorer.cli.dashboard file_explorer --path "path/to/your/folder"

**Explore with custom configuration:**

.. code-block:: bash

   python -m mango_dashboard.file_explorer.cli.dashboard file_explorer --path "./data" --config_path "./my_config.json" --editable 1

**Explore files in Google Cloud Storage:**

.. code-block:: bash

   python -m mango_dashboard.file_explorer.cli.dashboard file_explorer --path "gs://my-bucket/data" --gcp_credentials_path "./credentials.json"

Time Series Dashboard
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run the time series dashboard
   python -m mango_dashboard.time_series.cli.dashboard time_series

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

Troubleshooting
---------------

Error: "File does not exist"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Make sure to run the command from the ``mango`` project root:

.. code-block:: bash

   cd C:\Users\NataliaGorrin\Desktop\Proyectos_baobab\mango

Error: "TypeError: stat: path should be string, bytes, os.PathLike or integer, not NoneType"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This error has been fixed in the current version. If it persists, make sure you're using the latest version of the code.

Windows path issues
~~~~~~~~~~~~~~~~~~~

For Windows paths, use double quotes:

.. code-block:: bash

   python -m mango_dashboard.file_explorer.cli.dashboard file_explorer --path "G:\Mi unidad\data"

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