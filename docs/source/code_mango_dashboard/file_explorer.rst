File Explorer Dashboard
========================

The File Explorer Dashboard provides an interactive web interface for browsing, managing, and analyzing files from both local file systems and Google Cloud Storage. It offers a comprehensive file management solution with visualization capabilities.

Features
--------

- **Multi-source Support**: Browse files from local directories and Google Cloud Storage
- **File Type Support**: Handle various file formats including images, markdown, JSON, and HTML
- **Interactive Tree View**: Navigate through directory structures with an intuitive tree interface
- **File Preview**: Preview content of supported file types directly in the browser
- **Configuration Management**: Save and load dashboard configurations
- **Responsive Design**: Optimized for different screen sizes

Main Components
---------------

File Explorer App
~~~~~~~~~~~~~~~~~

.. autoclass:: mango_dashboard.file_explorer.dashboards.file_explorer_app.FileExplorerApp
   :members:
   :undoc-members:
   :show-inheritance:

File Explorer Handlers
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: mango_dashboard.file_explorer.dashboards.file_explorer_handlers.FileExplorerHandler
   :members:
   :undoc-members:
   :show-inheritance:

Local File System Handler
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: mango_dashboard.file_explorer.dashboards.file_explorer_handlers.LocalFileExplorerHandler
   :members:
   :undoc-members:
   :show-inheritance:

Google Cloud Storage Handler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: mango_dashboard.file_explorer.dashboards.file_explorer_handlers.GCPFileExplorerHandler
   :members:
   :undoc-members:
   :show-inheritance:

Usage
-----

The File Explorer Dashboard can be launched using Streamlit:

.. code-block:: bash

    streamlit run mango_dashboard/file_explorer/dashboards/file_explorer_app.py


