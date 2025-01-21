Logging
-------

The logging module provides advanced logging capabilities with support for console, file, and JSON logging formats. It includes color formatting for console output and customizable JSON fields for structured logging.

Features
~~~~~~~~
- Console logging with optional color formatting
- File logging with multiple format options 
- JSON logging with customizable fields
- Automatic log directory creation
- Configurable date and message formats

Basic Usage
~~~~~~~~~~

Here's a simple example of using the logger:

.. code-block:: python

    from mango.logging import get_configured_logger
    import logging

    # Basic console logger with colors
    logger = get_configured_logger(
        log_console_level=logging.DEBUG,
        mango_color=True
    )

    logger.debug("Debug message")
    logger.info("Info message") 
    logger.warning("Warning message")
    logger.error("Error message")

File Logging
~~~~~~~~~~~

To log to both console and file:

.. code-block:: python

    logger = get_configured_logger(
        log_console_level=logging.INFO,
        log_file_path="logs/app.log",
        log_file_level=logging.DEBUG,
        log_file_format="%(asctime)s | %(levelname)s || %(name)s: %(message)s",
        log_file_datefmt="%Y%m%d %H:%M:%S"
    )

    logger.info("This goes to both console and file")
    logger.debug("This only goes to file")

JSON Logging
~~~~~~~~~~~

For structured logging in JSON format:

.. code-block:: python

    logger = get_configured_logger(
        log_file_path="logs/app.json",
        json_fields=["level", "message", "time", "module"]
    )

    logger.info("This will be logged in JSON format")

API Reference
~~~~~~~~~~~~

get_configured_logger
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: mango.logging.logger.get_configured_logger

ColorFormatter
^^^^^^^^^^^^^
.. autoclass:: mango.logging.logger.ColorFormatter
    :members:
    :undoc-members:
    :show-inheritance:

JSONFormatter
^^^^^^^^^^^^
.. autoclass:: mango.logging.logger.JSONFormatter
    :members:
    :undoc-members:
    :show-inheritance:

JSONFileHandler
^^^^^^^^^^^^^^
.. autoclass:: mango.logging.logger.JSONFileHandler
    :members:
    :undoc-members:
    :show-inheritance:

Deprecated Functions
~~~~~~~~~~~~~~~~~~

get_basic_logger
^^^^^^^^^^^^^^^
.. autofunction:: mango.logging.logger.get_basic_logger

.. caution::
    The get_basic_logger function is deprecated and will be removed in a future version. Use get_configured_logger instead.

Chrono
~~~~~~
.. autoclass:: mango.logging.Chrono

Time decorator
~~~~~~~~~~~~~
.. autofunction:: mango.logging.log_time
