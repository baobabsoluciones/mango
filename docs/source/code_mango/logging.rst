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

Best Practices
~~~~~~~~~~~~~

Centralized Logger Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The recommended approach is to create a dedicated script for logger configuration (e.g., ``logger.py``) and import it across other modules. This ensures consistent logging behavior throughout your application by having a single source of truth for logger setup.

Creating the Logger Script
^^^^^^^^^^^^^^^^^^^^^^^^^^
Create a script named ``logger.py`` in your source directory to configure and return the application's central logger:

.. code-block:: python

    # Configuration for the application logger
    from mango.logging import get_configured_logger

    # Define log paths and create logger instance
    logger = get_configured_logger(
        logger_type="my_app",
        log_console_level=logging.INFO,
        mango_color=True,
        log_file_path="logs/app.log",
        log_file_level=logging.DEBUG,
    )

Using the Logger in Scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^^
In any script where you need logging, simply import the logger from ``logger.py``:

.. code-block:: python

    # Import the configured logger from the central configuration
    from logger import logger

    def main():
        # Log the start of script execution
        logger.info("Script started")
        value = 42
        # Example using f-string for logging
        logger.debug(f"Processing value: {value}")
        # Your code here

    if __name__ == "__main__":
        main()

This approach:
- Centralizes logger configuration in a single script
- Ensures consistent logger setup across all modules
- Simplifies logger usage by importing from ``logger.py``
- Avoids redundant logger configuration code


Logging Guidelines
^^^^^^^^^^^^^^^
- Use appropriate log levels:
    - DEBUG (10): Detailed information for debugging
    - INFO (20): General operational events
    - WARNING (30): Unexpected but handled situations
    - ERROR (40): Errors that prevent normal operation
    - CRITICAL (50): Critical issues requiring immediate attention

- Include contextual information in log messages
- Use structured logging (JSON) for machine parsing
- Avoid logging sensitive information

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
    :no-inherited-members:

JSONFormatter
^^^^^^^^^^^^
.. autoclass:: mango.logging.logger.JSONFormatter
    :members:
    :undoc-members:
    :no-inherited-members:

JSONFileHandler
^^^^^^^^^^^^^^
.. autoclass:: mango.logging.logger.JSONFileHandler
    :members:
    :undoc-members:
    :no-inherited-members:

Deprecated Functions
~~~~~~~~~~~~~~~~~~

get_basic_logger
^^^^^^^^^^^^^^^
.. autofunction:: mango.logging.logger.get_basic_logger

.. caution::
    The get_basic_logger function is deprecated and will be removed in a future version. Use get_configured_logger instead.

Chrono
~~~~~~
.. autoclass:: mango.logging.chrono.Chrono

Time decorator
~~~~~~~~~~~~~~
.. autofunction:: mango.logging.decorators.log_time
