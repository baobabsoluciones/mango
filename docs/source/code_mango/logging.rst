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
The recommended approach is to configure the logger once in your main script (e.g., ``main.py``) and then import and use it across other modules. This ensures consistent logging behavior throughout your application.

JSON Logging Best Practices
^^^^^^^^^^^^^^^^^^^^^^^
Since JSON log files are not configured in append mode (they are overwritten each time), it's recommended to use dynamic filenames that include timestamps. This ensures each execution creates a new log file and preserves historical logs:

.. code-block:: python

    # main.py
    from mango.logging import get_configured_logger
    import logging
    from datetime import datetime

    def setup_logger():
        """Configure and return the application's central logger."""
        # Generate timestamp for unique log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_log_path = f"logs/app_{timestamp}.json"
        
        return get_configured_logger(
            logger_type="my_app",
            log_console_level=logging.INFO,
            mango_color=True,
            # Regular log file in append mode
            log_file_path="logs/app.log",
            log_file_level=logging.DEBUG,
            # JSON log file with timestamp
            json_file_path=json_log_path,
            json_fields=["level", "message", "time", "module", "lineno"]
        )

Executing Secondary Scripts
^^^^^^^^^^^^^^^^^^^^^^^
When you need to run scripts other than ``main.py`` directly, you can use a simple try-except pattern to handle the logger configuration:

.. code-block:: python

    # secondary_script.py
    from logging import getLogger
    
    try:
        # Try to get the existing logger from main
        logger = getLogger("my_app")
        # Test if the logger is configured by attempting to log
        logger.debug("Testing logger")
    except Exception:
        # If logger not configured, create a new one
        from mango.logging import get_configured_logger
        import logging
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = get_configured_logger(
            logger_type="my_app",
            log_console_level=logging.INFO,
            mango_color=True,
            log_file_path="logs/app.log",
            log_file_level=logging.DEBUG,
            # JSON log file with timestamp
            json_file_path=f"logs/app_{timestamp}.json",
            json_fields=["level", "message", "time", "module", "lineno"]
        )
    
    def main():
        logger.info("Secondary script started")
        # Your code here
    
    if __name__ == "__main__":
        main()

This approach:
- Attempts to use the existing logger from main.py
- Falls back to creating a new logger if main.py's logger is not available
- Maintains consistent logger configuration
- Requires no additional files or wrappers
- Works whether the script is run directly or imported

Using the Logger in Other Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Import and use the configured logger in other modules:

.. code-block:: python

    # other_module.py
    from logging import getLogger

    # Get the same logger instance configured in main.py
    logger = getLogger("my_app")

    def some_function():
        logger.debug("Entering some_function")
        try:
            # Your code here
            logger.info("Operation successful")
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            raise

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
