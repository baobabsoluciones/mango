"""
Mango Logging Module

This module provides advanced logging capabilities with support for console, file,
and JSON logging formats. It includes color formatting for console output and
customizable JSON fields for structured logging.

Features:
    - Console logging with optional color formatting
    - File logging with multiple format options
    - JSON logging with customizable fields
    - Automatic log directory creation
    - Configurable date and message formats
"""

import copy
import json
import logging
import logging.config
import os
import sys
import warnings
from typing import Dict, List, Optional


class ColorFormatter(logging.Formatter):
    """
    Enhance log readability by applying different colors to console log messages.

    :cvar Dict[str, str] _ESCAPE_CODES: Terminal color escape codes for text formatting
    :cvar Dict[int, str] _LEVEL_COLORS: Mapping of log levels to their corresponding colors
    """

    _ESCAPE_CODES = {
        "reset": "\033[39;49;0m",
        "bold": "\033[01m",
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "purple": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bg_black": "\033[40m",
        "bg_red": "\033[41m",
        "bg_green": "\033[42m",
        "bg_yellow": "\033[43m",
        "bg_blue": "\033[44m",
        "bg_purple": "\033[45m",
        "bg_cyan": "\033[46m",
        "bg_white": "\033[47m",
        "bold_black": "\033[30;01m",
        "bold_red": "\033[31;01m",
        "bold_green": "\033[32;01m",
        "bold_yellow": "\033[33;01m",
        "bold_blue": "\033[34;01m",
        "bold_purple": "\033[35;01m",
        "bold_cyan": "\033[36;01m",
        "bold_white": "\033[37;01m",
    }

    _LEVEL_COLORS = {
        logging.DEBUG: _ESCAPE_CODES["blue"],
        logging.INFO: _ESCAPE_CODES["green"],
        logging.WARNING: _ESCAPE_CODES["yellow"],
        logging.ERROR: _ESCAPE_CODES["red"],
        logging.CRITICAL: _ESCAPE_CODES["bold_red"],
    }

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
        format: Optional[str] = None,
    ):
        """
        Initialize the ColorFormatter.

        :param Optional[str] fmt: Format string for log messages
        :param Optional[str] datefmt: Date format string
        :param str style: Format style (default is '%')
        :param Optional[str] format: Alternative argument for format (for compatibility)
        """
        # Prefer 'format' if provided, otherwise use 'fmt'
        effective_fmt = (
            format or fmt or "%(asctime)s | %(levelname)s | %(name)s: %(message)s"
        )

        # Call the parent constructor with the effective format
        super().__init__(fmt=effective_fmt, datefmt=datefmt, style=style)

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with appropriate color.

        :param logging.LogRecord record: The log record to format
        :return: The formatted log message with color codes
        :rtype: str
        """
        color = self._LEVEL_COLORS.get(record.levelno, self._ESCAPE_CODES["reset"])
        message = super().format(record)
        return f"{color}{message}{self._ESCAPE_CODES['reset']}"


class JSONFormatter(logging.Formatter):
    """
    A formatter that outputs log records as JSON.

    This formatter creates structured log output in JSON format with
    customizable fields. It maintains a list of log entries that can be
    serialized as a JSON array.

    :ivar List[str] fields: List of fields to include in JSON output
    :ivar str datefmt: Date format string for timestamp formatting
    :ivar List[Dict] _log_entries: Internal list of formatted log entries

    Available fields:
        * level: Log level name (e.g., INFO, DEBUG) (default)
        * message: Log message (default)
        * time: Timestamp of the log record (default)
        * name: Name of the logger (default)
        * module: Module name where the log was generated (default)
        * filename: Filename where the log was generated (default)
        * lineno: Line number in the source code where the log was generated (default)
        * funcName: Function name where the log was generated
        * pathname: Full pathname of the source file where the log was generated
        * process: Process ID of the process where the log was generated
        * processName: Process name where the log was generated
        * thread: Thread ID of the thread where the log was generated
        * threadName: Thread name where the log was generated
    """

    def __init__(
        self, fields: Optional[List[str]] = None, datefmt: Optional[str] = None
    ):
        """
        Initialize the JSONFormatter.

        :param Optional[List[str]] fields: List of fields to include in JSON output
        :param Optional[str] datefmt: Date format string for timestamp formatting
        """
        super().__init__()
        self.fields = fields or [
            "level",
            "message",
            "time",
            "name",
            "module",
            "filename",
            "lineno",
        ]
        self.datefmt = datefmt or "%Y-%m-%d %H:%M:%S"
        self._log_entries = []

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as a JSON object.

        :param record: The log record to format
        :type record: logging.LogRecord

        :returns: JSON string containing the formatted log entries
        :rtype: str
        """
        log_record = {}
        field_mapping = {
            "level": lambda: record.levelname,
            "message": lambda: record.getMessage(),
            "time": lambda: self.formatTime(record, self.datefmt),
            "name": lambda: record.name,
            "module": lambda: record.module,
            "filename": lambda: record.filename,
            "lineno": lambda: record.lineno,
            "funcName": lambda: record.funcName,
            "pathname": lambda: record.pathname,
            "process": lambda: record.process,
            "processName": lambda: record.processName,
            "thread": lambda: record.thread,
            "threadName": lambda: record.threadName,
        }

        for field in self.fields:
            if field in field_mapping:
                log_record[field] = field_mapping[field]()

        self._log_entries.append(log_record)
        return json.dumps(self._log_entries, indent=4)


class JSONFileHandler(logging.FileHandler):
    """A file handler that overwrites the file with each log entry.

    This handler is specifically designed for JSON logging, ensuring that
    the output file always contains valid JSON by overwriting it completely
    with each update.
    """

    def __init__(
        self,
        filename: str,
        mode: str = "w",
        encoding: Optional[str] = None,
        delay: bool = False,
    ):
        """Initialize the JSONFileHandler.

        :param str filename: Path to the log file
        :param str mode: File open mode (always 'w' for this handler)
        :param Optional[str] encoding: File encoding
        :param bool delay: Whether to delay file opening
        """
        super().__init__(filename, "w", encoding, delay)
        self.terminator = ""

    def emit(self, record: logging.LogRecord) -> None:
        """Write the log record to file.

        :param logging.LogRecord record: The log record to write
        """
        try:
            msg = self.format(record)
            with open(self.baseFilename, "w", encoding=self.encoding) as f:
                f.write(msg)
        except Exception:
            self.handleError(record)


# Default logging configurations
FORMATTERS = {
    "standard": {
        "format": "%(asctime)s | %(levelname)s | %(name)s: %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    },
    "standard_color": {
        "()": ColorFormatter,
        "format": "%(asctime)s | %(levelname)s | %(name)s: %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    },
    "json": {"()": JSONFormatter},
}

HANDLERS = {
    "stream": {
        "level": "INFO",
        "formatter": "standard",
        "class": "logging.StreamHandler",
    },
    "stream_color": {
        "level": "INFO",
        "formatter": "standard_color",
        "class": "logging.StreamHandler",
    },
    "file": {
        "level": "INFO",
        "formatter": "standard",
        "class": "logging.FileHandler",
        "filename": "app.log",
    },
}

LOGGING_DICT_DEFAULT = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": FORMATTERS,
    "handlers": HANDLERS,
    "loggers": {
        "mango_logging": {
            "handlers": ["stream"],
            "level": "INFO",
            "propagate": False,
        },
        "mango_logging_color": {
            "handlers": ["stream_color"],
            "level": "INFO",
            "propagate": False,
        },
    },
}


def ensure_log_directory(log_file_path: str) -> str:
    """
    Ensure the log directory exists and return the absolute path.

    :param str log_file_path: Relative or absolute path to the log file
    :return: Absolute path to the log file
    :rtype: str
    """
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    if not os.path.isabs(log_file_path):
        log_file_path = os.path.join(script_dir, log_file_path)

    log_dir = os.path.dirname(os.path.abspath(log_file_path))
    os.makedirs(log_dir, exist_ok=True)

    return log_file_path


def get_configured_logger(
    logger_type: str = "mango_logging",
    config_dict: Optional[Dict] = None,
    log_console_level: Optional[int] = None,
    log_console_format: Optional[str] = None,
    log_console_datefmt: Optional[str] = None,
    mango_color: bool = False,
    log_file_path: Optional[str] = None,
    log_file_mode: str = "a",
    log_file_level: Optional[int] = None,
    log_file_format: Optional[str] = None,
    log_file_datefmt: Optional[str] = None,
    json_fields: Optional[List[str]] = None,
) -> logging.Logger:
    """
    Configure and return a logger with flexible logging settings.

    Provides a comprehensive way to configure logging with support for
    console and file output, including advanced features like JSON formatting
    and colored console logs.

    :param logger_type: The type of logger to configure
    :param config_dict: Custom configuration dictionary
    :param log_console_level: Console logging level
    :param log_console_format: Console log message format
    :param log_console_datefmt: Console date format
    :param mango_color: Enable colored console output
    :param log_file_path: Path to log file
    :param log_file_mode: File open mode
    :param log_file_level: File logging level
    :param log_file_format: File log message format
    :param log_file_datefmt: File date format
    :param json_fields: Fields to include in JSON output. Options are:
        - level: Log level name (e.g., INFO, DEBUG) (default)
        - message: Log message (default)
        - time: Timestamp of the log record (default)
        - name: Name of the logger (default)
        - module: Module name where the log was generated (default)
        - filename: Filename where the log was generated (default)
        - lineno: Line number in the source code where the log was generated (default)
        - funcName: Function name where the log was generated
        - pathname: Full pathname of the source file where the log was generated
        - process: Process ID of the process where the log was generated
        - processName: Process name where the log was generated
        - thread: Thread ID of the thread where the log was generated
        - threadName: Thread name where the log was generated
    :return: Configured logger instance
    :rtype: logging.Logger
    :raises ValueError: If log file extension is invalid or logger type not found
    """
    # Special handling for root logger
    if logger_type == "root":
        logger = logging.getLogger()
        logger.handlers = []

        # Check if log_file_path is provided
        if log_file_path:
            if log_console_level or log_file_level:
                # Set the minimum level for the logger or 30 - WARNING (root logger default)
                logger.setLevel(min(log_console_level or 30, log_file_level or 30))
        else:
            logger.setLevel(log_console_level or 30)

        # Create a console handler with the specified format
        console_handler = logging.StreamHandler()

        # Use the specified format or default
        formatter_format = (
            log_console_format or "%(asctime)s | %(levelname)s | %(name)s: %(message)s"
        )
        formatter_datefmt = log_console_datefmt or "%Y-%m-%d"

        # Apply color formatting if mango_color is True
        if mango_color:
            formatter = ColorFormatter(fmt=formatter_format, datefmt=formatter_datefmt)
        else:
            formatter = logging.Formatter(
                fmt=formatter_format, datefmt=formatter_datefmt
            )

        console_handler.setFormatter(formatter)

        # Set handler level to match logger level or specified console level
        if log_console_level is not None:
            console_handler.setLevel(log_console_level)
        else:
            console_handler.setLevel(logger.level)

        # Add the handler to the root logger
        logger.addHandler(console_handler)

        # Add file handler if specified
        if log_file_path:
            file_extension = os.path.splitext(log_file_path)[1].lower()
            if file_extension not in {".txt", ".json", ".log"}:
                raise ValueError("Log file extension must be .txt, .json, or .log")

            log_file_path = ensure_log_directory(log_file_path)
            is_json_log = file_extension == ".json"

            if is_json_log:
                file_formatter = JSONFormatter(
                    fields=json_fields, datefmt=log_file_datefmt
                )
                file_handler = JSONFileHandler(log_file_path, mode="w")
            else:
                file_format = (
                    log_file_format
                    or "%(asctime)s | %(levelname)s | %(name)s: %(message)s"
                )
                file_formatter = logging.Formatter(
                    fmt=file_format, datefmt=log_file_datefmt or "%Y-%m-%d"
                )
                file_handler = logging.FileHandler(log_file_path, mode=log_file_mode)

            file_handler.setFormatter(file_formatter)
            if log_file_level is not None:
                file_handler.setLevel(log_file_level)
            else:
                file_handler.setLevel(logger.level)

            logger.addHandler(file_handler)

        return logger
    else:
        if "AIRFLOW_HOME" in os.environ:
            logger = logging.getLogger(__name__)
        else:
            logger = logging.getLogger(logger_type)

    # If no configuration is provided, use default
    logging_config = copy.deepcopy(config_dict or LOGGING_DICT_DEFAULT)

    # Dynamically add logger configuration if not exists
    if logger_type not in logging_config["loggers"]:
        logging_config["loggers"][logger_type] = {
            "handlers": ["stream"],
            "level": logging.INFO,
            "propagate": False,
        }

    # Remove file handler if not needed
    if not log_file_path and "file" in logging_config["handlers"]:
        if logger_type in logging_config["loggers"] or logger_type == "root":
            if "file" not in logging_config["loggers"].get(logger_type, {}).get(
                "handlers", []
            ):
                del logging_config["handlers"]["file"]

    # Configure console handler
    console_handler_name = "stream_color" if mango_color else "stream"
    console_formatter_name = "standard_color" if mango_color else "standard"
    console_handler_config = logging_config["handlers"][console_handler_name]

    if log_console_level is not None:
        console_handler_config["level"] = logging.getLevelName(log_console_level)

        # Special handling for root logger
        if logger_type == "root":
            logger.setLevel(logging.getLevelName(log_console_level))
        elif logger_type in logging_config["loggers"]:
            logging_config["loggers"][logger_type]["level"] = logging.getLevelName(
                log_console_level
            )

    if mango_color:
        console_handler_config["formatter"] = console_formatter_name

        # Special handling for root logger
        if logger_type == "root":
            logger.handlers = []  # Clear existing handlers
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                ColorFormatter(
                    fmt=log_console_format
                    or "%(asctime)s | %(levelname)s | %(name)s: %(message)s",
                    datefmt=log_console_datefmt or "%Y-%m-%d",
                )
            )
            logger.addHandler(console_handler)
        elif logger_type in logging_config["loggers"]:
            logging_config["loggers"][logger_type]["handlers"] = [console_handler_name]

    # Configure console format
    log_console_format = (
        log_console_format or "%(asctime)s | %(levelname)s | %(name)s: %(message)s"
    )
    log_console_datefmt = log_console_datefmt or "%Y-%m-%d"

    logging_config["formatters"][console_formatter_name] = {
        "()": ColorFormatter if mango_color else logging.Formatter,
        "fmt": log_console_format,
        "datefmt": log_console_datefmt,
    }

    # Configure file handler if needed
    if log_file_path:
        file_extension = os.path.splitext(log_file_path)[1].lower()
        if file_extension not in {".txt", ".json", ".log"}:
            raise ValueError("Log file extension must be .txt, .json, or .log")

        log_file_path = ensure_log_directory(log_file_path)
        is_json_log = file_extension == ".json"

        if is_json_log:
            formatter_config = {
                "()": JSONFormatter,
                "fields": json_fields or JSONFormatter().fields,
                "datefmt": log_file_datefmt,
            }
            handler_class = JSONFileHandler
        else:
            formatter_config = {
                "()": logging.Formatter,
                "format": log_file_format
                or "%(asctime)s | %(levelname)s | %(name)s: %(message)s",
                "datefmt": log_file_datefmt or "%Y-%m-%d",
            }
            handler_class = logging.FileHandler

        logging_config["formatters"]["file_formatter"] = formatter_config
        logging_config["handlers"]["file"] = {
            "level": logging.getLevelName(log_file_level or logging.INFO),
            "class": f"{handler_class.__module__}.{handler_class.__name__}",
            "filename": log_file_path,
            "mode": "w" if is_json_log else log_file_mode,
            "formatter": "file_formatter",
        }

        # Add file handler to logger's handlers
        if "file" not in logging_config["loggers"][logger_type]["handlers"]:
            logging_config["loggers"][logger_type]["handlers"].append("file")

    # If not root logger, apply configuration
    if logger_type != "root":
        logging.config.dictConfig(logging_config)

    if logger_type not in logging_config["loggers"] and logger_type != "root":
        raise ValueError(f"Logger type {logger_type} not found in logging config")

    return logger


def get_basic_logger(
    log_file: Optional[str] = None,
    console: bool = True,
    level: int = logging.INFO,
    format_str: str = "%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    datefmt: str = "%Y-%m-%d",
) -> logging.Logger:
    """Get a basic logger with console and/or file output (Deprecated).

    This function is deprecated and will be removed in a future version.
    Use get_configured_logger instead.

    :param log_file: Path to the log file
    :param console: Enable console output
    :param level: Logging level
    :param format_str: Log message format
    :param datefmt: Date format
    :return: Configured logger instance
    :rtype: logging.Logger
    :deprecated: Version 0.4.0: Use get_configured_logger instead
    """
    warnings.warn(
        "get_basic_logger is deprecated. Use get_configured_logger instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Determine the logger type
    logger_type = "root" if not console else "mango_logging"

    # Configure the logger
    logger = get_configured_logger(
        logger_type=logger_type,
        log_console_level=level if console else None,
        log_console_format=format_str,
        log_console_datefmt=datefmt,
        log_file_path=log_file,
        log_file_level=level if log_file else None,
        log_file_format=format_str,
        log_file_datefmt=datefmt,
    )

    return logger
