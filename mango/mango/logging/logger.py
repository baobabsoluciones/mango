import json
import logging
from datetime import datetime


class ColorFormatter(logging.Formatter):
    """
    ColorFormatter is a logging formatter that colors the log messages according to the log level.
    """

    # Color codes to escape the terminal
    ESCAPE_CODES = {
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

    # Map log levels to colors
    LEVEL_COLORS = {
        logging.DEBUG: ESCAPE_CODES["blue"],
        logging.INFO: ESCAPE_CODES["green"],
        logging.WARNING: ESCAPE_CODES["yellow"],
        logging.ERROR: ESCAPE_CODES["red"],
        logging.CRITICAL: ESCAPE_CODES["bold_red"],
    }

    def format(self, record):
        """
        Format the log message with color according to the log level.
        :param record: The log record to format.

        :return: The formatted log message with color.
        """
        # Apply color to message according to log level
        color = self.LEVEL_COLORS.get(record.levelno, self.ESCAPE_CODES["reset"])
        message = super().format(record)
        return f"{color}{message}{self.ESCAPE_CODES['reset']}"


class JSONFileHandler(logging.FileHandler):
    """
    Custom FileHandler that overwrites the file each time for JSON logging.
    """

    def __init__(self, filename, mode="w", encoding=None, delay=False):
        # Always use 'w' mode to overwrite the file
        super().__init__(filename, "w", encoding, delay)
        self.terminator = ""  # Disable automatic newline

    def emit(self, record):
        """
        Emit a record.
        """
        try:
            msg = self.format(record)
            # Write the JSON content directly, overwriting the file
            with open(self.baseFilename, "w", encoding=self.encoding) as f:
                f.write(msg)
        except Exception:
            self.handleError(record)


class JSONFormatter(logging.Formatter):
    """
    JSONFormatter is a logging formatter that formats the log messages as JSON.
    """

    def __init__(self, fields=None, datefmt=None, *args, **kwargs):
        """
        Initialize the JSONFormatter with optional fields and date format.

        :param fields: List of fields to include in the JSON log. Defaults to a standard set of fields.
        :param datefmt: Custom date format for the 'time' field.
        """
        super().__init__(*args, **kwargs)
        # Default fields if none are provided
        self.fields = fields or [
            "level",
            "message",
            "time",
            "name",
            "module",
            "filename",
            "lineno",
        ]
        # Custom date format
        self.datefmt = datefmt or "%Y-%m-%d %H:%M:%S"

        # Track log entries
        self._log_entries = []

    def format(self, record):
        """
        Format the log message as JSON.

        :param record: The log record to format.
        :return: The formatted log message as JSON.
        """
        # Create a dictionary with the specified fields
        log_record = {}
        if "level" in self.fields:
            log_record["level"] = record.levelname
        if "message" in self.fields:
            log_record["message"] = record.getMessage()
        if "time" in self.fields:
            log_record["time"] = self.formatTime(record, self.datefmt)
        if "name" in self.fields:
            log_record["name"] = record.name
        if "module" in self.fields:
            log_record["module"] = record.module
        if "filename" in self.fields:
            log_record["filename"] = record.filename
        if "lineno" in self.fields:
            log_record["lineno"] = record.lineno
        if "funcName" in self.fields:
            log_record["funcName"] = record.funcName

        # Add the current log entry to the list
        self._log_entries.append(log_record)

        # Return only the current list of entries as JSON
        return json.dumps(self._log_entries, indent=4)

    def formatTime(self, record, datefmt=None):
        """
        Override formatTime to use the specified date format.

        :param record: The log record.
        :param datefmt: Date format to use.
        :return: Formatted time string.
        """
        # Use the specified date format or the default
        format_to_use = datefmt or self.datefmt

        # Convert timestamp to datetime
        ct = datetime.fromtimestamp(record.created)

        # Format the datetime using the specified or default format
        return ct.strftime(format_to_use)


FORMATTERS = {
    "standard": {
        "format": "%(asctime)s | %(levelname)s | %(name)s: %(message)s",
        "datefmt": "%Y-%m-%d",
    },
    "standard_color": {
        "()": ColorFormatter,
        "format": "%(asctime)s | %(levelname)s | %(name)s: %(message)s",
        "datefmt": "%Y-%m-%d",
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
    "disable_existing_loggers": True,
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
    Ensure the log directory exists relative to the script's execution path.

    :param log_file_path: Relative or absolute log file path
    :return: Absolute path to the log file
    """
    import os
    import sys

    # Get the directory of the script being executed
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

    # If log_file_path is a relative path, join it with the script directory
    if not os.path.isabs(log_file_path):
        log_file_path = os.path.join(script_dir, log_file_path)

    # Ensure the log directory exists
    log_dir = os.path.dirname(os.path.abspath(log_file_path))
    os.makedirs(log_dir, exist_ok=True)

    return log_file_path


def get_configured_logger(
    logger_type: str = "mango_logging",
    config_dict: dict = None,
    log_console_level: int = None,
    log_console_format: str = None,
    log_console_datefmt: str = None,
    mango_color: bool = False,
    log_file_path: str = None,
    log_file_mode: str = "a",
    log_file_level: int = None,
    log_file_format: str = None,
    log_file_datefmt: str = None,
    json_fields: list = None,
) -> logging.Logger:
    """
    Get a configured logger with the specified parameters.

    :param logger_type: The type of logger to configure.
    :param config_dict: The configuration dictionary to use.
    :param log_console_level: The log level to set for console logging.
    :param log_console_format: The log format to use for console logging.
    :param log_console_datefmt: The log date format to use for console logging.
    :param mango_color: Whether to use color formatting for console logging.
    :param log_file_path: Path to the log file for file logging.
    :param log_file_mode: Mode for file logging.
    :param log_file_level: Log level for file logging.
    :param log_file_format: Format for file logging.
    :param log_file_datefmt: Date format for file logging.
    :param json_fields: Optional fields for JSON logging.
    :return: The configured logger.
    :raises ValueError: If the log file extension is not .txt, .json, or .log
    """
    import copy
    import logging.config

    # Deep copy configuration
    if config_dict is None:
        logging_config = copy.deepcopy(LOGGING_DICT_DEFAULT)
    else:
        logging_config = copy.deepcopy(config_dict)

    # Remove file handler if logger type doesn't have it and no log_file_path is specified
    if not log_file_path and "file" in logging_config["handlers"]:
        if logger_type in logging_config["loggers"]:
            if "file" not in logging_config["loggers"][logger_type]["handlers"]:
                del logging_config["handlers"]["file"]

    # Console handler configuration
    console_handler_name = "stream_color" if mango_color else "stream"
    console_formatter_name = "standard_color" if mango_color else "standard"
    console_handler_config = logging_config["handlers"][console_handler_name]

    # Set console log level
    if log_console_level is not None:
        console_handler_config["level"] = logging.getLevelName(log_console_level)
        logging_config["loggers"][logger_type]["level"] = logging.getLevelName(
            log_console_level
        )

    # Color formatting for console
    if mango_color:
        logging_config["handlers"][console_handler_name][
            "formatter"
        ] = console_formatter_name
        logging_config["loggers"][logger_type]["handlers"] = [console_handler_name]

    # Console format and date format
    log_console_format = (
        log_console_format or "%(asctime)s | %(levelname)s | %(name)s: %(message)s"
    )
    log_console_datefmt = log_console_datefmt or "%Y-%m-%d"

    logging_config["formatters"][console_formatter_name] = {
        "()": ColorFormatter if mango_color else logging.Formatter,
        "format": log_console_format,
        "datefmt": log_console_datefmt,
    }

    # File logging configuration
    if log_file_path:
        # Check file extension
        import os

        file_extension = os.path.splitext(log_file_path)[1].lower()
        allowed_extensions = [".txt", ".json", ".log"]
        if file_extension not in allowed_extensions:
            raise ValueError(f"Log file extension must be one of {allowed_extensions}")

        # Ensure log directory exists
        log_file_path = ensure_log_directory(log_file_path)

        # Determine if this is a JSON log file
        is_json_log = file_extension == ".json"

        # Determine file formatter and fields
        if is_json_log:
            # Use JSON formatter
            file_formatter_name = "json_formatter"

            # Use provided json_fields or default fields from JSONFormatter
            json_fields = json_fields or JSONFormatter().fields

            # Create JSON formatter configuration
            logging_config["formatters"][file_formatter_name] = {
                "()": JSONFormatter,
                "fields": json_fields,
                "datefmt": log_file_datefmt,
            }

            # Use custom JSONFileHandler for JSON logs
            file_handler_class = JSONFileHandler
        else:
            # Standard file logging
            file_formatter_name = "file_formatter"

            # Determine file formatter
            log_file_format = (
                log_file_format or "%(asctime)s | %(levelname)s | %(name)s: %(message)s"
            )
            log_file_datefmt = log_file_datefmt or "%Y-%m-%d"

            # Create standard formatter configuration
            logging_config["formatters"][file_formatter_name] = {
                "()": logging.Formatter,  # Always use standard Formatter for non-JSON logs
                "format": log_file_format,
                "datefmt": log_file_datefmt,
            }

            # Use standard FileHandler for non-JSON logs
            file_handler_class = logging.FileHandler

        # Create a new file handler configuration
        file_handler_config = {
            "level": logging.getLevelName(log_file_level or logging.INFO),
            "class": f"{file_handler_class.__module__}.{file_handler_class.__name__}",
            "filename": log_file_path,
            "mode": ("w" if is_json_log else log_file_mode),
            "formatter": file_formatter_name,
        }

        # Add the new file handler to handlers
        logging_config["handlers"]["file"] = file_handler_config

        # Update logger configuration to include file handler
        if "file" not in logging_config["loggers"][logger_type]["handlers"]:
            logging_config["loggers"][logger_type]["handlers"].append("file")

        # Set logger level to match file handler level
        logging_config["loggers"][logger_type]["level"] = logging.getLevelName(
            log_file_level or logging.INFO
        )

    # Apply logging configuration
    logging.config.dictConfig(logging_config)

    # Retrieve logger
    if logger_type not in logging_config["loggers"]:
        raise ValueError(f"Logger type {logger_type} not found in logging config")
    logger = logging.getLogger(logger_type)

    return logger


def get_basic_logger(
    log_file: str = None,
    console: bool = True,
    level: int = logging.INFO,
    format_str: str = "%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    datefmt: str = "%Y-%m-%d",
) -> logging.Logger:
    """
    DEPRECATED in version 0.4.0: Use get_configured_logger instead.
    Get a basic logger with console and/or file output.

    :param log_file: Path to the log file. If None, no file logging is used.
    :param console: Whether to enable console logging.
    :param level: The logging level to use.
    :param format_str: The format string for log messages.
    :param datefmt: The date format string.
    :return: The configured logger.
    """
    import warnings

    warnings.warn(
        "get_basic_logger is deprecated. Use get_configured_logger instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    logger_type = "mango_logging"
    if console and log_file:
        return get_configured_logger(
            logger_type=logger_type,
            log_console_level=level,
            log_console_format=format_str,
            log_console_datefmt=datefmt,
            log_file_path=log_file,
            log_file_level=level,
            log_file_format=format_str,
            log_file_datefmt=datefmt,
        )
    elif console:
        return get_configured_logger(
            logger_type=logger_type,
            log_console_level=level,
            log_console_format=format_str,
            log_console_datefmt=datefmt,
        )
    elif log_file:
        return get_configured_logger(
            logger_type=logger_type,
            log_file_path=log_file,
            log_file_level=level,
            log_file_format=format_str,
            log_file_datefmt=datefmt,
        )
    else:
        return get_configured_logger(
            logger_type=logger_type,
            log_console_level=level,
            log_console_format=format_str,
            log_console_datefmt=datefmt,
        )
