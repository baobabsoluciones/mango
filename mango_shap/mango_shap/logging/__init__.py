"""
Logging utilities for mango_shap.

This module provides advanced logging capabilities with support for console, file,
and JSON logging formats. It includes color formatting for console output and
customizable JSON fields for structured logging.

Example:
    >>> from mango_shap.logging import get_configured_logger
    >>>
    >>> # Get a configured logger
    >>> logger = get_configured_logger('mango_logging')
    >>> logger.info('SHAP analysis started')
    >>>
    >>> # Get a logger with colored output
    >>> colored_logger = get_configured_logger('mango_logging', mango_color=True)
    >>> colored_logger.info('Colored log message')
"""

from .logger import (
    get_configured_logger,
    ColorFormatter,
    JSONFormatter,
    JSONFileHandler,
)

__all__ = [
    "get_configured_logger",
    "ColorFormatter",
    "JSONFormatter",
    "JSONFileHandler",
]
