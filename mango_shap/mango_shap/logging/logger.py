"""Logging configuration for mango_shap."""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for the given name.

    :param name: Name of the logger
    :param level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :return: Logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    # Set level if provided
    if level:
        logger.setLevel(getattr(logging, level.upper()))

    return logger
