import logging
import time
from functools import wraps
from typing import Union


def log_time(logger: str = "root", level: Union[int, str] = "INFO"):
    """
    Decorator to automatically log function execution time.

    Creates a decorator that measures and logs the execution time of the
    decorated function. The timing information is logged using the specified
    logger and log level.

    :param logger: Name of the logger to use for timing messages
    :type logger: str
    :param level: Logging level for timing messages (default: "INFO")
    :type level: Union[int, str]
    :return: Decorator function that logs execution time
    :rtype: callable

    Example:
        >>> from mango.logging.decorators import log_time
        >>>
        >>> @log_time("my_logger", "DEBUG")
        >>> def slow_function():
        ...     time.sleep(1)
        ...     return "done"
        >>>
        >>> result = slow_function()  # Logs: "slow_function took 1.00 seconds"
    """
    # Convert string level to integer if needed
    if isinstance(level, str):

        level = getattr(logging, level.upper(), logging.INFO)

    logger_obj = logging.getLogger(logger)

    def log_decorator(func):
        """
        Create a decorator that logs function execution time.

        Wraps the provided function to measure and log its execution time
        using the configured logger and log level.

        :param func: Function to be decorated with timing logging
        :type func: callable
        :return: Wrapped function with timing logging
        :rtype: callable
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Execute the wrapped function and log its execution time.

            Measures the execution time of the wrapped function and logs
            the duration using the configured logger and log level.

            :param args: Positional arguments passed to the wrapped function
            :param kwargs: Keyword arguments passed to the wrapped function
            :return: Result returned by the wrapped function
            :rtype: Any
            """
            start = time.time()
            result = func(*args, **kwargs)
            duration = round(time.time() - start, 2)

            # Log with the specified level
            logger_obj.log(level, f"{func.__name__} took {duration} seconds")

            return result

        return wrapper

    return log_decorator
