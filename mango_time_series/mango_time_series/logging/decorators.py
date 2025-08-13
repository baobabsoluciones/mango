import logging
import time
from functools import wraps
from typing import Union


def log_time(logger: str = "root", level: Union[int, str] = logging.INFO):
    """
    The log_time function is a decorator that can receive a logger name and log level as parameters.

    :param str logger: the name of the logger to use (default: "root")
    :param level: logging level to use (default: logging.INFO)
                  Can be an integer (logging.INFO) or a string ("INFO", "DEBUG", etc.)
    :return: the log_decorator function
    """
    # Convert string level to integer if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger_obj = logging.getLogger(logger)

    def log_decorator(func):
        """
        The log_decorator function is a decorator that wraps the function func.
        It logs the function execution time with the specified log level.

        :param func: pass the function to be decorated
        :return: the result of the call to func
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            The wrapper function executes the function and logs its execution time.

            :param *args: pass a non-keyworded, variable-length argument list
            :param **kwargs: catch all keyword arguments that are passed to the function
            :return: The result of the function it wraps
            """
            start = time.time()
            result = func(*args, **kwargs)
            duration = round(time.time() - start, 2)

            # Log with the specified level
            logger_obj.log(level, f"{func.__name__} took {duration} seconds")

            return result

        return wrapper

    return log_decorator
