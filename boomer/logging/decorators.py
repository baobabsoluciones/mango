import time
from functools import wraps
import logging

logger = logging.getLogger("root")


def log_time(func):
    """
    The log_time function is a decorator that wraps the function func.
    It takes the arguments and keyword arguments passed to it, passes them on to func,
    and returns what func returns. However, before doing so it logs a message with log.info()
    describing what function is being called and how long it took to execute.

    :param func: pass the function to be decorated
    :return: the result of the call to func
    :doc-author: baobab soluciones
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        The wrapper function is a decorator that wraps the function func.
        It takes the arguments and keyword arguments passed to it, passes them on to func,
        and returns what func returns. However, before doing so it logs a message with log.info()
        describing what function is being called and how long it took to execute.

        :param *args: pass a non-keyworded, variable-length argument list
        :param **kwargs: catch all keyword arguments that are passed to the function
        :return: The result of the function it wraps
        :doc-author: baobab soluciones
        """
        start = time.time()
        result = func(*args, **kwargs)
        duration = round(time.time() - start, 2)
        logger.info(f"{func.__name__} took {duration} seconds")
        return result

    return wrapper
