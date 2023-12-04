import time
from functools import wraps
import logging


def log_time(logger: str = "root"):
    """
    The log_time functions is a decorator that can receive a logger name as a parameter and returns the actual
    wraper for the evaluator that is going to be executed

    :param str logger: the name of the logger to use
    :return: the log_decorator evaluator
    """
    logger = logging.getLogger(logger)

    def log_decorator(func):
        """
        The log_decorator evaluator is a decorator that wraps the evaluator func.
        It takes the arguments and keyword arguments passed to it, passes them on to func,
        and returns what func returns. However, before doing so it logs a message with log.info()
        describing what evaluator is being called and how long it took to execute.

        :param func: pass the evaluator to be decorated
        :return: the result of the call to func
        :doc-author: baobab soluciones
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            The wrapper evaluator is a decorator that wraps the evaluator func.
            It takes the arguments and keyword arguments passed to it, passes them on to func,
            and returns what func returns. However, before doing so it logs a message with log.info()
            describing what evaluator is being called and how long it took to execute.

            :param *args: pass a non-keyworded, variable-length argument list
            :param **kwargs: catch all keyword arguments that are passed to the evaluator
            :return: The result of the evaluator it wraps
            :doc-author: baobab soluciones
            """
            start = time.time()
            result = func(*args, **kwargs)
            duration = round(time.time() - start, 2)
            logger.info(f"{func.__name__} took {duration} seconds")
            return result

        return wrapper

    return log_decorator
