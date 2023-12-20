import numpy as np


def schwefel(x: np.array) -> float:
    """
    Schwefel function

    The Schwefel function is complex, with many local minima.

    The global minima is located at [420,9687, ..., 420,9687] with a value of 0

    :param :class:`np.array` x: input vector. The function is usually evaluated between [500, 500]
    :return: the value of the function
    :rtype: float
    :doc-author: baobab soluciones
    """
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def inverted_schwefel(x: np.array) -> float:
    """
    Inverted Schwefel function

    The Schwefel function is complex, with many local maxima.
    This version is used to test maximization.

    The global minima is located at [420,9687, ..., 420,9687] with a value of 0

    :param :class:`np.array` x: input vector. The function is usually evaluated between [500, 500]
    :return: the value of the function
    :rtype: float
    :doc-author: baobab soluciones
    """
    return -schwefel(x)
