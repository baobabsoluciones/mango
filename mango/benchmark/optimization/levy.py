from math import sin, pi
from typing import Union

import numpy as np


def levy(x: np.array) -> float:
    """
    Levy function.

    The global minima is at x = [1, 1, 1, ..., 1] and the function value is 0.

    :param x: array of floats. Each value is usually between -10 and 10.
    :type x: :class:`numpy.array`
    :return: the value of the function
    :rtype: float
    """
    w = 1 + (x - 1) / 4

    term1 = (sin(pi * w[0])) ** 2
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * (np.sin(pi * w[:-1] + 1)) ** 2))
    term3 = (w[-1] - 1) ** 2 * (1 + (sin(2 * pi * w[-1])) ** 2)

    return term1 + term2 + term3


def inverted_levy(x: np.array) -> float:
    """
    Inverted Levy function.

    :param x: array of floats. Each value is usually between -10 and 10.
    :type x: :class:`numpy.array`
    :return: the value of the function
    :rtype: float
    """
    return -levy(x)


def levy_function_no13(x: Union[np.array, list]) -> float:
    """
    Levy function N. 13.

    The global minima is at x = [1, 1] and the function value is 0.

    :param x: array or list of floats. Each value is usually between -10 and 10.
    :type x: :class:`numpy.array` or list
    :return: the value of the function
    :rtype: float
    :doc-author: baobab soluciones
    """
    return (
        sin(3 * pi * x[0]) ** 2
        + (x[0] - 1) ** 2 * (1 + sin(3 * pi * x[1]) ** 2)
        + (x[1] - 1) ** 2 * (1 + sin(2 * pi * x[1]) ** 2)
    )


def inverted_levy_no13(x: Union[np.array, list]) -> float:
    """
    Inverted Levy function N. 13.

    The global minima is at x = [1, 1] and the function value is 0.

    :param x: list of floats. Each value is usually between -10 and 10.
    :type x: :class:`numpy.array` or list
    :return: the value of the function
    :rtype: float
    :doc-author: baobab soluciones
    """
    return -levy_function_no13(x)
