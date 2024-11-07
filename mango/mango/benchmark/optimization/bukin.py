from math import sqrt

import numpy as np


def bukin_function_6(x: np.array) -> float:
    """
    Bukin function N. 6.

    The sixth Bukin function has many local minima, all of which lie in a ridge.

    The global minima is at (-10, 1) with a value of 0.

    :param x: array of floats. Values are between [-15, 5] for x1 and [-3, 3] for x2
    :type x: :class:`numpy.ndarray`
    :return: the value of the function
    :rtype: float
    """
    return 100 * sqrt(abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * abs(x[0] + 10)


def inverted_bukin_function_6(x: np.array) -> float:
    """
    Inverted Bukin function N. 6.

    :param x: array of floats. Values are between [-15, 5] for x1 and [-3, 3] for x2
    :type x: :class:`numpy.ndarray`
    :return: the value of the function
    :rtype: float
    """
    return -bukin_function_6(x)
