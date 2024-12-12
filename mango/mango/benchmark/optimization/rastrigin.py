from math import pi

import numpy as np


def rastrigin(x: np.array) -> float:
    """
    Rastrigin function.

    The Rastrigin function has several local minima. It is highly multimodal, but locations of the minima
    are regularly distributed.

    The global minima is at [0, 0, ..., 0] with a value of 0

    :param x: array of floats. Values usually are between -5.12 and 5.12.
    :type x: :class:`npumpy.array`
    :return: the value of the function
    :rtype: float
    """
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * pi * x))


def inverted_rastrigin(x: np.array) -> float:
    """
    Inverted Rastrigin function.

    The Rastrigin function has several local minima. It is highly multimodal, but locations of the minima
    are regularly distributed. This implementation inverts the function to test out the maximization

    :param x: array of floats. Values usually are between -5.12 and 5.12.
    :type x: :class:`npumpy.array`
    :return: the value of the function
    :rtype: float
    """
    return -rastrigin(x)
