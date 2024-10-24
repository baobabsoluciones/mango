from math import sin, sqrt

import numpy as np


def egg_holder(x: np.array) -> float:
    """
    Egg-holder function.

    The Egg-holder function has many widespread local minima, which are regularly distributed

    The function is usually evaluated on the square xi ∈ [-512, 512], for all i = 1, 2.
    The global mínima is located at x = (512, 404.2319) with a value of -959.6407

    :param x: array of floats
    :type x: :class:`numpy.ndarray`
    :return: the value of the function
    :rtype: float
    """
    return -(x[1] + 47) * sin(sqrt(abs(x[1] + x[0] / 2 + 47))) - x[0] * sin(
        sqrt(abs(x[0] - (x[1] + 47)))
    )


def inverted_egg_holder(x: np.array) -> float:
    """
    Inverted Egg-holder function.

    The Egg-holder function has many widespread local minima, which are regularly distributed
    This implementation inverts the function to test out the maximization

    :param x: array of floats
    :type x: :class:`numpy.ndarray`
    :return: the value of the function
    :rtype: float
    """
    return -egg_holder(x)
