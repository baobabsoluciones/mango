from math import sin, exp, sqrt, pi

import numpy as np


def cross_in_tray(x: np.array) -> float:
    """
    Cross-in-tray function.

    The Cross-in-tray function has many widespread local minima, which are regularly distributed

    The function is usually evaluated on the square xi ∈ [-10, 10], for all i = 1, 2.
    The global mínima is located at x = (1.34941, 1.34941), (-1.34941, 1.34941), (1.34941, -1.34941)
    and (-1.34941, -1.34941) with a value of -2.06261

    :param x: array of floats
    :type x: :class:`numpy.ndarray`
    :return: the value of the function
    :rtype: float
    """
    return (
        -0.0001
        * (
            abs(
                sin(x[0]) * sin(x[1]) * exp(abs(100 - sqrt(x[0] ** 2 + x[1] ** 2) / pi))
            )
            + 1
        )
        ** 0.1
    )


def inverted_cross_in_tray(x: np.array) -> float:
    """
    Inverted Cross-in-tray function.

    The Cross-in-tray function has many widespread local minima, which are regularly distributed
    This implementation inverts the function to test out the maximization

    :param x: array of floats
    :type x: :class:`numpy.ndarray`
    :return: the value of the function
    :rtype: float
    """
    return -cross_in_tray(x)
