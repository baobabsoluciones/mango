from math import exp, sqrt, pi, e

import numpy as np


def ackley(x: np.ndarray) -> float:
    """
    General Ackley function.

    The Ackley function is widely used for testing optimization algorithms.

    In its two-dimensional form, it is characterized by a nearly flat outer region, and a large hole at the centre.
    The function poses a risk for optimization algorithms, particularly hillclimbing algorithms,
    to be trapped in one of its many local minima.

    The global minimum point of the function is: f(x) = 0, at x = (0, ..., 0)

    :param x: numpy array of floats. The values are evaluated in the range [-32.768, 32.768]
    :type x: :class:`numpy.ndarray`
    :return: the value of the function
    :rtype: float
    :doc-author: baobab soluciones
    """
    return (
        -20 * exp(-0.2 * sqrt(1 / len(x) * np.sum(np.square(x))))
        - exp(1 / len(x) * np.sum(np.cos(2 * pi * x)))
        + e
        + 20
    )


def inverted_ackley(x: np.ndarray) -> float:
    """
    Inverted Ackley function.

    Some description of the function.

    :param x: numpy array of floats
    :type x: :class:`numpy.ndarray`
    :return: the value of the function
    :rtype: float
    """
    return -ackley(x)
