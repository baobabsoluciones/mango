from math import prod, cos, sqrt

import numpy as np


def griewank(x: np.array) -> float:
    """
    Griewank function.

    The Griewank function has many widespread local minima, which are regularly distributed

    :param :class:`np.array` x: array of floats
    :return: the value of the function
    :rtype: float
    """
    return (
        np.sum(np.square(x)) / 4000
        - np.prod(np.cos(x / np.sqrt(np.arange(1, x.shape[0] + 1))))
        + 1
    )


def inverted_griewank(x: np.array) -> float:
    """
    Inverted Griewank function.

    The Griewank function has many widespread local minima, which are regularly distributed
    This implementation inverts the function to test out the maximization

    :param :class:`np.array` x: array of floats
    :return: the value of the function
    :rtype: float
    """
    return -griewank(x)
