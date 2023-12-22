from math import sin, cos

import numpy as np


def dolan_function_no2(x: np.array) -> float:
    """
    Dolan function No 2.

    The global minima is at [8.39045925, 4.81424707, 7.34574133, 68.88246895, 3.85470806] with a value of 10^-5

    :param x: array of floats. Values are evaluated between [-100, 100]
    :type x: :class:`numpy.ndarray`
    :return: the value of the function
    :rtype: float
    """
    return (
        (x[0] + 1.7 * x[1]) * sin(x[0])
        - 1.5 * x[2]
        - 0.1 * x[3] * cos(x[4] + x[3] - x[0])
        + 0.2 * x[4] ** 2
        - x[1]
        - 1
    )


def inverted_dolan_function_no2(x: np.array) -> float:
    """
    Inverted Dolan function No 2.

    :param x: array of floats. Values are evaluated between [-100, 100]
    :type x: :class:`numpy.ndarray`
    :return: the value of the function
    :rtype: float
    """
    return -dolan_function_no2(x)
