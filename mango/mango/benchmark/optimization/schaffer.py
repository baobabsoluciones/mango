from math import sin, cos
from typing import Union

import numpy as np


def schaffer_function_no2(x: Union[np.array, list]) -> float:
    """
    Schaffer function N. 2.

    .. math:: f(x) = 0.5 + \\frac{\\sin^2(x_1^2 - x_2^2) - 0.5}{[1 + 0.001(x_1^2 + x_2^2)]^2}

    The global minima is at x [0, 0] with value 0

    :param x: A list of values. Values are evaluated between [-100, 100]
    :type x: :class:`numpy.array` or list
    :return: The value of the Schaffer function N. 2.
    :rtype: float
    """
    return 0.5 + (sin(x[0] ** 2 - x[1] ** 2) ** 2 - 0.5) / (
        (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2
    )


def inverted_schaffer_function_no2(x: Union[np.array, list]) -> float:
    """
    Inverted Schaffer function N. 2.

    This inverted version is used for maximization.

    .. math:: f(x) = -0.5 - \\frac{\\sin^2(x_1^2 - x_2^2) - 0.5}{[1 + 0.001(x_1^2 + x_2^2)]^2}

    The global maxima is at x [0, 0] with value 0

    :param x: A list of values. Values are evaluated between [-100, 100]
    :type x: :class:`numpy.array` or list
    :return: The value of the Inverted Schaffer function N. 2.
    :rtype: float
    """
    return -schaffer_function_no2(x)


def schaffer_function_no4(x: Union[np.array, list]) -> float:
    """
    Schaffer function N. 4.

    .. math:: f(x) = 0.5 + \\frac{\\cos^2(\\sin(|x_1^2 - x_2^2|)) - 0.5}{[1 + 0.001(x_1^2 + x_2^2)]^2}

    The global minima is at x [0, 1.253115] with value 0.292579

    :param x: array or list of values. Values are evaluated between [-100, 100]
    :type x: :class:`numpy.array` or list
    :return: The value of the Schaffer function N. 4.
    :rtype: float
    """
    return 0.5 + (cos(sin(abs(x[0] ** 2 - x[1] ** 2))) ** 2 - 0.5) / (
        (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2
    )


def inverted_schaffer_function_no4(x: Union[np.array, list]) -> float:
    """
    Inverted Schaffer function N. 4.

    This inverted version is used for maximization.

    .. math:: f(x) = -0.5 - \\frac{\\cos^2(\\sin(|x_1^2 - x_2^2|)) - 0.5}{[1 + 0.001(x_1^2 + x_2^2)]^2}

    The global maxima is at x [0, 1.253115] with value -0.292579

    :param x: array or list of values. Values are evaluated between [-100, 100]
    :type x: :class:`numpy.array` or list
    :return: The value of the Inverted Schaffer function N. 4.
    :rtype: float
    """
    return -schaffer_function_no4(x)
