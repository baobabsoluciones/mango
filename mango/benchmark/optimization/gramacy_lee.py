from math import sin, pi
from typing import Union

import numpy as np


def gramacy_lee(x: Union[np.array, list, float]) -> float:
    """
    Gramacy & Lee function.

    This function is multimodal, with a number of local minima. The global minimum is at x = 0.54.

    :param x: array or list with the value or the value itself. Each value is usually between -0.5 and 2.5.
    :type x: :class:`numpy.array`, list or float
    :return: the value of the function
    :rtype: float
    """
    if isinstance(x, float):
        x = [x]
    return sin(10 * pi * x[0]) / (2 * x[0]) + (x[0] - 1) ** 4


def inverted_gramacy_lee(x: Union[np.array, list, float]) -> float:
    """
    Inverted Gramacy & Lee function.

    This function is multimodal, with a number of local minima. The global minimum is at x = 0.54.

    :param x: array or list with the value or the value itself. Each value is usually between -0.5 and 2.5.
    :type x: :class:`numpy.array`, list or float
    :return: the value of the function
    :rtype: float
    """
    return -gramacy_lee(x)
