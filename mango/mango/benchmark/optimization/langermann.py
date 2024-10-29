from math import exp, pi, cos
from typing import Union

import numpy as np


def langermann(x: Union[np.array, list]) -> float:
    """
    Langermann function.

    The Langermann function is a multimodal problem. It has a fairly large number of local minima,
    widely separated and regularly distributed.

    This implementation is only for the two dimension version of the function using the values for a, c and m
    proposed by Molga & Smutnicki (2005) :cite:p:`molga2005test`.

    :param x: array or list of floats. Each value is usually between 0 and 10.
    :type x: :class:`numpy.array` or list
    :return: the value of the function
    :rtype: float
    """
    a = [
        [3, 5, 2, 1, 7],
        [5, 2, 1, 4, 9],
    ]
    c = [1, 2, 5, 2, 3]
    m = 5

    return sum(
        [
            c[i]
            * exp(-1 / pi * sum([(x[j] - a[j][i]) ** 2 for j in range(len(x))]))
            * cos(pi * sum([(x[j] - a[j][i] ** 2) for j in range(len(x))]))
            for i in range(m)
        ]
    )


def inverted_langermann(x: Union[np.array, list]) -> float:
    """
    Inverted Langermann function.

    The Langermann function is a multimodal problem. It has a fairly large number of local minima,
    widely separated and regularly distributed. This implementation inverts the function to test out the maximization

    :param x: array or list of floats. Each value is usually between 0 and 10.
    :type x: :class:`numpy.array` or list
    :return: the value of the function
    :rtype: float
    """
    return -langermann(x)
