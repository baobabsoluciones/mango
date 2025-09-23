from math import exp, pi, cos
from typing import Union

import numpy as np


def langermann(x: Union[np.ndarray, list]) -> float:
    """
    Compute the Langermann function value for optimization benchmarking.

    The Langermann function is a multimodal problem. It has a fairly large number of local minima,
    widely separated and regularly distributed. This function combines exponential and cosine terms
    to create a complex landscape with multiple local optima.

    This implementation is only for the two dimension version of the function using the values for a, c and m
    proposed by Molga & Smutnicki (2005).

    :param x: Input vector with 2 elements. Both elements usually in range [0, 10]
    :type x: Union[numpy.ndarray, list]
    :return: Function value at the given point
    :rtype: float
    :raises ValueError: If input doesn't have exactly 2 elements
    :raises IndexError: If input is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([2.0, 3.0])
        >>> result = langermann(x)
        >>> print(f"{result:.6f}")
        1.000000
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


def inverted_langermann(x: Union[np.ndarray, list]) -> float:
    """
    Compute the inverted Langermann function value for maximization problems.

    This function returns the negative of the standard Langermann function,
    effectively converting the minimization problem into a maximization problem.
    The global maximum is at the same point as the original function's minimum,
    but with a positive value.

    :param x: Input vector with 2 elements. Both elements usually in range [0, 10]
    :type x: Union[numpy.ndarray, list]
    :return: Negative function value at the given point
    :rtype: float
    :raises ValueError: If input doesn't have exactly 2 elements
    :raises IndexError: If input is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([2.0, 3.0])
        >>> result = inverted_langermann(x)
        >>> print(f"{result:.6f}")
        -1.000000
    """
    return -langermann(x)
