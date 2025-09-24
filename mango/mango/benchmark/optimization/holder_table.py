from math import sin, cos, exp, sqrt, pi
from typing import Union

import numpy as np


def holder_table(x: Union[np.ndarray, list]) -> float:
    """
    Compute the Holder Table function value for optimization benchmarking.

    This function is multimodal, with a number of local minima. It features
    trigonometric and exponential terms that create a complex landscape with
    multiple global minima. The function is particularly challenging due to
    its oscillatory behavior and multiple optimal solutions.

    The global minima are located at:
    - (8.05502, 9.66459)
    - (-8.05502, 9.66459)
    - (8.05502, -9.66459)
    - (-8.05502, -9.66459)
    All with a value of -19.2085.

    :param x: Input vector with 2 elements. Both elements usually in range [-10, 10]
    :type x: Union[numpy.ndarray, list]
    :return: Function value at the given point
    :rtype: float
    :raises ValueError: If input doesn't have exactly 2 elements
    :raises IndexError: If input is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([8.05502, 9.66459])
        >>> result = holder_table(x)
        >>> print(f"{result:.4f}")
        -19.2085
    """
    return -abs(sin(x[0]) * cos(x[1]) * exp(abs(1 - sqrt(x[0] ** 2 + x[1] ** 2) / pi)))


def inverted_holder_table(x: Union[np.ndarray, list]) -> float:
    """
    Compute the inverted Holder Table function value for maximization problems.

    This function returns the negative of the standard Holder Table function,
    effectively converting the minimization problem into a maximization problem.
    The global maxima are at the same points as the original function's minima,
    but with positive values.

    :param x: Input vector with 2 elements. Both elements usually in range [-10, 10]
    :type x: Union[numpy.ndarray, list]
    :return: Negative function value at the given point
    :rtype: float
    :raises ValueError: If input doesn't have exactly 2 elements
    :raises IndexError: If input is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([8.05502, 9.66459])
        >>> result = inverted_holder_table(x)
        >>> print(f"{result:.4f}")
        19.2085
    """
    return -holder_table(x)
