from math import sin, pi
from typing import Union

import numpy as np


def gramacy_lee(x: Union[np.ndarray, list, float]) -> float:
    """
    Compute the Gramacy & Lee function value for optimization benchmarking.

    This function is multimodal, with a number of local minima. It features
    oscillatory behavior combined with polynomial terms, creating a challenging
    landscape for optimization algorithms. The function is typically used for
    one-dimensional optimization problems.

    The global minimum is at x = 0.54.

    :param x: Input value as numpy array, list, or float. Value usually in range [-0.5, 2.5]
    :type x: Union[numpy.ndarray, list, float]
    :return: Function value at the given point
    :rtype: float
    :raises ValueError: If input is empty or invalid
    :raises IndexError: If input array/list is empty

    Example:
        >>> import numpy as np
        >>> result = gramacy_lee(0.54)
        >>> print(f"{result:.6f}")
        -0.869011
        >>> result = gramacy_lee([0.54])
        >>> print(f"{result:.6f}")
        -0.869011
    """
    if isinstance(x, float):
        x = [x]
    return sin(10 * pi * x[0]) / (2 * x[0]) + (x[0] - 1) ** 4


def inverted_gramacy_lee(x: Union[np.ndarray, list, float]) -> float:
    """
    Compute the inverted Gramacy & Lee function value for maximization problems.

    This function returns the negative of the standard Gramacy & Lee function,
    effectively converting the minimization problem into a maximization problem.
    The global maximum is at the same point as the original function's minimum,
    but with a positive value.

    :param x: Input value as numpy array, list, or float. Value usually in range [-0.5, 2.5]
    :type x: Union[numpy.ndarray, list, float]
    :return: Negative function value at the given point
    :rtype: float
    :raises ValueError: If input is empty or invalid
    :raises IndexError: If input array/list is empty

    Example:
        >>> import numpy as np
        >>> result = inverted_gramacy_lee(0.54)
        >>> print(f"{result:.6f}")
        0.869011
        >>> result = inverted_gramacy_lee([0.54])
        >>> print(f"{result:.6f}")
        0.869011
    """
    return -gramacy_lee(x)
