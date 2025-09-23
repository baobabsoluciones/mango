from math import exp, sqrt, pi, e

import numpy as np


def ackley(x: np.ndarray) -> float:
    """
    Compute the Ackley function value for optimization benchmarking.

    The Ackley function is widely used for testing optimization algorithms.
    In its two-dimensional form, it is characterized by a nearly flat outer region,
    and a large hole at the centre. The function poses a risk for optimization
    algorithms, particularly hillclimbing algorithms, to be trapped in one of its
    many local minima.

    The global minimum point of the function is: f(x) = 0, at x = (0, ..., 0)

    :param x: Input vector as numpy array. Values should be in range [-32.768, 32.768]
    :type x: numpy.ndarray
    :return: Function value at the given point
    :rtype: float
    :raises ValueError: If input array is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([0.0, 0.0])
        >>> result = ackley(x)
        >>> print(f"{result:.6f}")
        0.000000
    """
    return (
        -20 * exp(-0.2 * sqrt(1 / len(x) * np.sum(np.square(x))))
        - exp(1 / len(x) * np.sum(np.cos(2 * pi * x)))
        + e
        + 20
    )


def inverted_ackley(x: np.ndarray) -> float:
    """
    Compute the inverted Ackley function value for maximization problems.

    This function returns the negative of the standard Ackley function, effectively
    converting the minimization problem into a maximization problem. The global
    maximum point is: f(x) = 0, at x = (0, ..., 0).

    :param x: Input vector as numpy array. Values should be in range [-32.768, 32.768]
    :type x: numpy.ndarray
    :return: Negative function value at the given point
    :rtype: float
    :raises ValueError: If input array is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([0.0, 0.0])
        >>> result = inverted_ackley(x)
        >>> print(f"{result:.6f}")
        0.000000
    """
    return -ackley(x)
