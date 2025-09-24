from math import pi

import numpy as np


def rastrigin(x: np.ndarray) -> float:
    """
    Compute the Rastrigin function value for optimization benchmarking.

    The Rastrigin function has several local minima. It is highly multimodal, but locations of the minima
    are regularly distributed. This function combines a quadratic term with cosine oscillations,
    creating a complex landscape with numerous local optima.

    The global minimum is at [0, 0, ..., 0] with a value of 0.

    :param x: Input vector with n elements. Values usually in range [-5.12, 5.12]
    :type x: numpy.ndarray
    :return: Function value at the given point
    :rtype: float
    :raises ValueError: If input array is empty
    :raises IndexError: If input array is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([0.0, 0.0, 0.0])
        >>> result = rastrigin(x)
        >>> print(f"{result:.6f}")
        0.000000
    """
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * pi * x))


def inverted_rastrigin(x: np.ndarray) -> float:
    """
    Compute the inverted Rastrigin function value for maximization problems.

    This function returns the negative of the standard Rastrigin function,
    effectively converting the minimization problem into a maximization problem.
    The global maximum is at the same point as the original function's minimum,
    but with a positive value.

    :param x: Input vector with n elements. Values usually in range [-5.12, 5.12]
    :type x: numpy.ndarray
    :return: Negative function value at the given point
    :rtype: float
    :raises ValueError: If input array is empty
    :raises IndexError: If input array is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([0.0, 0.0, 0.0])
        >>> result = inverted_rastrigin(x)
        >>> print(f"{result:.6f}")
        0.000000
    """
    return -rastrigin(x)
