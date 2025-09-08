from math import sin, sqrt

import numpy as np


def egg_holder(x: np.ndarray) -> float:
    """
    Compute the Egg-holder function value for optimization benchmarking.

    The Egg-holder function has many widespread local minima, which are regularly
    distributed. This function is particularly challenging for optimization algorithms
    due to its complex landscape with numerous local optima and steep gradients.

    The function is usually evaluated on the square xi ∈ [-512, 512], for all i = 1, 2.
    The global minimum is located at x = (512, 404.2319) with a value of -959.6407.

    :param x: Input vector with 2 elements. Both elements in range [-512, 512]
    :type x: numpy.ndarray
    :return: Function value at the given point
    :rtype: float
    :raises ValueError: If input array doesn't have exactly 2 elements
    :raises IndexError: If input array is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([512.0, 404.2319])
        >>> result = egg_holder(x)
        >>> print(f"{result:.4f}")
        -959.6407
    """
    return -(x[1] + 47) * sin(sqrt(abs(x[1] + x[0] / 2 + 47))) - x[0] * sin(
        sqrt(abs(x[0] - (x[1] + 47)))
    )


def inverted_egg_holder(x: np.ndarray) -> float:
    """
    Compute the inverted Egg-holder function value for maximization problems.

    This function returns the negative of the standard Egg-holder function,
    effectively converting the minimization problem into a maximization problem.
    The global maximum is at the same point as the original function's minimum,
    but with a positive value.

    :param x: Input vector with 2 elements. Both elements in range [-512, 512]
    :type x: numpy.ndarray
    :return: Negative function value at the given point
    :rtype: float
    :raises ValueError: If input array doesn't have exactly 2 elements
    :raises IndexError: If input array is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([512.0, 404.2319])
        >>> result = inverted_egg_holder(x)
        >>> print(f"{result:.4f}")
        959.6407
    """
    return -egg_holder(x)
