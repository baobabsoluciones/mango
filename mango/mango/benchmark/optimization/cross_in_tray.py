from math import sin, exp, sqrt, pi

import numpy as np


def cross_in_tray(x: np.ndarray) -> float:
    """
    Compute the Cross-in-tray function value for optimization benchmarking.

    The Cross-in-tray function has many widespread local minima, which are regularly
    distributed. This function is particularly challenging for optimization algorithms
    due to its complex landscape with multiple global minima.

    The function is usually evaluated on the square xi ∈ [-10, 10], for all i = 1, 2.
    The global minima are located at:
    - x = (1.34941, 1.34941)
    - x = (-1.34941, 1.34941)
    - x = (1.34941, -1.34941)
    - x = (-1.34941, -1.34941)
    All with a value of -2.06261.

    :param x: Input vector with 2 elements. Both elements in range [-10, 10]
    :type x: numpy.ndarray
    :return: Function value at the given point
    :rtype: float
    :raises ValueError: If input array doesn't have exactly 2 elements
    :raises IndexError: If input array is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([1.34941, 1.34941])
        >>> result = cross_in_tray(x)
        >>> print(f"{result:.6f}")
        -2.062612
    """
    return (
        -0.0001
        * (
            abs(
                sin(x[0]) * sin(x[1]) * exp(abs(100 - sqrt(x[0] ** 2 + x[1] ** 2) / pi))
            )
            + 1
        )
        ** 0.1
    )


def inverted_cross_in_tray(x: np.ndarray) -> float:
    """
    Compute the inverted Cross-in-tray function value for maximization problems.

    This function returns the negative of the standard Cross-in-tray function,
    effectively converting the minimization problem into a maximization problem.
    The global maxima are located at the same points as the original function's
    minima, but with positive values.

    :param x: Input vector with 2 elements. Both elements in range [-10, 10]
    :type x: numpy.ndarray
    :return: Negative function value at the given point
    :rtype: float
    :raises ValueError: If input array doesn't have exactly 2 elements
    :raises IndexError: If input array is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([1.34941, 1.34941])
        >>> result = inverted_cross_in_tray(x)
        >>> print(f"{result:.6f}")
        2.062612
    """
    return -cross_in_tray(x)
