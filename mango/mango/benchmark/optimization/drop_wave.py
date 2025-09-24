from math import cos, sqrt

import numpy as np


def drop_wave(x: np.ndarray) -> float:
    """
    Compute the Drop-Wave function value for optimization benchmarking.

    The Drop-Wave function has a unique global minimum. It is multimodal, but the
    minima are regularly distributed. This function features a distinctive
    "drop" shape with oscillatory behavior that creates multiple local minima.

    The global minimum is located at x = (0, 0) with a value of -1.

    :param x: Input vector with 2 elements. Values usually in range [-5.12, 5.12]
    :type x: numpy.ndarray
    :return: Function value at the given point
    :rtype: float
    :raises ValueError: If input array doesn't have exactly 2 elements
    :raises IndexError: If input array is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([0.0, 0.0])
        >>> result = drop_wave(x)
        >>> print(f"{result:.6f}")
        -1.000000
    """
    return -(1 + cos(12 * sqrt(np.sum(np.square(x))))) / (
        0.5 * np.sum(np.square(x)) + 2
    )


def inverted_drop_wave(x: np.ndarray) -> float:
    """
    Compute the inverted Drop-Wave function value for maximization problems.

    This function returns the negative of the standard Drop-Wave function,
    effectively converting the minimization problem into a maximization problem.
    The global maximum is at the same point as the original function's minimum,
    but with a positive value.

    :param x: Input vector with 2 elements. Values usually in range [-5.12, 5.12]
    :type x: numpy.ndarray
    :return: Negative function value at the given point
    :rtype: float
    :raises ValueError: If input array doesn't have exactly 2 elements
    :raises IndexError: If input array is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([0.0, 0.0])
        >>> result = inverted_drop_wave(x)
        >>> print(f"{result:.6f}")
        1.000000
    """
    return -drop_wave(x)
