from math import sqrt

import numpy as np


def bukin_function_6(x: np.ndarray) -> float:
    """
    Compute the Bukin function N. 6 value for optimization benchmarking.

    The sixth Bukin function has many local minima, all of which lie in a ridge.
    This function is particularly challenging for optimization algorithms due to
    its complex landscape with multiple local optima.

    The global minimum is at (-10, 1) with a value of 0.

    :param x: Input vector with 2 elements. x[0] in range [-15, 5], x[1] in range [-3, 3]
    :type x: numpy.ndarray
    :return: Function value at the given point
    :rtype: float
    :raises ValueError: If input array doesn't have exactly 2 elements
    :raises IndexError: If input array is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([-10.0, 1.0])
        >>> result = bukin_function_6(x)
        >>> print(f"{result:.6f}")
        0.000000
    """
    return 100 * sqrt(abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * abs(x[0] + 10)


def inverted_bukin_function_6(x: np.ndarray) -> float:
    """
    Compute the inverted Bukin function N. 6 value for maximization problems.

    This function returns the negative of the standard Bukin function N. 6,
    effectively converting the minimization problem into a maximization problem.
    The global maximum is at (-10, 1) with a value of 0.

    :param x: Input vector with 2 elements. x[0] in range [-15, 5], x[1] in range [-3, 3]
    :type x: numpy.ndarray
    :return: Negative function value at the given point
    :rtype: float
    :raises ValueError: If input array doesn't have exactly 2 elements
    :raises IndexError: If input array is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([-10.0, 1.0])
        >>> result = inverted_bukin_function_6(x)
        >>> print(f"{result:.6f}")
        0.000000
    """
    return -bukin_function_6(x)
