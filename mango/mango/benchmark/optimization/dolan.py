from math import sin, cos

import numpy as np


def dolan_function_no2(x: np.ndarray) -> float:
    """
    Compute the Dolan function No. 2 value for optimization benchmarking.

    The Dolan function No. 2 is a 5-dimensional optimization test function with
    complex interactions between variables. It features trigonometric and polynomial
    terms that create a challenging landscape for optimization algorithms.

    The global minimum is at [8.39045925, 4.81424707, 7.34574133, 68.88246895, 3.85470806]
    with a value of approximately 10^-5.

    :param x: Input vector with 5 elements. All elements in range [-100, 100]
    :type x: numpy.ndarray
    :return: Function value at the given point
    :rtype: float
    :raises ValueError: If input array doesn't have exactly 5 elements
    :raises IndexError: If input array is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([8.39045925, 4.81424707, 7.34574133, 68.88246895, 3.85470806])
        >>> result = dolan_function_no2(x)
        >>> print(f"{result:.2e}")
        1.00e-05
    """
    return (
        (x[0] + 1.7 * x[1]) * sin(x[0])
        - 1.5 * x[2]
        - 0.1 * x[3] * cos(x[4] + x[3] - x[0])
        + 0.2 * x[4] ** 2
        - x[1]
        - 1
    )


def inverted_dolan_function_no2(x: np.ndarray) -> float:
    """
    Compute the inverted Dolan function No. 2 value for maximization problems.

    This function returns the negative of the standard Dolan function No. 2,
    effectively converting the minimization problem into a maximization problem.
    The global maximum is at the same point as the original function's minimum,
    but with a positive value.

    :param x: Input vector with 5 elements. All elements in range [-100, 100]
    :type x: numpy.ndarray
    :return: Negative function value at the given point
    :rtype: float
    :raises ValueError: If input array doesn't have exactly 5 elements
    :raises IndexError: If input array is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([8.39045925, 4.81424707, 7.34574133, 68.88246895, 3.85470806])
        >>> result = inverted_dolan_function_no2(x)
        >>> print(f"{result:.2e}")
        -1.00e-05
    """
    return -dolan_function_no2(x)
