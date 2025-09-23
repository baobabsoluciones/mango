from math import sin, pi
from typing import Union

import numpy as np


def levy(x: np.ndarray) -> float:
    """
    Compute the Levy function value for optimization benchmarking.

    The Levy function is a multimodal optimization test function with complex
    oscillatory behavior. It features trigonometric terms that create numerous
    local minima, making it challenging for optimization algorithms.

    The global minimum is at x = [1, 1, 1, ..., 1] and the function value is 0.

    :param x: Input vector with n elements. Each element usually in range [-10, 10]
    :type x: numpy.ndarray
    :return: Function value at the given point
    :rtype: float
    :raises ValueError: If input array is empty
    :raises IndexError: If input array is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([1.0, 1.0, 1.0])
        >>> result = levy(x)
        >>> print(f"{result:.6f}")
        0.000000
    """
    w = 1 + (x - 1) / 4

    term1 = (sin(pi * w[0])) ** 2
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * (np.sin(pi * w[:-1] + 1)) ** 2))
    term3 = (w[-1] - 1) ** 2 * (1 + (sin(2 * pi * w[-1])) ** 2)

    return term1 + term2 + term3


def inverted_levy(x: np.ndarray) -> float:
    """
    Compute the inverted Levy function value for maximization problems.

    This function returns the negative of the standard Levy function,
    effectively converting the minimization problem into a maximization problem.
    The global maximum is at the same point as the original function's minimum,
    but with a positive value.

    :param x: Input vector with n elements. Each element usually in range [-10, 10]
    :type x: numpy.ndarray
    :return: Negative function value at the given point
    :rtype: float
    :raises ValueError: If input array is empty
    :raises IndexError: If input array is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([1.0, 1.0, 1.0])
        >>> result = inverted_levy(x)
        >>> print(f"{result:.6f}")
        0.000000
    """
    return -levy(x)


def levy_function_no13(x: Union[np.ndarray, list]) -> float:
    """
    Compute the Levy function N. 13 value for optimization benchmarking.

    This is a 2D variant of the Levy function with specific trigonometric terms.
    It features oscillatory behavior that creates multiple local minima, making
    it challenging for optimization algorithms.

    The global minimum is at x = [1, 1] and the function value is 0.

    :param x: Input vector with 2 elements. Both elements usually in range [-10, 10]
    :type x: Union[numpy.ndarray, list]
    :return: Function value at the given point
    :rtype: float
    :raises ValueError: If input doesn't have exactly 2 elements
    :raises IndexError: If input is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([1.0, 1.0])
        >>> result = levy_function_no13(x)
        >>> print(f"{result:.6f}")
        0.000000
    """
    return (
        sin(3 * pi * x[0]) ** 2
        + (x[0] - 1) ** 2 * (1 + sin(3 * pi * x[1]) ** 2)
        + (x[1] - 1) ** 2 * (1 + sin(2 * pi * x[1]) ** 2)
    )


def inverted_levy_no13(x: Union[np.ndarray, list]) -> float:
    """
    Compute the inverted Levy function N. 13 value for maximization problems.

    This function returns the negative of the standard Levy function N. 13,
    effectively converting the minimization problem into a maximization problem.
    The global maximum is at the same point as the original function's minimum,
    but with a positive value.

    :param x: Input vector with 2 elements. Both elements usually in range [-10, 10]
    :type x: Union[numpy.ndarray, list]
    :return: Negative function value at the given point
    :rtype: float
    :raises ValueError: If input doesn't have exactly 2 elements
    :raises IndexError: If input is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([1.0, 1.0])
        >>> result = inverted_levy_no13(x)
        >>> print(f"{result:.6f}")
        0.000000
    """
    return -levy_function_no13(x)
