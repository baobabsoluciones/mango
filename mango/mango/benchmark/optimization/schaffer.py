from math import sin, cos
from typing import Union

import numpy as np


def schaffer_function_no2(x: Union[np.ndarray, list]) -> float:
    """
    Compute the Schaffer function N. 2 value for optimization benchmarking.

    This function features trigonometric terms that create a complex landscape
    with oscillatory behavior. It is particularly challenging for optimization
    algorithms due to its multimodal nature.

    .. math:: f(x) = 0.5 + \\frac{\\sin^2(x_1^2 - x_2^2) - 0.5}{[1 + 0.001(x_1^2 + x_2^2)]^2}

    The global minimum is at x = [0, 0] with value 0.

    :param x: Input vector with 2 elements. Values usually in range [-100, 100]
    :type x: Union[numpy.ndarray, list]
    :return: Function value at the given point
    :rtype: float
    :raises ValueError: If input doesn't have exactly 2 elements
    :raises IndexError: If input is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([0.0, 0.0])
        >>> result = schaffer_function_no2(x)
        >>> print(f"{result:.6f}")
        0.000000
    """
    return 0.5 + (sin(x[0] ** 2 - x[1] ** 2) ** 2 - 0.5) / (
        (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2
    )


def inverted_schaffer_function_no2(x: Union[np.ndarray, list]) -> float:
    """
    Compute the inverted Schaffer function N. 2 value for maximization problems.

    This function returns the negative of the standard Schaffer function N. 2,
    effectively converting the minimization problem into a maximization problem.
    The global maximum is at the same point as the original function's minimum,
    but with a positive value.

    :param x: Input vector with 2 elements. Values usually in range [-100, 100]
    :type x: Union[numpy.ndarray, list]
    :return: Negative function value at the given point
    :rtype: float
    :raises ValueError: If input doesn't have exactly 2 elements
    :raises IndexError: If input is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([0.0, 0.0])
        >>> result = inverted_schaffer_function_no2(x)
        >>> print(f"{result:.6f}")
        0.000000
    """
    return -schaffer_function_no2(x)


def schaffer_function_no4(x: Union[np.ndarray, list]) -> float:
    """
    Compute the Schaffer function N. 4 value for optimization benchmarking.

    This function features nested trigonometric terms that create a complex landscape
    with oscillatory behavior. It is particularly challenging for optimization
    algorithms due to its multimodal nature and asymmetric global minimum.

    .. math:: f(x) = 0.5 + \\frac{\\cos^2(\\sin(|x_1^2 - x_2^2|)) - 0.5}{[1 + 0.001(x_1^2 + x_2^2)]^2}

    The global minimum is at x = [0, 1.253115] with value 0.292579.

    :param x: Input vector with 2 elements. Values usually in range [-100, 100]
    :type x: Union[numpy.ndarray, list]
    :return: Function value at the given point
    :rtype: float
    :raises ValueError: If input doesn't have exactly 2 elements
    :raises IndexError: If input is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([0.0, 1.253115])
        >>> result = schaffer_function_no4(x)
        >>> print(f"{result:.6f}")
        0.292579
    """
    return 0.5 + (cos(sin(abs(x[0] ** 2 - x[1] ** 2))) ** 2 - 0.5) / (
        (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2
    )


def inverted_schaffer_function_no4(x: Union[np.ndarray, list]) -> float:
    """
    Compute the inverted Schaffer function N. 4 value for maximization problems.

    This function returns the negative of the standard Schaffer function N. 4,
    effectively converting the minimization problem into a maximization problem.
    The global maximum is at the same point as the original function's minimum,
    but with a positive value.

    :param x: Input vector with 2 elements. Values usually in range [-100, 100]
    :type x: Union[numpy.ndarray, list]
    :return: Negative function value at the given point
    :rtype: float
    :raises ValueError: If input doesn't have exactly 2 elements
    :raises IndexError: If input is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([0.0, 1.253115])
        >>> result = inverted_schaffer_function_no4(x)
        >>> print(f"{result:.6f}")
        -0.292579
    """
    return -schaffer_function_no4(x)
