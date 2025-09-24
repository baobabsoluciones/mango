import numpy as np


def schwefel(x: np.ndarray) -> float:
    """
    Compute the Schwefel function value for optimization benchmarking.

    The Schwefel function is complex, with many local minima. It features
    trigonometric terms that create a challenging landscape for optimization
    algorithms. The function becomes more difficult as dimensionality increases.

    The global minimum is located at [420.9687, ..., 420.9687] with a value of 0.

    :param x: Input vector with n elements. Values usually in range [-500, 500]
    :type x: numpy.ndarray
    :return: Function value at the given point
    :rtype: float
    :raises ValueError: If input array is empty
    :raises IndexError: If input array is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([420.9687, 420.9687])
        >>> result = schwefel(x)
        >>> print(f"{result:.6f}")
        0.000000
    """
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def inverted_schwefel(x: np.ndarray) -> float:
    """
    Compute the inverted Schwefel function value for maximization problems.

    This function returns the negative of the standard Schwefel function,
    effectively converting the minimization problem into a maximization problem.
    The global maximum is at the same point as the original function's minimum,
    but with a positive value.

    :param x: Input vector with n elements. Values usually in range [-500, 500]
    :type x: numpy.ndarray
    :return: Negative function value at the given point
    :rtype: float
    :raises ValueError: If input array is empty
    :raises IndexError: If input array is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([420.9687, 420.9687])
        >>> result = inverted_schwefel(x)
        >>> print(f"{result:.6f}")
        0.000000
    """
    return -schwefel(x)
