import numpy as np


def rosenbrock(x: np.ndarray) -> float:
    """
    Compute the Rosenbrock function value for optimization benchmarking.

    The Rosenbrock function, also referred to as the Valley or Banana function,
    is a popular test problem for gradient-based optimization algorithms.

    The function is unimodal, and the global minimum lies in a narrow, parabolic valley.
    However, even though this valley is easy to find, convergence to the minimum is difficult (Picheny et al., 2012).

    The global minimum is at [1, 1, ..., 1] with a value of 0.

    :param x: Input vector with n elements. Values usually in range [-5, 10]
    :type x: numpy.ndarray
    :return: Function value at the given point
    :rtype: float
    :raises ValueError: If input array is empty
    :raises IndexError: If input array is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([1.0, 1.0, 1.0])
        >>> result = rosenbrock(x)
        >>> print(f"{result:.6f}")
        0.000000
    """

    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0, axis=0)


def inverted_rosenbrock(x: np.ndarray) -> float:
    """
    Compute the inverted Rosenbrock function value for maximization problems.

    This function returns the negative of the standard Rosenbrock function,
    effectively converting the minimization problem into a maximization problem.
    The global maximum is at the same point as the original function's minimum,
    but with a positive value.

    :param x: Input vector with n elements. Values usually in range [-5, 10]
    :type x: numpy.ndarray
    :return: Negative function value at the given point
    :rtype: float
    :raises ValueError: If input array is empty
    :raises IndexError: If input array is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([1.0, 1.0, 1.0])
        >>> result = inverted_rosenbrock(x)
        >>> print(f"{result:.6f}")
        0.000000
    """
    return -rosenbrock(x)
