import numpy as np


def griewank(x: np.ndarray) -> float:
    """
    Compute the Griewank function value for optimization benchmarking.

    The Griewank function has many widespread local minima, which are regularly
    distributed. This function combines a quadratic term with a product of cosine
    terms, creating a complex landscape with numerous local optima. The function
    becomes more challenging as the dimensionality increases.

    The global minimum is located at [0, 0, ..., 0] with a value of 0.

    :param x: Input vector with n elements. Values usually in range [-600, 600]
    :type x: numpy.ndarray
    :return: Function value at the given point
    :rtype: float
    :raises ValueError: If input array is empty
    :raises IndexError: If input array is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([0.0, 0.0])
        >>> result = griewank(x)
        >>> print(f"{result:.6f}")
        0.000000
    """
    return (
        np.sum(np.square(x)) / 4000
        - np.prod(np.cos(x / np.sqrt(np.arange(1, x.shape[0] + 1))))
        + 1
    )


def inverted_griewank(x: np.ndarray) -> float:
    """
    Compute the inverted Griewank function value for maximization problems.

    This function returns the negative of the standard Griewank function,
    effectively converting the minimization problem into a maximization problem.
    The global maximum is at the same point as the original function's minimum,
    but with a positive value.

    :param x: Input vector with n elements. Values usually in range [-600, 600]
    :type x: numpy.ndarray
    :return: Negative function value at the given point
    :rtype: float
    :raises ValueError: If input array is empty
    :raises IndexError: If input array is empty

    Example:
        >>> import numpy as np
        >>> x = np.array([0.0, 0.0])
        >>> result = inverted_griewank(x)
        >>> print(f"{result:.6f}")
        0.000000
    """
    return -griewank(x)
