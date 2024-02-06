import numpy as np


def rosenbrock(x: np.array) -> float:
    """
    Rosenbrock function.

    The Rosenbrock function, also referred to as the Valley or Banana function,
    is a popular test problem for gradient-based optimization algorithms.

    The function is unimodal, and the global minimum lies in a narrow, parabolic valley.
    However, even though this valley is easy to find, convergence to the minimum is difficult (Picheny et al., 2012).

    The global minima is at [1, 1, ..., 1] with a value of 0

    :param x: array of floats. Values usually are between -5 and 10
    :type x: :class:`numpy.array`
    :return: the value of the function
    :rtype: float
    """

    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0, axis=0)


def inverted_rosenbrock(x: np.array) -> float:
    """
    Inverted Rosenbrock function.

    The Rosenbrock function, also referred to as the Valley or Banana function,
    is a popular test problem for gradient-based optimization algorithms.

    The function is unimodal, and the global minimum lies in a narrow, parabolic valley.
    However, even though this valley is easy to find, convergence to the minimum is difficult (Picheny et al., 2012).

    :param x: array of floats. Values ussually are between -5 and 10
    :type x: :class:`numpy.array`
    :return: the value of the function
    :rtype: float
    """
    return -rosenbrock(x)
