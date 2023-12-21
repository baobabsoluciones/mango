def rosenbrock(x: list) -> float:
    """
    Rosenbrock function.

    The Rosenbrock function, also referred to as the Valley or Banana function,
    is a popular test problem for gradient-based optimization algorithms.

    The function is unimodal, and the global minimum lies in a narrow, parabolic valley.
    However, even though this valley is easy to find, convergence to the minimum is difficult (Picheny et al., 2012).

    :param list x: list of floats. Values ussually are between -5 and 10
    :return: the value of the function
    :rtype: float
    """
    return sum(
        [100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1)]
    )


def inverted_rosenbrock(x: list) -> float:
    """
    Inverted Rosenbrock function.

    The Rosenbrock function, also referred to as the Valley or Banana function,
    is a popular test problem for gradient-based optimization algorithms.

    The function is unimodal, and the global minimum lies in a narrow, parabolic valley.
    However, even though this valley is easy to find, convergence to the minimum is difficult (Picheny et al., 2012).

    :param list x: list of floats. Values ussually are between -5 and 10
    :return: the value of the function
    :rtype: float
    """
    return -rosenbrock(x)
