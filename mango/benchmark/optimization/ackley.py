from math import exp, sqrt, cos, pi, e


def ackley(x: list) -> float:
    """
    General Ackley function.

    The Ackley function is widely used for testing optimization algorithms.
    In its two-dimensional form, it is characterized by a nearly flat outer region, and a large hole at the centre.
    The function poses a risk for optimization algorithms, particularly hillclimbing algorithms,
    to be trapped in one of its many local minima.

    The global minimum point of the function is: f(x) = 0, at x = (0, ..., 0)

    :param list x: list of floats. The values are evaluated in the range [-32.768, 32.768]
    :return: the value of the function
    :rtype: float
    """
    return (
        -20 * exp(-0.2 * sqrt(1 / len(x) * sum([i**2 for i in x])))
        - exp(1 / len(x) * sum([cos(2 * pi * i) for i in x]))
        + e
        + 20
    )


def inverted_ackley(x: list) -> float:
    """
    Inverted Ackley function.

    :param list x: list of floats
    :return: the value of the function
    :rtype: float
    """
    return -ackley(x)
