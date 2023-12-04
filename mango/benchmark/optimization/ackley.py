from math import exp, sqrt, cos, pi, e


def ackley(x: list) -> float:
    """
    General Ackley function.

    :param list x: list of floats
    :return: the value of the function
    :rtype: float
    """
    return (
        -20 * exp(-0.2 * sqrt(1 / len(x) * sum([i**2 for i in x])))
        - exp(1 / len(x) * sum([cos(2 * pi * i) for i in x]))
        + e
        + 20
    )
