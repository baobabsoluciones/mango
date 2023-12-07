from math import prod, cos, sqrt


def griewank(x: list) -> float:
    """
    Griewank function.

    The Griewank function has many widespread local minima, which are regularly distributed

    :param list x: list of floats
    :return: the value of the function
    :rtype: float
    """
    return (
        sum([i**2 / 4000 for i in x])
        - prod([cos(i / sqrt(idx + 1)) for idx, i in enumerate(x)])
        + 1
    )


def inverted_griewank(x: list) -> float:
    """
    Inverted Griewank function.

    The Griewank function has many widespread local minima, which are regularly distributed
    This implementation inverts the function to test out the maximization

    :param list x: list of floats
    :return: the value of the function
    :rtype: float
    """
    return -griewank(x)
