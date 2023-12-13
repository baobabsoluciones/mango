from math import sin, cos, exp, sqrt, pi


def holder_table(x: list) -> float:
    """
    Holder function.

    This function is multimodal, with a number of local minima.
    The global minima is at (8.05502, 9.66459), (-8.05502, 9.66459), (8.05502, -9.66459) and (-8.05502, -9.66459)
    with a value of -19.2085.

    :param list x: list of floats. Each value is usually between -10 and 10.
    :return: the value of the function
    :rtype: float
    """
    return -abs(sin(x[0]) * cos(x[1]) * exp(abs(1 - sqrt(x[0] ** 2 + x[1] ** 2) / pi)))


def inverted_holder_table(x: list) -> float:
    """
    Inverted Holder function.

    This function is multimodal, with a number of local minima.
    This inverted version is used for testin maximization

    The global maxima is at (8.05502, 9.66459), (-8.05502, 9.66459), (8.05502, -9.66459) and (-8.05502, -9.66459)
    with a value of 19.2085.

    :param list x: list of floats. Each value is usually between -10 and 10.
    :return: the value of the function
    :rtype: float
    """
    return -holder_table(x)
