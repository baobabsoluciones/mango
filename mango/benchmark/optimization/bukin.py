from math import sqrt


def bukin_function_6(x: list) -> float:
    """
    Bukin function N. 6.

    :param list x: list of floats
    :return: the value of the function
    :rtype: float
    """
    return 100 * sqrt(abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * abs(x[0] + 10)
