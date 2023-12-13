from math import sqrt


def bukin_function_6(x: list) -> float:
    """
    Bukin function N. 6.

    The sixth Bukin function has many local minima, all of which lie in a ridge.

    The global minima is at (-10, 1) with a value of 0.

    :param list x: list of floats. Values are between [-15, 5] for x1 and [-3, 3] for x2
    :return: the value of the function
    :rtype: float
    """
    return 100 * sqrt(abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * abs(x[0] + 10)


def inverted_bukin_function_6(x: list) -> float:
    """
    Inverted Bukin function N. 6.

    :param list x: list of floats
    :return: the value of the function
    :rtype: float
    """
    return -bukin_function_6(x)
