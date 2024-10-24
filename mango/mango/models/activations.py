import math


# noinspection PyTypeChecker
def sigmoid(x: float, factor: float = 1) -> float:
    """
    Sigmoid funtion with a factor to control the slope

    :param x: the input value
    :type x: float
    :param factor: the factor to control the slope, defaults to 1
    :type factor: float, optional
    :return: the sigmoid value
    :rtype: float
    """
    x = max(-60, min(60, factor * x))
    return 1 / (1 + math.exp(-x))


# noinspection PyTypeChecker
def tanh(x: float, factor: float = 1) -> float:
    """
    Tanh funtion with a factor to control the slope

    :param x: the input value
    :type x: float
    :param factor: the factor to control the slope, defaults to 1
    :type factor: float, optional
    :return: the tanh value
    """
    x = max(-60, min(60, factor * x))
    return math.tanh(x)
