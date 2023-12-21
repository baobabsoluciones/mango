from math import cos, pi


def rastrigin(x: list) -> float:
    """
    Rastrigin function.

    The Rastrigin function has several local minima. It is highly multimodal, but locations of the minima
    are regularly distributed.

    :param list x: list of floats. Values usually are between -5.12 and 5.12.
    :return: the value of the function
    :rtype: float
    """
    return 10 * len(x) + sum([i**2 - 10 * cos(2 * pi * i) for i in x])


def inverse_rastrigin(x: list) -> float:
    """
    Inverted Rastrigin function.

    The Rastrigin function has several local minima. It is highly multimodal, but locations of the minima
    are regularly distributed. This implementation inverts the function to test out the maximization

    :param list x: list of floats. Values usually are between -5.12 and 5.12.
    :return: the value of the function
    :rtype: float
    """
    return -rastrigin(x)
