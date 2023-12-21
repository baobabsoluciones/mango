from math import sin, pi


def gramacy_lee(x: list) -> float:
    """
    Gramacy & Lee function.

    This function is multimodal, with a number of local minima. The global minimum is at x = 0.54.

    :param list x: list of floats. Each value is usually between -0.5 and 2.5.
    :return: the value of the function
    :rtype: float
    """
    return sin(10 * pi * x[0]) / (2 * x[0]) + (x[0] - 1) ** 4


def inverted_gramacy_lee(x: list) -> float:
    """
    Inverted Gramacy & Lee function.

    This function is multimodal, with a number of local minima. The global minimum is at x = 0.54.

    :param list x: list of floats. Each value is usually between -0.5 and 2.5.
    :return: the value of the function
    :rtype: float
    """
    return -gramacy_lee(x)
