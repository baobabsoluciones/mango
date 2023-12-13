from math import cos, sqrt


def drop_wave(x: list) -> float:
    """
    Drop-Wave function.

    The Drop-Wave function has a unique global minimum. It is multimodal, but the minima are regularly distributed.

    The global minima is located at x = (0, 0) with a value of -1

    :param list x: list of floats. Values usually are between -5.12 and 5.12.
    :return: the value of the function
    :rtype: float
    """
    return -(1 + cos(12 * sqrt(sum([i**2 for i in x])))) / (
        0.5 * sum([i**2 for i in x]) + 2
    )


def inverted_drop_wave(x: list) -> float:
    """
    Inverted Drop-Wave function.

    The Drop-Wave function has a unique global minimum. It is multimodal, but the minima are regularly distributed.
    This implementation inverts the function to test out the maximization

    The global minima is located at x = (0, 0) with a value of -1

    :param list x: list of floats. Values usually are between -5.12 and 5.12.
    :return: the value of the function
    :rtype: float
    """
    return -drop_wave(x)
