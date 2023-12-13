from math import sin, exp, sqrt, pi


def cross_in_tray(x: list) -> float:
    """
    Cross-in-tray function.

    The Cross-in-tray function has many widespread local minima, which are regularly distributed

    The function is usually evaluated on the square xi ∈ [-10, 10], for all i = 1, 2.
    The global mínima is located at x = (1.34941, 1.34941), (-1.34941, 1.34941), (1.34941, -1.34941)
    and (-1.34941, -1.34941) with a value of -2.06261

    :param list x: list of floats
    :return: the value of the function
    :rtype: float
    """
    return (
        -0.0001
        * (
            abs(
                sin(x[0]) * sin(x[1]) * exp(abs(100 - sqrt(x[0] ** 2 + x[1] ** 2) / pi))
            )
            + 1
        )
        ** 0.1
    )


def inverted_cross_in_tray(x: list) -> float:
    """
    Inverted Cross-in-tray function.

    The Cross-in-tray function has many widespread local minima, which are regularly distributed
    This implementation inverts the function to test out the maximization

    :param list x: list of floats
    :return: the value of the function
    :rtype: float
    """
    return -cross_in_tray(x)
