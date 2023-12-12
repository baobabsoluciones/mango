from math import sin, cos


def dolan_function_no2(x: list) -> float:
    """
    Dolan function No 2.

    :param list x: list of floats
    :return: the value of the function
    :rtype: float
    """
    return (
        (x[0] + 1.7 * x[1]) * sin(x[0])
        - 1.5 * x[2]
        - 0.1 * x[3] * cos(x[4] + x[3] - x[0])
        + 0.2 * x[4] ** 2
        - x[1]
        - 1
    )


def inverted_dolan_function_no2(x: list) -> float:
    """
    Inverted Dolan function No 2.

    :param list x: list of floats
    :return: the value of the function
    :rtype: float
    """
    return -dolan_function_no2(x)
