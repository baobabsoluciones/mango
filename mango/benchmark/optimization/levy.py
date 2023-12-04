from math import sin, pi


def levy(x: list) -> float:
    """
    Levy function.

    :param list x: list of floats. Each value is usually between -10 and 10.
    :return: the value of the function
    :rtype: float
    """
    w = [1 + (i - 1) / 4 for i in x]

    term1 = (sin(pi * w[0])) ** 2
    term3 = (w[-1] - 1) ** 2 * (1 + (sin(2 * pi * w[-1])) ** 2)

    termi = [(wi - 1) ** 2 * (1 + 10 * (sin(pi * wi + 1)) ** 2) for wi in w[:-1]]

    term2 = sum(termi)

    return term1 + term2 + term3
