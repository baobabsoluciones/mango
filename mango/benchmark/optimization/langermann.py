from math import exp, pi, cos


def langermann(x: list) -> float:
    """
    Langermann function.

    The Langermann function is a multimodal problem. It has a fairly large number of local minima,
    widely separated and regularly distributed.

    :param list x: list of floats. Each value is usually between 0 and 10.
    :return: the value of the function
    :rtype: float
    """
    a = [
        [3, 5, 2, 1, 7],
        [5, 2, 1, 4, 9],
    ]
    c = [1, 2, 5, 2, 3]
    m = 5

    return sum(
        [
            c[i]
            * exp(-1 / pi * sum([(x[j] - a[j][i]) ** 2 for j in range(len(x))]))
            * cos(pi * sum([(x[j] - a[j][i] ** 2) for j in range(len(x))]))
            for i in range(m)
        ]
    )


def inverted_langermann(x: list) -> float:
    """
    Inverted Langermann function.

    The Langermann function is a multimodal problem. It has a fairly large number of local minima,
    widely separated and regularly distributed. This implementation inverts the function to test out the maximization

    :param list x: list of floats. Each value is usually between 0 and 10.
    :return: the value of the function
    :rtype: float
    """
    return -langermann(x)
