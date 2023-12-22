from math import cos, sqrt

import numpy as np


def drop_wave(x: np.array) -> float:
    """
    Drop-Wave function.

    The Drop-Wave function has a unique global minimum. It is multimodal, but the minima are regularly distributed.

    The global minima is located at x = (0, 0) with a value of -1

    :param x: array of floats. Values usually are between -5.12 and 5.12.
    :type x: :class:`numpy.ndarray`
    :return: the value of the function
    :rtype: float
    """
    return -(1 + cos(12 * sqrt(np.sum(np.square(x))))) / (
        0.5 * np.sum(np.square(x)) + 2
    )


def inverted_drop_wave(x: np.array) -> float:
    """
    Inverted Drop-Wave function.

    The Drop-Wave function has a unique global minimum. It is multimodal, but the minima are regularly distributed.
    This implementation inverts the function to test out the maximization

    The global minima is located at x = (0, 0) with a value of -1

    :param x: array of floats. Values usually are between -5.12 and 5.12.
    :type x: :class:`numpy.ndarray`
    :return: the value of the function
    :rtype: float
    """
    return -drop_wave(x)
