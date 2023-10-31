from math import sin, cos


def dolan_function(x):
    return (
        (x[0] + 1.7 * x[1]) * sin(x[0])
        - 1.5 * x[2]
        - 0.1 * x[3] * cos(x[4] + x[3] - x[0])
        + 0.2 * x[4] ** 2
        - x[1]
        - 1
    )
