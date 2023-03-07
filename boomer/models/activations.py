import math


def sigmoid(x, factor: float = 1):
    x = max(-60, min(60, factor * x))
    return 1 / (1 + math.exp(-x))


def tanh(x, factor: float = 1):
    x = max(-60, min(60, factor * x))
    return math.tanh(x)
