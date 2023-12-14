from abc import abstractmethod

import numpy as np


class Problem:
    def __init__(self):
        pass

    @abstractmethod
    def calculate_fitness(self, x: np.array) -> float:
        raise NotImplemented

    def __call__(self, x):
        return self.calculate_fitness(x)
