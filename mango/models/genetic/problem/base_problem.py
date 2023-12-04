from abc import abstractmethod


class Problem:
    def __init__(self):
        pass

    @abstractmethod
    def calculate_fitness(self, x):
        raise NotImplemented

    def __call__(self, x):
        return self.calculate_fitness(x)
