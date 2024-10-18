from abc import abstractmethod, ABC

import numpy as np


class Problem(ABC):
    """
    Metaclass to implement an abstract problem class for genetic algorithms.

    The problem class is used to define the fitness function for the genetic algorithm.
    """

    def __init__(self):
        """
        In this method all the logic to load initial data that is needed for the problem to calculate its fitness
        should be implemented
        """
        pass

    @abstractmethod
    def calculate_fitness(self, x: np.array) -> float:
        """
        Calculate the fitness of a given solution.

        This method has to be implemented on the subclasses

        :param x: Solution to calculate the fitness of.
        :type x: :class:`numpy.array`
        :return: Fitness of the solution.
        :rtype: float
        """
        raise NotImplemented

    def __call__(self, x):
        """
        Method to call the instance of the class as a method to calculate to return the fitness value.

        This way the class can be sued as a method.
        """
        return self.calculate_fitness(x)
