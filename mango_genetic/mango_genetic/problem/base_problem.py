from abc import abstractmethod, ABC

import numpy as np


class Problem(ABC):
    """
    Abstract base class for defining optimization problems in genetic algorithms.

    This class provides the interface for implementing fitness functions that
    can be used with genetic algorithms. Subclasses must implement the
    calculate_fitness method to define how solutions are evaluated.

    The class can be used as a callable object, making it compatible with
    genetic algorithm evaluators that expect function-like interfaces.
    """

    def __init__(self):
        """
        Initialize the problem instance.

        Override this method in subclasses to load any initial data or
        configuration needed for the problem to calculate fitness values.
        This is where you would typically load datasets, set parameters,
        or perform other initialization tasks specific to your problem.
        """
        pass

    @abstractmethod
    def calculate_fitness(self, x: np.array) -> float:
        """
        Calculate the fitness value for a given solution.

        This abstract method must be implemented by subclasses to define
        how solutions are evaluated. The fitness value represents how well
        a solution performs according to the optimization objective.

        :param x: Solution vector to evaluate
        :type x: numpy.ndarray
        :return: Fitness value of the solution
        :rtype: float

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplemented

    def __call__(self, x):
        """
        Make the problem instance callable like a function.

        Allows the problem instance to be used directly as a function,
        delegating to the calculate_fitness method. This makes the class
        compatible with genetic algorithm evaluators that expect function-like interfaces.

        :param x: Solution vector to evaluate
        :type x: numpy.ndarray
        :return: Fitness value of the solution
        :rtype: float

        Example:
            >>> problem = MyProblem()
            >>> fitness = problem(solution_vector)  # Equivalent to problem.calculate_fitness(solution_vector)
        """
        return self.calculate_fitness(x)
