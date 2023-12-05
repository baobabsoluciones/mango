from random import seed
from unittest import TestCase

from mango.benchmark.optimization import ackley
from mango.benchmark.optimization import bukin_function_6
from mango.benchmark.optimization import dolan_function_no2
from mango.models.genetic.config.config import GeneticBaseConfig
from mango.models.genetic.population.base_population import Population
from mango.models.genetic.problem.base_problem import Problem
from mango.tests.const import normalize_path


def dolan(x):
    pass


class TestGeneticAlgorithms(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ackley_2D(self):
        seed(33)
        config = GeneticBaseConfig(normalize_path("./data/test_ackley_2d.cfg"))
        population = Population(config, ackley)
        population.run()

        solution = [-0.3316358743328678, 0.01317055312258475]

        self.assertAlmostEqual(population.best.fitness, 2.347556662983397)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])

    def test_ackley_10D(self):
        seed(42)
        config = GeneticBaseConfig(normalize_path("./data/test_ackley_10d.cfg"))
        population = Population(config, ackley)
        population.run()

        solution = [
            -0.01309971982828273,
            -0.009824236218179294,
            0.21234663465939008,
            -0.11366943556181752,
            -0.026285307242666534,
            0.035387828688868694,
            0.07015370065710158,
            0.24108637455136517,
            0.031190632457850143,
            -0.010530681520364737,
        ]

        self.assertAlmostEqual(population.best.fitness, 0.9591826136127466)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])

    def test_bukin(self):
        seed(23)
        config = GeneticBaseConfig(normalize_path("./data/test_bukin.cfg"))
        population = Population(config, bukin_function_6)
        population.run()

        solution = [-8.830028013745961, 0.7790900403243501]

        self.assertAlmostEqual(population.best.fitness, 2.469151471238935)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])

    def test_custom_problem(self):
        class CustomProblem(Problem):
            def __init__(self):
                super().__init__()

            def calculate_fitness(self, x):
                return dolan_function_no2(x)

        seed(38)
        config = GeneticBaseConfig(normalize_path("./data/test_custom_problem.cfg"))
        problem = CustomProblem()
        population = Population(config, problem)
        population.run()

        solution = [
            99.00684234446081,
            98.4839235540964,
            99.44389866360481,
            -99.77425278761257,
            1.205824914327124,
        ]

        self.assertAlmostEqual(population.best.fitness, -523.8876176268644)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])
