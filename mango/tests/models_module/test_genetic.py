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
            0.005780531584417341,
            0.002532981327856094,
            -0.0002830464196912103,
            0.0035978701682818507,
            -0.009297998564329687,
            -0.010876453380193416,
            0.006082857233110294,
            0.002257310630493114,
            0.009310106593683124,
            0.03773227021181778,
        ]

        self.assertAlmostEqual(population.best.fitness, 0.06340265929127753)
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
            92.44640776087473,
            98.4839235540964,
            97.18940541471056,
            -75.34867669478973,
            1.565332634923024,
        ]

        self.assertAlmostEqual(population.best.fitness, -505.022072557116)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])
