from random import seed
from unittest import TestCase

from mango.benchmark.optimization import ackley, inverted_griewank
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

        solution = [-0.02712333924382193, -0.008804580617919555]

        self.assertAlmostEqual(population.best.fitness, 0.10217623013362598)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])

    def test_ackley_10D(self):
        seed(42)
        config = GeneticBaseConfig(normalize_path("./data/test_ackley_10d.cfg"))
        population = Population(config, ackley)
        population.run()

        solution = [
            -1.8523441888890307e-15,
            -9.806025732064819e-16,
            9.651034817088682e-16,
            2.197019125529338e-16,
            -9.183199962483439e-16,
            7.860456923012733e-16,
            -3.7203427977085037e-16,
            8.933583663345577e-16,
            -3.335263926046968e-17,
            9.242464293719362e-16,
        ]

        self.assertAlmostEqual(population.best.fitness, 3.552713678800501e-15)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])

    def test_bukin(self):
        seed(23)
        config = GeneticBaseConfig(normalize_path("./data/test_bukin.cfg"))
        population = Population(config, bukin_function_6)
        population.run()

        solution = [-5.790515976826087, 0.3353007527787817]

        self.assertAlmostEqual(population.best.fitness, 0.042094840231739136)
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
            -91.17322495278599,
            69.6966249577543,
            67.99413923036394,
            71.17276815342376,
            -5.180918017635875,
        ]

        self.assertAlmostEqual(population.best.fitness, -172.58125836935054)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])

    def test_inverted_griewank(self):
        seed(33)
        config = GeneticBaseConfig(
            normalize_path("./data/test_inverted_griewank_10d.cfg")
        )
        population = Population(config, inverted_griewank)
        population.run()

        solution = [
            -0.011761282771751104,
            4.444224598525348,
            -5.431886103415204,
            0.0906272308977834,
            7.022521500918264,
            -7.673298084983652,
            -8.071904722069078,
            -0.0920485550041308,
            0.020649306128330058,
            -9.885140546843235,
        ]

        self.assertAlmostEqual(population.best.fitness, -0.08601455153606641)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])
