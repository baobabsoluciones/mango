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

        self.assertAlmostEqual(population.best.fitness, 0.009386959004661577)
        self.assertAlmostEqual(population.best.genes[0], -0.000877432250149468)
        self.assertAlmostEqual(population.best.genes[1], -0.0030993080157841746)

    def test_ackley_10D(self):
        seed(42)
        config = GeneticBaseConfig(normalize_path("./data/test_ackley_10d.cfg"))
        population = Population(config, ackley)
        population.run()

        solution = [
            -0.004052163273080112,
            -0.02201369439172396,
            -0.007411533641743517,
            0.007270657567723049,
            -0.005729073398597961,
            0.001081986342576613,
            0.014568428240366416,
            0.006567255899831537,
            -0.009621024914956422,
            -0.005717096764783491,
        ]

        self.assertAlmostEqual(population.best.fitness, 0.04590547604851736)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])

    def test_bukin(self):
        seed(23)
        config = GeneticBaseConfig(normalize_path("./data/test_bukin.cfg"))
        population = Population(config, bukin_function_6)
        population.run()

        solution = [-4.334084163623883, 0.18786703978839192]

        self.assertAlmostEqual(population.best.fitness, 0.5484356791387585)
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
            98.97711814439049,
            99.912223441447,
            99.97991108646607,
            -99.73306449577049,
            0.7340978901546151,
        ]

        self.assertAlmostEqual(population.best.fitness, -529.5214065879054)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])
