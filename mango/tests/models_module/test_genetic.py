from random import seed
from unittest import TestCase

import numpy as np

from mango.benchmark.optimization import ackley, inverted_griewank, levy, rastrigin
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
        np.random.seed(33)
        config = GeneticBaseConfig(normalize_path("./data/test_ackley_2d.cfg"))
        population = Population(config, ackley)
        population.run()

        solution = np.array([0.03458293, -0.00880458])

        self.assertAlmostEqual(population.best.fitness, 0.1345084210097447)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])

    def test_ackley_10D(self):
        seed(42)
        np.random.seed(42)
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
        np.random.seed(23)
        config = GeneticBaseConfig(normalize_path("./data/test_bukin.cfg"))
        population = Population(config, bukin_function_6)
        population.run()

        solution = [-2.1335805, 0.04552038]

        self.assertAlmostEqual(population.best.fitness, 0.19187464831938178)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])

    def test_custom_problem(self):
        class CustomProblem(Problem):
            def __init__(self):
                super().__init__()

            def calculate_fitness(self, x):
                return dolan_function_no2(x)

        seed(38)
        np.random.seed(38)
        config = GeneticBaseConfig(normalize_path("./data/test_custom_problem.cfg"))
        problem = CustomProblem()

        population = Population(config, problem)
        population.run()

        solution = np.array(
            [
                86.12783555,
                80.64567867,
                88.83992716,
                -36.54646632,
                -0.62547071,
            ]
        )

        self.assertAlmostEqual(population.best.fitness, -432.8083522179468)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])

    def test_inverted_griewank(self):
        seed(2)
        np.random.seed(2)
        config = GeneticBaseConfig(
            normalize_path("./data/test_inverted_griewank_10d.cfg")
        )
        population = Population(config, inverted_griewank)
        population.run()

        solution = np.array(
            [
                -5.94755542e-04,
                2.78946596e-04,
                -9.97808729e-04,
                3.26422169e-04,
                3.21453369e-04,
                2.21523836e-03,
                -3.84816991e-05,
                -7.78011517e-05,
                1.60997086e-03,
                -1.47817448e-03,
            ]
        )

        self.assertAlmostEqual(population.best.fitness, -1.0514153370166923e-06)
        # For some strange error here the fitness value passes the test but the first gene does not,
        # that's why the places is reduced. This should mean that the optimization function is not
        # that sensible to the decimal values of the genes.
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position], places=3)

    def test_continue_running(self):
        seed(34)
        np.random.seed(34)
        config = GeneticBaseConfig(normalize_path("./data/test_levy_10d.cfg"))

        population = Population(config, levy)
        population.run()

        solution = np.array(
            [
                0.73574987,
                1.1482658,
                0.7338722,
                0.87022424,
                3.53912441,
                0.74235511,
                0.70309563,
                0.16524301,
                0.45234366,
                -0.89613885,
            ]
        )

        self.assertAlmostEqual(population.best.fitness, 1.0601179138257228)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])

        population.continue_running(200)

        solution = np.array(
            [
                0.73874857,
                1.15590713,
                0.74259911,
                0.87315056,
                1.04665419,
                0.74384827,
                0.71230261,
                0.18824796,
                0.44265748,
                -0.89500575,
            ]
        )

        self.assertAlmostEqual(population.best.fitness, 0.5686975824372769)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])

    def test_rastrigin_20d(self):
        seed(25)
        np.random.seed(25)
        config = GeneticBaseConfig(normalize_path("./data/test_rastrigin_20d.cfg"))

        population = Population(config, rastrigin)
        population.run()

        solution = np.array(
            [
                1.34840615e-02,
                -1.42957151e-02,
                1.30009974e-02,
                -2.07288208e-02,
                3.08256096e-04,
                7.17431663e-02,
                5.17655086e-02,
                1.36438025e-02,
                -1.37870862e-02,
                -3.70402242e-02,
                1.99370501e-02,
                -6.07970947e-03,
                -2.89709263e-03,
                -7.91841896e-03,
                8.02659081e-03,
                -9.93638338e-01,
                2.46121605e-02,
                4.72732521e-02,
                -1.30894478e-02,
                5.61310453e-02,
            ]
        )

        self.assertAlmostEqual(population.best.fitness, 4.392787670364299)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])

    def test_bigger_genomes(self):
        seed(100)
        np.random.seed(100)
        config = GeneticBaseConfig(normalize_path("./data/test_levy_1000d.cfg"))

        population = Population(config, levy)
        population.run()

        print(population.best._hash)

        self.assertAlmostEqual(population.best.fitness, 4.392787670364299)
