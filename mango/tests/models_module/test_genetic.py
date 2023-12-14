from random import seed
from unittest import TestCase

import numpy as np

from mango.benchmark.optimization import (
    ackley,
    inverted_griewank,
    levy,
    rastrigin,
    cross_in_tray,
    schwefel,
)
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
        print(f"Select parents: {self.population.internal_debug_counter}")
        print(f"Select parents inner: {self.population.select_parent_counter}")

    def test_cross_in_tray(self):
        seed(33)
        np.random.seed(33)
        config = GeneticBaseConfig(normalize_path("./data/test_cross_in_tray.cfg"))
        self.population = Population(config, cross_in_tray)
        self.population.run()

        solution = np.array([-1.330532, -1.35732278])

        self.assertAlmostEqual(self.population.best.fitness, -2.0625601772489146)
        for position, value in enumerate(self.population.best.genes):
            self.assertAlmostEqual(value, solution[position])

    def test_ackley(self):
        seed(42)
        np.random.seed(42)
        config = GeneticBaseConfig(normalize_path("./data/test_ackley.cfg"))
        self.population = Population(config, ackley)
        self.population.run()
        best_idx = 2019

        self.assertEqual(self.population.best.fitness, 0.0013287595059487955)
        self.assertEqual(self.population.best.idx, best_idx)

    def test_bukin(self):
        seed(23)
        np.random.seed(23)
        config = GeneticBaseConfig(normalize_path("./data/test_bukin.cfg"))
        self.population = Population(config, bukin_function_6)
        self.population.run()

        solution = np.array([-4.8699058, 0.23715526])

        self.assertAlmostEqual(self.population.best.fitness, 0.2650114528699309)
        for position, value in enumerate(self.population.best.genes):
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

        self.population = Population(config, problem)
        self.population.run()

        solution = np.array(
            [48.50515125, 99.77861667, 94.66805598, -99.69796702, 6.61408047]
        )

        self.assertAlmostEqual(self.population.best.fitness, -457.98922211094884)
        for position, value in enumerate(self.population.best.genes):
            self.assertAlmostEqual(value, solution[position])

    def test_inverted_griewank(self):
        seed(2)
        np.random.seed(2)
        config = GeneticBaseConfig(
            normalize_path("./data/test_inverted_griewank_10d.cfg")
        )
        self.population = Population(config, inverted_griewank)
        self.population.run()

        solution = np.array(
            [
                6.03311073e-05,
                -4.51454806e-04,
                -2.87190420e-03,
                4.47250605e-03,
                -6.48094691e-04,
                -2.82147509e-04,
                4.95029981e-03,
                -1.50351185e-03,
                -1.00338090e-03,
                -1.37725140e-04,
            ]
        )

        self.assertEqual(self.population.best.fitness, -5.939190922399362e-06)

        for position, value in enumerate(self.population.best.genes):
            self.assertAlmostEqual(value, solution[position])

    def test_continue_running(self):
        seed(34)
        np.random.seed(34)
        config = GeneticBaseConfig(normalize_path("./data/test_schwefel.cfg"))

        self.population = Population(config, schwefel)
        self.population.run()

        solution = np.array(
            [
                420.9093842,
                420.8662836,
                420.96870499,
                420.46678371,
                420.81416279,
                420.18168286,
                421.27713374,
                421.98149446,
                421.17685681,
                420.8241212,
                420.77162868,
                420.63757586,
                420.74841843,
                421.06334541,
                420.93782594,
                420.94873489,
                420.93164784,
                420.82082408,
                420.9530681,
                420.58314515,
            ]
        )

        self.assertAlmostEqual(self.population.best.fitness, 0.3124282655753632)
        for position, value in enumerate(self.population.best.genes):
            self.assertAlmostEqual(value, solution[position])

        self.population.continue_running(1000)

        solution = np.array(
            [
                420.9093842,
                420.8662836,
                420.96870499,
                420.65485923,
                420.88210439,
                420.96535357,
                421.27713374,
                421.72821781,
                421.17685681,
                420.8241212,
                420.77162868,
                420.63757586,
                420.74841843,
                421.06334541,
                420.93782594,
                420.94873489,
                420.93164784,
                420.82082408,
                420.92400862,
                420.58314515,
            ]
        )

        self.assertAlmostEqual(self.population.best.fitness, 0.1564223103414406)
        for position, value in enumerate(self.population.best.genes):
            self.assertAlmostEqual(value, solution[position])

    def test_rastrigin_20d(self):
        seed(25)
        np.random.seed(25)
        config = GeneticBaseConfig(normalize_path("./data/test_rastrigin.cfg"))

        self.population = Population(config, rastrigin)
        self.population.run()

        solution = np.array(
            [
                0.08048778,
                1.01996013,
                0.12595788,
                -2.01235437,
                0.01585046,
                -0.03250301,
                1.04164099,
                -0.04206057,
                -1.01786207,
                -0.05196861,
                0.03712523,
                1.00794357,
                -0.03372196,
                -0.11500333,
                -0.0336348,
                1.0669465,
                -0.94485628,
                -0.0430894,
                -0.95392393,
                0.95407875,
            ]
        )

        self.assertAlmostEqual(self.population.best.fitness, 23.881451399608125)
        for position, value in enumerate(self.population.best.genes):
            self.assertAlmostEqual(value, solution[position])

    def test_bigger_genomes(self):
        seed(100)
        np.random.seed(100)
        config = GeneticBaseConfig(normalize_path("./data/test_levy.cfg"))

        self.population = Population(config, levy)
        self.population.run()

        self.assertAlmostEqual(self.population.best.fitness, 29.59474317384334)
