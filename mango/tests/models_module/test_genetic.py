from random import seed
from unittest import TestCase

from mango.benchmark.optimization import ackley, inverted_griewank, levy
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

        solution = [0.04688254566426053, 0.05306303160494963]

        self.assertAlmostEqual(population.best.fitness, 0.3294397837892582)
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

        solution = [-3.92972796431617, 0.15442613259239604]

        self.assertAlmostEqual(population.best.fitness, 0.18261018071331875)
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
            86.02924663796051,
            91.34626467166257,
            75.75079061521132,
            71.17276815342376,
            -5.180918017635875,
        ]

        self.assertAlmostEqual(population.best.fitness, -428.72294223517395)
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
            6.309391234006169,
            4.50874683190745,
            5.452981628160172,
            0.21102096139689291,
            0.06444193222575913,
            -7.708571022040515,
            -8.2408106578037,
            0.14757221958266184,
            -0.11088277433051322,
            0.03817064084575229,
        ]

        self.assertAlmostEqual(population.best.fitness, -0.06420897800089553)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])

    def test_continue_running(self):
        seed(22)
        config = GeneticBaseConfig(normalize_path("./data/test_levy_10d.cfg"))

        population = Population(config, levy)
        population.run()

        solution = [
            1.046293677436979,
            1.045425234956985,
            0.6827048260235041,
            1.0766023419009219,
            -0.16208100990762414,
            -0.04221658979108796,
            0.9614474113850087,
            0.9998592579526275,
            0.92876804413118,
            1.5299647554782438,
        ]

        self.assertAlmostEqual(population.best.fitness, 0.2533584234516954)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])

        population.continue_running(200)

        solution = [
            1.043084062453158,
            1.0463364506301889,
            0.6839910498994877,
            1.0762212536890368,
            -0.162226471527582,
            -0.04224756064612065,
            0.9650844883660079,
            0.9991564122982302,
            0.9286365696445134,
            1.0113621127349337,
        ]

        self.assertAlmostEqual(population.best.fitness, 0.22555413869389201)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])
