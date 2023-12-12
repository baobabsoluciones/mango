from random import seed
from unittest import TestCase

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
        seed(2)
        config = GeneticBaseConfig(
            normalize_path("./data/test_inverted_griewank_10d.cfg")
        )
        population = Population(config, inverted_griewank)
        population.run()

        solution = [
            3.252847727829689,
            -4.433774179578112,
            -5.517210486615845,
            0.022216846503331618,
            6.9645301043392465,
            7.669082313457459,
            8.443211199288388,
            8.838636879568437,
            9.368930636234563,
            0.0018934479035296703,
        ]

        self.assertAlmostEqual(population.best.fitness, -0.1104620174877462)
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

    def test_rastrigin_20d(self):
        seed(22)
        config = GeneticBaseConfig(normalize_path("./data/test_rastrigin_20d.cfg"))

        population = Population(config, rastrigin)
        population.run()

        print(population.best.genes)

        solution = [
            0.026076077944644283,
            0.005086069396593729,
            -0.015973415638949717,
            -0.023799657814452146,
            0.06430399202537007,
            -0.01744462373686151,
            -0.014117319579181498,
            -0.03784225769599381,
            0.022639506452455826,
            0.011829175660904134,
            -0.07154841836692771,
            -0.05276671871202332,
            0.0014310285009333512,
            0.004973043612503503,
            0.03846729504120017,
            0.010889478458071977,
            0.02102003357697324,
            0.02660794416379897,
            -0.0020305370671653833,
            -0.0018585969342153064,
        ]

        self.assertAlmostEqual(population.best.fitness, 3.719865118417033)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])
