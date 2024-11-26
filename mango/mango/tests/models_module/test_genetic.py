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
from mango.models.genetic.individual import Individual
from mango.models.genetic.population.base_population import Population
from mango.models.genetic.problem.base_problem import Problem
from mango.models.genetic.shared.exceptions import GeneticDiversity
from mango.tests.const import normalize_path


class TestIndividual(TestCase):
    def setUp(self):
        seed(42)
        np.random.seed(42)

    def tearDown(self):
        pass

    def test_individual(self):
        config = GeneticBaseConfig(normalize_path("./data/test_ackley.cfg"))
        individual = Individual(config=config)
        self.assertEqual(individual.encoding, "real")
        self.assertEqual(individual.min_bound, -32.768)
        self.assertEqual(individual.max_bound, 32.768)
        self.assertEqual(individual.gene_length, 100)
        self.assertIsNone(individual.fitness)
        self.assertIsNone(individual.genes)
        self.assertIsNone(individual.parents_idx)
        self.assertEqual(individual.idx, 0)

    def test_create_random_individual(self):
        config = GeneticBaseConfig(normalize_path("./data/test_ackley.cfg"))
        individual = Individual.create_random_individual(1, config)
        self.assertEqual(individual.encoding, "real")
        self.assertEqual(individual.min_bound, -32.768)
        self.assertEqual(individual.max_bound, 32.768)
        self.assertEqual(individual.gene_length, 100)
        self.assertIsNone(individual.fitness)
        self.assertIsNotNone(individual.genes)
        self.assertIsNone(individual.parents_idx)
        self.assertEqual(individual.idx, 1)
        self.assertIsNotNone(individual._hash)

    def test_mutate_individual(self):
        config = GeneticBaseConfig(normalize_path("./data/test_ackley.cfg"))
        individual = Individual.create_random_individual(1, config)
        old_individual = individual.copy()
        old_hash = individual._hash
        individual.mutate(mutation_prob=0.8)
        self.assertNotEqual(old_hash, individual._hash)
        self.assertFalse(old_individual == individual)


class TestBaseGeneticAlgorithms(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ackley(self):
        seed(42)
        np.random.seed(42)
        config = GeneticBaseConfig(normalize_path("./data/test_ackley.cfg"))
        population = Population(config, ackley)
        population.run()
        best_idx = 15016

        self.assertEqual(population.best.fitness, 4.744205706207385)
        self.assertEqual(population.best.idx, best_idx)

    def test_bukin(self):
        seed(23)
        np.random.seed(23)
        config = GeneticBaseConfig(normalize_path("./data/test_bukin.cfg"))
        population = Population(config, bukin_function_6)
        population.run()

        solution = np.array([-4.01487863, 0.16113293])

        self.assertAlmostEqual(population.best.fitness, 0.83167899874918)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])

    def test_continue_running(self):
        seed(34)
        np.random.seed(34)
        config = GeneticBaseConfig(normalize_path("./data/test_schwefel.cfg"))

        population = Population(config, schwefel)
        population.run()

        solution = np.array(
            [
                423.84323733,
                397.64066193,
                401.07783185,
                415.25263103,
                419.4001648,
                -294.03490652,
                411.03999641,
                412.2421277,
                421.1858874,
                -312.37216189,
            ]
        )

        self.assertEqual(population.best.fitness, 400.2343490511935)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])

        population.continue_running(100)

        solution = np.array(
            [
                423.843238,
                397.90963535,
                420.94129611,
                414.48381684,
                419.40038507,
                421.49046255,
                421.1953659,
                412.24212787,
                421.08188928,
                392.03078943,
            ]
        )

        self.assertEqual(population.best.fitness, 180.97263523307947)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])

    def test_cross_in_tray(self):
        seed(33)
        np.random.seed(33)
        config = GeneticBaseConfig(normalize_path("./data/test_cross_in_tray.cfg"))
        population = Population(config, cross_in_tray)
        population.run()

        solution = np.array([1.36455618, 1.21460728])

        self.assertAlmostEqual(population.best.fitness, -2.0603701279263484)
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
        self.assertRaises(GeneticDiversity, population.run)

        solution = np.array(
            [48.50515125, 99.77861667, 94.66805598, -99.69796702, 6.61408047]
        )

        self.assertAlmostEqual(population.best.fitness, -457.98922211094884)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])

    def test_rastrigin_20d(self):
        seed(25)
        np.random.seed(25)
        config = GeneticBaseConfig(normalize_path("./data/test_rastrigin.cfg"))

        population = Population(config, rastrigin)
        self.assertRaises(GeneticDiversity, population.run)

        solution = np.array(
            [
                -0.09185608,
                -0.13252748,
                -0.10730447,
                0.12614708,
                1.04408277,
                1.11082311,
                -1.05350175,
                -0.05136119,
                -1.05360946,
                -0.05395838,
            ]
        )

        self.assertAlmostEqual(population.best.fitness, 19.57820108369802)
        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])


class TestInvertedFunctionsGenetic(TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_inverted_griewank(self):
        seed(2)
        np.random.seed(2)
        config = GeneticBaseConfig(
            normalize_path("./data/test_inverted_griewank_10d.cfg")
        )
        population = Population(config, inverted_griewank)
        population.run()

        print(population.best.genes)

        solution = np.array(
            [
                -2.92365297,
                -0.40048963,
                -5.20840551,
                0.03186117,
                0.73219018,
                0.80848836,
                0.60492159,
                0.83790558,
                0.14105634,
                0.4937259,
            ]
        )

        self.assertEqual(population.best.fitness, -0.24477631513274156)

        for position, value in enumerate(population.best.genes):
            self.assertAlmostEqual(value, solution[position])


class TestBiggerComplexGenetic(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_bigger_genomes(self):
        seed(100)
        np.random.seed(100)
        config = GeneticBaseConfig(normalize_path("./data/test_levy.cfg"))

        population = Population(config, levy)
        population.run()

        self.assertAlmostEqual(population.best.fitness, 26.994910606018838)
