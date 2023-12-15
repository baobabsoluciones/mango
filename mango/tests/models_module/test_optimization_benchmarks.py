import time

import numpy as np
from random import seed, uniform
from unittest import TestCase

from mango.benchmark.optimization import (
    ackley,
    inverted_ackley,
    bukin_function_6,
    inverted_bukin_function_6,
    cross_in_tray,
    inverted_cross_in_tray,
    dolan_function_no2,
    inverted_dolan_function_no2,
)


class TestOptimizationBenchmarks(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ackley(self):
        seed(22)
        np.random.seed(22)
        values = np.zeros(100)

        self.assertEqual(0.0, ackley(values))
        self.assertEqual(0.0, inverted_ackley(values))

        values = np.random.uniform(-32.768, 32.768, 100)

        self.assertEqual(21.380371270628963, ackley(values))
        self.assertEqual(-21.380371270628963, inverted_ackley(values))

    def test_bukin(self):
        seed(23)
        np.random.seed(23)
        values = np.array([-10, 1])

        self.assertEqual(0.0, bukin_function_6(values))
        self.assertEqual(0.0, inverted_bukin_function_6(values))

        values = np.random.uniform(15, 5, 2)

        self.assertEqual(213.849165663974, bukin_function_6(values))
        self.assertEqual(-213.849165663974, inverted_bukin_function_6(values))

    def test_cross_in_tray(self):
        seed(24)
        np.random.seed(24)

        values = [
            np.array([1.34941, 1.34941]),
            np.array([-1.34941, 1.34941]),
            np.array([1.34941, -1.34941]),
            np.array([-1.34941, -1.34941]),
        ]

        for group in values:
            self.assertEqual(-2.062611870820258, cross_in_tray(group))
            self.assertEqual(2.062611870820258, inverted_cross_in_tray(group))

        values = np.random.uniform(-10, 10, 2)

        self.assertEqual(-1.3383943647492602, cross_in_tray(values))
        self.assertEqual(1.3383943647492602, inverted_cross_in_tray(values))

    def test_dolan_function_no2(self):
        seed(25)
        np.random.seed(25)
        values = np.array([8.39045925, 4.81424707, 7.34574133, 68.88246895, 3.85470806])

        self.assertEqual(2.2149074663246893e-07, dolan_function_no2(values))
        self.assertEqual(-2.2149074663246893e-07, inverted_dolan_function_no2(values))

        values = np.random.uniform(-100, 100, 5)

        self.assertEqual(7.233634216388971, dolan_function_no2(values))
        self.assertEqual(-7.233634216388971, inverted_dolan_function_no2(values))

    def test_drop_wave_function(self):
        pass

    def test_egg_holder_function(self):
        pass

    def test_gramacy_lee_function(self):
        pass

    def test_griewank_function(self):
        pass

    def test_holder_table_function(self):
        pass

    def test_lagergren_function(self):
        pass

    def test_levy_function(self):
        pass

    def test_rastrigin_function(self):
        pass

    def test_rosenbrock_function(self):
        pass

    def test_schaffer_function(self):
        pass

    def test_schwefel_function(self):
        pass
