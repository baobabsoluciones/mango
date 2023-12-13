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
        values = [0 for _ in range(100)]

        self.assertEqual(0.0, ackley(values))
        self.assertEqual(0.0, inverted_ackley(values))

        values = [uniform(-32.768, 32.768) for _ in range(100)]

        self.assertAlmostEqual(21.370533416203124, ackley(values), delta=0.1)
        self.assertAlmostEqual(-21.370533416203124, inverted_ackley(values), delta=0.1)

    def test_bukin(self):
        seed(23)
        values = [-10, 1]

        self.assertEqual(0.0, bukin_function_6(values))
        self.assertEqual(0.0, inverted_bukin_function_6(values))

        values = [uniform(-15, 5), uniform(-3, 3)]

        self.assertAlmostEqual(160.4260596101911, bukin_function_6(values), delta=0.1)
        self.assertAlmostEqual(
            -160.4260596101911, inverted_bukin_function_6(values), delta=0.1
        )

    def test_cross_in_tray(self):
        seed(24)

        values = [
            [1.34941, 1.34941],
            [-1.34941, 1.34941],
            [1.34941, -1.34941],
            [-1.34941, -1.34941],
        ]

        for group in values:
            self.assertAlmostEqual(-2.06261, cross_in_tray(group), places=5)
            self.assertAlmostEqual(2.06261, inverted_cross_in_tray(group), places=5)

        values = [uniform(-10, 10) for _ in range(2)]

        self.assertAlmostEqual(-1.5716297958424037, cross_in_tray(values))
        self.assertAlmostEqual(1.5716297958424037, inverted_cross_in_tray(values))

    def test_dolan_function_no2(self):
        seed(25)
        values = [8.39045925, 4.81424707, 7.34574133, 68.88246895, 3.85470806]

        self.assertAlmostEqual(10e-5, dolan_function_no2(values), places=3)
        self.assertAlmostEqual(-10e-5, inverted_dolan_function_no2(values), places=3)

        values = [uniform(-100, 100) for _ in range(5)]

        self.assertAlmostEqual(972.8855228788035, dolan_function_no2(values))
        self.assertAlmostEqual(-972.8855228788035, inverted_dolan_function_no2(values))
