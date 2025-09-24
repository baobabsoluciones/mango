from random import seed, uniform
from unittest import TestCase

import numpy as np
from mango.benchmark import (
    drop_wave,
    inverted_drop_wave,
    egg_holder,
    inverted_egg_holder,
    gramacy_lee,
    inverted_gramacy_lee,
    griewank,
    inverted_griewank,
    holder_table,
    inverted_holder_table,
    levy,
    inverted_levy,
    levy_function_no13,
    inverted_levy_no13,
    rastrigin,
    rosenbrock,
    schaffer_function_no2,
    inverted_schaffer_function_no2,
    schaffer_function_no4,
    inverted_schaffer_function_no4,
    schwefel,
    inverted_schwefel,
)
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
from mango.benchmark.optimization.langermann import langermann, inverted_langermann
from mango.benchmark.optimization.rastrigin import inverted_rastrigin
from mango.benchmark.optimization.rosenbrock import inverted_rosenbrock


class TestOptimizationBenchmarks(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ackley(self):
        seed(22)
        np.random.seed(22)
        values = np.zeros(100)

        self.assertAlmostEqual(0.0, ackley(values))
        self.assertAlmostEqual(0.0, inverted_ackley(values))

        values = np.random.uniform(-32.768, 32.768, 100)

        self.assertAlmostEqual(21.380371270628963, ackley(values))
        self.assertAlmostEqual(-21.380371270628963, inverted_ackley(values))

    def test_bukin(self):
        seed(23)
        np.random.seed(23)
        values = np.array([-10, 1])

        self.assertAlmostEqual(0.0, bukin_function_6(values))
        self.assertAlmostEqual(0.0, inverted_bukin_function_6(values))

        values = np.random.uniform(15, 5, 2)

        self.assertAlmostEqual(213.849165663974, bukin_function_6(values))
        self.assertAlmostEqual(-213.849165663974, inverted_bukin_function_6(values))

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
            self.assertAlmostEqual(-2.062611870820258, cross_in_tray(group))
            self.assertAlmostEqual(2.062611870820258, inverted_cross_in_tray(group))

        values = np.random.uniform(-10, 10, 2)

        self.assertAlmostEqual(-1.3383943647492602, cross_in_tray(values))
        self.assertAlmostEqual(1.3383943647492602, inverted_cross_in_tray(values))

    def test_dolan_function_no2(self):
        seed(25)
        np.random.seed(25)
        values = np.array([8.39045925, 4.81424707, 7.34574133, 68.88246895, 3.85470806])

        self.assertAlmostEqual(2.2149074663246893e-07, dolan_function_no2(values))
        self.assertAlmostEqual(
            -2.2149074663246893e-07, inverted_dolan_function_no2(values)
        )

        values = np.random.uniform(-100, 100, 5)

        self.assertAlmostEqual(7.233634216388971, dolan_function_no2(values))
        self.assertAlmostEqual(-7.233634216388971, inverted_dolan_function_no2(values))

    def test_drop_wave_function(self):
        seed(26)
        np.random.seed(26)

        values = np.zeros(10)

        self.assertAlmostEqual(-1.0, drop_wave(values))
        self.assertAlmostEqual(1.0, inverted_drop_wave(values))

        values = np.random.uniform(low=-5.12, high=5.12, size=10)

        self.assertAlmostEqual(-0.0006846796655835945, drop_wave(values))
        self.assertAlmostEqual(0.0006846796655835945, inverted_drop_wave(values))

    def test_egg_holder_function(self):
        seed(27)
        np.random.seed(27)

        values = np.array([512, 404.2319])

        self.assertAlmostEqual(-959.6406627106155, egg_holder(values))
        self.assertAlmostEqual(959.6406627106155, inverted_egg_holder(values))

        values = np.random.uniform(-512, 512, 2)

        self.assertAlmostEqual(283.54519834471273, egg_holder(values))
        self.assertAlmostEqual(-283.54519834471273, inverted_egg_holder(values))

    def test_gramacy_lee_function(self):
        seed(28)
        np.random.seed(28)

        values = (np.array([0.54]), [0.54], 0.54)

        for v in values:
            self.assertAlmostEqual(-0.8358333254584753, gramacy_lee(v))
            self.assertAlmostEqual(0.8358333254584753, inverted_gramacy_lee(v))

        val = uniform(-0.5, 2.5)
        values = (np.array([val]), [val], val)
        print(values)

        for v in values:
            self.assertAlmostEqual(-1.0976975693308988, gramacy_lee(v))
            self.assertAlmostEqual(1.0976975693308988, inverted_gramacy_lee(v))

    def test_griewank_function(self):
        seed(29)
        np.random.seed(29)

        values = np.zeros(10)

        self.assertAlmostEqual(0.0, griewank(values))
        self.assertAlmostEqual(0.0, inverted_griewank(values))

        values = np.random.uniform(-600, 600, 10)

        self.assertAlmostEqual(264.94761340544994, griewank(values))
        self.assertAlmostEqual(-264.94761340544994, inverted_griewank(values))

    def test_holder_table_function(self):
        seed(30)
        np.random.seed(30)

        values = [8.05502, 9.66459]
        values = (np.array(values), values)

        for v in values:
            self.assertAlmostEqual(-19.208502567767606, holder_table(v))
            self.assertAlmostEqual(19.208502567767606, inverted_holder_table(v))

        values = np.random.uniform(-10, 10, 2)

        self.assertAlmostEqual(-0.22520163548036162, holder_table(values))
        self.assertAlmostEqual(0.22520163548036162, inverted_holder_table(values))

    def test_langermann_function(self):
        seed(31)
        np.random.seed(31)

        values = np.array([2.00299219, 1.00149609])

        self.assertAlmostEqual(5.16143849872291, langermann(values))
        self.assertAlmostEqual(-5.16143849872291, inverted_langermann(values))

    def test_levy_function(self):
        seed(32)
        np.random.seed(32)

        values = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        self.assertAlmostEqual(0.0, levy(values))
        self.assertAlmostEqual(0.0, inverted_levy(values))

        values = np.random.uniform(-10, 10, 10)

        self.assertAlmostEqual(99.15281096751649, levy(values))
        self.assertAlmostEqual(-99.15281096751649, inverted_levy(values))

        values = [1, 1]
        values = (np.array(values), values)

        for v in values:
            self.assertAlmostEqual(0.0, levy_function_no13(v))
            self.assertAlmostEqual(0.0, inverted_levy_no13(v))

        values = np.random.uniform(-10, 10, 2)

        self.assertAlmostEqual(159.3253171412186, levy_function_no13(values))
        self.assertAlmostEqual(-159.3253171412186, inverted_levy_no13(values))

    def test_rastrigin_function(self):
        seed(33)
        np.random.seed(33)

        values = np.zeros(10)

        self.assertAlmostEqual(0.0, rastrigin(values))
        self.assertAlmostEqual(0.0, inverted_rastrigin(values))

        values = np.random.uniform(-5.12, 5.12, 10)

        self.assertAlmostEqual(188.77313402387063, rastrigin(values))
        self.assertAlmostEqual(-188.77313402387063, inverted_rastrigin(values))

    def test_rosenbrock_function(self):
        seed(34)
        np.random.seed(34)

        values = np.ones(10)

        self.assertAlmostEqual(0.0, rosenbrock(values))
        self.assertAlmostEqual(0.0, inverted_rosenbrock(values))

        values = np.random.uniform(-5, 10, 10)

        self.assertAlmostEqual(990518.5847981748, rosenbrock(values))
        self.assertAlmostEqual(-990518.5847981748, inverted_rosenbrock(values))

    def test_schaffer_function(self):
        seed(35)
        np.random.seed(35)

        values = (np.array([0, 0]), [0, 0])

        for v in values:
            self.assertAlmostEqual(0.0, schaffer_function_no2(v))
            self.assertAlmostEqual(0.0, inverted_schaffer_function_no2(v))

        values = np.random.uniform(-100, 100, 2)

        self.assertAlmostEqual(0.5039215123824797, schaffer_function_no2(values))
        self.assertAlmostEqual(
            -0.5039215123824797, inverted_schaffer_function_no2(values)
        )

        values = (np.array([0, 1.25311]), [0, 1.253115])

        for v in values:
            self.assertAlmostEqual(0.2925786333928093, schaffer_function_no4(v))
            self.assertAlmostEqual(
                -0.2925786333928093, inverted_schaffer_function_no4(v)
            )

        values = np.random.uniform(-100, 100, 2)

        self.assertAlmostEqual(0.5032452712982894, schaffer_function_no4(values))
        self.assertAlmostEqual(
            -0.5032452712982894, inverted_schaffer_function_no4(values)
        )

    def test_schwefel_function(self):
        seed(36)
        np.random.seed(36)

        values = np.ones(10) * 420.968746

        self.assertAlmostEqual(0.00012727566263492918, schwefel(values))
        self.assertAlmostEqual(-0.00012727566263492918, inverted_schwefel(values))

        values = np.random.uniform(-500, 500, 10)

        self.assertAlmostEqual(4080.0940050073937, schwefel(values))
        self.assertAlmostEqual(-4080.0940050073937, inverted_schwefel(values))
