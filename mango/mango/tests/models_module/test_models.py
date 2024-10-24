from unittest import TestCase

import numpy as np
from mango.models import sigmoid, tanh, calculate_network_output


class ActivationTests(TestCase):
    def setUp(self):
        pass

    def tearDown(self) -> None:
        pass

    def test_sigmoid_activation(self):
        x = 1
        y = sigmoid(x)
        self.assertAlmostEqual(y, 0.7310585786300049)
        y = sigmoid(x, 2)
        self.assertAlmostEqual(y, 0.8807970779778823)

    def test_tahn_activation(self):
        x = 1
        y = tanh(x)
        self.assertAlmostEqual(y, 0.7615941559557649)
        y = tanh(x, 2)
        self.assertAlmostEqual(y, 0.9640275800758169)


class NetworkTests(TestCase):
    def setUp(self):
        pass

    def tearDown(self) -> None:
        pass

    def test_network(self):
        x, y = [(0, 0), (0, 1), (1, 0), (1, 1)], [0, 1, 1, 0]
        x = np.array(x)
        results = calculate_network_output(
            x, np.array([0.5, 0.5, 0, 0.5, 0]), 1, 2, [1], 1
        )
        self.assertEqual(
            True,
            np.allclose(
                results,
                np.array([[0.5621765], [0.57718538], [0.57718538], [0.59037826]]),
            ),
        )
