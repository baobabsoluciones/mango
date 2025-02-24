import unittest
import numpy as np
import tensorflow as tf

from mango_time_series.models.losses import mean_squared_error


class TestLosses(unittest.TestCase):
    def test_mean_squared_error_scalar(self):
        """
        Test mean squared error with scalar values
        """
        # Test with scalar values
        y_true = tf.constant([[1.0], [2.0], [3.0]])
        y_pred = tf.constant([[2.0], [2.0], [2.0]])

        # Expected MSE = ((2-1)^2 + (2-2)^2 + (2-3)^2) / 3 = (1 + 0 + 1) / 3 = 0.6666...
        expected = 0.6666667

        result = mean_squared_error(y_true, y_pred)
        np.testing.assert_almost_equal(result.numpy(), expected, decimal=6)

    def test_mean_squared_error_vector(self):
        """
        Test mean squared error with vector values
        """
        # Test with vector values
        y_true = tf.constant([[1.0, 2.0], [2.0, 3.0]])
        y_pred = tf.constant([[1.0, 1.0], [2.0, 2.0]])

        # Expected MSE = ((1-1)^2 + (1-2)^2 + (2-2)^2 + (2-3)^2) / 4 = (0 + 1 + 0 + 1) / 4 = 0.5
        expected = 0.5

        result = mean_squared_error(y_true, y_pred)
        np.testing.assert_almost_equal(result.numpy(), expected, decimal=6)

    def test_mean_squared_error_zero(self):
        """
        Test mean squared error with perfect predictions
        """
        # Test with perfect predictions
        y_true = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y_pred = tf.constant([[1.0, 2.0], [3.0, 4.0]])

        # Expected MSE should be 0 for perfect predictions
        expected = 0.0

        result = mean_squared_error(y_true, y_pred)
        np.testing.assert_almost_equal(result.numpy(), expected, decimal=6)

    def test_mean_squared_error_3d(self):
        """
        Test mean squared error with 3D tensors (batch, timesteps, features)
        """
        # Test with 3D tensors (batch, timesteps, features)
        y_true = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        y_pred = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 7.0]]])

        # Only last value differs by 1, MSE = 1^2 / 8 = 0.125 (8 total values)
        expected = 0.125

        result = mean_squared_error(y_true, y_pred)
        np.testing.assert_almost_equal(result.numpy(), expected, decimal=6)


if __name__ == "__main__":
    unittest.main()
