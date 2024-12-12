from unittest import TestCase

import numpy as np

from mango_time_series.time_series.heteroscedasticity import (
    detect_and_transform_heteroscedasticity,
    apply_boxcox_with_lambda,
    get_optimal_lambda,
)


class TestHeteroscedasticityTester(TestCase):

    def setUp(self):
        """
        Setup for the tests. We create sample data for heteroscedasticity detection.
        """
        np.random.seed(42)
        # Generate heteroscedasticity data
        self.series_with_heteroscedasticity = np.exp(
            np.linspace(1, 5, 100)
        ) + np.random.normal(0, 1, 100)
        # Generate homoscedasticity data
        self.series_without_heteroscedasticity = np.random.normal(0, 1, 100)

    def test_get_optimal_lambda_with_negative_values(self):
        """
        Test get_optimal_lambda with a series that contains negative values.
        """
        series_with_negative = (
            self.series_with_heteroscedasticity - 50
        )  # Introduce negative values
        optimal_lambda = get_optimal_lambda(series_with_negative)

        # Validate that the lambda is computed and that min_value adjustment was applied
        self.assertIsNotNone(optimal_lambda)

    def test_apply_boxcox_with_lambda_with_negative_values(self):
        """
        Test apply_boxcox_with_lambda with a series that contains negative values.
        """
        series_with_negative = (
            self.series_with_heteroscedasticity - 50
        )  # Introduce negative values
        lambda_value = 0.5
        transformed_series = apply_boxcox_with_lambda(
            series_with_negative, lambda_value
        )

        # Check that the series length remains consistent and transformation is applied
        self.assertEqual(len(transformed_series), len(series_with_negative))

    def test_get_optimal_lambda(self):
        """
        Test the calculation of the optimal lambda using the Box-Cox transformation.
        """
        optimal_lambda = get_optimal_lambda(self.series_with_heteroscedasticity)
        self.assertIsNotNone(optimal_lambda)

    def test_apply_boxcox_with_lambda(self):
        """
        Test the application of Box-Cox transformation with a given lambda value.
        """
        lambda_value = 0.5
        transformed_series = apply_boxcox_with_lambda(
            self.series_with_heteroscedasticity, lambda_value
        )
        self.assertEqual(
            len(transformed_series), len(self.series_with_heteroscedasticity)
        )

    def test_breusch_pagan_detection_and_transformation(self):
        """
        Test Breusch-Pagan detection method for heteroscedasticity and apply transformations if necessary.
        """
        # Test for a series with heteroscedasticity
        transformed_series, optimal_lambda = detect_and_transform_heteroscedasticity(
            series=self.series_with_heteroscedasticity
        )
        self.assertIsNotNone(optimal_lambda)
        self.assertEqual(
            len(transformed_series), len(self.series_with_heteroscedasticity)
        )

        # Test for a series without heteroscedasticity
        transformed_series, optimal_lambda = detect_and_transform_heteroscedasticity(
            series=self.series_without_heteroscedasticity
        )
        self.assertIsNone(optimal_lambda)
        self.assertEqual(
            len(transformed_series), len(self.series_without_heteroscedasticity)
        )

    def tearDown(self):
        pass
