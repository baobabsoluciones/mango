from unittest import TestCase
import numpy as np
import pandas as pd
from mango_time_series.mango.time_series.decomposition import SeasonalityDecompose

class TestSeasonalityDecompose(TestCase):

    def setUp(self):
        np.random.seed(42)
        self.series = pd.Series(
            10 * np.sin(2 * np.pi * np.arange(120) / 12) + np.random.normal(0, 1, 120),
            index=pd.date_range(start="1959-01-01", periods=120, freq="MS"))

        self.series_mstl = pd.Series(
            10 * np.sin(2 * np.pi * np.arange(365 * 4) / 7) + 5 * np.sin(
                2 * np.pi * np.arange(365 * 4) / 365) + np.random.normal(0, 1, 365 * 4),
            index=pd.date_range(start="2020-01-01", periods=365 * 4, freq="D")
        )
        # Create an instance of SeasonalityDecompose
        self.decomposer = SeasonalityDecompose()

    def test_decompose_stl(self):
        """
        Test the STL decomposition method.
        """
        trend, seasonal, resid = self.decomposer.decompose_stl(self.series, period=13)

        # Check that the outputs are of the correct length
        self.assertEqual(len(trend), len(self.series))
        self.assertEqual(len(seasonal), len(self.series))
        self.assertEqual(len(resid), len(self.series))

    def test_calculate_seasonal_strength(self):
        """
        Test the calculation of seasonal strength.
        """
        trend, seasonal, resid = self.decomposer.decompose_stl(self.series, period=13)
        fs = self.decomposer.calculate_seasonal_strength(seasonal, resid)

        # Check that the seasonal strength is a float and within reasonable bounds (0 <= Fs <= 1)
        self.assertIsInstance(fs, float)
        self.assertGreaterEqual(fs, 0)
        self.assertLessEqual(fs, 1)

    def test_detect_seasonality(self):
        """
        Test if the detect_seasonality method correctly identifies seasonality.
        """
        has_seasonality = self.decomposer.detect_seasonality(self.series, period=13)

        # The generated series has a clear yearly seasonality, so it should return True
        self.assertTrue(has_seasonality)

    def test_decompose_mstl(self):
        """
        Test the MSTL decomposition method.
        """
        periods = [7, 365]
        trend, seasonal, resid = self.decomposer.decompose_mstl(self.series_mstl, periods=periods)

        self.assertEqual(len(trend), len(self.series_mstl))
        self.assertEqual(len(seasonal), len(self.series_mstl))
        self.assertEqual(len(resid), len(self.series_mstl))

    def tearDown(self):
        pass

