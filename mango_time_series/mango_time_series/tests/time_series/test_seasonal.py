from unittest import TestCase

import numpy as np

from mango_time_series.time_series.seasonal import SeasonalityDetector


class TestSeasonalityDetector(TestCase):

    def setUp(self):
        """
        Setup for the tests. We create sample time series for seasonality detection.
        """
        np.random.seed(42)

        # Generate a time series with clear weekly seasonality (7-day period)
        self.weekly_seasonal_series = 10 * np.sin(
            2 * np.pi * np.arange(0, 365) / 7
        ) + np.random.normal(0, 1, 365)

        # Generate a time series with multiple seasonality: weekly and yearly
        self.multiple_seasonal_series = (
            10 * np.sin(2 * np.pi * np.arange(0, 365 * 2) / 7)
            + 50 * np.sin(2 * np.pi * np.arange(0, 365 * 2) / 365)
            + np.random.normal(0, 1, 365 * 2)
        )

        # Generate a random series with no seasonality (pure noise)
        self.series_noise = np.random.normal(0, 1, 365)

        # Generate a time series with clear annual seasonality (12-month period for monthly data)
        months = np.arange(0, 12 * 5)
        self.annual_seasonal_series = 20 * np.sin(
            2 * np.pi * months / 12
        ) + np.random.normal(0, 1, len(months))

        # Generate an hourly time series with daily seasonality (24-hour period)
        hours = np.arange(0, 24 * 7)  # 7 days of hourly data
        self.hourly_seasonal_series = 5 * np.sin(
            2 * np.pi * hours / 24
        ) + np.random.normal(0, 0.5, len(hours))

    def test_weekly_seasonality(self):
        """
        Test detection of weekly seasonality.
        """
        detector = SeasonalityDetector()
        detected_periods = detector.detect_seasonality(self.weekly_seasonal_series)
        self.assertIn(
            7.0, detected_periods, "Should detect weekly seasonality (7-day period)"
        )

    def test_multiple_seasonality(self):
        """
        Test detection of multiple seasonalities (weekly and yearly).
        """
        detector = SeasonalityDetector()
        detected_periods = detector.detect_seasonality(self.multiple_seasonal_series)
        self.assertIn(
            7.0, detected_periods, "Should detect weekly seasonality (7-day period)"
        )
        self.assertIn(
            365.0, detected_periods, "Should detect yearly seasonality (365-day period)"
        )

    def test_no_seasonality(self):
        """
        Test detection of no seasonality in a pure noise series.
        """
        detector = SeasonalityDetector()
        detected_periods = detector.detect_seasonality(self.series_noise)
        self.assertEqual(
            len(detected_periods),
            0,
            "No seasonality should be detected in pure noise series",
        )

    def test_acf_no_significant_lags(self):
        no_seasonality_series = np.random.normal(0, 1, 30)
        detected = SeasonalityDetector.detect_significant_seasonality_acf(
            no_seasonality_series, max_lag=30, acf_threshold=0.2, min_repetitions=3
        )
        self.assertEqual(detected, 0)

    def test_annual_seasonality_in_monthly_series(self):
        """
        Test detection of annual seasonality in monthly data (12-month period).
        """
        detector = SeasonalityDetector()
        detected_periods = detector.detect_seasonality(self.annual_seasonal_series)
        self.assertIn(
            12.0, detected_periods, "Should detect annual seasonality (12-month period)"
        )

    def test_hourly_seasonality(self):
        """
        Test detection of hourly seasonality (24-hour period).
        """
        detector = SeasonalityDetector()
        detected_periods = detector.detect_seasonality(self.hourly_seasonal_series)
        self.assertIn(
            24.0,
            detected_periods,
            "Should detect daily seasonality (24-hour period in hourly data)",
        )

    def tearDown(self):
        pass
