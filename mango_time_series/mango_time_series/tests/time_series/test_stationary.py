from datetime import datetime
from unittest import TestCase

import numpy as np
import polars as pl

from mango_time_series.time_series.stationary import StationaryTester


class TestStationaryTester(TestCase):

    def setUp(self):
        np.random.seed(42)

        # Dates for daily series in Polars
        dates_daily_polars = pl.date_range(
            start=datetime(2022, 1, 1),
            end=datetime(2022, 4, 10),
            interval="1d",
            eager=True,
        )

        # Dates for monthly series in Polars
        dates_monthly_polars = pl.date_range(
            start=datetime(2022, 1, 1),
            end=datetime(2026, 12, 1),
            interval="1mo",
            eager=True,
        )

        # Generate non-stationary series (with trend) for daily data
        self.series_with_trend = np.cumsum(np.random.normal(0, 1, 100)) + np.linspace(
            0, 10, 100
        )
        self.df_with_trend = pl.DataFrame(
            {"date": dates_daily_polars, "target": self.series_with_trend}
        )

        # Generate stationary series for daily data
        self.stationary_series = np.random.normal(0, 1, 100)
        self.df_stationary = pl.DataFrame(
            {"date": dates_daily_polars, "target": self.stationary_series}
        )

        # Generate seasonal series for daily data (7-day period for weekly seasonality)
        values = 10 * np.sin(2 * np.pi * np.arange(len(dates_monthly_polars)) / 12)
        values -= np.mean(values)
        self.df_seasonal = pl.DataFrame(
            {
                "date": dates_monthly_polars,
                "target": values,
            }
        )

        # Generate seasonal series for monthly data (12-month period for annual seasonality)
        values = 10 * np.sin(2 * np.pi * np.arange(60) / 12)
        values -= np.mean(values)
        self.df_monthly_seasonal = pl.DataFrame(
            {"date": dates_monthly_polars, "target": values}
        )

        # Generate a series with both annual seasonality and trend for monthly data
        self.series_with_trend_and_seasonality = (
            np.linspace(0, 20, 60)
            + 10 * np.sin(2 * np.pi * np.arange(60) / 12)
            + np.random.normal(0, 1, 60)
        )
        self.df_trend_seasonality = pl.DataFrame(
            {
                "date": dates_monthly_polars,
                "target": self.series_with_trend_and_seasonality,
            }
        )

        self.serie_soft_trend = 0.01 * np.arange(60) + np.random.normal(0, 0.2, 60)
        self.df_soft_trend = pl.DataFrame(
            {"date": dates_monthly_polars, "target": self.serie_soft_trend}
        )

        self.tester = StationaryTester()

    def test_stationary_series(self):
        """
        Test that a stationary series is detected as stationary and no transformation is applied.
        """
        df_transformed, d_regular, d_seasonal = self.tester.make_stationary(
            self.df_stationary, target_column="target", date_column="date"
        )

        # Check that no differencing was applied (d_regular and d_seasonal should be 0)
        self.assertEqual(d_regular, 0)
        self.assertEqual(d_seasonal, 0)

        # Check that the original data is returned
        target_original = self.df_stationary["target"].to_numpy()
        target_transformed = df_transformed["target"].to_numpy()

        self.assertTrue(np.array_equal(target_original, target_transformed))

    def test_non_stationary_series(self):
        """
        Test that a non-stationary series (with trend) is transformed into a stationary one.
        """
        df_transformed, d_regular, d_seasonal = self.tester.make_stationary(
            self.df_with_trend, target_column="target", date_column="date"
        )

        # Verify that regular differencing was applied
        self.assertGreater(d_regular, 0)
        self.assertEqual(d_seasonal, 0)

    def test_seasonal_series(self):
        """
        Test that a seasonal series is properly transformed by applying seasonal differencing.
        """
        df_transformed, d_regular, d_seasonal = self.tester.make_stationary(
            self.df_seasonal, target_column="target", date_column="date"
        )

        # Check that seasonal differencing was applied
        self.assertEqual(d_regular, 0)
        self.assertGreater(d_seasonal, 0)

    def test_monthly_series_with_annual_seasonality(self):
        """
        Test that a monthly series with annual seasonality is properly transformed by applying seasonal differencing.
        """
        df_transformed, d_regular, d_seasonal = self.tester.make_stationary(
            self.df_monthly_seasonal, target_column="target", date_column="date"
        )

        # Check that seasonal differencing was applied
        self.assertEqual(d_regular, 0)
        self.assertGreater(d_seasonal, 0)

    def test_series_with_trend_and_seasonality(self):
        """
        Test that a series with both trend and annual seasonality is properly transformed
        into a stationary one by applying both regular and seasonal differencing.
        """
        df_transformed, d_regular, d_seasonal = self.tester.make_stationary(
            self.df_trend_seasonality, target_column="target", date_column="date"
        )

        # Verify that both regular and seasonal differencing were applied
        self.assertGreater(d_regular, 0)
        self.assertGreater(d_seasonal, 0)

    def test_non_stationary_KPSS_trend(self):
        """
        Test that a non-stationary series (with trend) is transformed into a stationary one.
        """
        df_transformed, d_regular, d_seasonal = self.tester.make_stationary(
            self.df_soft_trend, target_column="target", date_column="date"
        )

        # Verify that regular differencing was applied
        self.assertGreater(d_regular, 0)
        self.assertEqual(d_seasonal, 0)

    def tearDown(self):
        pass
