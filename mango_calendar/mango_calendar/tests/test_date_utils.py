"""Unit tests for date_utils module."""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl

from mango_calendar.date_utils import (
    get_covid_lockdowns,
    get_holidays_df,
)


class TestDateUtils(unittest.TestCase):
    """Test cases for date_utils module."""

    def test_get_holidays_df_polars(self) -> None:
        """Test get_holidays_df with Polars output."""
        result = get_holidays_df(
            steps_back=2,
            steps_forward=3,
            start_year=2023,
            country="ES",
            output_format="polars",
        )

        # Check if result is a Polars DataFrame
        self.assertIsInstance(result, pl.DataFrame)

        # Check that datetime column exists
        self.assertIn("datetime", result.columns)

        # Check that result is not empty
        self.assertGreater(len(result), 0)

        # Check that datetime column has date type
        self.assertEqual(result["datetime"].dtype, pl.Date)

    def test_get_holidays_df_pandas(self) -> None:
        """Test get_holidays_df with Pandas output."""
        result = get_holidays_df(
            steps_back=2,
            steps_forward=3,
            start_year=2023,
            country="ES",
            output_format="pandas",
        )

        # Check if result is a Pandas DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check that datetime column exists
        self.assertIn("datetime", result.columns)

        # Check that result is not empty
        self.assertGreater(len(result), 0)

    def test_get_holidays_df_invalid_format(self) -> None:
        """Test get_holidays_df with invalid output format."""
        with self.assertRaises(ValueError) as context:
            get_holidays_df(steps_back=2, steps_forward=3, output_format="invalid")

        self.assertIn("should be either 'polars' or 'pandas'", str(context.exception))

    def test_get_holidays_df_parameters(self) -> None:
        """Test get_holidays_df with different parameters."""
        result = get_holidays_df(
            steps_back=5,
            steps_forward=10,
            start_year=2022,
            country="ES",
            output_format="polars",
        )

        # Check if result is a Polars DataFrame
        self.assertIsInstance(result, pl.DataFrame)

        # Check that result is not empty
        self.assertGreater(len(result), 0)

    def test_get_covid_lockdowns(self) -> None:
        """Test get_covid_lockdowns function."""
        result = get_covid_lockdowns()

        # Check if result is a Polars DataFrame
        self.assertIsInstance(result, pl.DataFrame)

        # Check required columns
        expected_columns = {"ds", "name", "lower_bound", "upper_bound"}
        self.assertTrue(expected_columns.issubset(set(result.columns)))

        # Check that all name values are "COVID"
        self.assertTrue(all(result["name"] == "COVID"))

        # Check that bounds are integers
        self.assertEqual(result["lower_bound"].dtype, pl.Int64)
        self.assertEqual(result["upper_bound"].dtype, pl.Int64)

        # Check that all bounds are 0
        self.assertTrue(all(result["lower_bound"] == 0))
        self.assertTrue(all(result["upper_bound"] == 0))

        # Check date range (should be about 2 years)
        self.assertGreater(len(result), 700)  # Approximately 2 years worth of days
        self.assertLess(len(result), 800)

    def test_get_covid_lockdowns_date_range(self) -> None:
        """Test that COVID lockdowns cover the expected date range."""
        result = get_covid_lockdowns()

        # Sort by date and get first and last rows to check date range
        sorted_result = result.sort("ds")
        first_date = sorted_result.select("ds").item(0, 0)
        last_date = sorted_result.select("ds").item(-1, 0)

        # Should start around March 1, 2020
        self.assertEqual(first_date.year, 2020)
        self.assertEqual(first_date.month, 3)
        self.assertEqual(first_date.day, 1)

        # Should end before March 1, 2022
        self.assertEqual(last_date.year, 2022)
        self.assertEqual(last_date.month, 2)

    @patch("mango_calendar.date_utils.get_calendar")
    def test_get_holidays_df_integration(self, mock_get_calendar: MagicMock) -> None:
        """Test integration with get_calendar function."""
        # Mock the get_calendar function to return test data
        mock_data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3, freq="D"),
                "name": ["Holiday1", "Holiday2", "Holiday3"],
                "distance": [0, 1, 2],
                "weight": [1.0, 0.8, 0.6],
            }
        )
        mock_get_calendar.return_value = mock_data

        get_holidays_df(steps_back=1, steps_forward=2, output_format="polars")

        # Check that get_calendar was called with correct parameters
        mock_get_calendar.assert_called_once()
        call_args = mock_get_calendar.call_args
        if call_args:
            kwargs = call_args[1]
            self.assertEqual(kwargs["communities"], True)
            self.assertEqual(kwargs["calendar_events"], True)
            self.assertEqual(kwargs["return_distances"], True)
            self.assertEqual(
                kwargs["distances_config"], {"steps_forward": 2, "steps_back": 1}
            )


if __name__ == "__main__":
    unittest.main()
