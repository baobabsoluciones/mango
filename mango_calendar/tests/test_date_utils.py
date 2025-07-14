"""Unit tests for date_utils module."""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl

from mango_calendar.date_utils import (
    get_holidays_df,
    get_mwc,
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

    def test_get_mwc(self) -> None:
        """Test get_mwc function."""
        result = get_mwc()

        # Check if result is a Polars DataFrame
        self.assertIsInstance(result, pl.DataFrame)

        # Check required columns
        expected_columns = {"datetime", "name", "distance"}
        self.assertTrue(expected_columns.issubset(set(result.columns)))

        # Check that all name values are "MWC"
        self.assertTrue(all(result["name"] == "MWC"))

        # Check that all distance values are 0
        self.assertTrue(all(result["distance"] == 0))

        # Check expected number of entries (should be 42 as per the data)
        self.assertEqual(len(result), 42)

        # Check datetime column type
        self.assertEqual(result["datetime"].dtype, pl.Date)

    def test_get_mwc_date_range(self) -> None:
        """Test that MWC dates cover the expected years."""
        result = get_mwc()

        # Extract years directly from Polars DataFrame
        years = (
            result.select(pl.col("datetime").dt.year().alias("year"))
            .unique()
            .get_column("year")
            .to_list()
        )

        # Should cover years from 2014 to 2024
        self.assertIn(2014, years)
        self.assertIn(2024, years)

        # Should be reasonable number of years
        self.assertGreaterEqual(len(years), 10)

    def test_get_mwc_month_range(self) -> None:
        """Test that MWC dates are in the expected months."""
        result = get_mwc()

        # Extract months directly from Polars DataFrame
        months = (
            result.select(pl.col("datetime").dt.month().alias("month"))
            .unique()
            .get_column("month")
            .to_list()
        )

        # MWC should be in February, March, June, or July
        expected_months = {2, 3, 6, 7}
        self.assertTrue(set(months).issubset(expected_months))

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
