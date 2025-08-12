"""Unit tests for calendar_features module."""

import unittest
import warnings

import pandas as pd

from mango_calendar.calendar_features import get_calendar


class TestCalendarFeatures(unittest.TestCase):
    """Test cases for calendar_features module."""

    def test_get_calendar_basic(self) -> None:
        """Test basic calendar functionality."""
        # Test with minimal parameters
        result = get_calendar(country="ES", start_year=2023, end_year=2024)

        self.assertIsInstance(result, pd.DataFrame)

        expected_columns = {"date", "name", "country_code"}
        self.assertTrue(expected_columns.issubset(set(result.columns)))

        self.assertTrue(all(result["country_code"] == "ES"))

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["date"]))
        self.assertTrue(pd.api.types.is_string_dtype(result["name"]))

    def test_get_calendar_with_communities(self) -> None:
        """Test calendar with communities enabled."""
        result = get_calendar(
            country="ES",
            start_year=2023,
            end_year=2024,
            communities=True,
            return_weights=True,
        )

        self.assertIsInstance(result, pd.DataFrame)

        expected_columns = {"date", "name", "country_code", "weight"}
        self.assertTrue(expected_columns.issubset(set(result.columns)))

        # Check that weights are numeric and reasonable
        self.assertTrue(pd.api.types.is_numeric_dtype(result["weight"]))
        self.assertTrue(all(result["weight"] >= 0))
        self.assertTrue(all(result["weight"] <= 1))

    def test_get_calendar_validation_errors(self) -> None:
        """Test calendar validation errors."""
        with self.assertRaises(ValueError):
            get_calendar(start_year=2025, end_year=2020)

        # Test communities without weights or distances
        with self.assertRaises(ValueError):
            get_calendar(communities=True, return_weights=False, return_distances=False)

    def test_get_calendar_with_distances(self) -> None:
        """Test calendar with distance calculation."""
        distances_config = {"steps_back": 3, "steps_forward": 7}

        result = get_calendar(
            country="ES",
            start_year=2023,
            end_year=2024,
            communities=True,
            return_distances=True,
            distances_config=distances_config,
        )

        self.assertIsInstance(result, pd.DataFrame)

        self.assertIn("distance", result.columns)

        self.assertTrue(all(result["distance"] >= -3))
        self.assertTrue(all(result["distance"] <= 7))

    def test_get_calendar_with_calendar_events(self) -> None:
        """Test calendar with calendar events (Black Friday)."""
        result = get_calendar(
            country="ES", start_year=2023, end_year=2024, calendar_events=True
        )

        black_friday_events = result[result["name"] == "Black Friday"]
        self.assertFalse(black_friday_events.empty)

    def test_warnings_handling(self) -> None:
        """Test that warnings are properly handled."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # This should trigger a warning
            get_calendar(
                country="ES",
                start_year=2023,
                end_year=2024,
                communities=False,
                return_weights=True,
            )

            # Check that a warning was issued
            self.assertEqual(len(w), 1)
            self.assertIn("return_weights", str(w[0].message))


if __name__ == "__main__":
    unittest.main()
