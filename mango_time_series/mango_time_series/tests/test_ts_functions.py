import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
import logging

from mango_time_series.utils.processing_time_series import aggregate_to_input_pl

# Define SERIES_CONFIGURATION here for the test
SERIES_CONF = {
    "KEY_COLS": ["product_id", "store_id"],
    "TIME_COL": "datetime",
    "VALUE_COL": "sales",
    "AGG_OPERATIONS": {"sales": "sum", "quantity": "sum"},
}


class TestAggregateToInput(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame
        data = {
            "datetime": pd.date_range(start="2023-01-01", periods=5, freq="D"),
            "product_id": [1, 1, 1, 2, 2],
            "store_id": [101, 101, 101, 102, 102],
            "sales": [10, 15, 10, 20, 25],
            "quantity": [1, 2, 1, 3, 4],
        }
        self.df = pd.DataFrame(data)

    def test_aggregate_daily(self):
        # Expected output after aggregation
        expected_data = {
            "datetime": pd.date_range(start="2023-01-01", periods=5, freq="D"),
            "product_id": [1, 1, 1, 2, 2],
            "store_id": [101, 101, 101, 102, 102],
            "sales": [10, 15, 10, 20, 25],
            "quantity": [1, 2, 1, 3, 4],
        }
        expected_df = pd.DataFrame(expected_data)

        result_df = aggregate_to_input_pl(self.df, "D", SERIES_CONF)
        # sort columns
        result_df = (
            result_df[expected_df.columns]
            .sort_values(by=["datetime", "product_id", "store_id"])
            .reset_index(drop=True)
        )
        expected_df = expected_df.sort_values(
            by=["datetime", "product_id", "store_id"]
        ).reset_index(drop=True)

        assert_frame_equal(result_df, expected_df)

    def test_aggregate_monthly(self):
        # Expected output after aggregation
        expected_data = {
            "product_id": [1, 2],
            "store_id": [101, 102],
            "datetime": pd.to_datetime(["2023-01-01", "2023-01-01"]),
            "sales": [35, 45],
            "quantity": [4, 7],
        }
        expected_df = pd.DataFrame(expected_data)
        # sort
        expected_df = expected_df.sort_values(
            by=expected_df.columns.to_list()
        ).reset_index(drop=True)

        result_df = aggregate_to_input_pl(self.df, "m", SERIES_CONF)
        # sort
        result_df = result_df[expected_df.columns]
        result_df = result_df.sort_values(by=result_df.columns.to_list()).reset_index(
            drop=True
        )

        assert_frame_equal(result_df, expected_df)
