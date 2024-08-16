from unittest import TestCase

from mango_time_series.mango.features.calendar_data import get_calendar
from mango_time_series.mango.tests.const import normalize_path
import pandas as pd

from pandas.testing import assert_frame_equal


class CalendarTests(TestCase):
    def setUp(self):
        self.start_year = 2010
        self.end_year = 2024
        self.steps_back = 5
        self.steps_forward = 6
        self.df_calendar = get_calendar(
            calendar_events=True,
            name_transformations=False,
            start_year=self.start_year,
            end_year=self.end_year,
        )
        self.expected_df_calendar = pd.read_csv(
            normalize_path("data/calendar.csv"), encoding="latin1", parse_dates=["date"]
        )

        self.df_calendar_com = get_calendar(
            calendar_events=False,
            name_transformations=True,
            start_year=self.start_year,
            end_year=self.end_year,
            communities=True,
        )
        self.expected_df_calendar_com = pd.read_csv(
            normalize_path("data/calendar_com.csv"),
            encoding="latin1",
            parse_dates=["date"],
        )

        self.df_calendar_com_pivot = get_calendar(
            calendar_events=False,
            name_transformations=True,
            start_year=self.start_year,
            end_year=self.end_year,
            communities=True,
            pivot=True,
        )
        self.expected_df_calendar_com_pivot = pd.read_csv(
            normalize_path("data/calendar_com_pivot.csv"),
            encoding="latin1",
            parse_dates=["date"],
        )

        self.df_calendar_pivot_keep_com = get_calendar(
            calendar_events=False,
            name_transformations=True,
            start_year=self.start_year,
            end_year=self.end_year,
            communities=True,
            pivot=True,
            pivot_keep_communities=True,
        )

        self.expected_df_calendar_pivot_keep_com = pd.read_csv(
            normalize_path("data/calendar_pivot_keep_com.csv"),
            encoding="latin1",
            parse_dates=["date"],
        )

        self.df_calendar_pivot_keep_com_distances = get_calendar(
            calendar_events=False,
            name_transformations=True,
            start_year=self.start_year,
            end_year=self.end_year,
            communities=True,
            pivot=True,
            return_distances=True,
            distances_config={
                "steps_back": self.steps_back,
                "steps_forward": self.steps_forward,
            },
            return_weights=False,
            pivot_keep_communities=True,
        )
        self.expected_df_calendar_pivot_keep_com_distances = pd.read_csv(
            normalize_path("data/calendar_pivot_keep_com_distances.csv"),
            encoding="latin1",
            parse_dates=["date"],
        )

        self.df_calendar_pivot_keep_com_distances_weights = get_calendar(
            calendar_events=False,
            name_transformations=True,
            start_year=self.start_year,
            end_year=self.end_year,
            communities=True,
            pivot=True,
            return_distances=True,
            distances_config={
                "steps_back": self.steps_back,
                "steps_forward": self.steps_forward,
            },
            return_weights=True,
            pivot_keep_communities=True,
        )

    def tearDown(self):
        pass

    def test_get_national_calendar(self):
        # Frame equal
        assert_frame_equal(self.df_calendar, self.expected_df_calendar)

        # Check years
        self.assertEqual(
            self.df_calendar["date"].dt.year.min(),
            self.start_year,
        )

        # Check names
        self.assertIn(
            "Black Friday",
            self.df_calendar["name"].unique(),
        )

        # Check country code
        self.assertEqual(
            self.df_calendar["country_code"].unique(),
            ["ES"],
        )

        # Check that there are no duplicates in subset date name
        self.assertEqual(
            self.df_calendar[["date", "name"]].drop_duplicates().shape[0],
            self.df_calendar.shape[0],
        )

    def test_get_communities_calendar(self):
        # Frame equal
        assert_frame_equal(self.df_calendar_com, self.expected_df_calendar_com)

        # Check years
        self.assertEqual(
            self.df_calendar_com["date"].dt.year.min(),
            self.start_year,
        )

        # Check names
        self.assertNotIn(
            "Black Friday",
            self.df_calendar_com["name"].unique(),
        )

        # Check country code
        self.assertEqual(
            self.df_calendar_com["country_code"].unique(),
            ["ES"],
        )

        # Check that there are no duplicates in subset date name community_code
        self.assertEqual(
            self.df_calendar_com[["date", "name", "community_code"]]
            .drop_duplicates()
            .shape[0],
            self.df_calendar_com.shape[0],
        )

        # Check Black Friday not in name
        self.assertNotIn(
            "Black Friday",
            self.df_calendar_com["name"].unique(),
        )

    def test_get_calendar_exceptions(self):
        # Check Exception start_year > end_year
        with self.assertRaises(ValueError):
            get_calendar(start_year=2020, end_year=2013)
        # Error
        with self.assertRaises(ValueError):
            get_calendar(
                start_year=2020,
                end_year=2021,
                return_weights=False,
                return_distances=False,
                communities=True,
            )

    def test_get_calendar_pivot(self):
        # Frame equal
        assert_frame_equal(
            self.df_calendar_com_pivot, self.expected_df_calendar_com_pivot
        )

        # Max value
        self.assertEqual(
            self.df_calendar_com_pivot.drop(columns=["date"]).max().max(),
            1,
        )

        # No duplicate dates
        self.assertEqual(
            self.df_calendar_com_pivot.date.drop_duplicates().shape[0],
            self.df_calendar_com_pivot.shape[0],
        )

    def test_get_calendar_pivot_keep_com(self):
        # Assert frame equal
        assert_frame_equal(
            self.df_calendar_pivot_keep_com, self.expected_df_calendar_pivot_keep_com
        )

        # Check that there are no duplicates in subset date name community_code
        self.assertEqual(
            self.df_calendar_pivot_keep_com[["date", "community_name"]]
            .drop_duplicates()
            .shape[0],
            self.df_calendar_pivot_keep_com.shape[0],
        )

        # Check there is no distance_ column and no weight_
        cols = [
            col
            for col in self.df_calendar_pivot_keep_com
            if col.startswith(("weight_", "distance_"))
        ]
        self.assertListEqual(cols, [])
        # When agg by date should give same dataframe as pivot com
        t = (
            self.df_calendar_pivot_keep_com.groupby("date")
            .max(numeric_only=True)
            .reset_index()
        )
        assert_frame_equal(t, self.expected_df_calendar_com_pivot)

    def test_get_calendar_pivot_keep_com_distances(self):
        assert_frame_equal(
            self.df_calendar_pivot_keep_com_distances,
            self.expected_df_calendar_pivot_keep_com_distances,
        )
        # Check there is no distance_ column and no weight_
        cols = [
            col
            for col in self.df_calendar_pivot_keep_com
            if col.startswith(("weight_", "distance_"))
        ]
        self.assertListEqual(cols, [])

        # Check min distance is - self.steps_back and max distance +self.steps_forward
        self.assertEqual(
            -self.df_calendar_pivot_keep_com_distances.min(numeric_only=True).min(),
            self.steps_back,
        )
        self.assertEqual(
            self.df_calendar_pivot_keep_com_distances.max(numeric_only=True).max(),
            self.steps_forward,
        )

    def test_calendar_pivot_keep_com_distances_weights(self):
        # Should be a mix between self.df_calendar_pivot_keep_com and self.df_calendar_pivot_keep_com_distances
        # Add prefix to columns
        t1 = self.expected_df_calendar_pivot_keep_com_distances.copy()
        t2 = self.expected_df_calendar_pivot_keep_com.copy()
        t1.columns = [
            (
                f"distance_{col}"
                if col not in ["date", "country_code", "community_name"]
                else col
            )
            for col in t1.columns
        ]
        t2.columns = [
            (
                f"weight_{col}"
                if col not in ["date", "country_code", "community_name"]
                else col
            )
            for col in t2.columns
        ]

        expected = pd.merge(
            t1, t2, how="left", on=["date", "country_code", "community_name"]
        )
        for col in [col for col in expected.columns if col.startswith("weight_")]:
            expected[col] = expected[col].fillna(0)

        # Columns same order
        expected = (
            expected[self.df_calendar_pivot_keep_com_distances_weights.columns]
            .copy()
            .sort_values("date")
        )

        assert_frame_equal(
            expected, self.df_calendar_pivot_keep_com_distances_weights, check_like=True
        )
