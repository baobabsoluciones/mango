from unittest import TestCase

from mango.data import get_calendar

try:
    import pandas as pd
except ImportError:
    pd = None


class CalendarTests(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_national_calendar(self):
        start_year = 2010
        df_calendar = get_calendar(
            calendar_events=True,
            name_transformations=False,
            start_year=start_year,
        )

        # Instance
        self.assertIsInstance(df_calendar, pd.DataFrame)

        # Columns
        self.assertEqual(
            list(df_calendar.columns),
            ["date", "name", "country_code"],
        )

        # Check years
        self.assertEqual(
            df_calendar["date"].dt.year.min(),
            start_year,
        )

        # Check names
        self.assertIn(
            "Black Friday",
            df_calendar["name"].unique(),
        )

        # Check country code
        self.assertEqual(
            df_calendar["country_code"].unique(),
            ["ES"],
        )

        # Check that there are no duplicates in subset date name
        self.assertEqual(
            df_calendar[["date", "name"]].drop_duplicates().shape[0],
            df_calendar.shape[0],
        )

    def test_get_communities_calendar(self):
        start_year = 2013
        df_calendar = get_calendar(
            calendar_events=False,
            name_transformations=True,
            start_year=start_year,
            communities=True,
        )

        # Instance
        self.assertIsInstance(df_calendar, pd.DataFrame)

        # Columns
        self.assertEqual(
            list(df_calendar.columns),
            ["date", "name", "country_code", "community_code", "community_name", "weight"],
        )

        # Check years
        self.assertEqual(
            df_calendar["date"].dt.year.min(),
            start_year,
        )

        # Check names
        self.assertNotIn(
            "Black Friday",
            df_calendar["name"].unique(),
        )

        # Check country code
        self.assertEqual(
            df_calendar["country_code"].unique(),
            ["ES"],
        )

        # Check that there are no duplicates in subset date name community_code
        self.assertEqual(
            df_calendar[["date", "name", "community_code"]].drop_duplicates().shape[0],
            df_calendar.shape[0],
        )

        # Check Black Friday not in name
        self.assertNotIn(
            "Black Friday",
            df_calendar["name"].unique(),
        )

        # Check Exception start_year > end_year
        with self.assertRaises(ValueError):
            get_calendar(start_year=2020, end_year=2013)
