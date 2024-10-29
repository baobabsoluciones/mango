from unittest import TestCase

from mango.data import get_calendar

try:
    import pandas as pd
except ImportError:
    pd = None


class CalendarTests(TestCase):
    def setUp(self):
        self.start_year = 2010
        self.df_calendar = get_calendar(
            calendar_events=True,
            name_transformations=False,
            start_year=self.start_year,
        )

        self.df_calendar_com = get_calendar(
            calendar_events=False,
            name_transformations=True,
            start_year=self.start_year,
            communities=True,
        )

        self.df_calendar_com_pivot = get_calendar(
            calendar_events=False,
            name_transformations=True,
            start_year=self.start_year,
            communities=True,
            pivot=True,
        )

    def tearDown(self):
        pass

    def test_get_national_calendar(self):
        # Instance
        self.assertIsInstance(self.df_calendar, pd.DataFrame)

        # Columns
        self.assertEqual(
            list(self.df_calendar.columns),
            ["date", "name", "country_code"],
        )

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
        # Instance
        self.assertIsInstance(self.df_calendar_com, pd.DataFrame)

        # Columns
        self.assertEqual(
            list(self.df_calendar_com.columns),
            [
                "date",
                "name",
                "country_code",
                "community_code",
                "community_name",
                "weight",
            ],
        )

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

        # Check Exception start_year > end_year
        with self.assertRaises(ValueError):
            get_calendar(start_year=2020, end_year=2013)

    def test_get_calendar_pivot(self):
        # Instance
        self.assertIsInstance(self.df_calendar_com_pivot, pd.DataFrame)

        # Columns
        _ = [
            self.assertIn(name, self.df_calendar_com_pivot.columns)
            for name in self.df_calendar_com.name.unique().tolist()
        ]

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
