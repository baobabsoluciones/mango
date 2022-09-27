from unittest import TestCase

from processing.date_functions import *


class DateTests(TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_get_date_from_string(self):
        string = "2022-09-27"
        date = get_date_from_string(string)
        self.assertIsInstance(date, datetime)
        self.assertEqual(date, datetime(2022, 9, 27))

    def test_get_date_time_from_string(self):
        string = "2022-09-27T09:00"
        date = get_date_time_from_string(string)
        self.assertIsInstance(date, datetime)
        self.assertEqual(date, datetime(2022, 9, 27, 9, 0))

    def test_get_date_string_from_ts(self):
        date = datetime(2022, 9, 27, 9, 0, 15)
        string = get_date_string_from_ts(date)
        self.assertIsInstance(string, str)
        self.assertEqual(string, "2022-09-27")

    def test_get_date_string_from_ts_string(self):
        ts_string = "2022-09-27T09:00"
        date_string = get_date_string_from_ts_string(ts_string)
        self.assertIsInstance(date_string, str)
        self.assertEqual(date_string, "2022-09-27")
