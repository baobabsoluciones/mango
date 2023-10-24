from unittest import TestCase
from datetime import date, datetime

from mango.processing import (
    get_date_from_string,
    get_date_time_from_string,
    get_date_string_from_ts,
    get_date_string_from_ts_string,
    get_hour_from_date_time,
    get_hour_from_string,
    date_add_weeks_days,
    date_time_add_minutes,
    get_time_slot_string,
    get_week_from_string,
    get_week_from_ts,
)


class DateTests(TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_get_date_from_string(self):
        string = "2022-09-27"
        dt = get_date_from_string(string)
        self.assertIsInstance(dt, datetime)
        self.assertEqual(dt, datetime(2022, 9, 27))

    def test_get_date_from_string_bad_string(self):
        string = "2022-09-274"
        self.assertRaises(ValueError, get_date_from_string, string)

    def test_get_date_time_from_string(self):
        string = "2022-09-27T09:00"
        dt = get_date_time_from_string(string)
        self.assertIsInstance(dt, datetime)
        self.assertEqual(dt, datetime(2022, 9, 27, 9, 0))

    def test_get_date_time_from_string_bad_string(self):
        string = "2022-09-27 09:00"
        self.assertRaises(ValueError, get_date_time_from_string, string)

    def test_get_date_string_from_ts(self):
        dt = datetime(2022, 9, 27, 9, 0, 15)
        string = get_date_string_from_ts(dt)
        self.assertIsInstance(string, str)
        self.assertEqual(string, "2022-09-27")

    def test_get_date_string_from_ts_string(self):
        ts_string = "2022-09-27T09:00"
        date_string = get_date_string_from_ts_string(ts_string)
        self.assertIsInstance(date_string, str)
        self.assertEqual(date_string, "2022-09-27")

    def test_get_date_string_from_ts_string_bad_string(self):
        ts_string = "09:00"
        date_string = get_date_string_from_ts_string(ts_string)
        self.assertIsInstance(date_string, str)
        self.assertNotEqual(date_string, "2022-09-27")

    def test_get_hour_from_ts(self):
        ts = datetime(2022, 9, 27, 9, 30, 15)
        hour = get_hour_from_date_time(ts)
        self.assertEqual(9.5, hour)
        self.assertIsInstance(hour, float)

    def test_get_hour_from_date(self):
        dt = date(2022, 9, 27)
        self.assertRaises(AttributeError, get_hour_from_date_time, dt)

    def test_get_hour_from_string(self):
        ts_string = "2022-09-27T09:30"
        hour = get_hour_from_string(ts_string)
        self.assertEqual(9.5, hour)
        self.assertIsInstance(hour, float)

    def test_get_hour_from_string_bad_string(self):
        ts_string = "2022-09-27 09:30"
        self.assertRaises(ValueError, get_hour_from_string, ts_string)

    def test_date_add_weeks_days(self):
        ts = datetime(2022, 9, 27, 9, 30, 15)
        weeks = 1
        days = 1
        result = date_add_weeks_days(ts, weeks, days)
        self.assertEqual(8, (result - ts).days)

    def test_date_add_weeks_days_bad_weeks(self):
        ts = datetime(2022, 9, 27, 9, 30, 15)
        weeks = "AAA"
        days = 1
        self.assertRaises(TypeError, date_add_weeks_days, ts, weeks, days)

    def test_date_add_weeks_days_bad_days(self):
        ts = datetime(2022, 9, 27, 9, 30, 15)
        weeks = 1
        days = "AAA"
        self.assertRaises(TypeError, date_add_weeks_days, ts, weeks, days)

    def test_date_time_add_minutes(self):
        ts = datetime(2022, 9, 27, 9, 30, 15)
        minutes = 30
        result = date_time_add_minutes(ts, minutes)
        self.assertEqual(30, (result - ts).seconds / 60)

    def test_date_time_add_minutes_bad_minutes(self):
        ts = datetime(2022, 9, 27, 9, 30, 15)
        minutes = "30"
        self.assertRaises(TypeError, date_time_add_minutes, ts, minutes)

    def test_get_time_slot_string(self):
        ts = datetime(2022, 9, 27, 9, 30, 15)
        result = get_time_slot_string(ts)
        self.assertEqual("2022-09-27T09:30", result)
        self.assertIsInstance(result, str)

    def test_get_time_slot_string_date(self):
        ts = date(2022, 9, 27)
        result = get_time_slot_string(ts)
        self.assertEqual("2022-09-27T00:00", result)
        self.assertIsInstance(result, str)

    def test_get_week_from_string(self):
        ts_string = "2022-09-27T09:00"
        result = get_week_from_string(ts_string)
        self.assertEqual(39, result)
        self.assertIsInstance(39, int)

    def test_get_week_from_ts(self):
        ts = datetime(2022, 9, 27, 9, 30, 15)
        result = get_week_from_ts(ts)
        self.assertEqual(39, result)
        self.assertIsInstance(39, int)
