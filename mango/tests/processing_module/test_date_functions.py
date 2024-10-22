from datetime import date, datetime
from unittest import TestCase

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
    to_tz,
    str_to_dt,
    str_to_d,
    dt_to_str,
    as_datetime,
    as_date,
    as_str,
    add_to_str_dt,
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

    def test_str_to_dt(self):
        """
        Test str_to_dt with various formats.
        :return:
        """
        string_1 = "2022-09-27 8:00:00"
        string_2 = "2022-09-27T8:00:00"
        string_3 = "2022-09-27 8:00"
        string_4 = "2022-09-27 8:00:00.000"
        dt1 = str_to_dt(string_1)
        dt2 = str_to_dt(string_2)
        dt3 = str_to_dt(string_3)
        dt4 = str_to_dt(string_4)
        self.assertIsInstance(dt1, datetime)
        self.assertEqual(dt1, datetime(2022, 9, 27, 8, 0, 0))
        self.assertIsInstance(dt1, datetime)
        self.assertEqual(dt2, datetime(2022, 9, 27, 8, 0, 0))
        self.assertIsInstance(dt2, datetime)
        self.assertEqual(dt3, datetime(2022, 9, 27, 8, 0, 0))
        self.assertIsInstance(dt4, datetime)
        self.assertEqual(dt4, datetime(2022, 9, 27, 8, 0, 0))

    def test_str_to_dt_fmt(self):
        string_1 = "2022-09-27 8h00"
        string_2 = "2022/09/27 8"
        string_3 = "2022-09-27 8:00:00 utc"
        string_4 = "2022-09-27"
        dt1 = str_to_dt(string_1, fmt="%Y-%m-%d %Hh%M")
        dt2 = str_to_dt(string_2, fmt="%Y/%m/%d %H")
        dt3 = str_to_dt(string_3, fmt="%Y-%m-%d %H:%M:%S utc")
        dt4 = str_to_dt(string_4, fmt="%Y-%m-%d")
        self.assertIsInstance(dt1, datetime)
        self.assertEqual(dt1, datetime(2022, 9, 27, 8, 0, 0))
        self.assertIsInstance(dt1, datetime)
        self.assertEqual(dt2, datetime(2022, 9, 27, 8, 0, 0))
        self.assertIsInstance(dt2, datetime)
        self.assertEqual(dt3, datetime(2022, 9, 27, 8, 0, 0))
        self.assertIsInstance(dt4, datetime)
        self.assertEqual(dt4, datetime(2022, 9, 27, 0, 0, 0))

    def test_str_to_d(self):
        """
        Test str_to_dt with various formats.
        :return:
        """
        string_1 = "2022-09-27"
        string_2 = "22/09/27"
        string_3 = "27-09-2022"
        string_4 = "2022-09-27 8:00:00"
        dt1 = str_to_d(string_1)
        dt2 = str_to_d(string_2)
        dt3 = str_to_d(string_3)
        dt4 = str_to_d(string_4)
        self.assertIsInstance(dt1, date)
        self.assertEqual(dt1, date(2022, 9, 27))
        self.assertIsInstance(dt1, date)
        self.assertEqual(dt2, date(2022, 9, 27))
        self.assertIsInstance(dt2, date)
        self.assertEqual(dt3, date(2022, 9, 27))
        self.assertIsInstance(dt4, date)
        self.assertEqual(dt4, date(2022, 9, 27))

    def test_str_to_dt_error(self):
        string_1 = "not a date"
        self.assertRaises(ValueError, str_to_dt, string_1)

    def test_dt_to_str(self):
        dt = datetime(2022, 9, 27, 8, 0, 0)
        default_string = dt_to_str(dt)
        date_string = dt_to_str(dt, fmt="%Y/%m/%d")
        date_time_string = dt_to_str(dt, fmt="%Y-%m-%dT%H:%M:%S")
        year_string = dt_to_str(dt, "%Y")
        self.assertEqual(default_string, "2022-09-27 08:00:00")
        self.assertEqual(date_string, "2022/09/27")
        self.assertEqual(date_time_string, "2022-09-27T08:00:00")
        self.assertEqual(year_string, "2022")

    def test_dt_to_str_2(self):
        dt = date(2022, 9, 27)
        default_string = dt_to_str(dt)
        date_string = dt_to_str(dt, fmt="%Y/%m/%d")
        date_time_string = dt_to_str(dt, fmt="%Y-%m-%dT%H:%M:%S")
        year_string = dt_to_str(dt, "%Y")
        self.assertEqual(default_string, "2022-09-27 00:00:00")
        self.assertEqual(date_string, "2022/09/27")
        self.assertEqual(date_time_string, "2022-09-27T00:00:00")
        self.assertEqual(year_string, "2022")

    def test_as_datetime(self):
        msg = "as_datetime works with strings"
        dt = as_datetime("2022-09-27 08:00:00")
        self.assertEqual(dt, datetime(2022, 9, 27, 8, 0, 0), msg=msg)

    def test_as_datetime_2(self):
        msg = "as_datetime works with date"
        dt = as_datetime(date(2022, 9, 27))
        self.assertEqual(dt, datetime(2022, 9, 27, 0, 0, 0), msg=msg)

    def test_as_datetime_3(self):
        msg = "as_datetime works with datetime"
        dt = as_datetime(datetime(2022, 9, 27, 8, 0, 0))
        self.assertEqual(dt, datetime(2022, 9, 27, 8, 0, 0), msg=msg)

    def test_as_datetime_fmt(self):
        msg = "as_datetime works with strings and format"
        dt = as_datetime("2022-09-27 08h00", "%Y-%m-%d %Hh%M")
        self.assertEqual(dt, datetime(2022, 9, 27, 8, 0, 0), msg=msg)

    def test_as_datetime_error(self):
        """
        as_datetime works get a ValueError if wrong inputs
        """
        self.assertRaises(ValueError, as_datetime, 5)
        self.assertRaises(ValueError, as_datetime, "not a date")
        self.assertRaises(ValueError, as_datetime, ["2023-12-23"])

    def test_as_date(self):
        msg = "as_date works with strings"
        dt = as_date("2022-09-27 08:00:00")
        self.assertEqual(dt, date(2022, 9, 27), msg=msg)

    def test_as_date_2(self):
        msg = "as_dateworks with date"
        dt = as_date(date(2022, 9, 27))
        self.assertEqual(dt, date(2022, 9, 27), msg=msg)

    def test_as_date_3(self):
        msg = "as_date works with datetime"
        dt = as_date(datetime(2022, 9, 27, 8, 0, 0))
        self.assertEqual(dt, date(2022, 9, 27), msg=msg)

    def test_as_date_4(self):
        msg = "as_date works with strings of date only"
        dt = as_date("2022-09-27")
        self.assertEqual(dt, date(2022, 9, 27), msg=msg)

    def test_as_date_fmt(self):
        msg = "as_datetime works with strings and format"
        dt = as_date("2022-09-27 08h00", "%Y-%m-%d %Hh%M")
        self.assertEqual(dt, date(2022, 9, 27), msg=msg)

    def test_as_date_error(self):
        """
        as_date works get a ValueError if wrong inputs
        """
        self.assertRaises(ValueError, as_date, 5)
        self.assertRaises(ValueError, as_date, "not a date")
        self.assertRaises(ValueError, as_date, ["2023-12-23"])

    def test_as_str(self):
        msg = "as_str works with strings"
        dt = as_str("2022-09-27 08:00:00")
        self.assertEqual(dt, "2022-09-27 08:00:00", msg=msg)

    def test_as_str_2(self):
        msg = "as_str works with date"
        dt = as_str(date(2022, 9, 27))
        self.assertEqual(dt, "2022-09-27 00:00:00", msg=msg)

    def test_as_str_3(self):
        msg = "as_str works with datetime"
        dt = as_str(datetime(2022, 9, 27, 8, 0, 0))
        self.assertEqual(dt, "2022-09-27 08:00:00", msg=msg)

    def test_as_str_fmt(self):
        msg = "as_str works with datetime and format"
        dt = as_str(datetime(2022, 9, 27, 8, 0, 0), "%Y-%m")
        self.assertEqual(dt, "2022-09", msg=msg)

    def test_as_str_fmt_2(self):
        msg = "as_str works with string and format"
        dt = as_str("2022-09-27 08:00:00", "%Y-%m")
        self.assertEqual(dt, "2022-09", msg=msg)

    def test_add_to_str_dt(self):
        msg = "add_to_str_date works with default formats"
        dt = add_to_str_dt("2022-09-27 08:00:00", hours=1)
        self.assertEqual(dt, "2022-09-27 09:00:00", msg=msg)

    def test_add_to_str_dt_2(self):
        msg = "add_to_str_date works with input format"
        dt = add_to_str_dt("2022-09-27 08h00", fmt_in="%Y-%m-%d %Hh%M", hours=1)
        self.assertEqual(dt, "2022-09-27 09:00:00", msg=msg)

    def test_add_to_str_dt_3(self):
        msg = "add_to_str_date works with output formats"
        dt = add_to_str_dt("2022-09-27 08:00:00", fmt_out="%Y-%m-%d %Hh%M", hours=1)
        self.assertEqual(dt, "2022-09-27 09h00", msg=msg)

    def test_add_to_str_dt_4(self):
        msg = "add_to_str_date works with dates"
        dt = add_to_str_dt("2022-09-27", hours=1)
        self.assertEqual(dt, "2022-09-27 01:00:00", msg=msg)

    def test_to_tz(self):
        msg = "to_tz should work in summer"
        dt = datetime(2022, 9, 12, 8, 0, 0)
        dt_tz = to_tz(dt)
        self.assertEqual(dt_tz, datetime(2022, 9, 12, 10, 0, 0), msg=msg)

    def test_to_tz_2(self):
        msg = "to_tz should work in winter"
        dt = datetime(2022, 1, 12, 8, 0, 0)
        dt_tz = to_tz(dt)
        self.assertEqual(dt_tz, datetime(2022, 1, 12, 9, 0, 0), msg=msg)

    def test_to_tz_dst_change(self):
        msg = "to_tz should work in hour change"
        dt = datetime(2023, 10, 29, 2, 20, 0)
        dt_tz = to_tz(dt)
        self.assertEqual(dt_tz, datetime(2023, 10, 29, 3, 20, 0), msg=msg)
