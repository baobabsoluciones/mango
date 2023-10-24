from unittest import TestCase
from datetime import date, datetime

from mango.processing.date_functions_2 import (
    str_to_dt,
    dt_to_str,
    as_datetime,
    as_date,
    as_str,
    add_to_str_dt,
    str_to_d,
    to_tz,
)


class DateTests(TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

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
