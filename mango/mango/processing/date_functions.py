from datetime import datetime, timedelta, date
from typing import Union, Iterable

import pytz

from .object_functions import as_list


def get_date_from_string(string: str) -> datetime:
    """
    Returns the datetime object set to 0 horus, 0 minutes and 0 seconds from a string

    :param str string: the datetime string
    :return: the datetime object
    :rtype: :class:`datetime`
    :raises ValueError: if the string does not have the format %Y-%m-%d
    """
    return datetime.strptime(string, "%Y-%m-%d")


def get_date_time_from_string(string: str) -> datetime:
    """
    Returns the datetime object from a string

    :param str string: the datetime string
    :return: the datetime object
    :rtype: :class:`datetime`
    :raises ValueError: if the string does not have the format %Y-%m-%dT%H:%M
    """
    return datetime.strptime(string, "%Y-%m-%dT%H:%M")


def get_date_string_from_ts(ts: datetime) -> str:
    """
    Returns a string representation of the date of a datetime object

    :param ts: the datetime object
    :type ts: :class:`datetime`
    :return: the datetime string
    :rtype: str
    """
    return datetime.strftime(ts, "%Y-%m-%d")


def get_date_string_from_ts_string(string: str) -> str:
    """
    Returns the date part from a datetime string if the datetime string has the following format: %Y-%m-%dT%H:%M

    :param str string: the datetime string
    :return: the date string
    :rtype: str
    """
    return string[0:10]


def get_hour_from_date_time(ts: datetime) -> float:
    """
    Returns the hours (in number) of the given datetime object

    :param ts: the datetime object
    :type ts: :class:`datetime`
    :return: the number of hours
    :rtype: float
    :raises AttributeError: if the passed object is a :class:`date` object instead of a :class:`datetime` object
    """
    return float(ts.hour + ts.minute / 60)


def get_hour_from_string(string: str) -> float:
    """
    Returns the hours (in number) of a given datetime in string format

    :param str string: the datetime string
    :return: the number of hours
    :rtype: float
    :raises ValueError: if the string does not have the format %Y-%m-%dT%H:%M
    """
    return get_hour_from_date_time(get_date_time_from_string(string))


def date_add_weeks_days(
    starting_date: datetime, weeks: int = 0, days: int = 0
) -> datetime:
    """
    Returns a datetime object from a starting date (datetime object) adding a given number of weeks and days

    :param starting_date: the starting datetime object.
    :type starting_date: :class:`datetime`.
    :param int weeks: the number of weeks to add.
    :param int days: the number of days to add.
    :return: the newly datetime object.
    :rtype: :class:`datetime`
    :raises TypeError: if weeks and days is not an integer value
    """
    return starting_date + timedelta(days=weeks * 7 + days)


def date_time_add_minutes(date: datetime, minutes: float = 0) -> datetime:
    """
    Returns a datetime from a date adding minutes

    :param date: the starting datetime object.
    :type date: :class:`datetime`.
    :param float minutes: the number of minutes to add
    :return: the newly datetime object.
    :rtype: :class:`datetime`.
    :raises TypeError: if minutes is not a float value
    """
    return date + timedelta(minutes=minutes)


def get_time_slot_string(ts: datetime) -> str:
    """
    Returns a string representing the datetime object with the following format: %Y-%m-%dT%H:%M

    :param ts: the datetime object
    :type ts: :class:`datetime`
    :return: the string
    :rtype: str
    """
    return datetime.strftime(ts, "%Y-%m-%dT%H:%M")


def get_week_from_string(string: str) -> int:
    """
    Returns the week number from a datetime string

    :param str string: the datetime string
    :return: the week number
    :rtype: int
    """
    return get_week_from_ts(get_date_time_from_string(string))


def get_week_from_ts(ts: datetime) -> int:
    """
    Returns the week number from a datetime object

    :param ts: the datetime object
    :type ts: :class:`datetime`
    :return: the week number
    :rtype: int
    """
    return ts.isocalendar()[1]


DATETIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S.%f",
]
DATE_FORMATS = ["%Y-%m-%d", "%Y/%m/%d", "%y-%m-%d", "%y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"]


def to_tz(dt: datetime, tz: str = "Europe/Madrid") -> datetime:
    """
    Transform an utc date to local timezone time (default Europe/Madrid).

    :param dt: a datetime
    :param tz: timezone name
    :return: datetime in local timezone
    """
    dt = pytz.utc.localize(dt)
    timezone = pytz.timezone(tz)
    dt = dt.astimezone(timezone)
    return dt.replace(tzinfo=None)


def str_to_dt(string: str, fmt: Union[str, Iterable] = None) -> datetime:
    """
    Transform a string into a datetime object.
    The function will try various standard format.

    :param string: the string
    :param fmt: (list or str) additional format to try on the string.
    :return: datetime
    """
    if fmt is not None:
        formats = as_list(fmt) + DATETIME_FORMATS + DATE_FORMATS
    else:
        formats = DATETIME_FORMATS + DATE_FORMATS

    for fmt in formats:
        try:
            return datetime.strptime(string, fmt)
        except ValueError:
            continue
    raise ValueError(f"string '{string}' does not match any datetime format")


def str_to_d(string: str, fmt: Union[str, Iterable] = None) -> date:
    """
    Transform a string into a date object.
    The function will try various standard format.

    :param string: the string
    :param fmt: (list or str) additional format to try on the string.
    :return: date
    """
    if fmt is not None:
        formats = as_list(fmt) + DATE_FORMATS
    else:
        formats = DATE_FORMATS + DATETIME_FORMATS

    for fmt in formats:
        try:
            dt = datetime.strptime(string, fmt)
            return date(dt.year, dt.month, dt.day)
        except ValueError:
            continue
    raise ValueError(f"string '{string}' does not match any datetime format")


def dt_to_str(dt: Union[date, datetime], fmt: str = None) -> str:
    """
    Transform a date or datetime object into a string.

    :param dt: datetime
    :param fmt: string format
    :return: str
    """
    if fmt is None:
        fmt = DATETIME_FORMATS[0]
    return datetime.strftime(dt, fmt)


def as_datetime(
    x: Union[date, datetime, str], fmt: Union[str, Iterable] = None
) -> datetime:
    """
    Coerce an object into a datetime object if possible.

    :param x: an object (string, date or datetime)
    :param fmt: (list or str) additional format to try on the string.
    :return: datetime
    """
    if isinstance(x, str):
        return str_to_dt(x, fmt)
    elif isinstance(x, datetime):
        return x
    elif isinstance(x, date):
        return datetime(x.year, x.month, x.day)
    else:
        raise ValueError(f"x is not a date: {x}")


def as_date(x: Union[date, datetime, str], fmt: Union[str, Iterable] = None) -> date:
    """
    Coerce an object into a datetime object if possible.

    :param x: an object (string, date or datetime)
    :param fmt: (list or str) additional format to try on the string.
    :return: date
    """
    if isinstance(x, str):
        dt = str_to_dt(x, fmt)
        return date(dt.year, dt.month, dt.day)
    elif isinstance(x, datetime):
        return date(x.year, x.month, x.day)
    elif isinstance(x, date):
        return x
    else:
        raise ValueError(f"x is not a date: {x}")


def as_str(x: Union[date, datetime, str], fmt: str = None) -> str:
    """
    Coerce a date like object to a string.
    If a format is given try to return a string in this format (even if it was already a string).

    :param x: object (date, str or datetime)
    :param fmt: datetime format
    :return: str
    """
    if isinstance(x, str):
        try:
            dt = as_datetime(x, fmt)
            return dt_to_str(dt, fmt)
        except ValueError:
            return x
    elif isinstance(x, date):
        return dt_to_str(x, fmt)
    else:
        raise ValueError(f"x is not a date: {x}")


def add_to_str_dt(
    x: str,
    fmt_in: Union[str, Iterable] = None,
    fmt_out: Union[str, Iterable] = None,
    **kwargs,
):
    """
    Add time to a date or datetime as a string and return the new datetime as a string.
    Example::

        result = add_to_str_dt("2024-01-01 05:00:00", hours=2)
        # result = "2024-01-01 07:00:00"

    :param x: a date as a string.
    :param fmt_in: datetime format of the input string (default: "%Y-%m-%d %H:%M:%S").
    :param fmt_out: datetime format of the output string (default: "%Y-%m-%d %H:%M:%S").
    :param kwargs: kwargs for timedelta (minutes, days, months...)
    :return: the new datetime as a string (in the same format)
    """
    if fmt_in:
        fmt_in = as_list(fmt_in) + DATE_FORMATS
    else:
        fmt_in = DATE_FORMATS
    new_date = as_datetime(x, fmt=fmt_in) + timedelta(**kwargs)
    return as_str(new_date, fmt_out)
