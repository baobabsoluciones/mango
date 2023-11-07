from datetime import timedelta, datetime, date
from typing import Iterable, Union

import pytz

from .object_functions import as_list

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
        formats = as_list(fmt) + DATETIME_FORMATS
    else:
        formats = DATETIME_FORMATS

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


def as_str(x, fmt: str = None) -> str:
    """
    Coerce an object to a string.
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
        raise ValueError(f"dt is not a date: {x}")


def add_to_str_dt(
    x,
    fmt_in: Union[str, Iterable] = None,
    fmt_out: Union[str, Iterable] = None,
    **kwargs,
):
    """
    Add time to a datetime as a string and return the new date as a string.

    :param date: a date as a string.
    :param fmt_in: datetime format of the input string (default: "%Y-%m-%d %H:%M:%S").
    :param fmt_out: datetime format of the output string (default: "%Y-%m-%d %H:%M:%S").
    :param kwargs: kwargs for timedelta (minutes, days, months...)
    :return: the new date as a string (in the same format)
    """
    if fmt_in:
        fmt_in = as_list(fmt_in) + DATE_FORMATS
    else:
        fmt_in = DATE_FORMATS
    new_date = as_datetime(x, fmt=fmt_in) + timedelta(**kwargs)
    return as_str(new_date, fmt_out)
