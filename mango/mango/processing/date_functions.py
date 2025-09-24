from datetime import datetime, timedelta, date
from typing import Union, Iterable

import pytz
from mango.logging import get_configured_logger

from .object_functions import as_list

log = get_configured_logger(__name__)


def get_date_from_string(string: str) -> datetime:
    """
    Convert a date string to a datetime object with time set to midnight.

    Parses a string in YYYY-MM-DD format and returns a datetime object
    with the time component set to 00:00:00.

    :param string: Date string in YYYY-MM-DD format
    :type string: str
    :return: Datetime object with time set to midnight
    :rtype: datetime
    :raises ValueError: If the string does not match the expected format

    Example:
        >>> get_date_from_string("2024-01-15")
        datetime.datetime(2024, 1, 15, 0, 0)
    """
    return datetime.strptime(string, "%Y-%m-%d")


def get_date_time_from_string(string: str) -> datetime:
    """
    Convert a datetime string to a datetime object.

    Parses a string in YYYY-MM-DDTHH:MM format and returns a datetime object
    with the specified date and time.

    :param string: Datetime string in YYYY-MM-DDTHH:MM format
    :type string: str
    :return: Datetime object with the parsed date and time
    :rtype: datetime
    :raises ValueError: If the string does not match the expected format

    Example:
        >>> get_date_time_from_string("2024-01-15T14:30")
        datetime.datetime(2024, 1, 15, 14, 30)
    """
    return datetime.strptime(string, "%Y-%m-%dT%H:%M")


def get_date_string_from_ts(ts: datetime) -> str:
    """
    Convert a datetime object to a date string.

    Extracts the date portion from a datetime object and returns it
    as a string in YYYY-MM-DD format.

    :param ts: Datetime object to convert
    :type ts: datetime
    :return: Date string in YYYY-MM-DD format
    :rtype: str

    Example:
        >>> dt = datetime(2024, 1, 15, 14, 30)
        >>> get_date_string_from_ts(dt)
        '2024-01-15'
    """
    return datetime.strftime(ts, "%Y-%m-%d")


def get_date_string_from_ts_string(string: str) -> str:
    """
    Extract the date portion from a datetime string.

    Extracts the first 10 characters (YYYY-MM-DD) from a datetime string
    in YYYY-MM-DDTHH:MM format.

    :param string: Datetime string in YYYY-MM-DDTHH:MM format
    :type string: str
    :return: Date string in YYYY-MM-DD format
    :rtype: str

    Example:
        >>> get_date_string_from_ts_string("2024-01-15T14:30")
        '2024-01-15'
    """
    return string[0:10]


def get_hour_from_date_time(ts: datetime) -> float:
    """
    Get the hour as a decimal number from a datetime object.

    Converts the hour and minute components to a decimal representation
    of hours (e.g., 14:30 becomes 14.5).

    :param ts: Datetime object to extract hours from
    :type ts: datetime
    :return: Hour as a decimal number (e.g., 14.5 for 14:30)
    :rtype: float
    :raises AttributeError: If the object is a date instead of datetime

    Example:
        >>> dt = datetime(2024, 1, 15, 14, 30)
        >>> get_hour_from_date_time(dt)
        14.5
    """
    return float(ts.hour + ts.minute / 60)


def get_hour_from_string(string: str) -> float:
    """
    Get the hour as a decimal number from a datetime string.

    Parses a datetime string and converts the hour and minute components
    to a decimal representation of hours.

    :param string: Datetime string in YYYY-MM-DDTHH:MM format
    :type string: str
    :return: Hour as a decimal number
    :rtype: float
    :raises ValueError: If the string does not match the expected format

    Example:
        >>> get_hour_from_string("2024-01-15T14:30")
        14.5
    """
    return get_hour_from_date_time(get_date_time_from_string(string))


def date_add_weeks_days(
    starting_date: datetime, weeks: int = 0, days: int = 0
) -> datetime:
    """
    Add weeks and days to a datetime object.

    Creates a new datetime object by adding the specified number of weeks
    and days to the starting date.

    :param starting_date: The base datetime object
    :type starting_date: datetime
    :param weeks: Number of weeks to add (default: 0)
    :type weeks: int
    :param days: Number of days to add (default: 0)
    :type days: int
    :return: New datetime object with added time
    :rtype: datetime
    :raises TypeError: If weeks or days are not integers

    Example:
        >>> dt = datetime(2024, 1, 15)
        >>> date_add_weeks_days(dt, weeks=2, days=3)
        datetime.datetime(2024, 2, 1)
    """
    return starting_date + timedelta(days=weeks * 7 + days)


def date_time_add_minutes(date: datetime, minutes: float = 0) -> datetime:
    """
    Add minutes to a datetime object.

    Creates a new datetime object by adding the specified number of minutes
    to the given datetime.

    :param date: The base datetime object
    :type date: datetime
    :param minutes: Number of minutes to add (default: 0)
    :type minutes: float
    :return: New datetime object with added minutes
    :rtype: datetime
    :raises TypeError: If minutes is not a numeric value

    Example:
        >>> dt = datetime(2024, 1, 15, 14, 30)
        >>> date_time_add_minutes(dt, minutes=90.5)
        datetime.datetime(2024, 1, 15, 16, 0, 30)
    """
    return date + timedelta(minutes=minutes)


def get_time_slot_string(ts: datetime) -> str:
    """
    Convert a datetime object to a time slot string.

    Formats a datetime object as a string in YYYY-MM-DDTHH:MM format,
    suitable for time slot representations.

    :param ts: Datetime object to format
    :type ts: datetime
    :return: Formatted datetime string
    :rtype: str

    Example:
        >>> dt = datetime(2024, 1, 15, 14, 30)
        >>> get_time_slot_string(dt)
        '2024-01-15T14:30'
    """
    return datetime.strftime(ts, "%Y-%m-%dT%H:%M")


def get_week_from_string(string: str) -> int:
    """
    Get the ISO week number from a datetime string.

    Parses a datetime string and returns the ISO week number of the year.

    :param string: Datetime string in YYYY-MM-DDTHH:MM format
    :type string: str
    :return: ISO week number (1-53)
    :rtype: int
    :raises ValueError: If the string does not match the expected format

    Example:
        >>> get_week_from_string("2024-01-15T14:30")
        3
    """
    return get_week_from_ts(get_date_time_from_string(string))


def get_week_from_ts(ts: datetime) -> int:
    """
    Get the ISO week number from a datetime object.

    Returns the ISO week number of the year for the given datetime.

    :param ts: Datetime object to extract week number from
    :type ts: datetime
    :return: ISO week number (1-53)
    :rtype: int

    Example:
        >>> dt = datetime(2024, 1, 15)
        >>> get_week_from_ts(dt)
        3
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
    Convert a UTC datetime to a local timezone.

    Transforms a UTC datetime object to the specified timezone.
    The resulting datetime will have the timezone information removed
    (naive datetime) but will represent the local time.

    :param dt: UTC datetime object to convert
    :type dt: datetime
    :param tz: Target timezone name (default: "Europe/Madrid")
    :type tz: str
    :return: Datetime in local timezone (naive)
    :rtype: datetime
    :raises ValueError: If timezone name is invalid

    Example:
        >>> utc_dt = datetime(2024, 1, 15, 12, 0)
        >>> to_tz(utc_dt, "Europe/Madrid")
        datetime.datetime(2024, 1, 15, 13, 0)
    """
    try:
        dt = pytz.utc.localize(dt)
        timezone = pytz.timezone(tz)
        dt = dt.astimezone(timezone)
        return dt.replace(tzinfo=None)
    except Exception as e:
        log.error(f"Error converting timezone: {e}")
        raise


def str_to_dt(string: str, fmt: Union[str, Iterable] = None) -> datetime:
    """
    Convert a string to a datetime object using multiple format attempts.

    Attempts to parse a string into a datetime object by trying various
    standard formats. Additional custom formats can be provided.

    :param string: String to convert to datetime
    :type string: str
    :param fmt: Additional format(s) to try (string or list of strings)
    :type fmt: Union[str, Iterable], optional
    :return: Parsed datetime object
    :rtype: datetime
    :raises ValueError: If no format matches the string

    Example:
        >>> str_to_dt("2024-01-15 14:30:00")
        datetime.datetime(2024, 1, 15, 14, 30)
        >>> str_to_dt("15/01/2024", ["%d/%m/%Y"])
        datetime.datetime(2024, 1, 15)
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

    log.error(f"Failed to parse datetime string: '{string}'")
    raise ValueError(f"string '{string}' does not match any datetime format")


def str_to_d(string: str, fmt: Union[str, Iterable] = None) -> date:
    """
    Convert a string to a date object using multiple format attempts.

    Attempts to parse a string into a date object by trying various
    standard formats. Additional custom formats can be provided.

    :param string: String to convert to date
    :type string: str
    :param fmt: Additional format(s) to try (string or list of strings)
    :type fmt: Union[str, Iterable], optional
    :return: Parsed date object
    :rtype: date
    :raises ValueError: If no format matches the string

    Example:
        >>> str_to_d("2024-01-15")
        datetime.date(2024, 1, 15)
        >>> str_to_d("15/01/2024", ["%d/%m/%Y"])
        datetime.date(2024, 1, 15)
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

    log.error(f"Failed to parse date string: '{string}'")
    raise ValueError(f"string '{string}' does not match any datetime format")


def dt_to_str(dt: Union[date, datetime], fmt: str = None) -> str:
    """
    Convert a date or datetime object to a string.

    Formats a date or datetime object as a string using the specified
    format. If no format is provided, uses the default datetime format.

    :param dt: Date or datetime object to convert
    :type dt: Union[date, datetime]
    :param fmt: Format string for the output (default: "%Y-%m-%d %H:%M:%S")
    :type fmt: str, optional
    :return: Formatted date/datetime string
    :rtype: str

    Example:
        >>> dt = datetime(2024, 1, 15, 14, 30)
        >>> dt_to_str(dt)
        '2024-01-15 14:30:00'
        >>> dt_to_str(dt, "%Y-%m-%d")
        '2024-01-15'
    """
    if fmt is None:
        fmt = DATETIME_FORMATS[0]
    return datetime.strftime(dt, fmt)


def as_datetime(
    x: Union[date, datetime, str], fmt: Union[str, Iterable] = None
) -> datetime:
    """
    Coerce an object to a datetime object.

    Converts various input types (string, date, datetime) to a datetime object.
    For strings, attempts multiple format parsing. For date objects, sets time to midnight.

    :param x: Object to convert (string, date, or datetime)
    :type x: Union[date, datetime, str]
    :param fmt: Additional format(s) to try for string parsing
    :type fmt: Union[str, Iterable], optional
    :return: Datetime object
    :rtype: datetime
    :raises ValueError: If the object cannot be converted to datetime

    Example:
        >>> as_datetime("2024-01-15")
        datetime.datetime(2024, 1, 15, 0, 0)
        >>> as_datetime(date(2024, 1, 15))
        datetime.datetime(2024, 1, 15, 0, 0)
    """
    if isinstance(x, str):
        return str_to_dt(x, fmt)
    elif isinstance(x, datetime):
        return x
    elif isinstance(x, date):
        return datetime(x.year, x.month, x.day)
    else:
        log.error(f"Cannot convert object to datetime: {type(x)}")
        raise ValueError(f"x is not a date: {x}")


def as_date(x: Union[date, datetime, str], fmt: Union[str, Iterable] = None) -> date:
    """
    Coerce an object to a date object.

    Converts various input types (string, date, datetime) to a date object.
    For strings and datetime objects, extracts only the date portion.

    :param x: Object to convert (string, date, or datetime)
    :type x: Union[date, datetime, str]
    :param fmt: Additional format(s) to try for string parsing
    :type fmt: Union[str, Iterable], optional
    :return: Date object
    :rtype: date
    :raises ValueError: If the object cannot be converted to date

    Example:
        >>> as_date("2024-01-15")
        datetime.date(2024, 1, 15)
        >>> as_date(datetime(2024, 1, 15, 14, 30))
        datetime.date(2024, 1, 15)
    """
    if isinstance(x, str):
        dt = str_to_dt(x, fmt)
        return date(dt.year, dt.month, dt.day)
    elif isinstance(x, datetime):
        return date(x.year, x.month, x.day)
    elif isinstance(x, date):
        return x
    else:
        log.error(f"Cannot convert object to date: {type(x)}")
        raise ValueError(f"x is not a date: {x}")


def as_str(x: Union[date, datetime, str], fmt: str = None) -> str:
    """
    Coerce a date-like object to a string.

    Converts date, datetime, or string objects to a formatted string.
    If the input is already a string and a format is specified, attempts
    to parse and reformat it.

    :param x: Object to convert (date, datetime, or string)
    :type x: Union[date, datetime, str]
    :param fmt: Format string for the output
    :type fmt: str, optional
    :return: Formatted string representation
    :rtype: str
    :raises ValueError: If the object cannot be converted to string

    Example:
        >>> as_str(datetime(2024, 1, 15, 14, 30))
        '2024-01-15 14:30:00'
        >>> as_str("2024-01-15", "%Y-%m-%d")
        '2024-01-15'
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
        log.error(f"Cannot convert object to string: {type(x)}")
        raise ValueError(f"x is not a date: {x}")


def add_to_str_dt(
    x: str,
    fmt_in: Union[str, Iterable] = None,
    fmt_out: Union[str, Iterable] = None,
    **kwargs,
):
    """
    Add time to a date/datetime string and return the result as a string.

    Parses a date/datetime string, adds the specified time duration,
    and returns the result as a formatted string.

    :param x: Date/datetime string to modify
    :type x: str
    :param fmt_in: Format(s) for parsing the input string
    :type fmt_in: Union[str, Iterable], optional
    :param fmt_out: Format for the output string
    :type fmt_out: Union[str, Iterable], optional
    :param kwargs: Time duration parameters for timedelta (days, hours, minutes, etc.)
    :return: New date/datetime as a formatted string
    :rtype: str
    :raises ValueError: If the input string cannot be parsed or timedelta parameters are invalid

    Example:
        >>> add_to_str_dt("2024-01-01 05:00:00", hours=2)
        '2024-01-01 07:00:00'
        >>> add_to_str_dt("2024-01-01", days=7, fmt_out="%Y-%m-%d")
        '2024-01-08'
    """
    try:
        if fmt_in:
            fmt_in = as_list(fmt_in) + DATE_FORMATS
        else:
            fmt_in = DATE_FORMATS
        new_date = as_datetime(x, fmt=fmt_in) + timedelta(**kwargs)
        return as_str(new_date, fmt_out)
    except Exception as e:
        log.error(f"Error adding time to datetime string: {e}")
        raise
