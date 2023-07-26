import pytz
from datetime import timedelta, datetime
from ..processing import as_list


DATETIME_FORMATS = [
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S.%f" "%Y-%m-%d %H:%M:%S.%f",
]


def to_madrid_tz(datetime):
    """
    Transform a date to local Madrid time.

    :param datetime: a datetime
    :return: datetime in madrid timezone
    """
    timezone = pytz.timezone("Europe/Madrid")
    return datetime + timezone.utcoffset(datetime)


def str_to_dt(str_dt: str, fmt: str = None) -> datetime:
    """
    Transform a string into a datetime object.
    The function will try various standard format.

    :param str_dt: the string
    :param fmt: (list or str) additional format to try on the string.
    :return: datetime
    """
    if fmt is not None:
        formats = as_list(fmt) + DATETIME_FORMATS
    else:
        formats = DATETIME_FORMATS

    for fmt in formats:
        try:
            return datetime.strptime(str_dt, fmt)
        except ValueError:
            continue
    raise ValueError(f"string {str_dt} does not match any datetime format")


def dt_to_str(dt, format=None):
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    if format is not None:
        formats += [format]

    for fmt in formats:
        try:
            return datetime.strftime(dt, fmt)
        except:
            continue
    raise ValueError(f"wrong date format: {dt}")


def as_datetime(x):
    if isinstance(x, str):
        return str_to_dt(x)
    elif isinstance(x, datetime):
        return x
    else:
        raise ValueError(f"dt is not a date: {x}")


def as_str(x):
    if isinstance(x, str):
        return x
    elif isinstance(x, datetime):
        return dt_to_str(x)
    else:
        raise ValueError(f"dt is not a date: {x}")


def add_str_date(date, format="%Y-%m-%d", **kwargs):
    """
    Add time to a date as a string and return the new date as a string.

    :param date: a date as a string (default format:"%Y-%m-%d")
    :param kwargs: kwargs for timedelta (minutes, days, months...)
    :return: the new date as a string (in the same format)
    """
    new_date = datetime.strptime(date, format) + timedelta(**kwargs)
    return datetime.strftime(new_date, format)
