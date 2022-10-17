from datetime import datetime, timedelta


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
