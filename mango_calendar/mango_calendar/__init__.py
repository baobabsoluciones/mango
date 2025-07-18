"""Mango Calendar package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mango-calendar")
except PackageNotFoundError:
    __version__ = "unknown"

from .calendar_features import get_calendar
from .date_utils import get_covid_lockdowns, get_holidays_df

__all__ = ["__version__", "get_calendar", "get_holidays_df", "get_covid_lockdowns"]
