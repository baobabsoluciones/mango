"""Mango Calendar package."""

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("mango-calendar")
except PackageNotFoundError:
    __version__ = "unknown"

from mango_calendar.mango_calendar import (
    calendar_features,
    date_utils,
    get_calendar,
    get_covid_lockdowns,
    get_holidays_df,
)

__all__ = [
    "__version__",
    "calendar_features",
    "date_utils",
    "get_calendar",
    "get_holidays_df",
    "get_covid_lockdowns",
]
