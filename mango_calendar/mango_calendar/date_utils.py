"""Utility functions for working with dates and calendars."""

from datetime import datetime, timedelta
from typing import Union

import pandas as pd
import polars as pl

from .calendar_features import get_calendar


def get_holidays_df(
    steps_back: int,
    steps_forward: int,
    start_year: int = 2014,
    country: str = "ES",
    output_format: str = "polars",
) -> Union[pl.DataFrame, pd.DataFrame]:
    """Get national holidays dataframe with window bounds

    :param steps_back: int, number of days to go back from the holiday.
    :param steps_forward: int, number of days to go forward from the holiday.
    :param start_year: int, start year for the holiday calendar.
    :param country: str, country code for holiday calendar.
    :param output_format: str, 'polars' or 'pandas' to specify the output format.
    :return: polars.DataFrame or pandas.DataFrame, holiday dataframe with window bounds.
    """
    if output_format not in ["polars", "pandas"]:
        raise ValueError(
            f"output_format should be either 'polars' or 'pandas', got {output_format}"
        )

    # Fetch the holidays data with specified parameters
    all_holidays = get_calendar(
        country,
        start_year=start_year,
        communities=True,
        calendar_events=True,
        return_distances=True,
        distances_config={"steps_forward": steps_forward, "steps_back": steps_back},
    )

    # Filter, clean, and process the data
    all_holidays = all_holidays[all_holidays["weight"] >= 0.5].copy()
    all_holidays = all_holidays[["date", "name", "distance"]].drop_duplicates(
        subset=["date", "name"]
    )
    all_holidays = all_holidays.rename(columns={"date": "datetime"})

    # Convert to Polars DataFrame and adjust date column
    all_holidays = pl.from_pandas(all_holidays).with_columns(
        pl.col("datetime").dt.date()
    )

    all_holidays = all_holidays.pivot(
        index="datetime",
        columns="name",
        values="distance",
    )

    if output_format == "pandas":
        return all_holidays.to_pandas()

    return all_holidays


def get_covid_lockdowns() -> pl.DataFrame:
    """Get COVID lockdown period as a dataframe

    :return: pd.DataFrame, COVID lockdown period dataframe
    """
    covid_start = datetime(2020, 3, 1)
    covid_end = datetime(2022, 3, 1)

    date_range = [
        covid_start + timedelta(i) for i in range(0, (covid_end - covid_start).days)
    ]

    covid_df = pl.DataFrame(
        {"ds": date_range, "name": "COVID", "lower_bound": 0, "upper_bound": 0}
    )

    return covid_df.with_columns(
        pl.col("ds").dt.cast_time_unit("ns"),
        pl.col("lower_bound").cast(pl.Int64),
        pl.col("upper_bound").cast(pl.Int64),
    )
