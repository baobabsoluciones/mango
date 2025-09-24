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
    """
    Get national holidays DataFrame with distance window bounds.


    This function retrieves holiday data for a specified country and creates a
    DataFrame with distance calculations around each holiday. It filters holidays
    by weight (>= 0.5), removes duplicates, and pivots the data to have each
    holiday as a separate column with distance values.

    The resulting DataFrame contains:
        - datetime: Date column (as date type)
        - Holiday columns: Each holiday name becomes a column with distance values

    :param steps_back: Number of days to go back from each holiday date
    :type steps_back: int
    :param steps_forward: Number of days to go forward from each holiday date
    :type steps_forward: int
    :param start_year: Starting year for the holiday calendar (default: 2014)
    :type start_year: int
    :param country: ISO 3166-1 alpha-2 country code for holiday calendar (default: "ES")
    :type country: str
    :param output_format: Output format, either 'polars' or 'pandas' (default: "polars")
    :type output_format: str
    :return: DataFrame with holidays as columns and distance values, in specified format
    :rtype: Union[pl.DataFrame, pd.DataFrame]
    :raises ValueError: If output_format is not 'polars' or 'pandas'

    Example:
        >>> # Get Spanish holidays with 7-day window in Polars format
        >>> holidays = get_holidays_df(
        ...     steps_back=7,
        ...     steps_forward=7,
        ...     start_year=2023,
        ...     country="ES",
        ...     output_format="polars"
        ... )
        >>> print(holidays.columns)
        ['datetime', 'New Year', 'Epiphany', 'Good Friday', ...]
        >>>
        >>> # Get holidays in Pandas format
        >>> holidays_pd = get_holidays_df(
        ...     steps_back=3,
        ...     steps_forward=3,
        ...     output_format="pandas"
        ... )
        >>> print(type(holidays_pd))
        <class 'pandas.core.frame.DataFrame'>
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
    """
    Get COVID-19 lockdown period as a Polars DataFrame.

    This function creates a DataFrame representing the COVID-19 lockdown period
    from March 1, 2020 to March 1, 2022. The DataFrame contains daily records
    with the lockdown name and boundary values set to 0.

    The resulting DataFrame contains:
        - ds: Date column (datetime with nanosecond precision)
        - name: Lockdown identifier ("COVID")
        - lower_bound: Lower boundary value (0)
        - upper_bound: Upper boundary value (0)

    :return: Polars DataFrame with COVID lockdown period data
    :rtype: pl.DataFrame

    Example:
        >>> covid_data = get_covid_lockdowns()
        >>> print(covid_data.shape)
        (730, 4)
        >>> print(covid_data.columns)
        ['ds', 'name', 'lower_bound', 'upper_bound']
        >>> print(covid_data['name'].unique())
        ['COVID']
        >>> # Check date range
        >>> print(covid_data['ds'].min())
        2020-03-01 00:00:00
        >>> print(covid_data['ds'].max())
        2022-02-28 00:00:00
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
