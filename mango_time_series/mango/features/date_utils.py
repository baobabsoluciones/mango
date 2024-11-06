from datetime import datetime, timedelta

import polars as pl

from mango_time_series.mango.features.calendar_features import get_calendar


def get_holidays_df(steps_back: int, steps_forward: int) -> pl.DataFrame:
    """
    Get national holidays dataframe with window bounds.

    :param steps_back: int, number of days to go back from the holiday.
    :param steps_forward: int, number of days to go forward
    from the holiday.
    :return: pd.DataFrame, holidays dataframe with window
    """
    start_year = 2014
    all_holidays = get_calendar(
        "ES", start_year=start_year, communities=True, calendar_events=True, return_distances=True, distances_config={"steps_forward": steps_forward, "steps_back": steps_back}
    )
    all_holidays = all_holidays[all_holidays["weight"] >= 0.5].copy()
    all_holidays = all_holidays[["date", "name", "distance"]]
    
    # Drop duplicates
    all_holidays = all_holidays.drop_duplicates(subset=["date", "name"])

    all_holidays = all_holidays.rename(columns={"date": "datetime"})
    all_holidays = pl.from_pandas(all_holidays).with_columns(
        pl.col("datetime").dt.date()
    )
    
    # Pivot with date as index
    all_holidays = all_holidays.pivot(index="datetime", on="name", values="distance")

    return all_holidays


def get_covid_lockdowns() -> pl.DataFrame:
    """
    Get COVID lockdown period as a dataframe.
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


def get_mwc() -> pl.DataFrame:
    """
    Get Mobile World Congress dates as a dataframe.
    :return: pd.DataFrame, Mobile World Congress dates dataframe
    """
    data = {
        "datetime": [
            "2024-02-26",
            "2024-02-27",
            "2024-02-28",
            "2024-02-29",
            "2023-02-27",
            "2023-02-28",
            "2023-03-01",
            "2023-03-02",
            "2022-02-28",
            "2022-03-01",
            "2022-03-02",
            "2022-03-03",
            "2021-06-28",
            "2021-06-29",
            "2021-06-30",
            "2021-07-01",
            "2019-02-25",
            "2019-02-26",
            "2019-02-27",
            "2019-02-28",
            "2018-02-26",
            "2018-02-27",
            "2018-02-28",
            "2018-02-27",
            "2018-02-28",
            "2018-03-01",
            "2017-02-27",
            "2017-02-28",
            "2017-03-01",
            "2017-03-02",
            "2016-02-22",
            "2016-02-23",
            "2016-02-24",
            "2016-02-25",
            "2015-03-02",
            "2015-03-03",
            "2015-03-04",
            "2015-03-05",
            "2014-02-24",
            "2014-02-25",
            "2014-02-26",
            "2014-02-27",
        ],
        "name": ["MWC"] * 42,
        "distance": [0] * 42,
    }
    return pl.DataFrame(data).with_columns(
        pl.col("datetime").dt.date()
    )
