import polars as pl
import re
from mango.logging import log_time
from mango.logging.logger import get_basic_logger

logger = get_basic_logger()


@log_time()
def create_season_unit_previous_date(df, time_unit):
    aux = (
        df.select(["forecast_origin", "season_unit"])
        .unique()
        .sort(["forecast_origin", "season_unit"])
        .with_columns(
            pl.col("forecast_origin").dt.weekday().alias("season_unit_origin")
        )
        .with_columns(
            (pl.col("season_unit") - pl.col("season_unit_origin")).alias("diff")
        )
        .with_columns(
            pl.when(pl.col("diff") > 0)
            .then(7 - pl.col("diff"))
            .otherwise(-pl.col("diff"))
            .alias("diff")
        )
        .with_columns(
            (
                pl.col("forecast_origin").dt.offset_by(
                    ("-" + pl.col("diff").cast(str) + time_unit)
                )
            ).alias("previous_dow_date")
        )
    )

    return aux


@log_time()
def create_recent_variables(
    df: pl.LazyFrame,
    SERIES_CONF: dict,
    window: list,
    lags: list,
    gap: int = 0,
    freq: int = 1,
    colname: str = "",
    season_unit=False,
) -> pl.LazyFrame:
    """
    Create rolling averages for the last window days, excluding the current row,
    but only for rows where horizon = 1. Creating one column for each window, and grouping
    by the key columns in SERIES_CONF.

    :param df: pd.DataFrame
    :param group_cols: list of columns to group by
    :param window: list of integers specifying the rolling window sizes in time unit.
    :param lags: list of integers specifying the lag sizes in time unit.
    :param gap: integer specifying how many previous rows to start the window from, excluding the current row.
    :param colname: string specifying the prefix for the new rolling average columns.
    :return: pd.DataFrame with new columns for each rolling average.
    """

    group_cols = SERIES_CONF["KEY_COLS"]

    if season_unit:
        group_cols = group_cols + ["season_unit"]

    variables_df = (
        df.select(group_cols + ["datetime", "y"])
        .unique()
        .sort(group_cols + ["datetime"])
    )
    # ROLLING AVERAGES
    for w in window:
        rolling_col_name = f"y_{colname}roll_{w}"
        variables_df = variables_df.with_columns(
            pl.col("y")
            .shift(gap)
            .rolling_mean(window_size=w)
            .over(group_cols)
            .alias(rolling_col_name)
        )

    # LAGS
    for l in lags:
        lag_col_name = f"y_{colname}lag_{l*freq}"
        variables_df = variables_df.with_columns(
            pl.col("y").shift(gap - 1 + l).over(group_cols).alias(lag_col_name)
        )

    variables_df = variables_df.drop("y")

    if season_unit:
        aux = create_season_unit_previous_date(
            df, time_unit=re.sub(r"\d", "", SERIES_CONF["TIME_PERIOD"])
        ).select(["forecast_origin", "season_unit", "previous_dow_date"])

        df = df.join(aux, on=["forecast_origin", "season_unit"], how="left")

        # remove season_unit from group_cols
        df_final = df.join(
            variables_df,
            left_on=group_cols + ["previous_dow_date"],
            right_on=group_cols + ["datetime"],
            how="left",
        ).drop(["previous_dow_date"])

    else:
        variables_df = variables_df.rename({"datetime": "forecast_origin"})
        df_final = df.join(
            variables_df, on=group_cols + ["forecast_origin"], how="left"
        )

    return df_final


@log_time()
def create_seasonal_variables(
    df: pl.LazyFrame,
    SERIES_CONF: dict,
    window: list,
    lags: list,
    season_unit: str,
    freq: int,
    gap: int = 0,
) -> pl.LazyFrame:
    """
    Create rolling averages for the last window days, excluding the current row,
    but only for rows where horizon = 1. Creating one column for each window, and grouping
    by the key columns in SERIES_CONF.

    :param df: pd.DataFrame
    :param SERIES_CONF: dictionary with the configuration of the series.
    :param window: list of integers specifying the rolling window sizes in time unit.
    :param lags: list of integers specifying the lag sizes in time unit.
    :param season_unit: string specifying the unit of the seasonality.
    :param freq: integer specifying the frequency of the seasonality.
    :param gap: integer specifying how many previous rows to start the window from, excluding the current row.
    :return: pd.DataFrame with new columns for each rolling average.
    """

    if season_unit == "day":
        df = df.with_columns(pl.col("datetime").dt.weekday().alias("season_unit"))
    elif season_unit == "week":
        df = df.with_columns(pl.col("datetime").dt.week().alias("season_unit"))
    elif season_unit == "month":
        df = df.with_columns(pl.col("datetime").dt.month().alias("season_unit"))

    df = create_recent_variables(
        df,
        SERIES_CONF,
        window,
        lags,
        gap,
        colname="sea_",
        freq=freq,
        season_unit=True,
    )

    # drop season_unit
    df = df.drop("season_unit")

    return df
