import re

import polars as pl

from mango_time_series.logging import log_time, get_configured_logger

logger = get_configured_logger()


@log_time()
def create_season_unit_previous_date(df: pl.DataFrame, time_unit: str) -> pl.DataFrame:
    """
    Create previous date mapping for seasonal unit analysis.

    Calculates the previous occurrence date for each seasonal unit (day of week)
    relative to the forecast origin. This is used for creating seasonal variables
    that reference the same day of week from previous periods.

    :param df: DataFrame containing forecast_origin and season_unit columns
    :type df: polars.DataFrame
    :param time_unit: Time unit string (e.g., 'd' for days, 'w' for weeks)
    :type time_unit: str
    :return: DataFrame with previous_dow_date column added
    :rtype: polars.DataFrame

    Note:
        - Calculates weekday difference between season_unit and forecast_origin
        - Adjusts for week boundaries (handles day-of-week wrapping)
        - Creates offset dates for seasonal variable creation
    """
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
    season_unit: bool = False,
) -> pl.LazyFrame:
    """
    Create rolling averages and lag variables for time series forecasting.

    Generates rolling average and lag features for time series data, with optional
    seasonal unit grouping. Creates multiple columns for different window sizes
    and lag periods, grouped by the key columns specified in SERIES_CONF.

    :param df: Input LazyFrame containing time series data
    :type df: polars.LazyFrame
    :param SERIES_CONF: Configuration dictionary containing KEY_COLS and TIME_PERIOD
    :type SERIES_CONF: dict
    :param window: List of window sizes for rolling averages
    :type window: list
    :param lags: List of lag periods to create
    :type lags: list
    :param gap: Number of periods to skip before starting window (default: 0)
    :type gap: int
    :param freq: Frequency multiplier for lag calculations (default: 1)
    :type freq: int
    :param colname: Prefix for new column names (default: "")
    :type colname: str
    :param season_unit: Whether to group by seasonal unit (default: False)
    :type season_unit: bool
    :return: LazyFrame with new rolling average and lag columns
    :rtype: polars.LazyFrame

    Note:
        - Rolling averages exclude current row (shifted by gap)
        - Lag variables are shifted by (gap - 1 + lag * freq)
        - Seasonal unit mode uses previous day-of-week dates for alignment
        - Column naming: y_{colname}roll_{window} and y_{colname}lag_{lag*freq}
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
    Create seasonal rolling averages and lag variables.

    Generates seasonal features by creating rolling averages and lag variables
    grouped by seasonal units (day of week, week, or month). Automatically
    extracts the appropriate seasonal unit from datetime and creates variables
    that capture seasonal patterns in the time series.

    :param df: Input LazyFrame containing time series data with datetime column
    :type df: polars.LazyFrame
    :param SERIES_CONF: Configuration dictionary containing KEY_COLS and TIME_PERIOD
    :type SERIES_CONF: dict
    :param window: List of window sizes for rolling averages
    :type window: list
    :param lags: List of lag periods to create
    :type lags: list
    :param season_unit: Seasonal unit type ('day', 'week', or 'month')
    :type season_unit: str
    :param freq: Frequency multiplier for lag calculations
    :type freq: int
    :param gap: Number of periods to skip before starting window (default: 0)
    :type gap: int
    :return: LazyFrame with seasonal rolling average and lag columns
    :rtype: polars.LazyFrame

    Note:
        - Extracts seasonal unit from datetime column based on season_unit parameter
        - Uses ``sea_`` prefix for seasonal variable column names
        - Removes season_unit column after processing
        - Supports day (weekday), week, and month seasonal units
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
