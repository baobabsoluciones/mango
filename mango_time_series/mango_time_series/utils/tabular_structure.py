import re

import pandas as pd
import polars as pl
from mango_time_series.logging import log_time, get_configured_logger

logger = get_configured_logger()


@log_time()
def create_tabular_structure(
    df: pd.DataFrame, horizon: int, SERIES_CONF: dict
) -> pd.DataFrame:
    """
    Create tabular structure for time series forecasting with multiple horizons.

    Transforms time series data into a tabular format suitable for machine learning
    by creating all possible combinations of forecast origins and horizons. Each
    row represents a forecast point with its corresponding horizon.

    :param df: DataFrame containing time series data with datetime column
    :type df: pandas.DataFrame
    :param horizon: Maximum forecast horizon to create (e.g., 7 for 7-day ahead forecasts)
    :type horizon: int
    :param SERIES_CONF: Configuration dictionary containing KEY_COLS and TIME_PERIOD settings
    :type SERIES_CONF: dict
    :return: DataFrame with tabular structure including horizon and forecast_origin columns
    :rtype: pandas.DataFrame

    Note:
        - Creates Cartesian product of original data with horizon range (1 to horizon)
        - Calculates forecast_origin by subtracting horizon from datetime
        - Handles different time periods (months vs other units)
        - Sorts result by key columns and datetime
        - Each row represents one forecast point for one horizon
    """

    logger.info("Creating tabular structure")

    # Create a copy of the dataframe
    df = df.copy()
    # range 1 to horizon
    df_horizon = pd.DataFrame({"horizon": range(1, horizon + 1)})

    # grid with all the possible combinations of original df
    df = df.merge(df_horizon, how="cross")
    # sort by KEY_COLS and datetime
    df = df.sort_values(SERIES_CONF["KEY_COLS"] + ["datetime"])

    # CREATE A NEW COLUMN WITH THE DATE OF THE FORECAST_origin
    # datetime - horizon with units from series_conf["TIME_PERIOD"]
    if SERIES_CONF["TIME_PERIOD_DESCR"] == "month":

        df["forecast_origin"] = [
            date - pd.DateOffset(months=months)
            for date, months in zip(df["datetime"], df["horizon"])
        ]
    else:
        df["forecast_origin"] = df["datetime"] - pd.to_timedelta(
            df["horizon"], unit=SERIES_CONF["TIME_PERIOD_PD"]
        )

    return df


@log_time()
def create_tabular_structure_pl(
    df: pl.LazyFrame, horizon: int, SERIES_CONF: dict
) -> pl.LazyFrame:
    """
    Create tabular structure for time series forecasting using Polars.

    Transforms time series data into a tabular format suitable for machine learning
    using Polars LazyFrame for efficient processing. Creates all possible combinations
    of forecast origins and horizons with optimized operations.

    :param df: LazyFrame containing time series data with datetime column
    :type df: polars.LazyFrame
    :param horizon: Maximum forecast horizon to create (e.g., 7 for 7-day ahead forecasts)
    :type horizon: int
    :param SERIES_CONF: Configuration dictionary containing TIME_PERIOD settings
    :type SERIES_CONF: dict
    :return: LazyFrame with tabular structure including horizon and forecast_origin columns
    :rtype: polars.LazyFrame

    Note:
        - Uses Polars cross join for efficient Cartesian product creation
        - Extracts time unit from TIME_PERIOD configuration
        - Calculates forecast_origin using Polars datetime offset operations
        - Maintains lazy evaluation for memory efficiency
        - Each row represents one forecast point for one horizon
    """

    # Logging info
    logger.info("Creating tabular structure")
    time_unit = re.sub(r"\d", "", SERIES_CONF["TIME_PERIOD"])

    # Create a DataFrame with the 'horizon' column ranging from 1 to horizon
    df_horizon = pl.LazyFrame({"horizon": range(1, horizon + 1)})

    # Perform a cross join (Cartesian product)
    df = df.join(df_horizon, how="cross")

    df = df.with_columns(
        [
            (
                pl.col("datetime").dt.offset_by(
                    ("-" + pl.col("horizon").cast(str) + time_unit)
                )
            ).alias("forecast_origin")
        ]
    )

    return df
