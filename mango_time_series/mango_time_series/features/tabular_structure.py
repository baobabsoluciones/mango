import re

import pandas as pd
import polars as pl

from mango.logging import log_time
from mango.logging.logger import get_basic_logger

logger = get_basic_logger()


@log_time()
def create_tabular_structure(df, horizon, SERIES_CONF):
    """
    Create a tabular structure for a time series dataframe
    :param df: pd.DataFrame
    :param horizon: int
    :param SERIES_CONF: dict
    :return: pd.DataFrame
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
    Create a tabular structure for a time series dataframe using Polars
    :param df: pl.DataFrame
    :param horizon: int
    :param SERIES_CONF: dict
    :return: pl.DataFrame
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
