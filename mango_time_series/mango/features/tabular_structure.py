import re

import pandas as pd
import polars as pl

from mango_base.mango.logging import log_time
from mango_base.mango.logging.logger import get_basic_logger
from mango_time_series.mango.features.moving_averages import (
    rolling_recent_averages,
    rolling_seasonal_averages,
)

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


from mango_time_series.mango.tests.example_timeseries_creation_polars import (
    test_timeseries_creation,
)
from mango_time_series.mango.utils.CONST import (
    SERIES_CONFIGURATION as SERIES_CONF,
    PARAMETERS,
)
from mango_time_series.mango.utils.processing import process_time_series


# df = test_timeseries_creation(1000)
df = pd.read_excel(
    r"G:\Unidades compartidas\clece_pmr_202207\proyecto\desarrollo\datos\time_series.xlsx"
)
df["airport"] = "BCN"

df = process_time_series(df, SERIES_CONF)

df = df.with_columns(pl.col("y").fill_null(0))
horizon = 56

## Alternativa con FOR
# all_keys = df.select(SERIES_CONF["KEY_COLS"]).collect().unique()
# df_all = pl.DataFrame()
# for i in all_keys.rows():
#     print("Store: ", i[0], "Product: ", i[1])
#     df_key = df.filter(pl.col("store") == i[0]).filter(pl.col("product") == i[1])
#
#     df_tab = create_tabular_structure_pl(df_key, horizon, SERIES_CONF)
#
#     df_tab = rolling_recent_averages(
#         df_tab, SERIES_CONF["KEY_COLS"], [7, 14, 28], gap=1
#     )
#     df_tab = rolling_seasonal_averages(
#         df_tab, SERIES_CONF["KEY_COLS"], [4], SERIES_CONF["TIME_PERIOD_DESCR"], gap=1
#     )
#
#     df_all = pl.concat([df_all, df_tab.collect()])

## Alternativa entera polars

df_big = create_tabular_structure_pl(df, horizon, SERIES_CONF)

df_big = rolling_recent_averages(df_big, SERIES_CONF["KEY_COLS"], [7, 14, 28], gap=1)
df_big = rolling_seasonal_averages(
    df_big, SERIES_CONF["KEY_COLS"], [4], SERIES_CONF["TIME_PERIOD_DESCR"], gap=1
)

df_all_pl = df_big.collect().sort("datetime")

df_pd = df_all_pl.to_pandas()

from mango_time_series.mango.validation.custom_folds import (
    create_recent_folds,
    create_recent_seasonal_folds,
)

ids = create_recent_folds(df_pd, 56, SERIES_CONF, SERIES_CONF["RECENT_FOLDS"])

sea_ids = create_recent_seasonal_folds(
    df_pd, 56, SERIES_CONF, PARAMETERS, SERIES_CONF["SEASONAL_FOLDS"]
)

df_ids1_tr = df_pd.loc[sea_ids[0][0]]
df_ids1_te = df_pd.loc[sea_ids[0][1]]
df_ids2_tr = df_pd.loc[sea_ids[1][0]]
df_ids2_te = df_pd.loc[sea_ids[1][1]]

min_date_tr1 = df_ids1_tr["datetime"].min()
max_date_tr1 = df_ids1_tr["datetime"].max()
min_date_te1 = df_ids1_te["datetime"].min()
max_date_te1 = df_ids1_te["datetime"].max()
min_date_tr2 = df_ids2_tr["datetime"].min()
max_date_tr2 = df_ids2_tr["datetime"].max()
min_date_te2 = df_ids2_te["datetime"].min()
max_date_te2 = df_ids2_te["datetime"].max()
logger.info("Tabular structure created")
# with colmns duration= end-start


# filter df_pl_sma on rows with horizon = 1
df_tab_filtered = df_all_pl.filter(pl.col("horizon") == 1)
df_pl_sma_sorted = df_tab_filtered.sort(by=SERIES_CONF["KEY_COLS"] + ["datetime"])

print("Olá mundo")
