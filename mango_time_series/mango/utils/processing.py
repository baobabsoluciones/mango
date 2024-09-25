import pandas as pd
import polars as pl

from mango_base.mango.logging import log_time
from mango_base.mango.logging.logger import get_basic_logger
from mango_time_series.mango.utils.processing_time_series import (
    create_dense_data_pl,
)

logger = get_basic_logger()


@log_time()
def aggregate_to_input(df: pd.DataFrame, freq: str, SERIES_CONF: dict) -> pd.DataFrame:
    """
    Aggregate data to the frequency defined in the input
    :param df: pd.DataFrame with the sales information
    :param freq: str with the frequency to aggregate the data
    :param group_cols: list with the columns to group by
    :return: pd.DataFrame
    """

    logger.info(f"Aggregating data to: {freq}")

    list_grouper = SERIES_CONF["KEY_COLS"] + [pd.Grouper(key="datetime", freq=freq)]

    # group by KEY_COLS
    df = df.groupby(list_grouper).agg(SERIES_CONF["AGG_OPERATIONS"]).reset_index()

    return df


@log_time()
def aggregate_to_input_pl(
    df: pd.DataFrame, freq: str, SERIES_CONF: dict
) -> pd.DataFrame:
    """
    Aggregate data to the frequency defined in the input
    :param df: pd.DataFrame with the sales information
    :param freq: str with the frequency to aggregate the data
    :param group_cols: list with the columns to group by
    :return: pd.DataFrame
    """

    logger.info(f"Aggregating data to: {freq}")
    # if freq is "m" or "MS" or "ME" freq_input = "mo"
    group_cols = SERIES_CONF["KEY_COLS"]
    agg_ops = SERIES_CONF["AGG_OPERATIONS"]

    if freq in ["m", "MS", "ME"]:
        freq_input = "1" + "mo"
    else:
        freq_input = "1" + freq.lower()

    def transform_agg_operations(agg_operations):
        operation_mapping = {
            "sum": lambda col: pl.col(col).sum(),
            "mean": lambda col: pl.col(col).mean(),
            "min": lambda col: pl.col(col).min(),
            "max": lambda col: pl.col(col).max(),
            "median": lambda col: pl.col(col).median(),
            # Add more operations as needed
        }
        transformed_operations = [
            operation_mapping[op](col) for col, op in agg_operations.items()
        ]
        return transformed_operations

    df_pl = pl.from_pandas(df.copy())

    df_agg = df_pl.group_by(
        group_cols + [pl.col("datetime").dt.truncate(freq_input)]
    ).agg(transform_agg_operations(agg_ops))

    return df_agg.to_pandas()


@log_time()
def aggregate_to_input_pllazy(
    df: pl.LazyFrame, freq: str, SERIES_CONF: dict
) -> pl.LazyFrame:
    """
    Aggregate data to the frequency defined in the input
    :param df: pd.DataFrame with the sales information
    :param freq: str with the frequency to aggregate the data
    :param SERIES_CONF: dict with the configuration of the series
    :return: pd.DataFrame
    """

    logger.info(f"Aggregating data to: {freq}")
    # if freq is "m" or "MS" or "ME" freq_input = "mo"
    group_cols = SERIES_CONF["KEY_COLS"]
    agg_ops = SERIES_CONF["AGG_OPERATIONS"]

    freq_input = "1" + freq.lower()

    def transform_agg_operations(agg_operations):
        operation_mapping = {
            "sum": lambda col: pl.col(col).sum(),
            "mean": lambda col: pl.col(col).mean(),
            "min": lambda col: pl.col(col).min(),
            "max": lambda col: pl.col(col).max(),
            "median": lambda col: pl.col(col).median(),
            # Add more operations as needed
        }
        transformed_operations = [
            operation_mapping[op](col) for col, op in agg_operations.items()
        ]
        return transformed_operations

    if isinstance(df, pl.DataFrame):
        df = df.lazy()

    df_agg = df.group_by(group_cols + [pl.col("datetime").dt.truncate(freq_input)]).agg(
        transform_agg_operations(agg_ops)
    )

    df_agg = df_agg.sort(group_cols + ["datetime"])

    return df_agg


def rename_to_common_ts_names(
    df: pd.DataFrame, time_col: str, value_col: str
) -> pd.DataFrame:
    """
    Rename columns to common time series names
    :param df: pd.DataFrame
    :param time_col: str with the name of the time column
    :param value_col: str with the name of the value column
    :return: pd.DataFrame
    """
    logger.info("Renaming columns to common time series names")

    # rename columns
    df = df.rename(columns={time_col: "datetime", value_col: "y"})

    # all columns to lowercase
    df.columns = df.columns.str.lower()

    return df


def rename_to_common_ts_names_pl(
    df: pl.LazyFrame, time_col: str, value_col: str
) -> pl.LazyFrame:
    """
    Rename columns to common time series names
    :param df: pl.DataFrame
    :param time_col: str with the name of the time column
    :param value_col: str with the name of the value column
    :return: pl.DataFrame
    """
    # Rename columns
    df = df.rename({time_col: "datetime", value_col: "y"})

    # Convert all column names to lowercase
    df = df.with_columns(
        [pl.col(c).alias(c.lower()) for c in df.collect_schema().names()]
    )

    return df


def drop_negative_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with negative sales
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """
    logger.info("Dropping rows with negative sales")

    # find rows with negative sales
    mask = df["y"] < 0
    logger.info(f"Dropping {mask.sum()} rows with negative sales")

    # drop rows with negative sales
    df = df[~mask]

    return df


def drop_negative_output_pl(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Drop rows with negative sales
    :param df: pl.LazyFrame
    :return: pl.LazyFrame
    """
    logger.info("Dropping rows with negative sales")

    # Add a lazy count of the negative rows
    negative_sales_count = (
        df.filter(pl.col("y") < 0).select(pl.count()).collect().item()
    )
    logger.info(f"Dropping {negative_sales_count} rows with negative sales")

    # Drop rows with negative sales in lazy mode
    df = df.filter(pl.col("y") >= 0)

    return df


@log_time()
def add_covid_mark(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add a column to mark the COVID period
    :param df: pl.LazyFrame with the sales information
    :return: pl.LazyFrame
    """
    logger.info("Adding COVID mark")

    # Define the start and end of the COVID period
    covid_start = pl.lit("2020-03-01").str.strptime(pl.Date, format="%Y-%m-%d")
    covid_end = pl.lit("2021-03-01").str.strptime(pl.Date, format="%Y-%m-%d")

    # Add the COVID mark as a new column
    df = df.with_columns(
        ((pl.col("datetime") >= covid_start) & (pl.col("datetime") < covid_end)).alias(
            "covid"
        )
    )

    return df


from typing import List


def create_lags_col(
    df: pd.DataFrame,
    col: str,
    lags: List[int],
    key_cols: List[str] = None,
    check_col: List[str] = None,
) -> pd.DataFrame:
    """
    The create_lags_col function creates lagged columns for a given dataframe.
    The function takes four arguments: df, col, lags, and key_cols.
    The df argument is the dataframe to which we want to add lagged columns.
    The col argument is the name of the column in the dataframe that we want to create lag variables for (e.g., 'units_value').
    The lags argument should be a list of integers representing how far back in time we want to shift our new lag features (e.g., [3, 6] would create two new lag features).
    The key_cols argument should be a list of columns that define each time series in the dataframe (e.g., ['prod_cod', 'country']).

    :param pd.DataFrame df: The dataframe to be manipulated.
    :param str col: The column to create lags for.
    :param list[int] lags: The list of lag values to create.
    :param list[str] key_cols: The list of columns that define each time series.
    :param list[str] check_col: The list of columns to check for discontinuities in the lagged series.
    :return: A dataframe with the lagged columns added.
    """

    # Ensure pandas and numpy are imported
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        raise ImportError("pandas and numpy need to be installed to use this function")

    # Make a copy of the input dataframe
    df_c = df.copy()

    # Group the dataframe by the key columns, if provided
    if key_cols is not None:
        grouped_df = df_c.groupby(key_cols)
    else:
        grouped_df = [("", df_c)]

    # Loop through the lag values
    for lag in lags:
        if lag != 0:  # Only create lags for non-zero lag values
            # Define the column name for the lag
            lag_col_name = (
                f"{col}_lag{abs(lag)}" if lag > 0 else f"{col}_lead{abs(lag)}"
            )

            # Apply the shift operation based on the lag value for each group
            for _, group in grouped_df:
                df_c.loc[group.index, lag_col_name] = group[col].shift(lag)

            # If a check column is provided, set lag values to NaN if there is a discontinuity
            if check_col is not None:
                for check in check_col:
                    if type(check) is list:
                        for c in check:
                            mask = df_c.groupby(key_cols)[check].transform(
                                lambda x: x != x.shift(lag)
                            )
                            df_c.loc[mask, lag_col_name] = np.nan
                    else:
                        mask = df_c.groupby(key_cols)[check].transform(
                            lambda x: x != x.shift(lag)
                        )
                        df_c.loc[mask, lag_col_name] = np.nan

    return df_c


def series_as_columns(df, SERIES_CONF):
    """
    Pivot the dataframe to have the series as columns
    :param df: pd.DataFrame
    :param key_cols: list with the columns to group by
    :param value_col: str with the name of the column to pivot
    :return: pd.DataFrame
    """
    logger.info("Pivoting the dataframe")
    key_cols = SERIES_CONF["KEY_COLS"]
    value_col = SERIES_CONF["VALUE_COL"]

    # pivot the dataframe
    pivot_df = df.pivot_table(
        index="datetime", columns=key_cols, values=value_col, aggfunc="sum"
    ).reset_index()

    # Flatten the multi-level columns and rename them
    pivot_df.columns = pivot_df.columns.map(lambda x: f"{x[0]}_{x[1]}")
    # reanme first column to datetime
    pivot_df = pivot_df.rename(columns={pivot_df.columns[0]: "datetime"})

    return pivot_df


def series_as_rows(df, SERIES_CONF):
    """
    Pivot the dataframe to have the series as columns
    :param df: pd.DataFrame
    :param key_cols: list with the columns to group by
    :param value_col: str with the name of the column to pivot
    :return: pd.DataFrame
    """
    logger.info("Pivoting the dataframe")
    value_col = SERIES_CONF["VALUE_COL"]
    key_cols = SERIES_CONF["KEY_COLS"]

    long_df = df.melt(
        id_vars="datetime",  # The columns to keep as is
        var_name="id",  # Name for the new identifier variable
        value_name=value_col,  # Name for the new values variable
    )
    # separate the id column into the key_cols taking into account the "_" as separator
    long_df[key_cols] = long_df["id"].str.split("_", expand=True)

    # drop the id column
    long_df = long_df.drop("id", axis=1)

    # arrange df datetime, key_cols, y
    long_df = long_df[["datetime"] + key_cols + [value_col]]

    return long_df


def process_time_series(
    df,
    SERIES_CONF,
    # add_kwargs
):
    # if df is a pd.DataFrame then convert to pl.DataFrame
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df).lazy()
    if isinstance(df, pl.DataFrame):
        df = df.lazy()

    # df = adapt_columns_cl(df)
    df = rename_to_common_ts_names_pl(
        df,
        time_col=SERIES_CONF["TIME_COL"],
        value_col=SERIES_CONF["VALUE_COL"],
    )

    # get_basic_stats_from_data(df)
    df = drop_negative_output_pl(df)
    df = aggregate_to_input_pllazy(df, "d", SERIES_CONF)

    df = create_dense_data_pl(
        df=df,
        id_cols=SERIES_CONF["KEY_COLS"],
        freq="d",
        min_max_by_id=True,
        date_end="2024-11-03",
        time_col="datetime",
    )

    if SERIES_CONF["TS_PARAMETERS"]["agg"] != "d":
        df = aggregate_to_input_pllazy(
            df, SERIES_CONF["TS_PARAMETERS"]["agg"], SERIES_CONF
        )

    df = add_covid_mark(df)

    return df
