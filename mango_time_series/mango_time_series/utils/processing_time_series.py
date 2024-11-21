import logging
import warnings
from datetime import datetime
from typing import Dict, List, Union

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import polars as pl
except ImportError:
    pl = None

from mango.logging import log_time
from mango.logging.logger import get_basic_logger

logger = get_basic_logger()


@log_time()
def aggregate_to_input(df: pd.DataFrame, freq: str, series_conf: dict) -> pd.DataFrame:
    """
    Aggregate data to the frequency defined in the input
    :param df: pd.DataFrame with the sales information
    :param freq: str with the frequency to aggregate the data
    :param series_conf: dict with the configuration of the series
    :return: pd.DataFrame
    """

    logger.info(f"Aggregating data to: {freq}")

    list_grouper = series_conf["KEY_COLS"] + [pd.Grouper(key="datetime", freq=freq)]

    # group by KEY_COLS
    df = df.groupby(list_grouper).agg(series_conf["AGG_OPERATIONS"]).reset_index()

    return df


@log_time()
def aggregate_to_input_pl(
    df: pd.DataFrame, freq: str, series_conf: dict
) -> pd.DataFrame:
    """
    Aggregate data to the frequency defined in the input
    :param df: pd.DataFrame with the sales information
    :param freq: str with the frequency to aggregate the data
    :param series_conf: dict with the configuration of the series.
    :return: pd.DataFrame
    """

    logger.info(f"Aggregating data to: {freq}")
    # if freq is "m" or "MS" or "ME" freq_input = "mo"
    group_cols = series_conf["KEY_COLS"]
    agg_ops = series_conf["AGG_OPERATIONS"]

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
    df: pl.LazyFrame, freq: str, series_conf: dict
) -> pl.LazyFrame:
    """
    Aggregate data to the frequency defined in the input
    :param df: pd.DataFrame with the sales information
    :param freq: str with the frequency to aggregate the data
    :param series_conf: dict with the configuration of the series
    :return: pd.DataFrame
    """

    logger.info(f"Aggregating data to: {freq}")
    # if freq is "m" or "MS" or "ME" freq_input = "mo"
    group_cols = series_conf["KEY_COLS"]
    agg_ops = series_conf["AGG_OPERATIONS"]

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


def series_as_columns(df, series_conf):
    """
    Pivot the dataframe to have the series as columns
    :param df: pd.DataFrame
    :param series_conf: dict with the configuration of the series
    :return: pd.DataFrame
    """
    logger.info("Pivoting the dataframe")
    key_cols = series_conf["KEY_COLS"]
    value_col = series_conf["VALUE_COL"]

    # pivot the dataframe
    pivot_df = df.pivot_table(
        index="datetime", columns=key_cols, values=value_col, aggfunc="sum"
    ).reset_index()

    # Flatten the multi-level columns and rename them
    pivot_df.columns = pivot_df.columns.map(lambda x: f"{x[0]}_{x[1]}")
    # reanme first column to datetime
    pivot_df = pivot_df.rename(columns={pivot_df.columns[0]: "datetime"})

    return pivot_df


def series_as_rows(df, series_conf):
    """
    Pivot the dataframe to have the series as columns
    :param df: pd.DataFrame
    :param series_conf: dict with the configuration of the series
    :return: pd.DataFrame
    """
    logger.info("Pivoting the dataframe")
    value_col = series_conf["VALUE_COL"]
    key_cols = series_conf["KEY_COLS"]

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
    df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    series_conf: Dict,
):
    """
    Process the time series data

    :param df: pd.DataFrame with the sales information
    :param series_conf: dict with the configuration of the series
    :return: pd.DataFrame
    """
    # if df is a pd.DataFrame then convert to pl.DataFrame
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df).lazy()
    if isinstance(df, pl.DataFrame):
        df = df.lazy()

    # df = adapt_columns_cl(df)
    df = rename_to_common_ts_names_pl(
        df,
        time_col=series_conf["TIME_COL"],
        value_col=series_conf["VALUE_COL"],
    )

    # get_basic_stats_from_data(df)
    df = drop_negative_output_pl(df)
    df = aggregate_to_input_pllazy(df, "d", series_conf)

    df = create_dense_data_pl(
        df=df,
        id_cols=series_conf["KEY_COLS"],
        freq="d",
        min_max_by_id=True,
        date_end="2024-10-01",
        time_col="datetime",
    )

    if series_conf["TS_PARAMETERS"]["agg"] != "d":
        df = aggregate_to_input_pllazy(
            df, series_conf["TS_PARAMETERS"]["agg"], series_conf
        )

    df = add_covid_mark(df)

    return df


@log_time()
def create_dense_data(
    df: pd.DataFrame,
    id_cols,
    freq: str,
    min_max_by_id: bool = None,
    date_init=None,
    date_end=None,
    time_col: str = "timeslot",
) -> pd.DataFrame:
    """
    Create a dense dataframe with a frequency of freq, given range of dates or inherited from the dataframe,
     using the id_cols as keys.
    :param df: dataframe to be expanded
    :param id_cols: list of columns to be used as keys
    :param freq: frequency of the new dataframe
    :param min_max_by_id: boolean to indicate if the range of dates is the min and max of the dataframe by id
    :param date_init: if it has a value, all initial dates will be set to this value
    :param date_end: if it has a value, all final dates will be set to this value
    :param time_col: string with name of the column with the time information
    :return: dataframe with all the dates using the frequency freq
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas need to be installed to use this function")
    df_w = df.copy()

    # get cols id_cols from df and drop duplicates
    df_id = df_w[id_cols].drop_duplicates()

    # if min_max_by_id is True, get the min and max of the time_col by id_cols
    if min_max_by_id:
        df_min_max = (
            df_w.groupby(id_cols, dropna=False)
            .agg({time_col: ["min", "max"]})
            .reset_index()
        )
        df_min_max.columns = id_cols + ["min_date", "max_date"]

        if date_init is not None:
            df_min_max["min_date"] = date_init

        if date_end is not None:
            df_min_max["max_date"] = date_end

        grid_min_date = df_min_max["min_date"].min()
        grid_max_date = df_min_max["max_date"].max()

    else:
        if date_init is not None:
            grid_min_date = date_init
        else:
            grid_min_date = df_w[time_col].min()

        if date_end is not None:
            grid_max_date = date_end
        else:
            grid_max_date = df_w[time_col].max()

    # create dataframe with column timeslot from grid_min_date to grid_max_date
    df_timeslots = pd.DataFrame(
        {time_col: pd.date_range(grid_min_date, grid_max_date, freq=freq)}
    )
    df_timeslots["key"] = 1

    # create dataframe with all possible combinations of id_cols
    df_id["key"] = 1
    df_grid = df_timeslots.merge(df_id, on="key", how="outer").drop("key", axis=1)

    # filter registers in df_grid using the min_date and max_date by id_cols
    if min_max_by_id:
        df_grid = df_grid.merge(df_min_max, on=id_cols, how="left")
        df_grid = df_grid[
            (df_grid[time_col] >= df_grid["min_date"])
            & (df_grid[time_col] <= df_grid["max_date"])
        ]
        df_grid = df_grid.drop(["min_date", "max_date"], axis=1)

    # merge df_grid with df_w
    df_w = df_grid.merge(df_w, on=id_cols + [time_col], how="left")

    return df_w


@log_time()
def create_dense_data_pl(
    df: pl.LazyFrame,
    id_cols,
    freq: str,
    min_max_by_id: bool = None,
    date_init=None,
    date_end=None,
    time_col: str = "timeslot",
) -> pl.LazyFrame:
    """
    Create a dense dataframe with a frequency of freq, given range of dates or inherited from the dataframe,
     using the id_cols as keys.
    :param df: dataframe to be expanded
    :param id_cols: list of columns to be used as keys
    :param freq: frequency of the new dataframe
    :param min_max_by_id: boolean to indicate if the range of dates is the min and max of the dataframe by id
    :param date_init: if it has a value, all initial dates will be set to this value
    :param date_end: if it has a value, all final dates will be set to this value
    :param time_col: string with name of the column with the time information
    :return: dataframe with all the dates using the frequency freq
    """

    df_w = df.collect()

    # Get cols id_cols from df and drop duplicates
    df_id = df_w.select(id_cols).unique()

    # If min_max_by_id is True, get the min and max of the time_col by id_cols
    if min_max_by_id:
        df_min_max = df_w.group_by(id_cols).agg(
            [
                pl.col(time_col).min().alias("min_date"),
                pl.col(time_col).max().alias("max_date"),
            ]
        )

        if date_init is not None:
            df_min_max = df_min_max.with_columns(
                pl.lit(date_init).str.to_datetime(format="%Y-%m-%d").alias("min_date")
            )

        if date_end is not None:
            df_min_max = df_min_max.with_columns(
                pl.lit(date_end).str.to_datetime(format="%Y-%m-%d").alias("max_date")
            )

        grid_min_date = df_min_max["min_date"].min()
        grid_max_date = df_min_max["max_date"].max()

    else:
        grid_min_date = date_init if date_init is not None else df_w[time_col].min()
        grid_max_date = date_end if date_end is not None else df_w[time_col].max()

    # if grid_min_date is not a datetime then convert it using datetime.strptime(grid_min_date, '%Y-%M-%D').date()
    if isinstance(grid_min_date, str):
        grid_min_date = datetime.strptime(grid_min_date, "%Y-%m-%d")
    if isinstance(grid_max_date, str):
        grid_max_date = datetime.strptime(grid_max_date, "%Y-%m-%d")

    # Create dataframe with column timeslot from grid_min_date to grid_max_date
    df_timeslots = pl.DataFrame(
        pl.date_range(
            grid_min_date,
            grid_max_date,
            ("1" + freq.lower()),
            eager=True,
        ).alias(time_col)
    )

    # Create dataframe with all possible combinations of id_cols
    df_id = df_id
    df_grid = df_timeslots.join(df_id, how="cross")

    # Filter registers in df_grid using the min_date and max_date by id_cols
    if min_max_by_id:
        df_grid = df_grid.join(df_min_max, on=id_cols, how="left")
        df_grid = df_grid.filter(
            (pl.col(time_col) >= pl.col("min_date"))
            & (pl.col(time_col) <= pl.col("max_date"))
        )
        df_grid = df_grid.drop(["min_date", "max_date"])

    # time_col is a Date, turn into datetime
    df_grid = df_grid.with_columns(pl.col(time_col).cast(pl.Datetime))
    df_w = df_w.with_columns(pl.col(time_col).cast(pl.Datetime))
    # Merge df_grid with df_w
    df_w = df_grid.join(df_w, on=id_cols + [time_col], how="left")

    return df_w.lazy()


@log_time()
def create_dense_data_pllazy(
    df: pl.LazyFrame,
    id_cols,
    freq: str,
    min_max_by_id: bool = None,
    date_init=None,
    date_end=None,
    time_col: str = "timeslot",
) -> pl.LazyFrame:
    """
    Create a dense dataframe with a frequency of freq, given range of dates or inherited from the dataframe,
     using the id_cols as keys.
    :param df: dataframe to be expanded
    :param id_cols: list of columns to be used as keys
    :param freq: frequency of the new dataframe
    :param min_max_by_id: boolean to indicate if the range of dates is the min and max of the dataframe by id
    :param date_init: if it has a value, all initial dates will be set to this value
    :param date_end: if it has a value, all final dates will be set to this value
    :param time_col: string with name of the column with the time information
    :return: dataframe with all the dates using the frequency freq
    """
    # If min_max_by_id is True, get the min and max of the time_col by id_cols
    if min_max_by_id:
        df_min_max = df.group_by(id_cols).agg(
            [
                pl.col(time_col).min().alias("min_date"),
                pl.col(time_col).max().alias("max_date"),
            ]
        )

        if date_init is not None:
            df_min_max = df_min_max.with_columns(
                pl.lit(date_init).str.to_datetime(format="%Y-%m-%d").alias("min_date")
            )

        if date_end is not None:
            df_min_max = df_min_max.with_columns(
                pl.lit(date_end).str.to_datetime(format="%Y-%m-%d").alias("max_date")
            )

        grid_min_date_expr = df_min_max.select(pl.col("min_date").min())
        grid_max_date_expr = df_min_max.select(pl.col("max_date").max())

    else:
        grid_min_date_expr = df.select(pl.col(time_col).min())
        grid_max_date_expr = df.select(pl.col(time_col).max())

    # Extract min and max dates from lazy expressions
    grid_min_date = grid_min_date_expr.collect().to_series().item()
    grid_max_date = grid_max_date_expr.collect().to_series().item()

    # Convert grid_min_date and grid_max_date to datetime if they are strings
    if isinstance(grid_min_date, str):
        grid_min_date = pl.datetime.strptime(grid_min_date, "%Y-%m-%d")
    if isinstance(grid_max_date, str):
        grid_max_date = pl.datetime.strptime(grid_max_date, "%Y-%m-%d")

    # Create dataframe with column timeslot from grid_min_date to grid_max_date
    df_timeslots = pl.LazyFrame(
        pl.date_range(
            grid_min_date,
            grid_max_date,
            ("1" + freq.lower()),
            eager=True,
        ).alias(time_col)
    )

    # Create dataframe with all possible combinations of id_cols
    df_id = df.select(id_cols).unique()
    df_grid = df_timeslots.join(df_id, how="cross")

    # Filter registers in df_grid using the min_date and max_date by id_cols
    if min_max_by_id:
        df_grid = df_grid.join(df_min_max, on=id_cols, how="left")
        df_grid = df_grid.filter(
            (pl.col(time_col) >= pl.col("min_date"))
            & (pl.col(time_col) <= pl.col("max_date"))
        )
        df_grid = df_grid.drop(["min_date", "max_date"])

    # time_col is a Date, turn into datetime
    df_grid = df_grid.with_columns(pl.col(time_col).cast(pl.Datetime))
    df = df.with_columns(pl.col(time_col).cast(pl.Datetime))
    # Merge df_grid with df_w
    df_w = df_grid.join(df, on=id_cols + [time_col], how="left")

    return df_w


def create_recurrent_dataset(
    data: np.array,
    look_back: int,
    include_output_lags: bool = False,
    lags: List[int] = None,
    output_last: bool = True,
):
    """
    The create_recurrent_dataset function creates a dataset for recurrent neural networks.
    The function takes in an array of data, and returns two arrays: one containing the input data,
    and another containing the output labels. The input is a 2D array with shape (num_samples, num_features).
    The input data output is a 3D array with shape (num_samples, look_back, num_features), while the labels output
    have a 1D array with shape (num_samples, ).

    The function allows to include the output lags in the input output data. If include_output_lags is True,
    the function will create the lags indicated on the lags' argument.

    The function allows for the label to be the first "column" of the input data, or the last "column" of the input data
    by setting the output_last argument to False or True, respectively.

    :param :class:`np.array` data: pass the data to be used for training
    :param int look_back: define the number of previous time steps to use as input variables
    to predict the next time period
    :param bool include_output_lags: decide whether the output lags should be included in the input data
    :param lags:sSpecify which lags should be included in the input data
    :param output_last: indicate if the label column is the first or last one in the original data
    :return: A tuple of numpy arrays: (input_data, labels)
    :rtype: tuple
    :doc-author: baobab soluciones
    """
    x, y = [], []
    if output_last:
        x_in = data[:, :-1]
        y_in = data[:, -1:]
    else:
        x_in = data[:, 1:]
        y_in = data[:, :1]

    if lags is None or not include_output_lags:
        max_lag = 0
    else:
        max_lag = max(lags)

    for i in range(max_lag, data.shape[0] - look_back):
        a = x_in[i : (i + look_back), :]

        if include_output_lags:
            lagged = np.empty((look_back, 1))
            for lag in lags:
                lagged = np.concatenate(
                    (
                        lagged,
                        y_in[i - lag : (i + look_back - lag)].reshape((look_back, 1)),
                    ),
                    axis=1,
                )
            lagged = lagged[:, 1:]

            x.append(np.concatenate((a, lagged), axis=1))
        else:
            x.append(a)

        if output_last:
            y.append(y_in[i + look_back])
        else:
            y.append(y_in[i + look_back])

    return np.array(x), np.array(y).reshape((data.shape[0] - look_back - max_lag,))


def get_corr_matrix(
    df: pd.DataFrame,
    n_top: int = None,
    threshold: int = None,
    date_col: str = None,
    years_corr: List = None,
    subset: List = None,
):
    """
    The get_corr_matrix function takes a dataframe and returns the correlation matrix of the columns.

    :param df: pd.DataFrame: Pass in the dataframe that we want to get the correlation matrix for
    :param n_top: int: Select the top n correlated variables
    :param threshold: int: Filter the correlation matrix by a threshold value
    :param date_col: str: Specify the name of the column that contains dates
    :param years_corr: List: Specify the years for which we want to calculate the correlation matrix
    :param subset: List: Specify a subset of columns to be used in the correlation matrix
    :param : Specify the number of top correlated variables to be returned
    :return: A correlation matrix of the dataframe
    :doc-author: baobab soluciones
    """
    if not date_col:
        date_col, as_index = get_date_col_candidate(df)
    else:
        as_index = False
    raise_if_inconsistency(df, date_col, as_index)  # Raises error if problems
    if not as_index:
        df = df.set_index(date_col)
    return get_corr_matrix_aux(df, years_corr, n_top, threshold, subset)


def get_date_col_candidate(df: pd.DataFrame):
    """
    The get_date_col_candidate function takes a dataframe as an input and returns the name of the column that is
    a datetime type. If there are no columns with datetime types, it will return None. It also returns a boolean value
    that indicates whether or not the index is a datetime type.

    :param df: pd.DataFrame: Pass the dataframe to the function
    :return: A list of columns that have datetime dtypes
    :doc-author: baobab soluciones
    """
    date_column = [
        column
        for column in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[column])
    ]
    if len(date_column) == 0:
        if isinstance(df.index, pd.DatetimeIndex):
            as_index = True
            date_column = None
            return date_column, as_index
        else:
            as_index = False
            date_column = None
            return date_column, as_index
    else:
        as_index = False
    return date_column, as_index


def raise_if_inconsistency(df: pd.DataFrame, date_col: str, as_index: bool):
    """
    The raise_if_inconsistency function raises a ValueError if the input dataframe is not in the correct format.

    :param df: pd.DataFrame: Pass the dataframe to the function
    :param date_col: str: Specify the name of the column that contains the date
    :param as_index: bool: Check if the dataframe is pivoted or not
    :return: A valueerror if the dataframe is not in the correct format
    :doc-author: baobab soluciones
    """
    if date_col is None and as_index is False:
        raise ValueError("Dataframe must contain one datetime column")
    elif date_col is None and as_index:
        dupli = df.index.duplicated().sum()
        if dupli > 0:
            columns_num = sum(
                pd.api.types.is_numeric_dtype(df[col]) for col in df.columns
            )
            if columns_num == len(df.columns):
                raise ValueError("There are duplicates in the index")
            else:
                data = {
                    "fecha": pd.date_range("2023-01-01", "2023-01-06"),
                    "ventas_loc1": [30, 50, 10, 25, 32, 45],
                    "ventas_loc2": [60, 31, 46, 43, 60, 20],
                }
                example = pd.DataFrame(data).set_index("fecha")
                raise ValueError(
                    f"Dataframe must be pivot:{print(example.to_markdown())}"
                )
        else:
            columns_num = sum(
                pd.api.types.is_numeric_dtype(df[col]) for col in df.columns
            )

            if columns_num != len(df.columns):
                raise ValueError("Not all columns in Dataframe are numerics")
    elif type(date_col) == list and len(date_col) > 1:
        raise ValueError("Dataframe must contain one datetime column")
    elif type(date_col) == list or type(date_col) == str:
        if type(date_col) == list:
            date_col = date_col[0]
        dupli = df[date_col].duplicated().sum()
        if dupli > 0:
            columns_num = sum(
                pd.api.types.is_numeric_dtype(df[col]) for col in df.columns
            )
            if columns_num == len(df.columns) - 1:
                raise ValueError("There are duplicates in the index")
            else:
                data = {
                    "fecha": pd.date_range("2023-01-01", "2023-01-06"),
                    "ventas_loc1": [30, 50, 10, 25, 32, 45],
                    "ventas_loc2": [60, 31, 46, 43, 60, 20],
                }
                example = pd.DataFrame(data).set_index("fecha")
                raise ValueError(
                    f"Dataframe must be pivot:{print(example.to_markdown())}"
                )
        else:
            columns_num = sum(
                pd.api.types.is_numeric_dtype(df[col]) for col in df.columns
            )

            if columns_num != len(df.columns) - 1:
                raise ValueError("Not all columns in Dataframe are numerics")


def get_corr_matrix_aux(
    df: pd.DataFrame,
    years_corr,
    n_top,
    threshold,
    subset: List = None,
) -> Dict[str, Dict[str, float]]:
    """
    The get_corr_matrix_aux function computes the correlation matrix of a dataframe and returns
    a dictionary with the top n correlations for each time series.
    The function can also filter by years, subset and threshold.

    :param df: pd.DataFrame: Pass the dataframe to the function
    :param years_corr: Filter the dataframe by year
    :param n_top: Get the top n correlations for each time series
    :param threshold: Filter the correlation matrix by a threshold value
    :param subset: List: Specify a subset of time series to compare the correlations with
    :param : Filter the dataframe by years
    :return: A dictionary with the names of the time series as keys and a list of tuples (name, correlation) as values
    :doc-author: baobab soluciones
    """
    if n_top is not None and n_top >= df.shape[1]:
        warnings.warn(
            "Number of n_top is bigger than number of columns of the dataframe"
        )

    if years_corr is not None:
        # Filter by years
        df = df[df.index.year.isin(years_corr)]

    if subset is None:
        logging.debug(
            f"Getting {n_top} top correlations for each time series in the dataframe"
        )
    else:
        logging.debug(
            f"Getting {n_top} top correlations for each time series in the dataframe with respect to the subset"
        )

    # Compute correlation matrix
    correlation_matrix = df.corr(method="pearson")
    # Make sure the correlation with itself is not the highest
    np.fill_diagonal(correlation_matrix.values, -100)

    # Filter by subset
    if subset:
        # Keep only as columns the time series in the subset
        correlation_matrix = correlation_matrix[
            correlation_matrix.columns.intersection(subset)
        ]
        # Drop rows in the subset to avoid comparing with the subset time series
        correlation_matrix = correlation_matrix.drop(index=subset, errors="ignore")

    # Get top n correlations for each time series
    top_correlations = {}
    if threshold is not None:
        for column in correlation_matrix.columns:
            name_correlations = correlation_matrix[column][
                correlation_matrix[column] > threshold
            ]
            if name_correlations.empty:
                warnings.warn(
                    "There are no rows that have a value greater than threshold, so it returns all rows"
                )
                top_correlations[column] = dict(correlation_matrix[column])

            else:
                name_correlations = dict(name_correlations)
                top_correlations[column] = name_correlations

    elif n_top is not None:
        for column in correlation_matrix.columns:
            name_correlations = dict(correlation_matrix[column].nlargest(n_top))
            top_correlations[column] = {
                index: value for index, value in name_correlations.items()
            }
    else:
        for column in correlation_matrix.columns:
            name_correlations = dict(correlation_matrix[column])
            top_correlations[column] = name_correlations
        warnings.warn(
            "n_top and threshold are None so top_correlations return all the correlations"
        )

    return top_correlations
