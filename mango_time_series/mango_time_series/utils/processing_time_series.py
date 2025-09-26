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

from mango_time_series.logging import log_time, get_configured_logger

logger = get_configured_logger()


@log_time()
def aggregate_to_input(df: pd.DataFrame, freq: str, series_conf: dict) -> pd.DataFrame:
    """
    Aggregate time series data to specified frequency using pandas.

    Groups the data by key columns and time frequency, then applies
    aggregation operations defined in the series configuration.

    :param df: DataFrame containing time series data
    :type df: pandas.DataFrame
    :param freq: Frequency string for aggregation (e.g., 'D', 'W', 'M')
    :type freq: str
    :param series_conf: Configuration dictionary containing KEY_COLS and AGG_OPERATIONS
    :type series_conf: dict
    :return: Aggregated DataFrame with specified frequency
    :rtype: pandas.DataFrame

    Note:
        - Uses pandas Grouper for time-based grouping
        - Applies aggregation operations from series_conf["AGG_OPERATIONS"]
        - Groups by both key columns and time frequency
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
    Aggregate time series data to specified frequency using Polars.

    Converts pandas DataFrame to Polars, groups by key columns and time frequency,
    applies aggregation operations, then converts back to pandas.

    :param df: DataFrame containing time series data
    :type df: pandas.DataFrame
    :param freq: Frequency string for aggregation (e.g., 'D', 'W', 'M')
    :type freq: str
    :param series_conf: Configuration dictionary containing KEY_COLS and AGG_OPERATIONS
    :type series_conf: dict
    :return: Aggregated DataFrame with specified frequency
    :rtype: pandas.DataFrame

    Note:
        - Converts pandas to Polars for processing, then back to pandas
        - Handles month frequency conversion ('m', 'MS', 'ME' -> 'mo')
        - Uses Polars datetime truncate for time-based grouping
        - Supports sum, mean, min, max, median aggregation operations
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
    Aggregate time series data to specified frequency using Polars LazyFrame.

    Groups LazyFrame by key columns and time frequency, applies aggregation
    operations, and returns sorted result as LazyFrame.

    :param df: LazyFrame containing time series data
    :type df: polars.LazyFrame
    :param freq: Frequency string for aggregation (e.g., 'D', 'W', 'M')
    :type freq: str
    :param series_conf: Configuration dictionary containing KEY_COLS and AGG_OPERATIONS
    :type series_conf: dict
    :return: Aggregated LazyFrame with specified frequency
    :rtype: polars.LazyFrame

    Note:
        - Works with Polars LazyFrame for efficient processing
        - Uses Polars datetime truncate for time-based grouping
        - Sorts result by key columns and datetime
        - Supports sum, mean, min, max, median aggregation operations
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
    Rename columns to standard time series naming convention.

    Standardizes column names by renaming time and value columns to
    'datetime' and 'y' respectively, and converts all column names to lowercase.

    :param df: DataFrame to rename columns in
    :type df: pandas.DataFrame
    :param time_col: Name of the time/datetime column
    :type time_col: str
    :param value_col: Name of the value/target column
    :type value_col: str
    :return: DataFrame with standardized column names
    :rtype: pandas.DataFrame

    Note:
        - Renames time_col to 'datetime' and value_col to 'y'
        - Converts all column names to lowercase
        - Standardizes naming for time series processing pipeline
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
    Rename columns to standard time series naming convention using Polars.

    Standardizes column names by renaming time and value columns to
    'datetime' and 'y' respectively, and converts all column names to lowercase.

    :param df: LazyFrame to rename columns in
    :type df: polars.LazyFrame
    :param time_col: Name of the time/datetime column
    :type time_col: str
    :param value_col: Name of the value/target column
    :type value_col: str
    :return: LazyFrame with standardized column names
    :rtype: polars.LazyFrame

    Note:
        - Renames time_col to 'datetime' and value_col to 'y'
        - Converts all column names to lowercase using with_columns
        - Standardizes naming for time series processing pipeline
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
    Remove rows with negative values from time series data.

    Filters out rows where the target variable (y) has negative values,
    which are typically not meaningful for sales or demand forecasting.

    :param df: DataFrame containing time series data with 'y' column
    :type df: pandas.DataFrame
    :return: DataFrame with negative value rows removed
    :rtype: pandas.DataFrame

    Note:
        - Removes rows where y < 0
        - Logs the number of rows being dropped
        - Preserves all other data and columns
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
    Remove rows with negative values from time series data using Polars.

    Filters out rows where the target variable (y) has negative values,
    which are typically not meaningful for sales or demand forecasting.
    Uses lazy evaluation for efficient processing.

    :param df: LazyFrame containing time series data with 'y' column
    :type df: polars.LazyFrame
    :return: LazyFrame with negative value rows removed
    :rtype: polars.LazyFrame

    Note:
        - Removes rows where y < 0 using filter operation
        - Logs the number of rows being dropped
        - Uses lazy evaluation for memory efficiency
        - Preserves all other data and columns
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
    Add COVID period indicator column to time series data.

    Creates a boolean column 'covid' that marks the COVID-19 pandemic period
    from March 2020 to March 2021, which can be used for analysis or modeling.

    :param df: LazyFrame containing time series data with datetime column
    :type df: polars.LazyFrame
    :return: LazyFrame with added 'covid' boolean column
    :rtype: polars.LazyFrame

    Note:
        - COVID period: 2020-03-01 to 2021-03-01
        - Creates boolean column where True indicates COVID period
        - Useful for identifying pandemic impact on time series patterns
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
    Create lagged columns for time series feature engineering.

    Generates lagged versions of a specified column for time series analysis.
    Supports both positive lags (past values) and negative lags (future values/leads).
    Can handle multiple time series by grouping on key columns and optionally
    handle discontinuities by checking for changes in specified columns.

    :param df: DataFrame to add lagged columns to
    :type df: pandas.DataFrame
    :param col: Name of the column to create lags for
    :type col: str
    :param lags: List of lag values (positive for past, negative for future)
    :type lags: list[int]
    :param key_cols: Columns that define each time series (for grouping)
    :type key_cols: list[str], optional
    :param check_col: Columns to check for discontinuities in lagged series
    :type check_col: list[str], optional
    :return: DataFrame with lagged columns added
    :rtype: pandas.DataFrame

    Note:
        - Positive lags create columns named '{col}_lag{lag}'
        - Negative lags create columns named '{col}_lead{abs(lag)}'
        - Groups by key_cols if provided, otherwise treats as single series
        - Sets lag values to NaN when discontinuities detected in check_col
        - Requires pandas and numpy to be installed
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


def series_as_columns(df: pd.DataFrame, series_conf: dict) -> pd.DataFrame:
    """
    Pivot time series data to wide format with series as columns.

    Transforms long-format time series data into wide format where each
    unique combination of key columns becomes a separate column. Uses
    sum aggregation for overlapping values.

    :param df: DataFrame in long format with time series data
    :type df: pandas.DataFrame
    :param series_conf: Configuration dictionary containing KEY_COLS and VALUE_COL
    :type series_conf: dict
    :return: DataFrame in wide format with series as columns
    :rtype: pandas.DataFrame

    Note:
        - Pivots using datetime as index and key_cols as columns
        - Uses sum aggregation for overlapping values
        - Flattens multi-level column names with underscore separator
        - Renames first column back to 'datetime'
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


def series_as_rows(df: pd.DataFrame, series_conf: dict) -> pd.DataFrame:
    """
    Pivot time series data to long format with series as rows.

    Transforms wide-format time series data into long format where each
    time series becomes a separate row. Splits column names to reconstruct
    the original key columns.

    :param df: DataFrame in wide format with series as columns
    :type df: pandas.DataFrame
    :param series_conf: Configuration dictionary containing KEY_COLS and VALUE_COL
    :type series_conf: dict
    :return: DataFrame in long format with series as rows
    :rtype: pandas.DataFrame

    Note:
        - Uses melt to transform wide to long format
        - Splits column names by underscore to reconstruct key columns
        - Arranges columns in order: datetime, key_cols, value_col
        - Removes temporary 'id' column after processing
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
) -> pl.LazyFrame:
    """
    Process time series data through complete preprocessing pipeline.

    Applies a comprehensive preprocessing pipeline including column standardization,
    negative value removal, aggregation, dense data creation, and COVID marking.
    Converts input to Polars LazyFrame for efficient processing.

    :param df: Input DataFrame (pandas, Polars, or LazyFrame)
    :type df: Union[pandas.DataFrame, polars.DataFrame, polars.LazyFrame]
    :param series_conf: Configuration dictionary containing processing parameters
    :type series_conf: dict
    :return: Processed LazyFrame with standardized time series data
    :rtype: polars.LazyFrame

    Note:
        - Converts input to LazyFrame for processing
        - Standardizes column names (TIME_COL -> datetime, VALUE_COL -> y)
        - Removes negative values from target variable
        - Aggregates to daily frequency, then to specified frequency
        - Creates dense data with complete date ranges
        - Adds COVID period indicator column
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
    id_cols: List[str],
    freq: str,
    min_max_by_id: bool = None,
    date_init: str = None,
    date_end: str = None,
    time_col: str = "timeslot",
) -> pd.DataFrame:
    """
    Create dense time series data with complete date ranges.

    Expands sparse time series data to include all dates within the specified
    frequency, filling missing dates with NaN values. Can use global date range
    or individual date ranges per ID group.

    :param df: DataFrame to expand with complete date ranges
    :type df: pandas.DataFrame
    :param id_cols: List of columns that identify unique time series
    :type id_cols: list[str]
    :param freq: Frequency string for date range generation (e.g., 'D', 'W', 'M')
    :type freq: str
    :param min_max_by_id: Whether to use individual min/max dates per ID (default: None)
    :type min_max_by_id: bool, optional
    :param date_init: Override start date for all series (default: None)
    :type date_init: str, optional
    :param date_end: Override end date for all series (default: None)
    :type date_end: str, optional
    :param time_col: Name of the time column (default: "timeslot")
    :type time_col: str
    :return: DataFrame with complete date ranges and original data merged
    :rtype: pandas.DataFrame

    Note:
        - Creates Cartesian product of all dates and ID combinations
        - Fills missing dates with NaN values
        - Requires pandas to be installed
        - Uses cross join to create complete date grid
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
    id_cols: List[str],
    freq: str,
    min_max_by_id: bool = None,
    date_init: str = None,
    date_end: str = None,
    time_col: str = "timeslot",
) -> pl.LazyFrame:
    """
    Create dense time series data with complete date ranges using Polars.

    Expands sparse time series data to include all dates within the specified
    frequency, filling missing dates with null values. Uses Polars for efficient
    processing with lazy evaluation.

    :param df: LazyFrame to expand with complete date ranges
    :type df: polars.LazyFrame
    :param id_cols: List of columns that identify unique time series
    :type id_cols: list[str]
    :param freq: Frequency string for date range generation (e.g., 'D', 'W', 'M')
    :type freq: str
    :param min_max_by_id: Whether to use individual min/max dates per ID (default: None)
    :type min_max_by_id: bool, optional
    :param date_init: Override start date for all series (default: None)
    :type date_init: str, optional
    :param date_end: Override end date for all series (default: None)
    :type date_end: str, optional
    :param time_col: Name of the time column (default: "timeslot")
    :type time_col: str
    :return: LazyFrame with complete date ranges and original data merged
    :rtype: polars.LazyFrame

    Note:
        - Collects LazyFrame for date range calculations
        - Creates cross join of all dates and ID combinations
        - Converts time column to datetime format
        - Returns LazyFrame for efficient processing
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
    data: np.ndarray,
    look_back: int,
    include_output_lags: bool = False,
    lags: List[int] = None,
    output_last: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create dataset for recurrent neural networks with time series sequences.

    Transforms 2D time series data into 3D sequences suitable for RNN training.
    Creates sliding windows of historical data as input and corresponding future
    values as targets. Supports optional output lag features and flexible target
    column positioning.

    :param data: 2D array with shape (num_samples, num_features)
    :type data: numpy.ndarray
    :param look_back: Number of previous time steps to use as input
    :type look_back: int
    :param include_output_lags: Whether to include output lag features (default: False)
    :type include_output_lags: bool
    :param lags: List of lag periods to include as features (default: None)
    :type lags: list[int], optional
    :param output_last: Whether target is last column (True) or first column (False)
    :type output_last: bool
    :return: Tuple containing (input_sequences, target_values)
    :rtype: tuple[numpy.ndarray, numpy.ndarray]

    Note:
        - Input shape: (num_samples, look_back, num_features)
        - Output shape: (num_samples,)
        - Supports both positive and negative lags
        - Handles output column positioning (first vs last)
        - Creates sequences with proper temporal alignment
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
    threshold: float = None,
    date_col: str = None,
    years_corr: List[int] = None,
    subset: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate correlation matrix for time series data with filtering options.

    Computes Pearson correlation matrix for time series data with various filtering
    and selection options. Automatically detects date columns and validates data format.
    Returns top correlations or correlations above threshold for each time series.

    :param df: DataFrame containing time series data
    :type df: pandas.DataFrame
    :param n_top: Number of top correlations to return per series (default: None)
    :type n_top: int, optional
    :param threshold: Minimum correlation threshold to filter results (default: None)
    :type threshold: float, optional
    :param date_col: Name of the date column (auto-detected if None)
    :type date_col: str, optional
    :param years_corr: List of years to filter data for correlation calculation (default: None)
    :type years_corr: list[int], optional
    :param subset: List of series names to focus correlation analysis on (default: None)
    :type subset: list[str], optional
    :return: Dictionary with series names as keys and correlation dictionaries as values
    :rtype: dict[str, dict[str, float]]

    Note:
        - Automatically detects datetime columns or uses index
        - Validates data format and raises errors for inconsistencies
        - Sets diagonal correlations to -100 to avoid self-correlation
        - Returns all correlations if both n_top and threshold are None
    """
    if not date_col:
        date_col, as_index = get_date_col_candidate(df)
    else:
        as_index = False
    raise_if_inconsistency(df, date_col, as_index)  # Raises error if problems
    if not as_index:
        df = df.set_index(date_col)
    return get_corr_matrix_aux(df, years_corr, n_top, threshold, subset)


def get_date_col_candidate(df: pd.DataFrame) -> tuple[List[str] | None, bool]:
    """
    Identify datetime columns in DataFrame for time series analysis.

    Searches for datetime columns in the DataFrame and determines whether
    the index contains datetime information. Returns the datetime column
    names and a boolean indicating if the index is datetime-based.

    :param df: DataFrame to analyze for datetime columns
    :type df: pandas.DataFrame
    :return: Tuple containing (datetime_columns, index_is_datetime)
    :rtype: tuple[list[str] or None, bool]

    Note:
        - Returns None for datetime_columns if no datetime columns found
        - Returns True for index_is_datetime if index is DatetimeIndex
        - Searches all columns for datetime64 dtypes
        - Used by correlation functions for automatic date detection
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


def raise_if_inconsistency(df: pd.DataFrame, date_col: str, as_index: bool) -> None:
    """
    Validate DataFrame format for time series correlation analysis.

    Performs comprehensive validation of DataFrame structure to ensure
    it's suitable for correlation analysis. Checks for datetime columns,
    duplicate indices, numeric columns, and proper pivoted format.

    :param df: DataFrame to validate
    :type df: pandas.DataFrame
    :param date_col: Name of the date column (or None if using index)
    :type date_col: str
    :param as_index: Whether the DataFrame uses datetime index
    :type as_index: bool
    :return: None (raises ValueError if validation fails)
    :rtype: None

    Raises:
        ValueError: If DataFrame format is inconsistent or invalid

    Note:
        - Validates presence of exactly one datetime column
        - Checks for duplicate indices in time series data
        - Ensures all non-datetime columns are numeric
        - Provides example format in error messages
        - Used by correlation functions for data validation
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
                raise ValueError(f"Dataframe must be pivot:\n{example.to_string()}")
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
                raise ValueError(f"Dataframe must be pivot:\n{example.to_string()}")
        else:
            columns_num = sum(
                pd.api.types.is_numeric_dtype(df[col]) for col in df.columns
            )

            if columns_num != len(df.columns) - 1:
                raise ValueError("Not all columns in Dataframe are numerics")


def get_corr_matrix_aux(
    df: pd.DataFrame,
    years_corr: List[int] = None,
    n_top: int = None,
    threshold: float = None,
    subset: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute correlation matrix with filtering and selection options.

    Calculates Pearson correlation matrix for time series data and returns
    filtered results based on various criteria. Supports year filtering,
    top N correlations, threshold filtering, and subset analysis.

    :param df: DataFrame with datetime index and numeric columns
    :type df: pandas.DataFrame
    :param years_corr: List of years to filter data for correlation calculation (default: None)
    :type years_corr: list[int], optional
    :param n_top: Number of top correlations to return per series (default: None)
    :type n_top: int, optional
    :param threshold: Minimum correlation threshold to filter results (default: None)
    :type threshold: float, optional
    :param subset: List of series names to focus correlation analysis on (default: None)
    :type subset: list[str], optional
    :return: Dictionary with series names as keys and correlation dictionaries as values
    :rtype: dict[str, dict[str, float]]

    Note:
        - Filters data by specified years before correlation calculation
        - Sets diagonal correlations to -100 to avoid self-correlation
        - Supports subset analysis for focused correlation studies
        - Returns all correlations if both n_top and threshold are None
        - Issues warnings for edge cases (no threshold matches, etc.)
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
