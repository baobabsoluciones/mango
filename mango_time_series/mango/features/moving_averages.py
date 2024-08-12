import polars as pl

from mango_base.mango.logging import log_time
from mango_base.mango.logging.logger import get_basic_logger

logger = get_basic_logger()


@log_time()
def rolling_recent_averages(
    df: pl.LazyFrame,
    group_cols: list,
    window: list,
    gap: int = 1,
    colname: str = "roll",
) -> pl.LazyFrame:
    """
    Create rolling averages for the last window days, excluding the current row,
    but only for rows where horizon = 1. Creating one column for each window, and grouping
    by the key columns in SERIES_CONF.

    :param df: pd.DataFrame
    :param group_cols: list of columns to group by
    :param window: list of integers specifying the rolling window sizes in days.
    :param gap: integer specifying how many previous rows to start the window from, excluding the current row.
    :param colname: string specifying the prefix for the new rolling average columns.
    :return: pd.DataFrame with new columns for each rolling average.
    """

    # Filter the DataFrame to include only rows where horizon == 1
    df_filtered = df.filter(pl.col("horizon") == 1)
    df_others = df.filter(pl.col("horizon") != 1)
    list_rolling_columns = []

    # Apply rolling averages for each window size to the filtered DataFrame
    for w in window:
        rolling_col_name = f"y_{colname}_{w}"
        list_rolling_columns.append(rolling_col_name)
        df_filtered = df_filtered.with_columns(
            pl.col("y")
            .shift(gap)  # Shift values by the gap to exclude the current row
            .rolling_mean(window_size=w)
            .over(group_cols)
            .alias(rolling_col_name)
        )

        # initialize the rolling column for the other rows
        df_others = df_others.with_columns(pl.lit(None).alias(rolling_col_name))

    # Collect the result to convert back to a DataFrame
    df_filtered_result = df_filtered
    df_others_result = df_others

    # concat the two dataframes
    df_final = pl.concat([df_filtered_result, df_others_result])
    df_final = df_final.sort(group_cols + ["forecast_origin", "datetime"])

    # grouped by key columns+forecast_origin rolling columns NA should be forward filled
    df_final = df_final.with_columns(
        [
            pl.col(col)
            .fill_null(strategy="forward")
            .over(group_cols + ["forecast_origin"])
            .alias(col)
            for col in list_rolling_columns
        ]
    )

    return df_final


@log_time()
def rolling_seasonal_averages(
    df: pl.LazyFrame,
    group_cols: list,
    window: list,
    season_unit: str,
    gap: int = 1,
) -> pl.LazyFrame:
    """
    Create rolling averages for the last window days, excluding the current row,
    but only for rows where horizon = 1. Creating one column for each window, and grouping
    by the key columns in SERIES_CONF.

    :param df: pd.DataFrame
    :param group_cols: list of columns to group by
    :param window: list of integers specifying the rolling window sizes in days.
    :param season_unit: string specifying the unit of the seasonality.
    :param gap: integer specifying how many previous rows to start the window from, excluding the current row.
    :return: pd.DataFrame with new columns for each rolling average.
    """

    if season_unit == "day":
        df = df.with_columns(pl.col("datetime").dt.weekday().alias("season_unit"))
    elif season_unit == "week":
        df = df.with_columns(pl.col("datetime").dt.week().alias("season_unit"))
    elif season_unit == "month":
        df = df.with_columns(pl.col("datetime").dt.month().alias("season_unit"))

    df = rolling_recent_averages(
        df,
        group_cols + ["season_unit"],
        window,
        gap,
        colname="sea_roll",
    )

    return df
