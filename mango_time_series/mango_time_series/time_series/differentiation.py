import polars as pl


def differentiate_target(df, group_cols, lag) -> pl.DataFrame:
    """
    Differentiate the target variable by the lag specified
    :param df: pl.DataFrame, input dataframe
    :param group_cols: list, columns to group by
    :param lag: int, lag to differentiate
    :return: pl.DataFrame, differentiated dataframe
    """

    df = df.with_columns(pl.col("y").alias("y_orig")).sort("datetime")
    df = df.with_columns(pl.col("y_orig").shift(lag).alias("y_orig_lagged")).sort(
        "datetime"
    )
    df = df.with_columns([pl.col("y").diff(lag).over(group_cols).alias(f"y")])
    # drop rows if y = null
    df = df.filter(pl.col("y").is_not_null())

    return df
