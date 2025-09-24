import polars as pl


def differentiate_target(df, group_cols, lag) -> pl.DataFrame:
    """
    Differentiate the target variable by applying lag-based differencing.

    Performs time series differentiation by calculating the difference between
    the current value and the value at the specified lag. This is useful for
    making non-stationary time series stationary by removing trends and
    seasonality. The original target values are preserved as 'y_orig' and
    'y_orig_lagged' columns for reference.

    :param df: Input DataFrame containing time series data
    :type df: polars.DataFrame
    :param group_cols: List of column names to group by for differentiation
    :type group_cols: list[str]
    :param lag: Number of periods to lag for differentiation
    :type lag: int
    :return: DataFrame with differentiated target variable and original values preserved
    :rtype: polars.DataFrame

    Note:
        - The DataFrame is sorted by 'datetime' column before processing
        - Rows with null values in the differentiated target are removed
        - Original target values are preserved in 'y_orig' and 'y_orig_lagged' columns

    Example:
        >>> df = pl.DataFrame({
        ...     "datetime": ["2023-01-01", "2023-01-02", "2023-01-03"],
        ...     "y": [100, 110, 120],
        ...     "group": ["A", "A", "A"]
        ... })
        >>> result = differentiate_target(df, ["group"], lag=1)
    """

    df = df.with_columns(pl.col("y").alias("y_orig")).sort("datetime")
    df = df.with_columns(pl.col("y_orig").shift(lag).alias("y_orig_lagged")).sort(
        "datetime"
    )
    df = df.with_columns([pl.col("y").diff(lag).over(group_cols).alias(f"y")])
    # drop rows if y = null
    df = df.filter(pl.col("y").is_not_null())

    return df
