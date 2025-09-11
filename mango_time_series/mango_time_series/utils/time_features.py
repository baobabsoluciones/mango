import numpy as np
import pandas as pd
import polars as pl


def create_time_features(df: pl.LazyFrame, SERIES_CONF: dict) -> pl.LazyFrame:
    """
    Create time-based features from datetime column based on time period description.

    Extracts relevant time features from the datetime column depending on the
    time period description in the series configuration. Creates different
    sets of features for monthly, daily, and weekly time series.

    :param df: LazyFrame containing time series data with datetime column
    :type df: polars.LazyFrame
    :param SERIES_CONF: Configuration dictionary containing TIME_PERIOD_DESCR
    :type SERIES_CONF: dict
    :return: LazyFrame with additional time feature columns
    :rtype: polars.LazyFrame

    Note:
        - For monthly data: adds 'month' and 'year' columns
        - For daily data: adds 'day', 'month', 'year', and 'weekday' columns
        - For weekly data: adds 'week' and 'year' columns
        - Features are extracted using Polars datetime methods
    """

    # if time period is month
    if SERIES_CONF["TIME_PERIOD_DESCR"] == "month":
        df = df.with_columns(
            pl.col("datetime").dt.month().alias("month"),
            pl.col("datetime").dt.year().alias("year"),
        )
    elif SERIES_CONF["TIME_PERIOD_DESCR"] == "day":
        df = df.with_columns(
            pl.col("datetime").dt.day().alias("day"),
            pl.col("datetime").dt.month().alias("month"),
            pl.col("datetime").dt.year().alias("year"),
            pl.col("datetime").dt.weekday().alias("weekday"),
        )
    elif SERIES_CONF["TIME_PERIOD_DESCR"] == "week":
        df = df.with_columns(
            pl.col("datetime").dt.week().alias("week"),
            pl.col("datetime").dt.year().alias("year"),
        )

    return df


def month_as_bspline(df: pl.DataFrame) -> pd.DataFrame:
    """
    Transform monthly seasonality into B-spline features for machine learning.

    Converts monthly seasonal patterns into smooth B-spline features that can
    be used effectively in machine learning models. Creates day-of-year features
    and applies B-spline transformation with periodic extrapolation.

    :param df: DataFrame containing time series data with datetime column
    :type df: polars.DataFrame
    :return: DataFrame with original data plus B-spline features
    :rtype: pandas.DataFrame

    Note:
        - Converts Polars DataFrame to pandas for sklearn compatibility
        - Creates day_of_year feature from datetime column
        - Applies B-spline transformation with 12 knots (monthly period)
        - Uses periodic extrapolation for smooth seasonal transitions
        - Removes intermediate day_of_year column after transformation
    """

    def spline_transformer(
        period: int, degree: int = 3, extrapolation: str = "periodic"
    ):
        """
        Create B-spline transformer for seasonal feature engineering.

        :param period: Seasonal period (e.g., 12 for monthly)
        :type period: int
        :param degree: Degree of B-spline (default: 3)
        :type degree: int
        :param extrapolation: Extrapolation method (default: "periodic")
        :type extrapolation: str
        :return: Configured SplineTransformer
        :rtype: sklearn.preprocessing.SplineTransformer
        """
        from sklearn.preprocessing import SplineTransformer

        return SplineTransformer(
            degree=degree,
            n_knots=period + 1,
            knots="uniform",
            extrapolation=extrapolation,
            include_bias=True,
        ).set_output(transform="pandas")

    if not isinstance(df, pd.DataFrame):
        df_pd = df.to_pandas()
    else:
        df_pd = df

    # create day_of_year
    df_pd["day_of_year"] = df_pd["datetime"].dt.dayofyear

    splines_month = spline_transformer(period=12).fit_transform(df_pd[["day_of_year"]])
    splines_month.columns = [f"spline{i}" for i in range(len(splines_month.columns))]

    df_splines = pd.concat([df_pd, splines_month], axis=1)
    df_splines = df_splines.drop(columns=["day_of_year"])

    return df_splines


def custom_weights(index: pd.DatetimeIndex) -> np.ndarray:
    """
    Create custom weights for time series data with specific period exclusion.

    Generates a weight array where values are set to 0 for a specific date range
    and 1 for all other dates. This is useful for excluding certain periods
    from model training or evaluation.

    :param index: DatetimeIndex containing the dates to weight
    :type index: pandas.DatetimeIndex
    :return: Array of weights (0 or 1) corresponding to each date
    :rtype: numpy.ndarray

    Note:
        - Sets weight to 0 for dates between 2012-06-01 and 2012-10-21
        - Sets weight to 1 for all other dates
        - Useful for excluding specific periods from analysis
        - Returns numpy array for efficient computation
    """
    weights = np.where((index >= "2012-06-01") & (index <= "2012-10-21"), 0, 1)

    return weights
