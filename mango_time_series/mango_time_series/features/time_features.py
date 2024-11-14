import pandas as pd
import polars as pl


def create_time_features(df: pl.LazyFrame, SERIES_CONF: dict):
    """
    Depending on SERIES_CONF['TIME_PERIOD_DESCR'] create time features for the dataframe

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


def month_as_bspline(df: pl.DataFrame):
    def spline_transformer(period, degree=3, extrapolation="periodic"):
        """
        Returns a transformer that applies B-spline transformation.
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


def custom_weights(index):
    """
    Return 0 if index is between 2012-06-01 and 2012-10-21.
    """
    weights = np.where((index >= "2012-06-01") & (index <= "2012-10-21"), 0, 1)

    return weights
