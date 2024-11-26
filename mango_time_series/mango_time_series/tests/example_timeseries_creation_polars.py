from datetime import date

import numpy as np
import polars as pl
from mango.logging import log_time

np.random.seed(42)


@log_time()
def test_timeseries_creation(num_prods=10, return_pandas=True):
    stores = ["Store_A", "Store_B"]
    products = [f"Product_{i}" for i in range(1, num_prods)]

    # Define the time period
    df_dates = pl.DataFrame(
        pl.date_range(date(2015, 1, 1), date(2024, 8, 1), "1d", eager=True).alias(
            "date"
        )
    )

    # Perform the Cartesian join
    df = (
        pl.DataFrame({"store": stores})
        .join(pl.DataFrame({"product": products}), how="cross")
        .join(df_dates, how="cross")
        .with_columns(pl.col("date").dt.weekday().alias("day_of_week"))
    )

    # Generate random sales data with weekly seasonality
    base_sales = 50
    amplitude = 20
    noise = np.random.randint(
        -10,
        10,
        size=len(df),
    )

    # Calculate sales with weekly seasonality
    df = df.with_columns(
        (
            pl.lit(base_sales)
            + pl.lit(amplitude) * (2 * np.pi * pl.col("day_of_week") / 7).sin()
            + pl.Series(noise)
        )
        .alias("sales")
        .round()
        .clip(0, None)
    )

    # drop column
    df = df.drop("day_of_week")

    # drop rows if store_A and product_1, and date < 2019-05-01, other products and stores keep
    df = df.filter(
        ~(
            (pl.col("store") == "Store_A")
            & (pl.col("product") == "Product_1")
            & (pl.col("date") < date(2019, 5, 1))
        )
    )

    if return_pandas:
        import pandas as pd

        df_pd = df.to_pandas()
        # drop 10% rows randomly
        df_pd = df_pd.sample(frac=0.9)
        df_pd["store"] = df_pd["store"].astype(str)
        df_pd["product"] = df_pd["product"].astype(str)
        # date to class datetime
        df_pd["date"] = pd.to_datetime(df_pd["date"])
        # sales to float
        df_pd["sales"] = df_pd["sales"].astype(float)

        return df_pd
    df = df.sample(fraction=0.9, with_replacement=False)

    return df


def test_timeseries_prediction_creation(num_prods=10, horizon=14, return_pandas=True):
    df_sales = test_timeseries_creation(num_prods=num_prods, return_pandas=False)
    # Emulate different horizons forecasts. For each horizon create a row
    df_horizons = pl.DataFrame(
        {
            "horizon": [i for i in range(1, horizon + 1)],
        }
    )
    df_sales = df_sales.join(df_horizons, how="cross")
    df_sales = df_sales.with_columns(
        (pl.col("date") - pl.duration(days=pl.col("horizon"))).alias("forecast_origin")
    )
    df_sales = df_sales.with_columns(
        (pl.col("sales") + np.random.randint(-10, 10, size=len(df_sales))).alias("f")
    )
    # Rename columns sales->y, date->datetime
    df_sales = df_sales.with_columns(
        pl.col("sales").alias("y"),
        pl.col("date").alias("datetime"),
        pl.col("horizon").alias("h"),
    )
    df_sales = df_sales.select(
        [
            "store",
            "product",
            "forecast_origin",
            "datetime",
            "h",
            "y",
            "f",
        ]
    )

    # Add column err,abs_err,perc_err,perc_abs_err
    df_sales = df_sales.with_columns(
        (pl.col("f") - pl.col("y")).alias("err"),
        (pl.col("f") - pl.col("y")).abs().alias("abs_err"),
        (pl.col("f") - pl.col("y") / pl.col("y")).alias("perc_err"),
        (pl.col("f") - pl.col("y") / pl.col("y")).abs().alias("perc_abs_err"),
    )
    if return_pandas:
        return df_sales.to_pandas()
    return df_sales
