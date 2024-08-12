from datetime import date

import numpy as np
import polars as pl

from mango_base.mango.logging import log_time

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
