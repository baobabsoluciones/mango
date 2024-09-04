import re

import numpy as np
import pandas as pd
import polars as pl

from mango_base.mango.logging import log_time
from mango_base.mango.logging.logger import get_basic_logger
from mango_time_series.mango.features.moving_averages import (
    create_recent_variables,
    create_seasonal_variables,
)
from mango_time_series.mango.features.time_features import month_as_bspline

logger = get_basic_logger()


@log_time()
def create_tabular_structure(df, horizon, SERIES_CONF):
    """
    Create a tabular structure for a time series dataframe
    :param df: pd.DataFrame
    :param horizon: int
    :param SERIES_CONF: dict
    :return: pd.DataFrame
    """

    logger.info("Creating tabular structure")

    # Create a copy of the dataframe
    df = df.copy()
    # range 1 to horizon
    df_horizon = pd.DataFrame({"horizon": range(1, horizon + 1)})

    # grid with all the possible combinations of original df
    df = df.merge(df_horizon, how="cross")
    # sort by KEY_COLS and datetime
    df = df.sort_values(SERIES_CONF["KEY_COLS"] + ["datetime"])

    # CREATE A NEW COLUMN WITH THE DATE OF THE FORECAST_origin
    # datetime - horizon with units from series_conf["TIME_PERIOD"]
    if SERIES_CONF["TIME_PERIOD_DESCR"] == "month":

        df["forecast_origin"] = [
            date - pd.DateOffset(months=months)
            for date, months in zip(df["datetime"], df["horizon"])
        ]
    else:
        df["forecast_origin"] = df["datetime"] - pd.to_timedelta(
            df["horizon"], unit=SERIES_CONF["TIME_PERIOD_PD"]
        )

    return df


@log_time()
def create_tabular_structure_pl(
    df: pl.LazyFrame, horizon: int, SERIES_CONF: dict
) -> pl.LazyFrame:
    """
    Create a tabular structure for a time series dataframe using Polars
    :param df: pl.DataFrame
    :param horizon: int
    :param SERIES_CONF: dict
    :return: pl.DataFrame
    """

    # Logging info
    logger.info("Creating tabular structure")
    time_unit = re.sub(r"\d", "", SERIES_CONF["TIME_PERIOD"])

    # Create a DataFrame with the 'horizon' column ranging from 1 to horizon
    df_horizon = pl.LazyFrame({"horizon": range(1, horizon + 1)})

    # Perform a cross join (Cartesian product)
    df = df.join(df_horizon, how="cross")

    df = df.with_columns(
        [
            (
                pl.col("datetime").dt.offset_by(
                    ("-" + pl.col("horizon").cast(str) + time_unit)
                )
            ).alias("forecast_origin")
        ]
    )

    return df


def differentiate_target(df, group_cols, lag):
    """
    Differentiate the target variable by the lag specified
    """

    df = df.with_columns(pl.col("y").alias("y_orig")).sort("datetime")
    df = df.with_columns(pl.col("y_orig").shift(lag).alias("y_orig_lagged")).sort(
        "datetime"
    )
    df = df.with_columns([pl.col("y").diff(lag).over(group_cols).alias(f"y")])
    # drop rows if y = null
    df = df.filter(pl.col("y").is_not_null())

    return df


from mango_time_series.mango.utils.CONST import (
    SERIES_CONFIGURATION as SERIES_CONF,
)
from mango_time_series.mango.utils.processing import process_time_series


# df = test_timeseries_creation(1000)
df = pd.read_excel(
    r"G:\Unidades compartidas\clece_pmr_202207\proyecto\desarrollo\datos\time_series.xlsx"
)
df["airport"] = "BCN"

df = process_time_series(df, SERIES_CONF)

df = df.with_columns(pl.col("y").fill_null(0))
horizon = 56

## Alternativa con FOR
# all_keys = df.select(SERIES_CONF["KEY_COLS"]).collect().unique()
# df_all = pl.DataFrame()
# for i in all_keys.rows():
#     print("Store: ", i[0], "Product: ", i[1])
#     df_key = df.filter(pl.col("store") == i[0]).filter(pl.col("product") == i[1])
#
#     df_tab = create_tabular_structure_pl(df_key, horizon, SERIES_CONF)
#
#     df_tab = rolling_recent_averages(
#         df_tab, SERIES_CONF["KEY_COLS"], [7, 14, 28], gap=1
#     )
#     df_tab = rolling_seasonal_averages(
#         df_tab, SERIES_CONF["KEY_COLS"], [4], SERIES_CONF["TIME_PERIOD_DESCR"], gap=1
#     )
#
#     df_all = pl.concat([df_all, df_tab.collect()])

## Alternativa entera polars

df = differentiate_target(df, SERIES_CONF["KEY_COLS"], 7)

df_big = create_tabular_structure_pl(df, horizon, SERIES_CONF)

df_big = create_recent_variables(
    df_big, SERIES_CONF, [7, 14, 28, 56], lags=[1, 2], gap=0
)

df_big = create_seasonal_variables(
    df=df_big,
    SERIES_CONF=SERIES_CONF,
    window=[2, 4],
    lags=[1, 4, 8],
    season_unit=SERIES_CONF["TIME_PERIOD_DESCR"],
    freq=SERIES_CONF["TS_PARAMETERS"]["season_period"],
)

df_all_pl = df_big.collect().sort("datetime")

df_pd = df_all_pl.to_pandas()

from mango_time_series.mango.validation.custom_folds import (
    create_recent_folds,
    create_recent_seasonal_folds,
)

df_pd = df_pd.dropna()


logger.info("Tabular structure created")
# with colmns duration= end-start


# filter df_pl_sma on rows with horizon = 1

df_pd = month_as_bspline(df_pd)
df_pd["weekday"] = df_pd["datetime"].dt.weekday
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# all but datetime and y
X_columns = df_pd.columns.tolist()
X_columns.remove("datetime")
X_columns.remove("y")
X_columns.remove("airport")
X_columns.remove("forecast_origin")
X_columns.remove("y_orig")
X_columns.remove("y_orig_lagged")

# target y
y_column = "y"

m = GradientBoostingRegressor()
# gridseacrhv with cv with cv_folds
# fit model with df_reduced
parameters = {
    # "loss":["deviance"],
    "learning_rate": [0.01],
    # "min_samples_split": [0.5],
    "max_depth": [10],
    "criterion": ["friedman_mse"],
    "subsample": [0.5],
    "n_estimators": [200],
}
df_pd = df_pd.reset_index()
date_end_train = "2024-02-01"
df_pd_tr = df_pd[df_pd["datetime"] < date_end_train]
df_pd_te = df_pd[df_pd["datetime"] >= date_end_train]
# df_spl = df_spl.dropna()
ids = create_recent_folds(df_pd_tr, 56, SERIES_CONF, SERIES_CONF["RECENT_FOLDS"])

sea_ids = create_recent_seasonal_folds(
    df_pd_tr, 56, SERIES_CONF, SERIES_CONF["SEASONAL_FOLDS"]
)

# paste ids and sea_ids
cv_folds = ids + sea_ids
clf = GridSearchCV(
    m,
    parameters,
    scoring="neg_mean_absolute_error",
    cv=cv_folds,
    verbose=2,
    return_train_score=True,
)

clf.fit(df_pd_tr[X_columns], df_pd_tr[y_column])

df_pd_te["f"] = clf.predict(df_pd_te[X_columns])

# t = df_pd_te[df_pd_te.forecast_origin == "2024-02-01 00:00:00.000"]
# t = t[["forecast_origin", "datetime", "weekday", "y", "f", "y_orig", "y_orig_lagged"]]
df_pd_te["err"] = df_pd_te["f"] - df_pd_te["y"]
df_pd_te = df_pd_te.sort_values(["datetime", "forecast_origin"])

df_pd_te["prev_err"] = df_pd_te.groupby(["weekday", "forecast_origin"])["err"].shift(1)
df_pd_te["cumsum_err"] = df_pd_te.groupby(["weekday", "forecast_origin"])[
    "prev_err"
].cumsum()

# calculate mae, mape

# create a ploty graph with forecast vs actual
import plotly.graph_objects as go

df_pd_te["f_orig"] = df_pd_te["f"] + df_pd_te["y_orig_lagged"]
y_plot = "y_orig"
f_plot = "f_orig2"
# cumsum error by weekday
df_pd_te = df_pd_te.sort_values(["datetime", "forecast_origin"])
df_pd_te["f_orig2"] = df_pd_te["f_orig"] + df_pd_te["cumsum_err"].fillna(0)


# cumsum error by weekday

df_pl = df_pd_te[df_pd_te.forecast_origin == "2024-02-01 00:00:00.000"]
fig = go.Figure()

# Add trace for actual data (y)
fig.add_trace(
    go.Scatter(x=df_pl["datetime"], y=df_pl[y_plot], mode="lines", name="Actual")
)

# Add trace for forecast data (f)
fig.add_trace(
    go.Scatter(x=df_pl["datetime"], y=df_pl[f_plot], mode="lines", name="Forecast")
)

# Update layout for better visualization
fig.update_layout(
    title="Actual vs Forecast",
    xaxis_title="Datetime",
    yaxis_title="Values",
    legend_title="Legend",
    hovermode="x unified",
)
# y starts from 0
fig.update_yaxes(range=[0, 1500])

# Show plot
fig.show()


from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# df_score = df_pd_te[df_pd_te.horizon <= 7]
df_score = df_pd_te
mae = mean_absolute_error(df_score[y_plot], df_score[f_plot])
mape = mean_absolute_percentage_error(df_score[y_plot], df_score[f_plot])
mae_snaive = mean_absolute_error(df_score[y_plot], df_score["y_orig_lagged"])
mape_snaive = mean_absolute_percentage_error(
    df_score[y_plot], df_score["y_orig_lagged"]
)
print(f"MAE model: {mae}")
print(f"MAPE model: {mape}")
print(f"MAE snaive: {mae_snaive}")
print(f"MAPE snaive: {mape_snaive}")

# df_pl_sma_sorted = df_tab_filtered.sort(by=SERIES_CONF["KEY_COLS"] + ["datetime"])

print("Olá mundo")
