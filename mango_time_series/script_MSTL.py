import pandas as pd

from mango_time_series.mango.timeseries_class.base_class import TimeSeriesProcessor
from mango_time_series.mango.timeseries_class.config_class import SeriesConfiguration

from mango_base.mango.logging.logger import get_basic_logger

logger = get_basic_logger()

# Load the data
df = pd.read_excel(
    r"G:\Unidades compartidas\clece_pmr_202207\proyecto\desarrollo\datos\time_series_synthetic_interpolate.xlsx"
)
df["airport"] = "BCN"
df.drop(columns=["PNP", "TOTAL"], inplace=True)

# Create an instance of SeriesConfiguration directly
config = SeriesConfiguration(
    key_cols=["airport"],
    time_period_descr="day",
    time_col="Fecha_Vuelo",
    value_col="FIN",
    recent_folds=2,
    seasonal_folds=1,
    agg_operations={"y": "sum"},
)

# Create an instance of TimeSeriesProcessor with the configuration
ts = TimeSeriesProcessor(config)

# Load and preprocess the data
ts.load_data(df)

# Convert to pandas DataFrame
df_pd = ts.data.collect().to_pandas()

df_pd = df_pd.reset_index()
df_pd = df_pd[["airport", "datetime", "y"]]
# rename to ds
df_pd = df_pd.rename(columns={"datetime": "ds", "airport": "unique_id"})
date_end_train = "2024-09-22"
df_pd_tr = df_pd[df_pd["ds"] < date_end_train]
df_pd_te = df_pd[df_pd["ds"] >= date_end_train]

from statsforecast.models import MSTL, AutoTheta, AutoARIMA, AutoETS
from statsforecast import StatsForecast

models = [
    MSTL(
        season_length=[7, 52 * 7],  # seasonalities of the time series
        trend_forecaster=AutoTheta(),  # trend forecaster
    )
]
sf = StatsForecast(
    models=models,  # model used to fit each time series
    freq="D",  # frequency of the data
    n_jobs=-1,
)
# sf = sf.fit(df=df_pd_tr)
logger.info("Cross validation started")
crossvalidation_df = sf.cross_validation(
    df=df_pd,
    h=56,  # Forecast horizon of 30 days
    step_size=7,  # Run forecasting process every 30 days
    n_windows=52,
)
logger.info("Cross validation finished")


# rename ds to datetime, cutoff to forecast_origin and MSTL to f
crossvalidation_df = crossvalidation_df.rename(
    columns={"ds": "datetime", "cutoff": "forecast_origin", "MSTL": "f"}
)
crossvalidation_df["h"] = (
    crossvalidation_df["datetime"] - crossvalidation_df["forecast_origin"]
).dt.days
# crossvalidation_df = crossvalidation_df[
#     (crossvalidation_df.datetime >= "2024-07-01")
#     & (crossvalidation_df.datetime <= "2024-09-21")
# ]


crossvalidation_df.to_excel("mstlautoTheta_crossvalidation_52w.xlsx", index=False)
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# df_score = df_pd_te[df_pd_te.horizon <= 7]
# df_score = df_pd_te
df_score = crossvalidation_df
mae = mean_absolute_error(df_score["y"], df_score["f"])
mape = mean_absolute_percentage_error(df_score["f"], df_score["f"])
# mae_snaive = mean_absolute_error(df_score[y_plot], df_score["y_orig_lagged"])
# mape_snaive = mean_absolute_percentage_error(
#     df_score[y_plot], df_score["y_orig_lagged"]
# )
print(f"MAE model: {mae}")
print(f"MAPE model: {mape}")
# print(f"MAE snaive: {mae_snaive}")
# print(f"MAPE snaive: {mape_snaive}")

# df_pl_sma_sorted = df_tab_filtered.sort(by=SERIES_CONF["KEY_COLS"] + ["datetime"])

print("Olá mundo")


# plot


# create a ploty graph with forecast vs actual
import plotly.graph_objects as go

# cumsum error by weekday
# df_pl = df_pd[["datetime", "y_orig_raw"]].drop_duplicates()
# df_pl = df_pd_te[df_pd_te.forecast_origin == "2024-08-01 00:00:00.000"]
df_pl = df_pd_te
# df_pl = df_pl[df_pl.datetime <= "2024-09-04 00:00:00.000"]
fig = go.Figure()

# Add trace for actual data (y)
# fig.add_trace(
#     go.Scatter(x=df_pl["datetime"], y=df_pl[y_plot], mode="lines", name="Actual")
# )

# # Add trace for forecast data (f)
fig.add_trace(go.Scatter(x=df_pl["ds"], y=df_pl[f_plot], mode="lines", name="Forecast"))

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
# drop values ds > '2024-09-21'
