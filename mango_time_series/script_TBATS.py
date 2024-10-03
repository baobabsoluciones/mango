import pandas as pd
from statsforecast.tbats import tbats_selection, tbats_forecast

from mango_time_series.mango.timeseries_class.config_class import SeriesConfiguration
from mango_time_series.mango.timeseries_class.base_class import TimeSeriesProcessor

# Load the data
df = pd.read_excel(
    r"G:\Unidades compartidas\clece_pmr_202207\proyecto\desarrollo\datos\time_series_synthetic_interpolate.xlsx"
)
df["aeropuerto"] = "BCN"
df.drop(columns=["PNP", "TOTAL"], inplace=True)

# Create an instance of SeriesConfiguration directly
config = SeriesConfiguration(
    key_cols=["aeropuerto"],
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
df_pd = df_pd[["aeropuerto", "datetime", "y"]]
df_pd = df_pd.rename(columns={"datetime": "ds", "aeropuerto": "unique_id"})

# Split the data into training and testing sets
date_end_train = "2024-09-01"
df_pd_tr = df_pd[df_pd["ds"] < date_end_train]
df_pd_te = df_pd[df_pd["ds"] >= date_end_train]

# TBATS model selection and forecasting
from statsforecast.models import MSTL, AutoTheta
from statsforecast import StatsForecast
import numpy as np

seasonal_periods = np.array([7, 7*52])
use_boxcox = None
bc_lower_bound = 0
bc_upper_bound = 1
use_trend = None
use_damped_trend = None
use_arma_errors = True

y = df_pd_tr["y"].to_numpy()
mod = tbats_selection(y, seasonal_periods, use_boxcox, bc_lower_bound, bc_upper_bound, use_trend, use_damped_trend, use_arma_errors)



h = 34  # Forecast horizon
fcst = tbats_forecast(mod, h)
forecast = fcst['mean']

if mod['BoxCox_lambda'] is not None:
    from coreforecast.scalers import inv_boxcox
    forecast = inv_boxcox(forecast, mod['BoxCox_lambda'])

df_pd_te["f"] = forecast

models = [MSTL(
    season_length=[7, 52 * 7],
    trend_forecaster=AutoTheta()
)]
sf = StatsForecast(
    models=models,
    freq='D',
    n_jobs=-1,
)
sf = sf.fit(df=df_pd_tr)

a = sf.predict(h=34)
a = a.reset_index()
a.index = df_pd_te.index
df_pd_te["f"] = a["MSTL"]

df_pd_te["err"] = df_pd_te["f"] - df_pd_te["y"]
df_pd_te = df_pd_te.sort_values(["datetime", "forecast_origin"])

df_pd_te["prev_err"] = df_pd_te.groupby(["weekday", "forecast_origin"])["err"].shift(1)
df_pd_te["cumsum_err"] = df_pd_te.groupby(["weekday", "forecast_origin"])["prev_err"].cumsum()
arr = df_pd_te["f_orig"].to_numpy()
df_pd_te["f_orig2"] = transformer.inverse_transform(arr)

df_pd_te["f_orig"] = df_pd_te["f"] + df_pd_te["y_orig_lagged"]

import plotly.graph_objects as go

df_pl = df_pd_te
fig = go.Figure()

fig.add_trace(
    go.Scatter(x=df_pl["ds"], y=df_pl["f"], mode="lines", name="Forecast")
)

fig.update_layout(
    title="Actual vs Forecast",
    xaxis_title="Datetime",
    yaxis_title="Values",
    legend_title="Legend",
    hovermode="x unified",
)
fig.update_yaxes(range=[0, 1500])
fig.show()

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

df_score = df_pl
mae = mean_absolute_error(df_score["y"], df_score["f"])
mape = mean_absolute_percentage_error(df_score["y"], df_score["f"])
mae_snaive = mean_absolute_error(df_score["y"], df_score["y_orig_lagged"])
mape_snaive = mean_absolute_percentage_error(df_score["y"], df_score["y_orig_lagged"])
print(f"MAE model: {mae}")
print(f"MAPE model: {mape}")
print(f"MAE snaive: {mae_snaive}")
print(f"MAPE snaive: {mape_snaive}")

print("Olá mundo")