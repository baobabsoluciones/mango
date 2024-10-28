from calendar import month

import pandas as pd
from Tools.demo.sortvisu import steps
from coreforecast.scalers import boxcox_lambda, boxcox, inv_boxcox
from statsmodels.base.transform import BoxCox

from mango_time_series.mango.features.date_utils import get_holidays_df
from mango_time_series.mango.timeseries_class.base_class import TimeSeriesProcessor
from mango_time_series.mango.timeseries_class.config_class import SeriesConfiguration

from mango_base.mango.logging.logger import get_basic_logger

logger = get_basic_logger()

# Load the data
df = pd.read_excel(
    r"G:\Unidades compartidas\clece_pmr_202207\proyecto\desarrollo\datos\time_series_synthetic_interpolate.xlsx"
)
df["aeropuerto"] = "BCN"
df = df[["datetime", "aeropuerto", "finalizadas"]]

# Create an instance of SeriesConfiguration directly
config = SeriesConfiguration(
    key_cols=["aeropuerto"],
    time_period_descr="day",
    time_col="datetime",
    value_col="finalizadas",
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
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NBEATSx

# Cargar datos
df_pd = df_pd.reset_index()
df_pd = df_pd[["aeropuerto", "datetime", "y"]]
hol = get_holidays_df(steps_back=7, steps_forward=7)
# replace nulls in hol with 99 with polars
hol = hol.fill_null(99).to_pandas()

df_pd  = pd.merge(df_pd, hol, on="datetime", how="left")
df_pd = df_pd.fillna(99)

df_pd = df_pd.rename(columns={"datetime": "ds", "aeropuerto": "unique_id"})


# Dividir en entrenamiento y prueba
date_end_train = "2024-07-27"
df_pd_tr = df_pd[df_pd["ds"] < date_end_train]
df_pd_te = df_pd[df_pd["ds"] >= date_end_train]

# get regressors names
regressors = df_pd_tr.columns.difference(["ds", "unique_id", "y"]).tolist()
# Definir el modelo NBEATS
model = NBEATSx(h=56, input_size=375,
               stack_types = ['identity', 'trend', 'seasonality'],
                futr_exog_list=regressors,
               max_steps=500,
               val_check_steps=10,
                scaler_type = 'robust'

            # loss='RMSE',
            #    early_stop_patience_steps=5
               )

logger.info("Model created")

# Define EarlyStopping callback to monitor `train_loss_epoch` instead of `ptl/val_loss`
nf = NeuralForecast(
    models=[model],
    freq='D',  # Frecuencia diaria
)

# Entrenar y predecir
# nf.fit(df=df_pd_tr, val_size=56, verbose=True)
# forecast_df = nf.predict()

# Cross validation
logger.info("Cross validation started")
crossvalidation_df = nf.cross_validation(
    df=df_pd,
    # h=56,
    step_size=7,
    n_windows=36,
)

logger.info("Cross validation finished")



# df_pd_na = df_pd_tr.dropna()
# aaa = fcst.preprocess(df_pd_na)
# cv_hist = cv.fit(
#     df_pd_na,
#     n_windows=4,
#     h=56,
#     params=lgb_params,
#     eval_every=5,
#     early_stopping_evals=10,
#     compute_cv_preds=True,
# # )
#
# # sf = sf.fit(df=df_pd_tr)
# logger.info("Cross validation started")
# crossvalidation_df = fcst.cross_validation(
#     df=df_pd,
#     n_windows=52,  # number of models to train/splits to perform
#     step_size=7,  # Run forecasting process every 30 days
#     h=56,  # length of the validation set in each window
#     xreg=fourier_terms
# )
# logger.info("Cross validation finished")
#
#
# # rename ds to datetime, cutoff to forecast_origin and MSTL to f
crossvalidation_df = crossvalidation_df.rename(
    columns={"ds": "datetime", "cutoff": "forecast_origin", "NBEATSx": "f"}
)
crossvalidation_df["h"] = (
    crossvalidation_df["datetime"] - crossvalidation_df["forecast_origin"]
).dt.days
# # crossvalidation_df = crossvalidation_df[
# #     (crossvalidation_df.datetime >= "2024-07-01")
# #     & (crossvalidation_df.datetime <= "2024-09-21")
# # ]
#
# # if unique_id has just one value, then drop it
# if len(crossvalidation_df["unique_id"].unique()) == 1:
#     crossvalidation_df.drop(columns=["unique_id"], inplace=True)
#
crossvalidation_df.to_excel("NBEATS.xlsx", index=False)
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
#
# # df_score = df_pd_te[df_pd_te.horizon <= 7]
# # df_score = df_pd_te
df_score = crossvalidation_df
# # drop rows with na
df_score = df_score.dropna()
mae = mean_absolute_error(df_score["y"], df_score["f"])
mape = mean_absolute_percentage_error(df_score["y"], df_score["f"])
# # mae_snaive = mean_absolute_error(df_score[y_plot], df_score["y_orig_lagged"])
# # mape_snaive = mean_absolute_percentage_error(
# #     df_score[y_plot], df_score["y_orig_lagged"]
# # )
# print(f"MAE model: {mae}")
# print(f"MAPE model: {mape}")
# # print(f"MAE snaive: {mae_snaive}")
# # print(f"MAPE snaive: {mape_snaive}")
#
# # df_pl_sma_sorted = df_tab_filtered.sort(by=SERIES_CONF["KEY_COLS"] + ["datetime"])
#
# print("Olá mundo")
#
#
# # plot
#
#
# # create a ploty graph with forecast vs actual
# import plotly.graph_objects as go
#
# # cumsum error by weekday
# # df_pl = df_pd[["datetime", "y_orig_raw"]].drop_duplicates()
# # df_pl = df_pd_te[df_pd_te.forecast_origin == "2024-08-01 00:00:00.000"]
# df_pl = df_pd_te
# # df_pl = df_pl[df_pl.datetime <= "2024-09-04 00:00:00.000"]
# fig = go.Figure()
#
# # Add trace for actual data (y)
# # fig.add_trace(
# #     go.Scatter(x=df_pl["datetime"], y=df_pl[y_plot], mode="lines", name="Actual")
# # )
#
# # # Add trace for forecast data (f)
# fig.add_trace(go.Scatter(x=df_pl["ds"], y=df_pl[f_plot], mode="lines", name="Forecast"))
#
# # Update layout for better visualization
# fig.update_layout(
#     title="Actual vs Forecast",
#     xaxis_title="Datetime",
#     yaxis_title="Values",
#     legend_title="Legend",
#     hovermode="x unified",
# )
# # y starts from 0
# fig.update_yaxes(range=[0, 1500])
#
# # Show plot
# fig.show()
# # drop values ds > '2024-09-21'
#
