import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import BoxCox, Diff
from darts.dataprocessing.pipeline import Pipeline
from darts.models import LightGBMModel
from darts.metrics import mae, mape
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import BoxCox

# Load the data
df = pd.read_excel(
    r"G:\Unidades compartidas\clece_pmr_202207\proyecto\desarrollo\datos\time_series_synthetic_interpolate.xlsx"
)
df["aeropuerto"] = "BCN"
df.drop(columns=["PNP", "TOTAL"], inplace=True)

# Convert to TimeSeries
series = TimeSeries.from_dataframe(df, time_col="Fecha_Vuelo", value_cols="FIN")


# BoxCox transformation
transformer = BoxCox()
transformer.fit(series)
series_transformed = transformer.transform(series)

# Create month feature
month_series = datetime_attribute_timeseries(series, attribute="month", one_hot=False)

# Split the data into training and testing sets
date_end_train = "2024-08-01"
train, test = series_transformed.split_before(pd.Timestamp(date_end_train))
past_cov = datetime_attribute_timeseries(train, attribute="month", one_hot=True)
future_cov = datetime_attribute_timeseries(test, attribute="month", one_hot=True)

# Train AutoLGBM model with specified lags and month as features
model = LightGBMModel(
    lags=[-1, -2, -7, -14, -364],
    lags_future_covariates=list(range(-7, 53)),
    output_chunk_length=len(test),
)
model.fit(train, future_covariates=past_cov)

# Forecast
# concat last 7 values of past_cov with future_cov
future_cov2 = past_cov[-7:].append(future_cov)
forecast = model.predict(len(test), future_covariates=future_cov2)

# Inverse transformations
forecast = transformer.inverse_transform(forecast)
test = transformer.inverse_transform(test)

# Evaluate the model
mae_val = mae(test, forecast)
mape_val = mape(test, forecast)
print(f"MAE: {mae_val}")
print(f"MAPE: {mape_val}")

# create a dataframe with df_test and a column with the forecast "f"
df_test = test.pd_dataframe()
df_test["f"] = forecast.values()
# index as col
df_test.reset_index(inplace=True)
# rename to datetime, y and f
df_test = df_test.rename(columns={"Fecha_Vuelo": "datetime", "FIN": "y"})
df_test["forecast_origin"] = pd.Timestamp(date_end_train)
# to csv no index
df_test.to_csv("forecast_LGBM2.csv", index=False)
