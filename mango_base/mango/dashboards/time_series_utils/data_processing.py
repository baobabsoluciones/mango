import pandas as pd
from mango_time_series.mango.utils.processing import aggregate_to_input
import streamlit as st

@st.cache_data
def aggregate_to_input_cache(data, freq, SERIES_CONF):
    return aggregate_to_input(data, freq, SERIES_CONF)

def process_data(data, columns_id, select_agr_tmp_dict, select_agr_tmp):
    # Convert forecast_origin to datetime
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["forecast_origin"] = pd.to_datetime(data["forecast_origin"])

    time_series = data[columns_id + ["datetime", "y"]].drop_duplicates()
    forecast = data.copy()

    # Aggregate the data
    time_series = aggregate_to_input(
        time_series,
        freq=select_agr_tmp_dict[select_agr_tmp],
        SERIES_CONF={"KEY_COLS": columns_id, "AGG_OPERATIONS": {"y": "sum"}},
    )
    forecast = aggregate_to_input_cache(
        forecast,
        freq=select_agr_tmp_dict[select_agr_tmp],
        SERIES_CONF={
            "KEY_COLS": columns_id + ["forecast_origin", "h"],
            "AGG_OPERATIONS": {
                "y": "sum",
                "f": "sum",
                "err": "mean",
                "abs_err": "mean",
                "perc_err": "mean",
                "perc_abs_err": "mean",
            },
        },
    )
    return time_series, forecast