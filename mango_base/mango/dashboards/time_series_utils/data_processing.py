import pandas as pd
import streamlit as st

from mango_time_series.mango.utils.processing import aggregate_to_input


@st.cache_data
def aggregate_to_input_cache(data, freq, SERIES_CONF):
    return aggregate_to_input(data, freq, SERIES_CONF)


def calculate_horizon(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Calculate the forecast horizon based on the selected frequency.
    :param df: pd.DataFrame with forecast data
    :param freq: str with the frequency
    :return: pd.DataFrame with an added column 'h' for the horizon
    """
    if freq == "h":
        df["h"] = (df["datetime"] - df["forecast_origin"]).dt.total_seconds() // 3600
    elif freq == "D":
        df["h"] = (df["datetime"] - df["forecast_origin"]).dt.days
    elif freq == "W":
        df["h"] = (df["datetime"] - df["forecast_origin"]).dt.days // 7
    elif freq == "MS":
        df["h"] = (df["datetime"].dt.year - df["forecast_origin"].dt.year) * 12 + (
            df["datetime"].dt.month - df["forecast_origin"].dt.month
        )
    elif freq == "QE":
        df["h"] = (df["datetime"].dt.year - df["forecast_origin"].dt.year) * 4 + (
            df["datetime"].dt.quarter - df["forecast_origin"].dt.quarter
        )
    elif freq == "YE":
        df["h"] = df["datetime"].dt.year - df["forecast_origin"].dt.year
    else:
        raise ValueError(f"Unsupported frequency: {freq}")

    return df


def process_data(data, columns_id, select_agr_tmp_dict, select_agr_tmp, UI_TEXT):
    data["datetime"] = pd.to_datetime(data["datetime"])
    if (
        st.session_state.get("visualization") == UI_TEXT["visualization_options"][0]
    ):  # "Exploration"
        time_series = data[["datetime", "y"]].drop_duplicates()
        forecast = None
    else:
        if "forecast_origin" in data.columns:
            data["forecast_origin"] = pd.to_datetime(data["forecast_origin"])

        time_series = data[columns_id + ["datetime", "y"]].drop_duplicates()
        forecast = data.copy()

        if "forecast_origin" in forecast.columns:
            forecast = aggregate_to_input_cache(
                forecast,
                freq=select_agr_tmp_dict[select_agr_tmp],
                SERIES_CONF={
                    "KEY_COLS": columns_id + ["forecast_origin", "model"],
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
            forecast = calculate_horizon(forecast, select_agr_tmp_dict[select_agr_tmp])

    time_series = aggregate_to_input(
        time_series,
        freq=select_agr_tmp_dict[select_agr_tmp],
        SERIES_CONF={"KEY_COLS": columns_id, "AGG_OPERATIONS": {"y": "sum"}},
    )

    return time_series, forecast


def calculate_min_diff_per_window(df):
    time_diffs = df["datetime"].diff().dropna().dt.total_seconds() / 86400
    time_diffs = abs(time_diffs)

    if time_diffs.min() >= 28:
        time_diffs = df["datetime"].diff().dropna().apply(lambda x: abs(x.days))
    min_time_diffs = time_diffs.min()
    return min_time_diffs
