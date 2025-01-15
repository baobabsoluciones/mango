import importlib.resources as pkg_resources
from typing import List, Dict

import jinja2
import pandas as pd
import streamlit as st

from mango_time_series.utils.processing_time_series import aggregate_to_input
from .constants import all_imports


@st.cache_data
def aggregate_to_input_cache(data, freq, series_conf):
    return aggregate_to_input(data, freq, series_conf)


@st.cache_data
def convert_df(df: pd.DataFrame):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode("utf-8")


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


def process_data(
    data: pd.DataFrame,
    columns_id: List,
    select_agr_tmp_dict: Dict,
    select_agr_tmp: str,
    ui_text: Dict,
):
    """
    Process the input data for the time series analysis.
    :param data: pd.DataFrame with the input data
    :param columns_id: list of str with the columns to use as identifiers
    :param select_agr_tmp_dict: dict with the aggregation options
    :param select_agr_tmp: str with the selected aggregation option
    :param ui_text: dict with the UI text
    :return: tuple with the aggregated time series and forecast data
    """
    data["datetime"] = pd.to_datetime(data["datetime"])
    if st.session_state.get("visualization") == ui_text["visualization_options"][0]:
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
                series_conf={
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
        series_conf={"KEY_COLS": columns_id, "AGG_OPERATIONS": {"y": "sum"}},
    )

    return time_series, forecast


def calculate_min_diff_per_window(df: pd.DataFrame) -> float:
    """
    Calculate the minimum time difference between consecutive timestamps in a time series.
    :param df: pd.DataFrame with a 'datetime' column
    :return: float with the minimum time difference
    """
    time_diffs = df["datetime"].diff().dropna().dt.total_seconds() / 86400
    time_diffs = abs(time_diffs)

    if time_diffs.min() >= 28:
        time_diffs = df["datetime"].diff().dropna().apply(lambda x: abs(x.days))
    min_time_diffs = time_diffs.min()
    return min_time_diffs


def render_script(
    models: List, horizon: int, step_size: int, n_windows: int, freq_code: str
):
    """
    Render the script for the forecast template.
    :param models: list of str with the model names
    :param horizon: int with the forecast horizon
    :param step_size: int with the step size
    :param n_windows: int with the number of windows
    :param freq_code: str with the frequency code
    :return: str with the rendered script
    """
    with pkg_resources.path(
        "mango_time_series.dashboards.time_series_utils", "forecast_template.py.j2"
    ) as template_path:
        templateLoader = jinja2.FileSystemLoader(searchpath=template_path.parent)
        templateEnv = jinja2.Environment(loader=templateLoader)

        template = templateEnv.get_template("forecast_template.py.j2")

        output_text = template.render(
            models=models,
            all_imports=all_imports,
            horizon=horizon,
            step_size=step_size,
            n_windows=n_windows,
            freq_code=freq_code,
        )
    return output_text
