import importlib.resources as pkg_resources
from typing import List, Dict

import jinja2
import pandas as pd
import streamlit as st
from mango_time_series.utils.processing_time_series import aggregate_to_input

from .constants import all_imports


@st.cache_data
def aggregate_to_input_cache(data, freq, series_conf):
    """Cache wrapper for aggregate_to_input function.

    This function provides caching for the aggregate_to_input operation
    to improve performance when processing time series data repeatedly.

    :param data: Time series data to aggregate
    :type data: pd.DataFrame
    :param freq: Frequency for aggregation
    :type freq: str
    :param series_conf: Configuration for series aggregation
    :type series_conf: dict
    :return: Aggregated time series data
    :rtype: pd.DataFrame
    """
    return aggregate_to_input(data, freq, series_conf)


@st.cache_data
def convert_df(df: pd.DataFrame):
    """Convert DataFrame to CSV format with UTF-8 encoding.

    This function converts a pandas DataFrame to CSV format and encodes it
    as UTF-8 bytes. The result is cached to prevent repeated computation
    on every Streamlit rerun.

    :param df: DataFrame to convert to CSV
    :type df: pd.DataFrame
    :return: CSV data encoded as UTF-8 bytes
    :rtype: bytes
    """
    return df.to_csv(index=False).encode("utf-8")


def calculate_horizon(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Calculate the forecast horizon based on the selected frequency.

    This function calculates the forecast horizon (h) by computing the time
    difference between the forecast datetime and the forecast origin.
    The calculation method depends on the frequency:
    - Hourly (h): Difference in hours
    - Daily (D): Difference in days
    - Weekly (W): Difference in weeks
    - Monthly (MS): Difference in months
    - Quarterly (QE): Difference in quarters
    - Yearly (YE): Difference in years

    :param df: DataFrame with forecast data containing 'datetime' and 'forecast_origin' columns
    :type df: pd.DataFrame
    :param freq: Frequency code for horizon calculation
    :type freq: str
    :return: DataFrame with added 'h' column containing horizon values
    :rtype: pd.DataFrame
    :raises ValueError: If the frequency is not supported

    Example:
        >>> df = pd.DataFrame({
        ...     'datetime': ['2023-01-02', '2023-01-03'],
        ...     'forecast_origin': ['2023-01-01', '2023-01-01']
        ... })
        >>> result = calculate_horizon(df, 'D')
        >>> print(result['h'].tolist())
        [1, 2]
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
    """Process input data for time series analysis and visualization.

    This function processes time series data based on the selected visualization
    type (exploration or forecast). It handles data aggregation, datetime conversion,
    and prepares data for both time series analysis and forecast evaluation.

    For exploration mode, it returns aggregated time series data.
    For forecast mode, it additionally processes forecast data with error metrics
    and calculates forecast horizons.

    :param data: Input DataFrame containing time series data
    :type data: pd.DataFrame
    :param columns_id: List of column names to use as identifiers
    :type columns_id: List[str]
    :param select_agr_tmp_dict: Dictionary mapping aggregation options to frequency codes
    :type select_agr_tmp_dict: Dict
    :param select_agr_tmp: Selected aggregation option key
    :type select_agr_tmp: str
    :param ui_text: Dictionary containing UI text and visualization options
    :type ui_text: Dict
    :return: Tuple containing (time_series_data, forecast_data)
    :rtype: tuple[pd.DataFrame, pd.DataFrame | None]

    Example:
        >>> data = pd.DataFrame({
        ...     'datetime': ['2023-01-01', '2023-01-02'],
        ...     'y': [100, 110],
        ...     'model': ['model1', 'model1']
        ... })
        >>> time_series, forecast = process_data(
        ...     data, ['model'], {'daily': 'D'}, 'daily', ui_text
        ... )
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
    """Calculate the minimum time difference between consecutive timestamps.

    This function calculates the minimum time difference between consecutive
    timestamps in a time series. It handles different time scales automatically:
    - For differences >= 28 days, it uses day-based calculation
    - For smaller differences, it uses second-based calculation converted to days

    :param df: DataFrame containing a 'datetime' column with timestamps
    :type df: pd.DataFrame
    :return: Minimum time difference in days (or days for large differences)
    :rtype: float

    Example:
        >>> df = pd.DataFrame({
        ...     'datetime': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-05'])
        ... })
        >>> min_diff = calculate_min_diff_per_window(df)
        >>> print(min_diff)
        1.0
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
    """Render a Python script from a Jinja2 template for forecast generation.

    This function generates a Python script by rendering a Jinja2 template
    with the provided parameters. The template is used to create forecast
    generation scripts with the specified models, horizon, and configuration.

    :param models: List of model names to include in the script
    :type models: List[str]
    :param horizon: Forecast horizon (number of periods to forecast)
    :type horizon: int
    :param step_size: Step size for rolling window forecasts
    :type step_size: int
    :param n_windows: Number of forecast windows
    :type n_windows: int
    :param freq_code: Frequency code for the time series
    :type freq_code: str
    :return: Rendered Python script as a string
    :rtype: str

    Example:
        >>> script = render_script(
        ...     models=['ARIMA', 'Prophet'],
        ...     horizon=30,
        ...     step_size=7,
        ...     n_windows=4,
        ...     freq_code='D'
        ... )
        >>> print(script[:100])
        # Generated forecast script...
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
