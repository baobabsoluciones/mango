import copy
from typing import List, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf
from streamlit_date_picker import date_range_picker, PickerType

from .data_processing import (
    calculate_min_diff_per_window,
)
from ...time_series.decomposition import SeasonalityDecompose
from ...time_series.seasonal import SeasonalityDetector
from ...utils.utils import cast_env_to_bool


def select_series(data: pd.DataFrame, columns: List, ui_text: Dict):
    """
    Create a sidebar to select a series from the DataFrame based on the values in the specified columns.
    :param data: The DataFrame containing the time series data.
    :param columns: The columns to use for filtering the series.
    :param ui_text: The dictionary containing the UI text.
    """
    st.sidebar.title(ui_text["select_series"])
    filtered_data = data.copy()
    for column in columns:
        filter_values = filtered_data[column].unique()

        # Create a selectbox for each filter
        selected_value = st.sidebar.selectbox(
            ui_text["choose_column"].format(column),
            filter_values,
            key=f"selectbox_{column}",
        )

        # Filter the DataFrame by the selected value
        filtered_data = filtered_data[filtered_data[column] == selected_value].copy()

    # Create a label for the selected series
    selected_item = {col: filtered_data.iloc[0][col] for col in columns}
    return selected_item


def plot_time_series(
    time_series: pd.DataFrame,
    selected_series: List,
    select_agr_tmp_dict: Dict,
    select_agr_tmp: str,
    ui_text: Dict,
):
    """
    Plot the selected time series data. The user can choose between different types of plots:
    - Original series: Plot the original time series data.
    - Series by year: Plot the time series data for each year.
    - STL: Plot the original series and its STL decomposition.
    - Lag analysis: Plot the ACF and PACF of the time series data.
    - Seasonality boxplot: Plot boxplots of the time series data by day of year, day of week, or month.
    - Periodogram: Plot the periodogram of the time series data.
    :param time_series: The DataFrame containing the time series data.
    :param selected_series: The list of selected series to plot.
    :param select_agr_tmp_dict: The dictionary mapping the temporal grouping options to their corresponding frequency.
    :param select_agr_tmp: The selected temporal grouping option.
    :param ui_text: The dictionary containing the UI text.
    """
    select_plot = st.selectbox(
        ui_text["choose_plot"],
        ui_text["plot_options"],
        label_visibility="collapsed",
    )
    seasonality_decompose = SeasonalityDecompose()
    seasonality_detector = SeasonalityDetector()

    # "Original series"
    if select_plot == ui_text["plot_options"][0]:
        date_range = date_range_picker(
            picker_type=PickerType.date,
            start=time_series["datetime"].min(),
            end=time_series["datetime"].max(),
            key="date_range_1",
        )
        if date_range:
            date_start = pd.to_datetime(date_range[0])
            date_end = pd.to_datetime(date_range[1])
        else:
            date_start = time_series["datetime"].min()
            date_end = time_series["datetime"].max()
        for serie in selected_series:
            selected_data = time_series.copy()

            filter_cond = f"datetime >= '{date_start}' & datetime <= '{date_end}' & "
            for col, col_value in serie.items():
                filter_cond += f"{col} == '{col_value}' & "

            # Remove the last & from the filter condition
            filter_cond = filter_cond[:-3]
            selected_data = selected_data.query(filter_cond)
            st.plotly_chart(
                px.line(
                    selected_data,
                    x="datetime",
                    y="y",
                    title="-".join(serie.values()),
                ),
                use_container_width=True,
            )
    # "Series by year"
    elif select_plot == ui_text["plot_options"][1]:
        st.markdown(
            """
            <style>
            /* Cambiar el color de las etiquetas seleccionadas en el multiselect */
            .stMultiSelect [data-baseweb="tag"] {
                background-color: #66b3ff !important;  /* Color azul */
                color: white !important;  /* Texto en blanco */
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        options = st.multiselect(
            ui_text["choose_years"],
            sorted(time_series["datetime"].dt.year.unique(), reverse=True),
        )
        for serie in selected_series:
            selected_data = time_series.copy()
            filter_cond = ""
            for col, col_value in serie.items():
                filter_cond += f"{col} == '{col_value}' & "

            # Remove the last & from the filter condition
            filter_cond = filter_cond[:-3]
            if filter_cond:
                selected_data = selected_data.query(filter_cond)
            for year in reversed(sorted(options)):
                selected_data_year = selected_data.query(f"datetime.dt.year == {year}")
                st.plotly_chart(
                    px.line(
                        selected_data_year,
                        x="datetime",
                        y="y",
                        title=f"{'-'.join(serie.values())} - {year}",
                    ),
                    use_container_width=True,
                )
    # STL
    elif select_plot == ui_text["plot_options"][2]:
        if not cast_env_to_bool("ENABLE_EXPERIMENTAL_FEATURES", default=False):
            st.warning(ui_text["experimental_features_warning"], icon="⚠️")
            return
        else:
            st.info(ui_text["experimental_features_info"], icon="ℹ️")
        for serie in selected_series:
            selected_data = time_series.copy()
            filter_cond = ""
            for col, col_value in serie.items():
                filter_cond += f"{col} == '{col_value}' & "

            # Remove the last & from the filter condition
            filter_cond = filter_cond[:-3]
            if filter_cond:
                selected_data = selected_data.query(filter_cond)
            selected_data_stl = selected_data.set_index("datetime")
            selected_data_stl = selected_data_stl.asfreq(
                select_agr_tmp_dict[select_agr_tmp]
            )
            ts_data = selected_data_stl["y"].ffill().values
            detected_periods = seasonality_detector.detect_seasonality(ts=ts_data)
            if detected_periods:
                st.write(f"{ui_text['stl']['periods_detected']} {detected_periods}")
            else:
                st.write(ui_text["stl"]["periods_detec"])

            try:
                if len(detected_periods) == 1:
                    # STL decomposition if only one period is detected
                    trend, seasonal, resid = seasonality_decompose.decompose_stl(
                        selected_data_stl["y"].ffill(), period=detected_periods[0]
                    )
                elif len(detected_periods) > 1:
                    # MSTL decomposition if multiple periods are detected
                    trend, seasonal, resid = seasonality_decompose.decompose_mstl(
                        selected_data_stl["y"].ffill(), periods=detected_periods
                    )
                else:
                    st.write(ui_text["stl"]["periods_detected"])
                    continue
            except ValueError:
                st.write(ui_text["stl"]["no_periods_detec"])
                continue

            fig1 = px.line(
                selected_data_stl["y"],
                title=ui_text["stl"]["stl_components"]["original"],
            )
            fig2 = px.line(trend, title=ui_text["stl"]["stl_components"]["trend"])
            fig3 = px.line(seasonal, title=ui_text["stl"]["stl_components"]["seasonal"])
            fig4 = px.line(resid, title=ui_text["stl"]["stl_components"]["residual"])
            # Put each plot in a subplot
            fig = sp.make_subplots(
                rows=4,
                cols=1,
                subplot_titles=[
                    ui_text["stl"]["stl_components"]["original"],
                    ui_text["stl"]["stl_components"]["trend"],
                    ui_text["stl"]["stl_components"]["seasonal"],
                    ui_text["stl"]["stl_components"]["residual"],
                ],
                shared_xaxes=True,
            )
            fig.add_trace(fig1.data[0], row=1, col=1)
            fig.add_trace(fig2.data[0], row=2, col=1)
            fig.add_trace(fig3.data[0], row=3, col=1)
            fig.add_trace(fig4.data[0], row=4, col=1)
            fig.update_layout(showlegend=False, title="-".join(serie.values()))

            fig.update_layout(
                showlegend=False,
                height=900,
                width=800,
            )

            st.plotly_chart(fig, use_container_width=True)
    # "Lag analysis"
    elif select_plot == ui_text["plot_options"][3]:
        for serie in selected_series:
            selected_data = time_series.copy()
            filter_cond = ""
            for col, col_value in serie.items():
                filter_cond += f"{col} == '{col_value}' & "

            # Remove the last & from the filter condition
            filter_cond = filter_cond[:-3]
            if filter_cond:
                selected_data = selected_data.query(filter_cond)

            if "uid" in selected_data.columns:
                selected_data = selected_data.drop(columns=["uid"])
            selected_data_lags = selected_data.set_index("datetime")
            selected_data_lags = selected_data_lags.asfreq(
                select_agr_tmp_dict[select_agr_tmp]
            )
            max_lags = int(len(selected_data_lags) / 2) - 1
            acf_array = acf(
                selected_data_lags.dropna(), nlags=min(35, max_lags), alpha=0.05
            )
            pacf_array = pacf(
                selected_data_lags.dropna(), nlags=min(35, max_lags), alpha=0.05
            )

            acf_lower_y = acf_array[1][:, 0] - acf_array[0]
            acf_upper_y = acf_array[1][:, 1] - acf_array[0]
            pacf_lower_y = pacf_array[1][:, 0] - pacf_array[0]
            pacf_upper_y = pacf_array[1][:, 1] - pacf_array[0]

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                subplot_titles=(
                    ui_text["lag_analysis"]["lag_analysis_components"]["acf"],
                    ui_text["lag_analysis"]["lag_analysis_components"]["pacf"],
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(acf_array[0])),
                    y=acf_array[0],
                    mode="markers",
                    marker=dict(color="#1f77b4", size=12),
                    name=ui_text["lag_analysis"]["lag_analysis_components"]["acf"],
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(acf_array[0])),
                    y=acf_upper_y,
                    mode="lines",
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(acf_array[0])),
                    y=acf_lower_y,
                    mode="lines",
                    fillcolor="rgba(32, 146, 230,0.3)",
                    fill="tonexty",
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            # Adding PACF subplot
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(pacf_array[0])),
                    y=pacf_array[0],
                    mode="markers",
                    marker=dict(color="#1f77b4", size=12),
                    name=ui_text["lag_analysis"]["lag_analysis_components"]["pacf"],
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(pacf_array[0])),
                    y=pacf_upper_y,
                    mode="lines",
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(pacf_array[0])),
                    y=pacf_lower_y,
                    mode="lines",
                    fillcolor="rgba(32, 146, 230,0.3)",
                    fill="tonexty",
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            for i in range(len(acf_array[0])):
                fig.add_shape(
                    type="line",
                    x0=i,
                    y0=0,
                    x1=i,
                    y1=acf_array[0][i],
                    line=dict(color="grey", width=1),
                    row=1,
                    col=1,
                )

            for i in range(len(pacf_array[0])):
                fig.add_shape(
                    type="line",
                    x0=i,
                    y0=0,
                    x1=i,
                    y1=pacf_array[0][i],
                    line=dict(color="grey", width=1),
                    row=2,
                    col=1,
                )

            fig.update_layout(showlegend=False, title="-".join(serie.values()))

            fig.update_layout(
                showlegend=False,
                height=900,
                width=800,
            )

            st.plotly_chart(fig, use_container_width=True)

    # "Seasonality boxplot"
    elif select_plot == ui_text["plot_options"][4]:
        selected_granularity = select_agr_tmp

        if selected_granularity == ui_text["daily"]:
            freq_options = ui_text["frequency_options"]
        elif selected_granularity == ui_text["weekly"]:
            freq_options = ui_text["frequency_options"][1:]
        elif selected_granularity == ui_text["monthly"]:
            freq_options = [ui_text["frequency_options"][2]]
        else:
            st.write(ui_text["boxplot_error"])
            return
        selected_freq = st.selectbox(ui_text["select_frequency"], freq_options)

        for serie in selected_series:
            selected_data = time_series.copy()
            filter_cond = ""
            for col, col_value in serie.items():
                filter_cond += f"{col} == '{col_value}' & "

            # Remove the last & from the filter condition
            filter_cond = filter_cond[:-3]
            if filter_cond:
                selected_data = selected_data.query(filter_cond)

            selected_data = selected_data.set_index("datetime")
            selected_data.index = pd.to_datetime(selected_data.index)

            if selected_freq == ui_text["frequency_options"][0]:
                selected_data["day_of_year"] = selected_data.index.dayofyear
                fig = px.box(
                    selected_data,
                    x="day_of_year",
                    y="y",
                    title=ui_text["boxplot_titles"]["daily"],
                )
                st.plotly_chart(fig, use_container_width=True)

            elif selected_freq == ui_text["frequency_options"][1]:
                selected_data["day_of_week"] = selected_data.index.weekday
                fig = px.box(
                    selected_data,
                    x="day_of_week",
                    y="y",
                    title=ui_text["boxplot_titles"]["weekly"],
                )
                st.plotly_chart(fig, use_container_width=True)

            elif selected_freq == ui_text["frequency_options"][2]:
                selected_data["month"] = selected_data.index.month
                fig = px.box(
                    selected_data,
                    x="month",
                    y="y",
                    title=ui_text["boxplot_titles"]["monthly"],
                )
                st.plotly_chart(fig, use_container_width=True)

    # "Periodogram"
    elif select_plot == ui_text["plot_options"][5]:
        if not cast_env_to_bool("ENABLE_EXPERIMENTAL_FEATURES", default=False):
            st.warning(ui_text["experimental_features_warning"], icon="⚠️")
            return
        else:
            st.info(ui_text["experimental_features_info"], icon="ℹ️")
        for serie in selected_series:
            selected_data = time_series.copy()
            filter_cond = ""
            for col, col_value in serie.items():
                filter_cond += f"{col} == '{col_value}' & "

            # Delete the last & from the filter condition
            filter_cond = filter_cond[:-3]
            if filter_cond:
                selected_data = selected_data.query(filter_cond)
            selected_data_periodogram = selected_data.set_index("datetime")
            selected_data_periodogram = selected_data_periodogram.asfreq(
                select_agr_tmp_dict[select_agr_tmp]
            )
            ts_data = selected_data_periodogram["y"].ffill().values

            significant_periods, filtered_periods, filtered_power_spectrum = (
                seasonality_detector.detect_seasonality_periodogram(
                    ts=ts_data,
                    min_period=2,
                    max_period=365,
                )
            )
            st.write(
                f"{ui_text['periodogram']['significant_periods']} {significant_periods}"
            )
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=filtered_periods,
                    y=filtered_power_spectrum,
                    mode="lines",
                    name=ui_text["periodogram"]["power_spectrum"],
                )
            )

            threshold_value = np.percentile(filtered_power_spectrum, 99)
            fig.add_shape(
                type="line",
                x0=filtered_periods.min(),
                x1=filtered_periods.max(),
                y0=threshold_value,
                y1=threshold_value,
                line=dict(color="red", dash="dash"),
            )
            fig.add_annotation(
                x=filtered_periods.mean(),
                y=threshold_value,
                text=ui_text["periodogram"]["percentile_threshold"],
                showarrow=False,
                yshift=10,
                font=dict(color="red"),
            )

            for period in significant_periods:
                fig.add_shape(
                    type="line",
                    x0=period,
                    y0=0,
                    x1=period,
                    y1=max(filtered_power_spectrum),
                    line=dict(color="green", dash="dot"),
                )
                fig.add_annotation(
                    x=period,
                    y=max(filtered_power_spectrum) * 0.9,
                    text=f"{period:.1f}",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-40,
                    font=dict(color="green"),
                )

            fig.update_layout(
                title=ui_text["periodogram"]["title"],
                xaxis_title=ui_text["periodogram"]["xaxis_title"],
                yaxis_title=ui_text["periodogram"]["yaxis_title"],
                showlegend=False,
            )

            st.plotly_chart(fig, use_container_width=True)


def setup_sidebar(time_series: pd.DataFrame, columns_id: List, ui_text: Dict):
    """
    Set up the sidebar for the time series analysis dashboard.
    :param time_series: The DataFrame containing the time series data.
    :param columns_id: The list of columns to use as identifiers for the series.
    :param ui_text: The dictionary containing the UI text.
    :return: The selected temporal grouping option, the selected visualization option, and the list of visualization options.
    """
    st.sidebar.title(ui_text["sidebar_title"])

    if "forecast" in st.session_state and st.session_state["forecast"] is not None:
        if (
            ui_text["visualization_options"][1]
            not in st.session_state["visualization_options"]
        ):
            st.session_state["visualization_options"].append(
                ui_text["visualization_options"][1]
            )

    elif "forecast_origin" not in time_series.columns or "f" not in time_series.columns:
        st.session_state["visualization_options"] = [
            ui_text["visualization_options"][0]
        ]

    else:
        st.session_state["visualization_options"] = ui_text["visualization_options"]

    visualization = st.sidebar.radio(
        ui_text["select_visualization"],
        ui_text["visualization_options"],
    )

    st.sidebar.title(ui_text["select_temporal_grouping"])
    all_tmp_agr = copy.deepcopy(ui_text["temporal_grouping_options"])

    if "forecast_origin" in time_series.columns and "f" in time_series.columns:
        min_diff_per_window = time_series.groupby("forecast_origin").apply(
            calculate_min_diff_per_window
        )
        min_diff = min_diff_per_window.min()

        if min_diff >= 1:
            all_tmp_agr.remove(ui_text["hourly"])
        if min_diff >= 7:
            all_tmp_agr.remove(ui_text["daily"])
        if min_diff >= 28:
            all_tmp_agr.remove(ui_text["weekly"])
        if min_diff >= 90:
            all_tmp_agr.remove(ui_text["monthly"])
        if min_diff >= 365:
            all_tmp_agr.remove(ui_text["quarterly"])

        if len(all_tmp_agr) == 0:
            st.write(ui_text["temporal_analysis_error"])
            return

        select_agr_tmp = st.sidebar.selectbox(
            ui_text["select_temporal_grouping"],
            all_tmp_agr,
            label_visibility="collapsed",
        )
    else:
        if "uid" in time_series.columns:
            time_series = time_series.sort_values(by=["uid", "datetime"])
        min_diff_per_window = calculate_min_diff_per_window(time_series)
        min_diff = min_diff_per_window.min()

        if min_diff >= 1:
            all_tmp_agr.remove(ui_text["hourly"])
        if min_diff >= 7:
            all_tmp_agr.remove(ui_text["daily"])
        if min_diff >= 28:
            all_tmp_agr.remove(ui_text["weekly"])
        if min_diff >= 90:
            all_tmp_agr.remove(ui_text["monthly"])
        if min_diff >= 365:
            all_tmp_agr.remove(ui_text["quarterly"])

        if len(all_tmp_agr) == 0:
            st.write(ui_text["temporal_analysis_error"])
            return

        select_agr_tmp = st.sidebar.selectbox(
            ui_text["select_temporal_grouping"],
            all_tmp_agr,
            label_visibility="collapsed",
        )

    if columns_id:
        time_series[columns_id] = time_series[columns_id].astype(str)
        # Setup select series
        selected_item = select_series(time_series, columns_id, ui_text)
        if st.sidebar.button(ui_text["add_selected_series"]):
            # Avoid adding duplicates
            if selected_item not in st.session_state["selected_series"]:
                st.session_state["selected_series"].append(selected_item)
            else:
                st.toast(ui_text["series_already_added"], icon="❌")
        # Display the selected series in the sidebar with remove button
        for idx, serie in enumerate(st.session_state["selected_series"]):
            serie = "-".join(serie.values())
            col1, col2 = st.sidebar.columns([8, 2])
            with col1:
                st.write(f"{idx + 1}. {serie}")
            with col2:
                if st.button("❌", key=f"remove_{idx}"):
                    st.session_state["selected_series"].pop(idx)
                    st.rerun()
        # Remove all selected series
        if st.session_state["selected_series"]:
            if st.sidebar.button(ui_text["remove_all_series"]):
                st.session_state["selected_series"] = []
                st.rerun()
    else:
        st.sidebar.write(ui_text["no_columns_to_filter"])
        st.session_state["selected_series"] = [{}]

    return (
        select_agr_tmp,
        visualization,
        st.session_state["visualization_options"],
        st.session_state["selected_series"],
    )


def plot_forecast(forecast: pd.DataFrame, selected_series: List, ui_text: Dict):
    """
    Plot the forecast for the selected series.
    :param forecast: The DataFrame containing the forecast data.
    :param selected_series: The list of selected series to plot.
    :param ui_text: The dictionary containing the UI text.
    """
    st.subheader(ui_text["forecast_plot_title"])
    if not selected_series:
        st.write(ui_text["select_series_to_plot"])
        return
    # Get dates for the selected series
    filter_cond = ""
    for serie in selected_series:
        filter_cond_serie = "("
        for col, col_value in serie.items():
            filter_cond_serie += f"{col} == '{col_value}' & "

        # Remove the last & from the filter condition
        filter_cond_series = filter_cond_serie[:-3] + ")"
        filter_cond += filter_cond_series + " | "
    filter_cond = filter_cond[:-3]
    if filter_cond != ")":
        forecast_restricted = forecast.query(filter_cond)
    else:
        forecast_restricted = forecast.copy()

    selected_date = st.date_input(
        ui_text["choose_date"],
        min_value=forecast_restricted["forecast_origin"].min(),
        max_value=forecast_restricted["forecast_origin"].max(),
        value=forecast_restricted["forecast_origin"].min(),
        label_visibility="collapsed",
    )
    for serie in selected_series:
        filter_cond = ""
        for col, col_value in serie.items():
            filter_cond += f"{col} == '{col_value}' & "

        # Remove the last & from the filter condition
        filter_cond = filter_cond[:-3]
        if filter_cond:
            selected_data = forecast_restricted.query(filter_cond)
        else:
            selected_data = forecast_restricted.copy()

        selected_data = selected_data[
            selected_data["forecast_origin"] == pd.to_datetime(selected_date)
        ]
        # Plot both real and predicted values x datetime shared y axis y and f column
        # weekday from datetime

        selected_data["weekday"] = selected_data["datetime"].dt.dayofweek
        # Use ui_text for DAY_NAME_DICT
        selected_data["weekday"] = selected_data["weekday"].map(
            ui_text["DAY_NAME_DICT"]
        )

        time_series = selected_data[["datetime", "y", "weekday"]].drop_duplicates()
        if "model" in selected_data.columns:
            fig = go.Figure()

            for i, (model_name, model_data) in enumerate(
                selected_data.groupby("model")
            ):
                fig.add_trace(
                    go.Scatter(
                        x=model_data["datetime"],
                        y=model_data["f"],
                        mode="lines",
                        name=f"Forecast ({model_name})",
                        line=dict(
                            color=px.colors.qualitative.Plotly[
                                i % len(px.colors.qualitative.Plotly)
                            ]
                        ),
                        hovertemplate=(
                            f"Model: {model_name}<br>"
                            + "datetime: %{x}<br>"
                            + "forecast: %{y}<br>"
                            + "Error: %{customdata[0]}<br>"
                            + "Abs Error: %{customdata[1]}<br>"
                            + "Perc Error: %{customdata[2]}<br>"
                            + "Perc Abs Error: %{customdata[3]}<br>"
                            + "Weekday: %{customdata[4]}"
                        ),
                        customdata=model_data[
                            ["err", "abs_err", "perc_err", "perc_abs_err", "weekday"]
                        ],
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=time_series["datetime"],
                    y=time_series["y"],
                    mode="lines",
                    name=ui_text["series_names"]["real"],
                    line=dict(color=px.colors.qualitative.Dark2[0]),
                    hovertemplate="datetime: %{x}<br>real: %{y}<br>Weekday: %{customdata[4]}",
                    customdata=time_series[["weekday"]],
                    opacity=1,
                )
            )

        else:
            fig = px.line(
                selected_data,
                x="datetime",
                y=["y", "f"],
                title="-".join(serie.values()),
                labels={
                    "datetime": ui_text["axis_labels"]["date"],
                    "value": ui_text["axis_labels"]["value"],
                },
                hover_data=["err", "abs_err", "perc_err", "perc_abs_err", "weekday"],
                range_y=[
                    selected_data[["y", "f"]].min().min() * 0.8,
                    selected_data[["y", "f"]].max().max() * 1.2,
                ],
            )

        newnames = {
            "y": ui_text["series_names"]["real"],
            "f": ui_text["series_names"]["forecast"],
        }

        fig.for_each_trace(
            lambda t: t.update(
                name=newnames.get(t.name, t.name),
                legendgroup=newnames.get(t.name, t.name),
                hovertemplate=t.hovertemplate.replace(
                    f"variable={t.name}", f"variable={newnames.get(t.name, t.name)}"
                )
                .replace("perc_abs_err", "Error porcentual absoluto")
                .replace("perc_err", "Error porcentual")
                .replace("abs_err", "Error absoluto")
                .replace("err", "Error")
                .replace("weekday", "Día de la semana"),
            )
        )

        st.plotly_chart(fig)


def plot_error_visualization(
    forecast: pd.DataFrame, selected_series: List, ui_text: Dict[str, str]
):
    """
    Plot the error visualization for the selected series.
    :param forecast: The DataFrame containing the forecast data.
    :param selected_series: The list of selected series to plot.
    :param ui_text: The dictionary containing the UI text.
    """
    st.subheader(ui_text["error_visualization_title"])

    # Add radio selector for filter type
    filter_type = st.radio(
        ui_text["select_filter_type"],
        [
            ui_text["datetime_filter"],
            ui_text["forecast_origin_filter"],
            ui_text["both_filters"],
        ],
        horizontal=True,
    )

    if filter_type in [ui_text["datetime_filter"], ui_text["both_filters"]]:
        st.write(ui_text["select_datetime_range"])
        date_range = date_range_picker(
            picker_type=PickerType.date,
            start=forecast["datetime"].min(),
            end=forecast["datetime"].max(),
            key="date_range_datetime",
        )
        if date_range:
            date_start = pd.to_datetime(date_range[0])
            date_end = pd.to_datetime(date_range[1])
        else:
            date_start = forecast["datetime"].min()
            date_end = forecast["datetime"].max()

    if filter_type in [ui_text["forecast_origin_filter"], ui_text["both_filters"]]:
        st.write(ui_text["select_forecast_origin_range"])
        forecast_origin_range = date_range_picker(
            picker_type=PickerType.date,
            start=forecast["forecast_origin"].min(),
            end=forecast["forecast_origin"].max(),
            key="date_range_forecast_origin",
        )
        if forecast_origin_range:
            forecast_origin_start = pd.to_datetime(forecast_origin_range[0])
            forecast_origin_end = pd.to_datetime(forecast_origin_range[1])
        else:
            forecast_origin_start = forecast["forecast_origin"].min()
            forecast_origin_end = forecast["forecast_origin"].max()

    data_dict = {}
    for idx, serie in enumerate(selected_series):
        selected_data = forecast.copy()
        filter_cond = ""

        if filter_type in [ui_text["datetime_filter"], ui_text["both_filters"]]:
            filter_cond += f"datetime >= '{date_start}' & datetime <= '{date_end}' & "

        if filter_type in [ui_text["forecast_origin_filter"], ui_text["both_filters"]]:
            filter_cond += f"forecast_origin >= '{forecast_origin_start}' & forecast_origin <= '{forecast_origin_end}' & "

        for col, col_value in serie.items():
            filter_cond += f"{col} == '{col_value}' & "

        # Remove the last & from the filter condition
        filter_cond = filter_cond[:-3]
        selected_data = selected_data.query(filter_cond)
        data_dict[idx] = selected_data.copy()

    # get maximum percentage error in selected data as a table in the streamplit. top10 rows
    models = sorted(
        set(model for serie in data_dict.values() for model in serie["model"].unique())
    )
    select_model = st.selectbox(ui_text["select_top_10"], models)
    for idx, serie in data_dict.items():
        filter = serie[serie["model"] == select_model]
        st.write(ui_text["top_10_errors"] + f": **{select_model}**")
        st.write(
            filter.nlargest(10, "perc_abs_err")[
                ["datetime", "forecast_origin", "model", "perc_abs_err"]
            ]
        )

    mean_or_median_error = st.radio(
        ui_text["show_median_or_mean"],
        ui_text["median_or_mean_options"],
        index=0,
        key="median_or_mean_pmrs_diarios",
        horizontal=True,
    )
    median_or_mean_trans = {
        ui_text["median_option"]: "median",
        ui_text["mean_option"]: "mean",
    }

    # Show mean or median overall perc_abs_err
    for idx, serie in data_dict.items():
        st.write(
            ui_text["error_message"].format(
                ui_text["mediana_mean_string_dict"][mean_or_median_error],
                f"**{serie['perc_abs_err'].agg(median_or_mean_trans[mean_or_median_error]):.2%}**",
            )
        )

        st.write(ui_text["aggregated_summary_title"] + ":")

        df_agg = serie.groupby("model", as_index=False).agg(
            y=("y", "mean"),
            f=("f", "mean"),
            err=("err", "mean"),
            abs_err=("abs_err", "mean"),
            perc_err=("perc_err", "mean"),
            perc_abs_err_mean=("perc_abs_err", "mean"),
            perc_abs_err_median=("perc_abs_err", "median"),
        )

        df_agg = df_agg.round(2)

        if mean_or_median_error == ui_text["mean_option"]:
            df_agg_filtered = df_agg[["model", "y", "f", "perc_abs_err_mean"]]
            df_agg_ordered = df_agg_filtered.sort_values(
                by="perc_abs_err_mean"
            ).reset_index(drop=True)
            st.write(df_agg_ordered)
            st.write(
                ui_text["best_error_message"].format(
                    f"**{df_agg_ordered['model'].iloc[0]}**",
                    f"**{df_agg_ordered['perc_abs_err_mean'].iloc[0]}**",
                )
            )
        elif mean_or_median_error == ui_text["median_option"]:
            df_agg_filtered = df_agg[["model", "y", "f", "perc_abs_err_median"]]
            df_agg_ordered = df_agg_filtered.sort_values(
                by="perc_abs_err_median"
            ).reset_index(drop=True)
            st.write(df_agg_ordered)

            if len(models) > 1:
                st.write(
                    ui_text["best_error_message"].format(
                        f"**{df_agg_ordered['model'].iloc[0]}**",
                        f"**{df_agg_ordered['perc_abs_err_median'].iloc[0]}**",
                    )
                )

    # Select which plot to show multiple allowed
    plot_options = st.multiselect(
        ui_text["select_plots"],
        ui_text["plot_options_error"],
        default=[],
        placeholder=ui_text["select_plots"],
        label_visibility="collapsed",
    )
    # "Box plot by horizon"
    if ui_text["plot_options_error"][0] in plot_options:
        st.write(f"### {ui_text['horizon_boxplot_title']}")
        # Box plot perc_abs_err by horizon
        for idx, serie in data_dict.items():
            # Show how many points for each horizon
            fig = px.box(serie, x="h", y="perc_abs_err", color="model")
            # Update yaxis to show % values
            fig.update_yaxes(tickformat=".2%")
            fig.update_layout(
                xaxis_title="Horizonte",
                yaxis_title="Error porcentual absoluto",
            )
            st.plotly_chart(fig)
            number_by_horizon = serie.groupby("h").size()
            if number_by_horizon.std() > 0:
                st.warning(ui_text["horizon_warning"])
    # "Box plot by datetime"
    if ui_text["plot_options_error"][1] in plot_options:
        st.write(f"### {ui_text['datetime_boxplot_title']}")
        # Box plot perc_abs_err by datetime columns depending on user selection
        dict_transformations = {
            ui_text["temporal_aggregation_options"][0]: lambda x: x.dayofweek,
            ui_text["temporal_aggregation_options"][1]: lambda x: x.month,
        }
        col_name_dict = {
            ui_text["temporal_aggregation_options"][0]: ui_text["day"],
            ui_text["temporal_aggregation_options"][1]: ui_text["month"],
        }
        select_agg = st.selectbox(
            ui_text["select_temporal_aggregation"],
            ui_text["temporal_aggregation_options"],
            key="select_agg",
        )

        for idx, serie in data_dict.items():
            # Apply the selected transformation to the datetime column
            transformed_datetime = serie["datetime"].apply(
                dict_transformations[select_agg]
            )
            # Use ui_text for ALL_DICT
            transformed_datetime = transformed_datetime.map(
                ui_text["ALL_DICT"][select_agg]
            )

            fig = px.box(
                serie,
                x=transformed_datetime,
                y="perc_abs_err",
                color="model",
                title=f"Box plot de {col_name_dict[select_agg]} para {idx}",
                labels={
                    "x": col_name_dict[select_agg],
                    "perc_abs_err": "Error porcentual absoluto",
                },
            )
            fig.update_yaxes(tickformat=".2%")
            st.plotly_chart(fig, use_container_width=True)
