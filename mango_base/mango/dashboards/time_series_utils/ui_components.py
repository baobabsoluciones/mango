import copy

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf
from streamlit_date_picker import date_range_picker, PickerType


def select_series(data, columns, UI_TEXT):
    st.sidebar.title(UI_TEXT["select_series"])
    filtered_data = data.copy()
    for column in columns:
        filter_values = filtered_data[column].unique()

        # Create a selectbox for each filter
        selected_value = st.sidebar.selectbox(
            UI_TEXT["choose_column"].format(column),
            filter_values,
            key=f"selectbox_{column}",
        )

        # Filter the DataFrame by the selected value
        filtered_data = filtered_data[filtered_data[column] == selected_value].copy()

    # Create a label for the selected series
    selected_item = {col: filtered_data.iloc[0][col] for col in columns}
    return selected_item


def plot_time_series(
    time_series, selected_series, select_agr_tmp_dict, select_agr_tmp, UI_TEXT
):
    select_plot = st.selectbox(
        UI_TEXT["choose_plot"],
        UI_TEXT["plot_options"],
        label_visibility="collapsed",
    )
    if select_plot == UI_TEXT["plot_options"][0]:  # "Original series"
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
    elif select_plot == UI_TEXT["plot_options"][1]:
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
        # "Series by year"
        options = st.multiselect(
            UI_TEXT["choose_years"],
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
    elif select_plot == UI_TEXT["plot_options"][2]:  # "STL"
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

            # Descomposición STL
            try:
                stl = STL(selected_data_stl["y"].ffill())
                result = stl.fit()
            except ValueError:
                st.write(
                    "No se puede realizar la descomposición STL para la serie seleccionada, "
                    "prueba otro nivel de agregación temporal."
                )
                continue
            fig1 = px.line(result.observed, title="Serie original")
            fig2 = px.line(result.trend, title="Tendencia")
            fig3 = px.line(result.seasonal, title="Estacionalidad")
            fig4 = px.line(result.resid, title="Residuales")
            # Put each plot in a subplot
            fig = sp.make_subplots(
                rows=4,
                cols=1,
                subplot_titles=[
                    "Serie Original",
                    "Tendencia",
                    "Estacionalidad",
                    "Residuales",
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

    elif select_plot == UI_TEXT["plot_options"][3]:  # "Lag analysis"
        for serie in selected_series:
            selected_data = time_series.copy()
            filter_cond = ""
            for col, col_value in serie.items():
                filter_cond += f"{col} == '{col_value}' & "

            # Remove the last & from the filter condition
            filter_cond = filter_cond[:-3]
            if filter_cond:
                selected_data = selected_data.query(filter_cond)
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
                    "Autocorrelation Function (ACF)",
                    "Partial Autocorrelation Function (PACF)",
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(acf_array[0])),
                    y=acf_array[0],
                    mode="markers",
                    marker=dict(color="#1f77b4", size=12),
                    name="ACF",
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
                    name="PACF",
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

    elif select_plot == UI_TEXT["plot_options"][4]:  # "Seasonality boxplot"
        selected_granularity = select_agr_tmp

        if selected_granularity == UI_TEXT["daily"]:
            freq_options = UI_TEXT["frequency_options"]
        elif selected_granularity == UI_TEXT["weekly"]:
            freq_options = UI_TEXT["frequency_options"][1:]
        elif selected_granularity == UI_TEXT["monthly"]:
            freq_options = [UI_TEXT["frequency_options"][2]]
        else:
            st.write(UI_TEXT["boxplot_error"])
            return
        selected_freq = st.selectbox(UI_TEXT["select_frequency"], freq_options)

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

            if selected_freq == UI_TEXT["frequency_options"][0]:
                selected_data["day_of_year"] = selected_data.index.dayofyear
                fig = px.box(
                    selected_data,
                    x="day_of_year",
                    y="y",
                    title=UI_TEXT["boxplot_titles"]["daily"],
                )
                st.plotly_chart(fig, use_container_width=True)

            elif selected_freq == UI_TEXT["frequency_options"][1]:
                selected_data["day_of_week"] = selected_data.index.weekday
                fig = px.box(
                    selected_data,
                    x="day_of_week",
                    y="y",
                    title=UI_TEXT["boxplot_titles"]["weekly"],
                )
                st.plotly_chart(fig, use_container_width=True)

            elif selected_freq == UI_TEXT["frequency_options"][2]:
                selected_data["month"] = selected_data.index.month
                fig = px.box(
                    selected_data,
                    x="month",
                    y="y",
                    title=UI_TEXT["boxplot_titles"]["monthly"],
                )
                st.plotly_chart(fig, use_container_width=True)


def setup_sidebar(time_series, columns_id, UI_TEXT):
    st.sidebar.title(UI_TEXT["sidebar_title"])
    if "forecast_origin" in time_series.columns and "f" in time_series.columns:
        visualization_options = UI_TEXT["visualization_options"]
    else:
        visualization_options = [UI_TEXT["visualization_options"][0]]

    visualization = st.sidebar.radio(
        UI_TEXT["select_visualization"],
        visualization_options,
    )
    st.sidebar.title(UI_TEXT["select_temporal_grouping"])
    all_tmp_agr = copy.deepcopy(UI_TEXT["temporal_grouping_options"])

    # Reduce the list given the detail in the data datetime column
    if time_series["datetime"].dt.hour.nunique() == 1:
        all_tmp_agr.remove(UI_TEXT["hourly"])
    if time_series["datetime"].dt.day.nunique() == 1:
        all_tmp_agr.remove(UI_TEXT["daily"])
    if time_series["datetime"].dt.isocalendar().week.nunique() == 1:
        all_tmp_agr.remove(UI_TEXT["weekly"])
    if time_series["datetime"].dt.month.nunique() == 1:
        all_tmp_agr.remove(UI_TEXT["monthly"])
    if time_series["datetime"].dt.quarter.nunique() == 1:
        all_tmp_agr.remove(UI_TEXT["quarterly"])
    if time_series["datetime"].dt.year.nunique() == 1:
        all_tmp_agr.remove(UI_TEXT["yearly"])
    if len(all_tmp_agr) == 0:
        st.write(UI_TEXT["temporal_analysis_error"])
        return
    select_agr_tmp = st.sidebar.selectbox(
        UI_TEXT["select_temporal_grouping"],
        all_tmp_agr,
        label_visibility="collapsed",
    )

    if columns_id:
        # Setup select series
        selected_item = select_series(time_series, columns_id, UI_TEXT)
        if st.sidebar.button(UI_TEXT["add_selected_series"]):
            # Avoid adding duplicates
            if selected_item not in st.session_state["selected_series"]:
                st.session_state["selected_series"].append(selected_item)
            else:
                st.toast(UI_TEXT["series_already_added"], icon="❌")

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
            if st.sidebar.button(UI_TEXT["remove_all_series"]):
                st.session_state["selected_series"] = []
                st.rerun()
    else:
        st.sidebar.write(UI_TEXT["no_columns_to_filter"])
        st.session_state["selected_series"] = [{}]

    return select_agr_tmp, visualization


def plot_forecast(forecast, selected_series, UI_TEXT):
    st.subheader(UI_TEXT["forecast_plot_title"])
    if not selected_series:
        st.write(UI_TEXT["select_series_to_plot"])
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
        UI_TEXT["choose_date"],
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
        # Use UI_TEXT for DAY_NAME_DICT
        selected_data["weekday"] = selected_data["weekday"].map(
            UI_TEXT["DAY_NAME_DICT"]
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
                    name=UI_TEXT["series_names"]["real"],
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
                    "datetime": UI_TEXT["axis_labels"]["date"],
                    "value": UI_TEXT["axis_labels"]["value"],
                },
                hover_data=["err", "abs_err", "perc_err", "perc_abs_err", "weekday"],
                range_y=[
                    selected_data[["y", "f"]].min().min() * 0.8,
                    selected_data[["y", "f"]].max().max() * 1.2,
                ],
            )

        newnames = {
            "y": UI_TEXT["series_names"]["real"],
            "f": UI_TEXT["series_names"]["forecast"],
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


def plot_error_visualization(forecast, selected_series, UI_TEXT):
    st.subheader(UI_TEXT["error_visualization_title"])

    # Add radio selector for filter type
    filter_type = st.radio(
        UI_TEXT["select_filter_type"],
        [
            UI_TEXT["datetime_filter"],
            UI_TEXT["forecast_origin_filter"],
            UI_TEXT["both_filters"],
        ],
        horizontal=True,
    )

    if filter_type in [UI_TEXT["datetime_filter"], UI_TEXT["both_filters"]]:
        st.write(UI_TEXT["select_datetime_range"])
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

    if filter_type in [UI_TEXT["forecast_origin_filter"], UI_TEXT["both_filters"]]:
        st.write(UI_TEXT["select_forecast_origin_range"])
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

        if filter_type in [UI_TEXT["datetime_filter"], UI_TEXT["both_filters"]]:
            filter_cond += f"datetime >= '{date_start}' & datetime <= '{date_end}' & "

        if filter_type in [UI_TEXT["forecast_origin_filter"], UI_TEXT["both_filters"]]:
            filter_cond += f"forecast_origin >= '{forecast_origin_start}' & forecast_origin <= '{forecast_origin_end}' & "

        for col, col_value in serie.items():
            filter_cond += f"{col} == '{col_value}' & "

        # Remove the last & from the filter condition
        filter_cond = filter_cond[:-3]
        selected_data = selected_data.query(filter_cond)
        data_dict[idx] = selected_data.copy()

    # get maximum percentage error in selected data as a table in the streamplit. top10 rows
    st.write(UI_TEXT["top_10_errors"])
    for idx, serie in data_dict.items():
        st.write(
            serie.nlargest(10, "perc_abs_err")[["datetime", "model", "perc_abs_err"]]
        )

    mean_or_median_error = st.radio(
        UI_TEXT["show_median_or_mean"],
        UI_TEXT["median_or_mean_options"],
        index=0,
        key="median_or_mean_pmrs_diarios",
        horizontal=True,
    )
    median_or_mean_trans = {
        UI_TEXT["median_option"]: "median",
        UI_TEXT["mean_option"]: "mean",
    }

    # Show mean or median overall perc_abs_err
    for idx, serie in data_dict.items():
        st.write(
            UI_TEXT["error_message"].format(
                UI_TEXT["mediana_mean_string_dict"][mean_or_median_error],
                f"**{serie['perc_abs_err'].agg(median_or_mean_trans[mean_or_median_error]):.2%}**",
            )
        )

    st.write(UI_TEXT["aggregated_summary_title"] + ":")

    df_agg = forecast.groupby(by=["model"], as_index=False).agg(
        y=("y", "mean"),
        f=("f", "mean"),
        err=("err", "mean"),
        abs_err=("abs_err", "mean"),
        perc_err=("perc_err", "mean"),
        perc_abs_err_mean=("perc_abs_err", "mean"),
        perc_abs_err_median=("perc_abs_err", "median"),
    )

    if mean_or_median_error == UI_TEXT["mean_option"]:
        df_agg_filtered = df_agg[["model", "y", "f", "perc_abs_err_mean"]]
        st.write(
            df_agg_filtered.rename(
                columns={
                    "perc_abs_err_mean": "Mean % Error",
                }
            )
        )
    else:
        df_agg_filtered = df_agg[["model", "y", "f", "perc_abs_err_mean"]]
        st.write(
            df_agg_filtered.rename(
                columns={
                    "perc_abs_err_median": "Median % Error"
                }
            )
        )

    # Select which plot to show multiple allowed
    plot_options = st.multiselect(
        UI_TEXT["select_plots"],
        UI_TEXT["plot_options_error"],
        default=[],
        placeholder=UI_TEXT["select_plots"],
        label_visibility="collapsed",
    )
    if UI_TEXT["plot_options_error"][0] in plot_options:  # "Box plot by horizon"
        st.write(f"### {UI_TEXT['horizon_boxplot_title']}")
        # Box plot perc_abs_err by horizon (# TODO: Handle multiple series)
        for idx, serie in data_dict.items():
            # Show how many points for each horizon
            fig = px.box(
                serie,
                x="h",
                y="perc_abs_err",
                color="model"
            )
            # Update yaxis to show % values
            fig.update_yaxes(tickformat=".2%")
            fig.update_layout(
                xaxis_title="Horizonte",
                yaxis_title="Error porcentual absoluto",
            )
            st.plotly_chart(fig)
            number_by_horizon = serie.groupby("h").size()
            if number_by_horizon.std() > 0:
                st.warning(UI_TEXT["horizon_warning"])
    if UI_TEXT["plot_options_error"][1] in plot_options:  # "Box plot by datetime"
        st.write(f"### {UI_TEXT['datetime_boxplot_title']}")
        # Box plot perc_abs_err by datetime columns depending on user selection
        dict_transformations = {
            UI_TEXT["temporal_aggregation_options"][0]: lambda x: x.dayofweek,
            UI_TEXT["temporal_aggregation_options"][1]: lambda x: x.month,
        }
        col_name_dict = {
            UI_TEXT["temporal_aggregation_options"][0]: UI_TEXT["day"],
            UI_TEXT["temporal_aggregation_options"][1]: UI_TEXT["month"],
        }
        select_agg = st.selectbox(
            UI_TEXT["select_temporal_aggregation"],
            UI_TEXT["temporal_aggregation_options"],
            key="select_agg",
        )

        for idx, serie in data_dict.items():
            # Apply the selected transformation to the datetime column
            transformed_datetime = serie["datetime"].apply(
                dict_transformations[select_agg]
            )
            # Use UI_TEXT for ALL_DICT
            transformed_datetime = transformed_datetime.map(
                UI_TEXT["ALL_DICT"][select_agg]
            )

            # Create a box plot
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
