import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.subplots as sp
from dateutil.rrule import weekday
from statsmodels.tsa.seasonal import STL
from streamlit_date_picker import date_range_picker, PickerType
from .constants import SELECT_AGR_TMP_DICT, DAY_NAME_DICT, MONTH_NAME_DICT, ALL_DICT


def select_series(data, columns):
    st.sidebar.title("Selecciona serie a analizar")
    filtered_data = data.copy()
    for column in columns:
        filter_values = filtered_data[column].unique()

        # Create a selectbox for each filter
        selected_value = st.sidebar.selectbox(
            f"Elige {column}:", filter_values, key=f"selectbox_{column}"
        )

        # Filter the DataFrame by the selected value
        filtered_data = filtered_data[filtered_data[column] == selected_value].copy()

    # Create a label for the selected series
    selected_item = {col: filtered_data.iloc[0][col] for col in columns}
    return selected_item


def plot_time_series(time_series, selected_series, select_agr_tmp_dict, select_agr_tmp):
    select_plot = st.selectbox(
        "Selecciona el gráfico",
        ["Serie original", "Serie por año", "STL"],
        label_visibility="collapsed",
    )
    if select_plot == "Serie original":
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
            filter_cond = f"datetime>='{date_start}' & datetime <= '{date_end}' & "
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
    elif select_plot == "Serie por año":
        options = st.multiselect(
            "Elige los años a visualizar",
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
    elif select_plot == "STL":
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


def setup_sidebar(time_series, columns_id):
    st.sidebar.title("Visualizaciones")
    visualization = st.sidebar.radio(
        "Selecciona la visualización",
        ["Time Series", "Forecast"],
    )
    st.sidebar.title("Selecciona la agrupación temporal de los datos")
    all_tmp_agr = ["hourly", "daily", "weekly", "monthly", "quarterly", "yearly"]

    # Reduce the list given the detail in the data datetime column
    if time_series["datetime"].dt.hour.nunique() == 1:
        all_tmp_agr.remove("hourly")
    if time_series["datetime"].dt.day.nunique() == 1:
        all_tmp_agr.remove("daily")
    if time_series["datetime"].dt.isocalendar().week.nunique() == 1:
        all_tmp_agr.remove("weekly")
    if time_series["datetime"].dt.month.nunique() == 1:
        all_tmp_agr.remove("monthly")
    if time_series["datetime"].dt.quarter.nunique() == 1:
        all_tmp_agr.remove("quarterly")
    if time_series["datetime"].dt.year.nunique() == 1:
        all_tmp_agr.remove("yearly")
    if len(all_tmp_agr) == 0:
        st.write("No se puede realizar el análisis temporal")
        return
    select_agr_tmp = st.sidebar.selectbox(
        "Selecciona la agrupación temporal de los datos",
        all_tmp_agr,
        label_visibility="collapsed",
    )

    # Select series
    if columns_id:
        selected_series = select_series(time_series, columns_id)
        st.session_state["selected_series"].append(selected_series)
    else:
        st.sidebar.write("No hay columnas para filtrar. Solo una serie detectada")
        st.session_state["selected_series"] = [{}]
    return select_agr_tmp, visualization


def plot_forecast(forecast, selected_series):
    st.subheader("Forecast plot")
    if not selected_series:
        st.write("Select at least one series to plot the forecast")
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
        "Choose a date",
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
        # map to DAY_NAME_DICT
        selected_data["weekday"] = selected_data["weekday"].map(DAY_NAME_DICT)
        fig = px.line(
            selected_data,
            x="datetime",
            y=["y", "f"],
            title=serie,
            labels={"datetime": "Fecha", "value": "Valor"},
            hover_data=["err", "abs_err", "perc_err", "perc_abs_err", "weekday"],
            range_y=[200, selected_data[["y", "f"]].max().max() * 1.2],
        )
        newnames = {"y": "Real", "f": "Pronóstico"}
        fig.for_each_trace(
            lambda t: t.update(
                name=newnames[t.name],
                legendgroup=newnames[t.name],
                hovertemplate=t.hovertemplate.replace(
                    f"variable={t.name}", f"variable={newnames[t.name]}"
                )
                .replace("perc_abs_err", "Error porcentual absoluto")
                .replace("perc_err", "Error porcentual")
                .replace("abs_err", "Error absoluto")
                .replace("err", "Error")
                .replace("weekday", "Día de la semana"),
            )
        )
        st.plotly_chart(fig)


def plot_error_visualization(forecast, selected_series):
    st.subheader("Error visualization")
    st.write(
        "**Seleccione el rango de fechas** *(columna datetime)* **para visualizar los errores de pronóstico**",
    )
    date_range = date_range_picker(
        picker_type=PickerType.date,
        start=forecast["datetime"].min(),
        end=forecast["datetime"].max(),
        key="date_range_2",
    )
    if date_range:
        date_start = pd.to_datetime(date_range[0])
        date_end = pd.to_datetime(date_range[1])
    else:
        date_start = forecast["datetime"].min()
        date_end = forecast["datetime"].max()
    data_dict = {}
    for idx, serie in enumerate(selected_series):
        selected_data = forecast.copy()
        filter_cond = f"datetime>='{date_start}' & datetime <= '{date_end}' & "
        for col, col_value in serie.items():
            filter_cond += f"{col} == '{col_value}' & "

        # Remove the last & from the filter condition
        filter_cond = filter_cond[:-3]
        selected_data = selected_data.query(filter_cond)
        data_dict[idx] = selected_data.copy()
    median_or_mean_trans = {"Mediana": "median", "Media": "mean"}
    mean_or_median_error = st.radio(
        "Mostrar mediana o media",
        ["Mediana", "Media"],
        index=0,
        key="median_or_mean_pmrs_diarios",
        horizontal=True,
    )

    # Show mean or median overall perc_abs_err
    for idx, serie in data_dict.items():
        st.write(
            f"El error porcentual absoluto medio de la serie es de "
            f"**{serie['perc_abs_err'].agg(median_or_mean_trans[mean_or_median_error]):.2%}**"
        )

    # Select which plot to show multiple allowed
    plot_options = st.multiselect(
        "Selecciona los gráficos a mostrar",
        ["Box plot por horizonte", "Box plot por datetime", "Scatter"],
        default=[],
        placeholder="Selecciona los gráficos a mostrar",
        label_visibility="collapsed",
    )
    if "Box plot por horizonte" in plot_options:
        st.write("### Box plot por horizonte")
        # Box plot perc_abs_err by horizon (# TODO: Handle multiple series)
        for idx, serie in data_dict.items():
            # Show how many points for each horizon
            fig = px.box(
                serie,
                x="h",
                y="perc_abs_err",
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
                st.warning(
                    "El número de puntos por horizonte es muy variable, revisa tu proceso de generación de pronósticos."
                    " Debes generar para cada datetime la misma cantidad de horizontes, "
                    "haciendo forecast_origin=forecast_origin-horizonte"
                    "con todos los horizontes que deseas pronosticar."
                )
    if "Box plot por datetime" in plot_options:
        st.write("### Box plot por datetime")
        # Box plot perc_abs_err by datetime columns depending on user selection
        dict_transformations = {
            "Diario": lambda x: x.dayofweek,
            "Mensual": lambda x: x.month,
        }
        col_name_dict = {
            "Diario": "Día",
            "Mensual": "Mes",
        }
        select_agg = st.selectbox(
            "Selecciona la agrupación temporal para el boxplot",
            ["Diario", "Mensual"],
            key="select_agg",
        )

        for idx, serie in data_dict.items():
            # Apply the selected transformation to the datetime column
            transformed_datetime = serie["datetime"].apply(
                dict_transformations[select_agg]
            )
            transformed_datetime = transformed_datetime.map(ALL_DICT[select_agg])

            # Create a box plot
            fig = px.box(
                serie,
                x=transformed_datetime,
                y="perc_abs_err",
                title=f"Box plot de {col_name_dict[select_agg]} para {idx}",
                labels={
                    "x": col_name_dict[select_agg],
                    "perc_abs_err": "Error porcentual absoluto",
                },
            )
            fig.update_yaxes(tickformat=".2%")
            st.plotly_chart(fig, use_container_width=True)
