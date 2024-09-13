import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from mango_time_series.mango.utils.processing import aggregate_to_input


# Cache the data to avoid loading it multiple times
@st.cache_data
def load_data(data_path):
    return pd.read_csv(data_path)


@st.cache_data
def aggregate_to_input_cache(data, freq, SERIES_CONF):
    return aggregate_to_input(data, freq, SERIES_CONF)


def select_series(data, columns):
    st.sidebar.title("Selecciona serie a analizar")
    filtered_data = data.copy()
    for column in columns:
        filter_values = filtered_data[column].unique()

        # Create a selectbox for each filter
        selected_value = st.sidebar.selectbox(
            f"Select {column} to filter:", filter_values, key=f"selectbox_{column}"
        )

        # Filter the DataFrame by the selected value
        filtered_data = filtered_data[filtered_data[column] == selected_value].copy()

    # Create a label for the selected series
    selected_item = " - ".join([str(filtered_data.iloc[0][col]) for col in columns])
    return selected_item


def interface_visualization(file, logo_path: str = None, project_name: str = None):
    st.set_page_config(
        page_title="Visualizacion",
        page_icon=requests.get(logo_path).content,
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    data = load_data(file)

    st.title(project_name)

    # Create a sidebar for navigation
    st.sidebar.title("Visualizaciones")

    # Add radious button to select visualization
    visualization = st.sidebar.radio(
        "Select visualization",
        ["Time Series", "Forecast"],
    )

    # Identify columns to be filtered
    columns_id = [
        col
        for col in data.columns
        if col
        not in [
            "forecast_origin",
            "datetime",
            "h",
            "y",
            "f",
            "err",
            "abs_err",
            "perc_err",
            "perc_abs_err",
        ]
    ]

    if columns_id == []:
        data["id"] = "1"
        columns_id = ["id"]

    # Convert forecast_origin to datetime
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["forecast_origin"] = pd.to_datetime(data["forecast_origin"])

    time_series = data[columns_id + ["datetime", "y"]].drop_duplicates()
    forecast = data.copy()

    # Make it possible to select the aggregation temporal
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
        "Select the agrupation to analyze the series:",
        all_tmp_agr,
    )

    select_agr_tmp_dict = {
        "hourly": "H",
        "daily": "D",
        "weekly": "W",
        "monthly": "M",
        "quarterly": "Q",
        "yearly": "YE",
    }

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

    # Manage selected series using session_state
    if "selected_series" not in st.session_state:
        st.session_state["selected_series"] = []

    selected_item = select_series(time_series, columns_id)

    if st.sidebar.button("Add selected series"):
        # Avoid adding duplicates
        if selected_item not in st.session_state["selected_series"]:
            st.session_state["selected_series"].append(selected_item)
        else:
            st.toast("The selected series is already in the list", icon="❌")

    # Display the selected series in the sidebar with remove button
    for idx, serie in enumerate(st.session_state["selected_series"]):
        col1, col2 = st.sidebar.columns(
            [8, 2]
        )  # Create two columns: 8 units for text, 2 units for button
        with col1:
            st.write(
                f"{idx + 1}. {serie}"
            )  # Display the series name in the first column
        with col2:
            if st.button(
                "❌", key=f"remove_{idx}"
            ):  # Display the button in the second column
                st.session_state["selected_series"].pop(
                    idx
                )  # Remove the series name from the first column
                st.rerun()  # Rerun the app to update the sidebar
        # Remove all selected series
    if st.session_state["selected_series"]:
        if st.sidebar.button("Remove all selected series"):
            st.session_state["selected_series"] = []
            st.rerun()

    # Plotly chart for the selected series
    if visualization == "Time Series":
        st.write("Selected series plot")
        for serie in st.session_state["selected_series"]:
            selected_data = time_series.copy()
            cols_to_filter = serie.split(" - ")
            filter_cond = ""
            for idx, col in enumerate(columns_id):
                filter_cond += f"{col} == '{cols_to_filter[idx]}' & "

            # Remove the last & from the filter condition
            filter_cond = filter_cond[:-3]
            selected_data = selected_data.query(filter_cond)
            st.write(selected_data)
            st.plotly_chart(px.line(selected_data, x="datetime", y="y", title=serie))

    elif visualization == "Forecast":
        st.write("Forecast plot")
        if not st.session_state["selected_series"]:
            st.write("Select at least one series to plot the forecast")
            return
        # Get dates for the selected series
        filter_cond = ""
        for serie in st.session_state["selected_series"]:
            cols_to_filter = serie.split(" - ")
            filter_cond_serie = "("
            for idx, col in enumerate(columns_id):
                filter_cond_serie += f"{col} == '{cols_to_filter[idx]}' & "

            # Remove the last & from the filter condition
            filter_cond_series = filter_cond_serie[:-3] + ")"
            filter_cond += filter_cond_series + " | "
        filter_cond = filter_cond[:-3]

        forecast_restricted = forecast.query(filter_cond)

        selected_date = st.date_input(
            "Choose a date",
            min_value=forecast_restricted["forecast_origin"].min(),
            max_value=forecast_restricted["forecast_origin"].max(),
            value=forecast_restricted["forecast_origin"].min(),
            label_visibility="collapsed",
        )
        for serie in st.session_state["selected_series"]:
            cols_to_filter = serie.split(" - ")
            filter_cond = ""
            for idx, col in enumerate(columns_id):
                filter_cond += f"{col} == '{cols_to_filter[idx]}' & "

            # Remove the last & from the filter condition
            filter_cond = filter_cond[:-3]
            selected_data = forecast_restricted.query(filter_cond)
            selected_data = selected_data[
                selected_data["forecast_origin"] == pd.to_datetime(selected_date)
            ]
            # Plot both real and predicted values x datetime shared y axis y and f column
            fig = px.line(
                selected_data,
                x="datetime",
                y=["y", "f"],
                title=serie,
                labels={"datetime": "Fecha", "value": "Valor"},
                hover_data=["err", "abs_err", "perc_err", "perc_abs_err"],
            )
            st.plotly_chart(fig)


if __name__ == "__main__":
    file = r"C:\Users\AntonioGonzález\Desktop\proyectos_baobab\mango\daily_forecast_error.csv"
    logo_path = r"https://www.multiserviciosaeroportuarios.com/wp-content/uploads/2024/03/cropped-Logo-transparente-blanco-Multiservicios-Aeroportuarios-Maero-1-192x192.png"
    interface_visualization(
        file=file, logo_path=logo_path, project_name="Testing limits of Montse"
    )
