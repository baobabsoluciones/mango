import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from statsmodels.tsa.seasonal import STL
import plotly.subplots as sp

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
            f"Elige {column}:", filter_values, key=f"selectbox_{column}"
        )

        # Filter the DataFrame by the selected value
        filtered_data = filtered_data[filtered_data[column] == selected_value].copy()

    # Create a label for the selected series
    selected_item = {col: filtered_data.iloc[0][col] for col in columns}
    return selected_item


def interface_visualization(file, logo_path: str = None, project_name: str = None):
    # SETUP web page
    st.set_page_config(
        page_title="Visualizacion",
        # page_icon=requests.get(logo_path).content,
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    st.title(project_name)

    # Manage selected series using session_state
    if "selected_series" not in st.session_state:
        st.session_state["selected_series"] = []

    # Setup data
    data = load_data(file)

    # Identify columns to be filtered
    columns_id = [
        col
        for col in data.columns
        if col
        not in [
            "origin_date",
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

    # TODO: change id
    if columns_id == []:
        data["id"] = "1"
        columns_id = ["id"]

    # Convert forecast_origin to datetime
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["origin_date"] = pd.to_datetime(data["origin_date"])

    time_series = data[columns_id + ["datetime", "y"]].drop_duplicates()
    forecast = data.copy()

    # Setup side bar
    # Create a sidebar for navigation
    st.sidebar.title("Visualizaciones")

    # Add radious button to select visualization
    visualization = st.sidebar.radio(
        "Select visualization",
        ["Time Series", "Forecast"],
    )
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
        "",
        all_tmp_agr,
        label_visibility="collapsed",
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
            "KEY_COLS": columns_id + ["origin_date"],
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

    # Setup select series
    selected_item = select_series(time_series, columns_id)
    if st.sidebar.button("Add selected series"):
        # Avoid adding duplicates
        if selected_item not in st.session_state["selected_series"]:
            st.session_state["selected_series"].append(selected_item)
        else:
            st.toast("The selected series is already in the list", icon="❌")

    # Display the selected series in the sidebar with remove button
    for idx, serie in enumerate(st.session_state["selected_series"]):
        serie = "-".join(serie.values())
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

    # Setup visualization
    # Plotly chart for the selected series
    if visualization == "Time Series":
        select_plot = st.selectbox("", ["Serie original", "Serie por año", "STL"], label_visibility='collapsed')
        if select_plot == "Serie original":
            from streamlit_date_picker import date_range_picker, PickerType
            # st.subheader("Serie original")
            date_range= date_range_picker(picker_type=PickerType.date, start=time_series["datetime"].min(),end=time_series["datetime"].max(), key="date_range_1")
            if date_range:
                date_start = pd.to_datetime(date_range[0])
                date_end = pd.to_datetime(date_range[1])
            else:
                date_start = time_series["datetime"].min()
                date_end = time_series["datetime"].max()
            for serie in st.session_state["selected_series"]:
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
            options = st.multiselect("Elige los años a visualizar",time_series["datetime"].dt.year.unique())
            for serie in st.session_state["selected_series"]:
                selected_data = time_series.copy()
                filter_cond = ""
                for col, col_value in serie.items():
                    filter_cond += f"{col} == '{col_value}' & "

                # Remove the last & from the filter condition
                filter_cond = filter_cond[:-3]
                selected_data = selected_data.query(filter_cond)
                for year in reversed(sorted(options)):
                    selected_data_year = selected_data.query(
                        f"datetime.dt.year == {year}"
                    )
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
            for serie in st.session_state["selected_series"]:
                selected_data = time_series.copy()
                filter_cond = ""
                for col, col_value in serie.items():
                    filter_cond += f"{col} == '{col_value}' & "

                # Remove the last & from the filter condition
                filter_cond = filter_cond[:-3]
                selected_data = selected_data.query(filter_cond)
                selected_data_stl = selected_data.set_index("datetime")
                selected_data_stl = selected_data_stl.asfreq(
                    select_agr_tmp_dict[select_agr_tmp]
                )

                # Descomposición STL

                stl = STL(selected_data_stl["y"].ffill())
                result = stl.fit()

                fig1 = px.line(result.observed, title="Serie original")
                fig2 = px.line(result.trend, title="Tendencia")
                fig3 = px.line(result.seasonal, title="Estacionalidad")
                fig4 = px.line(result.resid, title="Residuales")
                # Put each plot in a subplot
                fig = sp.make_subplots(
                    rows=4,
                    cols=1,
                    subplot_titles=["Serie Original","Tendencia", "Estacionalidad", "Residuales"],
                    shared_xaxes=True
                )
                fig.add_trace(fig1.data[0], row=1, col=1)
                fig.add_trace(fig2.data[0], row=2, col=1)
                fig.add_trace(fig3.data[0], row=3, col=1)
                fig.add_trace(fig4.data[0], row=4, col=1)
                fig.update_layout(showlegend=False, title='-'.join(serie.values()))

                fig.update_layout(
                    showlegend=False,
                    height=900,
                    width=800,
                )

                st.plotly_chart(fig, use_container_width=True)

    elif visualization == "Forecast":
        median_or_mean_trans = {"Mediana": "median", "Media": "mean"}
        st.session_state.median_or_mean_diario = st.radio(
            "Mostrar mediana o media",
            ["Mediana", "Media"],
            index=0,
            key="median_or_mean_pmrs_diarios",
            horizontal=True,
        )
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
            min_value=forecast_restricted["origin_date"].min(),
            max_value=forecast_restricted["origin_date"].max(),
            value=forecast_restricted["origin_date"].min(),
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
                selected_data["origin_date"] == pd.to_datetime(selected_date)
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
    file = r"C:\Users\MontserratMuñoz\Documents\codigo\streamlit_code\streamlit_code\data\test_multi_index.csv"
    logo_path = r"https://www.multiserviciosaeroportuarios.com/wp-content/uploads/2024/03/cropped-Logo-transparente-blanco-Multiservicios-Aeroportuarios-Maero-1-192x192.png"
    interface_visualization(
        file=file, logo_path=logo_path, project_name="Testing Dashboard Time Series"
    )
