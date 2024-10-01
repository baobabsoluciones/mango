import pandas as pd
import plotly.express as px
import plotly.subplots as sp
import streamlit as st

# from statsmodels.tsa.seasonal import STL
# from streamlit_date_picker import date_range_picker, PickerType

from mango_base.mango.dashboards.time_series_utils.data_loader import load_data
from mango_base.mango.dashboards.time_series_utils.data_processing import process_data
from mango_base.mango.dashboards.time_series_utils.ui_components import (
    select_series,
    plot_time_series,
    setup_sidebar,
    plot_forecast,
    plot_error_visualization,
)
from mango_base.mango.dashboards.time_series_utils.constants import SELECT_AGR_TMP_DICT
from mango_base.mango.dashboards.time_series_utils.file_uploader import (
    upload_files,
    manage_files,
)


def interface_visualization(file, logo_path: str = None, project_name: str = None):
    # SETUP web page
    st.set_page_config(
        page_title="Visualizacion",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    st.title(project_name)

    if not st.session_state.get("files_loaded"):
        files_loaded = upload_files()
        st.session_state["files_loaded"] = files_loaded
    else:
        # Sidebar was loaded so we can place a manage button
        manage_files(st.session_state["files_loaded"])

    if st.session_state.get("files_loaded"):
        # Manage selected series using session_state
        if "selected_series" not in st.session_state:
            st.session_state["selected_series"] = []

        data, visualization = load_data(st.session_state.get("files_loaded"))
        if data is not None:
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

            # Setup side bar and get aggregation settings
            select_agr_tmp, visualization = setup_sidebar(data, columns_id)

            time_series, forecast = process_data(
                data, columns_id, SELECT_AGR_TMP_DICT, select_agr_tmp
            )

            if visualization == "Exploración":
                plot_time_series(
                    time_series,
                    st.session_state["selected_series"],
                    SELECT_AGR_TMP_DICT,
                    select_agr_tmp,
                )
            elif visualization == "Forecast":
                plot_forecast(forecast, st.session_state["selected_series"])
                plot_error_visualization(forecast, st.session_state["selected_series"])


if __name__ == "__main__":
    file = "daily_forecast_error.csv"
    logo_path = r"https://www.multiserviciosaeroportuarios.com/wp-content/uploads/2024/03/cropped-Logo-transparente-blanco-Multiservicios-Aeroportuarios-Maero-1-192x192.png"
    interface_visualization(
        file=file, logo_path=logo_path, project_name="Testing Dashboard Time Series"
    )
