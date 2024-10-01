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
from mango_base.mango.dashboards.time_series_utils.ui_text_es import (
    UI_TEXT as UI_TEXT_ES,
)
from mango_base.mango.dashboards.time_series_utils.ui_text_en import (
    UI_TEXT as UI_TEXT_EN,
)
from mango_base.mango.dashboards.time_series_utils.ui_text_catala import (
    UI_TEXT as UI_TEXT_CATALA,
)


def interface_visualization(project_name: str = None):
    # SETUP web page
    st.set_page_config(
        page_title="Visualization",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    # Language selector
    language = st.sidebar.selectbox(
        "Language / Idioma / Llengua", ["English", "Español", "Català"]
    )

    # Load appropriate UI text
    if language == "English":
        UI_TEXT = UI_TEXT_EN
    elif language == "Español":
        UI_TEXT = UI_TEXT_ES
    elif language == "Català":
        UI_TEXT = UI_TEXT_CATALA
    else:
        UI_TEXT = UI_TEXT_EN

    st.title(project_name or UI_TEXT["page_title"])

    if not st.session_state.get("files_loaded"):
        files_loaded = upload_files(UI_TEXT)
        st.session_state["files_loaded"] = files_loaded
    else:
        # Sidebar was loaded so we can place a manage button
        manage_files(st.session_state["files_loaded"], UI_TEXT)

    if st.session_state.get("files_loaded"):
        # Manage selected series using session_state
        if "selected_series" not in st.session_state:
            st.session_state["selected_series"] = []

        data, visualization = load_data(st.session_state.get("files_loaded"), UI_TEXT)
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
            # Update the keys in SELECT_AGR_TMP_DICT with the translation in UI_TEXT
            final_select_agr_tmp_dict = {}
            for key, value in SELECT_AGR_TMP_DICT.items():
                if key in UI_TEXT:
                    final_select_agr_tmp_dict[UI_TEXT[key]] = value
            select_agr_tmp, visualization = setup_sidebar(data, columns_id, UI_TEXT)
            time_series, forecast = process_data(
                data, columns_id, final_select_agr_tmp_dict, select_agr_tmp, UI_TEXT
            )

            if visualization == UI_TEXT["visualization_options"][0]:  # "Exploration"
                plot_time_series(
                    time_series,
                    st.session_state["selected_series"],
                    final_select_agr_tmp_dict,
                    select_agr_tmp,
                    UI_TEXT,
                )
            elif visualization == UI_TEXT["visualization_options"][1]:  # "Forecast"
                plot_forecast(forecast, st.session_state["selected_series"], UI_TEXT)
                plot_error_visualization(
                    forecast, st.session_state["selected_series"], UI_TEXT
                )
            else:
                st.write(
                    UI_TEXT["visualization_not_implemented"].format(
                        UI_TEXT["visualization_options"][0],
                        UI_TEXT["visualization_options"][1],
                    )
                )


if __name__ == "__main__":
    interface_visualization()
