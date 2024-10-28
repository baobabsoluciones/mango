import streamlit as st
from statsforecast import StatsForecast

from mango_base.mango.dashboards.time_series_utils.constants import (
    SELECT_AGR_TMP_DICT,
    model_context,
)
from mango_base.mango.dashboards.time_series_utils.constants import default_models
from mango_base.mango.dashboards.time_series_utils.data_loader import load_data
from mango_base.mango.dashboards.time_series_utils.data_processing import (
    process_data,
    convert_df,
    aggregate_to_input_cache,
    render_script,
)
from mango_base.mango.dashboards.time_series_utils.file_uploader import (
    upload_files,
    manage_files,
)
from mango_base.mango.dashboards.time_series_utils.ui_components import (
    plot_time_series,
    setup_sidebar,
    plot_forecast,
    plot_error_visualization,
    adapt_values_based_on_series_length,
)
from mango_base.mango.dashboards.time_series_utils.ui_text_catala import (
    UI_TEXT as UI_TEXT_CATALA,
)
from mango_base.mango.dashboards.time_series_utils.ui_text_en import (
    UI_TEXT as UI_TEXT_EN,
)
from mango_base.mango.dashboards.time_series_utils.ui_text_es import (
    UI_TEXT as UI_TEXT_ES,
)


# from statsmodels.tsa.seasonal import STL
# from streamlit_date_picker import date_range_picker, PickerType


def interface_visualization(project_name: str = None):
    # SETUP web page
    st.set_page_config(
        page_title="Visualization",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    # Set up theme

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

    hide_label = (
        """
        <style>
            div[data-testid="stFileUploader"]>section[data-testid="stFileUploaderDropzone"]>button[data-testid="baseButton-secondary"] {
               color:white;
            }
            div[data-testid="stFileUploader"]>section[data-testid="stFileUploaderDropzone"]>button[data-testid="baseButton-secondary"]::after {
                content: "BUTTON_TEXT";
                color:black;
                display: block;
                position: absolute;
            }
            div[data-testid="stFileUploaderDropzoneInstructions"]>div>span {
               visibility:hidden;
            }
            div[data-testid="stFileUploaderDropzoneInstructions"]>div>span::after {
               content:"INSTRUCTIONS_TEXT";
               visibility:visible;
               display:block;
            }
             div[data-testid="stFileUploaderDropzoneInstructions"]>div>small {
               visibility:hidden;
            }
            div[data-testid="stFileUploaderDropzoneInstructions"]>div>small::before {
               content:"FILE_LIMITS";
               visibility:visible;
               display:block;
            }
        </style>
        """.replace(
            "BUTTON_TEXT", UI_TEXT["upload_button_text"]
        )
        .replace("INSTRUCTIONS_TEXT", UI_TEXT["upload_instructions"])
        .replace(
            "FILE_LIMITS",
            UI_TEXT["file_limits"].format(st.get_option("server.maxUploadSize")),
        )
    )
    st.markdown(hide_label, unsafe_allow_html=True)
    if not st.session_state.get("files_loaded"):
        files_loaded, no_model_column = upload_files(UI_TEXT)
        st.session_state["files_loaded"] = files_loaded
        st.session_state["no_model_column"] = no_model_column
    else:
        # Sidebar was loaded so we can place a manage button
        manage_files(st.session_state["files_loaded"], UI_TEXT)
    if st.session_state.get("files_loaded"):
        # Manage selected series using session_state
        if "selected_series" not in st.session_state:
            st.session_state["selected_series"] = []
        no_model_column = st.session_state.get("no_model_column", False)
        data, visualization = load_data(st.session_state.get("files_loaded"), UI_TEXT)

        if data is not None:
            columns_id = [
                col
                for col in data.columns
                if col
                not in [
                    "forecast_origin",
                    "datetime",
                    "model",
                    "h",
                    "y",
                    "f",
                    "err",
                    "abs_err",
                    "perc_err",
                    "perc_abs_err",
                ]
            ]

            if "visualization_options" not in st.session_state:
                st.session_state["visualization_options"] = [
                    UI_TEXT["visualization_options"]
                ]

            # Setup side bar and get aggregation settings
            # Update the keys in SELECT_AGR_TMP_DICT with the translation in UI_TEXT
            final_select_agr_tmp_dict = {}
            for key, value in SELECT_AGR_TMP_DICT.items():
                if key in UI_TEXT:
                    final_select_agr_tmp_dict[UI_TEXT[key]] = value
            select_agr_tmp, visualization, visualization_options = setup_sidebar(
                data, columns_id, UI_TEXT
            )
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

                if len(visualization_options) == 1:
                    if "forecast_activated" not in st.session_state:
                        st.session_state["forecast_activated"] = False

                    if not st.session_state["forecast_activated"]:
                        if st.sidebar.button(UI_TEXT["activate_button_train"]):
                            st.session_state["forecast_activated"] = True

                    if st.session_state["forecast_activated"]:
                        freq_code = final_select_agr_tmp_dict[select_agr_tmp]
                        series_length = len(time_series)
                        horizon_limit, step_size_limit, n_windows_limit = (
                            adapt_values_based_on_series_length(series_length)
                        )

                        st.sidebar.write(UI_TEXT["model_parameters"])

                        if "horizon" not in st.session_state:
                            st.session_state["horizon"] = min(10, horizon_limit)
                        if "step_size" not in st.session_state:
                            st.session_state["step_size"] = min(3, step_size_limit)
                        if "n_windows" not in st.session_state:
                            st.session_state["n_windows"] = min(5, n_windows_limit)

                        with st.sidebar.form(key="forecast_form"):
                            st.write(UI_TEXT["forecast_parameters"])
                            horizon = st.number_input(
                                UI_TEXT["horizon"],
                                min_value=1,
                                max_value=horizon_limit,
                                value=28,
                                help=UI_TEXT["explanation_horizon"],
                                # value=st.session_state["horizon"]
                            )
                            step_size = st.number_input(
                                UI_TEXT["step_size"],
                                min_value=1,
                                max_value=step_size_limit,
                                value=28,
                                help=UI_TEXT["explanation_step_size"],
                                # value=st.session_state["step_size"]
                            )
                            n_windows = st.number_input(
                                UI_TEXT["n_windows"],
                                min_value=1,
                                max_value=n_windows_limit,
                                value=3,
                                help=UI_TEXT["explanation_n_windows"],
                                # value=st.session_state["n_windows"]
                            )

                            st.link_button(
                                UI_TEXT["documentation"],
                                "https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/cross_validation.html#load-and-explore-the-data",
                                type="secondary",
                            )

                            train_button = st.form_submit_button(
                                label="Train", type="primary"
                            )
                        if train_button:
                            st.info("Starting forecast training")
                            st.session_state["horizon"] = horizon
                            st.session_state["step_size"] = step_size
                            st.session_state["n_windows"] = n_windows

                            models_to_use = {k: v for k, v in default_models.items()}

                            fcst = StatsForecast(
                                models=list(models_to_use.values()), freq=freq_code
                            )

                            time_serie = time_series.copy()

                            if not columns_id:
                                time_serie["unique_id"] = "id_1"
                                columns_id = "unique_id"
                            time_serie = time_serie.rename(
                                columns={"datetime": "ds", "y": "y"}
                            )
                            crossvalidation_df = fcst.cross_validation(
                                df=time_serie,
                                h=st.session_state["horizon"],
                                step_size=st.session_state["step_size"],
                                n_windows=st.session_state["n_windows"],
                                id_col=columns_id,
                            )

                            model_columns = [
                                col
                                for col in crossvalidation_df.columns
                                if col not in ["ds", "cutoff", "y"]
                            ]
                            data_long = crossvalidation_df.melt(
                                id_vars=["ds", "cutoff", "y"],
                                value_vars=model_columns,
                                var_name="model",
                                value_name="f",
                            )

                            data_long = data_long.rename(
                                columns={"cutoff": "forecast_origin", "ds": "datetime"}
                            )

                            if "err" not in data_long.columns:
                                data_long["err"] = data_long["y"] - data_long["f"]
                            if "abs_err" not in data_long.columns:
                                data_long["abs_err"] = data_long["err"].abs()
                            if "perc_err" not in data_long.columns:
                                data_long["perc_err"] = (
                                    data_long["err"] / data_long["y"]
                                )
                            if "perc_abs_err" not in data_long.columns:
                                data_long["perc_abs_err"] = (
                                    data_long["abs_err"] / data_long["y"]
                                )

                            # st.write(data_long)
                            st.session_state["forecast"] = data_long
                            if (
                                UI_TEXT["visualization_options"][1]
                                not in st.session_state["visualization_options"]
                            ):
                                st.session_state["visualization_options"].append(
                                    UI_TEXT["visualization_options"][1]
                                )

                            st.session_state["visualization"] = UI_TEXT[
                                "visualization_options"
                            ][1]
                            st.rerun()
            elif visualization == UI_TEXT["visualization_options"][1]:  # "Forecast"
                if "f" in forecast.columns and "err" in forecast.columns:
                    st.info(UI_TEXT["upload_forecast"])

                    plot_forecast(
                        forecast, st.session_state["selected_series"], UI_TEXT
                    )

                    plot_error_visualization(
                        forecast,
                        st.session_state["selected_series"],
                        UI_TEXT,
                    )

                elif (
                    "forecast" in st.session_state
                    and st.session_state["forecast"] is not None
                ):
                    st.info(UI_TEXT["message_forecast_baseline"])
                    forecast_st = st.session_state["forecast"].copy()
                    forecast = aggregate_to_input_cache(
                        forecast_st,
                        freq=final_select_agr_tmp_dict[select_agr_tmp],
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
                    data_csv = forecast.copy()
                    data_csv = convert_df(data_csv)
                    st.download_button(
                        label="Download data as CSV",
                        data=data_csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
                    freq_code = final_select_agr_tmp_dict[select_agr_tmp]

                    if st.download_button(
                        label=UI_TEXT["jinja_template"],
                        data=render_script(
                            models=model_context,
                            horizon=st.session_state["horizon"],
                            step_size=st.session_state["step_size"],
                            n_windows=st.session_state["n_windows"],
                            freq_code=freq_code,
                        ),
                        file_name="forecast_model.py",
                    ):
                        st.success(UI_TEXT["downloaded"])

                    plot_forecast(
                        forecast,
                        st.session_state["selected_series"],
                        UI_TEXT,
                    )

                    plot_error_visualization(
                        forecast,
                        st.session_state["selected_series"],
                        UI_TEXT,
                    )

                else:
                    st.warning(UI_TEXT["warning_no_forecast"])
            else:
                st.write(
                    UI_TEXT["visualization_not_implemented"].format(
                        UI_TEXT["visualization_options"][0],
                        UI_TEXT["visualization_options"][1],
                    )
                )


if __name__ == "__main__":
    interface_visualization()
