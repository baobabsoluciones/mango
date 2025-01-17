import os
from io import BytesIO

import PIL.Image
import pandas as pd
import requests
import streamlit as st
from statsforecast import StatsForecast

from mango_time_series.dashboards.time_series_utils.constants import (
    SELECT_AGR_TMP_DICT,
    model_context,
    default_models,
)
from mango_time_series.dashboards.time_series_utils.data_loader import (
    load_data,
)
from mango_time_series.dashboards.time_series_utils.data_processing import (
    process_data,
    convert_df,
    aggregate_to_input_cache,
    render_script,
)
from mango_time_series.dashboards.time_series_utils.file_uploader import (
    upload_files,
    manage_files,
)
from mango_time_series.dashboards.time_series_utils.ui_components import (
    plot_time_series,
    setup_sidebar,
    plot_forecast,
    plot_error_visualization,
)
from mango_time_series.dashboards.time_series_utils.ui_text_en import (
    UI_TEXT as UI_TEXT_EN,
)
from mango_time_series.dashboards.time_series_utils.ui_text_es import (
    UI_TEXT as UI_TEXT_ES,
)


def interface_visualization(
    project_name: str = None, logo_url: str = None, experimental_features: bool = False
):
    """
    Main interface for the time series visualization dashboard.
    :param project_name: str with the name of the project
    """

    def validate_logo(logo_path):
        if not logo_path:
            return None

        # Handle file:/// protocol and normalize path
        if logo_path.startswith("file:///"):
            logo_path = logo_path.replace("file:///", "")
            logo_path = os.path.normpath(logo_path)

        # Check if it's a local file
        if os.path.exists(logo_path):
            try:
                PIL.Image.open(logo_path)
                return logo_path
            except Exception:
                return None

        # If not a local file, try as URL
        if logo_path.startswith(("http://", "https://")):
            try:
                response = requests.get(logo_path, timeout=5)
                if response.status_code == 200:
                    image_data = BytesIO(response.content)
                    PIL.Image.open(image_data)
                    return logo_path
            except Exception:
                return None

        return None

    st.set_page_config(
        page_title=project_name,
        page_icon=validate_logo(logo_url),
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    # Set up theme

    # Language selector
    language = st.sidebar.selectbox(
        "Language / Idioma / Llengua", ["English", "Español"]
    )

    # Load appropriate UI text
    if language == "English":
        UI_TEXT = UI_TEXT_EN
    elif language == "Español":
        UI_TEXT = UI_TEXT_ES
    else:
        UI_TEXT = UI_TEXT_EN

    st.title(project_name)

    hide_label = (
        """
        <style>
            div[data-testid="stFileUploader"] button {
               background-color: white !important; /* Fondo azul */
               color: black !important; /* Texto blanco */
               cursor: pointer !important; /* Cambia el cursor al pasar */
               border: 1px solid #3d9df3; 
        }
            div[data-testid="stFileUploader"] button:hover {
               color: #3d9df3;
               border: 1px solid #3d9df3; 
            }

            div[data-testid="stFileUploader"]>section[data-testid="stFileUploaderDropzone"]>button[data-testid="baseButton-secondary"] {
               color:white;
               background-color: darkblue
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
            div[data-testid="stFormSubmitButton"] button {
               background-color: white !important;
               color: black !important; 
               cursor: pointer !important;
               border: 1px solid #3d9df3; 
            }
    
            div[data-testid="stFormSubmitButton"] button:hover {
               color: #3d9df3;
               border: 1px solid #3d9df3; 
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
    if not st.session_state.get("files_loaded_bool", False):
        files_loaded, no_model_column = upload_files(UI_TEXT)
        if files_loaded:
            st.session_state["files_loaded"] = files_loaded
            st.session_state["files_loaded_bool"] = True
            st.session_state["no_model_column"] = no_model_column
            st.rerun()
    else:
        # Sidebar was loaded so we can place a manage button
        manage_files(st.session_state["files_loaded"], UI_TEXT)
        if st.session_state.get("files_loaded_bool"):
            if st.session_state.get("files_loaded_bool", False):
                if "selected_series" not in st.session_state:
                    st.session_state["selected_series"] = []
                data, visualization = load_data(
                    st.session_state.get("files_loaded"), UI_TEXT
                )

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

                    columns_id_name = columns_id[0] if columns_id else None

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
                    (
                        select_agr_tmp,
                        visualization,
                        visualization_options,
                        selected_uid,
                    ) = setup_sidebar(data, columns_id, UI_TEXT, columns_id_name)

                    if selected_uid and selected_uid[0]:
                        current_uid = selected_uid[0].get(columns_id_name)
                        if "previous_uid" not in st.session_state:
                            st.session_state["previous_uid"] = current_uid

                        current_length = len(selected_uid)
                        if "previous_length" not in st.session_state:
                            st.session_state["previous_length"] = current_length

                        if current_length < st.session_state["previous_length"]:
                            # If the number of series selected is less than the previous one, the forecast of the current is maintained
                            st.session_state["forecast_activated"] = True
                            st.session_state["selected_series"] = selected_uid
                            st.session_state["previous_length"] = current_length

                        if current_length > st.session_state["previous_length"]:
                            # Reset forecast if the number of series selected is greater than the previous one
                            st.session_state["forecast_activated"] = False
                            st.session_state["forecast"] = None
                            st.session_state["selected_series"] = selected_uid
                            st.session_state["previous_length"] = current_length
                            st.rerun()

                    elif "previous_uid" in st.session_state:
                        st.session_state.pop("previous_uid", None)
                        st.session_state["forecast"] = None
                        st.rerun()

                    time_series, forecast = process_data(
                        data,
                        columns_id,
                        final_select_agr_tmp_dict,
                        select_agr_tmp,
                        UI_TEXT,
                    )
                    # "Exploration"
                    if visualization == UI_TEXT["visualization_options"][0]:
                        st.subheader(UI_TEXT["page_title_visualization"], divider=True)
                        plot_time_series(
                            time_series,
                            st.session_state["selected_series"],
                            final_select_agr_tmp_dict,
                            select_agr_tmp,
                            UI_TEXT,
                            columns_id_name=columns_id_name,
                            experimental_features=experimental_features,
                        )
                        if len(visualization_options) == 1:
                            if "forecast_activated" not in st.session_state:
                                st.session_state["forecast_activated"] = False

                            if len(st.session_state.get("selected_series", [])) > 2:
                                st.session_state["forecast_activated"] = False
                                st.warning(UI_TEXT["warning_no_forecast_possible"])
                            else:
                                if not st.session_state["forecast_activated"]:
                                    if st.sidebar.button(
                                        UI_TEXT["activate_button_train"]
                                    ):
                                        st.session_state["forecast_activated"] = True

                            if st.session_state["forecast_activated"]:
                                freq_code = final_select_agr_tmp_dict[select_agr_tmp]
                                st.sidebar.write(UI_TEXT["model_parameters"])

                                if "horizon" not in st.session_state:
                                    st.session_state["horizon"] = 1
                                if "step_size" not in st.session_state:
                                    st.session_state["step_size"] = 1
                                if "n_windows" not in st.session_state:
                                    st.session_state["n_windows"] = 1

                                if len(st.session_state["selected_series"]) == 1:
                                    with st.sidebar.form(key="forecast_form"):
                                        st.write(UI_TEXT["forecast_parameters"])
                                        horizon = st.number_input(
                                            UI_TEXT["horizon"],
                                            min_value=1,
                                            help=UI_TEXT["explanation_horizon"],
                                            value=st.session_state["horizon"],
                                        )
                                        step_size = st.number_input(
                                            UI_TEXT["step_size"],
                                            min_value=1,
                                            help=UI_TEXT["explanation_step_size"],
                                            value=st.session_state["step_size"],
                                        )
                                        n_windows = st.number_input(
                                            UI_TEXT["n_windows"],
                                            min_value=1,
                                            help=UI_TEXT["explanation_n_windows"],
                                            value=st.session_state["n_windows"],
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

                                        models_to_use = {
                                            k: v for k, v in default_models.items()
                                        }

                                        fcst = StatsForecast(
                                            models=list(models_to_use.values()),
                                            freq=freq_code,
                                        )

                                        if selected_uid[0]:
                                            columns_id = selected_uid[0].get(
                                                columns_id_name
                                            )
                                            time_series = time_series[
                                                time_series[columns_id_name]
                                                == columns_id
                                            ]

                                        time_serie = time_series.copy()

                                        if not columns_id:
                                            time_serie["unique_id"] = "id_1"
                                        else:
                                            time_serie = time_series.rename(
                                                columns={columns_id_name: "unique_id"}
                                            )
                                        time_serie = time_serie.rename(
                                            columns={"datetime": "ds", "y": "y"}
                                        )

                                        crossvalidation_df = fcst.cross_validation(
                                            df=time_serie,
                                            h=st.session_state["horizon"],
                                            step_size=st.session_state["step_size"],
                                            n_windows=st.session_state["n_windows"],
                                            id_col="unique_id",
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
                                            columns={
                                                "cutoff": "forecast_origin",
                                                "ds": "datetime",
                                            }
                                        )

                                        if "err" not in data_long.columns:
                                            data_long["err"] = (
                                                data_long["y"] - data_long["f"]
                                            )
                                        if "abs_err" not in data_long.columns:
                                            data_long["abs_err"] = data_long[
                                                "err"
                                            ].abs()
                                        if "perc_err" not in data_long.columns:
                                            data_long["perc_err"] = (
                                                data_long["err"] / data_long["y"]
                                            )
                                        if "perc_abs_err" not in data_long.columns:
                                            data_long["perc_abs_err"] = (
                                                data_long["abs_err"] / data_long["y"]
                                            )

                                        st.session_state["forecast"] = data_long
                                        if (
                                            UI_TEXT["visualization_options"][1]
                                            not in st.session_state[
                                                "visualization_options"
                                            ]
                                        ):
                                            st.session_state[
                                                "visualization_options"
                                            ].append(
                                                UI_TEXT["visualization_options"][1]
                                            )

                                        st.session_state["visualization"] = UI_TEXT[
                                            "visualization_options"
                                        ][1]
                                        st.rerun()

                                elif len(st.session_state["selected_series"]) == 2:
                                    if st.session_state["forecast_activated"]:
                                        freq_code = final_select_agr_tmp_dict[
                                            select_agr_tmp
                                        ]
                                        st.sidebar.write(UI_TEXT["model_parameters"])

                                        if "forecast_params" not in st.session_state:
                                            st.session_state["forecast_params"] = {
                                                series_id.get(columns_id_name): {
                                                    "horizon": 1,
                                                    "step_size": 1,
                                                    "n_windows": 1,
                                                }
                                                for series_id in st.session_state[
                                                    "selected_series"
                                                ]
                                            }
                                        if "saved_params" not in st.session_state:
                                            st.session_state["saved_params"] = {}
                                        for series_id in st.session_state[
                                            "selected_series"
                                        ]:
                                            uid = series_id.get(columns_id_name)
                                            with st.sidebar.form(
                                                key=f"forecast_form_{uid}"
                                            ):
                                                st.write(
                                                    f"Forecast parameters for series: {uid}"
                                                )
                                                horizon = st.number_input(
                                                    UI_TEXT["horizon_uid"].format(
                                                        uid=uid
                                                    ),
                                                    min_value=1,
                                                    value=st.session_state[
                                                        "forecast_params"
                                                    ][uid]["horizon"],
                                                    help=UI_TEXT["explanation_horizon"],
                                                )
                                                step_size = st.number_input(
                                                    UI_TEXT["step_size_uid"].format(
                                                        uid=uid
                                                    ),
                                                    min_value=1,
                                                    value=st.session_state[
                                                        "forecast_params"
                                                    ][uid]["step_size"],
                                                    help=UI_TEXT[
                                                        "explanation_step_size"
                                                    ],
                                                )
                                                n_windows = st.number_input(
                                                    UI_TEXT["n_windows_uid"].format(
                                                        uid=uid
                                                    ),
                                                    min_value=1,
                                                    value=st.session_state[
                                                        "forecast_params"
                                                    ][uid]["n_windows"],
                                                    help=UI_TEXT[
                                                        "explanation_n_windows"
                                                    ],
                                                )

                                                st.link_button(
                                                    UI_TEXT["documentation"],
                                                    "https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/cross_validation.html#load-and-explore-the-data",
                                                    type="secondary",
                                                )

                                                train_button = st.form_submit_button(
                                                    label="Save parameters"
                                                )

                                                if train_button:
                                                    st.session_state["forecast_params"][
                                                        uid
                                                    ] = {
                                                        "horizon": horizon,
                                                        "step_size": step_size,
                                                        "n_windows": n_windows,
                                                    }
                                                    st.session_state["saved_params"][
                                                        uid
                                                    ] = True
                                                if st.session_state["saved_params"].get(
                                                    uid, False
                                                ):
                                                    st.success(
                                                        f"Parameters saved for series: {uid}"
                                                    )
                                        if st.sidebar.button("Train"):
                                            forecasts = []
                                            for series_id in st.session_state[
                                                "selected_series"
                                            ]:
                                                uid = series_id.get(columns_id_name)
                                                time_series_filtered = (
                                                    time_series.copy()
                                                )

                                                time_series_filtered = (
                                                    time_series_filtered[
                                                        time_series_filtered[
                                                            columns_id_name
                                                        ]
                                                        == series_id.get(
                                                            columns_id_name
                                                        )
                                                    ]
                                                )

                                                time_series_filtered = time_series_filtered.rename(
                                                    columns={
                                                        "datetime": "ds",
                                                        "y": "y",
                                                        columns_id_name: "unique_id",
                                                    }
                                                )

                                                models_to_use = {
                                                    k: v
                                                    for k, v in default_models.items()
                                                }
                                                fcst = StatsForecast(
                                                    models=list(models_to_use.values()),
                                                    freq=freq_code,
                                                )

                                                params = st.session_state[
                                                    "forecast_params"
                                                ][uid]

                                                crossvalidation_df = (
                                                    fcst.cross_validation(
                                                        df=time_series_filtered,
                                                        h=params["horizon"],
                                                        step_size=params["step_size"],
                                                        n_windows=params["n_windows"],
                                                        id_col="unique_id",
                                                    )
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
                                                    columns={
                                                        "cutoff": "forecast_origin",
                                                        "ds": "datetime",
                                                    }
                                                )

                                                # Adding uid to the forecast
                                                data_long[columns_id_name] = uid
                                                data_long["err"] = (
                                                    data_long["y"] - data_long["f"]
                                                )
                                                data_long["abs_err"] = data_long[
                                                    "err"
                                                ].abs()
                                                data_long["perc_err"] = (
                                                    data_long["err"] / data_long["y"]
                                                )
                                                data_long["perc_abs_err"] = (
                                                    data_long["abs_err"]
                                                    / data_long["y"]
                                                )

                                                forecasts.append(data_long)

                                            st.session_state["forecast"] = forecasts

                                            if (
                                                UI_TEXT["visualization_options"][1]
                                                not in st.session_state[
                                                    "visualization_options"
                                                ]
                                            ):
                                                st.session_state[
                                                    "visualization_options"
                                                ].append(
                                                    UI_TEXT["visualization_options"][1]
                                                )

                                            st.session_state["visualization"] = UI_TEXT[
                                                "visualization_options"
                                            ][1]
                                            st.success(UI_TEXT["forecast_completed"])
                                            st.rerun()

                    # "Forecast"
                    elif visualization == UI_TEXT["visualization_options"][1]:
                        st.subheader(UI_TEXT["page_title_forecast"], divider=True)
                        if len(st.session_state["selected_series"]) > 2:
                            st.warning(UI_TEXT["warning_no_forecast_possible"])
                        else:
                            if "f" in forecast.columns and "err" in forecast.columns:
                                st.info(UI_TEXT["upload_forecast"])
                                plot_forecast(
                                    forecast=forecast,
                                    selected_series=st.session_state["selected_series"],
                                    ui_text=UI_TEXT,
                                    columns_id_name=columns_id_name,
                                )

                                plot_error_visualization(
                                    forecast=forecast,
                                    selected_series=st.session_state["selected_series"],
                                    ui_text=UI_TEXT,
                                    freq=final_select_agr_tmp_dict[select_agr_tmp],
                                    columns_id_name=columns_id_name,
                                )

                            elif (
                                "forecast" in st.session_state
                                and st.session_state["forecast"] is not None
                            ):
                                if len(st.session_state["selected_series"]) == 1:
                                    st.info(UI_TEXT["message_forecast_baseline"])
                                    forecast_st = st.session_state["forecast"].copy()

                                    if selected_uid and selected_uid[0]:
                                        columns_id_multiple = selected_uid[0].get(
                                            columns_id_name
                                        )
                                        if isinstance(columns_id_name, str):
                                            filtered_forecast_st = [
                                                df
                                                for df in forecast_st
                                                if columns_id_name in df.columns
                                                and (
                                                    df[columns_id_name]
                                                    == columns_id_multiple
                                                ).any()
                                            ]
                                            forecast_st = filtered_forecast_st[0]
                                        else:
                                            forecast_st[columns_id_name] = (
                                                columns_id_multiple
                                            )
                                    forecast = aggregate_to_input_cache(
                                        forecast_st,
                                        freq=final_select_agr_tmp_dict[select_agr_tmp],
                                        series_conf={
                                            "KEY_COLS": columns_id
                                            + ["forecast_origin", "model"],
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
                                    freq_code = final_select_agr_tmp_dict[
                                        select_agr_tmp
                                    ]

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
                                        forecast=forecast,
                                        selected_series=st.session_state[
                                            "selected_series"
                                        ],
                                        ui_text=UI_TEXT,
                                        columns_id_name=columns_id_name,
                                    )

                                    plot_error_visualization(
                                        forecast=forecast,
                                        selected_series=st.session_state[
                                            "selected_series"
                                        ],
                                        ui_text=UI_TEXT,
                                        freq=freq_code,
                                        columns_id_name=columns_id_name,
                                    )
                                elif len(st.session_state["selected_series"]) == 2:
                                    st.info(UI_TEXT["two_id_trained"])
                                    combined_forecasts = pd.concat(
                                        st.session_state["forecast"], ignore_index=True
                                    )

                                    data_csv = combined_forecasts.copy()
                                    data_csv = convert_df(data_csv)
                                    st.download_button(
                                        label="Download combined predictions as CSV",
                                        data=data_csv,
                                        file_name="combined_predictions.csv",
                                        mime="text/csv",
                                    )

                                    freq_code = final_select_agr_tmp_dict[
                                        select_agr_tmp
                                    ]

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
                                        forecast=combined_forecasts,
                                        selected_series=st.session_state[
                                            "selected_series"
                                        ],
                                        ui_text=UI_TEXT,
                                        columns_id_name=columns_id_name,
                                    )
                                    plot_error_visualization(
                                        forecast=combined_forecasts,
                                        selected_series=st.session_state[
                                            "selected_series"
                                        ],
                                        ui_text=UI_TEXT,
                                        freq=freq_code,
                                        columns_id_name=columns_id_name,
                                    )

                                else:
                                    st.warning(UI_TEXT["warning_no_forecast_possible"])

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
    project_name = os.environ.get("TS_DASHBOARD_PROJECT_NAME", "Project")
    logo_url = os.environ.get("TS_DASHBOARD_LOGO_URL", None)
    experimental_features = os.environ.get("TS_DASHBOARD_EXPERIMENTAL_FEATURES", False)
    if experimental_features.lower() in ["true", "t", "1"]:
        experimental_features = True
    else:
        experimental_features = False
    interface_visualization(
        project_name=project_name,
        logo_url=logo_url,
        experimental_features=experimental_features,
    )
