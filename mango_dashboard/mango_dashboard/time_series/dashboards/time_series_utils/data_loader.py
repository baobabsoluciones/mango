from typing import Dict

import pandas as pd
import streamlit as st


@st.cache_data
def load_data(files_loaded: Dict, ui_text: Dict):
    """
    Load data from the files loaded in the app.
    :param files_loaded: Dictionary with the files loaded in the app.
    :param ui_text: Dictionary with the text to display in the app.
    """
    # Assuming the first file is the main data file
    list_df = []
    list_visualization = []
    change_column_state = False

    for files_name, df in files_loaded.items():
        data = df["data"]

        if "forecast_origin" in data.columns and "f" in data.columns:
            # "Forecast"
            visualization = ui_text["visualization_options"][1]
            if "f" in data.columns:
                if "err" not in data.columns:
                    data["err"] = data["y"] - data["f"]
                if "abs_err" not in data.columns:
                    data["abs_err"] = data["err"].abs()
                if "perc_err" not in data.columns:
                    data["perc_err"] = data["err"] / data["y"]
                if "perc_abs_err" not in data.columns:
                    data["perc_abs_err"] = data["abs_err"] / data["y"]

                perc_cols = [col for col in data.columns if "perc" in col]
                data[perc_cols] = data[perc_cols].round(3)
                no_perc_cols = ["err", "abs_err"]
                data[no_perc_cols] = data[no_perc_cols].round(0)
            else:
                st.info(ui_text["f_column_missing"])
        else:
            # "Exploration"
            visualization = ui_text["visualization_options"][0]

        list_df.append(data)
        list_visualization.append(visualization)
        if "model" not in data.columns:
            change_column_state = True
            data["model"] = files_name

    if change_column_state and len(list_df) > 1:
        st.session_state["no_model_column"] = False

    df_final = pd.concat(list_df, ignore_index=True)
    # How to check if all list visualizations have the same value?
    check_visualization = all(
        elem == list_visualization[0] for elem in list_visualization
    )

    if not check_visualization:
        raise ValueError("Error visualization")

    return df_final, list_visualization[0]
