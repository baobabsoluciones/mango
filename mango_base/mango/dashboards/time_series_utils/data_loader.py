import pandas as pd
import streamlit as st


@st.cache_data
def load_data(files_loaded, UI_TEXT):
    # Assuming the first file is the main data file
    if files_loaded:
        main_file_name = list(files_loaded.keys())[0]
        data = files_loaded[main_file_name]["data"]

        if "forecast_origin" in data.columns and "f" in data.columns:
            visualization = UI_TEXT["visualization_options"][1]  # "Forecast"

            if "datetime" in data.columns and "forecast_origin" in data.columns:
                if "h" not in data.columns:
                    data["h"] = (data["datetime"] - data["forecast_origin"]).dt.days
            else:
                st.warning(UI_TEXT["datetime_warning"])

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
                st.info(UI_TEXT["f_column_missing"])
        else:
            visualization = UI_TEXT["visualization_options"][0]  # "Exploration"
            st.info(UI_TEXT["exploration_mode"])

        return data, visualization
    else:
        st.error(UI_TEXT["no_files_uploaded"])
        return None, None
