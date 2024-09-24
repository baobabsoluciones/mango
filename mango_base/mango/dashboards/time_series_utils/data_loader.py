import pandas as pd
import streamlit as st

@st.cache_data
def load_data(files_loaded):
    # Assuming the first file is the main data file
    if files_loaded:
        main_file_name = list(files_loaded.keys())[0]
        data = files_loaded[main_file_name]["data"]
        # Check if columns available
        if "h" not in data.columns:
            data["h"] = (data["datetime"]-data["forecast_origin"]).dt.days
        if "err" not in data.columns:
            data["err"] = data["y"]-data["f"]
        if "abs_err" not in data.columns:
            data["abs_err"] = data["err"].abs()
        if "perc_err" not in data.columns:
            data["perc_err"] = data["err"]/data["y"]
        if "perc_abs_err" not in data.columns:
            data["perc_abs_err"] = data["abs_err"]/data["y"]
        return data
    else:
        st.error("No files uploaded.")
        return None