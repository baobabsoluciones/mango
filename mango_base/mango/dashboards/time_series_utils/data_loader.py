import pandas as pd
import streamlit as st


@st.cache_data
def load_data(files_loaded):
    # Assuming the first file is the main data file
    if files_loaded:
        main_file_name = list(files_loaded.keys())[0]
        data = files_loaded[main_file_name]["data"]

        if "forecast_origin" in data.columns and "f" in data.columns:
            visualization = "Forecast"

            if "datetime" in data.columns and "forecast_origin" in data.columns:
                if "h" not in data.columns:
                    data["h"] = (data["datetime"] - data["forecast_origin"]).dt.days
            else:
                st.warning(
                    "Las columnas 'datetime' o 'forecast_origin' no están presentes para calcular 'h'."
                )

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
                st.info(
                    "La columna 'f' no está presente, por lo que no se calcularán los errores."
                )
        else:
            visualization = "Exploración"
            st.info(
                "El modo actual es 'Exploración', no se calcularán columnas de forecast ni errores."
            )

        return data, visualization
    else:
        st.error("No files uploaded.")
        return None, None
