import pandas as pd
import streamlit as st


def preview_data(
    uploaded_file, separator, decimal, thousands, encoding, date_format, UI_TEXT
):
    st.write(UI_TEXT["preview_title"])
    st.write(separator, decimal, thousands, encoding, date_format)
    data_frame = pd.read_csv(
        uploaded_file,
        sep=separator,
        decimal=decimal,
        thousands=thousands,
        encoding=encoding,
    )
    date_columns = []
    if "datetime" in data_frame.columns:
        date_columns.append("datetime")
    if "forecast_origin" in data_frame.columns:
        date_columns.append("forecast_origin")

    if date_columns:
        data_frame[date_columns] = data_frame[date_columns].apply(
            pd.to_datetime, format=date_format
        )

    st.write(data_frame.head())


def upload_files(UI_TEXT):
    files = st.file_uploader(
        UI_TEXT["upload_file"], type=["csv", "xlsx"], accept_multiple_files=True, key="file_uploader"
    )
    files_loaded = {}

    if files:
        for uploaded_file in files:
            st.write(f"{UI_TEXT['file_title']} {uploaded_file.name}")
            extension = uploaded_file.name.split(".")[-1]

            with st.form(key=f"form_{uploaded_file.name}"):
                file_name = st.text_input(
                    UI_TEXT["file_name"], value=uploaded_file.name
                )
                if extension == "csv":
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        separator = st.selectbox(
                            UI_TEXT["separator"], options=[",", ";", "|", "\t"], index=0
                        )
                    with col2:
                        decimal = st.text_input(
                            UI_TEXT["decimal"], value=".", help=UI_TEXT["decimal_help"]
                        )
                    with col3:
                        thousands = st.text_input(
                            UI_TEXT["thousands"],
                            value=",",
                            help=UI_TEXT["thousands_help"],
                        )
                    with col4:
                        encoding = st.text_input(
                            UI_TEXT["encoding"],
                            value="utf-8",
                            help=UI_TEXT["encoding_help"],
                        )
                    with col5:
                        date_format = st.text_input(
                            UI_TEXT["date_format"],
                            value="%Y-%m-%d",
                            help=UI_TEXT["date_format_help"],
                        )
                    data_frame = pd.read_csv(
                        uploaded_file,
                        sep=separator,
                        decimal=decimal,
                        thousands=thousands,
                        encoding=encoding,
                    )
                    date_columns = []
                    if "datetime" in data_frame.columns:
                        date_columns.append("datetime")
                    if "forecast_origin" in data_frame.columns:
                        date_columns.append("forecast_origin")

                    if date_columns:
                        data_frame[date_columns] = data_frame[date_columns].apply(
                            pd.to_datetime, format=date_format
                        )
                elif extension == "xlsx":
                    data_frame = pd.read_excel(uploaded_file)

                submit_button = st.form_submit_button(label=UI_TEXT["load_data"])

            if submit_button:
                if extension == "csv":
                    files_loaded[file_name] = {
                        "data": data_frame,
                        "separator": separator,
                        "decimal": decimal,
                        "thousands": thousands,
                        "encoding": encoding,
                        "date_format": date_format,
                        "file_name": file_name,
                    }
                elif extension == "xlsx":
                    files_loaded[file_name] = {
                        "data": data_frame,
                        "file_name": file_name,
                    }
    return files_loaded


def manage_files(files_loaded, UI_TEXT):
    st.sidebar.write(UI_TEXT["manage_files"])
    remaining_files = files_loaded.copy()
    for file_name, file_info in files_loaded.items():
        with st.sidebar.expander(file_name):
            st.write(f"{UI_TEXT['file_title']} {file_name}")

            with st.form(key=f"manage_form_{file_name}", border=0):
                file_name = st.text_input(
                    UI_TEXT["file_name"], value=file_info["file_name"]
                )
                if "separator" in file_info:
                    col1, col2 = st.columns(2)
                    with col1:
                        separator = st.text_input(
                            UI_TEXT["separator"],
                            value=file_info["separator"],
                            help=UI_TEXT["separator_help"],
                            disabled=True,
                        )
                        decimal = st.text_input(
                            UI_TEXT["decimal"],
                            value=file_info["decimal"],
                            help=UI_TEXT["decimal_help"],
                            disabled=True,
                        )
                    with col2:
                        thousands = st.text_input(
                            UI_TEXT["thousands"],
                            value=file_info["thousands"],
                            help=UI_TEXT["thousands_help"],
                            disabled=True,
                        )
                        encoding = st.text_input(
                            UI_TEXT["encoding"],
                            value=file_info["encoding"],
                            help=UI_TEXT["encoding_help"],
                            disabled=True,
                        )
                    date_format = st.text_input(
                        UI_TEXT["date_format"],
                        value=file_info["date_format"],
                        help=UI_TEXT["date_format_help"],
                    )
                col1, col2, col3 = st.columns(3)
                with col2:
                    update_button = st.form_submit_button(label=UI_TEXT["update_file"])
            col1, col2, col3 = st.columns(3)
            with col2:
                remove_button = st.button(UI_TEXT["remove_file"])
            if update_button:
                del remaining_files[file_info["file_name"]]
                remaining_files[file_name] = {
                    "data": file_info["data"],
                    "separator": separator,
                    "decimal": decimal,
                    "thousands": thousands,
                    "encoding": encoding,
                    "date_format": date_format,
                    "file_name": file_name,
                }

            if remove_button:
                del remaining_files[file_name]
    if remaining_files.keys() != files_loaded.keys():
        st.session_state["files_loaded"] = remaining_files
        st.rerun()
