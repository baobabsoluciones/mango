import pandas as pd
import streamlit as st
import io


def preview_data(uploaded_file, separator, decimal, thousands, encoding, date_format):
    st.write("### Preview")
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


def upload_files():
    files = st.file_uploader(
        "Upload a file", type=["csv", "xlsx"], accept_multiple_files=True
    )
    files_loaded = {}

    if files:
        for uploaded_file in files:
            st.write(f"### File: {uploaded_file.name}")
            extension = uploaded_file.name.split(".")[-1]

            # Create a form for user inputs
            with st.form(key=f"form_{uploaded_file.name}"):
                file_name = st.text_input("File name", value=uploaded_file.name)
                if extension == "csv":
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        separator = st.selectbox(
                            "Separator", options=[",", ";", "|", "\t"], index=0
                        )
                    with col2:
                        decimal = st.text_input(
                            "Decimal", value=".", help="e.g., '.', ','"
                        )
                    with col3:
                        thousands = st.text_input(
                            "Thousands", value=",", help="e.g., ',', '.', ' '"
                        )
                    with col4:
                        encoding = st.text_input(
                            "Encoding",
                            value="utf-8",
                            help="e.g., 'utf-8', 'latin1', 'ascii'",
                        )
                    with col5:
                        date_format = st.text_input(
                            "Date format",
                            value="%Y-%m-%d",
                            help="e.g., '%Y-%m-%d', '%d/%m/%Y'",
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

                submit_button = st.form_submit_button(label="Load Data")

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


def manage_files(files_loaded):
    st.sidebar.write("### Manage Uploaded Files")
    remaining_files = files_loaded.copy()
    for file_name, file_info in files_loaded.items():
        with st.sidebar.expander(file_name):
            st.write("### File: ", file_name)

            # Create a form for user inputs
            # Clear border
            with st.form(key=f"manage_form_{file_name}", border=0):
                file_name = st.text_input("File name", value=file_info["file_name"])
                if "separator" in file_info:
                    col1, col2 = st.columns(2)
                    with col1:
                        separator = st.text_input(
                            "Separator",
                            value=file_info["separator"],
                            help="e.g., ',', ';', '|', '\\t'",
                            disabled=True,
                        )
                        decimal = st.text_input(
                            "Decimal",
                            value=file_info["decimal"],
                            help="e.g., '.', ','",
                            disabled=True,
                        )
                    with col2:
                        thousands = st.text_input(
                            "Thousands",
                            value=file_info["thousands"],
                            help="e.g., ',', '.', ' '",
                            disabled=True,
                        )
                        encoding = st.text_input(
                            "Encoding",
                            value=file_info["encoding"],
                            help="e.g., 'utf-8', 'latin1', 'ascii'",
                            disabled=True,
                        )
                    date_format = st.text_input(
                        "Date format",
                        value=file_info["date_format"],
                        help="e.g., '%Y-%m-%d', '%d/%m/%Y'",
                    )
                # Center the buttons
                col1, col2, col3 = st.columns(3)
                with col2:
                    update_button = st.form_submit_button(label="Update File")
            col1, col2, col3 = st.columns(3)
            with col2:
                remove_button = st.button("Remove File")
            if update_button:
                # Update the CSV file with user-defined parameters
                # data_frame = pd.read_csv(
                #     file_info["raw_file"],
                #     sep=separator,
                #     decimal=decimal,
                #     thousands=thousands,
                #     encoding=encoding,
                #     parse_dates=True,
                #     infer_datetime_format=True,
                # )
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
