from typing import Dict

import pandas as pd
import streamlit as st


def preview_data(
    uploaded_file, separator, decimal, thousands, encoding, date_format, ui_text
):
    """Preview data from an uploaded file with specified parsing parameters.

    This function displays a preview of the uploaded file data using the
    specified parsing parameters. It handles CSV files with custom separators,
    decimal/thousands separators, encoding, and date formats. Automatically
    converts datetime columns if they exist in the data.

    :param uploaded_file: The uploaded file object from Streamlit
    :type uploaded_file: UploadedFile
    :param separator: Field separator used in the CSV file
    :type separator: str
    :param decimal: Decimal separator for numeric values
    :type decimal: str
    :param thousands: Thousands separator for numeric values
    :type thousands: str
    :param encoding: Text encoding of the file
    :type encoding: str
    :param date_format: Date format for datetime parsing
    :type date_format: str
    :param ui_text: Dictionary containing UI text labels
    :type ui_text: Dict
    :return: None
    :rtype: None

    Example:
        >>> preview_data(
        ...     uploaded_file, ',', '.', ',', 'utf-8', '%Y-%m-%d', ui_text
        ... )
        # Displays preview of the uploaded data
    """
    st.write(ui_text["preview_title"])
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


def upload_files(ui_text: Dict):
    """Upload and configure multiple files for time series analysis.

    This function provides a file upload interface that allows users to upload
    CSV and Excel files with custom parsing parameters. It creates a form-based
    interface for configuring file-specific settings like separators, encoding,
    and date formats. Automatically handles model column detection and creation.

    Features:
        - Support for CSV and Excel files
        - Custom parsing parameters per file
        - Automatic datetime column detection and conversion
        - Model column auto-creation if missing
        - Form-based configuration interface

    :param ui_text: Dictionary containing UI text labels and messages
    :type ui_text: Dict
    :return: Tuple containing (files_loaded_dict, no_model_column_flag)
    :rtype: tuple[Dict, bool]

    Example:
        >>> files_loaded, no_model = upload_files(ui_text)
        >>> print(f"Loaded {len(files_loaded)} files")
        Loaded 3 files
    """
    files = st.file_uploader(
        ui_text["upload_file"],
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key="file_uploader",
    )
    files_loaded = {}
    files_loaded_tmp = {}
    no_model_column = False

    if files:
        with st.form(key=f"form"):
            for uploaded_file in files:
                st.write(f"{ui_text['file_title']} {uploaded_file.name}")
                extension = uploaded_file.name.split(".")[-1]

                file_name = st.text_input(
                    ui_text["file_name"], value=uploaded_file.name
                )
                if extension == "csv":
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        separator = st.selectbox(
                            ui_text["separator"],
                            options=[",", ";", "|", "\t"],
                            index=0,
                            key=f"separator_{uploaded_file.name}",
                        )
                    with col2:
                        decimal = st.text_input(
                            ui_text["decimal"],
                            value=".",
                            help=ui_text["decimal_help"],
                            key=f"decimal_{uploaded_file.name}",
                        )
                    with col3:
                        thousands = st.text_input(
                            ui_text["thousands"],
                            value=",",
                            help=ui_text["thousands_help"],
                            key=f"thousands_{uploaded_file.name}",
                        )
                    with col4:
                        encoding = st.text_input(
                            ui_text["encoding"],
                            value="utf-8",
                            help=ui_text["encoding_help"],
                            key=f"encoding_{uploaded_file.name}",
                        )
                    with col5:
                        date_format = st.text_input(
                            ui_text["date_format"],
                            value="auto",
                            help=ui_text["date_format_help"],
                            key=f"date_format_{uploaded_file.name}",
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
                            pd.to_datetime,
                            format=date_format if date_format != "auto" else None,
                        )
                elif extension == "xlsx":
                    data_frame = pd.read_excel(uploaded_file)
                if "model" not in data_frame.columns:
                    data_frame["model"] = file_name
                    no_model_column = True

                if extension == "csv":
                    files_loaded_tmp[file_name] = {
                        "data": data_frame,
                        "separator": separator,
                        "decimal": decimal,
                        "thousands": thousands,
                        "encoding": encoding,
                        "date_format": date_format,
                        "file_name": file_name,
                    }
                elif extension == "xlsx":
                    files_loaded_tmp[file_name] = {
                        "data": data_frame,
                        "file_name": file_name,
                    }
            submit_button = st.form_submit_button(label=ui_text["load_data"])
        if submit_button:
            files_loaded.update(files_loaded_tmp)
    return files_loaded, no_model_column


def manage_files(files_loaded, ui_text):
    """Manage and update previously uploaded files.

    This function provides a management interface in the sidebar for updating
    or removing previously uploaded files. Users can modify file names, parsing
    parameters, and remove files from the analysis. Changes are applied
    immediately and trigger a Streamlit rerun.

    Features:
        - File name editing
        - Parsing parameter updates (for CSV files)
        - File removal functionality
        - Automatic session state updates
        - Model column synchronization

    :param files_loaded: Dictionary of currently loaded files with their configurations
    :type files_loaded: Dict
    :param ui_text: Dictionary containing UI text labels and messages
    :type ui_text: Dict
    :return: None (updates session state directly)
    :rtype: None

    Example:
        >>> manage_files(files_loaded, ui_text)
        # Updates files in sidebar with management options
    """
    st.sidebar.write(ui_text["manage_files"])
    remaining_files = files_loaded.copy()
    extension = list(remaining_files.keys())[0].split(".")[-1]

    for file_name_old, file_info in files_loaded.items():
        with st.sidebar.expander(file_name_old):
            st.write(f"{ui_text['file_title']} {file_name_old}")
            with st.form(key=f"manage_form_{file_name_old}", border=0):
                file_name = st.text_input(
                    ui_text["file_name"], value=file_info["file_name"]
                )

                if "separator" in file_info:
                    col1, col2 = st.columns(2)
                    with col1:
                        separator = st.text_input(
                            ui_text["separator"],
                            value=file_info["separator"],
                            help=ui_text["separator_help"],
                            # disabled=True,
                            key=f"separator_{file_name_old}",
                        )
                        decimal = st.text_input(
                            ui_text["decimal"],
                            value=file_info["decimal"],
                            help=ui_text["decimal_help"],
                            # disabled=True,
                            key=f"decimal_{file_name_old}",
                        )
                    with col2:
                        thousands = st.text_input(
                            ui_text["thousands"],
                            value=file_info["thousands"],
                            help=ui_text["thousands_help"],
                            # disabled=True,
                            key=f"thousands_{file_name_old}",
                        )
                        encoding = st.text_input(
                            ui_text["encoding"],
                            value=file_info["encoding"],
                            help=ui_text["encoding_help"],
                            # disabled=True,
                            key=f"encoding_{file_name_old}",
                        )
                    date_format = st.text_input(
                        ui_text["date_format"],
                        value=file_info["date_format"],
                        help=ui_text["date_format_help"],
                        key=f"date_format_{file_name_old}",
                    )
                col1, col2, col3 = st.columns(3)
                with col2:
                    update_button = st.form_submit_button(label=ui_text["update_file"])
            col1, col2, col3 = st.columns(3)
            with col2:
                remove_button = st.button(
                    ui_text["remove_file"], key=f"remove_{file_name_old}"
                )
            if update_button:

                del remaining_files[file_info["file_name"]]
                df = file_info["data"].copy()
                if (df["model"] == file_name_old).all():
                    df = file_info["data"].copy()
                    df["model"] = file_name

                if extension == "csv":
                    remaining_files[file_name] = {
                        "data": file_info["data"],
                        "separator": separator,
                        "decimal": decimal,
                        "thousands": thousands,
                        "encoding": encoding,
                        "date_format": date_format,
                        "file_name": file_name,
                    }
                elif extension == "xlsx":
                    remaining_files[file_name] = {
                        "data": file_info["data"],
                        "file_name": file_name,
                    }
                st.session_state["files_loaded"] = remaining_files
                st.session_state["selected_series"] = []
                st.rerun()
            if remove_button:
                del remaining_files[file_name]

    if remaining_files.keys() != files_loaded.keys():
        st.session_state["files_loaded"] = remaining_files
        st.rerun()
