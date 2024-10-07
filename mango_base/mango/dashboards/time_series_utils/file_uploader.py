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
        UI_TEXT["upload_file"],
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
                st.write(f"{UI_TEXT['file_title']} {uploaded_file.name}")
                extension = uploaded_file.name.split(".")[-1]

                file_name = st.text_input(
                    UI_TEXT["file_name"], value=uploaded_file.name
                )
                if extension == "csv":
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        separator = st.selectbox(
                            UI_TEXT["separator"], options=[",", ";", "|", "\t"], index=0, key=f"separator_{uploaded_file.name}"
                        )
                    with col2:
                        decimal = st.text_input(
                            UI_TEXT["decimal"], value=".", help=UI_TEXT["decimal_help"], key=f"decimal_{uploaded_file.name}"
                        )
                    with col3:
                        thousands = st.text_input(
                            UI_TEXT["thousands"],
                            value=",",
                            help=UI_TEXT["thousands_help"], key=f"thousands_{uploaded_file.name}"
                        )
                    with col4:
                        encoding = st.text_input(
                            UI_TEXT["encoding"],
                            value="utf-8",
                            help=UI_TEXT["encoding_help"], key=f"encoding_{uploaded_file.name}"
                        )
                    with col5:
                        date_format = st.text_input(
                            UI_TEXT["date_format"],
                            value="auto",
                            help=UI_TEXT["date_format_help"], key=f"date_format_{uploaded_file.name}"
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
                            pd.to_datetime, format=date_format if date_format!="auto" else None
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
            submit_button = st.form_submit_button(label=UI_TEXT["load_data"])
        if submit_button:
            files_loaded.update(files_loaded_tmp)
    return files_loaded, no_model_column


def manage_files(files_loaded, UI_TEXT):
    st.sidebar.write(UI_TEXT["manage_files"])
    remaining_files = files_loaded.copy()
    extension = list(remaining_files.keys())[0].split(".")[-1]

    for file_name_old, file_info in files_loaded.items():
        with st.sidebar.expander(file_name_old):
            st.write(f"{UI_TEXT['file_title']} {file_name_old}")
            no_model_column = False
            with st.form(key=f"manage_form_{file_name_old}", border=0):
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
                            # disabled=True,
                            key=f"separator_{file_name_old}",
                        )
                        decimal = st.text_input(
                            UI_TEXT["decimal"],
                            value=file_info["decimal"],
                            help=UI_TEXT["decimal_help"],
                            # disabled=True,
                            key=f"decimal_{file_name_old}",
                        )
                    with col2:
                        thousands = st.text_input(
                            UI_TEXT["thousands"],
                            value=file_info["thousands"],
                            help=UI_TEXT["thousands_help"],
                            # disabled=True,
                            key=f"thousands_{file_name_old}",
                        )
                        encoding = st.text_input(
                            UI_TEXT["encoding"],
                            value=file_info["encoding"],
                            help=UI_TEXT["encoding_help"],
                            # disabled=True,
                            key=f"encoding_{file_name_old}",
                        )
                    date_format = st.text_input(
                        UI_TEXT["date_format"],
                        value=file_info["date_format"],
                        help=UI_TEXT["date_format_help"],
                        key=f"date_format_{file_name_old}",
                    )
                col1, col2, col3 = st.columns(3)
                with col2:
                    update_button = st.form_submit_button(label=UI_TEXT["update_file"])
            col1, col2, col3 = st.columns(3)
            with col2:
                remove_button = st.button(UI_TEXT["remove_file"], key=f"remove_{file_name_old}")
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

    # if st.sidebar.button(UI_TEXT["add_new_file"]):
    #     new_files = upload_files(UI_TEXT)
    #     files_loaded.update(new_files)
    #     st.session_state["files_loaded"] = files_loaded

    if remaining_files.keys() != files_loaded.keys():
        st.session_state["files_loaded"] = remaining_files
        st.rerun()

    return
