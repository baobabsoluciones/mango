import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image


def _file_selector(folder_path: str):
    paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filename = os.path.join(root, file)
            paths.append(filename)

    # Selectbox
    path_selected = st.selectbox(
        "Select a file",
        paths,
        placeholder="Choose an option",
        index=None,
        format_func=lambda x: x.replace(folder_path, ""),
    )
    if path_selected:
        return os.path.join(folder_path, path_selected)
    else:
        return ""


def _render_file_content(path_selected: str):
    if path_selected.endswith(".csv"):
        df = pd.read_csv(path_selected)
        st.dataframe(df)

    elif path_selected.endswith(".png") or path_selected.endswith(".jpg") or path_selected.endswith(".jpeg"):
        image = Image.open(path_selected)
        with st.spinner("Wait for it..."):
            st.image(image)

    elif path_selected.endswith(".html"):
        with st.spinner("Wait for it..."):
            with open(path_selected, "r") as f:
                components.html(f.read(), height=500)

    elif path_selected == "":
        pass
    else:
        st.warning("This file format is not supported yet. Please try another file.")


def app(
    dir_path: str = r"G:\Unidades compartidas\mango\desarrollo\datos\file_explorer_folder",
):
    st.title("File Explorer")
    path_selected = _file_selector(folder_path=dir_path)

    # Render file content
    _render_file_content(path_selected=path_selected)


if __name__ == "__main__":
    app()
