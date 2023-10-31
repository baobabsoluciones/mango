import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# Config
APP_CONFIG = {
    "title": "Mango Explorer App",
    "header": "Folder path structure",
    "icon": ":mango:",
    "layout": "wide",  # centered
    "dir_path": r"G:\Unidades compartidas\mango\desarrollo\datos\file_explorer_folder",
    "logo": r"G:\Unidades compartidas\mango\desarrollo\datos\logo.png",
}


class DisplayablePath(object):
    display_filename_prefix_middle = "├──"
    display_filename_prefix_last = "└──"
    display_parent_prefix_middle = "    "
    display_parent_prefix_last = "│   "

    def __init__(self, path, parent_path, is_last):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + "/"
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(
            list(path for path in root.iterdir() if criteria(path)),
            key=lambda s: str(s).lower(),
        )
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(
                    path, parent=displayable_root, is_last=is_last, criteria=criteria
                )
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + "/"
        return self.path.name

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (
            self.display_filename_prefix_last
            if self.is_last
            else self.display_filename_prefix_middle
        )

        parts = ["{!s} {!s}".format(_filename_prefix, self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(
                self.display_parent_prefix_middle
                if parent.is_last
                else self.display_parent_prefix_last
            )
            parent = parent.parent

        return "".join(reversed(parts))


def _element_selector(
    folder_path: str,
    element_type: str = "file",
    key: str = None,
    dict_layout: dict = None,
):
    paths = []
    for root, dirs, files in os.walk(folder_path):
        if element_type == "file":
            for element in files:
                path = os.path.join(root, element)
                paths.append(path)
        elif element_type == "folder":
            for element in dirs:
                path = os.path.join(root, element)
                paths.append(path)
        else:
            raise ValueError(
                "element_type must be 'file' or 'folder', but got {}".format(
                    element_type
                )
            )

    # Selectbox
    if element_type == "folder":
        paths = [os.path.basename(folder_path)] + paths
        # Return index of paths that match with the key
        default_index = (
            [i for i, s in enumerate(paths) if dict_layout[key] in s][0]
            if key in dict_layout.keys()
            else 0
        )

    else:
        # Return index of paths that match with the key
        default_index = (
            [i for i, s in enumerate(paths) if dict_layout[key] in s][0]
            if key in dict_layout.keys()
            else None
        )

    element_selected = st.selectbox(
        f"Select a {element_type}",
        paths,
        placeholder="Choose an option",
        index=default_index,
        format_func=lambda x: x.replace(folder_path, ""),
        key=key,
    )
    if element_selected:
        return os.path.join(folder_path, element_selected)
    else:
        return ""


def _render_file_content(path_selected: str):
    if path_selected.endswith(".csv"):
        df = pd.read_csv(path_selected)
        st.dataframe(df, use_container_width=True)

    elif (
        path_selected.endswith(".png")
        or path_selected.endswith(".jpg")
        or path_selected.endswith(".jpeg")
    ):
        image = Image.open(path_selected)
        with st.spinner("Wait for it..."):
            st.image(image)

    elif path_selected.endswith(".html"):
        with st.spinner("Wait for it..."):
            with open(path_selected, "r") as f:
                components.html(f.read(), height=500)

    elif path_selected.endswith(".xlsx"):
        with st.spinner("Wait for it..."):
            excel_file = pd.ExcelFile(path_selected)
            sheets = excel_file.sheet_names
            sheet_selected = st.selectbox(
                "Select a sheet", sheets, index=0, format_func=lambda x: x
            )
            df = pd.read_excel(excel_file, sheet_name=sheet_selected)
            st.dataframe(df, use_container_width=True)

    elif path_selected.endswith(".md"):
        with st.spinner("Wait for it..."):
            text_md = Path(path_selected).read_text()
            st.markdown(text_md, unsafe_allow_html=True)

    elif path_selected == "":
        pass
    else:
        st.warning("This file format is not supported yet. Please try another file.")


def _render_tree_folder(dir_path: str):
    paths = DisplayablePath.make_tree(Path(dir_path), criteria=lambda p: p.is_dir())
    displayable_path = ""
    for path in paths:
        displayable_path = displayable_path + "  \n" + path.displayable()
    st.info(displayable_path)


def _render_dropdown(dict_layout: dict, i_row: int, i_col: int):
    # Select folder
    dict_layout[f"folder_{i_row}_{i_col}"] = _element_selector(
        folder_path=APP_CONFIG["dir_path"],
        element_type="folder",
        key=f"folder_{i_row}_{i_col}",
        dict_layout=dict_layout,
    )
    # Select file
    if os.path.basename(dict_layout[f"folder_{i_row}_{i_col}"]) == os.path.basename(
        APP_CONFIG["dir_path"]
    ):
        dict_layout[f"folder_{i_row}_{i_col}"] = APP_CONFIG["dir_path"]
    dict_layout[f"file_{i_row}_{i_col}"] = _element_selector(
        folder_path=dict_layout[f"folder_{i_row}_{i_col}"],
        element_type="file",
        key=f"file_{i_row}_{i_col}",
        dict_layout=dict_layout,
    )

    # Render file content
    _render_file_content(path_selected=dict_layout[f"file_{i_row}_{i_col}"])


def _render_body_content(n_cols: int, n_rows: int, dict_layout: dict = {}):
    for i_row in range(1, n_rows + 1):
        if n_cols == 1:
            _render_dropdown(dict_layout=dict_layout, i_row=i_row, i_col=1)

        elif n_cols == 2:
            # Create columns display
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                _render_dropdown(dict_layout=dict_layout, i_row=i_row, i_col=1)

            with col2_2:
                _render_dropdown(dict_layout=dict_layout, i_row=i_row, i_col=2)
    return dict_layout


def _save_config(dict_update: dict):
    APP_CONFIG.update(dict_update)
    with open("config.json", "w") as f:
        json.dump(APP_CONFIG, f)


def app():
    # Load config
    if os.path.isfile("config.json"):
        with open("config.json", "r") as f:
            APP_CONFIG.update(json.load(f))

    # Set page config
    st.set_page_config(
        page_title=APP_CONFIG["title"],
        page_icon=APP_CONFIG["icon"],
        layout=APP_CONFIG["layout"],
        initial_sidebar_state="collapsed",
    )

    # Create columns display
    col3_1, col3_2, col3_3 = st.columns(3)
    with col3_1:
        st.image(Image.open(APP_CONFIG["logo"]), width=150)
        st.markdown(
            """<style>button[title="View fullscreen"]{visibility: hidden;}</style>""",
            unsafe_allow_html=True,
        )
    with col3_2:
        st.title(APP_CONFIG["title"])
        st.markdown(
            """<style>.st-emotion-cache-15zrgzn {display: none;}</style>""",
            unsafe_allow_html=True,
        )
    st.header(APP_CONFIG["header"])

    with st.sidebar:
        config_dict_update = {}
        st.header("Configuration")

        # Select folder
        config_dict_update["dir_path"] = st.text_input(
            "Folder", value=APP_CONFIG["dir_path"], max_chars=None, key="input_dir_path"
        )
        if not os.path.isdir(st.session_state.input_dir_path):
            st.error("Please enter a valid folder path")

        # Select number of columns and rows
        config_dict_update["n_cols"] = st.number_input(
            "Column Levels",
            key="config_cols",
            min_value=1,
            max_value=2,
            value=APP_CONFIG["n_cols"] if "n_cols" in APP_CONFIG.keys() else 1,
        )
        config_dict_update["n_rows"] = st.number_input(
            "Row Levels",
            key="config_rows",
            min_value=1,
            max_value=10,
            value=APP_CONFIG["n_rows"] if "n_rows" in APP_CONFIG.keys() else 1,
        )

        # Save config
        st.button(
            "Apply configuration",
            key="save_config",
            on_click=_save_config,
            args=(config_dict_update,),
        )

    # Render folder tree
    _render_tree_folder(dir_path=APP_CONFIG["dir_path"])

    # Render body content
    APP_CONFIG["dict_layout"] = _render_body_content(
        n_cols=config_dict_update["n_cols"],
        n_rows=config_dict_update["n_rows"],
        dict_layout=APP_CONFIG["dict_layout"]
        if "dict_layout" in APP_CONFIG.keys()
        else {},
    )


if __name__ == "__main__":
    app()
