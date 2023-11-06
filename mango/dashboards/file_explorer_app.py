import json
import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image


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


class FileExplorerApp:
    def __init__(
        self, path: str = None, editable: bool = True, config_path: str = None
    ):
        # Config
        _APP_CONFIG = {
            "logo_path": os.path.join(os.path.dirname(__file__), "assets", "logo.png"),
            "title": "Mango Explorer App",
            "header": "Folder path structure",
            "icon": ":mango:",
            "layout": "wide",
            "dir_path": os.getcwd(),
            "dict_layout": {},
        }
        if config_path is None:
            if os.path.exists(os.path.join(os.getcwd(), "config.json")):
                self.config_path = os.path.join(os.getcwd(), "config.json")
                with open(os.path.join(os.getcwd(), "config.json"), "r") as f:
                    config_dict = json.load(f)
            else:
                self.config_path = os.path.join(os.getcwd(), "config.json")
                config_dict = _APP_CONFIG
        else:
            if not os.path.exists(config_path):
                self.config_path = os.path.join(os.getcwd(), "config.json")
                config_dict = _APP_CONFIG
            else:
                self.config_path = config_path
                with open(config_path, "r") as f:
                    config_dict = json.load(f)

        if path:
            if os.path.exists(path):
                config_dict["dir_path"] = path

        # Set config
        self.config = config_dict.copy()
        self.new_config = config_dict.copy()
        self.editable = editable

        # Set Streamlit config
        st.set_page_config(
            page_title=self.config["title"],
            page_icon=self.config["icon"],
            layout=self.config["layout"],
            initial_sidebar_state="collapsed",
        )

    def _render_header(self):
        # Create columns display
        col3_1, col3_2, col3_3 = st.columns(3)
        with col3_1:
            st.image(Image.open(self.config["logo_path"]), width=150)
            st.markdown(
                """<style>button[title="View fullscreen"]{visibility: hidden;}</style>""",
                unsafe_allow_html=True,
            )
        with col3_2:
            st.title(self.config["title"])
            st.markdown(
                """<style>.st-emotion-cache-15zrgzn {display: none;}</style>""",
                unsafe_allow_html=True,
            )
        if self.editable:
            st.header(self.config["header"])

    def _render_configuration(self):
        with st.sidebar:
            st.header("Configuration")

            # Select folder
            self.new_config["dir_path"] = st.text_input(
                "Folder",
                value=self.config["dir_path"],
                max_chars=None,
                key="input_dir_path",
                disabled=True,
            )
            if not os.path.isdir(self.new_config["dir_path"]):
                st.error("Please enter a valid folder path")

            # Select config path
            self.config_path = st.text_input(
                "Configuration path",
                value=self.config_path,
                max_chars=None,
                key="input_config_path",
                disabled=True,
            )
            if not os.path.basename(self.config_path).endswith(
                ".json"
            ) or not os.path.exists(os.path.dirname(self.config_path)):
                st.error(
                    "Please enter a valid configuration path: *.json in correct folder"
                )

            # Select title
            self.new_config["title"] = st.text_input(
                "Title",
                value=self.config["title"],
                max_chars=None,
                key="title",
            )

            # Select number of columns and rows
            self.new_config["n_cols"] = st.number_input(
                "Column Levels",
                key="config_cols",
                min_value=1,
                max_value=2,
                value=self.new_config["n_cols"]
                if "n_cols" in self.new_config.keys()
                else 1,
            )
            self.new_config["n_rows"] = st.number_input(
                "Row Levels",
                key="config_rows",
                min_value=1,
                max_value=10,
                value=self.new_config["n_rows"]
                if "n_rows" in self.new_config.keys()
                else 1,
            )

            # Save config
            if os.path.isdir(self.new_config["dir_path"]) and (
                os.path.basename(self.config_path).endswith(".json")
                and os.path.exists(os.path.dirname(self.config_path))
            ):
                st.button(
                    "Apply & Save Configuration",
                    key="save_config",
                    on_click=self._save_config,
                )

    def _render_tree_folder(self):
        paths = DisplayablePath.make_tree(
            Path(self.config["dir_path"]), criteria=lambda p: p.is_dir()
        )
        displayable_path = ""
        for path in paths:
            displayable_path = displayable_path + "  \n" + path.displayable()
        st.info(displayable_path)

    def _render_dropdown(self, i_row: int, i_col: int):
        if self.editable:
            # Select folder
            self._element_selector(
                folder_path=self.config["dir_path"],
                element_type="folder",
                key=f"folder_{i_row}_{i_col}",
            )

            # Select file
            self._element_selector(
                folder_path=st.session_state[f"folder_{i_row}_{i_col}"]
                if f"folder_{i_row}_{i_col}" in st.session_state.keys()
                else self.config["dir_path"],
                element_type="file",
                key=f"file_{i_row}_{i_col}",
            )

            # Render file content
            self._render_file_content(
                path_selected=st.session_state[f"file_{i_row}_{i_col}"]
                if f"file_{i_row}_{i_col}" in st.session_state.keys()
                else ""
            )

        else:
            # Render file content
            self._render_file_content(
                path_selected=self.config["dict_layout"][f"file_{i_row}_{i_col}"]
                if f"file_{i_row}_{i_col}" in self.config["dict_layout"].keys()
                else ""
            )

    def _element_selector(
        self,
        folder_path: str,
        element_type: str = "file",
        key: str = None,
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
            paths = [folder_path] + paths
        else:
            paths = ["Select a file"] + paths
        # Return index of paths that match with the key
        default_index = (
            [i for i, s in enumerate(paths) if self.config["dict_layout"][key] in s][0]
            if key in self.config["dict_layout"].keys()
            else 0
        )
        st.selectbox(
            f"Select a {element_type}",
            paths,
            placeholder="Choose an option",
            index=default_index,
            format_func=lambda x: x.replace(folder_path, "")
            if x != self.config["dir_path"]
            else os.path.basename(x),
            key=key,
        )

    def _render_file_content(self, path_selected: str):
        if (
            path_selected is None
            or path_selected == ""
            or path_selected == "Select a file"
        ):
            pass
        elif path_selected.endswith(".csv"):
            df = pd.read_csv(path_selected)
            st.dataframe(df.astype(str), use_container_width=True)

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
                list_of_tabs = st.tabs(sheets)
                dict_of_tabs = {}
                for i, tab in enumerate(list_of_tabs):
                    dict_of_tabs[sheets[i]] = tab

                for key_tab, tab in dict_of_tabs.items():
                    with tab:
                        df = pd.read_excel(excel_file, sheet_name=key_tab)
                        st.dataframe(df.astype(str), use_container_width=True)

        elif path_selected.endswith(".md"):
            with st.spinner("Wait for it..."):
                text_md = Path(path_selected).read_text()
                st.markdown(text_md, unsafe_allow_html=True)

        else:
            st.warning(
                "This file format is not supported yet. Please try another file."
            )

    def _render_body_content(self):
        n_cols = self.config["n_cols"] if "n_cols" in self.config.keys() else 1
        n_rows = self.config["n_rows"] if "n_rows" in self.config.keys() else 1

        for i_row in range(1, n_rows + 1):
            st.markdown("---")
            if n_cols == 1:
                self._render_dropdown(i_row=i_row, i_col=1)

            elif n_cols == 2:
                # Create columns display
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    self._render_dropdown(i_row=i_row, i_col=1)

                with col2_2:
                    self._render_dropdown(i_row=i_row, i_col=2)

    def _save_config(self):
        for row in range(1, self.new_config["n_rows"] + 1):
            for col in range(1, self.new_config["n_cols"] + 1):
                try:
                    if st.session_state[f"folder_{row}_{col}"] != None:
                        self.new_config["dict_layout"][
                            f"folder_{row}_{col}"
                        ] = st.session_state[f"folder_{row}_{col}"]
                except:
                    pass
                try:
                    if st.session_state[f"file_{row}_{col}"] != None:
                        self.new_config["dict_layout"][
                            f"file_{row}_{col}"
                        ] = st.session_state[f"file_{row}_{col}"]
                except:
                    pass
        # Drop keys that are not in the new config
        keys_to_del = [
            key_layout
            for key_layout in self.config["dict_layout"].keys()
            if int(key_layout.split("_")[2]) > self.new_config["n_cols"]
        ]
        _ = [
            self.new_config["dict_layout"].pop(key_to_del) for key_to_del in keys_to_del
        ]
        keys_to_del = [
            key_layout
            for key_layout in self.config["dict_layout"].keys()
            if int(key_layout.split("_")[1]) > self.new_config["n_rows"]
        ]
        _ = [
            self.new_config["dict_layout"].pop(key_to_del) for key_to_del in keys_to_del
        ]

        # Change folder
        if self.new_config["dir_path"] != self.config["dir_path"]:
            self.new_config["dict_layout"] = {}

        self.config.update(self.new_config)
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, sort_keys=True, indent=4)

    def run(self):
        # Render header
        self._render_header()

        if self.editable:
            # Render configuration
            self._render_configuration()

        if self.editable:
            with st.spinner("Wait for it..."):
                # Render folder tree
                self._render_tree_folder()

        # Render body content
        self._render_body_content()

        # Not Render footer
        st.markdown(
            "<style>footer{visibility: hidden;}</style>", unsafe_allow_html=True
        )


if __name__ == "__main__":
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--editable", type=int, default=True, choices=[0, 1])
    args = parser.parse_args()
    path = args.path
    config_path = args.config_path
    editable = args.editable

    # Run app
    app = FileExplorerApp(path=path, editable=editable, config_path=config_path)
    app.run()
