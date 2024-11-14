import json
import os
import re
import uuid
import webbrowser
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import plotly
import streamlit as st
from PIL import Image
from mango.processing import write_json, load_json
from mango.table import Table


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
    # Config
    _APP_CONFIG = {
        "logo_path": os.path.join("assets", "logo.png"),
        "title": "Mango Explorer App",
        "header": "Folder path structure",
        "icon": ":mango:",
        "layout": "wide",
        "dir_path": os.getcwd(),
        "dict_layout": {},
    }

    def __init__(
        self, path: str = None, editable: bool = True, config_path: str = None
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines all its attributes.


        :param self: Represent the instance of the class
        :param str path: Set the default path of the app
        :param bool editable: Determine whether the user can edit the config file or not
        :param str config_path: Set the path to a config file
        :doc-author: baobab soluciones
        """
        if config_path is None:
            if os.path.exists(os.path.join(os.getcwd(), "config.json")):
                self.config_path = os.path.join(os.getcwd(), "config.json")
                with open(os.path.join(os.getcwd(), "config.json"), "r") as f:
                    config_dict = json.load(f)
            else:
                self.config_path = os.path.join(os.getcwd(), "config.json")
                config_dict = self._APP_CONFIG
        else:
            if not os.path.exists(config_path):
                if os.path.exists(os.path.dirname(config_path)):
                    self.config_path = os.path.join(
                        os.path.dirname(config_path), "config.json"
                    )
                    config_dict = self._APP_CONFIG
                else:
                    self.config_path = os.path.join(os.getcwd(), "config.json")
                    config_dict = self._APP_CONFIG
            else:
                self.config_path = config_path
                with open(config_path, "r") as f:
                    config_dict = json.load(f)

        if path:
            if os.path.exists(path):
                config_dict["dir_path"] = path

        # Set config
        self.config = config_dict.copy()
        if self.config.get("dict_layout", None) is None:
            self.config["dict_layout"] = self._APP_CONFIG["dict_layout"]
        if editable is None:
            if self.config.get("editable", None) is not None:
                self.editable = self.config["editable"]
            else:
                self.editable = True
        else:
            self.editable = True

        # Set Streamlit config
        st.set_page_config(
            page_title=self.config.get("title", self._APP_CONFIG["title"]),
            page_icon=self.config.get("icon", self._APP_CONFIG["icon"]),
            layout=self.config.get("layout", self._APP_CONFIG["layout"]),
            initial_sidebar_state="collapsed",
        )

    def _render_header(self):
        """
        The _render_header function is a helper function that renders the header of the app.
        It takes in no arguments and returns nothing. It uses Streamlit's st module to render
        the logo, title, and header of the app.

        :param self: Refer to the instance of the class
        :return: None.
        :doc-author: baobab soluciones
        """
        # Create columns display
        col3_1, col3_2, col3_3 = st.columns(3)
        with col3_1:
            if self.config.get("logo_path") is not None and os.path.exists(
                self.config["logo_path"]
            ):
                path_logo = self.config["logo_path"]
            else:
                path_logo = os.path.join(
                    os.path.dirname(__file__), self._APP_CONFIG["logo_path"]
                )

            st.image(
                Image.open(path_logo),
                width=150,
            )
            st.markdown(
                """<style>button[title="View fullscreen"]{visibility: hidden;}</style>""",
                unsafe_allow_html=True,
            )
        with col3_2:
            st.title(self.config.get("title", self._APP_CONFIG["title"]))
            st.markdown(
                """<style>.st-emotion-cache-15zrgzn {display: none;}</style>""",
                unsafe_allow_html=True,
            )
        if self.editable:
            st.header(self.config.get("header", self._APP_CONFIG["header"]))

    def _render_configuration(self):
        """
        The _render_configuration function is used to render the configuration sidebar.
        It allows the user to select a folder, a config path, and set title and number of columns/rows.
        The function also checks if the selected folder exists and if it does not, an error message is displayed.
        Similarly for config path: it must be in *.json format in correct directory.

        :param self: Refer to the object itself
        :return: None
        :doc-author: baobab soluciones
        """
        with st.sidebar:
            st.header("Configuration")

            # Select folder
            st.text_input(
                "Folder",
                value=self.config["dir_path"],
                max_chars=None,
                key="input_dir_path",
                disabled=True,
            )
            if not os.path.isdir(self.config["dir_path"]):
                st.error("Please enter a valid folder path")

            # Select config path
            st.text_input(
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
            self.config["title"] = st.text_input(
                "Title",
                value=self.config.get("title", self._APP_CONFIG["title"]),
                max_chars=None,
                key="title",
            )

            # Select number of columns and rows
            self.config["n_rows"] = st.number_input(
                "Row Levels",
                key="config_rows",
                min_value=1,
                max_value=100,
                value=self.config.get("n_rows", 1),
            )
            for row in range(1, self.config["n_rows"] + 1):
                self.config[f"n_cols_{row}"] = st.number_input(
                    f"Column Levels row {row}",
                    key=f"n_cols_{row}",
                    min_value=1,
                    max_value=2,
                    value=self.config.get(f"n_cols_{row}", 1),
                )
            for row in range(1, self.config["n_rows"] + 1):
                for col in range(1, self.config[f"n_cols_{row}"] + 1):
                    try:
                        path_row_col = st.session_state[f"file_{row}_{col}"]
                    except:
                        path_row_col = self.config["dict_layout"].get(
                            f"file_{row}_{col}", None
                        )
                        if path_row_col == None:
                            continue
                    key_row_col = f"file_{row}_{col}"
                    if path_row_col != None:
                        if (
                            path_row_col.endswith(".html")
                            or path_row_col.endswith(".jpg")
                            or path_row_col.endswith(".png")
                            or path_row_col.endswith(".jpeg")
                        ):

                            def _validate_number_text_input(text):
                                if text == "None" or text == "" or text == None:
                                    return None
                                elif text.isdecimal():
                                    return int(text)
                                else:
                                    return "Error"

                            number_w = st.text_input(
                                f"Width: {os.path.basename(path_row_col)}",
                                value=self.config.get(f"width_{key_row_col}", None),
                                key=f"width_{key_row_col}",
                            )
                            number_w = _validate_number_text_input(number_w)
                            if number_w == "Error":
                                st.error("Input must be number")
                            else:
                                self.config[f"width_{key_row_col}"] = number_w
                            if path_row_col.endswith(".html"):
                                number_h = st.text_input(
                                    f"Height: {os.path.basename(path_row_col)}",
                                    value=self.config.get(f"height_{key_row_col}", 500),
                                    key=f"height_{key_row_col}",
                                )
                                number_h = _validate_number_text_input(number_h)
                                if number_h == "Error":
                                    st.error("Input must be number")
                                else:
                                    self.config[f"height_{key_row_col}"] = number_h

            # Change to editable
            self.config["editable"] = st.checkbox(
                "Editable",
                key="editable_checkbox",
                value=bool(self.editable),
            )
            # Save config
            if os.path.isdir(self.config["dir_path"]) and (
                os.path.basename(self.config_path).endswith(".json")
                and os.path.exists(os.path.dirname(self.config_path))
            ):
                st.button(
                    "Apply & Save Configuration",
                    key="save_config",
                    on_click=self._save_config,
                )

    def _render_tree_folder(self):
        """
        The _render_tree_folder function is a helper function that renders the folder tree of the directory path specified in config.
        It uses DisplayablePath.make_tree to create a list of paths, and then iterates through them to display each one.

        :param self: Bind the method to an object
        :return: None
        :doc-author: baobab soluciones
        """
        paths = DisplayablePath.make_tree(
            Path(self.config["dir_path"]), criteria=lambda p: p.is_dir()
        )
        displayable_path = ""
        for path in paths:
            displayable_path = displayable_path + "  \n" + path.displayable()
        st.info(displayable_path)

    def _render_dropdown(self, i_row: int, i_col: int):
        """
        The _render_dropdown function is a helper function that renders the dropdown menu for selecting files.

        :param self: Access the attributes and methods of the class
        :param int i_row: Specify the row number of the file to be rendered
        :param int i_col: Specify the column number of the file to be rendered
        :return: None
        :doc-author: baobab soluciones
        """
        if self.editable:
            # Select folder
            self._element_selector(
                folder_path=self.config["dir_path"],
                element_type="folder",
                key=f"folder_{i_row}_{i_col}",
            )

            # Select file
            self._element_selector(
                folder_path=st.session_state.get(
                    f"folder_{i_row}_{i_col}", self.config["dir_path"]
                ),
                element_type="file",
                key=f"file_{i_row}_{i_col}",
            )

            # Render file content
            self._render_file_content(
                path_selected=st.session_state.get(f"file_{i_row}_{i_col}", ""),
                key=f"file_{i_row}_{i_col}",
            )

        else:
            # Render file content
            self._render_file_content(
                path_selected=self.config["dict_layout"].get(
                    f"file_{i_row}_{i_col}", ""
                ),
                key=f"file_{i_row}_{i_col}",
            )

    def _element_selector(
        self,
        folder_path: str,
        element_type: str = "file",
        key: str = None,
    ):
        """
        The _element_selector function is a helper function that allows the user to select an element from a folder.
        The function takes in three arguments:
            - self: The class instance of the Streamlit app. This is required for all functions within this class, as it allows us to access other functions and variables within the same class.
            - folder_path (str): The path of the folder where we want to select an element from.
                For example, if we want to select a file from our data directory, then we would pass in `self._data_dir` as our argument for `folder_path`.


        :param self: Bind the method to an object
        :param str folder_path: Specify the path of the folder to be searched
        :param str element_type: Specify whether the function should return a list of files or folders
        :param str key: Identify the selectbox in the config file
        :return: None
        :doc-author: baobab soluciones
        """
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
            [i for i, s in enumerate(paths) if self.config["dict_layout"][key] in s]
            if key in self.config["dict_layout"].keys()
            else 0
        )
        if default_index != 0:
            if len(default_index) == 0:
                default_index = 0
            else:
                default_index = default_index[0]
        st.selectbox(
            f"Select a {element_type}",
            paths,
            placeholder="Choose an option",
            index=default_index,
            format_func=lambda x: (
                x.replace(folder_path, "")
                if x != self.config["dir_path"]
                else os.path.basename(x)
            ),
            key=key,
        )

    @classmethod
    def _save_dataframe(cls, edited_df, path_selected, dict_of_tabs: dict = None):
        if path_selected.endswith(".csv"):
            edited_df.to_csv(path_selected, index=False)
        elif path_selected.endswith(".xlsx"):
            with pd.ExcelWriter(path_selected) as writer:
                for sheet in dict_of_tabs.keys():
                    dict_of_tabs[sheet]["df"].to_excel(
                        writer, sheet_name=sheet, index=False
                    )
        elif path_selected.endswith(".json"):
            write_json(edited_df, path_selected)

    @classmethod
    def _read_plot_from_html(cls, path_selected):
        encodings_to_try = ["utf-8", "latin-1", "cp1252"]

        for encoding in encodings_to_try:
            try:
                with open(path_selected, encoding=encoding) as f:
                    html = f.read()

                # Read
                call_arg_str = re.findall(
                    r"Plotly\.newPlot\((.*)\)", html[-(2**16) :]
                )[0]
                call_args = json.loads(f"[{call_arg_str}]")
                plotly_json = {"data": call_args[1], "layout": call_args[2]}
                return plotly.io.from_json(json.dumps(plotly_json))

            except:
                pass

    def _render_file_content(self, path_selected: str, key: str):
        """
        The _render_file_content function is a helper function that renders the content of a file in the Streamlit app.
        It takes as input:
            - path_selected: The path to the selected file. It can be None, an empty string or Select a file. In these cases, nothing is rendered.
            - If it ends with .csv, then it renders its content as a dataframe using st.dataframe().
            - If it ends with .png or .jpg or .jpeg, then it renders its content as an image using st.image().
            - If it ends with .html.
            - If it ends with .xlsx, then it renders its content as a dataframe using st.dataframe().

        :param self: Represent the instance of the class
        :param path_selected: str: Specify the path of the file to be rendered
        :return: None
        :doc-author: baobab soluciones
        """
        if (
            path_selected is None
            or path_selected == ""
            or path_selected == "Select a file"
        ):
            pass
        elif path_selected.endswith(".csv"):
            df = pd.read_csv(path_selected)
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                num_rows="dynamic",
                key=f"{path_selected}_editor_{key}",
            )
            if self.editable:
                st.button(
                    "Save",
                    on_click=self._save_dataframe,
                    args=(edited_df, path_selected),
                    key=f"{path_selected}_csv_{key}",
                )

        elif (
            path_selected.endswith(".png")
            or path_selected.endswith(".jpg")
            or path_selected.endswith(".jpeg")
        ):
            image = Image.open(path_selected)
            with st.spinner("Wait for it..."):
                st.image(image, width=self.config.get(f"width_{key}", None))

        elif path_selected.endswith(".html"):
            with st.spinner("Wait for it..."):
                try:
                    fig = self._read_plot_from_html(path_selected)
                    fig.update_layout(
                        height=self.config.get(f"height_{key}", 500),
                        width=self.config.get(f"width_{key}", None),
                    )
                    st.plotly_chart(
                        fig,
                        use_container_width=(
                            True
                            if self.config.get(f"width_{key}", None) is None
                            else False
                        ),
                        theme=None,
                    )
                except:
                    st.warning(
                        "The rendering of the HTML file failed. Please, notify mango@baobabsoluciones.es"
                    )

                def _open_html():
                    webbrowser.open_new_tab(path_selected)

                st.button(
                    "Open",
                    on_click=_open_html,
                    key=f"{path_selected}_html_{uuid.uuid4()}",
                )

        elif path_selected.endswith(".xlsx"):
            with st.spinner("Wait for it..."):
                excel_file = pd.ExcelFile(path_selected)
                sheets = excel_file.sheet_names
                list_of_tabs = st.tabs(sheets)
                try:
                    dict_of_tabs = {
                        sheets[i]: {
                            "tab": tab,
                            "df": pd.read_excel(excel_file, sheet_name=i),
                        }
                        for i, tab in enumerate(list_of_tabs)
                    }
                except Exception as e:
                    st.warning(f"The rendering of the Excel file failed. Error: {e}.")

                for key_tab, tab in dict_of_tabs.items():
                    with tab["tab"]:
                        df = tab["df"]
                        edited_df = st.data_editor(
                            df,
                            use_container_width=True,
                            num_rows="dynamic",
                            key=f"{key_tab}_{path_selected}_{key}",
                        )
                        tab["df"] = edited_df
                        if self.editable:
                            st.button(
                                "Save",
                                on_click=self._save_dataframe,
                                args=(edited_df, path_selected, dict_of_tabs),
                                key=f"{tab}_{path_selected}_xlsx_{key}",
                            )

        elif path_selected.endswith(".json"):
            with st.spinner("Wait for it..."):
                with open(path_selected, "r") as f:
                    data = json.load(f)
                    st.checkbox("As table", value=False, key=f"json_df_{key}")
                    if st.session_state[f"json_df_{key}"]:
                        sheets = list(data.keys())
                        list_of_tabs = st.tabs(sheets)
                        try:
                            dict_of_tabs = {
                                sheets[i]: {
                                    "tab": tab,
                                    "df": pd.DataFrame.from_dict(data[sheets[i]]),
                                }
                                for i, tab in enumerate(list_of_tabs)
                            }
                            for key_tab, tab in dict_of_tabs.items():
                                with tab["tab"]:
                                    edited_df = st.data_editor(
                                        tab["df"],
                                        use_container_width=True,
                                        key=f"{key_tab}_{path_selected}_{key}",
                                        num_rows="dynamic",
                                    )
                                    dict_edited = Table.from_pandas(
                                        edited_df
                                    ).replace_nan()
                                    data[key_tab] = dict_edited
                                    if self.editable:
                                        st.button(
                                            "Save",
                                            on_click=self._save_dataframe,
                                            args=(data, path_selected),
                                            key=f"{tab['tab']}_json_{path_selected}_{key}",
                                        )
                        except Exception as e:
                            st.warning(
                                f"The rendering of the JSON as file failed. Error: {e}"
                            )
                    else:
                        st.json(data)

        elif path_selected.endswith(".md"):
            with st.spinner("Wait for it..."):
                text_md = Path(path_selected).read_text()
                st.markdown(text_md, unsafe_allow_html=True)

        else:
            st.warning(
                "This file format is not supported yet. Please try another file."
            )

    def _render_body_content(self):
        """
        The _render_body_content function is responsible for rendering the body content of the widget.

        :param self: Refer to the object itself
        :return: None
        :doc-author: baobab soluciones
        """
        n_rows = self.config.get("n_rows", 1)

        for i_row in range(1, n_rows + 1):
            st.markdown("---")
            if self.config.get(f"n_cols_{i_row}", 1) == 1:
                self._render_dropdown(i_row=i_row, i_col=1)

            elif self.config.get(f"n_cols_{i_row}", 1) == 2:
                # Create columns display
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    self._render_dropdown(i_row=i_row, i_col=1)

                with col2_2:
                    self._render_dropdown(i_row=i_row, i_col=2)

    def _save_config(self):
        """
        The _save_config function is called when the user clicks on the &quot;Save&quot; button in the sidebar.
        It updates self.config with all of the new values that were entered into st.sidebar, and then saves
        the updated config to a json file.

        :param self: Refer to the object itself
        :return: The new configuration
        :doc-author: baobab soluciones
        """
        # Get max of config row in dict_layout
        for row in range(1, 101):
            col = self.config.get(f"n_cols_{row}", 1)
            for i_col in range(1, col + 1):
                try:
                    if st.session_state[f"folder_{row}_{i_col}"] != None:
                        self.config["dict_layout"][f"folder_{row}_{i_col}"] = (
                            st.session_state[f"folder_{row}_{i_col}"]
                        )
                except:
                    if (
                        self.config["dict_layout"].get(f"folder_{row}_{i_col}", None)
                        != None
                    ):
                        self.config["dict_layout"].pop(f"folder_{row}_{i_col}")
                try:
                    if st.session_state[f"file_{row}_{i_col}"] != None:
                        self.config["dict_layout"][f"file_{row}_{i_col}"] = (
                            st.session_state[f"file_{row}_{i_col}"]
                        )
                except:
                    if (
                        self.config["dict_layout"].get(f"file_{row}_{i_col}", None)
                        != None
                    ):
                        self.config["dict_layout"].pop(f"file_{row}_{i_col}")

        with open(self.config_path, "w") as f:
            json.dump(self.config, f, sort_keys=True, indent=4)

    def run(self):
        """
        The run function is the main entry point for the Streamlit app.
        It renders all of the components in order, and then returns a dictionary
        of values that can be used by other functions.

        :param self: Represent the instance of the class
        :return: None
        :doc-author: baobab soluciones
        """
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
    parser.add_argument("--editable", type=int, default=None, choices=[0, 1, -1])
    args = parser.parse_args()
    path = args.path
    config_path = args.config_path
    editable = None if args.editable == -1 else args.editable

    # Run app
    app = FileExplorerApp(path=path, editable=editable, config_path=config_path)
    app.run()
