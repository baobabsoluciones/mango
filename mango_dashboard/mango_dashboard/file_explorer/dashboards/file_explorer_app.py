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
from mango.table import Table

from mango_dashboard.file_explorer.dashboards.file_explorer_handlers import (
    FileExplorerHandler,
    GCPFileExplorerHandler,
    LocalFileExplorerHandler,
)


class DisplayablePath(object):
    """A class for creating tree-like directory structure displays.

    This class provides functionality to create a visual tree representation
    of directory structures using ASCII characters. It supports hierarchical
    display with proper indentation and tree-like connectors.

    Attributes:
        display_filename_prefix_middle (str): Prefix for middle items in tree
        display_filename_prefix_last (str): Prefix for last items in tree
        display_parent_prefix_middle (str): Indentation for middle parent levels
        display_parent_prefix_last (str): Indentation for last parent levels
    """

    display_filename_prefix_middle = "├──"
    display_filename_prefix_last = "└──"
    display_parent_prefix_middle = "    "
    display_parent_prefix_last = "│   "

    def __init__(self, path, parent_path, is_last):
        """Initialize a DisplayablePath instance.

        :param path: The file or directory path to display
        :type path: str or Path
        :param parent_path: The parent DisplayablePath instance (None for root)
        :type parent_path: DisplayablePath or None
        :param is_last: Whether this is the last item in its parent's children
        :type is_last: bool
        """
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        """Get the display name for the path.

        Returns the name of the file or directory with a trailing slash
        for directories to distinguish them from files.

        :return: Display name with trailing slash for directories
        :rtype: str
        """
        if self.path.is_dir():
            return self.path.name + "/"
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        """Generate a tree structure of DisplayablePath objects.

        Creates a hierarchical tree structure starting from the root path.
        Recursively processes directories and applies filtering criteria.
        Children are sorted alphabetically (case-insensitive).

        :param root: Root directory path to start the tree from
        :type root: str or Path
        :param parent: Parent DisplayablePath instance (None for root)
        :type parent: DisplayablePath or None
        :param is_last: Whether this is the last item in its parent's children
        :type is_last: bool
        :param criteria: Function to filter paths (default: include all)
        :type criteria: callable or None
        :return: Generator yielding DisplayablePath objects in tree order
        :rtype: Generator[DisplayablePath]

        Example:
            >>> for path in DisplayablePath.make_tree('/some/directory'):
            ...     print(path.displayable())
            /some/directory/
            ├── subdir1/
            │   ├── file1.txt
            │   └── file2.txt
            └── subdir2/
        """
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
        """Default criteria function that includes all paths.

        :param path: Path to evaluate
        :type path: Path
        :return: Always True (include all paths)
        :rtype: bool
        """
        return True

    def displayable(self):
        """Generate the complete display string for this path.

        Creates a tree-like string representation with proper indentation
        and connectors based on the hierarchical position in the tree.

        :return: Formatted string representation of the path in tree format
        :rtype: str

        Example:
            >>> path = DisplayablePath('/some/dir/file.txt', parent, True)
            >>> print(path.displayable())
            └── file.txt
        """
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
    """A Streamlit-based file explorer application.

    This class provides a web-based interface for exploring and managing files
    and directories. It supports various file formats including CSV, Excel, JSON,
    images, HTML plots, and Markdown files. The application can work with both
    local file systems and Google Cloud Storage.

    Features:
        - Interactive file browsing with tree view
        - File content preview and editing
        - Support for multiple file formats
        - Configurable layout and display options
        - Integration with GCP storage
    """

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
        self,
        path: str = None,
        editable: bool = True,
        config_path: str = None,
        file_handler: FileExplorerHandler = None,
    ):
        """Initialize the FileExplorerApp instance.

        Sets up the Streamlit application with configuration, file handler,
        and initial settings. Loads configuration from file or uses defaults.
        Configures Streamlit page settings and validates paths.

        :param path: Default directory path for the file explorer
        :type path: str or None
        :param editable: Whether the interface should allow editing and configuration
        :type editable: bool
        :param config_path: Path to configuration JSON file
        :type config_path: str or None
        :param file_handler: File handler for accessing files (local or GCP)
        :type file_handler: FileExplorerHandler or None
        :raises FileNotFoundError: If config file path doesn't exist
        :raises json.JSONDecodeError: If config file contains invalid JSON

        Example:
            >>> app = FileExplorerApp(
            ...     path="/path/to/directory",
            ...     editable=True,
            ...     config_path="config.json"
            ... )
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

        # Set file handler
        self.file_handler = file_handler

        if path:
            if self.file_handler.path_exists(path):
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
        """Render the application header with logo, title, and header text.

        Creates a three-column layout displaying the application logo,
        title, and header text. Handles logo path resolution and applies
        custom CSS styling to hide certain UI elements.

        :return: None
        :rtype: None
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
        """Render the configuration sidebar with user input controls.

        Creates a sidebar with configuration options including folder selection,
        config path, title, layout settings (rows/columns), and file-specific
        settings like width/height for images and HTML files. Validates inputs
        and provides error messages for invalid configurations.

        :return: None
        :rtype: None
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
        """Render the directory tree structure using DisplayablePath.

        Creates a visual tree representation of the directory structure
        starting from the configured directory path. Only displays directories
        (not files) in the tree view.

        :return: None
        :rtype: None
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
        Render dropdown selectors and file content for a specific grid position.

        Creates folder and file selection dropdowns for the specified row and column
        position in the grid layout. Renders the selected file content using the
        appropriate handler based on file type.

        :param i_row: Row index in the grid layout (1-based)
        :type i_row: int
        :param i_col: Column index in the grid layout (1-based)
        :type i_col: int
        :return: None
        :rtype: None
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
        Create a selectbox for choosing files or folders from a directory.

        Generates a Streamlit selectbox populated with files or folders from the
        specified directory path. Handles default selection based on saved
        configuration and formats display names appropriately.

        :param folder_path: Path to the directory to browse
        :type folder_path: str
        :param element_type: Type of elements to display ('file' or 'folder')
        :type element_type: str
        :param key: Unique key for the selectbox widget
        :type key: str or None
        :return: None
        :rtype: None
        """
        paths = self.file_handler.get_file_or_folder_paths(
            path=folder_path, element_type=element_type
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

    def _save_dataframe(self, edited_df, path_selected, dict_of_tabs: dict = None):
        if path_selected.endswith(".csv"):
            edited_df.to_csv(path_selected, index=False)
        elif path_selected.endswith(".xlsx"):
            with pd.ExcelWriter(path_selected) as writer:
                for sheet in dict_of_tabs.keys():
                    dict_of_tabs[sheet]["df"].to_excel(
                        writer, sheet_name=sheet, index=False
                    )
        elif path_selected.endswith(".json"):
            self.file_handler.write_json_fe(path_selected, edited_df)

    def _read_plot_from_html(self, path_selected):
        encodings_to_try = ["utf-8", "latin-1", "cp1252"]

        for encoding in encodings_to_try:
            try:
                html = self.file_handler.read_html(path_selected, encoding=encoding)

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
        Render file content based on file type with appropriate handlers.

        Displays file content using the appropriate Streamlit component based on
        the file extension. Supports CSV, Excel, JSON, images, HTML plots,
        and Markdown files. Provides editing capabilities for data files and
        interactive features for HTML content.

        Supported file types:
            - CSV: Editable data table with save functionality
            - Excel: Multi-sheet data tables with editing
            - JSON: Raw JSON or tabular view with editing
            - Images (PNG, JPG, JPEG): Image display with configurable width
            - HTML: Plotly charts with interactive features
            - Markdown: Rendered markdown content

        :param path_selected: Path to the file to render
        :type path_selected: str
        :param key: Unique key for the file content widget
        :type key: str
        :return: None
        :rtype: None
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
            image = self.file_handler.read_img(path=path_selected)
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
                data = self.file_handler.read_json(path_selected)
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
                                dict_edited = Table.from_pandas(edited_df).replace_nan()
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
                text_md = self.file_handler.read_markdown(path_selected)
                st.markdown(text_md, unsafe_allow_html=True)

        else:
            st.warning(
                "This file format is not supported yet. Please try another file."
            )

    def _render_body_content(self):
        """
        Render the main body content with configurable grid layout.

        Creates a grid layout based on the configuration settings for rows and columns.
        Each grid cell contains file selection dropdowns and content rendering.
        Supports both single-column and two-column layouts per row.

        :return: None
        :rtype: None
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
        Save the current configuration to a JSON file.

        Collects all current configuration values from the Streamlit session state,
        updates the internal config dictionary, and saves it to the configured
        JSON file path. Handles layout configuration for all grid positions.

        :return: None
        :rtype: None
        :raises IOError: If unable to write to the configuration file
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
        Run the complete Streamlit file explorer application.

        Main entry point that orchestrates the rendering of all application
        components in the correct order: header, configuration sidebar,
        folder tree, and main content area. Handles error cases gracefully
        with appropriate user feedback.

        :return: None
        :rtype: None
        """
        # Render header
        self._render_header()

        if self.editable:
            # Render configuration
            self._render_configuration()

        if self.editable:
            with st.spinner("Wait for it..."):
                try:
                    # Render folder tree
                    self._render_tree_folder()
                except:
                    if "GCPFileExplorerHandler" in self.file_handler.__class__.__name__:
                        st.warning(
                            "GCPFileExplorerHandler does not support folder tree."
                        )
                    else:
                        st.warning("The rendering of the folder tree failed.")

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
    parser.add_argument("--gcp_credentials_path", type=str, default=None)
    args = parser.parse_args()
    path = args.path
    config_path = args.config_path
    editable = None if args.editable == -1 else args.editable
    gcp_path = args.gcp_credentials_path
    if gcp_path is not None:
        if not os.path.exists(gcp_path):
            raise ValueError("The GCP credentials path does not exist")
        else:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_path

    # Select the file handler
    if os.path.exists(path):
        file_handler = LocalFileExplorerHandler(path=path)
    else:
        if path.startswith("gs://"):
            file_handler = GCPFileExplorerHandler(
                path=path, gcp_credentials_path=gcp_path
            )
        else:
            raise ValueError("The path must be a valid local path or a GCP path")

    # Run app
    app = FileExplorerApp(
        path=path, editable=editable, config_path=config_path, file_handler=file_handler
    )
    app.run()
