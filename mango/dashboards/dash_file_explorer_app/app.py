import os
import pathlib
from pathlib import Path
from typing import List, Dict

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Config
_APP_CONFIG = {
    "title": "Mango Explorer App",
    "dir_path": r"G:\Unidades compartidas\mango\desarrollo\datos\file_explorer_folder",
    "dict_layout": {},
}

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = _APP_CONFIG["title"]
server = app.server
app.config["suppress_callback_exceptions"] = True

APP_PATH = str(pathlib.Path(__file__).parent.resolve())


def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(
                id="banner-text",
                children=[
                    html.H5(_APP_CONFIG["title"]),
                ],
            ),
            html.Div(
                id="banner-logo",
                children=[
                    html.Button(
                        id="learn-more-button", children="LEARN MORE", n_clicks=0
                    ),
                    html.A(
                        html.Img(id="logo", src=app.get_asset_url("logo.png")),
                        href="https://baobabsoluciones.es/",
                        target="_blank",
                    ),
                ],
            ),
        ],
    )


def generate_modal():
    return html.Div(
        id="modal",
        className="modal",
        children=(
            html.Div(
                id="modal-container",
                className="modal-container",
                children=[
                    html.Div(
                        className="close-container",
                        children=html.Button(
                            "Close",
                            id="markdown_close",
                            n_clicks=0,
                            className="closeButton",
                        ),
                    ),
                    html.Div(
                        className="modal-text",
                        children=dcc.Markdown(
                            children=(
                                """
                                ###### About this app
                                This is a file explorer to visualize files and folders of any directory.
                                
                                Made by [baobab soluciones](https://baobabsoluciones.es/).
                                """
                            )
                        ),
                    ),
                ],
            )
        ),
    )


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


def build_tree_folder():
    """
    The _render_tree_folder function is a helper function that renders the folder tree of the directory path specified in config.
    It uses DisplayablePath.make_tree to create a list of paths, and then iterates through them to display each one.

    :param self: Bind the method to an object
    :return: None
    :doc-author: baobab soluciones
    """
    paths = DisplayablePath.make_tree(
        Path(_APP_CONFIG["dir_path"]), criteria=lambda p: p.is_dir()
    )
    displayable_path = ""
    for path in paths:
        displayable_path = displayable_path + "  \n" + path.displayable()
    return displayable_path


def _get_options_by_path(
    path: str, element_type: str = "folder"
) -> List[Dict[str, str]]:
    paths = []
    for root, dirs, files in os.walk(path):
        if element_type == "file":
            for element in files:
                path_i = os.path.join(root, element)
                paths.append(path_i)
        elif element_type == "folder":
            for element in dirs:
                path_i = os.path.join(root, element)
                paths.append(path_i)
        else:
            raise ValueError(
                "element_type must be 'file' or 'folder', but got {}".format(
                    element_type
                )
            )
    return list(
        {
            "label": path_i.replace(path, ""),
            "value": path_i,
        }
        for path_i in paths
    )


app.layout = html.Div(
    id="big-app-container",
    children=[
        build_banner(),
        html.Div(
            id="app-container",
            children=[
                dcc.Markdown(children=(build_tree_folder())),
                html.Div(
                    id="metric-select-menu",
                    children=[
                        html.Label(
                            id="metric-select-title",
                            children="Selecciona la carpeta",
                        ),
                        dcc.Dropdown(
                            id="dropdown_folder",
                            options=_get_options_by_path(
                                _APP_CONFIG["dir_path"], "folder"
                            ),
                            value=None,
                        ),
                    ],
                ),
                html.Div(
                    id="metric-select-menu",
                    children=[
                        html.Label(
                            id="metric-select-title",
                            children="Selecciona el fichero",
                        ),
                        dcc.Dropdown(
                            id="dropdown_file",
                            value=None,
                        ),
                    ],
                ),
                # Main app
                html.Div(
                    id="status-container",
                    children=[],
                ),
            ],
        ),
        generate_modal(),
    ],
)


@app.callback(
    Output("modal", "style"),
    [Input("learn-more-button", "n_clicks"), Input("markdown_close", "n_clicks")],
)
def update_click_output(button_click, close_click):
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "learn-more-button":
            return {"display": "block"}

    return {"display": "none"}


@app.callback(
    output=Output("dropdown_file", "options"),
    inputs=[
        Input(f"dropdown_folder", "value"),
    ],
)
def show_selector_file(path_folder):
    return _get_options_by_path(path_folder, "file")


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
