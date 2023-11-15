import os
import pathlib

import dash
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output

# Config
_APP_CONFIG = {
    "title": "Mango Explorer App",
    "dir_path": os.getcwd(),
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


app.layout = html.Div(
    id="big-app-container",
    children=[
        build_banner(),
        html.Div(
            id="app-container",
            children=[
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


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
