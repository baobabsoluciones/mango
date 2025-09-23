import os

import click
from mango_dashboard.file_explorer.cli import cli


@cli.group(name="dashboard", help="Commands in the dashboard")
def dashboard():
    pass


@dashboard.command(
    name="file_explorer", help="Creates the data folder explorer dashboard"
)
@click.option("--path", "-p", default=os.getcwd(), help="Path to the folder to explore")
@click.option(
    "--editable",
    "-e",
    help="Enable editing of dashboard structure",
    default=-1,
    type=int,
)
@click.option(
    "--config_path",
    "-c",
    default=os.path.join(os.getcwd(), "config.json"),
    help="Path to the folder to explore",
)
def file_explorer(path, editable, config_path):
    # Python Run os command
    relative_path = "../dashboards/file_explorer_app.py"

    absolute_path = os.path.join(os.path.dirname(__file__), relative_path)

    os.system(
        rf'streamlit run {absolute_path} -- --path "{path}" --editable {editable} --config_path "{config_path}"'
    )
