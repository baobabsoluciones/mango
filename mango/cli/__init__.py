import os

import click

cli_path = os.path.dirname(os.path.abspath(__file__))


@click.group(name="mango", help="Commands in the mango cli")
def cli():
    pass


@cli.command(name="dashboard", help="Creates the data folder explorer dashboard")
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
def dashboard(path, editable, config_path):
    # Python Run os command
    relative_path = "../dashboards/file_explorer_app.py"

    absolute_path = os.path.join(cli_path, relative_path)

    os.system(
        rf'streamlit run {absolute_path} -- --path "{path}" --editable {editable} --config_path "{config_path}"'
    )
