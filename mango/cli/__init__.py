import os

import click


@click.group(name="mango", help="Commands in the mango cli")
def cli():
    pass


@cli.command(name="dashboard", help="Creates the data folder explorer dashboard")
@click.option("--path", "-p", default=os.getcwd(), help="Path to the folder to explore")
@click.option(
    "--editable",
    "-e",
    help="Enable editing of dashboard structure",
    default=True,
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
    os.system(
        rf'streamlit run mango/dashboards/file_explorer.py -- --config_path "{path}" --editable {editable} --config_path "{config_path}"'
    )
