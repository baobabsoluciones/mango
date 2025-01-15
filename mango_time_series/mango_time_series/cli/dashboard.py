import os
import sys

import click

from mango_time_series.cli import cli


@cli.group(name="dashboard", help="Commands in the dashboard")
def dashboard():
    pass


@dashboard.command(name="time_series", help="Creates the time series dashboard")
@click.option(
    "--project_name",
    help="Name of the project to display in dashboard. Can be set in TS_DASHBOARD_PROJECT_NAME env variable",
    type=str,
    default=None,
)
@click.option(
    "--logo_url",
    help="Logo URL. Can be https:// or file:/// for local files. Can be set in TS_DASHBOARD_LOGO_URL env variable",
    type=str,
    default=None,
)
@click.option(
    "--experimental_features",
    "-e",
    help="Enable experimental features as a 0 or 1. Can be set as env in TS_DASHBOARD_EXPERIMENTAL_FEATURES",
    type=str,
    default=None,
)
def time_series(project_name, logo_url, experimental_features):
    # Python Run os command
    path_to_app = os.path.join(
        os.path.dirname(__file__), "..", "dashboards", "time_series_app.py"
    )
    if project_name is None:
        project_name = os.getenv("TS_DASHBOARD_PROJECT_NAME", "Project")

    if logo_url is None:
        logo_url = os.getenv("TS_DASHBOARD_LOGO_URL", "")

    if experimental_features is None:
        experimental_features = os.getenv("TS_DASHBOARD_EXPERIMENTAL_FEATURES", "0")
    assert experimental_features in [
        "0",
        "1",
    ], "TS_DASHBOARD_EXPERIMENTAL_FEATURES must be set to 0 or 1"

    # Setup env variables
    os.environ["TS_DASHBOARD_PROJECT_NAME"] = project_name
    os.environ["TS_DASHBOARD_LOGO_URL"] = logo_url
    os.environ["TS_DASHBOARD_EXPERIMENTAL_FEATURES"] = experimental_features

    # Make sure we use the correct python with dependencies installed
    python_path = sys.executable
    streamlit_path = os.path.join(os.path.dirname(python_path), "streamlit")
    os.system(f'{streamlit_path} run "{path_to_app}" --theme.primaryColor=#3d9df3')
