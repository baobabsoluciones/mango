import os
import subprocess
import sys
from pathlib import Path

import click

from mango_dashboard.time_series.cli import cli


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

    os.environ["TS_DASHBOARD_PROJECT_NAME"] = project_name
    os.environ["TS_DASHBOARD_LOGO_URL"] = logo_url
    os.environ["TS_DASHBOARD_EXPERIMENTAL_FEATURES"] = experimental_features

    path_to_app = (
        Path(__file__).resolve().parent.parent / "dashboards" / "time_series_app.py"
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(path_to_app),
            "--theme.primaryColor=#3d9df3",
        ],
        check=False,
    )
