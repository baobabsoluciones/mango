from mango_time_series.cli import cli
import sys
import os
import click


@cli.group(name="dashboard", help="Commands in the dashboard")
def dashboard():
    pass


@dashboard.command(name="time_series", help="Creates the time series dashboard")
@click.option(
    "--experimental_features",
    "-e",
    help="Enable experimental features",
    default=-1,
    type=int,
)
def time_series(experimental_features):
    # Python Run os command
    path_to_app = os.path.join(
        os.path.dirname(__file__), "..", "dashboards", "time_series_app.py"
    )
    if experimental_features == 1:
        os.environ["ENABLE_EXPERIMENTAL_FEATURES"] = "1"
    else:
        os.environ["ENABLE_EXPERIMENTAL_FEATURES"] = "0"

    python_path = sys.executable
    streamlit_path = os.path.join(os.path.dirname(python_path), "streamlit")
    os.system(f'{streamlit_path} run "{path_to_app}"')
