import os

import click

cli_path = os.path.dirname(os.path.abspath(__file__))


@click.group(name="mango_time_series", help="Commands in the mango_time_series cli")
def cli():
    pass


# Import subcommands to register them (needs to be after cli to avoid circular imports)
from mango_time_series.cli.dashboard import dashboard

cli.add_command(dashboard)
