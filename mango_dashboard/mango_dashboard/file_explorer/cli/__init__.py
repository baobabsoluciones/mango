import os

import click

cli_path = os.path.dirname(os.path.abspath(__file__))


@click.group(name="mango_file_explorer", help="Commands in the mango_file_explorer cli")
def cli():
    pass


# Import subcommands to register them (needs to be after cli to avoid circular imports)
from mango_dashboard.file_explorer.cli.dashboard import dashboard

cli.add_command(dashboard)
