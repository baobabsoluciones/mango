import click

@click.group(name="mango", help="Commands in the mango cli")
def cli():
    pass

@cli.command(name="dashboard", help="Creates the data folder explorer dashboard")
@click.option("--path", "-p", default=".", help="Path to the folder to explore")
@click.option("--editable", "-e", is_flag=True, help="Enable editing of dashboard structure")
def dashboard(path, editable):
    from mango.dashboards import FileExplorerApp

    app = FileExplorerApp(path)
    app.run()
