import click
import traffik
import traffik.config as config
from traffik.graph_dataset import build_graph


def validate_cityname(ctx, param, value):
    if not value.lower() in config.CITIES:
        raise click.BadParameter("Should be a valid city dataset")
    return value


@click.group(help="Command line interface for the traffik Python package.")
@click.version_option(version=traffik.__version__, message="%(version)s")
@click.pass_context
def cli(ctx):
    """Execute the main mercantile command"""
    ctx.obj = {}


@cli.command()
@click.option(
    "--city", callback=validate_cityname, help="The city dataset to be processed."
)
def process(city):
    build_graph(city)
