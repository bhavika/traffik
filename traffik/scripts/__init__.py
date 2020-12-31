import click
import traffik
import traffik.config as config
from traffik.build_graph_dataset import (
    build_graph,
    build_static_grid,
    build_nodes_edges,
)
from dotenv import load_dotenv


def validate_cityname(
    ctx: click.core.Context, param: click.core.Parameter, value: str
) -> str:
    if value.lower() in config.CITIES:
        return value
    else:
        raise click.BadParameter("Should be a valid city dataset")


@click.group(help="Command line interface for the traffik Python package.")
@click.version_option(version=traffik.__version__, message="%(version)s")
@click.pass_context
def cli(ctx):
    """Execute the main traffik command"""
    load_dotenv(verbose=False)
    ctx.obj = {}


@cli.command("process")
@click.option(
    "--city", callback=validate_cityname, help="The city dataset to be processed."
)
@click.option(
    "--data-type",
    type=click.Choice(
        [config.MAX_VOLUME, config.AVG_TOTAL_VOLUME], case_sensitive=False
    ),
)
@click.option(
    "--mode",
    type=click.Choice(
        [config.TRAINING_DIR, config.VALIDATION_DIR, config.TESTING_DIR],
        case_sensitive=False,
    ),
    help="One of training, validation or testing"
)
@click.option("--volume-filter", type=int)
def process(city, data_type, mode, volume_filter):
    image_size = [495, 436]
    build_static_grid(city, image_size, data_type)
    build_nodes_edges(city, mode, data_type, None, volume_filter)
    build_graph(city)
