import click
import os
import traffik
from traffik.helpers import reproducibility
import traffik.config as config
from traffik.dataset import (
    build_graph,
    build_static_grid,
    build_nodes_edges,
    combine_grids,
)
from dotenv import load_dotenv
import wandb

wandb.init(project=os.getenv("WANDB_PROJECT"))
reproducibility()


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


@cli.command("prep")
@click.option(
    "--city", callback=validate_cityname, help="The city dataset to be processed."
)
@click.option(
    "--data-type",
    type=click.Choice(
        [config.MAX_VOLUME, config.AVG_TOTAL_VOLUME], case_sensitive=False
    ),
)
def prep(city, data_type):
    grids = []
    image_size = [495, 436]

    run = wandb.init(job_type="prep")

    artifact = wandb.Artifact(
        "gnn-dataset",
        type="dataset",
        description="Data from the prep stage",
        metadata={
            "solution": "traffic4cast2020-place4",
            "dataset": "road network grid",
        },
    )
    wandb.config.image_size = image_size
    wandb.config.city = city
    wandb.config.data_type = data_type

    for m in config.modes:
        grids.append(
            build_static_grid(city, image_size, m, data_type, artifact=artifact)
        )
    grid = combine_grids(city, grids[0], grids[1], grids[2], data_type, save=True)
    if isinstance(grid, str):
        artifact.add_file(grid, name="combined-grid")
    run.log_artifact(artifact)


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
        [config.TRAINING_DIR, config.VALIDATION_DIR, config.TESTING_DIR, "all"],
        case_sensitive=False,
    ),
    help="One of training, validation or testing",
)
@click.option("--volume-filter", type=int)
def process(city, data_type, mode, volume_filter):
    city_road_network = os.path.join(
        os.getenv("DATA_DIR"), config.INTERMEDIATE_DIR, f"{city}_roads_{data_type}.npy"
    )

    run = wandb.init(job_type="process")

    artifact = wandb.Artifact(
        "graph-dataset",
        type="dataset",
        description="Data from the process stage",
        metadata={
            "solution": "traffic4cast2020-place4",
            "dataset": "nodes, edges, mask files",
        },
    )

    if os.path.exists(city_road_network):
        build_nodes_edges(city, data_type, volume_filter, artifact)
    else:
        raise Exception(
            f"The {city_road_network} file has not been created yet. Run `traffik prep` first."
        )

    if mode == "all":
        [build_graph(city, m, artifact=artifact) for m in config.modes]
    else:
        build_graph(city, mode, artifact)

    run.log_artifact(artifact)
