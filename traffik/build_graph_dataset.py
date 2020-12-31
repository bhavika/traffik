import os
import numpy as np
import h5py
from tqdm import tqdm
import traffik.config as config
from traffik.logger import logger
from typing import List
import torch


def setup(city: str):
    logger.info("Setting up processed directory for", city=city)

    output_path = os.path.join(os.getenv("DATA_DIR"), config.PROCESSED_DIR)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(os.path.join(output_path, city)):
        os.makedirs(os.path.join(output_path, city))

    node_handle = os.path.join(output_path, city, f"{city}_nodes_5.npy")
    edge_handle = os.path.join(output_path, city, f"{city}_edges_5.npy")

    return {"output_path": output_path, "node": node_handle, "edge": edge_handle}


def build_graph(city: str):
    locations = setup(city)
    nodes = np.load(locations["node"])

    for mode in [config.TRAINING_DIR, config.VALIDATION_DIR, config.TESTING_DIR]:
        logger.debug("Start building graph dataset for", city=city, mode=mode)
        raw_data = os.path.join(os.getenv("DATA_DIR"), city, mode)

        hf_handle = h5py.File(
            os.path.join(locations["output_path"], city, f"{city}_{mode}_5.h5"), "w"
        )

        for f in os.listdir(raw_data):
            reader = h5py.File(os.path.join(raw_data, f), "r")
            data = reader[list(reader.keys())[0]]  # key is 'array'
            if mode == config.TESTING_DIR:
                graph_data = np.array(data)[:, :, nodes[:, 0], nodes[:, 1], :]
            else:
                graph_data = np.array(data)[:, nodes[:, 0], nodes[:, 1], :]
            reader.close()
            hf_handle.create_dataset(f, data=graph_data, compression="lzf")
        hf_handle.close()


def get_road_network(source_dir: str, image_size: List, testing: bool, data_type: str):
    grid = np.zeros(image_size)
    files = os.listdir(source_dir)
    total_files = len(files)

    logger.debug(
        "Start processing road network for ",
        source_dir=source_dir,
        testing=testing,
        data_type=data_type,
    )

    for f in tqdm(files):
        reader = h5py.File(os.path.join(source_dir, f), "r")
        data = reader[list(reader.keys())[0]]

        if testing:
            if data_type == config.AVG_TOTAL_VOLUME:
                data = np.mean(data, axis=0)
            else:
                data = np.max(data, axis=0)
        elif data_type == config.MAX_VOLUME:
            d_max = np.array(data)[:, :, :, [0, 2, 4, 6]].max(3).max(0)
            grid = np.maximum(grid, d_max)
        elif data_type == config.AVG_TOTAL_VOLUME:
            d_max = np.array(data)[:, :, :, [0, 2, 4, 6]].mean(3).mean(0)
            grid += d_max / total_files
        else:
            d_max = np.array(data).reshape(-1, 495, 436).sum(0)
            grid += d_max
        logger.info(
            f"[get_road_network] Current size {(grid > 0.1).sum() / (495 * 436)} of image"
        )
    return grid


def process_grid(
    city: str, image_size: List, mode: str, data_type: str, save: bool = True
) -> np.array:
    grid_handle = os.path.join(
        os.getenv("DATA_DIR"),
        config.INTERMEDIATE_DIR,
        f"{city}_{mode}_roads_{data_type}.npy",
    )

    source_dir = os.path.join(os.getenv("DATA_DIR"), city, mode)

    if os.path.isfile(grid_handle):
        logger.info("[process_grid] Reading existing grid file:", file=grid_handle)
        grid = np.load(grid_handle)
    else:
        logger.info(
            "[process_grid] Creating new road network grid file for",
            city=city,
            mode=mode,
            data_type=data_type,
        )
        grid = get_road_network(source_dir, image_size, mode == "testing", data_type)
        if save:
            logger.info(
                "[process_grid] Saving processed grid for",
                city=city,
                data_type=data_type,
                destination=grid_handle,
            )
            np.save(grid_handle, grid)
    return grid


def combine_grids(
    city, train_grid, validation_grid, test_grid, data_type, save=True
) -> np.array:
    """
    Combines the training, validation and test grids into one grid.
    :return:
    """
    grid = np.maximum(train_grid, validation_grid)
    grid = np.maximum(grid, test_grid)

    if save:
        fname = f"{city}_roads_{data_type}.npy"
        logger.info(
            "[combine_grids] Saving combined grid for ",
            city=city,
            data_type=data_type,
            destination=fname,
        )
        np.save(
            os.path.join(os.getenv("DATA_DIR"), config.INTERMEDIATE_DIR, fname), grid
        )
    return grid


def build_static_grid(city: str, image_size: List, data_type: str):
    """
    Calculates overall max volume for each pixel across all channels.
    """

    if not os.path.exists(os.path.join(os.getenv("DATA_DIR"), config.INTERMEDIATE_DIR)):
        os.makedirs(os.path.join(os.getenv("DATA_DIR"), config.INTERMEDIATE_DIR))

    logger.info(
        "[build_static_grid] Calculating and saving the road network for training data."
    )

    train_grid = process_grid(
        city, image_size, config.TRAINING_DIR, data_type, save=True
    )
    road_percentage = (train_grid != 0).sum() / (image_size[0] * image_size[1])
    logger.info(
        "[build_static_grid] Training images show the road network covers percentage of image",
        cover=road_percentage,
    )

    validation_grid = process_grid(
        city, image_size, config.VALIDATION_DIR, data_type, save=True
    )
    road_percentage = (validation_grid != 0).sum() / (image_size[0] * image_size[1])
    logger.info(
        "[build_static_grid] Validation images show the road network covers percentage of image",
        cover=road_percentage,
    )

    test_grid = process_grid(city, image_size, config.TESTING_DIR, data_type, save=True)
    road_percentage = (test_grid != 0).sum() / (image_size[0] * image_size[1])
    logger.info(
        "[build_static_grid] Test images show the road network covers percentage of image",
        cover=road_percentage,
    )

    logger.info(
        "[build_static_grid] Combining train, validation and test grids into one."
    )
    combined_grid = combine_grids(
        city, train_grid, validation_grid, test_grid, data_type
    )
    road_percentage = (combined_grid != 0).sum() / (image_size[0] * image_size[1])
    logger.info(
        "[build_static_grid] Combined images show a road network covers percentage of image",
        cover=road_percentage,
    )

    assert (
        np.subtract((test_grid != 0) * 1, (combined_grid != 0) * 1) == 1
    ).sum() == 0, "Seems like there is activity in image areas in the test set that isn't in the train set."


def get_edges_and_unconnected_nodes(node_coordinates):
    edge_start = []
    edge_to = []
    unconnected_nodes = []

    for idx in range(len(node_coordinates)):
        i, j = node_coordinates[idx]
        all_nodes = np.where(
            (node_coordinates[:, 0] >= (i - 1))
            & (node_coordinates[:, 0] <= (i + 1))
            & (node_coordinates[:, 1] >= (j - 1))
            & (node_coordinates[:, 1] <= (j + 1))
        )
        adj_nodes = all_nodes[0][all_nodes[0] != idx]
        if len(adj_nodes) > 0:
            start_adj = np.ones(len(adj_nodes)) * idx
            end_adj = adj_nodes
            edge_start = np.append(edge_start, start_adj)
            edge_to = np.append(edge_to, end_adj)
        else:
            unconnected_nodes = np.append(unconnected_nodes, idx)
    edge_start = edge_start.astype(int)
    edge_to = edge_to.astype(int)
    if len(unconnected_nodes) > 0:
        unconnected_nodes = unconnected_nodes.astype(int)
    edge_idx = [edge_start, edge_to]
    return edge_idx, unconnected_nodes


def build_nodes_edges(city, source_dir, mode, data_type, testval, volume_filter):
    fname = os.path.join(source_dir, f"{city}_{mode}_roads_{data_type}.npy")
    logger.info("Loading file", file=fname)
    max_volume = np.load(fname)

    filtered_grid = max_volume > volume_filter
    graph_coverage = filtered_grid.sum() / (495 * 436)
    logger.info("Ratio of image covered by road network", ratio=graph_coverage)

    all_node_coordinates = np.array(np.where(filtered_grid)).transpose()
    edge_idx, uncon_nodes = get_edges_and_unconnected_nodes(all_node_coordinates)

    logger.info("Number of nodes", nodes=len(all_node_coordinates))
    logger.info("Number of edges", edges=len(edge_idx[0]))
    logger.info("Number of unconnected nodes", unconnected_nodes=len(uncon_nodes))

    logger.info(f"Rerunning with only connected nodes")
    logger.info(f"Must be faster ways of doing this.....")
    connected_nodes = all_node_coordinates[np.unique(edge_idx[0])]
    edge_idx, uncon_nodes = get_edges_and_unconnected_nodes(connected_nodes)
    logger.info("Number of nodes", nodes=len(all_node_coordinates))
    logger.info("Number of edges", edges=len(edge_idx[0]))
    logger.info("Number of unconnected nodes", unconnected_nodes=len(uncon_nodes))

    logger.info(f"Saving...")
    if testval is None:
        node_file = os.path.join(
            os.getenv("DATA_DIR"),
            config.INTERMEDIATE_DIR,
            f"{city}_nodes_{volume_filter}.npy",
        )
        edge_file = os.path.join(
            os.getenv("DATA_DIR"),
            config.INTERMEDIATE_DIR,
            f"{city}_edges_{volume_filter}.npy",
        )
        unc_node_file = os.path.join(
            os.getenv("DATA_DIR"),
            config.INTERMEDIATE_DIR,
            f"{city}_unc_nodes_{volume_filter}.npy",
        )
        mask_file = os.path.join(
            os.getenv("DATA_DIR"),
            config.INTERMEDIATE_DIR,
            f"{city}_Mask_{volume_filter}.pt",
        )
    else:
        node_file = os.path.join(
            os.getenv("DATA_DIR"),
            config.INTERMEDIATE_DIR,
            f"{city}_nodes_{volume_filter}_{testval}.npy",
        )
        edge_file = os.path.join(
            os.getenv("DATA_DIR"),
            config.INTERMEDIATE_DIR,
            f"{city}_edges_{volume_filter}_{testval}.npy",
        )
        unc_node_file = os.path.join(
            os.getenv("DATA_DIR"),
            config.INTERMEDIATE_DIR,
            f"{city}_unc_nodes_{volume_filter}_{testval}.npy",
        )
        mask_file = os.path.join(
            os.getenv("DATA_DIR"),
            config.INTERMEDIATE_DIR,
            f"{city}_Mask_{volume_filter}.pt",
        )

    logger.info("Saving nodes to file", file=node_file)
    np.save(node_file, connected_nodes)
    logger.info("Saving edges to file", file=edge_file)
    np.save(edge_file, edge_idx)
    logger.info("Saving unconnected nodes to file", file=unc_node_file)
    np.save(unc_node_file, uncon_nodes)
    mask = torch.zeros([495, 436]).byte()
    mask[connected_nodes[:, 0], connected_nodes[:, 1]] = 1
    logger.info("Saving mask to file", file=mask_file)
    torch.save(mask, mask_file)
