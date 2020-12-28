import os
import numpy as np
import h5py
from dotenv import load_dotenv
import traffik.config as config
from traffik.logger import logger


def setup(city: str):
    output_path = os.path.join(os.getenv("DATA_DIR"), config.PROCESSED_DIR)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(os.path.join(output_path, city)):
        os.makedirs(os.path.join(output_path, city))

    node_handle = os.path.join(output_path, city, f"{city}_nodes_5.npy")
    edge_handle = os.path.join(output_path, city, f"{city}_edges_5.npy")

    return {"output_path": output_path, "node": node_handle, "edge": edge_handle}


def build_graph(city: str):
    load_dotenv(verbose=False)

    locations = setup(city)
    nodes = np.load(locations["node"])

    for mode in [config.TRAINING_DIR, config.VALIDATION_DIR, config.TESTING_DIR]:
        logger.msg("Start building graph dataset for", city=city, mode=mode)

        raw_data = os.path.join(os.getenv("DATA_DIR"), city, mode)

        hf_handle = h5py.File(
            os.path.join(locations["output_path"], city, f"{city}_{mode}_5.h5"), "w"
        )

        for f in os.listdir(raw_data):
            reader = h5py.File(os.path.join(raw_data, f), "r")
            data = reader[list(reader.keys())[0]]
            if mode == config.TESTING_DIR:
                graph_data = np.array(data)[:, :, nodes[:, 0], nodes[:, 1], :]
            else:
                graph_data = np.array(data)[:, nodes[:, 0], nodes[:, 1], :]
            reader.close()
            hf_handle.create_dataset(f, data=graph_data, compression="lzf")
        hf_handle.close()
