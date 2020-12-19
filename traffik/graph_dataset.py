import torch
import os
import pandas as pd
import numpy as np
import h5py
from dotenv import load_dotenv
import traffik.config as config
from traffik.logger import logger


def build_graph(city):
    load_dotenv(verbose=False)

    train_raw = os.path.join(os.getenv("DATA_DIR"), city, config.TRAINING_DIR)
    valid_raw = os.path.join(os.getenv("DATA_DIR"), city, config.VALIDATION_DIR)
    test_raw = os.path.join(os.getenv("DATA_DIR"), city, config.TESTING_DIR)

    node_handle = os.path.join(config.PROCESSED_DIR, city, f"{city}_nodes_5.npy")
    edge_handle = os.path.join(config.PROCESSED_DIR, city, f"{city}_edges_5.npy")

    for mode in [config.TRAINING_DIR, config.VALIDATION_DIR, config.TESTING_DIR]:
        logger.msg("Starting:", mode=mode)
