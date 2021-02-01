import wandb
import numpy as np
import os
import json
import random
import torch
from traffik.config import INTERMEDIATE_DIR
from typing import Union
from traffik.logger import logger
from tqdm import tqdm
import h5py
from torch_geometric.data import Data


def data_logger(data_obj: Union[str, np.ndarray], description: str = "train"):
    run_id = wandb.run.id

    if isinstance(data_obj, str):
        if os.path.exists(data_obj):
            logger.info("[data_logger] Saving file to W&B", file=data_obj)
            wandb.save(data_obj)

    elif isinstance(data_obj, np.ndarray):
        f = os.path.join(INTERMEDIATE_DIR, f"{description}_{run_id}.csv")
        np.savetxt(f)
        logger.info("[data_logger] Saving numpy array to W&B", file=f)
        wandb.save(f)


def reproducibility():
    seed = 7
    random.seed(seed)
    torch.manual_seed = seed
    wandb.config.manual_seed = seed

    if torch.cuda.is_available():
        torch.cuda.manual_seed = seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_test_slots():
    with open("data/test_slots.json", "r") as json_file:
        test_slots = json.load(json_file)
        test_slots = {k: v for each in test_slots for k, v in each.items()}
    return test_slots


def create_submissions(model, city, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.freeze()
    model.eval()

    test_slots = get_test_slots()

    for date, frame in tqdm(test_slots.items()):
        data = []
        with h5py.File(
            os.path.join(
                os.getenv("DATA_DIR"), INTERMEDIATE_DIR, f"{city}_testing_5.h5"
            ),
            "r",
        ) as f:
            test_data = f[f"{date}_test.h5"]

            for i in range(len(frame)):
                x = test_data[i]
                slice_id = frame[i] + 12

                x = dataset.get_training_data(x, dataset.city, 12, slice_id=slice_id)
                data = Data(x=x, edge_index=dataset.edges)
                out = model(data.to(device))

                zeros_img = torch.zeros([1, 12, 495, 436, 8], dtype=torch.float)
                out = dataset.convert_graph_minibatch_y_to_image(
                    out.to("cpu"), zeros_img
                )
                out = out[:, dataset.forward_steps - 1, :, :, :]
                out = out.round()
                out = torch.clamp(out, min=0, max=255)
                out = out.byte()
                if i == 0:
                    prediction = out
                else:
                    prediction = torch.cat((prediction, out), 0)

        with h5py.File(
            os.path.join(
                os.getenv("DATA_DIR"), f"{INTERMEDIATE_DIR}/{data}_test.h5", "w"
            )
        ) as fdata:
            fdata.create_dataset("array", data=prediction, compression="gzip")
