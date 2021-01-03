import wandb
import numpy as np
import os
import random
import torch
from traffik.config import INTERMEDIATE_DIR
from typing import Union
from traffik.logger import logger


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
