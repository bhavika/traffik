import wandb
import numpy as np
import os
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
