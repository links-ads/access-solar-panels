import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import yaml
from pydantic import BaseSettings
from torch.utils.data import DataLoader
from tqdm import tqdm


def current_timestamp() -> str:
    return datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M")


def generate_id() -> str:
    return str(uuid4())


def prepare_folder(root_folder: Path, experiment_id: str = ""):
    if isinstance(root_folder, str):
        root_folder = Path(root_folder)
    full_path = root_folder / experiment_id
    if not full_path.exists():
        os.makedirs(str(full_path))
    return full_path


def prepare_base_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(name)-24s: %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M',
    )


def prepare_file_logging(experiment_folder: Path, filename: str = "output.log") -> None:
    logger = logging.getLogger()
    handler = logging.FileHandler(experiment_folder / filename)
    handler.setLevel(logging.INFO)
    # get the handler from the base handler
    handler.setFormatter(logger.handlers[0].formatter)
    logger.addHandler(handler)


def progressbar(dataloder: DataLoader, epoch: int = 0, stage: str = "train"):
    pbar = tqdm(dataloder, file=sys.stdout, unit="batch", postfix={"loss": "--"})
    pbar.set_description(f"Epoch {epoch:<3d} - {stage}")
    return pbar


def store_config(config: BaseSettings, path: Path) -> None:
    with open(str(path), "w") as file:
        yaml.dump(config.dict(), file)
