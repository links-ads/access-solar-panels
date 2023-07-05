import logging
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from solarnet.datasets import (DydasSegmentationDataset)
from solarnet.preprocessing import ImageSplitter, MaskMaker
from solarnet.utils.ml import (compute_class_weights)

LOG = logging.getLogger(__name__)


def make_masks(data_folder='data'):
    """Generates raster masks from solar panel polygons (USGS dataset)

    Args:
        data_folder (str, optional): folder containing data. Defaults to 'data'.
    """
    mask_maker = MaskMaker(data_folder=Path(data_folder))
    mask_maker.process()


def split_images(data_folder='data', imsize=256, empty_ratio=2):
    """Split the available images into imsize x imsize tiles.
    Store every tile with solar panels, and 2x the same quantity of empty tiles.

    Args:
        data_folder (str, optional): folder with original masks to be split. Defaults to 'data'.
        imsize (int, optional): size of a single tile. Defaults to 256.
        empty_ratio (int, optional): ratio of empty tiles w.r.t. solar tiles. Defaults to 2.
    """
    splitter = ImageSplitter(data_folder=Path(data_folder))
    splitter.process(imsize=imsize, empty_ratio=empty_ratio)


def compute_stats(data_folder: str,
                  output_folder: str = "data",
                  store_stats: bool = True,
                  weight_smoothing: float = 0.2):
    data_folder = Path(data_folder)
    output_folder = Path(output_folder)
    assert data_folder.exists(), f"Data folder '{data_folder}' does not exist"
    assert output_folder.exists(), f"Output folder '{output_folder}' does not exist"

    LOG.info("Computing stats for data: %s", str(data_folder))
    dataset = DydasSegmentationDataset(data_folder=data_folder, transform=None, channels=4)
    loader = DataLoader(dataset, batch_size=1, num_workers=1)

    total_result = {index: 0 for index in dataset.categories.keys()}
    for _, label in tqdm(loader):
        values, counts = np.unique(label.numpy(), return_counts=True)
        for v, c in zip(values, counts):
            total_result[v] += c

    LOG.info("Raw counts:")
    for value, total in total_result.items():
        LOG.info(f"{dataset.categories[value]:>10s}: {total:>9d}")

    LOG.info("Computing class weights...")
    weights = compute_class_weights(total_result, smoothing=weight_smoothing, clip=10.0)
    for index, weight in weights.items():
        LOG.info(f"({index}) {dataset.categories[index]:>10s}: {weight:>.2f}")

    if store_stats:
        LOG.info("Storing data in: %s", str(output_folder))
        weights_arr = np.array(list(weights.values()))
        np.save(output_folder / "class_weights.npy", weights_arr)
    LOG.info("Done!")
