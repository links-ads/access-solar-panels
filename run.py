import click
from tqdm.contrib.logging import logging_redirect_tqdm

from solarnet import tasks
from solarnet.cli import command
from solarnet.config import (ClassifierSettings, SegmenterTestSettings, SegmenterTrainSettings,
                             SSLSegmenterTrainSettings)
from solarnet.utils.common import prepare_base_logging


@click.group()
def cli():
    pass


@click.command()
@click.option("--data-folder", type=str, required=True)
def generate_masks(data_folder: str):
    """Saves masks for each .tif image in the raw dataset. Masks are saved
    in  <org_folder>_mask/<org_filename>.tif where <org_folder> should be the
    city name, as defined in `data/README.md`.

    Parameters
    ----------
    data_folder: pathlib.Path
        Path of the data folder, which should be set up as described in `data/README.md`
    """
    return tasks.make_masks(data_folder)


@click.command()
@click.option("--data-folder", type=str, required=True)
@click.option("--imsize", type=int)
@click.option("--empty_ratio", type=int)
def generate_tiles(data_folder: str, imsize: int = 256, empty_ratio: int = 2):
    """Generates images (and their corresponding masks) of height = width = imsize
    for input into the models.

    Args:
        data_folder (pathlib.Path): Path of the data folder, which should be set up as described in `data/README.md`
        imsize (int): The size of the images to be generated, default: 224
        empty_ratio (int): The ratio of images without solar panels to images with solar panels.
            Because images without solar panels are randomly sampled with limited
            patience, having this number slightly > 1 yields a roughly 1:1 ratio. Default: 2
    """
    return tasks.split_images(data_folder, imsize, empty_ratio)


@click.command()
@click.option("--data-folder", type=str, required=True)
@click.option("--output-folder", type=str, default="data", required=False)
@click.option("--store-stats", type=bool, default=True, required=False)
@click.option("--weight-smoothing", type=float, default=0.1, required=False)
def compute_stats(data_folder: str, output_folder: str, store_stats: bool, weight_smoothing: float):
    tasks.compute_stats(data_folder, output_folder, store_stats=store_stats, weight_smoothing=weight_smoothing)


@command(config=ClassifierSettings)
def train_classifier(config: ClassifierSettings):
    """Train the classifier only, on a pretraining dataset (USGS)

    Args:
        config (ClassifierSettings): settings object
    """
    return tasks.train_classifier(config)


@command(config=SegmenterTrainSettings)
def train_segmenter(config: SegmenterTrainSettings):
    """Train the segmenter, starting from stratch, or from a previous classifier/encoder.

    Args:
        config (SegmenterSettings): settings object for segmentation tasks.
    """
    return tasks.train_segmenter(config)


@command(config=SSLSegmenterTrainSettings)
def train_segmenter_ssl(config: SSLSegmenterTrainSettings):
    """Train the segmenter in a Semi-Supervised Learning setup.

    Args:
        config (SegmenterSettings): settings object for segmentation tasks.
    """
    return tasks.train_segmenter_ssl(config)


@command(config=SegmenterTestSettings)
def test_segmenter(config: SegmenterTestSettings):
    """Test a specified segmenter, indicating which experiment and optionally which run/model/dataset.

    Args:
        config (SegmenterTestSettings): segmentation settings for testing purposes.
    """
    return tasks.test_segmenter(config)


@command(config=SegmenterTestSettings)
def test_large_tiles(config: SegmenterTestSettings):
    """Test a specified segmenter, indicating which experiment and optionally which run/model/dataset.

    Args:
        config (SegmenterTestSettings): segmentation settings for testing purposes.
    """
    return tasks.infer_large_tiles(config)


@command(config=SegmenterTestSettings)
def test_segmenter_ssl(config: SegmenterTestSettings):
    """Run tests on  the given experiment, specific for self-supervised learning.

    Args:
        config (SegmenterTestSettings): segmentation settings for testing.
    """
    return tasks.test_segmenter_ssl(config)


if __name__ == '__main__':
    # configure basic logging
    cli.add_command(generate_masks)
    cli.add_command(generate_tiles)
    cli.add_command(compute_stats)
    cli.add_command(train_classifier)
    cli.add_command(train_segmenter)
    cli.add_command(test_segmenter)
    cli.add_command(test_large_tiles)
    cli.add_command(train_segmenter_ssl)
    cli.add_command(test_segmenter_ssl)
    prepare_base_logging()
    with logging_redirect_tqdm():
        cli()
