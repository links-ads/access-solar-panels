import logging
from collections import OrderedDict
from pathlib import Path
import numpy as np
import rasterio

import torch
import torch.nn as nn
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from solarnet.config import (SegmenterTestSettings, SegmenterTrainSettings)
from solarnet.datasets import (BinaryDydasSegDataset, DydasSegmentationDataset, DydasInferenceDataset)
from solarnet.datasets.transforms import (segment_test_transforms)
from solarnet.logging.tensorboard import TensorBoardLogger
from solarnet.metrics import ConfusionMatrix, F1Score, IoU, Precision, Recall
from solarnet.models.testing import test_model
from solarnet.utils.common import (prepare_file_logging, prepare_folder)
from solarnet.utils.ml import (find_best_checkpoint, make_grid, mask_to_rgb, plot_confusion_matrix, seed_everything,
                               seed_worker)
from solarnet.tasks.utils import _load_class_weights, _load_model
from solarnet.utils.smooth_tiles import predict_smooth_windowing

LOG = logging.getLogger(__name__)


def test_segmenter(test_cfg: SegmenterTestSettings):
    device = torch.device(test_cfg.trainer.device)

    # prepare folders and check previous stuff
    experiment_folder = Path(test_cfg.output_folder)
    train_config_file = experiment_folder / "segmenter-config.yaml"
    assert train_config_file.exists(), f"Missing training configuration for experiment: {experiment_folder.stem}"

    # load the training configuration
    with open(str(train_config_file), "r", encoding="utf-8") as file:
        LOG.info("Loading config from: %s", str(train_config_file))
        train_cfg = yaml.load(file, Loader=yaml.Loader)
    # reload object and seed with same settings
    train_cfg = SegmenterTrainSettings(**train_cfg)
    seed_everything(train_cfg.seed, strict=train_cfg.deterministic)

    model_folder = experiment_folder / "models"
    prepare_file_logging(experiment_folder, filename="segmenter-test.log")
    LOG.info("Testing semantic segmentation results")

    dataset_cls = DydasSegmentationDataset if train_cfg.multiclass else BinaryDydasSegDataset
    num_classes = len(dataset_cls.categories)
    class_labels = list(dataset_cls.categories.values())

    data_folder = Path(train_cfg.data_folder)
    dataset = dataset_cls(data_folder=data_folder / "test",
                          transform=segment_test_transforms(in_channels=train_cfg.input_channels),
                          channels=train_cfg.input_channels)

    test_dataloader = DataLoader(
        dataset,
        batch_size=1,  # better for storing
        num_workers=test_cfg.trainer.num_workers,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=seed_worker)

    model = _load_model(train_cfg, num_classes=num_classes, norm_layer=nn.BatchNorm2d)
    model.to(device)
    target_name = test_cfg.model_name or "segmenter_*.pth"
    ckpt_path = find_best_checkpoint(model_folder, model_name=target_name)
    LOG.info("Loading checkpoint: %s", str(ckpt_path))
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint)

    # create a metric dictionary: reduction is none so that we can reduce ourselves
    reduct = test_cfg.metric_reduction
    metrics = OrderedDict([(f"{reduct}_f1", F1Score(num_classes=num_classes, reduction=reduct).to(device)),
                           (f"{reduct}_iou", IoU(num_classes=num_classes, reduction=reduct).to(device)),
                           (f"{reduct}_precision", Precision(num_classes=num_classes, reduction=reduct).to(device)),
                           (f"{reduct}_recall", Recall(num_classes=num_classes, reduction=reduct).to(device)),
                           ("class_f1", F1Score(num_classes=num_classes, reduction=None).to(device)),
                           ("class_iou", IoU(num_classes=num_classes, reduction=None).to(device)),
                           ("class_precision", Precision(num_classes=num_classes, reduction=None).to(device)),
                           ("class_recall", Recall(num_classes=num_classes, reduction=None).to(device)),
                           ("conf_mat", ConfusionMatrix(num_classes=num_classes).to(device))])

    LOG.info("Computing forward pass...")
    logger = TensorBoardLogger(log_folder=experiment_folder / "logs", filename_suffix="seg")
    logger_exclude = [name for name in metrics.keys() if name.startswith("class") or name == "conf_mat"]

    criterion = train_cfg.loss(**_load_class_weights(train_cfg.class_weights, train_cfg.multiclass)).to(device)
    _, scores, (images, y_true, y_pred) = test_model(model=model,
                                                     criterion=criterion,
                                                     test_dataloader=test_dataloader,
                                                     metrics=metrics,
                                                     logger=logger,
                                                     logger_exclude=logger_exclude,
                                                     device=device)

    num_images = len(images)

    LOG.info("Storing predicted images: %s)", str(test_cfg.store_predictions).lower())
    for i in tqdm(range(num_images)):
        if test_cfg.store_predictions:
            results_folder = prepare_folder(experiment_folder / "images")
            rgb_true = mask_to_rgb(y_true[i].squeeze(0), palette=dataset_cls.palette)
            rgb_pred = mask_to_rgb(y_pred[i].squeeze(0), palette=dataset_cls.palette)
            grid = make_grid(images[i], rgb_true, rgb_pred)
            plt.imsave(results_folder / f"{i:06d}.png", grid)

    LOG.info("Average results, reduction: '%s'", reduct)
    for i, (name, score) in enumerate(scores.items()):
        # only printing reduced metrics
        if name.startswith(reduct):
            LOG.info(f"{name:<20s}: {score.item():.4f}")

    LOG.info("Class-wise results, no reduction")
    header = f"{'score':<20s}  " + "|".join([f"{lab:<15s}" for lab in class_labels])
    LOG.info(header)
    for i, (name, score) in enumerate(scores.items()):
        if not name.startswith(reduct) and name != "conf_mat":
            scores_str = [f"{v:.4f}" for v in score]
            scores_str = "|".join(f"{s:<15s}" for s in scores_str)
            LOG.info(f"{name:<20s}: {scores_str}")

    LOG.info("Plotting confusion matrix...")
    cm_name = f"cm_{Path(ckpt_path).stem}"
    plot_folder = prepare_folder(experiment_folder / "plots")
    plot_confusion_matrix(scores["conf_mat"].cpu().numpy(),
                          destination=plot_folder / f"{cm_name}.png",
                          labels=class_labels,
                          title=cm_name,
                          normalize=False)
    LOG.info("Testing done!")


def infer_large_tiles(test_cfg: SegmenterTestSettings):
    LOG.info("Testing semantic segmentation results on large tiles")
    device = torch.device(test_cfg.trainer.device)

    # prepare folders and check previous stuff
    experiment_folder = Path(test_cfg.output_folder)
    train_config_file = experiment_folder / "segmenter-config.yaml"
    assert train_config_file.exists(), f"Missing training configuration for experiment: {experiment_folder.stem}"

    # check image file exists and is not empty
    assert test_cfg.large_images_file.exists(), f"Missing large images file: {test_cfg.large_images_file}"
    image_names = test_cfg.large_images_file.read_text().splitlines()
    assert len(image_names) > 0, f"Empty large images file: {test_cfg.large_images_file}"

    # load the training configuration
    with open(str(train_config_file), "r", encoding="utf-8") as file:
        LOG.info("Loading config from: %s", str(train_config_file))
        train_cfg = yaml.load(file, Loader=yaml.Loader)
    # reload object and seed with same settings
    train_cfg = SegmenterTrainSettings(**train_cfg)
    seed_everything(train_cfg.seed, strict=train_cfg.deterministic)

    model_folder = experiment_folder / "models"
    prepare_file_logging(experiment_folder, filename="segmenter-test.log")
    LOG.info("Testing semantic segmentation results")

    num_classes = len(DydasInferenceDataset.categories)
    data_folder = Path(test_cfg.data_folder)
    dataset = DydasInferenceDataset(data_folder=data_folder,
                                    images=image_names,
                                    channels=test_cfg.input_channels,
                                    transform=segment_test_transforms(in_channels=test_cfg.input_channels))

    # define a callback with forward
    def callback(patches: torch.Tensor) -> torch.Tensor:
        patch_preds = model(patches.to(device))
        return patch_preds.detach().cpu()

    model = _load_model(train_cfg, num_classes=num_classes, norm_layer=nn.BatchNorm2d)
    model.to(device)
    target_name = test_cfg.model_name or "segmenter_*.pth"
    ckpt_path = find_best_checkpoint(model_folder, model_name=target_name)
    LOG.info("Loading checkpoint: %s", str(ckpt_path))
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint)

    with torch.no_grad():
        model.eval()
        for batch in tqdm(dataset):
            image, image_name, src_profile, src_transform = batch
            # run the callback
            LOG.info("Running inference on image: %s", image_name)
            prediction = predict_smooth_windowing(image,
                                                  tile_size=test_cfg.tile_size,
                                                  subdivisions=2,
                                                  prediction_fn=callback,
                                                  batch_size=test_cfg.trainer.batch_size,
                                                  channels_first=True,
                                                  num_classes=num_classes)
            prediction = prediction.argmax(dim=0).type(torch.uint8).numpy()
            # save the predictions
            results_folder = prepare_folder(experiment_folder / "large_images")
            # rgb_pred = mask_to_rgb(prediction, palette=DydasInferenceDataset.palette, channels_first=True)
            image_file = results_folder / image_name

            LOG.info("Saving prediction to: %s", str(image_file))
            profile = src_profile.copy()
            profile.update(transform=src_transform, count=1)
            with rasterio.open(str(image_file), "w", **profile) as dst:
                dst.write(np.expand_dims(prediction, axis=0))
