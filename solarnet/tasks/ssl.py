import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from solarnet.config import Metrics, SSLSegmenterTrainSettings
from solarnet.datasets import SSLDydasDataset, DydasSegmentationDataset
from solarnet.datasets.transforms import segment_test_transforms, segment_train_transforms
from solarnet.logging.tensorboard import TensorBoardLogger
from solarnet.trainer.callbacks import EarlyStopping, EarlyStoppingCriterion
from solarnet.trainer.ssl import SSLTrainer
from solarnet.utils.common import store_config
from solarnet.utils.ml import init_experiment, mask_to_rgb, seed_worker, string_summary
from solarnet.tasks.utils import _load_class_weights, _load_model
from PIL import Image

from solarnet.utils.transforms import Denormalize

LOG = logging.getLogger(__name__)


def save_tensor_as_image(tensor: torch.Tensor, path: Path, palette: dict, channels: int = 3) -> None:
    denormalizer = Denormalize(mean=[0.485, 0.456, 0.406, 0.485], std=[0.229, 0.224, 0.225, 0.229])
    if tensor.ndim == 3:
        tensor = tensor.cpu().detach().permute(1, 2, 0)
        tensor = denormalizer(tensor)
        tensor = tensor[:, :, :3]
    else:
        tensor = mask_to_rgb(tensor.numpy(), palette=palette, channels_first=True)
        tensor = torch.from_numpy(tensor).permute(1, 2, 0)
    tensor = (tensor.numpy()).astype('uint8')
    image = Image.fromarray(tensor)
    image.save(path)


def train_segmenter_ssl(cfg: SSLSegmenterTrainSettings):
    """Trains the segmentation model using a standard supervised approach.

    Args:
        cfg (SegmenterTrainSettings): settings of the segmentation network.
    """
    data_folder = Path(cfg.data_folder)
    device = torch.device(cfg.trainer.device)
    exp_id, out_folder, model_folder, logs_folder = init_experiment(config=cfg, log_name="segmenter.log")
    store_config(cfg, path=out_folder / "segmenter-config.yaml")

    # prepare dataset
    num_classes = len(DydasSegmentationDataset.categories)
    dataset_train = SSLDydasDataset(labeled_folder=data_folder / "train",
                                    unlabeled_folder=data_folder / "unlabeled",
                                    transform=segment_train_transforms(in_channels=cfg.input_channels),
                                    unlabeled_transform=segment_train_transforms(in_channels=cfg.input_channels),
                                    channels=cfg.input_channels)
    dataset_val = DydasSegmentationDataset(data_folder=data_folder / "val",
                                           transform=segment_test_transforms(in_channels=cfg.input_channels),
                                           channels=cfg.input_channels)
    dataset_test = DydasSegmentationDataset(data_folder=data_folder / "test",
                                            transform=segment_test_transforms(in_channels=cfg.input_channels),
                                            channels=cfg.input_channels)

    LOG.info(
        "Training samples: %d, validation samples: %d, testing samples: %d",
        len(dataset_train),
        len(dataset_val),
        len(dataset_test),
    )

    # for i in range(10):
    #     x, y, xm, ym = dataset_train[i]
    #     save_tensor_as_image(x, Path(f"debug/train_{i}.png"), palette=DydasSegmentationDataset.palette)
    #     save_tensor_as_image(y, Path(f"debug/train_{i}_label.png"), palette=DydasSegmentationDataset.palette)
    #     save_tensor_as_image(xm, Path(f"debug/train_{i}_mix.png"), palette=DydasSegmentationDataset.palette)
    #     save_tensor_as_image(ym, Path(f"debug/train_{i}_mix_label.png"), palette=DydasSegmentationDataset.palette)
    # exit(0)

    train_dataloader = DataLoader(dataset_train,
                                  batch_size=cfg.trainer.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.trainer.num_workers,
                                  pin_memory=True,
                                  worker_init_fn=seed_worker)
    val_dataloader = DataLoader(dataset_val,
                                batch_size=cfg.trainer.batch_size,
                                num_workers=cfg.trainer.num_workers,
                                pin_memory=True,
                                worker_init_fn=seed_worker)

    # load model
    LOG.info("Starting experiment: %s", exp_id)
    model = _load_model(cfg, num_classes=num_classes, norm_layer=nn.BatchNorm2d)
    # print model summary
    input_size = (cfg.input_channels, cfg.image_size, cfg.image_size)
    LOG.info("Model\n%s", string_summary(model, input_size=input_size, batch_size=cfg.trainer.batch_size))
    LOG.info("Training on device: %s", str(cfg.trainer.device))
    model = model.to(device=device)

    # prepare loss, optimizer and scheduler
    loss = cfg.loss(**_load_class_weights(cfg.class_weights, cfg.multiclass)).to(device)
    encoder_lr = cfg.enc_lr or cfg.trainer.lr
    decoder_lr = cfg.trainer.lr
    params = [{
        "params": model.encoder_parameters(),
        "lr": encoder_lr
    }, {
        "params": model.decoder_parameters(),
        "lr": decoder_lr
    }]
    optimizer = cfg.optimizer(params, lr=cfg.trainer.lr, weight_decay=cfg.trainer.weight_decay)
    scheduler = cfg.scheduler(optimizer)
    # prepare metrics
    train_metrics = {e.name: e.value(num_classes=num_classes).to(device) for e in Metrics}
    valid_metrics = {e.name: e.value(num_classes=num_classes).to(device) for e in Metrics}
    monitored_metric = cfg.monitor.name
    # prepare logger
    logger = TensorBoardLogger(log_folder=logs_folder, filename_suffix="seg", comment=cfg.comment)
    logger.log_model(model, input_size=(1, cfg.input_channels, cfg.image_size, cfg.image_size), device=device)
    # prepare callbacks
    early_stopping = EarlyStopping(call_every=1,
                                   metric=monitored_metric,
                                   criterion=EarlyStoppingCriterion.maximum,
                                   patience=cfg.trainer.patience)
    # train classifier
    trainer = SSLTrainer(model=model,
                         criterion=loss,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         train_metrics=train_metrics,
                         val_metrics=valid_metrics,
                         logger=logger,
                         device=device)
    trainer.add_callback(early_stopping)
    trainer.fit(train_dataloader=train_dataloader, val_dataloader=val_dataloader, max_epochs=cfg.trainer.max_epochs)

    LOG.info(f"Training completed at epoch {trainer.current_epoch:<2d} "
             f"(best {monitored_metric}: {trainer.best_score:.4f})")
    model_path = model_folder / f"segmenter_{monitored_metric}-{trainer.best_score:.4f}.pth"
    torch.save(model.state_dict(), model_path)
    LOG.info("Experiment '%s' completed", exp_id)
