import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from solarnet.config import (ClassifierSettings, Metrics, SegmenterTrainSettings)
from solarnet.datasets import (BinaryDydasSegDataset, DydasSegmentationDataset, USGSClassifierDataset, mask_set)
from solarnet.datasets.transforms import (classif_test_transforms, classif_train_transforms, segment_test_transforms,
                                          segment_train_transforms)
from solarnet.logging.tensorboard import TensorBoardLogger
from solarnet.models import Classifier, train_model
from solarnet.trainer.base import Trainer
from solarnet.trainer.callbacks import EarlyStopping, EarlyStoppingCriterion
from solarnet.utils.common import (progressbar, store_config)
from solarnet.utils.ml import (init_experiment, seed_worker, string_summary)
from solarnet.tasks.utils import _load_class_weights, _load_model

LOG = logging.getLogger(__name__)


def train_classifier(cfg: ClassifierSettings):
    """Trains the classifier for pretraining.

    Args:
        cfg (ClassifierSettings): settings of the classifier.
    """
    data_folder = Path(cfg.data_folder)
    device = torch.device(cfg.device)
    exp_id, out_folder, model_folder, logs_folder = init_experiment(config=cfg, log_name="classifier-out.log")
    store_config(cfg, path=out_folder / "classifier-config.yaml")

    # load model
    LOG.info("Starting experiment: %s", exp_id)
    model = Classifier(backbone=cfg.backbone, pretrained=True, num_classes=1)
    LOG.info("Model\n%s", string_summary(model, input_size=(3, 256, 256), batch_size=cfg.batch_size))
    LOG.info("Training on device: %s", str(cfg.device))
    model = model.to(device=device)

    logger = TensorBoardLogger(log_folder=logs_folder, filename_suffix="clf")
    dataset = USGSClassifierDataset(data_folder=data_folder, transform=classif_train_transforms())
    # prepare splits using boolean masks
    train_mask, val_mask, test_mask = mask_set(len(dataset), cfg.val_size, cfg.test_size)
    dataset.add_mask(train_mask)
    train_dataloader = DataLoader(dataset,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  pin_memory=True,
                                  worker_init_fn=seed_worker)
    val_dataloader = DataLoader(USGSClassifierDataset(mask=val_mask,
                                                      data_folder=data_folder,
                                                      transform=classif_test_transforms()),
                                batch_size=cfg.batch_size,
                                num_workers=8,
                                pin_memory=True,
                                worker_init_fn=seed_worker)
    test_dataloader = DataLoader(USGSClassifierDataset(mask=test_mask,
                                                       data_folder=data_folder,
                                                       transform=classif_test_transforms()),
                                 batch_size=cfg.batch_size,
                                 num_workers=8,
                                 pin_memory=True,
                                 worker_init_fn=seed_worker)

    # prepare loss, optimizer and scheduler
    loss = nn.BCEWithLogitsLoss().to(device)
    optimizer = Adam(params=model.parameters(), lr=cfg.lr)
    scheduler = ExponentialLR(optimizer, gamma=0.85)
    # prepare metrics
    metrics = {
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
    }
    monitored_metric = "f1"
    # train classifier
    epoch, best_loss, best_score = train_model(model=model,
                                               criterion=loss,
                                               optimizer=optimizer,
                                               scheduler=scheduler,
                                               train_dataloader=train_dataloader,
                                               val_dataloader=val_dataloader,
                                               logger=logger,
                                               metrics=metrics,
                                               monitor=monitored_metric,
                                               device=device,
                                               patience=cfg.patience,
                                               max_epochs=cfg.max_epochs)

    LOG.info(f"Training completed at epoch {epoch:<2d} "
             f"(best loss: {best_loss:.4f}, best {monitored_metric}: {best_score:.4f})")
    model_path = model_folder / f"classifier_loss-{best_loss:.4f}_{monitored_metric}-{best_score:.4f}.pth"
    torch.save(model.state_dict(), model_path)

    # save predictions for analysis
    LOG.info("Generating test results")
    pred, true = [], []
    test_tqdm = progressbar(test_dataloader, epoch=epoch, stage="testing")
    with torch.no_grad():
        for test_x, test_y in test_tqdm:
            test_x = test_x.to(device, non_blocking=True)
            test_y = test_y.to(device, non_blocking=True)
            test_pred = model(test_x)
            pred.append(torch.sigmoid(test_pred.squeeze(1)).cpu().numpy())
            true.append(test_y.cpu().numpy())

    np.save(model_folder / 'classifier_pred.npy', np.concatenate(pred))
    np.save(model_folder / 'classifier_true.npy', np.concatenate(true))
    LOG.info("Experiment '%s' completed", exp_id)


def train_segmenter(cfg: SegmenterTrainSettings):
    """Trains the segmentation model using a standard supervised approach.

    Args:
        cfg (SegmenterTrainSettings): settings of the segmentation network.
    """
    data_folder = Path(cfg.data_folder)
    device = torch.device(cfg.trainer.device)
    exp_id, out_folder, model_folder, logs_folder = init_experiment(config=cfg, log_name="segmenter.log")
    store_config(cfg, path=out_folder / "segmenter-config.yaml")

    # prepare dataset
    dataset_cls = DydasSegmentationDataset if cfg.multiclass else BinaryDydasSegDataset
    num_classes = len(dataset_cls.categories)
    dataset_train = dataset_cls(data_folder=data_folder / "train",
                                transform=segment_train_transforms(in_channels=cfg.input_channels),
                                channels=cfg.input_channels)
    dataset_val = dataset_cls(data_folder=data_folder / "val",
                              transform=segment_test_transforms(in_channels=cfg.input_channels),
                              channels=cfg.input_channels)
    dataset_test = dataset_cls(data_folder=data_folder / "test",
                               transform=segment_test_transforms(in_channels=cfg.input_channels),
                               channels=cfg.input_channels)

    LOG.info(
        "Training samples: %d, validation samples: %d, testing samples: %d",
        len(dataset_train),
        len(dataset_val),
        len(dataset_test),
    )
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
    trainer = Trainer(model=model,
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
