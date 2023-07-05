from __future__ import annotations

import logging
from enum import Enum
from posix import listdir
from typing import TYPE_CHECKING, Any, Dict

import torch
from solarnet.logging import BaseLogger
from solarnet.logging.tensorboard import TensorBoardLogger
from solarnet.metrics import Metric
from solarnet.trainer.base import Trainer
from torch import nn
from torch.optim import Optimizer

if TYPE_CHECKING:
    from trainer.callbacks import BaseCallback

LOG = logging.getLogger(__name__)


class TrainerStage(str, Enum):
    train = "train"
    val = "val"
    test = "test"


class SSLTrainer(Trainer):

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: Optimizer,
                 scheduler: Any,
                 train_metrics: Dict[str, Metric],
                 val_metrics: Dict[str, Metric],
                 logger: BaseLogger = None,
                 device: torch.device = None) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = {TrainerStage.train.value: train_metrics, TrainerStage.val.value: val_metrics}
        self.logger = logger or TensorBoardLogger()
        self.device = device or torch.device("gpu")
        # internal state
        self.current_epoch = -1
        self.global_step = -1
        # internal monitoring
        self.current_scores = {TrainerStage.train.value: dict(), TrainerStage.val.value: dict()}
        self.best_epoch = None
        self.best_score = None
        self.best_state_dict = None
        self.callbacks: listdir[BaseCallback] = list()

    def train_batch(self, batch: Any) -> torch.Tensor:
        x, y, xm, ym = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        xm = xm.to(self.device, non_blocking=True)
        ym = ym.to(self.device, non_blocking=True)
        preds = self.model(x)
        preds_m = self.model(xm)
        loss = self.criterion(preds, y) + self.criterion(preds_m, ym)
        self._update_metrics(y_true=y, y_pred=preds, stage=TrainerStage.train)
        return loss

    def validation_batch(self, batch: Any):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        preds = self.model(x)
        loss = self.criterion(preds, y)
        self._update_metrics(y_true=y, y_pred=preds, stage=TrainerStage.val)
        return loss

    def test_batch(self, batch: Any):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        preds = self.model(x)
        loss = self.criterion(preds, y)
        self._update_metrics(y_true=y, y_pred=preds, stage=TrainerStage.test)
        return loss, (x, y, torch.argmax(preds, dim=1))
