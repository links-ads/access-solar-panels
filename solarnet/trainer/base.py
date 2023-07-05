from __future__ import annotations

import logging
import time
from enum import Enum
from posix import listdir
from typing import TYPE_CHECKING, Any, Dict, Iterable

import numpy as np
import torch
from solarnet.logging import BaseLogger
from solarnet.logging.tensorboard import TensorBoardLogger
from solarnet.metrics import Metric
from solarnet.utils.common import progressbar
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from trainer.callbacks import BaseCallback

LOG = logging.getLogger(__name__)


class TrainerStage(str, Enum):
    train = "train"
    val = "val"
    test = "test"


class Trainer:

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

    def _update_metrics(self,
                        y_true: torch.Tensor,
                        y_pred: torch.Tensor,
                        stage: TrainerStage = TrainerStage.train) -> None:
        with torch.no_grad():
            for metric in self.metrics[stage.value].values():
                metric(y_true, y_pred)

    def _compute_metrics(self, stage: TrainerStage = TrainerStage.train) -> None:
        result = dict()
        with torch.no_grad():
            for name, metric in self.metrics[stage.value].items():
                result[name] = metric.compute()
        self.current_scores[stage.value] = result

    def _reset_metrics(self, stage: TrainerStage = TrainerStage.train) -> None:
        for metric in self.metrics[stage.value].values():
            metric.reset()

    def _log_metrics(self, stage: TrainerStage = TrainerStage.train, exclude: Iterable[str] = None) -> None:
        log_strings = []
        scores = self.current_scores[stage.value]
        for metric_name, score in scores.items():
            if exclude is not None and metric_name in exclude:
                continue
            self.logger.log_scalar(f"{stage.value}/{metric_name}", score)
            log_strings.append(f"{stage.value}/{metric_name}: {score:.4f}")
        LOG.info(", ".join(log_strings))

    def add_callback(self, callback: BaseCallback) -> Trainer:
        self.callbacks.append(callback)
        return self

    def setup_callbacks(self) -> None:
        for callback in self.callbacks:
            callback.setup(self)

    def dispose_callbacks(self) -> None:
        for callback in self.callbacks:
            callback.dispose(self)

    def step(self) -> None:
        self.global_step += 1
        self.logger.step()

    def train_epoch_start(self):
        self._reset_metrics(stage=TrainerStage.train)

    def train_batch(self, batch: Any) -> torch.Tensor:
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        preds = self.model(x)
        loss = self.criterion(preds, y)
        self._update_metrics(y_true=y, y_pred=preds, stage=TrainerStage.train)
        return loss

    def train_epoch_end(self, train_losses: list, train_times: list):
        with torch.no_grad():
            self._compute_metrics(stage=TrainerStage.train)
        self.logger.log_scalar("train/loss", np.mean(train_losses))
        self.logger.log_scalar("train/time", np.mean(train_times))
        self._log_metrics(stage=TrainerStage.train)

    def validation_epoch_start(self):
        self._reset_metrics(stage=TrainerStage.val)

    def validation_batch(self, batch: Any):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        preds = self.model(x)
        loss = self.criterion(preds, y)
        self._update_metrics(y_true=y, y_pred=preds, stage=TrainerStage.val)
        return loss

    def validation_epoch_end(self, val_losses: list, val_times: list):
        with torch.no_grad():
            self._compute_metrics(stage=TrainerStage.val)
        self.logger.log_scalar("val/loss", np.mean(val_losses))
        self.logger.log_scalar("val/time", np.mean(val_times))
        self._log_metrics(stage=TrainerStage.val)

    def test_batch(self, batch: Any):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        preds = self.model(x)
        loss = self.criterion(preds, y)
        self._update_metrics(y_true=y, y_pred=preds, stage=TrainerStage.test)
        return loss, (x, y, torch.argmax(preds, dim=1))

    def train_epoch(self, epoch: int, train_dataloader: DataLoader) -> Any:
        losses, timings = [], []
        train_tqdm = progressbar(train_dataloader, epoch=epoch, stage=TrainerStage.train.value)

        self.model.train()
        for batch in train_tqdm:
            start = time.time()
            self.optimizer.zero_grad()
            loss = self.train_batch(batch=batch)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            # measure elapsed time
            elapsed = (time.time() - start)
            # store training info
            loss_val = loss.item()
            train_tqdm.set_postfix({"loss": loss_val})
            self.logger.log_scalar("train/loss_iter", loss_val)
            self.logger.log_scalar("train/lr", self.optimizer.param_groups[0]["lr"])
            self.logger.log_scalar("train/time_iter", elapsed)
            losses.append(loss_val)
            timings.append(elapsed)
            # step the logger
            self.step()

        return losses, timings

    def validation_epoch(self, epoch: int, val_dataloader: DataLoader) -> Any:
        val_tqdm = progressbar(val_dataloader, epoch=epoch, stage=TrainerStage.val.value)
        losses, timings = [], []

        with torch.no_grad():
            self.model.eval()
            for batch in val_tqdm:
                start = time.time()
                loss = self.validation_batch(batch=batch)
                elapsed = (time.time() - start)
                loss_val = loss.item()
                val_tqdm.set_postfix({"loss": loss_val})
                # we do not log 'iter' versions for loss and timings, since we do notadvance the logger step
                # during validation (also, it's kind of useless)
                losses.append(loss_val)
                timings.append(elapsed)

        return losses, timings

    def fit(self, train_dataloader: DataLoader, val_dataloader: DataLoader = None, max_epochs: int = 100):
        self.best_state_dict = self.model.state_dict()
        self.setup_callbacks()
        self.global_step = 0

        for curr_epoch in range(max_epochs):
            self.current_epoch = curr_epoch
            LOG.info(f"[Epoch {self.current_epoch:>2d}]")
            try:
                self.train_epoch_start()
                t_losses, t_times = self.train_epoch(epoch=self.current_epoch, train_dataloader=train_dataloader)
                self.train_epoch_end(t_losses, t_times)

                if val_dataloader is not None:
                    self.validation_epoch_start()
                    v_losses, v_times = self.validation_epoch(epoch=self.current_epoch, val_dataloader=val_dataloader)
                    self.validation_epoch_end(v_losses, v_times)

                for callback in self.callbacks:
                    callback(self)

            except KeyboardInterrupt:
                LOG.info("[Epoch %2d] Interrupting training", curr_epoch)
                break

        self.dispose_callbacks()
        return self

    def predict(self, test_dataloader: DataLoader, metrics: Dict[str, Metric], logger_exclude: Iterable[str]):
        self.metrics[TrainerStage.test.value] = metrics
        self._reset_metrics(stage=TrainerStage.test)
        test_tqdm = progressbar(test_dataloader, stage=TrainerStage.test.value)
        losses, timings, results = [], [], []

        with torch.no_grad():
            self.model.eval()
            for batch in test_tqdm:
                start = time.time()
                loss, data = self.test_batch(batch=batch)
                elapsed = (time.time() - start)
                loss_value = loss.item()
                test_tqdm.set_postfix({"loss": loss_value})
                # we do not log 'iter' versions, as for validation
                losses.append(loss_value)
                timings.append(elapsed)
                results.append(data)

            self.logger.log_scalar("test/loss", np.mean(losses))
            self.logger.log_scalar("test/time", np.mean(timings))
            self._compute_metrics(stage=TrainerStage.test)
            self._log_metrics(stage=TrainerStage.test, exclude=logger_exclude)

        return losses, results
