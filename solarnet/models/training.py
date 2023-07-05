import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from solarnet.logging import BaseLogger
from solarnet.metrics import Metric
from solarnet.utils.common import progressbar
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

LOG = logging.getLogger(__name__)


#@torch.no_grad
def _update_metrics(metrics: Dict[str, Metric], y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
    for metric in metrics.values():
        metric(y_true, y_pred)


def _compute_metrics(metrics: Dict[str, Metric]) -> Dict[str, torch.Tensor]:
    result = dict()
    for name, metric in metrics.items():
        result[name] = metric.compute()
    return result


def _reset_metrics(metrics: Dict[str, Metric]) -> None:
    for metric in metrics.values():
        metric.reset()


def _log_metrics(stage: str, scores: Dict[str, float], logger: Optional[BaseLogger] = None) -> None:
    log_strings = []
    for metric_name, score in scores.items():
        logger.log_scalar(f"{stage}/{metric_name}", score)
        log_strings.append(f"{stage}/{metric_name}: {score:.4f}")
    LOG.info(", ".join(log_strings))


def _adaptive_squeeze(x: torch.Tensor, apply_sigmoid: bool = False):
    if x.ndim == 4:
        is_binary = x.size(1) == 1
        x = x.squeeze(1) if is_binary else torch.argmax(x, dim=1)
    if apply_sigmoid:
        x = torch.sigmoid(x)
    return x


def _train_model_epoch(epoch: int,
                       model: torch.nn.Module,
                       criterion: nn.Module,
                       optimizer: Optimizer,
                       scheduler: nn.Module,
                       train_dataloader: DataLoader,
                       val_dataloader: DataLoader,
                       metrics: Dict[str, Callable],
                       logger: BaseLogger,
                       device: str = "cpu") -> Tuple[Tuple[np.ndarray, Dict], Tuple[np.ndarray, Dict]]:
    """Completes a single training epoch, with a single validation loop.

    Args:
        epoch (int): current epoch
        model (torch.nn.Module): current model
        criterion (nn.Module): loss instance
        optimizer (Optimizer): optimizer instance
        scheduler (nn.Module): scheduler instance
        train_dataloader (DataLoader): train dataloader
        val_dataloader (DataLoader): validation dataloader
        logger (BaseLogger): logger to store stuff
        device (str, optional): current device. Defaults to "cpu".

    Returns:
        Tuple[Tuple[np.ndarray, Dict], Tuple[np.ndarray, Dict]]: two tuples, containing losses and best metric
        for train and validation sets.
    """
    t_losses, v_losses, timings = [], [], []
    train_tqdm = progressbar(train_dataloader, epoch=epoch, stage="training")
    _reset_metrics(metrics["train"])
    _reset_metrics(metrics["val"])

    model.train()
    for x, y in train_tqdm:
        start = time.time()
        optimizer.zero_grad()
        _handle_train_batch_supervised(model, criterion, batch, device=device)
        preds = model(x)
        # step
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        # update metrics
        _update_metrics(metrics["train"], y_true=y, y_pred=preds)
        #store training info
        loss_val = loss.item()
        elapsed = (time.time() - start)
        train_tqdm.set_postfix({"loss": loss_val})
        logger.log_scalar("train/loss_iter", loss_val)
        logger.log_scalar("train/lr", optimizer.param_groups[0]["lr"])
        logger.log_scalar("train/time_iter", elapsed)
        t_losses.append(loss_val)
        timings.append(elapsed)
        # step the logger
        logger.step()

    v_losses = _eval_model_epoch(epoch=epoch,
                                 model=model,
                                 criterion=criterion,
                                 val_dataloader=val_dataloader,
                                 metrics=metrics["val"],
                                 device=device)
    t_scores = _compute_metrics(metrics["train"])
    v_scores = _compute_metrics(metrics["val"])

    #log stuff
    logger.log_scalar("train/loss", np.mean(t_losses))
    logger.log_scalar("train/time", np.mean(timings))
    logger.log_scalar("val/loss", np.mean(v_losses))
    LOG.info(f"[Epoch {epoch:>2d}]")
    _log_metrics(stage="train", scores=t_scores, logger=logger)
    _log_metrics(stage="val", scores=v_scores, logger=logger)

    return (np.array(t_losses), t_scores), (np.array(v_losses), v_scores)


def _eval_model_epoch(epoch: int,
                      model: nn.Module,
                      criterion: nn.Module,
                      val_dataloader: DataLoader,
                      metrics: Dict[str, Metric],
                      device: torch.device = "cpu") -> Tuple[List[float], List[Any], List[Any]]:
    """Completes a validation loop of the given dataloader.

    Args:
        epoch (int): current epoch
        model (nn.Module): model in training
        criterion (nn.Module): loss instance
        val_dataloader (DataLoader): validation dataloder
        device (torch.device): current device

    Returns:
        Tuple[List[float], Lis[Any], List[Any]]: Returns <list of loss values, list of true data, list of predictions>
    """
    val_tqdm = progressbar(val_dataloader, epoch=epoch, stage="validation")
    v_losses = []

    with torch.no_grad():
        model.eval()
        for val_x, val_y in val_tqdm:
            val_x = val_x.to(device, non_blocking=True)
            val_y = val_y.to(device, non_blocking=True)
            val_preds = model(val_x)

            val_loss = criterion(val_preds, val_y)
            v_losses.append(val_loss.item())
            val_tqdm.set_postfix({"loss": val_loss.item()})

            _update_metrics(metrics, y_true=val_y, y_pred=val_preds)

    return v_losses


def train_model(model: nn.Module,
                criterion: nn.Module,
                optimizer: Optimizer,
                scheduler: nn.Module,
                train_dataloader: DataLoader,
                val_dataloader: DataLoader,
                logger: BaseLogger,
                metrics: Dict[str, Dict[str, Metric]],
                monitor: str,
                device: str = "cpu",
                patience: int = 10,
                max_epochs: int = 100) -> Tuple[int, float, float]:
    """Train a new classifier to discern tiles containing panels and empty tiles.

    Args:
        model (nn.Module): model instance to be trained
        criterion (nn.Module): loss instance for the model
        optimizer (Optimizer): optimizer instance instantiated with model parameters
        scheduler (nn.Module): scheduler instance, already initialized with the given optimizer
        train_dataloader (DataLoader): dataloader for the training set
        val_dataloader (DataLoader): dataloader for the validation set
        logger (BaseLogger): logger instance to store loss and stuff
        device (str, optional): which device to train on. Defaults to "cpu".
        patience (int, optional): patience for early stopping. Defaults to 10.
        max_epochs (int, optional): maximum amount of epochs to train on. Defaults to 100.

    Returns:
        Tuple[int, float, float]: epoch, best loss, best metric. Epoch can be < max_epochs with early stopping
    """
    assert monitor in metrics["val"],\
        f"Monitored metric '{monitor}' not in available metrics ({str(list(metrics.keys()))})"

    best_state_dict = model.state_dict()
    best_metric = 0.0
    best_loss = float("inf")
    patience_counter = 0

    for curr_epoch in range(max_epochs):
        try:
            _, (v_losses, v_scores) = _train_model_epoch(epoch=curr_epoch,
                                                         model=model,
                                                         criterion=criterion,
                                                         optimizer=optimizer,
                                                         scheduler=scheduler,
                                                         train_dataloader=train_dataloader,
                                                         val_dataloader=val_dataloader,
                                                         metrics=metrics,
                                                         logger=logger,
                                                         device=device)
        except KeyboardInterrupt:
            LOG.info("[Epoch %2d] Gracefully stopping training.", curr_epoch)
            break
        # store best loss
        epoch_val_loss = np.mean(v_losses)
        if not best_loss or epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
        # store best metric
        v_metric = v_scores[monitor]
        if not best_metric or v_metric > best_metric:
            best_metric = v_metric
            patience_counter = 0
            best_state_dict = model.state_dict()
        else:
            patience_counter += 1
            LOG.info("[Epoch %2d] Early stopping patience increased to: %d/%d", curr_epoch, patience_counter, patience)
            if patience_counter == patience:
                LOG.info("[Epoch %2d] Early stopping triggered", curr_epoch)
                model.load_state_dict(best_state_dict)
                # stop the cycle
                break

    return curr_epoch, best_loss, best_metric
