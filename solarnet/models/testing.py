from typing import Dict, Iterable

import numpy as np
import torch
from solarnet.logging import BaseLogger
from solarnet.metrics import Metric
from solarnet.utils.common import progressbar
from torch import nn
from torch.utils.data import DataLoader


def test_model(model: nn.Module,
               criterion: nn.Module,
               test_dataloader: DataLoader,
               metrics: Dict[str, Metric],
               logger: BaseLogger,
               logger_exclude: Iterable[str] = ("conf_mat"),
               device: str = "cpu"):
    model.eval()
    test_tqdm = progressbar(test_dataloader, stage="testing")
    scores = dict()
    losses = []
    inputs, y_pred, y_true = [], [], []

    with torch.no_grad():
        for test_x, test_y in test_tqdm:
            test_x = test_x.to(device, non_blocking=True)
            test_y = test_y.to(device, non_blocking=True)
            test_preds = model(test_x)
            test_loss = criterion(test_preds, test_y).item()
            logger.log_scalar("test/loss_iter", test_loss)
            losses.append(test_loss)
            # save predictions to generate plots
            inputs.append(test_x.cpu())
            y_pred.append(torch.argmax(test_preds, dim=1).cpu().numpy())
            y_true.append(test_y.cpu().numpy())
            # update metric states
            for metric in metrics.values():
                metric(test_y, test_preds)
            logger.step()

        # gather and log results
        logger.step(iteration=0)
        logger.log_scalar("test/loss", np.mean(losses))
        for name, metric in metrics.items():
            score: torch.Tensor = metric.compute()
            scores[name] = score
            # confusion matrices and class-wise metrics are not a scalar
            if name not in logger_exclude:
                scalar = score if score.ndim == 0 else score.mean()
                logger.log_scalar(f"test/{name}", scalar.item())

    return losses, scores, (inputs, y_true, y_pred)
