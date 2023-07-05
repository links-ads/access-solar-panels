import torch
from solarnet.cli import Initializer
from solarnet.utils.ml import one_hot_batch
from torch import nn
from torch.nn import functional as func


class FocalTverskyLoss(nn.Module):
    """Custom implementation
    """

    def __init__(self,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 gamma: float = 2.0,
                 ignore_index: int = 255,
                 weights: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weights = weights
        self.ignore_index = ignore_index

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = preds.size(1)
        onehot = one_hot_batch(targets, num_classes=num_classes, ignore_index=self.ignore_index)
        onehot = onehot.float().to(preds.device)
        probs = func.softmax(preds, dim=1)

        # sum over batch, height width, leave classes (dim 1)
        dims = (0, 2, 3)
        tp = (onehot * probs).sum(dim=dims)
        fp = (probs * (1 - onehot)).sum(dim=dims)
        fn = ((1 - probs) * onehot).sum(dim=dims)

        index = self.weights * (tp / (tp + self.alpha * fp + self.beta * fn))
        return (1 - index.mean())**self.gamma


class CombinedLoss(nn.Module):

    def __init__(self, criterion_a: Initializer, criterion_b: Initializer, alpha: float = 0.5):
        super().__init__()
        self.criterion_a = criterion_a()
        self.criterion_b = criterion_b()
        self.alpha = alpha

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_a = self.criterion_a(preds, targets)
        loss_b = self.criterion_b(preds, targets)
        return self.alpha * loss_a + (1 - self.alpha) * loss_b
