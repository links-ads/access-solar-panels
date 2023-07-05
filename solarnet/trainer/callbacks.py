import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from solarnet.trainer.base import Trainer

LOG = logging.getLogger(__name__)


class BaseCallback:

    def __init__(self, call_every: int = 1, call_once: int = None) -> None:
        assert call_every is not None or call_once is not None, "Specify at least one between call_every and call_once"
        if call_every is not None:
            assert call_every > 0, "call_every should be >= 1"
        if call_once is not None:
            assert call_once >= 0, "call_once should be >= 0"
        self.call_every = call_every
        self.call_once = call_once
        self.expired = False

    def __call__(self, trainer: "Trainer", *args: Any, **kwds: Any) -> Any:
        # early exit for one-time callbacks
        if self.expired:
            return
        if self.call_once is not None and self.call_once == trainer.current_epoch:
            data = self.call(trainer, *args, **kwds)
            self.expired = True
            return data
        if self.call_every is not None:
            if (trainer.current_epoch % self.call_every) == 0:
                return self.call(trainer, *args, **kwds)

    def setup(self, trainer: "Trainer"):
        pass

    def call(self, trainer: "Trainer", *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Callback not implemented!")

    def dispose(self, trainer: "Trainer"):
        pass


class EarlyStoppingCriterion(Enum):
    minimum = torch.lt
    maximum = torch.gt


class EarlyStopping(BaseCallback):

    criteria = {"min": torch.lt, "max": torch.gt}

    def __init__(self,
                 call_every: int,
                 metric: str,
                 criterion: EarlyStoppingCriterion.minimum,
                 patience: int = 10) -> None:
        super().__init__(call_every=call_every)
        self.metric = metric
        self.criterion = criterion.value
        self.patience = patience
        self.patience_counter = None

    def setup(self, trainer: "Trainer"):
        metrics = trainer.metrics["val"]
        if self.metric not in metrics:
            raise ValueError(f"Monitored metric '{self.metric}' not in validation metrics: {list(metrics.keys())}")
        self.patience_counter = 0

    def call(self, trainer: "Trainer", *args: Any, **kwargs: Any) -> Any:
        current_score = trainer.current_scores["val"][self.metric]
        if trainer.best_score is None or self.criterion(current_score, trainer.best_score):
            self.patience_counter = 0
            trainer.best_score = current_score
            trainer.best_epoch = trainer.current_epoch
            trainer.best_state_dict = trainer.model.state_dict()
        else:
            self.patience_counter += 1
            LOG.info("[Epoch %2d] Early stopping patience increased to: %d/%d", trainer.current_epoch,
                     self.patience_counter, self.patience)
            if self.patience_counter == self.patience:
                LOG.info("[Epoch %2d] Early stopping triggered", trainer.current_epoch)
                trainer.model.load_state_dict(trainer.best_state_dict)
                # stop iterating with an exceptionm it will be caught by the training loop
                raise KeyboardInterrupt

    def dispose(self, trainer: "Trainer"):
        self.patience_counter = 0
