from abc import ABC, abstractmethod

import numpy as np
from torch import nn


class BaseLogger(ABC):

    @abstractmethod
    def step(self, iteration: int = None) -> None:
        raise NotImplementedError()

    @abstractmethod
    def log_model(self, model: nn.Module, input_size: tuple, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def log_scalar(self, name: str, value: float, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def log_image(self, name: str, image: np.ndarray, **kwargs) -> None:
        raise NotImplementedError()
