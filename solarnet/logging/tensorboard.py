from pathlib import Path
from typing import Union

import numpy as np
import torch
from solarnet.logging import BaseLogger
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger(BaseLogger):

    def __init__(self,
                 log_folder: Path = Path("logs"),
                 filename_suffix: str = "",
                 current_step: int = 0,
                 comment: str = "") -> None:
        super().__init__()
        self.log = SummaryWriter(log_dir=log_folder, filename_suffix=filename_suffix, comment=comment)
        self.current_step = current_step

    def step(self, iteration: int = None) -> None:
        if not iteration:
            self.current_step += 1
        else:
            self.current_step = iteration

    def log_model(self,
                  model: nn.Module,
                  input_size: Union[tuple, list] = (1, 4, 256, 256),
                  device: str = "cpu") -> None:
        if isinstance(input_size, list):
            sample_input = [torch.rand(size, device=device) for size in input_size]
        else:
            sample_input = torch.rand(input_size, device=device)
        self.log.add_graph(model, input_to_model=sample_input)

    def log_scalar(self, name: str, value: float, **kwargs) -> None:
        self.log.add_scalar(name, value, global_step=self.current_step, **kwargs)

    def log_image(self, name: str, image: np.ndarray, **kwargs) -> None:
        self.log.add_image(name, image, global_step=self.current_step, **kwargs)
