from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


class USGSClassifierDataset:

    def __init__(self, data_folder: Path, transform: Callable = None, mask: Optional[List[bool]] = None) -> None:
        self.transform = transform
        solar_files = list((data_folder / "solar" / "org").glob("*.tif"))
        empty_files = list((data_folder / "empty" / "org").glob("*.tif"))
        assert len(solar_files) > 0, "No images containing solar panels found!"
        assert len(empty_files) > 0, "No images without solar panels found!"
        # set 0 if image belongs to empty files, else 1, this is the clf ground truth
        self.y = torch.as_tensor([1 for _ in solar_files] + [0 for _ in empty_files]).float()
        self.x_files = solar_files + empty_files
        if mask is not None:
            self.add_mask(mask)

    def add_mask(self, mask: List[bool]) -> None:
        """Add a mask to the data
        """
        assert len(mask) == len(self.x_files), \
            f"Mask is the wrong size! Expected {len(self.x_files)}, got {len(mask)}"
        self.y = torch.as_tensor(self.y.cpu().numpy()[mask])
        self.x_files = [x for include, x in zip(mask, self.x_files) if include]

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.y[index]
        # read channels last
        x = np.array(Image.open(self.x_files[index]))
        if self.transform:
            x = self.transform(image=x)["image"]
        return x, y
