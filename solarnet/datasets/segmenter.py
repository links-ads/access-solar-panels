from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import rasterio as rio
import torch
from torch.utils.data import Dataset
from skimage import measure


def imread(path: Path, channels: int = 3, channels_first: bool = False) -> np.ndarray:
    with rio.open(path) as src:
        img = src.read(list(range(1, channels + 1)))
        if not channels_first and img.ndim == 3:
            img = np.moveaxis(img, 0, -1)
    return img


class MaskedDataset(Dataset):

    def add_mask(self, mask: List[bool]) -> None:
        """Filters out files and masks not required for the current dataset split.add()

        Args:
            mask (List[bool]): list of bollean values, one for each tile, to decide whether to keep it or not.
        """
        assert len(mask) == len(self.solar_files), \
            f"Mask is the wrong size! Expected {len(self.solar_files)}, got {len(mask)}"
        self.solar_files = [x for include, x in zip(mask, self.solar_files) if include]
        self.mask_files = [x for include, x in zip(mask, self.mask_files) if include]


class USGSSegmentationDataset:

    def __init__(self, data_folder: Path, transform: Callable = None, mask: Optional[List[bool]] = None) -> None:
        self.transform = transform

        # We will only segment the images which we know have solar panels in them; the
        # other images should be filtered out by the classifier
        solar_folder = data_folder / 'solar'
        self.solar_files = list((solar_folder / 'org').glob("*.tif"))
        self.mask_files = [solar_folder / 'mask' / f.name for f in self.solar_files]
        assert len(self.solar_files) > 0, "No images found!"
        assert len(self.solar_files) == len(self.mask_files), "Length mismatch between images and masks!"
        if mask is not None:
            self.add_mask(mask)

    def __len__(self) -> int:
        return len(self.solar_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = imread(self.solar_files[index])
        y = imread(self.mask_files[index])
        if self.transform is not None:
            pair = self.transform(image=x, mask=y)
            x = pair.get("image")
            y = pair.get("mask")
        return x, y


class DydasSegmentationDataset(MaskedDataset):

    categories = {0: "background", 1: "mono", 2: "poly"}
    palette = {0: (0, 0, 0), 1: (0, 255, 0), 2: (255, 0, 255)}

    def __init__(self,
                 data_folder: Path,
                 transform: Callable = None,
                 mask: List[bool] = None,
                 channels: int = 4,
                 ignore_index: int = 255) -> None:
        self.transform = transform
        self.channels = channels
        self.ignore_index = ignore_index
        # find images and masks inside the specified folder
        self.solar_files = sorted(list((data_folder / "img_dir").glob("*.tif")))
        self.mask_files = sorted(list((data_folder / "ann_dir").glob("*.tif")))
        # check consistency
        assert len(self.solar_files) == len(self.mask_files), "Images and masks mismatch!"
        for img_path, msk_path in zip(self.solar_files, self.mask_files):
            assert img_path.stem == msk_path.stem, \
                f"Image and mask mismatch: '{img_path.stem}' - '{msk_path.stem}'"
        # mask files if provided
        if mask is not None:
            self.add_mask(mask)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = imread(self.solar_files[index], channels=self.channels)
        y = imread(self.mask_files[index], channels=1).squeeze(-1)
        if self.transform is not None:
            pair = self.transform(image=x, mask=y)
            x = pair.get("image")
            y = pair.get("mask")
        else:
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)
        return x, y.long()

    def __len__(self) -> int:
        return len(self.solar_files)


class DydasInferenceDataset(DydasSegmentationDataset):

    def __init__(self, data_folder: Path, images: List[str], transform: Callable = None, channels: int = 4) -> None:
        image_paths = [data_folder / f"{img}.tif" for img in images]
        for path in image_paths:
            assert path.exists(), f"Image not found: {path}"
        self.image_paths = image_paths
        self.transform = transform
        self.channels = channels

    def __getitem__(self, index: int) -> torch.Tensor:
        img_path = self.image_paths[index]
        profile = {}
        trf = None
        with rio.open(img_path) as src:
            profile = src.profile
            trf = src.transform
            x = src.read().transpose(1, 2, 0)[:, :, :self.channels]
        if self.transform is not None:
            x = self.transform(image=x)["image"]
        else:
            x = torch.from_numpy(x).float()
        return x, img_path.name, profile, trf

    def __len__(self) -> int:
        return len(self.image_paths)


class BinaryDydasSegDataset(DydasSegmentationDataset):

    categories = {0: "background", 1: "panel"}
    palette = {0: (0, 0, 0), 1: (255, 255, 255)}

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = imread(self.solar_files[index], channels=self.channels)
        y = imread(self.mask_files[index], channels=1)
        # save image as png for debug
        # no clue why sometimes it has (X, X, 1) instead of (X, X)
        if y.ndim > 2:
            y = y.squeeze(axis=-1)

        ignored = y == self.ignore_index
        y = (y > 0).astype(np.float32)
        y[ignored] = self.ignore_index

        if self.transform is not None:
            pair = self.transform(image=x, mask=y)
            x = pair.get("image")
            y = pair.get("mask")
        return x, y


class SSLDydasDataset(DydasSegmentationDataset):

    def __init__(self,
                 labeled_folder: Path,
                 unlabeled_folder: Path,
                 transform: Callable = None,
                 unlabeled_transform: Callable = None,
                 mask: List[bool] = None,
                 channels: int = 4,
                 ignore_index: int = 255) -> None:
        super().__init__(labeled_folder, transform, mask, channels, ignore_index)
        self.unlabeled_transform = unlabeled_transform
        self.unlabeled_files = sorted(list((unlabeled_folder).glob("*.tif")))
        assert len(self.unlabeled_files) > 0, "No unlabeled images found!"

    def __len__(self) -> int:
        return len(self.solar_files) * len(self.unlabeled_files)

    def generate_class_mask(self, label: np.ndarray, classes: np.ndarray) -> np.ndarray:
        # make the list a 3D tensor with size 1 x 1 x N
        classes = np.expand_dims(classes, axis=(0, 1))
        # make the label a 3D tensor with size H x W x 1
        label = np.expand_dims(label, axis=-1)
        class_mask = (label == classes).sum(-1)
        return torch.from_numpy(class_mask)

    def get_mixing_mask(self, label: np.ndarray):
        # for each label, do the following:
        # - isolate single component (sort of instance masks)
        # - filter away any 255 from the label by creating a mapping <instance ID: class index>
        label = label.squeeze(0)
        nobkg = label.numpy().copy()
        nobkg[label == 0] = 255
        instance_mask = measure.label(nobkg, background=255)
        label_instances = np.unique(instance_mask, return_counts=False)
        label_instances = label_instances[label_instances != 0]
        # for each instance, get the class from the first pixel (assuming thery are all the same)
        # indexing is done on [1, 512, 512] => [batch][HxW][first element]
        # convert array of instances into classes, eg:
        # [1, 2, 3, 4, 5, 6]
        # [3, 3, 2, 2, 0, 0]
        num_patches = len(label_instances)
        num_samples = int((num_patches + num_patches % 2) / 2)
        chosen = np.random.choice(label_instances, size=num_samples, replace=False)
        return self.generate_class_mask(label=instance_mask, classes=chosen)

    def mix_images(self, x: torch.Tensor, y: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self.get_mixing_mask(y)
        # mix images and labels
        mixed_x = x * mask + u * (1 - mask)
        mixed_y = y * mask.long()
        return mixed_x, mixed_y

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        labeled_index = index // len(self.unlabeled_files)
        unlabeled_index = index % len(self.unlabeled_files)
        x, y = super().__getitem__(labeled_index)
        u = imread(self.unlabeled_files[unlabeled_index], channels=self.channels)
        if self.unlabeled_transform is not None:
            u = self.unlabeled_transform(image=u)["image"]
        else:
            u = torch.from_numpy(u)
        xm, ym = self.mix_images(x, y, u)
        return x, y, xm, ym
