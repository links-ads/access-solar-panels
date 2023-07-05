from solarnet.tasks.processing import make_masks, split_images, compute_stats
from solarnet.tasks.training import train_classifier, train_segmenter
from solarnet.tasks.testing import test_segmenter, infer_large_tiles
from solarnet.tasks.ssl import train_segmenter_ssl

__all__ = [
    "make_masks",
    "split_images",
    "compute_stats",
    "train_classifier",
    "train_segmenter",
    "test_segmenter",
    "infer_large_tiles",
    "train_segmenter_ssl",
]
