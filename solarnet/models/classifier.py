from pathlib import Path

import timm
import torch
from timm.models.layers import create_classifier
from torch import nn


class Classifier(nn.Module):
    """A ResNet34 Model

    Attributes:
        imagenet_base: boolean, default: True
            Whether or not to load weights pretrained on imagenet
    """

    def __init__(self,
                 backbone: str = "resnet50",
                 pretrained: bool = True,
                 from_checkpoint: Path = None,
                 num_classes: int = 1,
                 global_pool: str = "avg") -> None:
        super().__init__()
        ckpt_path = str(from_checkpoint) if from_checkpoint else ""
        self.num_classes = num_classes
        self.backbone = timm.create_model(model_name=backbone,
                                          pretrained=pretrained,
                                          checkpoint_path=ckpt_path,
                                          num_classes=0,
                                          global_pool='')
        self.avgpool, self.fc = create_classifier(self.backbone.num_features, num_classes, pool_type=global_pool)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        return self.fc(x)
