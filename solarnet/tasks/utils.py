import logging
from typing import Optional, Type

import numpy as np
import torch
import torch.nn as nn

from solarnet.config import (Models, SegmenterTrainSettings, SSLSegmenterTrainSettings)
from solarnet.models.segmenter import (AdaptNet, DeepLabV3Plus, Segmenter, SSLMixedModel, SSLUnet, UNet)

LOG = logging.getLogger(__name__)


def _load_class_weights(class_weights: Optional[str] = None, multiclass: bool = True):
    # load class weights, if any
    loss_args = dict()
    if multiclass and class_weights is not None:
        weights = np.load(class_weights).astype(np.float32)
        loss_args["weight"] = torch.from_numpy(weights)
        LOG.info("Using class weights: %s", str(weights))
    return loss_args


def _load_model(config: SegmenterTrainSettings,
                num_classes: int,
                norm_layer: Type[nn.Module] = nn.BatchNorm2d) -> Segmenter:
    if config.model == Models.deeplabv3:
        model = DeepLabV3Plus(encoder=config.encoder,
                              input_channels=config.input_channels,
                              pretrained=config.enc_pretrained,
                              encoder_weights=config.enc_weights,
                              input_size=config.image_size,
                              num_classes=num_classes,
                              norm_layer=norm_layer)
    elif config.model == Models.adaptnet:
        model = AdaptNet(encoder=config.encoder,
                         input_channels=config.input_channels,
                         pretrained=config.enc_pretrained,
                         encoder_weights=config.enc_weights,
                         input_size=config.image_size,
                         num_classes=num_classes,
                         norm_layer=norm_layer)
    elif config.model == Models.unet:
        model = UNet(encoder=config.encoder,
                     input_channels=config.input_channels,
                     pretrained=config.enc_pretrained,
                     encoder_weights=config.enc_weights,
                     num_classes=num_classes,
                     norm_layer=norm_layer)
    return model


def _load_model_ssl(config: SSLSegmenterTrainSettings,
                    num_classes: int,
                    stage: str = "train",
                    norm_layer: Type[nn.Module] = nn.BatchNorm2d) -> SSLMixedModel:
    if config.model == Models.unet:
        assert stage in ("train", "eval"), f"Unrecognized stage: '{stage}'"
        if stage == "train":
            model = SSLMixedModel(
                encoder=config.encoder_rgb,
                rgb_pretrained=config.pretrained_rgb,
                rgb_checkpoint=config.weights_rgb,
                ir_pretrained=False,
                segment_classes=num_classes,
                pretext_classes=4,  # TODO: remove hardcoded param with SSL config
                norm_layer=norm_layer)
        else:
            model = SSLUnet(encoder=config.encoder_rgb, num_classes=num_classes, norm_layer=norm_layer)
        return model
    else:
        raise NotImplementedError("SSL currently supports UNets only")
