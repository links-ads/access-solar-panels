from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Tuple, Type

import timm
import torch
from solarnet.models.modules import (SSMA, ASPPModule, DecoderAdaptNet, DecoderUNet, DecoderV3, DecoderV3Plus,
                                     EfficientASPP, RotationHead)
from solarnet.utils.ml import expand_input, initialize_weights
from timm.models.features import FeatureListNet
from torch import nn


class Segmenter(nn.Module, ABC):

    def __init__(self,
                 encoder: str,
                 input_channels: int = 4,
                 pretrained: bool = False,
                 checkpoint: Path = None,
                 output_stride: int = 16,
                 layers: Tuple[int, ...] = (1, 4),
                 input_layer: str = None,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d) -> None:
        super().__init__()
        ckpt_path = str(checkpoint) if checkpoint else None
        self.encoder: FeatureListNet = timm.create_model(encoder,
                                                         pretrained=pretrained,
                                                         checkpoint_path=ckpt_path,
                                                         features_only=True,
                                                         output_stride=output_stride,
                                                         out_indices=layers,
                                                         norm_layer=norm_layer)
        # get the first set of 7x7 layers in the weights with shape [64, 3, 7, 7]
        if input_channels > 3:
            self.encoder = expand_input(self.encoder, input_layer=input_layer, copy_channel=0)

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        return self.encoder.parameters()

    @abstractmethod
    def decoder_parameters(self) -> Iterator[nn.Parameter]:
        raise NotImplementedError("Implement in a subclass")


class SSLSegmenter(nn.Module, ABC):

    def __init__(self,
                 encoder: str,
                 rgb_pretrained: bool = False,
                 rgb_checkpoint: Path = None,
                 ir_pretrained: bool = False,
                 ir_checkpoint: Path = None,
                 output_stride: int = 16,
                 layers: Tuple[int, ...] = (1, 4),
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d) -> None:
        super().__init__()
        ckpt_path_rgb = str(rgb_checkpoint) if rgb_checkpoint else None
        ckpt_path_ir = str(ir_checkpoint) if ir_checkpoint else None

        self.encoder_rgb: FeatureListNet = timm.create_model(encoder,
                                                             pretrained=rgb_pretrained,
                                                             checkpoint_path=ckpt_path_rgb,
                                                             features_only=True,
                                                             output_stride=output_stride,
                                                             out_indices=layers,
                                                             norm_layer=norm_layer)
        self.encoder_ir: FeatureListNet = timm.create_model(encoder,
                                                            in_chans=1,
                                                            pretrained=ir_pretrained,
                                                            checkpoint_path=ckpt_path_ir,
                                                            features_only=True,
                                                            output_stride=output_stride,
                                                            out_indices=layers,
                                                            norm_layer=norm_layer)

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        for iterator in (self.encoder_rgb.parameters(), self.encoder_ir.parameters()):
            for param in iterator:
                yield param

    @abstractmethod
    def decoder_parameters(self) -> Iterator[nn.Parameter]:
        raise NotImplementedError("Implement in a subclass")


class DeepLabV3(Segmenter):

    def __init__(self,
                 encoder: str = "resnet50",
                 encoder_weights: str = None,
                 input_channels: int = 3,
                 input_size: int = 512,
                 output_stride: int = 16,
                 num_classes: int = 1,
                 pretrained: bool = False,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d) -> None:
        assert output_stride in (8, 16), f"Invalid output stride: '{output_stride}'"
        super().__init__(encoder=encoder,
                         input_channels=input_channels,
                         pretrained=pretrained,
                         checkpoint=encoder_weights,
                         output_stride=output_stride,
                         norm_layer=norm_layer,
                         layers=(4,))
        # only one layer
        channels = self.encoder.feature_info.channels()[0]
        reduction = self.encoder.feature_info.reduction()[0]
        aspp_channels = 256
        self.aspp = ASPPModule(in_size=int(input_size / reduction),
                               in_channels=channels,
                               output_stride=output_stride,
                               out_channels=aspp_channels,
                               batch_norm=norm_layer)
        self.decoder = DecoderV3(in_channels=aspp_channels,
                                 output_stride=output_stride,
                                 output_channels=num_classes,
                                 batch_norm=norm_layer)
        self.aspp.apply(initialize_weights)
        self.decoder.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.aspp(*x)
        return self.decoder(x)

    def decoder_parameters(self) -> Iterator[nn.Parameter]:
        for iterator in (self.app.parameters(), self.decoder.parameters()):
            for param in iterator:
                yield param


class DeepLabV3Plus(Segmenter):

    def __init__(self,
                 encoder: str = "resnet50",
                 encoder_weights: str = None,
                 input_channels: int = 3,
                 aspp_channels: int = 256,
                 input_size: int = 512,
                 output_stride: int = 16,
                 num_classes: int = 1,
                 pretrained: bool = False,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d) -> None:
        assert output_stride in (8, 16), f"Invalid output stride: '{output_stride}'"
        super().__init__(encoder=encoder,
                         input_channels=input_channels,
                         pretrained=pretrained,
                         checkpoint=encoder_weights,
                         output_stride=output_stride,
                         norm_layer=norm_layer,
                         layers=(1, 4))
        low_channels, high_channels = self.encoder.feature_info.channels()
        _, high_reduction = self.encoder.feature_info.reduction()
        self.aspp = ASPPModule(in_size=int(input_size / high_reduction),
                               in_channels=high_channels,
                               output_stride=output_stride,
                               out_channels=aspp_channels,
                               batch_norm=norm_layer)
        self.decoder = DecoderV3Plus(skip_channels=low_channels,
                                     aspp_channels=aspp_channels,
                                     output_stride=output_stride,
                                     output_channels=num_classes,
                                     batch_norm=norm_layer)
        self.aspp.apply(initialize_weights)
        self.decoder.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip, x = self.encoder(x)
        x = self.aspp(x)
        return self.decoder(x, skip)

    def decoder_parameters(self) -> Iterator[nn.Parameter]:
        for iterator in (self.aspp.parameters(), self.decoder.parameters()):
            for param in iterator:
                yield param


class AdaptNet(Segmenter):

    def __init__(self,
                 encoder: str = "resnet50",
                 encoder_weights: str = None,
                 input_channels: int = 3,
                 aspp_channels: int = 256,
                 input_size: int = 512,
                 output_stride: int = 16,
                 num_classes: int = 1,
                 pretrained: bool = False,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d) -> None:
        super().__init__(encoder=encoder,
                         input_channels=input_channels,
                         pretrained=pretrained,
                         checkpoint=encoder_weights,
                         output_stride=output_stride,
                         norm_layer=norm_layer,
                         layers=(0, 1, 4))
        # extract info about encoder layers, add 1 to reductions to account for input
        low, mid, high = self.encoder.feature_info.channels()
        reductions = [1] + self.encoder.feature_info.reduction()

        self.aspp = EfficientASPP(in_size=int(input_size / reductions[-1]),
                                  in_channels=high,
                                  output_stride=output_stride,
                                  out_channels=aspp_channels,
                                  batch_norm=norm_layer)
        # compute how much should we upsample, from lowest to input size (thanks to the [1] from before)
        upsamples = [reductions[i + 1] // reductions[i] for i in range(len(reductions) - 1)]
        self.decoder = DecoderAdaptNet(skip_channels=(low, mid),
                                       skip_upsamples=upsamples[:-1],
                                       aspp_upsample=upsamples[-1],
                                       aspp_channels=aspp_channels,
                                       output_channels=num_classes,
                                       batch_norm=norm_layer)
        self.decoder.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low, mid, x = self.encoder(x)
        x = self.aspp(x)
        return self.decoder(x, mid, low)

    def decoder_parameters(self) -> Iterator[nn.Parameter]:
        for iterator in (self.aspp.parameters(), self.decoder.parameters()):
            for param in iterator:
                yield param


class UNet(Segmenter):

    def __init__(self,
                 encoder: str = "resnet50",
                 encoder_weights: str = None,
                 input_channels: int = 3,
                 output_stride: int = 16,
                 num_classes: int = 1,
                 pretrained: bool = False,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d) -> None:
        super().__init__(encoder,
                         input_channels=input_channels,
                         pretrained=pretrained,
                         checkpoint=encoder_weights,
                         output_stride=output_stride,
                         layers=tuple(i for i in range(5)),
                         norm_layer=norm_layer)
        self.decoder = DecoderUNet(feature_channels=self.encoder.feature_info.channels(),
                                   feature_reductions=self.encoder.feature_info.reduction(),
                                   output_channels=num_classes,
                                   norm_layer=norm_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.decoder(*features)

    def decoder_parameters(self) -> Iterator[nn.Parameter]:
        return self.decoder.parameters()


class SSLUnet(SSLSegmenter):

    def __init__(self,
                 encoder: str = "resnet50",
                 rgb_pretrained: bool = False,
                 rgb_checkpoint: Path = None,
                 ir_pretrained: bool = False,
                 ir_checkpoint: Path = None,
                 output_stride: int = 16,
                 num_classes: int = 1,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d) -> None:
        super().__init__(encoder,
                         rgb_pretrained=rgb_pretrained,
                         rgb_checkpoint=rgb_checkpoint,
                         ir_pretrained=ir_pretrained,
                         ir_checkpoint=ir_checkpoint,
                         output_stride=output_stride,
                         layers=tuple(i for i in range(5)),
                         norm_layer=norm_layer)
        self.ssmas = nn.ModuleList()
        for rgb_chs, ir_chs in zip(self.encoder_rgb.feature_info.channels(), self.encoder_ir.feature_info.channels()):
            self.ssmas.append(SSMA(rgb_channels=rgb_chs, ir_channels=ir_chs, norm_layer=norm_layer))
        self.decoder = DecoderUNet(feature_channels=self.encoder_rgb.feature_info.channels(),
                                   feature_reductions=self.encoder_rgb.feature_info.reduction(),
                                   output_channels=num_classes,
                                   norm_layer=norm_layer)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        rgb_features = self.encoder_rgb(rgb)
        ir_features = self.encoder_ir(ir)
        out_features = []
        for i in range(len(rgb_features)):
            out_features.append(self.ssmas[i](rgb_features[i], ir_features[i]))
        return self.decoder(*out_features)

    def decoder_parameters(self) -> Iterator[nn.Parameter]:
        for iterator in (self.ssmas.parameters(), self.decoder.parameters()):
            for param in iterator:
                yield param


class SSLMixedModel(SSLUnet):

    def __init__(self,
                 encoder: str = "resnet50",
                 rgb_pretrained: bool = False,
                 rgb_checkpoint: Path = None,
                 ir_pretrained: bool = False,
                 ir_checkpoint: Path = None,
                 output_stride: int = 16,
                 segment_classes: int = 1,
                 pretext_classes: int = 4,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d) -> None:
        super().__init__(encoder=encoder,
                         rgb_pretrained=rgb_pretrained,
                         rgb_checkpoint=rgb_checkpoint,
                         ir_pretrained=ir_pretrained,
                         ir_checkpoint=ir_checkpoint,
                         output_stride=output_stride,
                         num_classes=segment_classes,
                         norm_layer=norm_layer)
        channels = self.encoder_ir.feature_info.channels()[-1]
        self.head = RotationHead(input_dim=channels, hidden_dim=128, num_classes=pretext_classes, norm_layer=norm_layer)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        rgb_features = self.encoder_rgb(rgb)
        ir_features = self.encoder_ir(ir)
        out_features = [ssma(rgb, ir) for ssma, rgb, ir in zip(self.ssmas, rgb_features, ir_features)]

        ssl_logits = self.head(out_features[-1])
        seg_logits = self.decoder(*out_features)
        return ssl_logits, seg_logits
