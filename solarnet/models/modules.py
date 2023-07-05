from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple, Type

import torch
from torch import nn


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling module: this block is responsible for the multi-scale feature extraction,
    using multiple parallel convolutional blocks (conv, bn, relu) with different dilations.
    The four feature groups are then recombined into a single tensor together with an upscaled average pooling
    (that contrasts information loss), then again processed by a 1x1 convolution + dropout
    """

    def __init__(self,
                 in_size: int = 32,
                 in_channels: int = 2048,
                 output_stride: int = 16,
                 out_channels: int = 256,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d):
        """Creates a new Atrous spatial Pyramid Pooling block. This module is responsible
        for the extraction of features at different scales from the input tensor (which is
        an encoder version of the image with high depth and low height/width).
        The module combines these multi-scale features into a single tensor via 1x convolutions

        Args:
            in_size (int, optional): Size of the input tensor, defaults to 32 for the last layer of ResNet50/101.
            in_channels (int, optional): Channels of the input tensor, defaults to 2048 for ResNet50/101.
            dilations (Tuple[int, int, int, int], optional): dilations, depending on stride. Defaults to (1, 6, 12, 18).
            out_channels (int, optional): Number of output channels. Defaults to 256.
            batch_norm (Type[nn.Module], optional): batch normalization layer. Defaults to nn.BatchNorm2d.
        """
        super().__init__()
        dil_factor = int(output_stride // 16)  # equals 1 or 2 if os = 8
        dilations = tuple(v * dil_factor for v in (1, 6, 12, 18))
        self.aspp1 = self.aspp_block(in_channels, 256, 1, 0, dilations[0], batch_norm=batch_norm)
        self.aspp2 = self.aspp_block(in_channels, 256, 3, dilations[1], dilations[1], batch_norm=batch_norm)
        self.aspp3 = self.aspp_block(in_channels, 256, 3, dilations[2], dilations[2], batch_norm=batch_norm)
        self.aspp4 = self.aspp_block(in_channels, 256, 3, dilations[3], dilations[3], batch_norm=batch_norm)
        # this is redoncolous, but it's described in the paper: bring it down to 1x1 tensor and upscale, yapf: disable
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                     nn.Conv2d(in_channels,256,kernel_size=1, bias=False),
                                     batch_norm(256),
                                     nn.ReLU(inplace=True),
                                     nn.Upsample((in_size, in_size), mode="bilinear", align_corners=True))
        self.merge = self.aspp_block(256 * 5, out_channels, kernel=1, padding=0, dilation=1, batch_norm=batch_norm)
        self.dropout = nn.Dropout(p=0.5)
        # yapf: enable

    def aspp_block(self, in_channels: int, out_channels: int, kernel: int, padding: int, dilation: int,
                   batch_norm: Type[nn.Module]) -> nn.Sequential:
        """Creates a basic ASPP block, a sequential module with convolution, batch normalization and relu activation.
        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels (usually fixed to 256)
        :type out_channels: int
        :param kernel: kernel size for the convolution (usually 3)
        :type kernel: int
        :param padding: convolution padding, usually equal to the dilation, unless no dilation is applied
        :type padding: int
        :param dilation: dilation for the atrous convolution, depends on ASPPVariant
        :type dilation: int
        :param batch_norm: batch normalization class yet to be instantiated
        :type batch_norm: Type[nn.Module]
        :return: sequential block representing an ASPP component
        :rtype: nn.Sequential
        """
        module = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel,
                      stride=1,
                      padding=padding,
                      dilation=dilation,
                      bias=False), batch_norm(out_channels), nn.ReLU(inplace=True))
        return module

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Computes a forward pass on the ASPP module.
        The same input is processed five times with different dilations. Output sizes are the same,
        except for the pooled layer, which requires an upscaling.
        :param batch: input tensor with dimensions [batch, channels, height, width]
        :type batch: torch.Tensor
        :return: output tensor with dimensions [batch, 256, height, width]
        :rtype: torch.Tensor
        """
        x1 = self.aspp1(batch)
        x2 = self.aspp2(batch)
        x3 = self.aspp3(batch)
        x4 = self.aspp4(batch)
        x5 = self.avgpool(batch)
        x5 = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.merge(x5)
        return self.dropout(x)


class EfficientASPP(ASPPModule):
    """Variant of the standard ASPP block, using a bottleneck approach similar to the ResNet one.
    The aim of the bottleneck is to reduce the number of parameters, while increasing the performance.
    Implementation follows: https://doi.org/10.1007/s11263-019-01188-y
    """

    def __init__(self,
                 in_size: int = 32,
                 in_channels: int = 2048,
                 output_stride: int = 16,
                 out_channels: int = 256,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d):
        """Creates a new eASPP block.

        Args:
            in_size (int, optional): spatial input dimension of the encoder (supposed squared). Defaults to 32.
            in_channels (int, optional): number of input channels from the decoder. Defaults to 2048.
            output_stride (int, optional): output stride from the decoder. Defaults to 16.
            out_channels (int, optional): number of output channels of the ASPP. Defaults to 256.
            batch_norm (Type[nn.Module], optional): batch normalization layer. Defaults to nn.BatchNorm2d.
        """
        super(ASPPModule, self).__init__()
        dil_factor = int(output_stride // 16)  # equals 1 or 2 if os = 8
        dilations = tuple(v * dil_factor for v in (1, 6, 12, 18))
        self.aspp1 = self.aspp_block(in_channels, 256, 1, 0, dilations[0], batch_norm=batch_norm)
        self.aspp2 = self.assp_bottleneck(in_channels, 256, 3, dilations[1], dilations[1], batch_norm=batch_norm)
        self.aspp3 = self.assp_bottleneck(in_channels, 256, 3, dilations[2], dilations[2], batch_norm=batch_norm)
        self.aspp4 = self.assp_bottleneck(in_channels, 256, 3, dilations[3], dilations[3], batch_norm=batch_norm)
        # this is redoncolous, but it's described in the paper: bring it down to 1x1 tensor and upscale, yapf: disable
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                     nn.Conv2d(in_channels,256,kernel_size=1, bias=False),
                                     batch_norm(256),
                                     nn.ReLU(inplace=True),
                                     nn.Upsample((in_size, in_size), mode="bilinear", align_corners=True))
        self.merge = self.aspp_block(256 * 5, out_channels, kernel=1, padding=0, dilation=1, batch_norm=batch_norm)
        self.dropout = nn.Dropout(p=0.5)
        # yapf: enable

    def assp_bottleneck(self, in_channels: int, out_channels: int, kernel: int, padding: int, dilation: int,
                        batch_norm: Type[nn.Module]) -> nn.Sequential:
        """Creates a new ASPP bottleneck branch, consisting of one 1x1 convolution to reduce feature channels,
        followed by two 3x3 with increased dilation and lower channels, then again a 1x1 convolution to restore
        the number of output channels.

        Args:
            in_channels (int): number of input channels, depends on the encoder
            out_channels (int): number of output channels, usually hardcoded to 256
            kernel (int): kernel size for the middle convolutions, usually 3x3
            padding (int): padding for the middle convolutions, varies with the branch
            dilation (int): dilation for the middle convolutions, varies with the branch
            batch_norm (Type[nn.Module]): batch normalization module

        Returns:
            nn.Sequential: sequential module representing the bottleneck ASPP branch
        """
        mid_channels = out_channels // 4
        modules = list()
        modules.extend(list(self.aspp_block(in_channels, mid_channels, 1, 0, 1, batch_norm)))
        modules.extend(list(self.aspp_block(mid_channels, mid_channels, kernel, padding, dilation, batch_norm)))
        modules.extend(list(self.aspp_block(mid_channels, mid_channels, kernel, padding, dilation, batch_norm)))
        modules.extend(list(self.aspp_block(mid_channels, out_channels, 1, 0, 1, batch_norm)))
        return nn.Sequential(*modules)


class Decoder(nn.Module, ABC):
    """Simple abstract class to define a decoder.
    """


class DecoderV3(Decoder):
    """Decoder for DeepLabV3, consisting of a double convolution and a direct 16X upsampling.
    This is clearly not the best for performance, but, if memory is a problem, this can save a little space.
    """

    def __init__(self,
                 in_channels: int = 256,
                 output_stride: int = 16,
                 output_channels: int = 1,
                 dropout: float = 0.1,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d):
        """Decoder output for the simpler DeepLabV3: this module simply processes the ASPP output
        and upscales it to the input size.The 3x3 convolution and the dropout do not appear in the paper,
        but they are implemented in the official release.
        :param output_stride: scaling factor of the backbone, defaults to 16
        :type output_stride: int, optional
        :param output_channels: number of classes in the output mask, defaults to 1
        :type output_channels: int, optional
        :param dropout: dropout probability before the final convolution, defaults to 0.1
        :type dropout: float, optional
        :param batch_norm: batch normalization class, defaults to nn.BatchNorm2d
        :type batch_norm: Type[nn.Module], optional
        """
        # yapf: disable
        super(DecoderV3, self).__init__(nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
                                        batch_norm(256),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=dropout),
                                        nn.Conv2d(256, output_channels, kernel_size=1),
                                        nn.Upsample(scale_factor=output_stride, mode="bilinear", align_corners=True))
        # yapf: enable


class DecoderV3Plus(Decoder):
    """DeepLabV3+ decoder branch, with a skip branch embedding low level
    features (higher resolution) into the highly dimensional output. This typically
    produces much better results than a naive 16x upsampling.
    Original paper: https://arxiv.org/abs/1802.02611
    """

    def __init__(self,
                 skip_channels: int,
                 aspp_channels: int = 256,
                 output_stride: int = 16,
                 output_channels: int = 1,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d):
        """Returns a new Decoder for DeepLabV3+.
        The upsampling is divided into two parts: a fixed 4x from 128 to 512, and a 2x or 4x
        from 32 or 64 (when input=512x512) to 128, depending on the output stride.
        :param low_level_channels: how many channels on the lo-level skip branch
        :type low_level_channels: int
        :param output_stride: downscaling factor of the backbone, defaults to 16
        :type output_stride: int, optional
        :param output_channels: how many outputs, defaults to 1
        :type output_channels: int, optional
        :param batch_norm: batch normalization module, defaults to nn.BatchNorm2d
        :type batch_norm: Type[nn.Module], optional
        """
        super().__init__()
        low_up_factor = 4
        high_up_factor = output_stride / low_up_factor
        self.low_level = nn.Sequential(nn.Conv2d(skip_channels, 48, 1, bias=False), batch_norm(48),
                                       nn.ReLU(inplace=True))
        self.upsample = nn.Upsample(scale_factor=high_up_factor, mode="bilinear", align_corners=True)

        # Table 2, best performance with two 3x3 convs, yapf: disable
        self.output = nn.Sequential(nn.Conv2d(48 + aspp_channels, 256, 3, stride=1, padding=1, bias=False),
                                    batch_norm(256),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.5),
                                    nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
                                    batch_norm(256),
                                    nn.ReLU(inplace=True), nn.Dropout(0.1),
                                    nn.Conv2d(256, output_channels, 1, stride=1),
                                    nn.Upsample(scale_factor=low_up_factor, mode="bilinear", align_corners=True),
                                    nn.Dropout(0.1))
        # yapf: enable
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass on the decoder. Low-level features 'skip' are processed and merged
        with the upsampled high-level features 'x'. The output then restores the tensor
        to the original height and width.
        :param x: high-level features, [batch, 2048, X, X], where X = input size / output stride
        :type x: torch.Tensor
        :param skip: low-level features, [batch, Y, 128, 128] where Y = 256 for ResNet, 128 for Xception
        :type skip: torch.Tensor
        :return: tensor with the final output, [batch, classes, input height, input width]
        :rtype: torch.Tensor
        """
        skip = self.low_level(skip)
        x = self.upsample(x)
        return self.output(torch.cat((skip, x), dim=1))


class DecoderAdaptNet(Decoder):

    def __init__(self,
                 skip_channels: Tuple[int, int] = (64, 512),
                 skip_upsamples: Tuple[int, int] = (2, 4),
                 skip_outputs: Tuple[int, int] = (48, 256),
                 aspp_upsample: int = 2,
                 aspp_channels: int = 256,
                 output_channels: int = 1,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d):
        super().__init__()
        # decoder uses encoder layers in inverse order
        self.skip1 = self.skip_connection(skip_channels[1], skip_outputs[1], batch_norm=batch_norm)
        self.skip2 = self.skip_connection(skip_channels[0], skip_outputs[0], batch_norm=batch_norm)
        self.upsample1 = nn.Upsample(scale_factor=aspp_upsample, mode="bilinear", align_corners=True)
        self.upconv1 = self.upconv_block(in_channels=(skip_outputs[1] + aspp_channels),
                                         mid_channels=128,
                                         out_channels=256,
                                         batch_norm=batch_norm)
        self.upsample2 = nn.Upsample(scale_factor=skip_upsamples[1], mode="bilinear", align_corners=True)
        self.upconv2 = self.upconv_block(in_channels=(skip_outputs[0] + 256),
                                         mid_channels=158,
                                         out_channels=256,
                                         batch_norm=batch_norm)
        self.upsample3 = nn.Upsample(scale_factor=skip_upsamples[0], mode="bilinear", align_corners=True)
        self.head = nn.Sequential(nn.Dropout(p=0.5), nn.Conv2d(256, output_channels, kernel_size=1))

    def skip_connection(self, in_channels: int, out_channels: int, batch_norm: Type[nn.Module]):
        # yapf: disable
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                             batch_norm(out_channels),
                             nn.ReLU(inplace=True))

    def upconv_block(self, in_channels: int, mid_channels: int, out_channels: int, batch_norm: Type[nn.Module]):
        return nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1, padding=0, bias=False),
                             batch_norm(mid_channels),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
                             batch_norm(out_channels),
                             nn.ReLU(inplace=True))
        # yapf: enable

    def forward(self, x: torch.Tensor, mid_level: torch.Tensor, low_level: torch.Tensor) -> torch.Tensor:
        x = self.upsample1(x)
        x2 = torch.cat((x, self.skip1(mid_level)), dim=1)
        x = x + self.upconv1(x2)
        x = self.upsample2(x)
        x3 = torch.cat((x, self.skip2(low_level)), dim=1)
        x = x + self.upconv2(x3)
        x = self.upsample3(x)
        return self.head(x)


class UNetDecodeBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 skip_channels: int,
                 out_channels: int,
                 scale_factor: int = 2,
                 bilinear: bool = True,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d):
        super().__init__()
        upsampled_channels = in_channels // 2
        self.upsample = self._upsampling(in_channels, upsampled_channels, factor=scale_factor, bilinear=bilinear)
        self.conv = self._upconv(upsampled_channels + skip_channels, out_channels, norm_layer=norm_layer)

    def _upsampling(self, in_channels: int, out_channels: int, factor: int, bilinear: bool = True):
        if bilinear:
            return nn.Sequential(nn.Upsample(scale_factor=factor, mode="bilinear", align_corners=True),
                                 nn.Conv2d(in_channels, out_channels, kernel_size=1))
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=factor, stride=factor)

    def _upconv(self,
                in_channels: int,
                out_channels: int,
                norm_layer: Type[nn.Module] = nn.BatchNorm2d) -> nn.Sequential:
        # yapf: disable
        # mid_channels = (in_channels + out_channels) // 2
        mid_channels = out_channels
        return nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                             norm_layer(mid_channels), nn.ReLU(inplace=True),
                             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                             norm_layer(out_channels), nn.ReLU(inplace=True))
        # yapf: enable

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        combined = torch.cat((x, skip), dim=1)
        return x + self.conv(combined)


class UNetHead(nn.Module):

    def __init__(self, in_channels: int, num_classes: int, scale_factor: int = 2, dropout_prob: float = 0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
        self.out = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.upsample(x)
        return self.out(x)


class DecoderUNet(Decoder):

    def __init__(self,
                 feature_channels: List[int],
                 feature_reductions: List[int],
                 bilinear: bool = True,
                 output_channels: int = 1,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d):
        super().__init__()
        # invert sequences to decode
        channels = feature_channels[::-1]
        reductions = feature_reductions[::-1] + [1]
        scaling_factors = [int(reductions[i] // reductions[i + 1]) for i in range(len(reductions) - 1)]

        self.up1 = UNetDecodeBlock(channels[0], channels[1], channels[0] // 2, scaling_factors[0], bilinear, norm_layer)
        self.up2 = UNetDecodeBlock(channels[1], channels[2], channels[1] // 2, scaling_factors[1], bilinear, norm_layer)
        self.up3 = UNetDecodeBlock(channels[2], channels[3], channels[2] // 2, scaling_factors[2], bilinear, norm_layer)
        self.up4 = UNetDecodeBlock(channels[3], channels[4], channels[3] // 2, scaling_factors[3], bilinear, norm_layer)
        self.out = UNetHead(channels[3] // 2, output_channels, scaling_factors[4])

    def forward(self, *features: Iterable[torch.Tensor]) -> torch.Tensor:
        x1, x2, x3, x4, x5 = features
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)


class SSMA(nn.Module):

    def __init__(self, rgb_channels: int, ir_channels: int, norm_layer: Type[nn.Module] = nn.BatchNorm2d):
        super().__init__()
        # RGB and IR channel count should be the same, so (256 + 256) / 4 = 128 => 256 / 2
        total_chs = rgb_channels + ir_channels
        bottleneck_chs = (rgb_channels + ir_channels) // 4
        self.bottleneck = nn.Sequential(nn.Conv2d(total_chs, bottleneck_chs, kernel_size=3, padding=1, bias=False),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(bottleneck_chs, total_chs, kernel_size=3, padding=1, bias=False),
                                        nn.Sigmoid())
        self.out_bn = norm_layer(total_chs)
        self.out_conv = nn.Conv2d(total_chs, rgb_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        x1 = torch.cat((rgb, ir), dim=1)
        x = self.bottleneck(x1)
        x = self.out_bn(x1 + x)
        return self.out_conv(x)


class RotationHead(nn.Sequential):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 100,
                 num_classes: int = 4,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d):
        super().__init__(nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1, bias=True),
                         nn.AdaptiveAvgPool2d(output_size=1), norm_layer(hidden_dim), nn.Flatten(),
                         nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(hidden_dim, num_classes))
