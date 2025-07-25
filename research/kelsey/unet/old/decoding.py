# Code adapted from https://github.com/fepegar/unet

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from conv import ConvolutionalBlock

CHANNELS_DIMENSION = 1
UPSAMPLING_MODES = (
    'nearest',
    'linear',
    'bilinear',
    'bicubic',
    'trilinear',
)

class Decoder(nn.Module):
    def __init__(
            self,
            in_channels_skip_connection: int,
            dimensions: int,
            upsampling_type: str,
            num_decoding_blocks: int,
            normalization: Optional[str],
            preactivation: bool = False,
            residual: bool = False,
            padding: int = 0,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            initial_dilation: Optional[int] = None,
            dropout: float = 0
            ):
        super().__init__()
        self.dilation = initial_dilation
        self.decoding_blocks = nn.ModuleList()

        for _ in range(num_decoding_blocks):
            decoding_block = DecodingBlock(
                in_channels_skip_connection,
                dimensions,
                upsampling_type=upsampling_type,
                normalization=normalization,
                preactivation=preactivation,
                residual=residual,
                padding=padding,
                padding_mode=padding_mode,
                activation=activation,
                dilation=self.dilation,
                dropout=dropout
            )
            self.decoding_blocks.append(decoding_block)
            in_channels_skip_connection //= 2
            if self.dilation is not None:
                self.dilation //= 2

    def forward(self, skip_connections, x):
        zipped = zip(reversed(skip_connections), self.decoding_blocks)
        for skip_connection, decoding_block in zipped:
            x = decoding_block(skip_connection, x)
        return x


class DecodingBlock(nn.Module):
    def __init__(
            self,
            in_channels_skip_connection: int,
            dimensions: int,
            upsampling_type: str,
            normalization: Optional[str],
            preactivation: bool = False,
            residual: bool = False,
            padding: int = 0,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            dilation: Optional[int] = None,
            dropout: float = 0
            ):
        super().__init__()

        self.residual = residual

        if upsampling_type == 'conv':
            in_channels = out_channels = 2 * in_channels_skip_connection
            class_name = 'ConvTranspose{}d'.format(dimensions)
            conv_class = getattr(nn, class_name)
            self.upsample = conv_class(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            raise NotImplementedError()

        in_channels_first = in_channels_skip_connection * 2
        out_channels = in_channels_skip_connection

        self.conv1 = ConvolutionalBlock(
            dimensions,
            in_channels_first,
            out_channels,
            normalization=normalization,
            preactivation=preactivation,
            padding_mode=padding_mode,
            activation=activation,
            dropout=dropout
        )

        in_channels_second = out_channels

        self.conv2 = ConvolutionalBlock(
            dimensions,
            in_channels_second,
            out_channels,
            normalization=normalization,
            preactivation=preactivation,
            padding_mode=padding_mode,
            activation=activation,
            dropout=dropout
        )

        if residual:
            self.conv_residual = ConvolutionalBlock(
                dimensions,
                in_channels_first,
                out_channels,
                kernel_size=1,
                normalization=None,
                activation=None
            )

    def forward(self, skip_connection, x):
        x = self.upsample(x)
        x = torch.cat((skip_connection, x), dim=CHANNELS_DIMENSION)

        if self.residual:
            connection = self.conv_residual(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x += connection
        else:
            x = self.conv1(x)
            x = self.conv2(x)
        return x


'''
# To do: Implement additional upsampling modes (if of interest)
# Currently only supporting conv
def get_upsampling_layer(upsampling_type: str) -> nn.Upsample:
    if upsampling_type not in UPSAMPLING_MODES:
        message = (
            'Upsampling type is "{}"'
            ' but should be one of the following: {}'
        )
        message = message.format(upsampling_type, UPSAMPLING_MODES)
        raise ValueError(message)
    upsample = nn.Upsample(
        scale_factor=2,
        mode=upsampling_type,
        align_corners=False,
    )
    return upsample
'''




