from typing import Optional
import torch.nn as nn
from encoding import Encoder, EncodingBlock
from decoding import Decoder
from conv import ConvolutionalBlock

class UNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_classes: int,
            out_channels_first_layer: int = 64,
            dimensions: int=2,
            num_encoding_blocks: int=5,
            normalization: Optional[str] = None,
            pooling_type: str = 'max',
            upsampling_type: str = 'conv',
            preactivation: bool = False,
            residual: bool = False,
            padding: int = 0,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            initial_dilation: Optional[int] = None,
            dropout: float = 0,
            use_bias: bool=True,
            use_sigmoid: bool = False
            ):
        super().__init__()
        depth = num_encoding_blocks - 1

        # Encoder
        self.encoder = Encoder(
            in_channels,
            out_channels_first_layer,
            dimensions,
            pooling_type,
            depth,
            normalization=normalization,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dropout=dropout
        )

        # Bottom of UNet (last encoding block)
        in_channels = self.encoder.out_channels
        out_channels_first = 2 * in_channels

        self.bottom_block = EncodingBlock(
            in_channels,
            out_channels_first,
            dimensions,
            normalization,
            pooling_type = None,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=self.encoder.dilation,
            dropout=dropout
        )

        # Decoder
        in_channels = self.bottom_block.out_channels
        in_channels_skip_connection = out_channels_first_layer * 2
        num_decoding_blocks = depth
        self.decoder = Decoder(
            in_channels_skip_connection,
            dimensions,
            upsampling_type,
            num_decoding_blocks,
            normalization=normalization,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            initial_dilation=self.encoder.dilation,
            dropout=dropout
        )

        in_channels = out_channels_first_layer
        self.output = ConvolutionalBlock(
            dimensions,
            in_channels,
            out_channels=out_classes,
            kernel_size=1,
            activation=None,
            normalization=None
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip_connections, encoding = self.encoder(x)
        encoding = self.bottom_block(encoding)
        x = self.decoder(skip_connections, encoding)
        if self.use_sigmoid:
            return self.sigmoid(x)
        else:
            return x


import torch
random_data = torch.rand((1, 1, 28, 28))
my_nn = UNet(1,1)
result = my_nn(random_data)
