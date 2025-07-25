# Code adapted from https://github.com/fepegar/unet

import torch.nn as nn
from typing import Optional

class ConvolutionalBlock(nn.Module):
    def __init__(self,
                 dimensions: int,
                 in_channels: int,
                 out_channels: int,
                 normalization: Optional[str] = None,
                 kernel_size: int = 2,
                 activation: Optional[str] = 'ReLU',
                 preactivation: bool = False,
                 padding_mode: str = 'zeros',
                 dropout: float = 0
                 ):
        super().__init__()                  # super lets you avoid referring to the base class explicitly

        block = nn.ModuleList()             # Holds submodules in a list

        class_name = 'Conv{}d'.format(dimensions)
        conv_class = getattr(nn, class_name)
        no_bias = not preactivation and (normalization is not None)
        conv_layer = conv_class(
            in_channels,
            out_channels,
            kernel_size,
            padding= (kernel_size + 1) // 2 - 1,
            padding_mode=padding_mode,
            bias = not no_bias
        )

        norm_layer = None
        if normalization is not None:
            class_name = '{}Norm{}d'.format(normalization.capitalize(), dimensions)
            norm_class = getattr(nn, class_name)
            num_features = in_channels if preactivation else out_channels
            norm_layer = norm_class(num_features)

        activation_layer = None
        if activation is not None:
            activation_layer = getattr(nn, activation)()

        if preactivation:
            self.add_if_not_none(block, norm_layer)
            self.add_if_not_none(block, activation_layer)
            self.add_if_not_none(block, conv_layer)
        else:
            self.add_if_not_none(block, conv_layer)
            self.add_if_not_none(block, norm_layer)
            self.add_if_not_none(block, activation_layer)

        dropout_layer = None
        if dropout:
            class_name = 'Dropout{}d'.format(dimensions)
            dropout_class = getattr(nn, class_name)
            dropout_layer = dropout_class(p=dropout)
            self.add_if_not_none(block, dropout_layer)

        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer
        self.dropout_layer = dropout_layer

        # Sequential container, modules will be added in the order they are passed into the container
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

    @staticmethod
    def add_if_not_none(module_list, module):
        if module is not None:
            module_list.append(module)













