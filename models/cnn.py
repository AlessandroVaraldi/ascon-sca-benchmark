#!/usr/bin/env python3
"""
CNN1D: 1D CNN for side-channel profiling attacks.

Designed to work with the Ascon pipeline:
- Input: traces [B, T] or [B, 1, T]
- Output: logits [B, num_classes]
"""

from typing import Tuple

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    1D-CNN profiling model for SCA-style traces.

    Architecture:
    - optional input AvgPool (downsampling)
    - stack of Conv1d -> BN -> ReLU -> MaxPool
    - global average pooling over time
    - small MLP head

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for single power trace).
    num_classes : int
        Number of output classes (2 for single bit).
    base_channels : int
        Number of channels in the first conv layer.
    num_conv_layers : int
        Number of convolutional blocks.
    kernel_size : int
        Kernel size for all conv layers (odd value recommended).
    pool_size : int
        MaxPool1d kernel/stride.
    dropout : float
        Dropout rate in the classifier.
    use_input_avgpool : bool
        If True, apply an initial AvgPool1d to the input (simple downsampling).
    input_pool_kernel : int
        Kernel/stride for the input AvgPool1d.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
        num_conv_layers: int = 4,
        kernel_size: int = 11,
        pool_size: int = 2,
        dropout: float = 0.5,
        use_input_avgpool: bool = False,
        input_pool_kernel: int = 2,
    ):
        super().__init__()

        self.use_input_avgpool = use_input_avgpool
        if use_input_avgpool:
            self.input_pool = nn.AvgPool1d(kernel_size=input_pool_kernel, stride=input_pool_kernel)

        layers = []
        c_in = in_channels
        padding = (kernel_size - 1) // 2

        # Convolutional stack
        for i in range(num_conv_layers):
            c_out = base_channels * (2 ** i)  # 32, 64, 128, ...
            layers.append(
                nn.Conv1d(
                    c_in,
                    c_out,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm1d(c_out))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))
            c_in = c_out

        self.conv = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # [B, C, 1]

        self.classifier = nn.Sequential(
            nn.Flatten(),          # [B, C]
            nn.Linear(c_in, c_in),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(c_in, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T] or [B, 1, T]
        returns logits: [B, num_classes]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B,1,T]

        if self.use_input_avgpool:
            x = self.input_pool(x)

        x = self.conv(x)              # [B, C, T']
        x = self.global_pool(x)       # [B, C, 1]
        logits = self.classifier(x)   # [B, num_classes]
        return logits
