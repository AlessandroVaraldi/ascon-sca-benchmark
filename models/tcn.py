#!/usr/bin/env python3
"""
TCN: CNN + TCN-style dilated residual blocks.

Designed to work with the Ascon pipeline:
- Input: traces [B, T] or [B, 1, T]
- Output: logits [B, num_classes]
"""

from typing import Tuple

import torch
import torch.nn as nn


class _TCNBlock(nn.Module):
    """
    Simple TCN-style residual block with dilated convolutions.
    Non-causal, padding chosen to preserve length.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return self.relu(out + residual)


class TCN(nn.Module):
    """
    CNN + TCN hybrid profiling model for SCA traces.

    Pipeline:
    - optional input AvgPool
    - CNN stack (local feature extraction & downsampling)
    - stack of TCN-style dilated residual blocks (temporal integration)
    - global average pooling over time
    - small MLP classifier

    Parameters
    ----------
    in_channels : int
        Number of input channels (1).
    num_classes : int
        Output classes (2 for a bit).
    base_channels : int
        Channels in first conv layer.
    num_conv_layers : int
        Number of conv blocks before TCN.
    kernel_size : int
        Kernel size for conv blocks.
    pool_size : int
        MaxPool1d kernel/stride.
    dropout : float
        Dropout in classifier.
    use_input_avgpool : bool
        Optional input downsampling.
    input_pool_kernel : int
        AvgPool kernel/stride at input.
    tcn_depth : int
        Number of TCN residual blocks.
    tcn_kernel_size : int
        Kernel size for TCN blocks.
    tcn_dropout : float
        Dropout in TCN blocks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
        num_conv_layers: int = 3,
        kernel_size: int = 11,
        pool_size: int = 2,
        dropout: float = 0.3,
        use_input_avgpool: bool = False,
        input_pool_kernel: int = 2,
        tcn_depth: int = 2,
        tcn_kernel_size: int = 5,
        tcn_dropout: float = 0.1,
    ):
        super().__init__()

        self.use_input_avgpool = use_input_avgpool
        if use_input_avgpool:
            self.input_pool = nn.AvgPool1d(kernel_size=input_pool_kernel, stride=input_pool_kernel)

        # CNN stem
        cnn_layers = []
        c_in = in_channels
        padding = (kernel_size - 1) // 2

        for i in range(num_conv_layers):
            c_out = base_channels * (2 ** i)
            cnn_layers.append(
                nn.Conv1d(
                    c_in,
                    c_out,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            cnn_layers.append(nn.BatchNorm1d(c_out))
            cnn_layers.append(nn.ReLU(inplace=True))
            cnn_layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))
            c_in = c_out

        self.cnn = nn.Sequential(*cnn_layers)
        feature_channels = c_in  # channels after CNN

        # TCN blocks with increasing dilation
        tcn_blocks = []
        for i in range(tcn_depth):
            dilation = 2 ** i
            tcn_blocks.append(
                _TCNBlock(
                    channels=feature_channels,
                    kernel_size=tcn_kernel_size,
                    dilation=dilation,
                    dropout=tcn_dropout,
                )
            )
        self.tcn = nn.Sequential(*tcn_blocks)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),                 # [B, C]
            nn.Linear(feature_channels, feature_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_channels, num_classes),
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

        x = self.cnn(x)          # [B, C, T']
        x = self.tcn(x)          # [B, C, T']
        x = self.global_pool(x)  # [B, C, 1]
        logits = self.classifier(x)
        return logits
