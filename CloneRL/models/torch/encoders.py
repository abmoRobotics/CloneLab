from __future__ import annotations

import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    activations = {
        "leaky_relu": nn.LeakyReLU(),
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "elu": nn.ELU(),
        "relu6": nn.ReLU6(),
        "selu": nn.SELU(),
    }
    if name not in activations:
        raise ValueError(f"Activation function {name} not supported.")
    return activations[name]


class ConvEncoder(nn.Module):
    """Convolutional image/depth encoder used by the default CloneRL models."""

    def __init__(
        self,
        in_channels: int,
        input_dim: list[int] | tuple[int, int] = (224, 224),
        encoder_features: list[int] | tuple[int, ...] = (8, 16, 32, 64),
        fc_features: list[int] | tuple[int, ...] = (120, 60),
        encoder_activation: str = "leaky_relu",
    ):
        super().__init__()
        padding = 1
        stride = 1
        kernel_size = 3
        self.encoder_layers = nn.ModuleList()

        for feature in encoder_features:
            self.encoder_layers.append(
                nn.Conv2d(in_channels, feature, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
            )
            self.encoder_layers.append(nn.BatchNorm2d(feature))
            self.encoder_layers.append(get_activation(encoder_activation))
            self.encoder_layers.append(
                nn.Conv2d(feature, feature, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
            )
            self.encoder_layers.append(nn.BatchNorm2d(feature))
            self.encoder_layers.append(get_activation(encoder_activation))
            self.encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feature

        flattened = list(input_dim)
        for _ in encoder_features:
            width = (flattened[0] - kernel_size + 2 * padding) // stride + 1
            height = (flattened[1] - kernel_size + 2 * padding) // stride + 1
            width = (width - 2) // 2 + 1
            height = (height - 2) // 2 + 1
            flattened = [width, height]

        in_features = flattened[0] * flattened[1] * in_channels
        self.fc_layers = nn.ModuleList()
        for feature in fc_features:
            self.fc_layers.append(nn.Linear(in_features, feature))
            self.fc_layers.append(nn.LeakyReLU())
            in_features = feature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_sequence = x.ndim == 5
        if is_sequence:
            batch_size, seq_len, channels, height, width = x.shape
            x = x.view(batch_size * seq_len, channels, height, width)

        for layer in self.encoder_layers:
            x = layer(x)

        x = x.reshape(x.size(0), -1)
        for layer in self.fc_layers:
            x = layer(x)

        if is_sequence:
            x = x.view(batch_size, seq_len, x.shape[1])
        return x


conv_encoder = ConvEncoder
