from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from CloneRL.models.torch.encoders import ConvEncoder, get_activation


class GaussianImageActor(nn.Module):
    def __init__(
        self,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        proprioception_channels: int = 3,
        encoder_channels: int = 60,
        image_channels: int = 1,
        depth_channels: int = 1,
        action_dim: int = 2,
        mlp_features: list[int] | tuple[int, ...] = (256, 160, 128),
        image_input_dim: list[int] | tuple[int, int] = (224, 224),
        image_encoder_features: list[int] | tuple[int, ...] = (8, 16, 32, 64),
        image_fc_features: list[int] | tuple[int, ...] = (120, 60),
        activation: str = "leaky_relu",
        dropout_rate: float = 0,
        use_batch_norm: bool = False,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        state_independent_log_std: bool = True,
    ):
        super().__init__()
        self.device = device
        self.proprioception_channels = proprioception_channels
        self.encoder_channels = encoder_channels
        self.action_dim = action_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.state_independent_log_std = state_independent_log_std
        self.image_channels = image_channels
        self.depth_channels = depth_channels

        self.image_encoder = ConvEncoder(image_channels, image_input_dim, image_encoder_features, image_fc_features)
        self.depth_encoder = ConvEncoder(depth_channels, image_input_dim, image_encoder_features, image_fc_features)
        self.encoder_channels = image_fc_features[-1] * 2

        self.mlp = _make_mlp(
            proprioception_channels + self.encoder_channels,
            mlp_features,
            activation,
            dropout_rate,
            use_batch_norm,
        )
        final_features = (
            mlp_features[-1]
            if mlp_features
            else proprioception_channels + self.encoder_channels
        )
        self.mean_head = nn.Linear(final_features, action_dim)
        if state_independent_log_std:
            self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        else:
            self.log_std_head = nn.Linear(final_features, action_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state: Dict[str, torch.Tensor]):
        x = _encode_feedforward_state(
            state,
            self.image_encoder,
            self.depth_encoder,
            self.proprioception_channels,
        )
        for layer in self.mlp:
            x = layer(x)

        mu = torch.tanh(self.mean_head(x))
        if self.state_independent_log_std:
            log_std = torch.clamp(self.log_std, min=self.min_log_std, max=self.max_log_std)
            log_std = log_std.expand_as(mu)
        else:
            log_std = torch.clamp(self.log_std_head(x), min=self.min_log_std, max=self.max_log_std)
        return torch.distributions.Normal(mu, log_std.exp())

    def get_action(self, state: Dict[str, torch.Tensor], deterministic: bool = True):
        with torch.no_grad():
            dist = self.forward(state)
            return dist.mean if deterministic else dist.sample()


class ImageValue(nn.Module):
    def __init__(
        self,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        proprioception_channels: int = 3,
        encoder_channels: int = 60,
        image_channels: int = 1,
        depth_channels: int = 1,
        mlp_features: list[int] | tuple[int, ...] = (256, 160, 128),
        image_input_dim: list[int] | tuple[int, int] = (224, 224),
        image_encoder_features: list[int] | tuple[int, ...] = (8, 16, 32, 64),
        image_fc_features: list[int] | tuple[int, ...] = (120, 60),
        activation: str = "leaky_relu",
        dropout_rate: float = 0,
        use_batch_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.proprioception_channels = proprioception_channels
        self.encoder_channels = encoder_channels
        self.image_channels = image_channels
        self.depth_channels = depth_channels

        self.image_encoder = ConvEncoder(image_channels, image_input_dim, image_encoder_features, image_fc_features)
        self.depth_encoder = ConvEncoder(depth_channels, image_input_dim, image_encoder_features, image_fc_features)
        self.encoder_channels = image_fc_features[-1] * 2
        self.mlp = _make_mlp(
            proprioception_channels + self.encoder_channels,
            mlp_features,
            activation,
            dropout_rate,
            use_batch_norm,
        )
        final_features = (
            mlp_features[-1]
            if mlp_features
            else proprioception_channels + self.encoder_channels
        )
        self.value_head = nn.Linear(final_features, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state: Dict[str, torch.Tensor]):
        x = _encode_feedforward_state(
            state,
            self.image_encoder,
            self.depth_encoder,
            self.proprioception_channels,
        )
        for layer in self.mlp:
            x = layer(x)
        return self.value_head(x)


class ImageQ(nn.Module):
    def __init__(
        self,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        proprioception_channels: int = 3,
        encoder_channels: int = 60,
        image_channels: int = 1,
        depth_channels: int = 1,
        action_dim: int = 2,
        mlp_features: list[int] | tuple[int, ...] = (256, 160, 128),
        image_input_dim: list[int] | tuple[int, int] = (224, 224),
        image_encoder_features: list[int] | tuple[int, ...] = (8, 16, 32, 64),
        image_fc_features: list[int] | tuple[int, ...] = (120, 60),
        activation: str = "leaky_relu",
        dropout_rate: float = 0,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        self.device = device
        self.proprioception_channels = proprioception_channels
        self.encoder_channels = encoder_channels
        self.action_dim = action_dim
        self.image_channels = image_channels
        self.depth_channels = depth_channels

        self.image_encoder = ConvEncoder(image_channels, image_input_dim, image_encoder_features, image_fc_features)
        self.depth_encoder = ConvEncoder(depth_channels, image_input_dim, image_encoder_features, image_fc_features)
        self.encoder_channels = image_fc_features[-1] * 2
        self.mlp = _make_mlp(
            proprioception_channels + self.encoder_channels + action_dim,
            mlp_features,
            activation,
            dropout_rate,
            use_batch_norm,
        )
        final_features = (
            mlp_features[-1]
            if mlp_features
            else proprioception_channels + self.encoder_channels + action_dim
        )
        self.q_head = nn.Linear(final_features, 1)

    def forward(self, state: Dict[str, torch.Tensor], action: torch.Tensor):
        x_img = self.image_encoder(state["image"])
        x_depth = self.depth_encoder(state["depth"])
        proprioceptive = state["proprioceptive"][..., : self.proprioception_channels]
        dim = 2 if x_img.ndim == 3 else 1
        x = torch.cat([proprioceptive, x_img, x_depth, action], dim=dim)
        for layer in self.mlp:
            x = layer(x)
        return self.q_head(x)


class TwinImageQ(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        device = kwargs.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.q1 = ImageQ(**kwargs).to(device)
        self.q2 = ImageQ(**kwargs).to(device)

    def forward(self, state: Dict[str, torch.Tensor], action: torch.Tensor):
        return self.q1(state, action), self.q2(state, action)

    def q1_forward(self, state: Dict[str, torch.Tensor], action: torch.Tensor):
        return self.q1(state, action)


def _make_mlp(
    in_features: int,
    features: list[int] | tuple[int, ...],
    activation: str,
    dropout_rate: float,
    use_batch_norm: bool,
) -> nn.ModuleList:
    layers = nn.ModuleList()
    for feature in features:
        layers.append(nn.Linear(in_features, feature))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(feature))
        layers.append(get_activation(activation))
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        in_features = feature
    return layers


def _encode_feedforward_state(
    state: Dict[str, torch.Tensor],
    image_encoder: nn.Module,
    depth_encoder: nn.Module,
    proprioception_channels: int,
) -> torch.Tensor:
    x_img = image_encoder(state["image"])
    x_depth = depth_encoder(state["depth"])
    proprioceptive = state["proprioceptive"][..., :proprioception_channels]
    dim = 2 if x_img.ndim == 3 else 1
    return torch.cat([proprioceptive, x_img, x_depth], dim=dim)


actor_gaussian_image = GaussianImageActor
v_image = ImageValue
q_image = ImageQ
TwinQ_image = TwinImageQ
