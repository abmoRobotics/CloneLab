from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from CloneRL.models.torch.encoders import ConvEncoder


class GRUActorGaussian(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        depth_channels: int = 1,
        image_size: list[int] | tuple[int, int] = (224, 224),
        proprioception_channels: int = 3,
        action_dim: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        image_encoder_features: list[int] | tuple[int, ...] = (8, 16, 32, 64),
        image_fc_features: list[int] | tuple[int, ...] = (120, 60),
        mlp_features: list[int] | tuple[int, ...] = (256, 160, 128),
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        state_independent_log_std: bool = True,
        device: str | None = None,
    ):
        super().__init__()
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hidden_val = None
        self.action_dim = action_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.state_independent_log_std = state_independent_log_std
        self.proprioception_channels = proprioception_channels
        self.image_channels = image_channels
        self.depth_channels = depth_channels

        encoder_output_size = 0
        self.image_encoder = _make_encoder(image_channels, image_size, image_encoder_features, image_fc_features)
        if self.image_encoder is not None:
            encoder_output_size += image_fc_features[-1]
        self.depth_encoder = _make_encoder(depth_channels, image_size, image_encoder_features, image_fc_features)
        if self.depth_encoder is not None:
            encoder_output_size += image_fc_features[-1]

        self.gru = nn.GRU(
            input_size=proprioception_channels + encoder_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.mlp = _make_mlp(hidden_size, mlp_features)
        final_features = mlp_features[-1] if mlp_features else hidden_size
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

    def forward(self, state: Dict[str, torch.Tensor], hidden: torch.Tensor | None = None):
        x, batch, seq_len = _recurrent_features(
            state,
            self.image_encoder,
            self.depth_encoder,
            self.proprioception_channels,
        )
        if hidden is None:
            hidden = torch.zeros(self.num_layers, batch, self.hidden_size, device=x.device)
        out, hidden = self.gru(x, hidden)
        out = _apply_step_mlp(out, self.mlp, seq_len)

        mu = torch.tanh(self.mean_head(out))
        if self.state_independent_log_std:
            log_std = torch.clamp(self.log_std, min=self.min_log_std, max=self.max_log_std)
            log_std = log_std.expand_as(mu)
        else:
            log_std = torch.clamp(self.log_std_head(out), min=self.min_log_std, max=self.max_log_std)
        return torch.distributions.Normal(mu, log_std.exp()), hidden

    def get_action(self, state: Dict[str, torch.Tensor], deterministic: bool = True):
        with torch.no_grad():
            dist, self.hidden_val = self.forward(state, self.hidden_val)
            action = dist.mean if deterministic else dist.sample()
            return action.squeeze(1) if action.dim() == 3 else action

    def reset_hidden(self, batch_size: int = 1):
        self.hidden_val = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

    def validate(self, state: Dict[str, torch.Tensor], done: torch.Tensor):
        with torch.no_grad():
            dist, new_hidden = self.forward(state, self.hidden_val)
            if done is not None:
                done_flat = done.view(-1) if done.dim() > 1 else done
                not_done = ~done_flat.bool()
                keep = not_done.unsqueeze(0).unsqueeze(2).expand_as(new_hidden)
                self.hidden_val = new_hidden * keep.float()
            else:
                self.hidden_val = new_hidden
            action = dist.mean
            return action.squeeze(1) if action.dim() == 3 else action


class GRUValue(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        depth_channels: int = 1,
        image_size: list[int] | tuple[int, int] = (224, 224),
        proprioception_channels: int = 3,
        hidden_size: int = 128,
        num_layers: int = 2,
        image_encoder_features: list[int] | tuple[int, ...] = (8, 16, 32, 64),
        image_fc_features: list[int] | tuple[int, ...] = (120, 60),
        mlp_features: list[int] | tuple[int, ...] = (256, 160, 128),
        device: str | None = None,
    ):
        super().__init__()
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hidden_val = None
        self.proprioception_channels = proprioception_channels
        self.image_channels = image_channels
        self.depth_channels = depth_channels

        encoder_output_size = 0
        self.image_encoder = _make_encoder(image_channels, image_size, image_encoder_features, image_fc_features)
        if self.image_encoder is not None:
            encoder_output_size += image_fc_features[-1]
        self.depth_encoder = _make_encoder(depth_channels, image_size, image_encoder_features, image_fc_features)
        if self.depth_encoder is not None:
            encoder_output_size += image_fc_features[-1]

        self.gru = nn.GRU(
            input_size=proprioception_channels + encoder_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.mlp = _make_mlp(hidden_size, mlp_features)
        final_features = mlp_features[-1] if mlp_features else hidden_size
        self.value_head = nn.Linear(final_features, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state: Dict[str, torch.Tensor], hidden: torch.Tensor | None = None):
        x, batch, seq_len = _recurrent_features(
            state,
            self.image_encoder,
            self.depth_encoder,
            self.proprioception_channels,
        )
        if hidden is None:
            hidden = torch.zeros(self.num_layers, batch, self.hidden_size, device=x.device)
        out, hidden = self.gru(x, hidden)
        out = _apply_step_mlp(out, self.mlp, seq_len)
        return self.value_head(out), hidden

    def reset_hidden(self, batch_size: int = 1):
        self.hidden_val = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)


class GRUQNetwork(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        depth_channels: int = 1,
        image_size: list[int] | tuple[int, int] = (224, 224),
        proprioception_channels: int = 3,
        action_dim: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        image_encoder_features: list[int] | tuple[int, ...] = (8, 16, 32, 64),
        image_fc_features: list[int] | tuple[int, ...] = (120, 60),
        mlp_features: list[int] | tuple[int, ...] = (256, 160, 128),
        device: str | None = None,
    ):
        super().__init__()
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hidden_val = None
        self.proprioception_channels = proprioception_channels
        self.action_dim = action_dim
        self.image_channels = image_channels
        self.depth_channels = depth_channels

        encoder_output_size = 0
        self.image_encoder = _make_encoder(image_channels, image_size, image_encoder_features, image_fc_features)
        if self.image_encoder is not None:
            encoder_output_size += image_fc_features[-1]
        self.depth_encoder = _make_encoder(depth_channels, image_size, image_encoder_features, image_fc_features)
        if self.depth_encoder is not None:
            encoder_output_size += image_fc_features[-1]

        self.gru = nn.GRU(
            input_size=proprioception_channels + encoder_output_size + action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.mlp = _make_mlp(hidden_size, mlp_features)
        final_features = mlp_features[-1] if mlp_features else hidden_size
        self.q_head = nn.Linear(final_features, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ):
        if action.dim() == 2:
            action = action.unsqueeze(1)
        x, batch, seq_len = _recurrent_features(
            state,
            self.image_encoder,
            self.depth_encoder,
            self.proprioception_channels,
        )
        x = torch.cat([x, action], dim=2)
        if hidden is None:
            hidden = torch.zeros(self.num_layers, batch, self.hidden_size, device=x.device)
        out, hidden = self.gru(x, hidden)
        out = _apply_step_mlp(out, self.mlp, seq_len)
        return self.q_head(out), hidden

    def reset_hidden(self, batch_size: int = 1):
        self.hidden_val = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)


class GRUTwinQ(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = kwargs.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        self.q1 = GRUQNetwork(**kwargs)
        self.q2 = GRUQNetwork(**kwargs)

    def forward(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        hidden1: torch.Tensor | None = None,
        hidden2: torch.Tensor | None = None,
    ):
        q1, hidden1 = self.q1(state, action, hidden1)
        q2, hidden2 = self.q2(state, action, hidden2)
        return q1, q2, hidden1, hidden2

    def q1_forward(self, state: Dict[str, torch.Tensor], action: torch.Tensor, hidden: torch.Tensor | None = None):
        return self.q1(state, action, hidden)

    def reset_hidden(self, batch_size: int = 1):
        self.q1.reset_hidden(batch_size)
        self.q2.reset_hidden(batch_size)


def _make_encoder(
    channels: int,
    image_size: list[int] | tuple[int, int],
    encoder_features: list[int] | tuple[int, ...],
    fc_features: list[int] | tuple[int, ...],
) -> ConvEncoder | None:
    if channels <= 0:
        return None
    return ConvEncoder(channels, image_size, encoder_features, fc_features)


def _make_mlp(in_features: int, features: list[int] | tuple[int, ...]) -> nn.ModuleList:
    layers = nn.ModuleList()
    for feature in features:
        layers.append(nn.Linear(in_features, feature))
        layers.append(nn.LeakyReLU())
        in_features = feature
    return layers


def _recurrent_features(
    state: Dict[str, torch.Tensor],
    image_encoder: ConvEncoder | None,
    depth_encoder: ConvEncoder | None,
    proprioception_channels: int,
) -> tuple[torch.Tensor, int, int]:
    proprioceptive = state["proprioceptive"]
    if proprioceptive.dim() == 2:
        proprioceptive = proprioceptive.unsqueeze(1)
    batch = proprioceptive.shape[0]
    seq_len = proprioceptive.shape[1]

    features = [proprioceptive[..., :proprioception_channels]]
    if image_encoder is not None and "image" in state:
        features.append(_encode_sequence(state["image"], image_encoder, batch, seq_len))
    if depth_encoder is not None and "depth" in state:
        features.append(_encode_sequence(state["depth"], depth_encoder, batch, seq_len))
    return torch.cat(features, dim=2), batch, seq_len


def _encode_sequence(x: torch.Tensor, encoder: ConvEncoder, batch: int, seq_len: int) -> torch.Tensor:
    if x.dim() == 4:
        x = x.unsqueeze(1)
    _, _, channels, height, width = x.shape
    x = x.view(batch * seq_len, channels, height, width)
    x = encoder(x)
    return x.view(batch, seq_len, -1)


def _apply_step_mlp(x: torch.Tensor, mlp: nn.ModuleList, seq_len: int) -> torch.Tensor:
    results = []
    for t in range(seq_len):
        step = x[:, t, :]
        for layer in mlp:
            step = layer(step)
        results.append(step)
    return torch.stack(results, dim=1)
