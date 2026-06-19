from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from CloneRL.models.torch.dino_da_recurrent import DinoDAVisualEncoder


RISK_PREFERENCE_KEY = "risk_preference"


class _RiskConditionedDinoDARecurrentBase(nn.Module):
    def _init_recurrent_base(
        self,
        *,
        dino_token_count: int,
        dino_dim: int,
        dino_proj_dim: int,
        depth_channels: int,
        depth_height: int,
        depth_width: int,
        depth_token_dim: int,
        visual_token_dim: int,
        visual_dim: int,
        proprioception_channels: int,
        gru_hidden_size: int,
        gru_num_layers: int,
        num_queries: int,
        num_heads: int,
        num_pool_blocks: int,
        dropout: float,
        action_input_dim: int = 0,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dino_token_count = int(dino_token_count)
        self.dino_dim = int(dino_dim)
        self.depth_channels = int(depth_channels)
        self.depth_height = int(depth_height)
        self.depth_width = int(depth_width)
        self.visual_dim = int(visual_dim)
        self.proprioception_channels = int(proprioception_channels)
        self.gru_hidden_size = int(gru_hidden_size)
        self.gru_num_layers = int(gru_num_layers)
        self.action_input_dim = int(action_input_dim)
        self.hidden_val = None

        self.visual_encoder = DinoDAVisualEncoder(
            dino_token_count=dino_token_count,
            dino_dim=dino_dim,
            dino_proj_dim=dino_proj_dim,
            depth_channels=depth_channels,
            depth_height=depth_height,
            depth_width=depth_width,
            depth_token_dim=depth_token_dim,
            visual_token_dim=visual_token_dim,
            visual_dim=visual_dim,
            num_queries=num_queries,
            num_heads=num_heads,
            num_pool_blocks=num_pool_blocks,
            dropout=dropout,
        )
        self.gru = nn.GRU(
            input_size=visual_dim + proprioception_channels + action_input_dim + 1,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
            bidirectional=False,
        )

    def _recurrent_input(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        dino_tokens = _state_value(state, "dino_tokens", "dino_tokens_t")
        da_depth = _state_value(state, "da_depth", "da_depth_t")
        proprio = _state_value(state, "proprioceptive", "proprio")
        risk_preference = _state_value(state, RISK_PREFERENCE_KEY)

        dino_tokens, da_depth, proprio = self._ensure_sequence_inputs(
            dino_tokens,
            da_depth,
            proprio,
        )
        batch, seq_len = dino_tokens.shape[:2]
        risk_preference = self._ensure_risk_preference(risk_preference, batch, seq_len, dino_tokens.device)

        dino_flat = dino_tokens.reshape(batch * seq_len, self.dino_token_count, self.dino_dim)
        da_flat = da_depth.reshape(
            batch * seq_len,
            self.depth_channels,
            self.depth_height,
            self.depth_width,
        )
        visual_flat = self.visual_encoder(dino_flat, da_flat)
        visual = visual_flat.reshape(batch, seq_len, self.visual_dim)

        features = [
            visual,
            proprio[..., : self.proprioception_channels],
        ]
        if self.action_input_dim:
            if action is None:
                raise ValueError("Risk-conditioned critic requires action input.")
            if action.dim() == 2:
                action = action.unsqueeze(1)
            action = action.to(device=visual.device, dtype=torch.float32)
            if action.shape[:2] != (batch, seq_len):
                raise ValueError(
                    f"Expected actions [B, S, A] with B={batch}, S={seq_len}, got {tuple(action.shape)}."
                )
            features.append(action[..., : self.action_input_dim])
        features.append(risk_preference)
        return torch.cat(features, dim=-1), batch

    def _ensure_sequence_inputs(
        self,
        dino_tokens: torch.Tensor,
        da_depth: torch.Tensor,
        proprio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        dino_tokens = dino_tokens.to(device=device)
        da_depth = da_depth.to(device=device)
        proprio = proprio.to(device=device, dtype=torch.float32)

        if dino_tokens.dim() == 3:
            dino_tokens = dino_tokens.unsqueeze(1)
        if da_depth.dim() == 4:
            da_depth = da_depth.unsqueeze(1)
        if proprio.dim() == 2:
            proprio = proprio.unsqueeze(1)

        if dino_tokens.dim() != 4:
            raise ValueError(f"Expected DINO tokens [B, S, 577, 384], got {tuple(dino_tokens.shape)}.")
        if tuple(dino_tokens.shape[-2:]) != (self.dino_token_count, self.dino_dim):
            expected = (self.dino_token_count, self.dino_dim)
            raise ValueError(f"Expected DINO token trailing shape {expected}, got {tuple(dino_tokens.shape[-2:])}.")
        if da_depth.dim() != 5:
            raise ValueError(f"Expected DA3 depth [B, S, 1, H, W], got {tuple(da_depth.shape)}.")
        expected_depth = (self.depth_channels, self.depth_height, self.depth_width)
        if tuple(da_depth.shape[-3:]) != expected_depth:
            raise ValueError(
                f"Expected DA3 depth trailing shape {expected_depth}, got {tuple(da_depth.shape[-3:])}."
            )
        if proprio.dim() != 3:
            raise ValueError(f"Expected proprioception [B, S, P], got {tuple(proprio.shape)}.")
        return dino_tokens, da_depth, proprio

    @staticmethod
    def _ensure_risk_preference(
        risk_preference: torch.Tensor,
        batch: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        risk_preference = risk_preference.to(device=device, dtype=torch.float32)
        if risk_preference.dim() == 1:
            risk_preference = risk_preference[:, None, None]
        elif risk_preference.dim() == 2:
            risk_preference = risk_preference.unsqueeze(1)
        if risk_preference.dim() != 3 or risk_preference.shape[0] != batch or risk_preference.shape[-1] != 1:
            raise ValueError(
                f"Expected risk preference [B], [B, 1], or [B, S, 1], got {tuple(risk_preference.shape)}."
            )
        if risk_preference.shape[1] == 1:
            risk_preference = risk_preference.expand(-1, seq_len, -1)
        elif risk_preference.shape[1] != seq_len:
            raise ValueError(
                f"Risk preference sequence length must be 1 or {seq_len}, got {risk_preference.shape[1]}."
            )
        if torch.any((risk_preference < 0) | (risk_preference > 1)):
            raise ValueError("Risk preference alpha must lie in [0, 1].")
        return risk_preference

    def _initial_hidden(self, batch: int, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            self.gru_num_layers,
            batch,
            self.gru_hidden_size,
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )

    def reset_hidden(self, batch_size: int = 1) -> None:
        self.hidden_val = torch.zeros(
            self.gru_num_layers,
            batch_size,
            self.gru_hidden_size,
            device=next(self.parameters()).device,
        )


class RiskConditionedDinoDARecurrentActorGaussian(_RiskConditionedDinoDARecurrentBase):
    def __init__(
        self,
        dino_token_count: int = 577,
        dino_dim: int = 384,
        dino_proj_dim: int = 192,
        depth_channels: int = 1,
        depth_height: int = 72,
        depth_width: int = 128,
        depth_token_dim: int = 64,
        visual_token_dim: int = 256,
        visual_dim: int = 512,
        proprioception_channels: int = 3,
        action_dim: int = 2,
        gru_hidden_size: int = 256,
        gru_num_layers: int = 2,
        num_queries: int = 8,
        num_heads: int = 4,
        num_pool_blocks: int = 2,
        dropout: float = 0.1,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
        state_independent_log_std: bool = True,
        device: str | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.state_independent_log_std = bool(state_independent_log_std)
        self._init_recurrent_base(
            dino_token_count=dino_token_count,
            dino_dim=dino_dim,
            dino_proj_dim=dino_proj_dim,
            depth_channels=depth_channels,
            depth_height=depth_height,
            depth_width=depth_width,
            depth_token_dim=depth_token_dim,
            visual_token_dim=visual_token_dim,
            visual_dim=visual_dim,
            proprioception_channels=proprioception_channels,
            gru_hidden_size=gru_hidden_size,
            gru_num_layers=gru_num_layers,
            num_queries=num_queries,
            num_heads=num_heads,
            num_pool_blocks=num_pool_blocks,
            dropout=dropout,
            device=device,
        )
        self.actor_body = nn.Sequential(
            nn.Linear(gru_hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.mean_head = nn.Sequential(nn.Linear(128, action_dim), nn.Tanh())
        if self.state_independent_log_std:
            self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        else:
            self.log_std_head = nn.Linear(128, action_dim)
        _init_linear_weights(self)

    def forward(self, state: Dict[str, torch.Tensor], hidden: torch.Tensor | None = None):
        gru_input, batch = self._recurrent_input(state)
        if hidden is None:
            hidden = self._initial_hidden(batch, gru_input)
        sequence, hidden = self.gru(gru_input, hidden)
        features = self.actor_body(sequence)
        mean = self.mean_head(features)
        if self.state_independent_log_std:
            log_std = self.log_std.clamp(self.min_log_std, self.max_log_std).expand_as(mean)
        else:
            log_std = self.log_std_head(features).clamp(self.min_log_std, self.max_log_std)
        return torch.distributions.Normal(mean, log_std.exp()), hidden

    def get_action(self, state: Dict[str, torch.Tensor], deterministic: bool = True):
        with torch.no_grad():
            distribution, self.hidden_val = self.forward(state, self.hidden_val)
            action = distribution.mean if deterministic else distribution.sample()
            return action.squeeze(1) if action.dim() == 3 and action.shape[1] == 1 else action

    def validate(self, state: Dict[str, torch.Tensor], done: torch.Tensor | None = None):
        with torch.no_grad():
            distribution, new_hidden = self.forward(state, self.hidden_val)
            if done is not None:
                done_flat = done.view(-1) if done.dim() > 1 else done
                keep = (~done_flat.bool()).unsqueeze(0).unsqueeze(2).expand_as(new_hidden)
                self.hidden_val = new_hidden * keep.float()
            else:
                self.hidden_val = new_hidden
            action = distribution.mean
            return action.squeeze(1) if action.dim() == 3 and action.shape[1] == 1 else action


class RiskConditionedDinoDARecurrentValue(_RiskConditionedDinoDARecurrentBase):
    def __init__(
        self,
        dino_token_count: int = 577,
        dino_dim: int = 384,
        dino_proj_dim: int = 192,
        depth_channels: int = 1,
        depth_height: int = 72,
        depth_width: int = 128,
        depth_token_dim: int = 64,
        visual_token_dim: int = 256,
        visual_dim: int = 512,
        proprioception_channels: int = 3,
        gru_hidden_size: int = 256,
        gru_num_layers: int = 2,
        num_queries: int = 8,
        num_heads: int = 4,
        num_pool_blocks: int = 2,
        dropout: float = 0.1,
        device: str | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        self._init_recurrent_base(
            dino_token_count=dino_token_count,
            dino_dim=dino_dim,
            dino_proj_dim=dino_proj_dim,
            depth_channels=depth_channels,
            depth_height=depth_height,
            depth_width=depth_width,
            depth_token_dim=depth_token_dim,
            visual_token_dim=visual_token_dim,
            visual_dim=visual_dim,
            proprioception_channels=proprioception_channels,
            gru_hidden_size=gru_hidden_size,
            gru_num_layers=gru_num_layers,
            num_queries=num_queries,
            num_heads=num_heads,
            num_pool_blocks=num_pool_blocks,
            dropout=dropout,
            device=device,
        )
        self.value_head = _scalar_head(gru_hidden_size, dropout)
        _init_linear_weights(self)

    def forward(self, state: Dict[str, torch.Tensor], hidden: torch.Tensor | None = None):
        gru_input, batch = self._recurrent_input(state)
        if hidden is None:
            hidden = self._initial_hidden(batch, gru_input)
        sequence, hidden = self.gru(gru_input, hidden)
        return self.value_head(sequence), hidden


class RiskConditionedDinoDARecurrentQNetwork(_RiskConditionedDinoDARecurrentBase):
    def __init__(
        self,
        dino_token_count: int = 577,
        dino_dim: int = 384,
        dino_proj_dim: int = 192,
        depth_channels: int = 1,
        depth_height: int = 72,
        depth_width: int = 128,
        depth_token_dim: int = 64,
        visual_token_dim: int = 256,
        visual_dim: int = 512,
        proprioception_channels: int = 3,
        action_dim: int = 2,
        gru_hidden_size: int = 256,
        gru_num_layers: int = 2,
        num_queries: int = 8,
        num_heads: int = 4,
        num_pool_blocks: int = 2,
        dropout: float = 0.1,
        device: str | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self._init_recurrent_base(
            dino_token_count=dino_token_count,
            dino_dim=dino_dim,
            dino_proj_dim=dino_proj_dim,
            depth_channels=depth_channels,
            depth_height=depth_height,
            depth_width=depth_width,
            depth_token_dim=depth_token_dim,
            visual_token_dim=visual_token_dim,
            visual_dim=visual_dim,
            proprioception_channels=proprioception_channels,
            gru_hidden_size=gru_hidden_size,
            gru_num_layers=gru_num_layers,
            num_queries=num_queries,
            num_heads=num_heads,
            num_pool_blocks=num_pool_blocks,
            dropout=dropout,
            action_input_dim=action_dim,
            device=device,
        )
        self.q_head = _scalar_head(gru_hidden_size, dropout)
        _init_linear_weights(self)

    def forward(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ):
        gru_input, batch = self._recurrent_input(state, action)
        if hidden is None:
            hidden = self._initial_hidden(batch, gru_input)
        sequence, hidden = self.gru(gru_input, hidden)
        return self.q_head(sequence), hidden


class RiskConditionedDinoDARecurrentTwinQ(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.device = kwargs.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        self.q1 = RiskConditionedDinoDARecurrentQNetwork(**kwargs)
        self.q2 = RiskConditionedDinoDARecurrentQNetwork(**kwargs)

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

    def reset_hidden(self, batch_size: int = 1) -> None:
        self.q1.reset_hidden(batch_size)
        self.q2.reset_hidden(batch_size)


def _scalar_head(input_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Linear(128, 1),
    )


def _init_linear_weights(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


def _state_value(state: Dict[str, torch.Tensor], *keys: str) -> torch.Tensor:
    for key in keys:
        if key in state:
            return state[key]
    raise KeyError(f"State is missing required key. Tried: {', '.join(keys)}")

