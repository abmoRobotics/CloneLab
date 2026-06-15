from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoDABCRecurrentPolicy(nn.Module):
    """BC-only recurrent student policy for cached DINOv3 tokens and DA3 depth."""

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
        gru_hidden_size: int = 512,
        gru_num_layers: int = 2,
        num_queries: int = 8,
        num_heads: int = 4,
        num_pool_blocks: int = 2,
        dropout: float = 0.1,
        device: str | None = None,
        **_: object,
    ):
        super().__init__()
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dino_token_count = int(dino_token_count)
        self.dino_dim = int(dino_dim)
        self.dino_proj_dim = int(dino_proj_dim)
        self.depth_channels = int(depth_channels)
        self.depth_height = int(depth_height)
        self.depth_width = int(depth_width)
        self.depth_token_dim = int(depth_token_dim)
        self.visual_token_dim = int(visual_token_dim)
        self.visual_dim = int(visual_dim)
        self.proprioception_channels = int(proprioception_channels)
        self.action_dim = int(action_dim)
        self.gru_hidden_size = int(gru_hidden_size)
        self.gru_num_layers = int(gru_num_layers)
        self.num_queries = int(num_queries)
        self.hidden_val = None

        if self.dino_token_count != 577 or self.dino_dim != 384:
            raise ValueError("DinoDABCRecurrentPolicy currently expects DINO tokens [577, 384].")
        if (self.depth_channels, self.depth_height, self.depth_width) != (1, 72, 128):
            raise ValueError("DinoDABCRecurrentPolicy currently expects DA3 depth [1, 72, 128].")

        self.dino_projection = nn.Sequential(
            nn.LayerNorm(dino_dim),
            nn.Linear(dino_dim, dino_proj_dim),
            nn.GELU(),
            nn.Linear(dino_proj_dim, dino_proj_dim),
        )

        self.depth_cnn = nn.Sequential(
            nn.Conv2d(depth_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        self.depth_token_projection = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, depth_token_dim),
        )
        self.depth_gate = nn.Sequential(
            nn.Linear(dino_proj_dim + 128, 128),
            nn.GELU(),
            nn.Linear(128, depth_token_dim),
            nn.Sigmoid(),
        )

        self.token_mlp = nn.Sequential(
            nn.LayerNorm(dino_proj_dim + depth_token_dim),
            nn.Linear(dino_proj_dim + depth_token_dim, visual_token_dim),
            nn.GELU(),
            nn.Linear(visual_token_dim, visual_token_dim),
        )
        self.cls_projection = nn.Linear(dino_proj_dim, visual_token_dim)
        self.query_tokens = nn.Parameter(torch.randn(num_queries, visual_token_dim) * 0.02)
        self.pool_blocks = nn.ModuleList(
            [PerceiverPoolingBlock(visual_token_dim, num_heads, dropout=dropout) for _ in range(num_pool_blocks)]
        )
        self.visual_summary = nn.Sequential(
            nn.Linear(visual_token_dim * 3, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, visual_dim),
            nn.LayerNorm(visual_dim),
        )

        self.gru = nn.GRU(
            input_size=visual_dim + proprioception_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.action_head = nn.Sequential(
            nn.Linear(gru_hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state: Dict[str, torch.Tensor], hidden: torch.Tensor | None = None):
        dino_tokens = _state_value(state, "dino_tokens", "dino_tokens_t")
        da_depth = _state_value(state, "da_depth", "da_depth_t")
        proprio = _state_value(state, "proprioceptive", "proprio")

        dino_tokens, da_depth, proprio = self._ensure_sequence_inputs(dino_tokens, da_depth, proprio)
        batch, seq_len = dino_tokens.shape[:2]

        dino_flat = dino_tokens.reshape(batch * seq_len, self.dino_token_count, self.dino_dim)
        da_flat = da_depth.reshape(batch * seq_len, self.depth_channels, self.depth_height, self.depth_width)

        z_visual_flat = self.encode_visual(dino_flat, da_flat)
        z_visual = z_visual_flat.reshape(batch, seq_len, self.visual_dim)
        z_gru_input = torch.cat([z_visual, proprio[..., : self.proprioception_channels]], dim=-1)

        if hidden is None:
            hidden = torch.zeros(self.gru_num_layers, batch, self.gru_hidden_size, device=z_gru_input.device)
        h_seq, hidden = self.gru(z_gru_input, hidden)
        return self.action_head(h_seq), hidden

    def encode_visual(self, dino_tokens: torch.Tensor, da_depth: torch.Tensor) -> torch.Tensor:
        dino_tokens = dino_tokens.float()
        da_depth = da_depth.float()

        dino_projected = self.dino_projection(dino_tokens)
        dino_cls = dino_projected[:, 0]
        dino_patch = dino_projected[:, 1:]

        depth_feat = self.depth_cnn(da_depth)
        depth_tokens = depth_feat.flatten(2).transpose(1, 2)
        depth_tokens = self.depth_token_projection(depth_tokens)
        depth_global = depth_feat.mean(dim=(2, 3))

        depth_gate = self.depth_gate(torch.cat([dino_cls, depth_global], dim=-1))
        depth_tokens_gated = depth_tokens * depth_gate[:, None, :]

        fused_patch_tokens = torch.cat([dino_patch, depth_tokens_gated], dim=-1)
        fused_patch_tokens = self.token_mlp(fused_patch_tokens)
        cls_256 = self.cls_projection(dino_cls)
        visual_tokens = torch.cat([cls_256[:, None, :], fused_patch_tokens], dim=1)

        latent_tokens = self.query_tokens.unsqueeze(0).expand(dino_tokens.shape[0], -1, -1)
        for block in self.pool_blocks:
            latent_tokens = block(latent_tokens, visual_tokens)

        latent_mean = latent_tokens.mean(dim=1)
        latent_max = latent_tokens.max(dim=1).values
        visual_summary = torch.cat([latent_mean, latent_max, cls_256], dim=-1)
        return self.visual_summary(visual_summary)

    def get_action(self, state: Dict[str, torch.Tensor], deterministic: bool = True):
        with torch.no_grad():
            action, self.hidden_val = self.forward(state, self.hidden_val)
            return action.squeeze(1) if action.dim() == 3 and action.shape[1] == 1 else action

    def reset_hidden(self, batch_size: int = 1):
        self.hidden_val = torch.zeros(self.gru_num_layers, batch_size, self.gru_hidden_size).to(self.device)

    def validate(self, state: Dict[str, torch.Tensor], done: torch.Tensor | None = None):
        with torch.no_grad():
            action, new_hidden = self.forward(state, self.hidden_val)
            if done is not None:
                done_flat = done.view(-1) if done.dim() > 1 else done
                not_done = ~done_flat.bool()
                keep = not_done.unsqueeze(0).unsqueeze(2).expand_as(new_hidden)
                self.hidden_val = new_hidden * keep.float()
            else:
                self.hidden_val = new_hidden
            return action.squeeze(1) if action.dim() == 3 and action.shape[1] == 1 else action

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
            raise ValueError(f"Expected dino tokens [B, S, 577, 384], got {tuple(dino_tokens.shape)}.")
        if da_depth.dim() != 5:
            raise ValueError(f"Expected DA3 depth [B, S, 1, 72, 128], got {tuple(da_depth.shape)}.")
        if proprio.dim() != 3:
            raise ValueError(f"Expected proprio [B, S, 3], got {tuple(proprio.shape)}.")
        return dino_tokens, da_depth, proprio


class DinoDAVisualEncoder(nn.Module):
    """Frame encoder for cached DINOv3 tokens plus DA3 pseudo-depth."""

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
        num_queries: int = 8,
        num_heads: int = 4,
        num_pool_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dino_token_count = int(dino_token_count)
        self.dino_dim = int(dino_dim)
        self.dino_proj_dim = int(dino_proj_dim)
        self.depth_channels = int(depth_channels)
        self.depth_height = int(depth_height)
        self.depth_width = int(depth_width)
        self.depth_token_dim = int(depth_token_dim)
        self.visual_token_dim = int(visual_token_dim)
        self.visual_dim = int(visual_dim)
        self.num_queries = int(num_queries)
        self.dino_patch_grid = (18, 32)

        if self.dino_token_count != 577 or self.dino_dim != 384:
            raise ValueError("DinoDAVisualEncoder currently expects DINO tokens [577, 384].")
        if self.depth_channels != 1:
            raise ValueError("DinoDAVisualEncoder currently expects one DA3 depth channel.")

        self.dino_projection = nn.Sequential(
            nn.LayerNorm(dino_dim),
            nn.Linear(dino_dim, dino_proj_dim),
            nn.GELU(),
            nn.Linear(dino_proj_dim, dino_proj_dim),
        )

        self.depth_cnn = nn.Sequential(
            nn.Conv2d(depth_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        self.depth_token_projection = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, depth_token_dim),
        )
        self.depth_gate = nn.Sequential(
            nn.Linear(dino_proj_dim + 128, 128),
            nn.GELU(),
            nn.Linear(128, depth_token_dim),
            nn.Sigmoid(),
        )

        self.token_mlp = nn.Sequential(
            nn.LayerNorm(dino_proj_dim + depth_token_dim),
            nn.Linear(dino_proj_dim + depth_token_dim, visual_token_dim),
            nn.GELU(),
            nn.Linear(visual_token_dim, visual_token_dim),
        )
        self.cls_projection = nn.Linear(dino_proj_dim, visual_token_dim)
        self.query_tokens = nn.Parameter(torch.randn(num_queries, visual_token_dim) * 0.02)
        self.pool_blocks = nn.ModuleList(
            [PerceiverPoolingBlock(visual_token_dim, num_heads, dropout=dropout) for _ in range(num_pool_blocks)]
        )
        self.visual_summary = nn.Sequential(
            nn.Linear(visual_token_dim * 3, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, visual_dim),
            nn.LayerNorm(visual_dim),
        )

    def forward(self, dino_tokens: torch.Tensor, da_depth: torch.Tensor) -> torch.Tensor:
        return self.encode_visual(dino_tokens, da_depth)

    def encode_visual(self, dino_tokens: torch.Tensor, da_depth: torch.Tensor) -> torch.Tensor:
        dino_tokens = dino_tokens.float()
        da_depth = da_depth.float()

        dino_projected = self.dino_projection(dino_tokens)
        dino_cls = dino_projected[:, 0]
        dino_patch = dino_projected[:, 1:]

        depth_feat = self.depth_cnn(da_depth)
        if depth_feat.shape[-2:] != self.dino_patch_grid:
            depth_feat = F.adaptive_avg_pool2d(depth_feat, self.dino_patch_grid)
        depth_tokens = depth_feat.flatten(2).transpose(1, 2)
        depth_tokens = self.depth_token_projection(depth_tokens)
        depth_global = depth_feat.mean(dim=(2, 3))

        if depth_tokens.shape[1] != dino_patch.shape[1]:
            raise ValueError(
                "Depth token grid does not match DINO patch grid: "
                f"{depth_tokens.shape[1]} depth tokens vs {dino_patch.shape[1]} DINO patch tokens. "
                "Check the configured DINO token count and DA depth aspect ratio."
            )

        depth_gate = self.depth_gate(torch.cat([dino_cls, depth_global], dim=-1))
        depth_tokens_gated = depth_tokens * depth_gate[:, None, :]

        fused_patch_tokens = torch.cat([dino_patch, depth_tokens_gated], dim=-1)
        fused_patch_tokens = self.token_mlp(fused_patch_tokens)
        cls_256 = self.cls_projection(dino_cls)
        visual_tokens = torch.cat([cls_256[:, None, :], fused_patch_tokens], dim=1)

        latent_tokens = self.query_tokens.unsqueeze(0).expand(dino_tokens.shape[0], -1, -1)
        for block in self.pool_blocks:
            latent_tokens = block(latent_tokens, visual_tokens)

        latent_mean = latent_tokens.mean(dim=1)
        latent_max = latent_tokens.max(dim=1).values
        visual_summary = torch.cat([latent_mean, latent_max, cls_256], dim=-1)
        return self.visual_summary(visual_summary)


class _DinoDARecurrentBase(nn.Module):
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
        recurrent_action_dim: int = 0,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dino_token_count = int(dino_token_count)
        self.dino_dim = int(dino_dim)
        self.dino_proj_dim = int(dino_proj_dim)
        self.depth_channels = int(depth_channels)
        self.depth_height = int(depth_height)
        self.depth_width = int(depth_width)
        self.depth_token_dim = int(depth_token_dim)
        self.visual_token_dim = int(visual_token_dim)
        self.visual_dim = int(visual_dim)
        self.proprioception_channels = int(proprioception_channels)
        self.gru_hidden_size = int(gru_hidden_size)
        self.gru_num_layers = int(gru_num_layers)
        self.num_queries = int(num_queries)
        self.recurrent_action_dim = int(recurrent_action_dim)
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
            input_size=visual_dim + proprioception_channels + recurrent_action_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
            bidirectional=False,
        )

    def encode_visual(self, dino_tokens: torch.Tensor, da_depth: torch.Tensor) -> torch.Tensor:
        return self.visual_encoder.encode_visual(dino_tokens, da_depth)

    def _recurrent_input(self, state: Dict[str, torch.Tensor], action: torch.Tensor | None = None):
        dino_tokens = _state_value(state, "dino_tokens", "dino_tokens_t")
        da_depth = _state_value(state, "da_depth", "da_depth_t")
        proprio = _state_value(state, "proprioceptive", "proprio")

        dino_tokens, da_depth, proprio = self._ensure_sequence_inputs(dino_tokens, da_depth, proprio)
        batch, seq_len = dino_tokens.shape[:2]

        dino_flat = dino_tokens.reshape(batch * seq_len, self.dino_token_count, self.dino_dim)
        da_flat = da_depth.reshape(batch * seq_len, self.depth_channels, self.depth_height, self.depth_width)

        z_visual_flat = self.encode_visual(dino_flat, da_flat)
        z_visual = z_visual_flat.reshape(batch, seq_len, self.visual_dim)
        features = [z_visual, proprio[..., : self.proprioception_channels]]
        if self.recurrent_action_dim > 0:
            if action is None:
                raise ValueError("This recurrent module requires action input.")
            if action.dim() == 2:
                action = action.unsqueeze(1)
            action = action.to(device=z_visual.device, dtype=torch.float32)
            if action.shape[:2] != (batch, seq_len):
                raise ValueError(f"Expected actions [B, S, A] with B={batch}, S={seq_len}, got {tuple(action.shape)}.")
            features.append(action[..., : self.recurrent_action_dim])
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
            raise ValueError(f"Expected dino tokens [B, S, 577, 384], got {tuple(dino_tokens.shape)}.")
        if da_depth.dim() != 5:
            raise ValueError(f"Expected DA3 depth [B, S, 1, H, W], got {tuple(da_depth.shape)}.")
        if tuple(da_depth.shape[-3:]) != (self.depth_channels, self.depth_height, self.depth_width):
            expected = (self.depth_channels, self.depth_height, self.depth_width)
            raise ValueError(f"Expected DA3 depth trailing shape {expected}, got {tuple(da_depth.shape[-3:])}.")
        if proprio.dim() != 3:
            raise ValueError(f"Expected proprio [B, S, 3], got {tuple(proprio.shape)}.")
        return dino_tokens, da_depth, proprio


class DinoDARecurrentActorGaussian(_DinoDARecurrentBase):
    """Gaussian recurrent actor for DINO/DA IQL."""

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
    ):
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
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state: Dict[str, torch.Tensor], hidden: torch.Tensor | None = None):
        gru_input, batch = self._recurrent_input(state)
        if hidden is None:
            hidden = torch.zeros(self.gru_num_layers, batch, self.gru_hidden_size, device=gru_input.device)
        h_seq, hidden = self.gru(gru_input, hidden)
        features = self.actor_body(h_seq)
        mean = self.mean_head(features)
        if self.state_independent_log_std:
            log_std = torch.clamp(self.log_std, min=self.min_log_std, max=self.max_log_std).expand_as(mean)
        else:
            log_std = torch.clamp(self.log_std_head(features), min=self.min_log_std, max=self.max_log_std)
        return torch.distributions.Normal(mean, log_std.exp()), hidden

    def get_action(self, state: Dict[str, torch.Tensor], deterministic: bool = True):
        with torch.no_grad():
            dist, self.hidden_val = self.forward(state, self.hidden_val)
            action = dist.mean if deterministic else dist.sample()
            return action.squeeze(1) if action.dim() == 3 and action.shape[1] == 1 else action

    def reset_hidden(self, batch_size: int = 1):
        self.hidden_val = torch.zeros(self.gru_num_layers, batch_size, self.gru_hidden_size).to(self.device)

    def validate(self, state: Dict[str, torch.Tensor], done: torch.Tensor | None = None):
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
            return action.squeeze(1) if action.dim() == 3 and action.shape[1] == 1 else action


class DinoDARecurrentValue(_DinoDARecurrentBase):
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
    ):
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
        self.value_head = nn.Sequential(
            nn.Linear(gru_hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state: Dict[str, torch.Tensor], hidden: torch.Tensor | None = None):
        gru_input, batch = self._recurrent_input(state)
        if hidden is None:
            hidden = torch.zeros(self.gru_num_layers, batch, self.gru_hidden_size, device=gru_input.device)
        h_seq, hidden = self.gru(gru_input, hidden)
        return self.value_head(h_seq), hidden

    def reset_hidden(self, batch_size: int = 1):
        self.hidden_val = torch.zeros(self.gru_num_layers, batch_size, self.gru_hidden_size).to(self.device)


class DinoDARecurrentQNetwork(_DinoDARecurrentBase):
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
    ):
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
            recurrent_action_dim=action_dim,
            device=device,
        )
        self.q_head = nn.Sequential(
            nn.Linear(gru_hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
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
        gru_input, batch = self._recurrent_input(state, action)
        if hidden is None:
            hidden = torch.zeros(self.gru_num_layers, batch, self.gru_hidden_size, device=gru_input.device)
        h_seq, hidden = self.gru(gru_input, hidden)
        return self.q_head(h_seq), hidden

    def reset_hidden(self, batch_size: int = 1):
        self.hidden_val = torch.zeros(self.gru_num_layers, batch_size, self.gru_hidden_size).to(self.device)


class DinoDARecurrentTwinQ(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = kwargs.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        self.q1 = DinoDARecurrentQNetwork(**kwargs)
        self.q2 = DinoDARecurrentQNetwork(**kwargs)

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

    def reset_hidden(self, batch_size: int = 1):
        self.q1.reset_hidden(batch_size)
        self.q2.reset_hidden(batch_size)


class PerceiverPoolingBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.query_cross_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.query_self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Linear(512, dim),
        )

    def forward(self, queries: torch.Tensor, visual_tokens: torch.Tensor) -> torch.Tensor:
        query_norm = self.query_cross_norm(queries)
        context_norm = self.context_norm(visual_tokens)
        cross_out, _ = self.cross_attn(query_norm, context_norm, context_norm, need_weights=False)
        queries = queries + cross_out

        self_query = self.query_self_norm(queries)
        self_out, _ = self.self_attn(self_query, self_query, self_query, need_weights=False)
        queries = queries + self_out
        return queries + self.mlp(self.mlp_norm(queries))


def _state_value(state: Dict[str, torch.Tensor], *keys: str) -> torch.Tensor:
    for key in keys:
        if key in state:
            return state[key]
    raise KeyError(f"State is missing required key. Tried: {', '.join(keys)}")
