from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


SCHEMA_NAME = "rlroverlab.offline_dino_da3_v1"


def is_rlroverlab_dino_da_features(file_path: str | Path) -> bool:
    """Return True when an HDF5 file contains precomputed DINO/DA3 observations."""
    try:
        with h5py.File(file_path, "r") as file:
            return file.attrs.get("schema_name") == SCHEMA_NAME
    except OSError:
        return False


class RLRoverLabDinoDARandomSequenceDataset(Dataset):
    """Random contiguous-sequence loader for precomputed DINO/DA3 BC-RNN data."""

    def __init__(
        self,
        file_path: str,
        sequence_length: int = 8,
        min_idx: int = 0,
        max_idx: Optional[int] = None,
        total_samples: int = 100000,
        proprioceptive_keys: List[str] = ["distance", "heading", "angle_diff"],
        return_hidden_reset_mask: bool = False,
    ):
        self.file_path = str(file_path)
        self.sequence_length = int(sequence_length)
        self.min_idx = int(min_idx)
        self.max_idx = max_idx
        self.total_samples = int(total_samples)
        self.proprioceptive_keys = list(proprioceptive_keys)
        self.return_hidden_reset_mask = bool(return_hidden_reset_mask)

        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")

        with h5py.File(self.file_path, "r") as file:
            self._validate_file(file)
            self.transition_count = int(file.attrs.get("total_transitions", len(file["transitions/actions"])))
            self.observation_count = int(file.attrs.get("total_observations", len(file["observations/dino_tokens_t"])))
            self.obs_offsets = file["index/obs_offsets"][:].astype(np.int64)
            self.transition_offsets = file["index/transition_offsets"][:].astype(np.int64)
            self.episode_lengths = file["index/episode_lengths"][:].astype(np.int64)

        self.episode_count = int(len(self.episode_lengths))
        episode_stop = self.episode_count if max_idx is None else min(int(max_idx), self.episode_count)
        episode_start = max(0, min(self.min_idx, episode_stop))
        self.selected_episode_ids = np.arange(episode_start, episode_stop, dtype=np.int64)
        if self.selected_episode_ids.size == 0:
            raise ValueError(f"No episodes selected from {self.file_path}: min_idx={min_idx}, max_idx={max_idx}")

        self.valid_sequences = self._sequence_starts()

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int):
        episode, start_idx = self.valid_sequences[np.random.randint(0, len(self.valid_sequences))]
        end_idx = start_idx + self.sequence_length
        transition_start = int(self.transition_offsets[episode]) + start_idx
        obs_start = int(self.obs_offsets[episode]) + start_idx

        with h5py.File(self.file_path, "r") as file:
            transition_slice = slice(transition_start, transition_start + self.sequence_length)
            obs_slice = slice(obs_start, obs_start + self.sequence_length)
            next_obs_slice = slice(obs_start + 1, obs_start + self.sequence_length + 1)

            actions, rewards, dones = self._read_transition_arrays(file, transition_slice)
            obs = self._read_observation_sequence(file, obs_slice)
            next_obs = self._read_observation_sequence(file, next_obs_slice)

        weights = torch.ones_like(rewards)
        masks = torch.ones_like(rewards)
        if not self.return_hidden_reset_mask:
            return obs, actions, rewards, next_obs, dones, weights, masks

        hidden_reset = torch.zeros(self.sequence_length, 1, dtype=torch.float32)
        hidden_reset[0] = 1.0 if start_idx == 0 else 0.0
        for timestep in range(self.sequence_length - 1):
            if bool(dones[timestep].item()):
                hidden_reset[timestep + 1] = 1.0
        return obs, actions, rewards, next_obs, dones, weights, masks, hidden_reset

    def _validate_file(self, file: h5py.File) -> None:
        schema = file.attrs.get("schema_name")
        if schema != SCHEMA_NAME:
            raise ValueError(f"Expected precomputed DINO/DA3 schema {SCHEMA_NAME!r}, got {schema!r}.")

        required_paths = (
            "observations/dino_tokens_t",
            "observations/da_depth_t",
            "observations/state",
            "transitions/actions",
            "transitions/rewards",
            "index/episode_lengths",
            "index/obs_offsets",
            "index/transition_offsets",
        )
        for path in required_paths:
            if path not in file:
                raise ValueError(f"Precomputed DINO/DA3 dataset is missing required path: {path}")
        for key in self.proprioceptive_keys:
            if f"observations/state/{key}" not in file:
                raise ValueError(f"Precomputed DINO/DA3 dataset is missing proprioceptive key: {key}")

        dino_shape = file["observations/dino_tokens_t"].shape[1:]
        da_shape = file["observations/da_depth_t"].shape[1:]
        if dino_shape != (577, 384):
            raise ValueError(f"Expected observations/dino_tokens_t shape [N, 577, 384], got [N, {dino_shape}].")
        if da_shape != (1, 72, 128):
            raise ValueError(f"Expected observations/da_depth_t shape [N, 1, 72, 128], got [N, {da_shape}].")

    def _sequence_starts(self) -> List[Tuple[int, int]]:
        starts: List[Tuple[int, int]] = []
        for episode in self.selected_episode_ids:
            length = int(self.episode_lengths[episode])
            max_start = length - self.sequence_length
            if max_start < 0:
                continue
            starts.extend((int(episode), start) for start in range(max_start + 1))
        if not starts:
            raise ValueError(
                f"No valid DINO/DA3 sequences found. Selected episodes must contain at least "
                f"{self.sequence_length} transitions."
            )
        return starts

    def _read_transition_arrays(self, file: h5py.File, transition_slice: slice) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions = torch.from_numpy(np.asarray(file["transitions/actions"][transition_slice])).float()
        rewards = torch.from_numpy(np.asarray(file["transitions/rewards"][transition_slice])).float()
        if "transitions/dones" in file:
            dones_np = np.asarray(file["transitions/dones"][transition_slice])
        else:
            terminals = np.asarray(file["transitions/terminals"][transition_slice], dtype=np.bool_)
            timeouts = np.asarray(file["transitions/timeouts"][transition_slice], dtype=np.bool_)
            dones_np = np.logical_or(terminals, timeouts)
        dones = torch.from_numpy(np.asarray(dones_np))

        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        rewards = _ensure_column_tensor(rewards)
        dones = _ensure_column_tensor(dones)
        return actions, rewards, dones

    def _read_observation_sequence(self, file: h5py.File, obs_slice: slice) -> Dict[str, torch.Tensor]:
        proprio_tensors = [
            torch.from_numpy(np.asarray(file[f"observations/state/{key}"][obs_slice])).float()
            for key in self.proprioceptive_keys
        ]
        proprioceptive = torch.cat([_ensure_column_tensor(tensor) for tensor in proprio_tensors], dim=-1)
        return {
            "dino_tokens": torch.from_numpy(np.asarray(file["observations/dino_tokens_t"][obs_slice])),
            "da_depth": torch.from_numpy(np.asarray(file["observations/da_depth_t"][obs_slice])),
            "proprioceptive": proprioceptive,
        }


def _ensure_column_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.reshape(1)
    if tensor.ndim == 1:
        return tensor.unsqueeze(-1)
    return tensor
