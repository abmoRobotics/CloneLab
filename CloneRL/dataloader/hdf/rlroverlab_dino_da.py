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
    ok, _ = rlroverlab_dino_da_feature_status(file_path)
    return ok


def rlroverlab_dino_da_feature_status(file_path: str | Path) -> Tuple[bool, str]:
    """Return whether an HDF5 file is a complete cached DINO/DA3 dataset, plus a diagnostic."""
    try:
        with h5py.File(file_path, "r") as file:
            schema = file.attrs.get("schema_name")
            if schema != SCHEMA_NAME:
                return False, f"schema_name={schema!r}, expected {SCHEMA_NAME!r}"
            writer_status = file.attrs.get("writer_status")
            if writer_status is not None and writer_status != "complete":
                return False, f"writer_status={writer_status!r}"
            transitions = int(file.attrs.get("total_transitions", len(file.get("transitions/actions", ()))))
            if transitions <= 0:
                return False, f"total_transitions={transitions}"
            return True, f"complete DINO/DA3 dataset with {transitions} transitions"
    except BlockingIOError as exc:
        return False, f"locked by another process: {exc}"
    except OSError as exc:
        return False, f"not readable as HDF5: {exc}"


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
        return_next_obs: bool = False,
        expected_da_shape: Optional[Tuple[int, int, int]] = None,
    ):
        self.file_path = str(file_path)
        self.sequence_length = int(sequence_length)
        self.min_idx = int(min_idx)
        self.max_idx = max_idx
        self.total_samples = int(total_samples)
        self.proprioceptive_keys = list(proprioceptive_keys)
        self.return_hidden_reset_mask = bool(return_hidden_reset_mask)
        self.return_next_obs = bool(return_next_obs)
        self.expected_da_shape = expected_da_shape
        self._file: Optional[h5py.File] = None

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
        transition_start = int(self.transition_offsets[episode]) + start_idx
        obs_start = int(self.obs_offsets[episode]) + start_idx

        file = self._get_file()
        transition_slice = slice(transition_start, transition_start + self.sequence_length)
        obs_slice = slice(obs_start, obs_start + self.sequence_length)

        actions, rewards, dones = self._read_transition_arrays(file, transition_slice)
        obs = self._read_observation_sequence(file, obs_slice)

        if self.return_next_obs:
            next_obs_slice = slice(obs_start + 1, obs_start + self.sequence_length + 1)
            next_obs = self._read_observation_sequence(file, next_obs_slice)
        else:
            # BC-RNN does not use next_state; avoid a second large random HDF5 read
            # and a duplicate DINO-token transfer to the GPU.
            next_obs = {}

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

    def _get_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.file_path, "r")
        return self._file

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __del__(self):
        file = getattr(self, "_file", None)
        if file is not None:
            try:
                file.close()
            except Exception:
                pass

    def _validate_file(self, file: h5py.File) -> None:
        schema = file.attrs.get("schema_name")
        if schema != SCHEMA_NAME:
            raise ValueError(f"Expected precomputed DINO/DA3 schema {SCHEMA_NAME!r}, got {schema!r}.")
        writer_status = file.attrs.get("writer_status")
        if writer_status is not None and writer_status != "complete":
            raise ValueError(f"Precomputed DINO/DA3 dataset is not complete: writer_status={writer_status!r}.")

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
        expected_da_shape = self.expected_da_shape or da_shape
        if tuple(da_shape) != tuple(expected_da_shape):
            raise ValueError(
                f"Expected observations/da_depth_t shape [N, {tuple(expected_da_shape)}], got [N, {da_shape}]."
            )

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

    def _read_transition_arrays(
        self,
        file: h5py.File,
        transition_slice: slice,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


class RLRoverLabDinoDAMultiFileRandomSequenceDataset(Dataset):
    """Random sequence loader over multiple cached DINO/DA3 HDF5 files.

    This is used for DAgger aggregation: keep the original large teacher dataset
    in place, add small student-visited shards, and sample from the union without
    materializing a second huge HDF5 file.
    """

    def __init__(
        self,
        datasets: List[RLRoverLabDinoDARandomSequenceDataset],
        total_samples: int = 100000,
    ):
        if not datasets:
            raise ValueError("At least one DINO/DA dataset is required.")
        self.datasets = list(datasets)
        self.total_samples = int(total_samples)
        weights = np.asarray([len(dataset.valid_sequences) for dataset in self.datasets], dtype=np.float64)
        if np.any(weights <= 0):
            raise ValueError("All DINO/DA datasets must contain at least one valid sequence.")
        self.sample_probabilities = weights / weights.sum()

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int):
        dataset_index = int(np.random.choice(len(self.datasets), p=self.sample_probabilities))
        return self.datasets[dataset_index][idx]


def _ensure_column_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.reshape(1)
    if tensor.ndim == 1:
        return tensor.unsqueeze(-1)
    return tensor
