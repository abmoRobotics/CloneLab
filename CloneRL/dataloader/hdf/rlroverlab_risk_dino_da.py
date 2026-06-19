from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch

from CloneRL.dataloader.hdf.rlroverlab_dino_da import (
    RLRoverLabDinoDAMultiFileRandomSequenceDataset,
    RLRoverLabDinoDARandomSequenceDataset,
)


RISK_CLEARANCE_STATE_KEY = "_risk_clearance"
DEFAULT_CLEARANCE_PATH = "transitions/extra/risk/min_distance_to_rock"


class RLRoverLabRiskDinoDARandomSequenceDataset(RLRoverLabDinoDARandomSequenceDataset):
    """DINO/DA recurrent dataset with privileged transition-level clearance."""

    def __init__(
        self,
        file_path: str,
        *,
        clearance_path: str = DEFAULT_CLEARANCE_PATH,
        sequence_length: int = 8,
        min_idx: int = 0,
        max_idx: Optional[int] = None,
        total_samples: int = 100000,
        proprioceptive_keys: Optional[list[str]] = None,
        return_hidden_reset_mask: bool = False,
        return_next_obs: bool = True,
        expected_da_shape: Optional[tuple[int, int, int]] = None,
    ) -> None:
        self.clearance_path = str(clearance_path)
        super().__init__(
            file_path=file_path,
            sequence_length=sequence_length,
            min_idx=min_idx,
            max_idx=max_idx,
            total_samples=total_samples,
            proprioceptive_keys=proprioceptive_keys or ["distance", "heading", "angle_diff"],
            return_hidden_reset_mask=return_hidden_reset_mask,
            return_next_obs=return_next_obs,
            expected_da_shape=expected_da_shape,
        )
        with h5py.File(self.file_path, "r") as file:
            self._validate_clearance(file, Path(self.file_path))

    def __getitem__(self, idx: int):
        episode, start_idx = self.valid_sequences[np.random.randint(0, len(self.valid_sequences))]
        transition_start = int(self.transition_offsets[episode]) + start_idx
        obs_start = int(self.obs_offsets[episode]) + start_idx

        file = self._get_file()
        transition_slice = slice(transition_start, transition_start + self.sequence_length)
        obs_slice = slice(obs_start, obs_start + self.sequence_length)

        actions, rewards, dones = self._read_transition_arrays(file, transition_slice)
        clearance = torch.from_numpy(np.asarray(file[self.clearance_path][transition_slice])).float()
        clearance = _ensure_column_tensor(clearance)

        obs = self._read_observation_sequence(file, obs_slice)
        obs[RISK_CLEARANCE_STATE_KEY] = clearance

        if self.return_next_obs:
            next_obs_slice = slice(obs_start + 1, obs_start + self.sequence_length + 1)
            next_obs = self._read_observation_sequence(file, next_obs_slice)
        else:
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

    def _validate_clearance(self, file: h5py.File, path: Path) -> None:
        if self.clearance_path not in file:
            raise ValueError(f"Risk dataset {path} is missing clearance path: {self.clearance_path}")

        clearance = file[self.clearance_path]
        if len(clearance) != self.transition_count:
            raise ValueError(
                f"Clearance dataset has {len(clearance)} rows but the file has "
                f"{self.transition_count} transitions."
            )
        if clearance.ndim not in (1, 2) or (clearance.ndim == 2 and clearance.shape[1:] != (1,)):
            raise ValueError(
                f"Expected scalar transition clearance [N] or [N, 1], got {clearance.shape}."
            )
        clearance_values = np.asarray(clearance[:])
        if not np.isfinite(clearance_values).all():
            raise ValueError(f"Clearance dataset {self.clearance_path} contains non-finite values.")
        if np.any(clearance_values < 0):
            raise ValueError(f"Clearance dataset {self.clearance_path} contains negative values.")


class RLRoverLabRiskDinoDAMultiFileRandomSequenceDataset(
    RLRoverLabDinoDAMultiFileRandomSequenceDataset
):
    """Weighted union of risk-aware cached DINO/DA sequence datasets."""


def _ensure_column_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.reshape(1)
    if tensor.ndim == 1:
        return tensor.unsqueeze(-1)
    return tensor
