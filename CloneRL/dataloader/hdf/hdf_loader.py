from typing import Dict, List, Tuple, Union

import gymnasium as gym
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# Default mapper for imitation learning
HDF_DEFAULT_IL_MAPPER = {
    "observations": "observations",
    "actions": "actions",
}

# Default mapper for offline reinforcement learning
HDF_DEFAULT_ORL_MAPPER = {
    "observations": "observations",
    "actions": "actions",
    "rewards": "rewards",
    "next_observation": "next_observations",
    "dones": "dones",
}


class HDF5Dataset(Dataset):
    """HDF5 dataset for imitation learning and offline reinforcement learning."""
    def __init__(self, file_path: str, mapper: Dict[str, str], min_idx=0, max_idx=None):
        self.file_path = file_path
        self.mapper = mapper
        self.min_idx = min_idx
        self.max_idx = max_idx

        self.file = h5py.File(self.file_path, "r")
        if self.max_idx is None:
            self.max_idx = len(self.file[self.mapper["observations"]])



    def __len__(self):
        return self.max_idx - self.min_idx

    def __getitem__(self, idx):
        idx += self.min_idx
        data = {}

        for key, value in self.mapper.items():
            data[key] = torch.from_numpy(np.atleast_1d(self.file[value][idx]))

        return data

    def close(self):
        self.file.close()
