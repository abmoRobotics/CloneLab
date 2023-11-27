from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
import gymnasium as gym

import torch
from torch.utils.data import Dataset

# Default mapper for imitation learning
HDF_DEFAULT_IL_MAPPER = {
    "observation": "observation",
    "action": "action",
}

# Default mapper for offline reinforcement learning
HDF_DEFAULT_ORL_MAPPER = {
    "observation": "observation",
    "action": "action",
    "reward": "reward",
    "next_observation": "next_observation",
    "done": "done",
}


class HDF5Dataset(Dataset):
    """HDF5 dataset for imitation learning and offline reinforcement learning."""
    def __init__(self, file_path: str, mapper: Dict[str, str]):
        self.file_path = file_path
        self.mapper = mapper

        self.file = h5py.File(self.file_path, "r")
        
    
    def __len__(self):
        return len(self.file[self.mapper["observation"]])
    
    def __getitem__(self, idx):

        data = {}

        for key, value in self.mapper.items():
            data[key] = torch.from_numpy(np.atleast_1d(self.file[value][idx]))

        return data

    def close(self):
        self.file.close()