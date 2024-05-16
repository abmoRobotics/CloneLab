from typing import Dict, List, Tuple, Union

import gymnasium as gym
import h5py
import numpy as np
import torch
import xarray as xr
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
    "next_observation": "observations",
    "dones": "terminated",
    "rgb": "rgb",
    "depth": "depth",
}


class HDF5Dataset(Dataset):
    """HDF5 dataset for imitation learning and offline reinforcement learning."""

    def __init__(self, file_path: str, mapper: Dict[str, str], min_idx=0, max_idx=None):
        self.file_path = file_path
        self.mapper = mapper
        self.min_idx = min_idx
        self.max_idx = max_idx
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.weights = self._calculate_weights()
        if self.max_idx is None:
            self.max_idx = len(self.file[self.mapper["observations"]])

    def __len__(self):
        return self.max_idx - self.min_idx

    def __getitem__(self, idx):
        raise NotImplementedError

    def close(self):
        self.file.close()

    def _calculate_weights(self):
        with h5py.File(self.file_path, 'r') as file:
            actions = file[self.mapper["actions"]][self.min_idx:self.max_idx]

        # Define the histogram bins for the actions
        bins = np.linspace(actions.min(), actions.max(), 40)

        # Calculate histograms for each column in actions
        histograms = np.array([np.histogram(actions[:, i], bins=bins)[0] for i in range(actions.shape[1])]).T

        # Calculate the inverse of the histograms to form the basis of weights, with a small constant for numerical stability
        weights = 1.0 / (histograms + 1e-6)
        normalized_weights = weights / weights.sum(axis=1, keepdims=True)

        # Interpolate weights for each action according to bins
        interpolated_weights = np.array([np.interp(actions[:, i], bins[:-1], normalized_weights[:, i])
                                        for i in range(actions.shape[1])]).T

        # Average the interpolated weights to form the final weight vector
        # weight_sum = 1 / actions.shape[1] * np.sum(interpolated_weights, axis=0)
        # multiply the weights in axis 0
        # weight_sum = np.prod(interpolated_weights, axis=0)
        # Convert the weights to a tensor
        return torch.from_numpy(interpolated_weights).float().to(self.device) * 100

    def get_weights(self, idx):
        return self.weights[idx - self.min_idx]  # .unsqueeze(-1)

    def get_obs_depth(self, idx, file):
        image = file[self.mapper["depth"]][idx]
        obs = {"proprioceptive": torch.from_numpy(np.atleast_1d(file[self.mapper["observations"]][idx, :4])),
               "image": torch.from_numpy(np.atleast_1d(image)).unsqueeze(0)}
        # Get to correct device
        for key, value in obs.items():
            obs[key] = value.to(self.device)
            obs[key] = torch.nan_to_num(obs[key], nan=256.0)
            obs[key] = torch.clamp(obs[key], min=0.0, max=256.0)
        return obs

    def get_obs_rgb(self, idx):
        rgb = torch.from_numpy(np.atleast_1d(self.file[self.mapper["rgb"]][idx]))
        obs = {"proprioceptive": torch.from_numpy(np.atleast_1d(self.file[self.mapper["observations"]][idx, :4])),
               "image": rgb}

        for key, value in obs.items():
            obs[key] = value.to(self.device)
        return obs

    def get_obs_depth_rgb(self, idx, file):
        depth = np.expand_dims(file[self.mapper["depth"]][idx], 2)
        rgb = file[self.mapper["rgb"]][idx]

        image = np.concatenate((depth, rgb), axis=2)
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = torch.nan_to_num(image, nan=256.0)
        image[0] = torch.clamp(image[0], min=0.0, max=256.0)
        obs = {"proprioceptive": torch.from_numpy(np.atleast_1d(file[self.mapper["observations"]][idx, :4])),
               "image": image}

        for key, value in obs.items():
            obs[key] = value.to(self.device)

        return obs


class EpisodicHDF5Dataset(HDF5Dataset):
    def __init__(self, file_path: str, mapper: Dict[str, str], min_idx=0, max_idx=None, observations: List[str] = ["base"]):
        super().__init__(file_path, mapper, min_idx, max_idx, observations)
        self.episodes = self._calculate_episodes()

    def _calculate_episodes(self) -> List[np.ndarray]:
        with h5py.File(self.file_path, 'r') as file:
            dones = file[self.mapper["dones"]][self.min_idx:self.max_idx]

        episode_indices = np.where(dones)[0]
        all_episodes = np.split(np.arange(self.min_idx, self.max_idx), episode_indices + 1)

        # Filter out episodes that are too short
        episodes = [episode for episode in all_episodes if len(episode) > 5]
        return episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        episode = self.episodes[idx]

        data = {key: [] for key in ["obs", "actions", "rewards", "next_obs", "dones", "weights"]}

        with h5py.File(self.file_path, 'r') as file:
            for i in episode:
                data["obs"].append(self.get_obs_depth_rgb(i, file))
                data["next_obs"].append(self.get_obs_depth_rgb(i + 1, file))
                data["weights"].append(self.get_weights(i))
                for key in ["actions", "rewards", "dones"]:
                    numpy_data = np.atleast_1d(file[self.mapper[key]][i])
                    tensor_data = torch.from_numpy(numpy_data).to(self.device)
                    data[key].append(tensor_data)

        for key in data:
            if type(data[key][0]) == dict:
                data[key] = {sub_key: torch.stack([d[sub_key] for d in data[key]]).to(self.device)
                             for sub_key in data[key][0]}
            else:
                data[key] = torch.stack(data[key]).to(self.device)

        return data["obs"], data["actions"], data["rewards"], data["next_obs"], data["dones"], data["weights"]


class CloneLabDataset(HDF5Dataset):
    """Dataset class for the CloneLab dataset. The dataset can be either episodic or non-episodic.

    Args:
        file_path (str): The path to the HDF5 file.
        mapper (Dict[str, str]): A dictionary containing the dataset keys.
        min_idx (int, optional): The minimum index of the dataset. Defaults to 0.
        max_idx ([type], optional): The maximum index of the dataset. Defaults to None.
        episodic (bool, optional): Whether the dataset is episodic. Defaults to False.

    Returns:
        CloneLabDataset: A dataset object for the CloneLab dataset.
    """

    def __init__(self, file_path: str, mapper: Dict[str, str], min_idx=0, max_idx=None, episodic: bool = False):
        super().__init__(file_path, mapper, min_idx, max_idx)
        """Initialize the CloneLab dataset."""
        self.episodic = episodic
        if self.episodic:
            self.episodes = self._calculate_episodes()

    def _calculate_episodes(self) -> List[np.ndarray]:
        """Extract episodes from the dataset, based on the 'dones' key.

        Returns:
            List[np.ndarray]: A list with arrays containing the indices of the episodes.
        """
        with h5py.File(self.file_path, 'r') as file:
            dones = file[self.mapper["dones"]][self.min_idx:self.max_idx]
        episode_indices = np.where(dones)[0]
        all_episodes = np.split(np.arange(self.min_idx, self.max_idx), episode_indices + 1)
        episodes = [episode for episode in all_episodes if len(episode) > 5]
        return episodes

    def __len__(self) -> int:
        """Return the number of episodes if the dataset is episodic, otherwise the number of samples.

        Returns:
            int: The number of samples or episodes in the dataset.
        """
        if self.episodic:
            return len(self.episodes)
        return self.max_idx - self.min_idx if self.max_idx else float('inf')

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Return a single sample from the dataset. If the dataset is episodic, return an entire episode.

        Args:
            idx (int): The index of the sample or episode.

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor,
            torch.Tensor]: A tuple containing the observations, actions, rewards, next observations, dones,
            and weights."""
        if self.episodic:
            return self._get_episode_item(idx)
        else:
            return self._get_single_item(idx)

    def _get_single_item(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Return a single sample from the dataset.

        Args:
            idx (int): The index of the sample.

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor,
            torch.Tensor]: A tuple containing the observations, actions, rewards, next observations, dones, and weights.
        """
        with h5py.File(self.file_path, 'r') as file:
            idx += self.min_idx
            obs = self.get_obs_depth_rgb(idx, file)
            action = torch.from_numpy(np.atleast_1d(file[self.mapper["actions"]][idx])).to(self.device)
            reward = torch.from_numpy(np.atleast_1d(file[self.mapper["rewards"]][idx])).to(self.device)
            next_obs = self.get_obs_depth_rgb(idx + 1, file)
        weights = self.get_weights(idx)
        done = torch.zeros_like(reward)
        masks = torch.ones_like(reward)
        return obs, action, reward, next_obs, done, weights, masks

    def _get_episode_item(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Return an entire episode from the dataset based on the episode index.

        Args:
            idx (int): The index of the episode.

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor,
            torch.Tensor]: A tuple containing the observations, actions, rewards, next observations, dones, and weights.
        """
        episode = self.episodes[idx]
        data = {key: [] for key in ["obs", "actions", "rewards", "next_obs", "dones", "weights"]}
        with h5py.File(self.file_path, 'r') as file:
            for i in episode:
                data["obs"].append(self.get_obs_depth_rgb(i, file))
                data["next_obs"].append(self.get_obs_depth_rgb(i + 1, file))
                data["weights"].append(self.get_weights(i))
                for key in ["actions", "rewards", "dones"]:
                    numpy_data = np.atleast_1d(file[self.mapper[key]][i])
                    tensor_data = torch.from_numpy(numpy_data).to(self.device)
                    data[key].append(tensor_data)
        for key in data:
            if isinstance(data[key][0], dict):
                data[key] = {sub_key: torch.stack([d[sub_key] for d in data[key]]).to(self.device)
                             for sub_key in data[key][0]}
            else:
                data[key] = torch.stack(data[key]).to(self.device)
        return data["obs"], data["actions"], data["rewards"], data["next_obs"], data["dones"], data["weights"]
