"""HDF5 Dataloader for GRU/RNN-based models.

This module provides dataset classes that return sequential data suitable for
training recurrent neural networks (GRU, LSTM, RNN). The dataloaders return
sequences of observations with shape (batch, seq_len, ...) to provide temporal
context for recurrent models.
"""

from typing import Dict, List, Tuple, Optional
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class HDF5SequenceDataset(Dataset):
    """HDF5 dataset that returns sequences of observations for GRU/RNN training.
    
    This dataset returns contiguous sequences of frames for training recurrent models.
    Each item is a sequence of (observations, actions, rewards, next_observations, dones, weights, masks).
    
    Args:
        file_path: Path to the HDF5 file.
        sequence_length: Length of sequences to return.
        min_idx: Minimum episode index to use.
        max_idx: Maximum episode index to use.
        proprioceptive_keys: List of keys for proprioceptive observations.
        stride: Stride between consecutive sequences (default: 1).
        overlap_episodes: Whether sequences can span episode boundaries (default: False).
    """

    def __init__(
        self,
        file_path: str,
        sequence_length: int = 16,
        min_idx: int = 0,
        max_idx: Optional[int] = None,
        proprioceptive_keys: List[str] = ['angle_diff', 'distance', 'heading'],
        stride: int = 1,
        overlap_episodes: bool = False,
    ):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.min_idx = min_idx
        self.max_idx = max_idx
        self.proprioceptive_keys = proprioceptive_keys
        self.stride = stride
        self.overlap_episodes = overlap_episodes

        with h5py.File(self.file_path, 'r') as file:
            if 'data' not in file:
                raise ValueError("HDF5 file must contain a 'data' group.")
            demo_keys = sorted(list(file['data'].keys()))

            if max_idx is None:
                max_idx = len(demo_keys)

            self.demo_keys = demo_keys[min_idx:max_idx]
            
            # Build a list of valid sequence start indices
            self.sequence_starts = []
            for demo_key in self.demo_keys:
                demo_group = file['data'][demo_key]
                num_samples = demo_group.attrs.get('num_samples', len(demo_group['actions']))
                
                # Calculate valid starting points for sequences
                # We need at least sequence_length frames, and we need next_obs for the last frame
                max_start = num_samples - sequence_length
                for start_idx in range(0, max_start, stride):
                    self.sequence_starts.append((demo_key, start_idx))

    def __len__(self) -> int:
        return len(self.sequence_starts)

    def _extract_obs_sequence(self, obs_group, start_idx: int, end_idx: int) -> Dict[str, torch.Tensor]:
        """Extract a sequence of observations.
        
        Args:
            obs_group: HDF5 group containing observation data.
            start_idx: Starting index of the sequence.
            end_idx: Ending index of the sequence (exclusive).
            
        Returns:
            Dictionary containing observation tensors with shape (seq_len, ...).
        """
        # Process images
        rgb = torch.from_numpy(obs_group['rgb_image'][start_idx:end_idx]).float()
        depth = torch.from_numpy(obs_group['depth_image'][start_idx:end_idx]).float()

        # from (N, H, W, C) to (N, C, H, W)
        rgb = rgb.permute(0, 3, 1, 2)
        depth = depth.permute(0, 3, 1, 2)

        # Process depth
        depth = torch.nan_to_num(depth, nan=6.0)
        depth = torch.where(torch.isinf(depth), torch.tensor(6.0), depth)
        depth = torch.clamp(depth, min=0.0, max=6.0)

        # Normalize RGB to [0, 1] for image encoder
        image = rgb / 255.0

        # Process proprioceptive data
        proprio_tensors = [
            torch.from_numpy(obs_group[key][start_idx:end_idx]).float()
            for key in self.proprioceptive_keys
        ]
        proprioceptive = torch.cat(proprio_tensors, dim=1)

        obs_dict = {
            "proprioceptive": proprioceptive,
            "image": image,
            "depth": depth,
            "rgb": rgb,
        }

        return obs_dict

    def __getitem__(self, idx: int) -> Tuple:
        """Get a sequence of data.
        
        Args:
            idx: Index of the sequence.
            
        Returns:
            Tuple of (obs, actions, rewards, next_obs, dones, weights, masks).
            Each tensor has shape (seq_len, ...).
        """
        demo_key, start_idx = self.sequence_starts[idx]
        end_idx = start_idx + self.sequence_length

        with h5py.File(self.file_path, 'r') as file:
            demo_group = file['data'][demo_key]

            # Load sequence data
            actions = torch.from_numpy(demo_group['actions'][start_idx:end_idx]).float()
            rewards = torch.from_numpy(demo_group['rewards'][start_idx:end_idx]).float()
            dones = torch.from_numpy(demo_group['dones'][start_idx:end_idx])

            # Ensure proper shape for 1D arrays
            if actions.ndim == 1:
                actions = actions.unsqueeze(-1)
            if rewards.ndim == 1:
                rewards = rewards.unsqueeze(-1)
            if dones.ndim == 1:
                dones = dones.unsqueeze(-1)

            # Extract observations
            obs = self._extract_obs_sequence(demo_group['obs'], start_idx, end_idx)
            next_obs = self._extract_obs_sequence(demo_group['next_obs'], start_idx, end_idx)

        # Weights and masks placeholders
        weights = torch.ones_like(rewards)
        masks = torch.ones_like(rewards)

        # Move to device
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        masks = masks.to(self.device)

        for key in obs:
            obs[key] = obs[key].to(self.device)
            next_obs[key] = next_obs[key].to(self.device)

        return obs, actions, rewards, next_obs, dones, weights, masks


class HDF5EpisodicGRUDataset(Dataset):
    """HDF5 dataset that returns full episodes for GRU/RNN training.
    
    This dataset returns complete episodes, padding shorter episodes to a maximum length.
    Suitable for training with variable-length sequences using masking.
    
    Args:
        file_path: Path to the HDF5 file.
        min_idx: Minimum episode index to use.
        max_idx: Maximum episode index to use.
        max_episode_length: Maximum episode length (for padding).
        proprioceptive_keys: List of keys for proprioceptive observations.
        pad_episodes: Whether to pad episodes to max_episode_length (default: True).
    """

    def __init__(
        self,
        file_path: str,
        min_idx: int = 0,
        max_idx: Optional[int] = None,
        max_episode_length: Optional[int] = None,
        proprioceptive_keys: List[str] = ['angle_diff', 'distance', 'heading'],
        pad_episodes: bool = True,
    ):
        self.file_path = file_path
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.min_idx = min_idx
        self.max_idx = max_idx
        self.proprioceptive_keys = proprioceptive_keys
        self.pad_episodes = pad_episodes

        with h5py.File(self.file_path, 'r') as file:
            if 'data' not in file:
                raise ValueError("HDF5 file must contain a 'data' group.")
            demo_keys = sorted(list(file['data'].keys()))

            if max_idx is None:
                max_idx = len(demo_keys)

            self.demo_keys = demo_keys[min_idx:max_idx]

            # Determine max episode length if not provided
            if max_episode_length is None:
                max_len = 0
                for demo_key in self.demo_keys:
                    demo_group = file['data'][demo_key]
                    num_samples = demo_group.attrs.get('num_samples', len(demo_group['actions']))
                    max_len = max(max_len, num_samples)
                self.max_episode_length = max_len
            else:
                self.max_episode_length = max_episode_length

    def __len__(self) -> int:
        return len(self.demo_keys)

    def _extract_obs_episode(self, obs_group) -> Dict[str, torch.Tensor]:
        """Extract all observations from an episode.
        
        Args:
            obs_group: HDF5 group containing observation data.
            
        Returns:
            Dictionary containing observation tensors with shape (episode_len, ...).
        """
        # Process images
        rgb = torch.from_numpy(obs_group['rgb_image'][:]).float()
        depth = torch.from_numpy(obs_group['depth_image'][:]).float()

        # from (N, H, W, C) to (N, C, H, W)
        rgb = rgb.permute(0, 3, 1, 2)
        depth = depth.permute(0, 3, 1, 2)

        # Process depth
        depth = torch.nan_to_num(depth, nan=6.0)
        depth = torch.where(torch.isinf(depth), torch.tensor(6.0), depth)
        depth = torch.clamp(depth, min=0.0, max=6.0)

        # Normalize RGB to [0, 1] for image encoder
        image = rgb / 255.0

        # Process proprioceptive data
        proprio_tensors = [
            torch.from_numpy(obs_group[key][:]).float()
            for key in self.proprioceptive_keys
        ]
        proprioceptive = torch.cat(proprio_tensors, dim=1)

        obs_dict = {
            "proprioceptive": proprioceptive,
            "image": image,
            "depth": depth,
            "rgb": rgb,
        }

        return obs_dict

    def _pad_tensor(self, tensor: torch.Tensor, target_length: int) -> torch.Tensor:
        """Pad a tensor to the target length along the first dimension.
        
        Args:
            tensor: Input tensor with shape (seq_len, ...).
            target_length: Target sequence length.
            
        Returns:
            Padded tensor with shape (target_length, ...).
        """
        current_length = tensor.shape[0]
        if current_length >= target_length:
            return tensor[:target_length]
        
        padding_size = target_length - current_length
        pad_shape = (padding_size,) + tensor.shape[1:]
        padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding], dim=0)

    def _pad_obs_dict(self, obs: Dict[str, torch.Tensor], target_length: int) -> Dict[str, torch.Tensor]:
        """Pad all tensors in an observation dictionary.
        
        Args:
            obs: Dictionary of observation tensors.
            target_length: Target sequence length.
            
        Returns:
            Dictionary of padded observation tensors.
        """
        return {key: self._pad_tensor(tensor, target_length) for key, tensor in obs.items()}

    def __getitem__(self, idx: int) -> Tuple:
        """Get a full episode of data.
        
        Args:
            idx: Index of the episode.
            
        Returns:
            Tuple of (obs, actions, rewards, next_obs, dones, weights, masks).
            Each tensor has shape (episode_len or max_episode_length, ...).
        """
        demo_key = self.demo_keys[idx]

        with h5py.File(self.file_path, 'r') as file:
            demo_group = file['data'][demo_key]
            num_samples = demo_group.attrs.get('num_samples', len(demo_group['actions']))

            # Load full episode data (excluding last timestep)
            end_idx = num_samples - 1 if num_samples > 1 else num_samples
            
            actions = torch.from_numpy(demo_group['actions'][:end_idx]).float()
            rewards = torch.from_numpy(demo_group['rewards'][:end_idx]).float()
            dones = torch.from_numpy(demo_group['dones'][:end_idx])

            # Ensure proper shape for 1D arrays
            if actions.ndim == 1:
                actions = actions.unsqueeze(-1)
            if rewards.ndim == 1:
                rewards = rewards.unsqueeze(-1)
            if dones.ndim == 1:
                dones = dones.unsqueeze(-1)

            # Extract observations
            obs = self._extract_obs_episode(demo_group['obs'])
            next_obs = self._extract_obs_episode(demo_group['next_obs'])

            # Slice to exclude last timestep
            obs = {key: tensor[:end_idx] for key, tensor in obs.items()}
            next_obs = {key: tensor[:end_idx] for key, tensor in next_obs.items()}

        # Create masks (1 for valid timesteps, 0 for padding)
        actual_length = actions.shape[0]
        masks = torch.ones(actual_length, 1)
        weights = torch.ones_like(rewards)

        # Pad if requested
        if self.pad_episodes:
            target_length = self.max_episode_length - 1  # -1 because we exclude last timestep
            actions = self._pad_tensor(actions, target_length)
            rewards = self._pad_tensor(rewards, target_length)
            dones = self._pad_tensor(dones, target_length)
            weights = self._pad_tensor(weights, target_length)
            masks = self._pad_tensor(masks, target_length)
            obs = self._pad_obs_dict(obs, target_length)
            next_obs = self._pad_obs_dict(next_obs, target_length)

        # Move to device
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        masks = masks.to(self.device)

        for key in obs:
            obs[key] = obs[key].to(self.device)
            next_obs[key] = next_obs[key].to(self.device)

        return obs, actions, rewards, next_obs, dones, weights, masks


class HDF5RandomSequenceGRUDataset(Dataset):
    """HDF5 dataset that returns random sequences for GRU/RNN training.
    
    This dataset randomly samples sequences from episodes, providing a virtual
    dataset size for training. This is useful for creating larger training sets
    through random sampling.
    
    Args:
        file_path: Path to the HDF5 file.
        sequence_length: Length of sequences to return.
        min_idx: Minimum episode index to use.
        max_idx: Maximum episode index to use.
        total_samples: Virtual dataset size for sampling.
        proprioceptive_keys: List of keys for proprioceptive observations.
        image_mode: Image mode for the 'image' key in observations.
            - "rgb": Use RGB normalized to [0, 1] (3 channels)
            - "grayscale": Use grayscale normalized to [0, 1] (1 channel)
            - "depth": Use depth only (1 channel)
            - "rgbd": Use RGB + depth (4 channels)
            - "grayscale_depth": Use grayscale + depth (2 channels)
            - None or "none": No image encoder, only depth is provided separately
    """

    def __init__(
        self,
        file_path: str,
        sequence_length: int = 16,
        min_idx: int = 0,
        max_idx: Optional[int] = None,
        total_samples: int = 100000,
        proprioceptive_keys: List[str] = ['angle_diff', 'distance', 'heading'],
        image_mode: Optional[str] = "rgb",  # "rgb", "grayscale", "depth", "rgbd", "grayscale_depth", or None/"none"
    ):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.min_idx = min_idx
        self.max_idx = max_idx
        self.total_samples = total_samples
        self.proprioceptive_keys = proprioceptive_keys
        self.image_mode = image_mode.lower() if isinstance(image_mode, str) else None
        
        # Validate image_mode
        valid_modes = ["rgb", "grayscale", "depth", "rgbd", "grayscale_depth", "none", None]
        if self.image_mode not in valid_modes and self.image_mode is not None:
            raise ValueError(f"image_mode must be one of {valid_modes}, got {image_mode}")

        with h5py.File(self.file_path, 'r') as file:
            if 'data' not in file:
                raise ValueError("HDF5 file must contain a 'data' group.")
            demo_keys = sorted(list(file['data'].keys()))

            if max_idx is None:
                max_idx = len(demo_keys)

            self.demo_keys = demo_keys[min_idx:max_idx]

            # Build a list of valid sequence start indices
            self.valid_sequences = []
            for demo_key in self.demo_keys:
                demo_group = file['data'][demo_key]
                num_samples = demo_group.attrs.get('num_samples', len(demo_group['actions']))
                
                # Calculate valid starting points for sequences
                max_start = num_samples - sequence_length
                if max_start > 0:
                    for start_idx in range(max_start):
                        self.valid_sequences.append((demo_key, start_idx))

        if len(self.valid_sequences) == 0:
            raise ValueError(
                f"No valid sequences found. Check that episodes have at least "
                f"{sequence_length} frames."
            )

    def __len__(self) -> int:
        return self.total_samples

    def _extract_obs_sequence(self, obs_group, start_idx: int, end_idx: int) -> Dict[str, torch.Tensor]:
        """Extract a sequence of observations.
        
        Args:
            obs_group: HDF5 group containing observation data.
            start_idx: Starting index of the sequence.
            end_idx: Ending index of the sequence (exclusive).
            
        Returns:
            Dictionary containing observation tensors with shape (seq_len, ...).
        """
        # Process depth (always needed)
        depth = torch.from_numpy(obs_group['depth_image'][start_idx:end_idx]).float()
        depth = depth.permute(0, 3, 1, 2)  # from (N, H, W, C) to (N, C, H, W)
        depth = torch.nan_to_num(depth, nan=6.0)
        depth = torch.where(torch.isinf(depth), torch.tensor(6.0), depth)
        depth = torch.clamp(depth, min=0.0, max=6.0)

        # Helper function to compute grayscale
        def compute_grayscale(rgb_tensor):
            """Convert RGB to grayscale using standard weights."""
            grayscale = rgb_tensor[:, 0] * 0.2989 + rgb_tensor[:, 1] * 0.5870 + rgb_tensor[:, 2] * 0.1140
            grayscale = grayscale / 255.0
            return grayscale.unsqueeze(1)  # (N, 1, H, W)

        # Process image based on mode
        rgb = None
        if self.image_mode == "rgb":
            # RGB only (3 channels)
            rgb = torch.from_numpy(obs_group['rgb_image'][start_idx:end_idx]).float()
            rgb = rgb.permute(0, 3, 1, 2)  # from (N, H, W, C) to (N, C, H, W)
            image = rgb / 255.0
        elif self.image_mode == "grayscale":
            # Grayscale only (1 channel)
            rgb = torch.from_numpy(obs_group['rgb_image'][start_idx:end_idx]).float()
            rgb = rgb.permute(0, 3, 1, 2)
            image = compute_grayscale(rgb)
        elif self.image_mode == "depth":
            # Depth only (1 channel)
            image = depth.clone()
        elif self.image_mode == "rgbd":
            # RGB + Depth (4 channels)
            rgb = torch.from_numpy(obs_group['rgb_image'][start_idx:end_idx]).float()
            rgb = rgb.permute(0, 3, 1, 2)
            rgb_normalized = rgb / 255.0
            image = torch.cat([rgb_normalized, depth], dim=1)  # (N, 4, H, W)
        elif self.image_mode == "grayscale_depth":
            # Grayscale + Depth (2 channels)
            rgb = torch.from_numpy(obs_group['rgb_image'][start_idx:end_idx]).float()
            rgb = rgb.permute(0, 3, 1, 2)
            grayscale = compute_grayscale(rgb)
            image = torch.cat([grayscale, depth], dim=1)  # (N, 2, H, W)
        else:  # None or "none" - no image encoder
            image = None

        # # --- OLD CODE (depth + grayscale) ---
        # # Process images
        # rgb = torch.from_numpy(obs_group['rgb_image'][start_idx:end_idx]).float()
        # depth = torch.from_numpy(obs_group['depth_image'][start_idx:end_idx]).float()
        # # from (N, H, W, C) to (N, C, H, W)
        # rgb = rgb.permute(0, 3, 1, 2)
        # depth = depth.permute(0, 3, 1, 2)
        # # Process depth
        # depth = torch.nan_to_num(depth, nan=6.0)
        # depth = torch.where(torch.isinf(depth), torch.tensor(6.0), depth)
        # depth = torch.clamp(depth, min=0.0, max=6.0)
        # # Convert RGB to grayscale
        # grayscale = rgb[:, 0] * 0.2989 + rgb[:, 1] * 0.5870 + rgb[:, 2] * 0.1140
        # grayscale = grayscale / 255
        # grayscale = grayscale.unsqueeze(1)
        # # Combine depth and grayscale
        # image = torch.cat([depth, grayscale], dim=1)
        # # --- END OLD CODE ---

        # Process proprioceptive data
        proprio_tensors = [
            torch.from_numpy(obs_group[key][start_idx:end_idx]).float()
            for key in self.proprioceptive_keys
        ]
        proprioceptive = torch.cat(proprio_tensors, dim=1)

        obs_dict = {
            "proprioceptive": proprioceptive,
            "depth": depth,
        }
        
        # Only add image if it's being used
        if image is not None:
            obs_dict["image"] = image
        if rgb is not None:
            obs_dict["rgb"] = rgb

        return obs_dict

    def __getitem__(self, idx: int) -> Tuple:
        """Get a random sequence of data.
        
        Args:
            idx: Index (used for virtual dataset size, actual sampling is random).
            
        Returns:
            Tuple of (obs, actions, rewards, next_obs, dones, weights, masks).
            Each tensor has shape (seq_len, ...).
        """
        # Randomly select a valid sequence
        random_idx = np.random.randint(0, len(self.valid_sequences))
        demo_key, start_idx = self.valid_sequences[random_idx]
        end_idx = start_idx + self.sequence_length

        with h5py.File(self.file_path, 'r') as file:
            demo_group = file['data'][demo_key]

            # Load sequence data
            actions = torch.from_numpy(demo_group['actions'][start_idx:end_idx]).float()
            rewards = torch.from_numpy(demo_group['rewards'][start_idx:end_idx]).float()
            dones = torch.from_numpy(demo_group['dones'][start_idx:end_idx])

            # Ensure proper shape for 1D arrays
            if actions.ndim == 1:
                actions = actions.unsqueeze(-1)
            if rewards.ndim == 1:
                rewards = rewards.unsqueeze(-1)
            if dones.ndim == 1:
                dones = dones.unsqueeze(-1)

            # Extract observations
            obs = self._extract_obs_sequence(demo_group['obs'], start_idx, end_idx)
            next_obs = self._extract_obs_sequence(demo_group['next_obs'], start_idx, end_idx)

        # Weights and masks placeholders
        weights = torch.ones_like(rewards)
        masks = torch.ones_like(rewards)

        # Move to device
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        masks = masks.to(self.device)

        for key in obs:
            obs[key] = obs[key].to(self.device)
            next_obs[key] = next_obs[key].to(self.device)

        return obs, actions, rewards, next_obs, dones, weights, masks


class HDF5SlidingWindowGRUDataset(Dataset):
    """HDF5 dataset with sliding window approach for GRU/RNN training.
    
    This dataset uses a sliding window to create overlapping sequences from episodes,
    which can improve training by providing more context variations.
    
    Args:
        file_path: Path to the HDF5 file.
        sequence_length: Length of sequences to return.
        min_idx: Minimum episode index to use.
        max_idx: Maximum episode index to use.
        proprioceptive_keys: List of keys for proprioceptive observations.
        window_stride: Stride of the sliding window (default: 1).
        return_hidden_reset_mask: Whether to return mask indicating episode boundaries.
    """

    def __init__(
        self,
        file_path: str,
        sequence_length: int = 16,
        min_idx: int = 0,
        max_idx: Optional[int] = None,
        proprioceptive_keys: List[str] = ['angle_diff', 'distance', 'heading'],
        window_stride: int = 1,
        return_hidden_reset_mask: bool = True,
    ):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.min_idx = min_idx
        self.max_idx = max_idx
        self.proprioceptive_keys = proprioceptive_keys
        self.window_stride = window_stride
        self.return_hidden_reset_mask = return_hidden_reset_mask

        with h5py.File(self.file_path, 'r') as file:
            if 'data' not in file:
                raise ValueError("HDF5 file must contain a 'data' group.")
            demo_keys = sorted(list(file['data'].keys()))

            if max_idx is None:
                max_idx = len(demo_keys)

            self.demo_keys = demo_keys[min_idx:max_idx]

            # Build index of all valid windows
            self.windows = []
            for demo_key in self.demo_keys:
                demo_group = file['data'][demo_key]
                num_samples = demo_group.attrs.get('num_samples', len(demo_group['actions']))
                
                # Create sliding windows with specified stride
                for start_idx in range(0, num_samples - sequence_length, window_stride):
                    self.windows.append((demo_key, start_idx))

    def __len__(self) -> int:
        return len(self.windows)

    def _extract_obs_sequence(self, obs_group, start_idx: int, end_idx: int) -> Dict[str, torch.Tensor]:
        """Extract a sequence of observations.
        
        Args:
            obs_group: HDF5 group containing observation data.
            start_idx: Starting index of the sequence.
            end_idx: Ending index of the sequence (exclusive).
            
        Returns:
            Dictionary containing observation tensors with shape (seq_len, ...).
        """
        # Process images
        rgb = torch.from_numpy(obs_group['rgb_image'][start_idx:end_idx]).float()
        depth = torch.from_numpy(obs_group['depth_image'][start_idx:end_idx]).float()

        # from (N, H, W, C) to (N, C, H, W)
        rgb = rgb.permute(0, 3, 1, 2)
        depth = depth.permute(0, 3, 1, 2)

        # Process depth
        depth = torch.nan_to_num(depth, nan=6.0)
        depth = torch.where(torch.isinf(depth), torch.tensor(6.0), depth)
        depth = torch.clamp(depth, min=0.0, max=6.0)

        # Normalize RGB to [0, 1] for image encoder
        image = rgb / 255.0

        # Process proprioceptive data
        proprio_tensors = [
            torch.from_numpy(obs_group[key][start_idx:end_idx]).float()
            for key in self.proprioceptive_keys
        ]
        proprioceptive = torch.cat(proprio_tensors, dim=1)

        obs_dict = {
            "proprioceptive": proprioceptive,
            "image": image,
            "depth": depth,
            "rgb": rgb,
        }

        return obs_dict

    def __getitem__(self, idx: int) -> Tuple:
        """Get a sequence from a sliding window.
        
        Args:
            idx: Index of the window.
            
        Returns:
            Tuple of (obs, actions, rewards, next_obs, dones, weights, masks).
            Each tensor has shape (seq_len, ...).
            If return_hidden_reset_mask is True, also returns hidden_reset_mask.
        """
        demo_key, start_idx = self.windows[idx]
        end_idx = start_idx + self.sequence_length

        with h5py.File(self.file_path, 'r') as file:
            demo_group = file['data'][demo_key]

            # Load sequence data
            actions = torch.from_numpy(demo_group['actions'][start_idx:end_idx]).float()
            rewards = torch.from_numpy(demo_group['rewards'][start_idx:end_idx]).float()
            dones = torch.from_numpy(demo_group['dones'][start_idx:end_idx])

            # Ensure proper shape for 1D arrays
            if actions.ndim == 1:
                actions = actions.unsqueeze(-1)
            if rewards.ndim == 1:
                rewards = rewards.unsqueeze(-1)
            if dones.ndim == 1:
                dones = dones.unsqueeze(-1)

            # Extract observations
            obs = self._extract_obs_sequence(demo_group['obs'], start_idx, end_idx)
            next_obs = self._extract_obs_sequence(demo_group['next_obs'], start_idx, end_idx)

        # Weights and masks placeholders
        weights = torch.ones_like(rewards)
        masks = torch.ones_like(rewards)

        # Create hidden reset mask (indicates when hidden state should be reset)
        # Reset at the beginning of each sequence and after each done
        hidden_reset_mask = torch.zeros(self.sequence_length, 1)
        hidden_reset_mask[0] = 1.0 if start_idx == 0 else 0.0  # Reset at episode start
        
        # Also mark resets after done signals (except the last timestep)
        for i in range(self.sequence_length - 1):
            if dones[i].item():
                hidden_reset_mask[i + 1] = 1.0

        # Move to device
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        masks = masks.to(self.device)
        hidden_reset_mask = hidden_reset_mask.to(self.device)

        for key in obs:
            obs[key] = obs[key].to(self.device)
            next_obs[key] = next_obs[key].to(self.device)

        if self.return_hidden_reset_mask:
            return obs, actions, rewards, next_obs, dones, weights, masks, hidden_reset_mask
        return obs, actions, rewards, next_obs, dones, weights, masks
