from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


SCHEMA_NAME = "rlroverlab.offline_rgbd_v2"
VALID_IMAGE_MODES = {"rgb", "grayscale", "depth", "rgbd", "grayscale_depth", "none", None}
VALID_DECODE_BACKENDS = {"cuda", "pillow"}


def is_rlroverlab_compressed_rgbd(file_path: str | Path) -> bool:
    """Return True when an HDF5 file uses the RLRoverLab compressed RGB-D schema."""
    try:
        with h5py.File(file_path, "r") as file:
            return file.attrs.get("schema_name") == SCHEMA_NAME
    except OSError:
        return False


def _normalize_image_mode(image_mode: Optional[str]) -> Optional[str]:
    mode = image_mode.lower() if isinstance(image_mode, str) else image_mode
    if mode not in VALID_IMAGE_MODES:
        raise ValueError(f"image_mode must be one of {sorted(v for v in VALID_IMAGE_MODES if v is not None)}, got {image_mode}")
    return mode


def _normalize_decode_backend(decode_backend: str) -> str:
    backend = decode_backend.lower()
    if backend not in VALID_DECODE_BACKENDS:
        raise ValueError(f"decode_backend must be one of {sorted(VALID_DECODE_BACKENDS)}, got {decode_backend}")
    return backend


def _model_size_to_hw(image_size: Optional[Sequence[int]]) -> Optional[Tuple[int, int]]:
    if image_size is None:
        return None
    if len(image_size) != 2:
        raise ValueError(f"Expected image_size as [width, height], got {image_size}")
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size values must be positive, got {image_size}")
    return height, width


def _resize_chw(tensor: torch.Tensor, size_hw: Optional[Tuple[int, int]]) -> torch.Tensor:
    if size_hw is None or tuple(tensor.shape[-2:]) == size_hw:
        return tensor
    kwargs = {
        "size": size_hw,
        "mode": "bilinear",
        "align_corners": False,
    }
    try:
        return F.interpolate(tensor.unsqueeze(0), antialias=True, **kwargs).squeeze(0)
    except TypeError:
        return F.interpolate(tensor.unsqueeze(0), **kwargs).squeeze(0)


def _ensure_column_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.reshape(1)
    if tensor.ndim == 1:
        return tensor.unsqueeze(-1)
    return tensor


def _encoded_to_uint8_numpy(encoded: np.ndarray) -> np.ndarray:
    return np.asarray(encoded, dtype=np.uint8)


class _CompressedRGBDBase(Dataset):
    def __init__(
        self,
        file_path: str,
        min_idx: int = 0,
        max_idx: Optional[int] = None,
        total_samples: int = 120000,
        proprioceptive_keys: List[str] = ["angle_diff", "distance", "heading"],
        image_size: Optional[Sequence[int]] = None,
        image_mode: Optional[str] = "rgb",
        device: str | torch.device = "cuda:0" if torch.cuda.is_available() else "cpu",
        decode_backend: str = "cuda",
        include_raw_rgb: bool = False,
    ):
        self.file_path = str(file_path)
        self.min_idx = int(min_idx)
        self.max_idx = max_idx
        self.total_samples = int(total_samples)
        self.proprioceptive_keys = list(proprioceptive_keys)
        self.image_size_hw = _model_size_to_hw(image_size)
        self.image_mode = _normalize_image_mode(image_mode)
        self.device = torch.device(device)
        self.decode_backend = _normalize_decode_backend(decode_backend)
        self.include_raw_rgb = bool(include_raw_rgb)
        self._torchvision_decode_jpeg = None
        self._torchvision_image_read_mode = None
        self._nvimgcodec = None
        self._nvimgcodec_decoder = None

        if self.decode_backend == "cuda":
            if self.device.type != "cuda":
                raise ValueError("decode_backend='cuda' requires a CUDA training device.")
            if not torch.cuda.is_available():
                raise ValueError("decode_backend='cuda' requires torch.cuda.is_available().")

        with h5py.File(self.file_path, "r") as file:
            schema = file.attrs.get("schema_name")
            if schema != SCHEMA_NAME:
                raise ValueError(
                    f"Expected compressed RLRoverLab schema {SCHEMA_NAME!r}, got {schema!r} in {self.file_path}"
                )
            if bool(file.attrs.get("zero_structural_duplication", False)) is not True:
                raise ValueError("Compressed RLRoverLab datasets must declare zero_structural_duplication=True.")
            for path in (
                "observations/rgb_jpeg",
                "observations/depth_jp2",
                "observations/state",
                "transitions/actions",
                "transitions/rewards",
                "index/obs_index",
                "index/next_obs_index",
                "index/episode_id",
                "index/episode_transition_index",
                "index/episode_lengths",
                "index/obs_offsets",
                "index/transition_offsets",
            ):
                if path not in file:
                    raise ValueError(f"Compressed RLRoverLab dataset is missing required path: {path}")
            for key in self.proprioceptive_keys:
                if f"observations/state/{key}" not in file:
                    raise ValueError(f"Compressed RLRoverLab dataset is missing proprioceptive key: {key}")

            self.transition_count = int(file.attrs.get("total_transitions", len(file["transitions/actions"])))
            self.observation_count = int(file.attrs.get("total_observations", len(file["observations/rgb_jpeg"])))
            self.rgb_scale = float(file.attrs.get("recommended_rgb_scale", 1.0 / 255.0))
            self.depth_scale_m = float(file.attrs.get("recommended_depth_scale_m", 0.001))

            self.obs_index = file["index/obs_index"][:].astype(np.int64)
            self.next_obs_index = file["index/next_obs_index"][:].astype(np.int64)
            self.episode_id = file["index/episode_id"][:].astype(np.int64)
            self.episode_transition_index = file["index/episode_transition_index"][:].astype(np.int64)
            self.episode_lengths = file["index/episode_lengths"][:].astype(np.int64)
            self.obs_offsets = file["index/obs_offsets"][:].astype(np.int64)
            self.transition_offsets = file["index/transition_offsets"][:].astype(np.int64)

        self.episode_count = int(len(self.episode_lengths))
        episode_stop = self.episode_count if max_idx is None else min(int(max_idx), self.episode_count)
        episode_start = max(0, min(self.min_idx, episode_stop))
        self.selected_episode_ids = np.arange(episode_start, episode_stop, dtype=np.int64)
        if self.selected_episode_ids.size == 0:
            raise ValueError(f"No episodes selected from {self.file_path}: min_idx={min_idx}, max_idx={max_idx}")

    def _transition_indices_for_selected_episodes(self, min_episode_timestep: int = 0) -> np.ndarray:
        indices: List[np.ndarray] = []
        for episode in self.selected_episode_ids:
            start = int(self.transition_offsets[episode])
            length = int(self.episode_lengths[episode])
            local_start = min_episode_timestep
            if local_start >= length:
                continue
            indices.append(np.arange(start + local_start, start + length, dtype=np.int64))
        if not indices:
            raise ValueError("No valid transitions found for the selected compressed RLRoverLab episodes.")
        return np.concatenate(indices)

    def _sequence_starts(self, sequence_length: int) -> List[Tuple[int, int]]:
        starts: List[Tuple[int, int]] = []
        for episode in self.selected_episode_ids:
            length = int(self.episode_lengths[episode])
            max_start = length - int(sequence_length)
            if max_start < 0:
                continue
            starts.extend((int(episode), start) for start in range(max_start + 1))
        if not starts:
            raise ValueError(
                f"No valid compressed RLRoverLab sequences found. Selected episodes must contain at least "
                f"{sequence_length} transitions."
            )
        return starts

    def _read_transition_arrays(self, file: h5py.File, transition_indices: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions = torch.from_numpy(np.asarray(file["transitions/actions"][transition_indices])).float()
        rewards = torch.from_numpy(np.asarray(file["transitions/rewards"][transition_indices])).float()
        if "transitions/dones" in file:
            dones_np = np.asarray(file["transitions/dones"][transition_indices])
        else:
            terminals = np.asarray(file["transitions/terminals"][transition_indices], dtype=np.bool_)
            timeouts = np.asarray(file["transitions/timeouts"][transition_indices], dtype=np.bool_)
            dones_np = np.logical_or(terminals, timeouts)
        dones = torch.from_numpy(np.asarray(dones_np))

        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        rewards = _ensure_column_tensor(rewards)
        dones = _ensure_column_tensor(dones)
        return actions.to(self.device), rewards.to(self.device), dones.to(self.device)

    def _read_observation(self, file: h5py.File, obs_index: int) -> Dict[str, torch.Tensor]:
        rgb_raw = self._decode_rgb(file["observations/rgb_jpeg"][obs_index])
        depth = self._decode_depth(file["observations/depth_jp2"][obs_index])
        rgb_normalized = rgb_raw * self.rgb_scale

        rgb_raw = _resize_chw(rgb_raw, self.image_size_hw)
        rgb_normalized = _resize_chw(rgb_normalized, self.image_size_hw)
        depth = _resize_chw(depth, self.image_size_hw)

        proprio_tensors = [
            torch.from_numpy(np.atleast_1d(file[f"observations/state/{key}"][obs_index])).float()
            for key in self.proprioceptive_keys
        ]
        proprioceptive = torch.cat(proprio_tensors, dim=0)

        obs: Dict[str, torch.Tensor] = {
            "proprioceptive": proprioceptive,
            "depth": depth,
        }
        image = self._make_image_tensor(rgb_normalized, depth)
        if image is not None:
            obs["image"] = image
        if self.include_raw_rgb:
            obs["rgb"] = rgb_raw
        return {key: value.to(self.device) for key, value in obs.items()}

    def _read_observation_sequence(self, file: h5py.File, obs_indices: np.ndarray) -> Dict[str, torch.Tensor]:
        observations = [self._read_observation(file, int(obs_index)) for obs_index in obs_indices]
        return {
            key: torch.stack([obs[key] for obs in observations], dim=0).to(self.device)
            for key in observations[0].keys()
        }

    def _decode_rgb(self, encoded: np.ndarray) -> torch.Tensor:
        if self.decode_backend == "cuda":
            return self._decode_rgb_cuda(encoded)
        return self._decode_rgb_pillow(encoded)

    def _decode_depth(self, encoded: np.ndarray) -> torch.Tensor:
        if self.decode_backend == "cuda":
            return self._decode_depth_cuda(encoded)
        return self._decode_depth_pillow(encoded)

    def _decode_rgb_cuda(self, encoded: np.ndarray) -> torch.Tensor:
        decode_jpeg, image_read_mode = self._require_torchvision_jpeg_decoder()
        encoded_cpu = torch.from_numpy(_encoded_to_uint8_numpy(encoded).copy())
        decoded = decode_jpeg(encoded_cpu, mode=image_read_mode.RGB, device=str(self.device))
        return decoded.float()

    def _decode_depth_cuda(self, encoded: np.ndarray) -> torch.Tensor:
        nvimgcodec = self._require_nvimgcodec()
        stream = nvimgcodec.CodeStream(_encoded_to_uint8_numpy(encoded))
        cuda_stream = torch.cuda.current_stream(self.device).cuda_stream
        params = nvimgcodec.DecodeParams()
        params.allow_any_depth = True
        params.color_spec = nvimgcodec.ColorSpec.GRAY
        decoded = self._nvimgcodec_decoder.decode(stream, params=params, cuda_stream=cuda_stream)
        if decoded is None:
            raise RuntimeError("nvImageCodec/nvJPEG2000 failed to decode a compressed RLRoverLab depth JPEG2000 frame.")

        if hasattr(decoded, "to_dlpack"):
            depth = torch.from_dlpack(decoded.to_dlpack())
        else:
            depth = torch.as_tensor(decoded, device=self.device)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise RuntimeError(f"Expected decoded depth to have shape (H, W) or (H, W, C), got {tuple(depth.shape)}")
        depth = depth.unsqueeze(0).float()
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.clamp(depth * self.depth_scale_m, min=0.0, max=6.0)

    def _decode_rgb_pillow(self, encoded: np.ndarray) -> torch.Tensor:
        try:
            image = Image.open(io.BytesIO(_encoded_to_uint8_numpy(encoded).tobytes())).convert("RGB")
        except UnidentifiedImageError as exc:
            raise RuntimeError("Pillow failed to decode a compressed RLRoverLab RGB JPEG frame.") from exc
        array = np.asarray(image, dtype=np.uint8).copy()
        return torch.from_numpy(array).permute(2, 0, 1).float()

    def _decode_depth_pillow(self, encoded: np.ndarray) -> torch.Tensor:
        try:
            image = Image.open(io.BytesIO(_encoded_to_uint8_numpy(encoded).tobytes()))
            array = np.asarray(image)
        except UnidentifiedImageError as exc:
            raise RuntimeError(
                "Pillow failed to decode a compressed RLRoverLab depth JPEG2000 frame. "
                "Install Pillow with JPEG2000/OpenJPEG support."
            ) from exc

        if array.ndim == 3:
            array = array[..., 0]
        depth = torch.from_numpy(np.asarray(array, dtype=np.float32).copy()).unsqueeze(0)
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = torch.clamp(depth * self.depth_scale_m, min=0.0, max=6.0)
        return depth

    def _require_torchvision_jpeg_decoder(self):
        if self._torchvision_decode_jpeg is None or self._torchvision_image_read_mode is None:
            try:
                from torchvision.io import ImageReadMode, decode_jpeg
            except ImportError as exc:
                raise RuntimeError(
                    "CUDA RGB JPEG decode requires torchvision with CUDA image I/O support. "
                    "Install torchvision matching the local torch/CUDA build."
                ) from exc
            self._torchvision_decode_jpeg = decode_jpeg
            self._torchvision_image_read_mode = ImageReadMode
        return self._torchvision_decode_jpeg, self._torchvision_image_read_mode

    def _require_nvimgcodec(self):
        if self._nvimgcodec is None or self._nvimgcodec_decoder is None:
            try:
                from nvidia import nvimgcodec
            except ImportError as exc:
                raise RuntimeError(
                    "CUDA depth JPEG2000 decode requires NVIDIA nvImageCodec with nvJPEG2000 support. "
                    "Install nvidia-nvimgcodec-cu12[nvjpeg2k] or the matching CUDA wheel for this system."
                ) from exc

            backend = nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY)
            self._nvimgcodec = nvimgcodec
            self._nvimgcodec_decoder = nvimgcodec.Decoder(
                device_id=self.device.index if self.device.index is not None else torch.cuda.current_device(),
                backends=[backend],
                options=":fancy_upsampling=0",
            )
        return self._nvimgcodec

    def _make_image_tensor(self, rgb: torch.Tensor, depth: torch.Tensor) -> Optional[torch.Tensor]:
        if self.image_mode == "rgb":
            return rgb
        if self.image_mode == "grayscale":
            return self._grayscale(rgb)
        if self.image_mode == "depth":
            return depth.clone()
        if self.image_mode == "rgbd":
            return torch.cat([rgb, depth], dim=0)
        if self.image_mode == "grayscale_depth":
            return torch.cat([self._grayscale(rgb), depth], dim=0)
        return None

    @staticmethod
    def _grayscale(rgb: torch.Tensor) -> torch.Tensor:
        return (rgb[0] * 0.2989 + rgb[1] * 0.5870 + rgb[2] * 0.1140).unsqueeze(0)


class RLRoverLabCompressedRGBDDatasetRandom(_CompressedRGBDBase):
    """Random-transition loader for RLRoverLab's compressed zero-duplication RGB-D HDF5 files."""

    def __init__(
        self,
        file_path: str,
        min_idx: int = 0,
        max_idx: Optional[int] = None,
        total_samples: int = 120000,
        proprioceptive_keys: List[str] = ["angle_diff", "distance", "heading"],
        image_size: Optional[Sequence[int]] = None,
        image_mode: Optional[str] = "rgb",
        use_frame_stacking: bool = False,
        frame_stack_stride: int = 3,
        num_stacked_frames: int = 3,
        device: str | torch.device = "cuda:0" if torch.cuda.is_available() else "cpu",
        decode_backend: str = "cuda",
        include_raw_rgb: bool = False,
    ):
        super().__init__(
            file_path=file_path,
            min_idx=min_idx,
            max_idx=max_idx,
            total_samples=total_samples,
            proprioceptive_keys=proprioceptive_keys,
            image_size=image_size,
            image_mode=image_mode,
            device=device,
            decode_backend=decode_backend,
            include_raw_rgb=include_raw_rgb,
        )
        self.use_frame_stacking = bool(use_frame_stacking)
        self.frame_stack_stride = int(frame_stack_stride)
        self.num_stacked_frames = int(num_stacked_frames)
        if self.frame_stack_stride <= 0:
            raise ValueError("frame_stack_stride must be positive.")
        if self.num_stacked_frames <= 0:
            raise ValueError("num_stacked_frames must be positive.")

        self.min_history_timestep = (
            (self.num_stacked_frames - 1) * self.frame_stack_stride if self.use_frame_stacking else 0
        )
        self.valid_transition_indices = self._transition_indices_for_selected_episodes(self.min_history_timestep)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int):
        transition_index = int(self.valid_transition_indices[np.random.randint(0, len(self.valid_transition_indices))])
        with h5py.File(self.file_path, "r") as file:
            transition_indices = np.asarray([transition_index], dtype=np.int64)
            actions, rewards, dones = self._read_transition_arrays(file, transition_indices)
            if self.use_frame_stacking:
                obs = self._read_stacked_transition_observation(file, transition_index, next_observation=False)
                next_obs = self._read_stacked_transition_observation(file, transition_index, next_observation=True)
            else:
                obs = self._read_observation(file, int(self.obs_index[transition_index]))
                next_obs = self._read_observation(file, int(self.next_obs_index[transition_index]))

        action = actions[0]
        reward = rewards[0]
        done = dones[0]
        weight = torch.ones_like(reward)
        mask = torch.ones_like(reward)
        return obs, action, reward, next_obs, done, weight, mask

    def _read_stacked_transition_observation(
        self,
        file: h5py.File,
        transition_index: int,
        next_observation: bool,
    ) -> Dict[str, torch.Tensor]:
        episode = int(self.episode_id[transition_index])
        episode_timestep = int(self.episode_transition_index[transition_index])
        local_timesteps = [
            episode_timestep - i * self.frame_stack_stride
            for i in range(self.num_stacked_frames - 1, -1, -1)
        ]
        offset = int(self.obs_offsets[episode])
        shift = 1 if next_observation else 0
        obs_indices = np.asarray([offset + timestep + shift for timestep in local_timesteps], dtype=np.int64)
        sequence = self._read_observation_sequence(file, obs_indices)

        stacked: Dict[str, torch.Tensor] = {}
        for key, value in sequence.items():
            if key in {"image", "depth", "rgb"}:
                stacked[key] = torch.cat([frame for frame in value], dim=0)
            else:
                stacked[key] = value[-1]
        return stacked


class RLRoverLabCompressedRGBDRandomSequenceDataset(_CompressedRGBDBase):
    """Random contiguous-sequence loader for recurrent BC/IQL on compressed RLRoverLab RGB-D files."""

    def __init__(
        self,
        file_path: str,
        sequence_length: int = 16,
        min_idx: int = 0,
        max_idx: Optional[int] = None,
        total_samples: int = 100000,
        proprioceptive_keys: List[str] = ["angle_diff", "distance", "heading"],
        image_size: Optional[Sequence[int]] = None,
        image_mode: Optional[str] = "rgb",
        return_hidden_reset_mask: bool = False,
        device: str | torch.device = "cuda:0" if torch.cuda.is_available() else "cpu",
        decode_backend: str = "cuda",
        include_raw_rgb: bool = False,
    ):
        super().__init__(
            file_path=file_path,
            min_idx=min_idx,
            max_idx=max_idx,
            total_samples=total_samples,
            proprioceptive_keys=proprioceptive_keys,
            image_size=image_size,
            image_mode=image_mode,
            device=device,
            decode_backend=decode_backend,
            include_raw_rgb=include_raw_rgb,
        )
        self.sequence_length = int(sequence_length)
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        self.return_hidden_reset_mask = bool(return_hidden_reset_mask)
        self.valid_sequences = self._sequence_starts(self.sequence_length)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int):
        episode, start_idx = self.valid_sequences[np.random.randint(0, len(self.valid_sequences))]
        end_idx = start_idx + self.sequence_length
        transition_start = int(self.transition_offsets[episode]) + start_idx
        transition_indices = np.arange(transition_start, transition_start + self.sequence_length, dtype=np.int64)
        obs_offset = int(self.obs_offsets[episode])
        obs_indices = np.arange(obs_offset + start_idx, obs_offset + end_idx, dtype=np.int64)
        next_obs_indices = obs_indices + 1

        with h5py.File(self.file_path, "r") as file:
            actions, rewards, dones = self._read_transition_arrays(file, transition_indices)
            obs = self._read_observation_sequence(file, obs_indices)
            next_obs = self._read_observation_sequence(file, next_obs_indices)

        weights = torch.ones_like(rewards)
        masks = torch.ones_like(rewards)
        if not self.return_hidden_reset_mask:
            return obs, actions, rewards, next_obs, dones, weights, masks

        hidden_reset = torch.zeros(self.sequence_length, 1, dtype=torch.float32, device=self.device)
        hidden_reset[0] = 1.0 if start_idx == 0 else 0.0
        for timestep in range(self.sequence_length - 1):
            if bool(dones[timestep].item()):
                hidden_reset[timestep + 1] = 1.0
        return obs, actions, rewards, next_obs, dones, weights, masks, hidden_reset
