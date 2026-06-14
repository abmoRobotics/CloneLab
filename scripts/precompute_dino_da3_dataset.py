#!/usr/bin/env python3
"""Precompute DINOv3 tokens and Depth Anything 3 pseudo-depth for CloneLab HDF5 data.

The input is expected to use the RLRoverLab compressed RGB-D v2 schema:

    observations/rgb_jpeg
    observations/depth_jp2

The output keeps the transition/index/state structure, drops the compressed
visual payloads, and writes:

    observations/dino_tokens_t  float16 [N, 577, 384]
    observations/da_depth_t     float16 [N, 1, 72, 128]

Use --feature-backend mock for a quick local smoke test that exercises the HDF5
copying, RGB JPEG decoding, chunked writes, resume metadata, and shape checks
without downloading model weights.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


SOURCE_SCHEMA = "rlroverlab.offline_rgbd_v2"
OUTPUT_SCHEMA = "rlroverlab.offline_dino_da3_v1"

DINO_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
DA3_MODEL_ID = "depth-anything/DA3-SMALL"

DINO_SIZE_WH = (512, 288)
DINO_TOKENS = 577
DINO_DIM = 384
DA_DEPTH_HW = (72, 128)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class BatchFeatures:
    dino_tokens: np.ndarray
    da_depth: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute DINOv3 ViT-S/16 tokens and Depth Anything 3 pseudo-depth for CloneLab datasets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/rover_rgbd_hd720_expert_400k.hdf5"),
        help="Input compressed RLRoverLab HDF5 file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/rover_dino_da3_512x288_400k.hdf5"),
        help="Output HDF5 file to create.",
    )
    parser.add_argument(
        "--feature-backend",
        choices=("real", "mock"),
        default="real",
        help="Use real DINOv3/DA3 models or deterministic mock tensors for smoke tests.",
    )
    parser.add_argument("--dino-model", default=DINO_MODEL_ID, help="Hugging Face DINOv3 model id or local path.")
    parser.add_argument("--da3-model", default=DA3_MODEL_ID, help="Depth Anything 3 model id or local path.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="Torch device.")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="Autocast dtype used for model inference on CUDA.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="DINO batch size.")
    parser.add_argument(
        "--da3-batch-size",
        type=int,
        default=1,
        help=(
            "Depth Anything 3 inference batch size. Keep 1 for independent per-frame pseudo-depth; "
            "larger values let DA3 treat a batch as a multi-view set."
        ),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First observation index to process when creating a new file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of observations to process. Useful for smoke tests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing output file from its feature_observation_count_written attribute.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    parser.add_argument(
        "--store-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="HDF5 dtype for the cached feature tensors.",
    )
    parser.add_argument(
        "--compression",
        choices=("lzf", "gzip", "none"),
        default="lzf",
        help="HDF5 compression for feature datasets.",
    )
    parser.add_argument("--gzip-level", type=int, default=4, help="Compression level when --compression=gzip.")
    parser.add_argument("--dino-width", type=int, default=DINO_SIZE_WH[0], help="DINO input width.")
    parser.add_argument("--dino-height", type=int, default=DINO_SIZE_WH[1], help="DINO input height.")
    parser.add_argument("--da-depth-height", type=int, default=DA_DEPTH_HW[0], help="Cached DA3 depth height.")
    parser.add_argument("--da-depth-width", type=int, default=DA_DEPTH_HW[1], help="Cached DA3 depth width.")
    parser.add_argument(
        "--normalize-da-depth",
        action="store_true",
        help="Per-frame min/max normalize DA3 depth before saving. Default saves raw model depth resized to target shape.",
    )
    parser.add_argument("--da3-process-res", type=int, default=504, help="Depth Anything 3 process_res argument.")
    parser.add_argument(
        "--da3-process-res-method",
        default="upper_bound_resize",
        help="Depth Anything 3 process_res_method argument.",
    )
    parser.add_argument(
        "--show-model-logs",
        action="store_true",
        help="Do not suppress verbose model inference logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    validate_source_path(args.input)

    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive.")

    if args.output.exists():
        if args.overwrite:
            args.output.unlink()
        elif not args.resume:
            raise FileExistsError(f"{args.output} already exists. Use --resume or --overwrite.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    extractor = None
    if not args.output.exists():
        extractor = build_extractor(args)
        initialize_output_file(args)

    with h5py.File(args.output, "r+") as out_file:
        start, stop = processing_range(args, out_file)
        if start >= stop:
            out_file.attrs["feature_complete"] = True
            print(f"[INFO] Nothing to do; output is already processed through observation {start}.")
            return

    if extractor is None:
        extractor = build_extractor(args)
    process_observations(args, extractor, start, stop)
    verify_output(args, stop)


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.da3_batch_size <= 0:
        raise ValueError("--da3-batch-size must be positive.")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided.")
    if args.dino_width % 16 != 0 or args.dino_height % 16 != 0:
        raise ValueError("--dino-width and --dino-height must be multiples of DINOv3 patch size 16.")

    patch_tokens = (args.dino_width // 16) * (args.dino_height // 16)
    expected_without_registers = 1 + patch_tokens
    if expected_without_registers != DINO_TOKENS:
        raise ValueError(
            f"DINO input {args.dino_width}x{args.dino_height} gives {expected_without_registers} CLS+patch tokens; "
            f"this script is configured to write {DINO_TOKENS}. Use 512x288 for [577, 384]."
        )


def validate_source_path(path: Path) -> None:
    with h5py.File(path, "r") as file:
        validate_input_file(file, path)


def initialize_output_file(args: argparse.Namespace) -> None:
    with h5py.File(args.input, "r") as src:
        validate_input_file(src, args.input)
        observation_count = int(src.attrs["total_observations"])

        with h5py.File(args.output, "w") as dst:
            copy_root_attrs(src, dst, args)
            copy_static_groups(src, dst)
            create_feature_datasets(dst, observation_count, args)

            dst.attrs["feature_complete"] = False
            dst.attrs["feature_observation_start_index"] = int(args.start_index)
            dst.attrs["feature_observation_count_written"] = int(args.start_index)
            dst.flush()


def validate_input_file(file: h5py.File, path: Path) -> None:
    schema_name = file.attrs.get("schema_name")
    if schema_name != SOURCE_SCHEMA:
        raise ValueError(f"Expected {SOURCE_SCHEMA!r} input schema in {path}, got {schema_name!r}.")
    required_paths = (
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
    )
    for required_path in required_paths:
        if required_path not in file:
            raise ValueError(f"Input dataset is missing required path: {required_path}")


def copy_root_attrs(src: h5py.File, dst: h5py.File, args: argparse.Namespace) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value
    dst.attrs["schema_name"] = OUTPUT_SCHEMA
    dst.attrs["source_schema_name"] = src.attrs.get("schema_name", "")
    dst.attrs["source_file"] = str(args.input)
    dst.attrs["visual_profile"] = "dino_v3_vits16_tokens_da3_depth"
    dst.attrs["visual_storage_contract"] = "precomputed_float_features"
    dst.attrs["rgb_codec"] = "removed_after_precompute"
    dst.attrs["depth_codec"] = "removed_after_precompute"
    dst.attrs["dino_model"] = args.dino_model
    dst.attrs["dino_input_width"] = int(args.dino_width)
    dst.attrs["dino_input_height"] = int(args.dino_height)
    dst.attrs["dino_tokens_shape"] = json.dumps([DINO_TOKENS, DINO_DIM])
    dst.attrs["dino_register_tokens_removed_when_present"] = True
    dst.attrs["da3_model"] = args.da3_model
    dst.attrs["da_depth_shape"] = json.dumps([1, int(args.da_depth_height), int(args.da_depth_width)])
    dst.attrs["da_depth_normalized_per_frame"] = bool(args.normalize_da_depth)
    dst.attrs["feature_store_dtype"] = args.store_dtype
    dst.attrs["feature_backend"] = args.feature_backend
    dst.attrs["writer_status"] = "in_progress"


def copy_static_groups(src: h5py.File, dst: h5py.File) -> None:
    for name in ("data", "episodes", "index", "transitions"):
        if name in src:
            src.copy(name, dst)

    observations = dst.create_group("observations")
    if "observations" in src:
        for key, value in src["observations"].attrs.items():
            observations.attrs[key] = value
    src.copy("observations/state", observations, name="state")


def create_feature_datasets(dst: h5py.File, observation_count: int, args: argparse.Namespace) -> None:
    store_dtype = np.float16 if args.store_dtype == "float16" else np.float32
    compression_kwargs = compression_options(args)

    dst["observations"].create_dataset(
        "dino_tokens_t",
        shape=(observation_count, DINO_TOKENS, DINO_DIM),
        dtype=store_dtype,
        chunks=(min(1, observation_count), DINO_TOKENS, DINO_DIM),
        **compression_kwargs,
    )
    dst["observations"].create_dataset(
        "da_depth_t",
        shape=(observation_count, 1, int(args.da_depth_height), int(args.da_depth_width)),
        dtype=store_dtype,
        chunks=(min(16, observation_count), 1, int(args.da_depth_height), int(args.da_depth_width)),
        **compression_kwargs,
    )


def compression_options(args: argparse.Namespace) -> dict:
    if args.compression == "none":
        return {}
    if args.compression == "gzip":
        return {"compression": "gzip", "compression_opts": int(args.gzip_level), "shuffle": True}
    return {"compression": "lzf", "shuffle": True}


def processing_range(args: argparse.Namespace, out_file: h5py.File) -> tuple[int, int]:
    observation_count = int(out_file.attrs["total_observations"])
    configured_start = int(out_file.attrs.get("feature_observation_start_index", args.start_index))

    if args.resume:
        start = int(out_file.attrs.get("feature_observation_count_written", configured_start))
    else:
        start = int(args.start_index)

    stop = observation_count if args.limit is None else min(observation_count, start + int(args.limit))
    if start < 0 or start > observation_count:
        raise ValueError(f"Invalid start index {start} for {observation_count} observations.")
    if stop > observation_count:
        raise ValueError(f"Invalid stop index {stop} for {observation_count} observations.")
    return start, stop


def build_extractor(args: argparse.Namespace):
    if args.feature_backend == "mock":
        return MockFeatureExtractor(args)
    return RealFeatureExtractor(args)


def process_observations(args: argparse.Namespace, extractor, start: int, stop: int) -> None:
    started = time.time()
    total = stop - start
    batch_size = int(args.batch_size)
    print(f"[INFO] Processing observations [{start}, {stop}) from {args.input}")

    with h5py.File(args.input, "r") as src, h5py.File(args.output, "r+") as dst:
        rgb_dataset = src["observations/rgb_jpeg"]
        dino_dataset = dst["observations/dino_tokens_t"]
        da_dataset = dst["observations/da_depth_t"]
        batch_starts = range(start, stop, batch_size)

        with tqdm(total=total, unit="obs", desc="Precomputing features", initial=0) as progress:
            for batch_start in batch_starts:
                batch_stop = min(stop, batch_start + batch_size)
                images = [decode_rgb_jpeg(rgb_dataset[index]) for index in range(batch_start, batch_stop)]
                features = extractor.extract(images)

                expected_dino = (batch_stop - batch_start, DINO_TOKENS, DINO_DIM)
                expected_da = (batch_stop - batch_start, 1, int(args.da_depth_height), int(args.da_depth_width))
                if features.dino_tokens.shape != expected_dino:
                    raise RuntimeError(
                        f"DINO tokens shape mismatch: got {features.dino_tokens.shape}, expected {expected_dino}."
                    )
                if features.da_depth.shape != expected_da:
                    raise RuntimeError(f"DA3 depth shape mismatch: got {features.da_depth.shape}, expected {expected_da}.")

                dino_dataset[batch_start:batch_stop] = features.dino_tokens.astype(dino_dataset.dtype, copy=False)
                da_dataset[batch_start:batch_stop] = features.da_depth.astype(da_dataset.dtype, copy=False)
                dst.attrs["feature_observation_count_written"] = int(batch_stop)
                dst.flush()

                processed = batch_stop - start
                elapsed = max(time.time() - started, 1e-6)
                progress.update(batch_stop - batch_start)
                progress.set_postfix(
                    absolute_index=batch_stop,
                    rate=f"{processed / elapsed:.2f} obs/s",
                )

        dst.attrs["feature_complete"] = int(dst.attrs["feature_observation_count_written"]) >= int(dst.attrs["total_observations"])
        dst.attrs["writer_status"] = "complete" if dst.attrs["feature_complete"] else "partial"
        dst.flush()

    elapsed = max(time.time() - started, 1e-6)
    print(f"[INFO] Finished {total} observations in {elapsed / 60:.1f} min ({total / elapsed:.2f} obs/s).")


def decode_rgb_jpeg(encoded: np.ndarray) -> Image.Image:
    try:
        return Image.open(io.BytesIO(np.asarray(encoded, dtype=np.uint8).tobytes())).convert("RGB")
    except UnidentifiedImageError as exc:
        raise RuntimeError("Pillow failed to decode an RGB JPEG frame from observations/rgb_jpeg.") from exc


class MockFeatureExtractor:
    def __init__(self, args: argparse.Namespace):
        self.da_depth_hw = (int(args.da_depth_height), int(args.da_depth_width))

    def extract(self, images: list[Image.Image]) -> BatchFeatures:
        dino = np.empty((len(images), DINO_TOKENS, DINO_DIM), dtype=np.float32)
        da_depth = np.empty((len(images), 1, self.da_depth_hw[0], self.da_depth_hw[1]), dtype=np.float32)
        for batch_index, image in enumerate(images):
            arr = np.asarray(image.resize((16, 16), Image.Resampling.BILINEAR), dtype=np.float32)
            seed = float(arr.mean() / 255.0)
            token_axis = np.linspace(0.0, 1.0, DINO_TOKENS, dtype=np.float32)[:, None]
            dim_axis = np.linspace(0.0, 1.0, DINO_DIM, dtype=np.float32)[None, :]
            dino[batch_index] = seed + token_axis * 0.01 + dim_axis * 0.001

            gray = np.asarray(image.convert("L").resize((self.da_depth_hw[1], self.da_depth_hw[0]), Image.Resampling.BILINEAR))
            da_depth[batch_index, 0] = gray.astype(np.float32) / 255.0
        return BatchFeatures(dino_tokens=dino, da_depth=da_depth)


class RealFeatureExtractor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device)
        self.autocast_dtype = dtype_from_name(args.dtype)
        self.da_depth_hw = (int(args.da_depth_height), int(args.da_depth_width))
        self.dino_size_wh = (int(args.dino_width), int(args.dino_height))

        self.dino = self._load_dino_model(args)
        self.da3 = self._load_da3_model(args)

        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _load_dino_model(self, args: argparse.Namespace) -> torch.nn.Module:
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "Real DINOv3 extraction requires transformers. Install it with:\n"
                "  python -m pip install 'transformers>=4.56' huggingface_hub safetensors"
            ) from exc

        model = AutoModel.from_pretrained(args.dino_model)
        model = model.to(self.device)
        model.eval()
        return model

    def _load_da3_model(self, args: argparse.Namespace):
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:
            raise RuntimeError(
                "Real DA3 extraction requires the Depth Anything 3 package. Install it with:\n"
                "  python -m pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"
            ) from exc

        model = DepthAnything3.from_pretrained(args.da3_model)
        model = model.to(device=self.device)
        model.eval()
        return model

    def extract(self, images: list[Image.Image]) -> BatchFeatures:
        dino_tokens = self.extract_dino(images)
        da_depth = self.extract_da3(images)
        return BatchFeatures(dino_tokens=dino_tokens, da_depth=da_depth)

    @torch.inference_mode()
    def extract_dino(self, images: list[Image.Image]) -> np.ndarray:
        pixel_values = torch.stack([self.dino_preprocess(image) for image in images], dim=0).to(self.device)
        with maybe_autocast(self.device, self.autocast_dtype):
            outputs = self.dino(pixel_values=pixel_values)

        tokens = outputs.last_hidden_state
        tokens = strip_register_tokens_if_needed(tokens)
        if tokens.shape[1:] != (DINO_TOKENS, DINO_DIM):
            raise RuntimeError(
                f"Expected DINO tokens [B, {DINO_TOKENS}, {DINO_DIM}], got {tuple(tokens.shape)}. "
                "Check --dino-width/--dino-height and the selected DINO model."
            )
        return tokens.float().cpu().numpy()

    def dino_preprocess(self, image: Image.Image) -> torch.Tensor:
        resized = image.resize(self.dino_size_wh, Image.Resampling.BICUBIC)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return (tensor.to(self.device) - self.mean[0]) / self.std[0]

    @torch.inference_mode()
    def extract_da3(self, images: list[Image.Image]) -> np.ndarray:
        outputs: list[np.ndarray] = []
        da3_batch_size = int(self.args.da3_batch_size)
        for chunk in chunked(images, da3_batch_size):
            prediction = self.run_da3_inference(chunk)
            depth = torch.as_tensor(np.asarray(prediction.depth), dtype=torch.float32, device=self.device)
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)
            if depth.ndim != 3:
                raise RuntimeError(f"Expected DA3 depth [N, H, W], got {tuple(depth.shape)}.")

            depth = depth.unsqueeze(1)
            depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            if self.args.normalize_da_depth:
                depth = normalize_depth_per_frame(depth)
            depth = F.interpolate(depth, size=self.da_depth_hw, mode="bilinear", align_corners=False)
            outputs.append(depth.float().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def run_da3_inference(self, images: list[Image.Image]):
        image_arrays = [np.asarray(image, dtype=np.uint8) for image in images]
        kwargs = {
            "process_res": int(self.args.da3_process_res),
            "process_res_method": self.args.da3_process_res_method,
            "export_dir": None,
            "export_format": "mini_npz",
        }
        with maybe_suppress_output(enabled=not self.args.show_model_logs):
            try:
                return self.da3.inference(image=image_arrays, **kwargs)
            except TypeError:
                return self.da3.inference(image_arrays, **kwargs)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


class maybe_autocast:
    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.enabled = device.type == "cuda" and dtype != torch.float32
        self.context = torch.autocast(device_type=device.type, dtype=dtype, enabled=self.enabled)

    def __enter__(self):
        return self.context.__enter__()

    def __exit__(self, exc_type, exc, traceback):
        return self.context.__exit__(exc_type, exc, traceback)


@contextlib.contextmanager
def maybe_suppress_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def strip_register_tokens_if_needed(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.shape[1] == DINO_TOKENS:
        return tokens
    if tokens.shape[1] == DINO_TOKENS + 4:
        return torch.cat([tokens[:, :1], tokens[:, 5:]], dim=1)
    return tokens


def normalize_depth_per_frame(depth: torch.Tensor) -> torch.Tensor:
    flat = depth.flatten(start_dim=1)
    mins = flat.min(dim=1).values.view(-1, 1, 1, 1)
    maxs = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return (depth - mins) / (maxs - mins).clamp_min(1e-6)


def chunked(items: list[Image.Image], size: int) -> Iterable[list[Image.Image]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def verify_output(args: argparse.Namespace, processed_stop: int) -> None:
    with h5py.File(args.output, "r") as file:
        if "observations/rgb_jpeg" in file or "observations/depth_jp2" in file:
            raise RuntimeError("Output still contains compressed RGB/depth datasets.")
        dino = file["observations/dino_tokens_t"]
        da_depth = file["observations/da_depth_t"]
        if dino.shape[1:] != (DINO_TOKENS, DINO_DIM):
            raise RuntimeError(f"Invalid dino_tokens_t shape: {dino.shape}")
        if da_depth.shape[1:] != (1, int(args.da_depth_height), int(args.da_depth_width)):
            raise RuntimeError(f"Invalid da_depth_t shape: {da_depth.shape}")
        if int(file.attrs["feature_observation_count_written"]) < processed_stop:
            raise RuntimeError("Output progress metadata was not updated correctly.")
    print(f"[INFO] Verified {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
