from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from CloneRL.dataloader.hdf import RLRoverLabCompressedRGBDDatasetRandom
from CloneRL.dataloader.hdf.hdf_loader import HDF5DictDatasetRandom


DEFAULT_DATASETS = {
    "wvga": {
        "legacy": "bench_wvga_legacy_250_20260521.hdf5",
        "compressed": "bench_wvga_compressed_250_20260521.hdf5",
    },
    "hd720": {
        "legacy": "bench_hd720_legacy_250_20260521.hdf5",
        "compressed": "bench_hd720_compressed_250_20260521.hdf5",
    },
    "hd1080": {
        "legacy": "bench_hd1080_legacy_250_ep2_20260521.hdf5",
        "compressed": "bench_hd1080_compressed_250_ep10_rerun_20260521.hdf5",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Benchmark legacy vs compressed RLRoverLab dataloaders.")
    parser.add_argument(
        "--datasets_root",
        type=Path,
        default=Path("/home/anton/ws/projects/RLRoverLab/datasets"),
        help="Folder containing the benchmark HDF5 files.",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=["wvga", "hd720", "hd1080"],
        choices=sorted(DEFAULT_DATASETS),
        help="Dataset pairs to benchmark.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--warmup_batches", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[160, 90], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--compressed_decode_backend",
        choices=("cuda", "pillow"),
        default="cuda",
        help="Use cuda for nvJPEG/nvJPEG2000, or pillow for CPU fallback debugging.",
    )
    parser.add_argument("--output_csv", type=Path, default=None)
    return parser.parse_args()


class LegacyModelReadyDataset(Dataset):
    """Adapt the legacy loader to the same model-ready shape used by the compressed loader."""

    def __init__(self, file_path: str, total_samples: int, image_size: list[int], device: str):
        self.dataset = HDF5DictDatasetRandom(file_path, total_samples=total_samples)
        self.size_hw = (int(image_size[1]), int(image_size[0]))
        self.device = torch.device(device)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        obs, action, reward, next_obs, done, weight, mask = self.dataset[idx]
        obs = self._resize_obs(obs)
        next_obs = self._resize_obs(next_obs)
        return obs, action.to(self.device), reward.to(self.device), next_obs, done.to(self.device), weight.to(self.device), mask.to(self.device)

    def _resize_obs(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key, value in obs.items():
            value = value.to(self.device)
            if key in {"image", "depth", "rgb"} and value.ndim == 3:
                value = _resize_chw(value, self.size_hw)
            out[key] = value
        return out


def _resize_chw(tensor: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    if tuple(tensor.shape[-2:]) == size_hw:
        return tensor
    kwargs = {"size": size_hw, "mode": "bilinear", "align_corners": False}
    try:
        return F.interpolate(tensor.unsqueeze(0), antialias=True, **kwargs).squeeze(0)
    except TypeError:
        return F.interpolate(tensor.unsqueeze(0), **kwargs).squeeze(0)


def _flatten_tensors(value: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _flatten_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _flatten_tensors(item)


def _tensor_bytes(value: Any) -> int:
    return sum(t.numel() * t.element_size() for t in _flatten_tensors(value))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _peak_allocated_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**2


def _peak_reserved_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_reserved(device) / 1024**2


def benchmark_dataset(name: str, dataset: Dataset, args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    iterator = iter(loader)

    for _ in range(args.warmup_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        _sync(device)
        del batch

    _sync(device)
    _reset_peak(device)
    gc.collect()

    sample_count = 0
    payload_bytes = 0
    start = time.perf_counter()
    for _ in range(args.batches):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        _sync(device)
        sample_count += int(batch[1].shape[0])
        payload_bytes += _tensor_bytes(batch)
        del batch
    _sync(device)
    elapsed = time.perf_counter() - start

    return {
        "loader": name,
        "samples": sample_count,
        "seconds": elapsed,
        "samples_per_sec": sample_count / elapsed if elapsed > 0 else float("inf"),
        "avg_payload_mb_per_batch": payload_bytes / args.batches / 1024**2,
        "peak_allocated_mb": _peak_allocated_mb(device),
        "peak_reserved_mb": _peak_reserved_mb(device),
    }


def format_table(rows: list[Dict[str, Any]]) -> str:
    columns = [
        "resolution",
        "loader",
        "samples_per_sec",
        "seconds",
        "avg_payload_mb_per_batch",
        "peak_allocated_mb",
        "peak_reserved_mb",
        "speedup_vs_legacy",
        "peak_allocated_delta_mb",
    ]
    widths = {column: len(column) for column in columns}
    rendered_rows = []
    for row in rows:
        rendered = {}
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.2f}"
            else:
                value = str(value)
            rendered[column] = value
            widths[column] = max(widths[column], len(value))
        rendered_rows.append(rendered)

    header = " | ".join(column.ljust(widths[column]) for column in columns)
    sep = "-+-".join("-" * widths[column] for column in columns)
    body = [" | ".join(row[column].ljust(widths[column]) for column in columns) for row in rendered_rows]
    return "\n".join([header, sep, *body])


def main() -> None:
    args = parse_args()
    total_samples = max(args.batch_size * (args.batches + args.warmup_batches + 2), 512)
    rows: list[Dict[str, Any]] = []

    for resolution in args.resolutions:
        files = DEFAULT_DATASETS[resolution]
        legacy_path = args.datasets_root / files["legacy"]
        compressed_path = args.datasets_root / files["compressed"]

        if not legacy_path.exists():
            print(f"[WARN] Missing legacy file for {resolution}: {legacy_path}")
            continue
        if not compressed_path.exists():
            print(f"[WARN] Missing compressed file for {resolution}: {compressed_path}")
            continue

        print(f"[INFO] Benchmarking {resolution} legacy: {legacy_path.name}")
        legacy = LegacyModelReadyDataset(
            str(legacy_path),
            total_samples=total_samples,
            image_size=args.image_size,
            device=args.device,
        )
        legacy_result = benchmark_dataset("legacy", legacy, args)
        legacy_result["resolution"] = resolution
        rows.append(legacy_result)

        print(f"[INFO] Benchmarking {resolution} compressed: {compressed_path.name}")
        compressed = RLRoverLabCompressedRGBDDatasetRandom(
            str(compressed_path),
            total_samples=total_samples,
            image_size=args.image_size,
            device=args.device,
            decode_backend=args.compressed_decode_backend,
        )
        compressed_result = benchmark_dataset("compressed", compressed, args)
        compressed_result["resolution"] = resolution
        compressed_result["speedup_vs_legacy"] = (
            compressed_result["samples_per_sec"] / legacy_result["samples_per_sec"]
            if legacy_result["samples_per_sec"] > 0
            else float("inf")
        )
        compressed_result["peak_allocated_delta_mb"] = (
            compressed_result["peak_allocated_mb"] - legacy_result["peak_allocated_mb"]
        )
        rows.append(compressed_result)

    baseline_by_resolution = {
        row["resolution"]: row for row in rows if row["loader"] == "legacy"
    }
    for row in rows:
        if row["loader"] == "legacy":
            row["speedup_vs_legacy"] = 1.0
            row["peak_allocated_delta_mb"] = 0.0
        elif "speedup_vs_legacy" not in row:
            baseline = baseline_by_resolution.get(row["resolution"])
            if baseline is not None:
                row["speedup_vs_legacy"] = row["samples_per_sec"] / baseline["samples_per_sec"]
                row["peak_allocated_delta_mb"] = row["peak_allocated_mb"] - baseline["peak_allocated_mb"]

    print()
    print(format_table(rows))

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] Wrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
