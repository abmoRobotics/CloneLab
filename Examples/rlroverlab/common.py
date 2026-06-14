from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_TASK = "AAURoverEnvRGBDRaw-v0"

DEFAULT_EVAL_CMD_CONTAINER = (
    "/isaac-sim/python.sh /workspace/rlroverlab/examples/04_clonelab/eval_policy.py"
)
DEFAULT_EVAL_CMD_HOST = (
    "docker exec rover-lab-base "
    "/isaac-sim/python.sh /workspace/rlroverlab/examples/04_clonelab/eval_policy.py"
)

FEEDFORWARD_ACTOR_CONFIG: dict[str, Any] = {
    "proprioception_channels": 3,
    "image_channels": 3,
    "depth_channels": 1,
    "action_dim": 2,
    "mlp_features": [512, 256, 128, 64],
    "image_input_dim": [160, 90],
    "image_encoder_features": [8, 16, 32, 64],
    "image_fc_features": [160, 120, 60],
    "activation": "leaky_relu",
    "dropout_rate": 0,
    "use_batch_norm": False,
}

FEEDFORWARD_VALUE_CONFIG: dict[str, Any] = {
    key: value for key, value in FEEDFORWARD_ACTOR_CONFIG.items() if key != "action_dim"
}

RECURRENT_BASE_CONFIG: dict[str, Any] = {
    "image_channels": 3,
    "depth_channels": 1,
    "image_size": [160, 90],
    "proprioception_channels": 3,
    "hidden_size": 128,
    "num_layers": 2,
    "image_encoder_features": [8, 16, 32, 64],
    "image_fc_features": [120, 60],
    "mlp_features": [512, 256, 160, 128],
    "device": "cuda:0",
}


def image_channels_for_mode(image_mode: str) -> int:
    mode = image_mode.lower()
    if mode == "rgb":
        return 3
    if mode in {"grayscale", "depth"}:
        return 1
    if mode == "rgbd":
        return 4
    if mode == "grayscale_depth":
        return 2
    if mode in {"none", "null"}:
        return 0
    raise ValueError(f"Unsupported image_mode: {image_mode}")


def build_module(factory_spec: str, config: dict[str, Any], device: str):
    from CloneRL.models.factory import build_model

    return build_model(factory_spec, config, device)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def checkpoint_dir_from_wandb() -> Path:
    import wandb

    if wandb.run is None:
        raise RuntimeError("No active W&B run; cannot infer CloneLab checkpoint directory.")
    return Path("runs") / wandb.run.project / wandb.run.id / "checkpoints"


def save_export_config(
    checkpoint_dir: Path,
    policy_type: str,
    actor_factory: str,
    actor_config: dict[str, Any],
    checkpoint_name: str = "final_model.pt",
) -> Path:
    from CloneRL.export.policy import write_export_config

    path = write_export_config(
        checkpoint_dir=checkpoint_dir,
        policy_type=policy_type,
        model_factory=actor_factory,
        model_config=actor_config,
        checkpoint_name=checkpoint_name,
    )
    print(f"[INFO] Wrote export config: {path}")
    return path


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=str, default=None, help="Training HDF5 dataset.")
    parser.add_argument("--val_dataset", type=str, default=None, help="Validation HDF5 dataset. Defaults to --dataset.")
    parser.add_argument(
        "--dataset_format",
        type=str,
        default="auto",
        choices=("auto", "legacy", "compressed_rgbd", "dino_da"),
        help=(
            "HDF5 layout. 'auto' uses the compressed RLRoverLab loader or the cached DINO/DA loader "
            "when those schemas are detected."
        ),
    )
    parser.add_argument(
        "--compressed_decode_backend",
        type=str,
        default="cuda",
        choices=("cuda", "pillow"),
        help="Decoder for compressed RGB-D files. 'cuda' uses nvJPEG/nvJPEG2000; 'pillow' is a CPU debug fallback.",
    )
    parser.add_argument("--min_idx", type=int, default=0, help="Minimum training episode index.")
    parser.add_argument("--max_idx", type=int, default=None, help="Maximum training episode index.")
    parser.add_argument("--val_min_idx", type=int, default=0, help="Minimum validation episode index.")
    parser.add_argument("--val_max_idx", type=int, default=100, help="Maximum validation episode index.")
    parser.add_argument("--total_samples", type=int, default=240000, help="Virtual training sample count.")
    parser.add_argument("--val_samples", type=int, default=10000, help="Virtual validation sample count.")
    parser.add_argument(
        "--proprioceptive_keys",
        nargs="+",
        default=["angle_diff", "distance", "heading"],
        help="Observation keys concatenated into CloneLab proprioceptive state.",
    )


def resolve_dataset_format(file_path: str, requested_format: str = "auto") -> str:
    if requested_format == "legacy":
        return "legacy"
    if requested_format not in {"auto", "compressed_rgbd", "dino_da"}:
        raise ValueError(f"Unsupported dataset_format: {requested_format}")

    from CloneRL.dataloader.hdf import is_rlroverlab_compressed_rgbd
    from CloneRL.dataloader.hdf import is_rlroverlab_dino_da_features

    is_compressed = is_rlroverlab_compressed_rgbd(file_path)
    is_dino_da = is_rlroverlab_dino_da_features(file_path)
    if requested_format == "compressed_rgbd" and not is_compressed:
        raise ValueError(f"--dataset_format compressed_rgbd was requested, but {file_path} is not a compressed v2 file.")
    if requested_format == "dino_da" and not is_dino_da:
        raise ValueError(f"--dataset_format dino_da was requested, but {file_path} is not a cached DINO/DA file.")
    if requested_format == "dino_da":
        return "dino_da"
    if is_dino_da:
        return "dino_da"
    return "compressed_rgbd" if is_compressed else "legacy"


def build_feedforward_hdf5_dataset(
    file_path: str,
    args: argparse.Namespace,
    model_config: dict[str, Any],
    *,
    min_idx: int,
    max_idx: int | None,
    total_samples: int,
):
    dataset_format = resolve_dataset_format(file_path, getattr(args, "dataset_format", "auto"))
    if dataset_format == "dino_da":
        raise ValueError("Cached DINO/DA datasets are only supported by recurrent BC training for now.")
    if dataset_format == "compressed_rgbd":
        from CloneRL.dataloader.hdf import RLRoverLabCompressedRGBDDatasetRandom

        _configure_compressed_decode_workers(args)
        print(f"[INFO] Loading compressed RLRoverLab RGB-D dataset: {file_path}")
        return RLRoverLabCompressedRGBDDatasetRandom(
            file_path,
            min_idx=min_idx,
            max_idx=max_idx,
            total_samples=total_samples,
            proprioceptive_keys=args.proprioceptive_keys,
            image_size=model_config.get("image_input_dim") or model_config.get("image_size"),
            image_mode=getattr(args, "image_mode", "rgb"),
            use_frame_stacking=getattr(args, "frame_stacking", False),
            frame_stack_stride=getattr(args, "frame_stack_stride", 3),
            num_stacked_frames=getattr(args, "num_stacked_frames", 3),
            device=args.device,
            decode_backend=args.compressed_decode_backend,
        )

    from CloneRL.dataloader.hdf.hdf_loader import HDF5DictDatasetRandom

    return HDF5DictDatasetRandom(
        file_path,
        min_idx=min_idx,
        max_idx=max_idx,
        total_samples=total_samples,
        proprioceptive_keys=args.proprioceptive_keys,
        use_frame_stacking=getattr(args, "frame_stacking", False),
        frame_stack_stride=getattr(args, "frame_stack_stride", 3),
        num_stacked_frames=getattr(args, "num_stacked_frames", 3),
    )


def build_recurrent_hdf5_dataset(
    file_path: str,
    args: argparse.Namespace,
    model_config: dict[str, Any],
    *,
    min_idx: int,
    max_idx: int | None,
    total_samples: int,
):
    dataset_format = resolve_dataset_format(file_path, getattr(args, "dataset_format", "auto"))
    if dataset_format == "dino_da":
        from CloneRL.dataloader.hdf import RLRoverLabDinoDARandomSequenceDataset

        print(f"[INFO] Loading cached DINO/DA sequence dataset: {file_path}")
        return RLRoverLabDinoDARandomSequenceDataset(
            file_path,
            sequence_length=args.sequence_length,
            min_idx=min_idx,
            max_idx=max_idx,
            total_samples=total_samples,
            proprioceptive_keys=args.proprioceptive_keys,
        )

    if dataset_format == "compressed_rgbd":
        from CloneRL.dataloader.hdf import RLRoverLabCompressedRGBDRandomSequenceDataset

        _configure_compressed_decode_workers(args)
        print(f"[INFO] Loading compressed RLRoverLab RGB-D sequence dataset: {file_path}")
        return RLRoverLabCompressedRGBDRandomSequenceDataset(
            file_path,
            sequence_length=args.sequence_length,
            min_idx=min_idx,
            max_idx=max_idx,
            total_samples=total_samples,
            proprioceptive_keys=args.proprioceptive_keys,
            image_size=model_config.get("image_size") or model_config.get("image_input_dim"),
            image_mode=args.image_mode,
            device=args.device,
            decode_backend=args.compressed_decode_backend,
        )

    from CloneRL.dataloader.hdf import HDF5RandomSequenceGRUDataset

    return HDF5RandomSequenceGRUDataset(
        file_path,
        sequence_length=args.sequence_length,
        min_idx=min_idx,
        max_idx=max_idx,
        total_samples=total_samples,
        proprioceptive_keys=args.proprioceptive_keys,
        image_mode=args.image_mode,
    )


def _configure_compressed_decode_workers(args: argparse.Namespace) -> None:
    if args.compressed_decode_backend == "cuda" and getattr(args, "num_workers", 0) != 0:
        print("[INFO] CUDA compressed RGB-D decode runs in the main process; setting num_workers=0.")
        args.num_workers = 0


def add_trainer_args(parser: argparse.ArgumentParser, batch_size: int, epochs: int) -> None:
    parser.add_argument("--batch_size", type=int, default=batch_size, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=epochs, help="Number of training epochs.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch factor.")
    parser.add_argument("--save_freq", type=int, default=2, help="Checkpoint frequency in epochs.")
    parser.add_argument("--validation_freq", type=int, default=1, help="Validation frequency in epochs.")
    parser.add_argument("--log_freq", type=int, default=100, help="Batch logging frequency.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device.")
    parser.add_argument("--wandb_project", type=str, default=None, help="Optional W&B project name.")
    parser.add_argument("--wandb_mode", type=str, default=None, help="Optional W&B mode, for example offline.")


def add_frame_stack_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--frame_stacking", action="store_true", default=False, help="Use strided frame stacking.")
    parser.add_argument("--frame_stack_stride", type=int, default=3, help="Stride between stacked frames.")
    parser.add_argument("--num_stacked_frames", type=int, default=3, help="Number of frames to stack.")


def add_eval_args(parser: argparse.ArgumentParser, recurrent: bool = False) -> None:
    parser.add_argument(
        "--eval_after_train",
        action="store_true",
        default=False,
        help="Run RLRoverLab eval after training.",
    )
    parser.add_argument("--rlroverlab_eval_cmd", type=str, default=None, help="Command prefix for RLRoverLab eval.")
    parser.add_argument("--eval_task", type=str, default=DEFAULT_TASK, help="RLRoverLab task to evaluate.")
    parser.add_argument("--eval_steps", type=int, default=1000, help="RLRoverLab eval steps.")
    parser.add_argument("--eval_num_envs", type=int, default=None, help="RLRoverLab eval num_envs.")
    parser.add_argument("--eval_checkpoint_name", type=str, default="best_model.pt", help="Actor checkpoint filename.")
    parser.add_argument("--eval_metrics_out", type=str, default=None, help="Optional eval metrics JSON path.")
    parser.add_argument(
        "--eval_recurrent",
        action="store_true",
        default=recurrent,
        help="Evaluate with recurrent state resets.",
    )


def configure_wandb_env(args: argparse.Namespace) -> None:
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode


def default_eval_cmd() -> str:
    if Path("/workspace/rlroverlab/examples/04_clonelab/eval_policy.py").exists():
        return DEFAULT_EVAL_CMD_CONTAINER
    return DEFAULT_EVAL_CMD_HOST


def run_eval_after_train(args: argparse.Namespace, checkpoint_dir: Path) -> None:
    if not args.eval_after_train:
        return

    metrics_out = args.eval_metrics_out
    if metrics_out is None:
        metrics_out = str(checkpoint_dir / "rlroverlab_eval.json")

    command = shlex.split(args.rlroverlab_eval_cmd or default_eval_cmd())
    command.extend(
        [
            "--task",
            args.eval_task,
            "--checkpoint",
            str(checkpoint_dir),
            "--checkpoint_name",
            args.eval_checkpoint_name,
            "--steps",
            str(args.eval_steps),
            "--metrics_out",
            metrics_out,
        ]
    )
    if args.eval_num_envs is not None:
        command.extend(["--num_envs", str(args.eval_num_envs)])
    if args.eval_recurrent:
        command.append("--recurrent")

    print("[INFO] Running RLRoverLab evaluation:")
    print("[INFO] " + shlex.join(command))
    subprocess.run(command, check=True)

    metrics_path = Path(metrics_out)
    if metrics_path.exists():
        print(f"[INFO] RLRoverLab metrics written to {metrics_path}")
