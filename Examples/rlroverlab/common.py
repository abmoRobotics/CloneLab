from __future__ import annotations

import argparse
import json
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


def load_json_config(path: str | None, defaults: dict[str, Any]) -> dict[str, Any]:
    config = dict(defaults)
    if path is None:
        return config

    with open(path, encoding="utf-8") as file:
        user_config = json.load(file)
    if not isinstance(user_config, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(user_config).__name__}")
    config.update(user_config)
    return config


def load_object(spec: str):
    if ":" in spec:
        module_name, object_name = spec.split(":", 1)
    else:
        module_name, object_name = spec.rsplit(".", 1)
    module = __import__(module_name, fromlist=[object_name])
    return getattr(module, object_name)


def build_module(factory_spec: str, config: dict[str, Any], device: str):
    module = load_object(factory_spec)(**config)
    return module.to(device)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def checkpoint_dir_from_wandb() -> Path:
    import wandb

    if wandb.run is None:
        raise RuntimeError("No active W&B run; cannot infer CloneLab checkpoint directory.")
    return Path("runs") / wandb.run.project / wandb.run.id / "checkpoints"


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=str, required=True, help="Training HDF5 dataset.")
    parser.add_argument("--val_dataset", type=str, default=None, help="Validation HDF5 dataset. Defaults to --dataset.")
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
