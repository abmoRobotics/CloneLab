"""Training and Evaluation script for Behavior Cloning with Recurrent Models (BC-RNN).

This script demonstrates how to use the BehaviourCloningRNN algorithm with GRU-based
models for imitation learning benchmarking against IQL-RNN.
"""

from typing import Dict, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
import gymnasium as gym

from models_cai import GRUActorGaussian
from CloneRL.utils import set_seed
from CloneRL.algorithms.torch.imitation_learning.bc import BehaviourCloningRNN
from CloneRL.dataloader.hdf.hdf_loader_gru import HDF5RandomSequenceGRUDataset
from CloneRL.trainers.torch.sequential_recurrent import SequentialRecurrentTrainer as Trainer


# Parse arguments
parser = argparse.ArgumentParser("Behavior Cloning RNN Training Script")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="AAURoverEnvSimple-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

set_seed(12345)


def _print_cfg(d, indent=0) -> None:
    """Print the environment configuration."""
    for key, value in d.items():
        if isinstance(value, dict):
            _print_cfg(value, indent + 1)
        else:
            print("  |   " * indent + f"  |-- {key}: {value}")


def load_isaaclab_env(
    task_name: str = "",
    num_envs: Optional[int] = None,
    headless: Optional[bool] = None,
    cli_args: Sequence[str] = [],
    show_cfg: bool = True,
):
    """Load Isaac Lab environment."""
    import argparse
    import atexit
    import gymnasium

    # Check task from command line arguments
    defined = False
    for arg in sys.argv:
        if arg.startswith("--task"):
            defined = True
            break
    
    if defined:
        arg_index = sys.argv.index("--task") + 1
        if arg_index >= len(sys.argv):
            raise ValueError(
                "No task name defined. Set the task_name parameter or use --task <task_name> as command line argument"
            )
        if task_name and task_name != sys.argv[arg_index]:
            print(f"Overriding task ({task_name}) with command line argument ({sys.argv[arg_index]})")
    else:
        if task_name:
            sys.argv.append("--task")
            sys.argv.append(task_name)
        else:
            raise ValueError(
                "No task name defined. Set the task_name parameter or use --task <task_name> as command line argument"
            )

    # Check num_envs from command line arguments
    defined = False
    for arg in sys.argv:
        if arg.startswith("--num_envs"):
            defined = True
            break
    
    if defined:
        if num_envs is not None:
            print("Overriding num_envs with command line argument (--num_envs)")
    elif num_envs is not None and num_envs > 0:
        sys.argv.append("--num_envs")
        sys.argv.append(str(num_envs))

    # Check headless from command line arguments
    defined = False
    for arg in sys.argv:
        if arg.startswith("--headless"):
            defined = True
            break
    
    if defined:
        if headless is not None:
            print("Overriding headless with command line argument (--headless)")
    elif headless is not None:
        sys.argv.append("--headless")

    # Parse arguments
    parser = argparse.ArgumentParser("Isaac Lab: Omniverse Robotics Environments!")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--distributed", action="store_true", default=False)

    # Launch simulation
    try:
        from omni.isaac.lab.app import AppLauncher
    except ModuleNotFoundError:
        from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)

    @atexit.register
    def close_the_simulator():
        app_launcher.app.close()

    try:
        import omni.isaac.lab_tasks
        import rover_envs.envs.navigation.robots.aau_rover
        from omni.isaac.lab_tasks.utils import parse_env_cfg
    except ModuleNotFoundError:
        import isaaclab_tasks
        import rover_envs.envs.navigation.robots.aau_rover
        from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)

    if show_cfg:
        print(f"\nIsaac Lab environment ({args.task})")
        try:
            _print_cfg(cfg)
        except AttributeError:
            pass

    env = gymnasium.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video else None)
    return env


def train_bc_rnn():
    """Train Behavior Cloning RNN model."""
    
    print("="*60)
    print("Behavior Cloning RNN (BC-RNN) Training")
    print("="*60)
    
    # Dataset path
    data = "/home/robotlab/ws/RLRoverLab/datasets/new_camera_pos_160_90.hdf5"
    
    # Sequence configuration
    sequence_length = 16
    
    # ============================================================
    # IMAGE MODE CONFIGURATION
    # ============================================================
    # Options:
    #   - "rgb": Use RGB image (3 channels) for image encoder
    #   - "grayscale": Use grayscale image (1 channel) for image encoder
    #   - "depth": Use depth only (1 channel) for image encoder
    #   - "rgbd": Use RGB + depth (4 channels) for image encoder
    #   - "grayscale_depth": Use grayscale + depth (2 channels) for image encoder
    #   - None or "none": No image encoder, only depth is provided separately
    #
    # use_depth_encoder: Set to False to disable the separate depth encoder
    #   - If True (default), a separate depth encoder processes depth data
    #   - If False + image_mode=None, the model is purely proprioceptive (no images)
    # ============================================================
    image_mode = "grayscale"  # Change this to switch modes
    use_depth_encoder = True  # Set to False for proprioceptive-only
    
    # Set image_channels based on image_mode
    if image_mode == "rgb":
        image_channels = 3
    elif image_mode == "grayscale":
        image_channels = 1
    elif image_mode == "depth":
        image_channels = 1
    elif image_mode == "rgbd":
        image_channels = 4
    elif image_mode == "grayscale_depth":
        image_channels = 2
    else:  # None or "none"
        image_channels = 0  # No image encoder
    
    # Set depth_channels based on use_depth_encoder
    depth_channels = 1 if use_depth_encoder else 0
    
    print(f"Image mode: {image_mode} ({image_channels} channels)")
    print(f"Depth encoder: {'enabled' if use_depth_encoder else 'disabled'} ({depth_channels} channels)")
    
    # Create sequential datasets
    # Use episodes 100+ for training, 0-99 for validation
    dataset = HDF5RandomSequenceGRUDataset(
        file_path=data,
        sequence_length=sequence_length,    
        min_idx=100,
        max_idx=5000,
        total_samples=50000,
        proprioceptive_keys=['angle_diff', 'distance', 'heading'],
        image_mode=image_mode,  # Pass image_mode to dataloader
    )
    
    dataset_val = HDF5RandomSequenceGRUDataset(
        file_path=data,
        sequence_length=sequence_length,
        min_idx=0,
        max_idx=100,
        total_samples=5000,
        proprioceptive_keys=['angle_diff', 'distance', 'heading'],
        image_mode=image_mode,  # Pass image_mode to dataloader
    )
    
    print(f"Training dataset size: {len(dataset)}")
    print(f"Validation dataset size: {len(dataset_val)}")
    print(f"Sequence length: {sequence_length}")
    
    # Base GRU Model configuration
    # Note: The GRU dataloader provides based on image_mode:
    #   - "rgb": 'image' = RGB normalized to [0, 1] (3 channels)
    #   - "depth": 'image' = depth only (1 channel)
    #   - None/"none": no 'image' key, only 'depth' is provided
    #   - 'depth': depth only (1 channel) - always provided for depth_encoder
    gru_base_config = {
        "image_channels": image_channels,  # Set based on image_mode (3 for rgb, 1 for depth, 0 for none)
        "depth_channels": depth_channels,  # Set based on use_depth_encoder
        "image_size": [160, 90],  # Image dimensions [W, H]
        "proprioception_channels": 3,  # angle_diff, distance, heading
        "hidden_size": 128,       # GRU hidden state size
        "num_layers": 2,          # Number of GRU layers
        "image_encoder_features": [8, 16, 32, 64],
        "image_fc_features": [160, 80, 60],
        "mlp_features": [512, 256, 160, 128],
        "device": "cuda:0",
    }
    
    # Actor-specific config (includes action_dim and Gaussian output params)
    actor_config = {
        **gru_base_config,
        "action_dim": 2,
        "min_log_std": -20.0,
        "max_log_std": 2.0,
        "state_independent_log_std": True,
    }
    
    # BC-RNN algorithm configuration
    bc_config = {
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "lr_scheduler": None,
        "reset_hidden_on_done": True,
    }
    
    # Create GRU actor model
    print("Creating GRU actor model...")
    actor = GRUActorGaussian(**actor_config).to("cuda:0")
    
    # Print model info
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Actor parameters: {count_parameters(actor):,}")
    
    # Create BC-RNN agent
    agent = BehaviourCloningRNN(
        actor_policy=actor,
        cfg=bc_config,
        **bc_config,
    )
    
    # Trainer configuration
    trainer_config = {
        "batch_size": 16,          # Smaller batch size for sequences
        "epochs": 4,
        "num_workers": 12,
        "prefetch_factor": 2,
        "shuffle": True,
        "early_stopping_patience": 10,
        "save_freq": 5,
        "validation_freq": 1,
        "log_freq": 50,
        "mixed_precision": True,
        "sequence_length": sequence_length,
        "image_mode": image_mode,  # Pass image_mode to trainer for evaluation
    }
    
    # Create recurrent trainer
    trainer = Trainer(
        cfg=trainer_config,
        policy=agent,
        dataset=dataset,
        val_dataset=dataset_val,
    )
    
    # Start training
    print("\n" + "="*60)
    print("Starting BC-RNN Training")
    print("="*60)
    trainer.train()
    
    return trainer


def eval_bc_rnn(trainer: Trainer):
    """Evaluate the trained BC-RNN policy."""
    
    env = load_isaaclab_env(task_name="AAURoverEnvRGBDRaw-v0")
    
    print("\n" + "="*60)
    print("Starting BC-RNN Evaluation")
    print("="*60)
    
    trainer.evaluate(
        env,
        num_steps=10000,
        deterministic=True,
    )


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    print("="*60)
    print("Behavior Cloning with Recurrent Models (BC-RNN) Benchmark")
    print("="*60)
    
    trainer = train_bc_rnn()
    eval_bc_rnn(trainer)
