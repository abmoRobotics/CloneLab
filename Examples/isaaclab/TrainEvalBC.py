"""Training and Evaluation script for Behavior Cloning (BC).

This script demonstrates how to use the BehaviourCloning algorithm with
standard feedforward models for imitation learning benchmarking against IQL.
"""

from typing import Dict, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
import gymnasium as gym

from models_cai import actor_gaussian_image
from CloneRL.utils import set_seed
from CloneRL.algorithms.torch.imitation_learning.bc import BehaviourCloning
from CloneRL.dataloader.hdf.hdf_loader import HDF5DictDatasetRandom
from CloneRL.trainers.torch.sequential import SequentialTrainer as Trainer


# Parse arguments
parser = argparse.ArgumentParser("Behavior Cloning Training Script")
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


def train_bc():
    """Train Behavior Cloning model."""
    
    print("="*60)
    print("Behavior Cloning (BC) Training")
    print("="*60)
    
    # Dataset path
    data = "/home/robotlab/ws/RLRoverLab/datasets/new_camera_pos_160_90.hdf5"
    
    # Dataset configuration
    use_frame_stacking = False
    frame_stack_stride = 3
    num_stacked_frames = 3
    
    # Create datasets
    dataset = HDF5DictDatasetRandom(
        data, 
        min_idx=10000, 
        total_samples=240000,
        use_frame_stacking=use_frame_stacking,
        frame_stack_stride=frame_stack_stride, 
        num_stacked_frames=num_stacked_frames
    )
    dataset_val = HDF5DictDatasetRandom(
        data, 
        min_idx=1, 
        max_idx=100, 
        total_samples=10000,
        use_frame_stacking=use_frame_stacking,
        frame_stack_stride=frame_stack_stride,
        num_stacked_frames=num_stacked_frames
    )
    
    print(f"Training dataset size: {len(dataset)}")
    print(f"Validation dataset size: {len(dataset_val)}")
    
    # Channel multiplier for frame stacking
    image_channel_multiplier = num_stacked_frames if use_frame_stacking else 1
    depth_channel_multiplier = num_stacked_frames if use_frame_stacking else 1
    
    # Model configuration
    model_config = {
        "proprioception_channels": 3,
        "image_channels": 3 * image_channel_multiplier,  # RGB: 3 or 9 channels
        "depth_channels": 1 * depth_channel_multiplier,  # Depth: 1 or 3 channels
        "action_dim": 2,
        "mlp_features": [512, 256, 128, 64],
        "image_input_dim": [160, 90],
        "image_encoder_features": [8, 16, 32, 64],
        "image_fc_features": [160, 120, 60],
        "activation": "leaky_relu",
        "dropout_rate": 0,
        "use_batch_norm": False
    }
    
    # BC algorithm configuration
    bc_config = {
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "lr_scheduler": None,
    }
    
    # Create actor model
    actor = actor_gaussian_image(**model_config).to("cuda:0")
    
    # Print model info
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Actor parameters: {count_parameters(actor):,}")
    
    # Create BC agent
    agent = BehaviourCloning(
        actor_policy=actor,
        cfg=bc_config,
        **bc_config,
    )
    
    # Trainer configuration
    trainer_config = {
        "batch_size": 100,
        "epochs": 1,
        "num_workers": 16,
        "shuffle": True,
        "early_stopping_patience": 10,
        "save_freq": 2,
        "validation_freq": 1,
        "log_freq": 50,
        "mixed_precision": True,
    }
    
    # Create trainer
    trainer = Trainer(
        cfg=trainer_config,
        policy=agent,
        dataset=dataset,
        val_dataset=dataset_val,
    )
    
    # Start training
    print("\nStarting BC Training...")
    trainer.train()
    
    return trainer


def eval_bc(trainer: Trainer):
    """Evaluate the trained BC policy."""
    
    env = load_isaaclab_env(task_name="AAURoverEnvRGBDRaw-v0")
    
    print("\n" + "="*60)
    print("Starting BC Evaluation")
    print("="*60)
    
    trainer.evaluate(
        env,
        num_steps=10000,
        use_frame_stacking=False,
        frame_stack_stride=3,
        num_stacked_frames=3
    )


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    print("="*60)
    print("Behavior Cloning (BC) Benchmark")
    print("="*60)
    
    trainer = train_bc()
    eval_bc(trainer)
