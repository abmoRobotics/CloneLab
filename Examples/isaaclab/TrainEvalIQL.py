from typing import Dict
from typing import Optional, Sequence
#import rover_envs
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
import isaaclab
#from isaaclab.app import AppLauncher
from models import TwinQ_image, actor_gaussian_image, v_image
# from skrl.utils import set_seed
from CloneRL.utils import set_seed
#from skrl.envs.loaders.torch import load_isaaclab_env
from CloneRL.algorithms.torch.offline_rl.iql import IQL
from CloneRL.dataloader.hdf import (HDF_DEFAULT_IL_MAPPER,
                                    HDF_DEFAULT_ORL_MAPPER, HDF5Dataset)
from CloneRL.dataloader.hdf.hdf_loader import (
    CloneLabDataset,HDF5DictDataset, HDF5DictDataset2, HDF5DictDatasetRandom)
from CloneRL.trainers.torch.sequential import SequentialTrainer as Trainer
import gymnasium as gym
parser = argparse.ArgumentParser("Welcome to Isaac Lab: Omniverse Robotics Environments!")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="AAURoverEnvSimple-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--agent", type=str, default="PPO", help="Name of the agent.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")


set_seed(12345)

def _print_cfg(d, indent=0) -> None:
    """Print the environment configuration

    :param d: The dictionary to print
    :type d: dict
    :param indent: The indentation level (default: ``0``)
    :type indent: int, optional
    """
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
    import argparse
    import atexit
    import gymnasium

    # check task from command line arguments
    defined = False
    for arg in sys.argv:
        if arg.startswith("--task"):
            defined = True
            break
    # get task name from command line arguments
    if defined:
        arg_index = sys.argv.index("--task") + 1
        if arg_index >= len(sys.argv):
            raise ValueError(
                "No task name defined. Set the task_name parameter or use --task <task_name> as command line argument"
            )
        if task_name and task_name != sys.argv[arg_index]:
            print(f"Overriding task ({task_name}) with command line argument ({sys.argv[arg_index]})")
            #logger.warning(f"Overriding task ({task_name}) with command line argument ({sys.argv[arg_index]})")
    # get task name from function arguments
    else:
        if task_name:
            sys.argv.append("--task")
            sys.argv.append(task_name)
        else:
            raise ValueError(
                "No task name defined. Set the task_name parameter or use --task <task_name> as command line argument"
            )

    # check num_envs from command line arguments
    defined = False
    for arg in sys.argv:
        if arg.startswith("--num_envs"):
            defined = True
            break
    # get num_envs from command line arguments
    if defined:
        if num_envs is not None:
            print("Overriding num_envs with command line argument (--num_envs)")
            #logger.warning("Overriding num_envs with command line argument (--num_envs)")
    # get num_envs from function arguments
    elif num_envs is not None and num_envs > 0:
        sys.argv.append("--num_envs")
        sys.argv.append(str(num_envs))

    # check headless from command line arguments
    defined = False
    for arg in sys.argv:
        if arg.startswith("--headless"):
            defined = True
            break
    # get headless from command line arguments
    if defined:
        if headless is not None:
            print("Overriding headless with command line argument (--headless)")
            #logger.warning("Overriding headless with command line argument (--headless)")
    # get headless from function arguments
    elif headless is not None:
        sys.argv.append("--headless")

    # others command line arguments
    sys.argv += cli_args

    # parse arguments
    parser = argparse.ArgumentParser("Isaac Lab: Omniverse Robotics Environments!")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument(
        "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
    )

    # launch the simulation app
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
        import omni.isaac.lab_tasks  # type: ignore
        import rover_envs.envs.navigation.robots.aau_rover  # noqa: F401
        from omni.isaac.lab_tasks.utils import parse_env_cfg  # type: ignore
    except ModuleNotFoundError:
        import isaaclab_tasks  # type: ignore
        import rover_envs.envs.navigation.robots.aau_rover  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg  # type: ignore

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)

    # print config
    if show_cfg:
        print(f"\nIsaac Lab environment ({args.task})")
        try:
            _print_cfg(cfg)
        except AttributeError as e:
            pass

    # load environment
    env = gymnasium.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video else None)

    return env

def train_iql():

    # Define path to HDF5 file and mapper for training and validation
    # data = "/media/anton/T7 Shield/University/1. Master/Datasets/1. Simulation/dataset1/with_rgb_and_depth_2.hdf5"
    # data = "/media/anton/T7 Shield/University/1. Master/Datasets/1. Simulation/dataset4_180_320/with_rgb_and_depth_0.hdf5"
    #data = "/media/anton/T7 Shield/University/1. Master/Datasets/1. Simulation/dataset3/with_rgb_and_depth_0.hdf5"
    #data = "/home/robotlab/Documents/datasets/dataset_new2_combined.hdf5" # OLD ONE HERE
    #data2 = "/home/robotlab/Documents/datasets/combined_dataset_last_episodes.hdf5" # OLD ONE HERE
    #data = "/home/robotlab/ws/RLRoverLab/datasets/dataset_old_camera_pos.hdf5"
    data = "/home/robotlab/Documents/datasets/dataset_new2.hdf5"
   # data2 = "/home/robotlab/Documents/datasets/dataset_new2.hdf5"
    #data = "/media/anton/T7 Shield/University/PHD/rover_simulation_datasets/dataset_new2.hdf5"

    # Define what data to use typically "observations" and "actions", but for t his example we train on depth aswell
    # HDF_DEFAULT_ORL_MAPPER = {
    #     "observations": "observations",
    #     "actions": "actions",
    #     "depth": "depth"
    # }
    HDF_DEFAULT_ORL_MAPPER = {
        "observations": "observations",
        "actions": "actions",
        "rewards": "rewards",
        "next_observation": "observations",
        "dones": "terminated",
        "depth": "depth",
        "rgb": "rgb"
    }

    # Define the dataset and validation dataset, we use the same dataset for both here
    dataset = HDF5DictDatasetRandom(data, min_idx=10000, total_samples=120000)
    dataset_val = HDF5DictDatasetRandom(
        data, min_idx=1, max_idx=100, total_samples=10000)

    # Define model configurations
    model_config = {
        "proprioception_channels": 3,
        "image_channels": 2,
        "action_dim": 2,
        "mlp_features": [512, 256, 128, 64],
        "image_input_dim": [224, 224],
        "image_encoder_features": [8, 16, 32, 64],
        "image_fc_features": [120, 60],
        "activation": "leaky_relu",
        "dropout_rate": 0,
        "use_batch_norm": False
    }
    
    iql_config = {
        "actions_lr": 1e-3,
        "value_lr": 3e-4,
        "critic_lr": 3e-4,
        "discount": 0.99,
        "tau": 0.005,
        "expectile": 0.8,
        "temperature": 0.1,
        "target_update_freq": 1,
    }

    # Define model with improved configurations
    actor = actor_gaussian_image(**model_config).to("cuda:0")
    critic = TwinQ_image(**model_config).to("cuda:0")
    value = v_image(**model_config).to("cuda:0")

    # Choose the algorithm to train with
    agent = IQL(actor_policy=actor,
                value_policy=value,
                critic_policy=critic,
                cfg=iql_config)

    #def env_loader(): return wrap_env(load_isaac_orbit_env(task_name="AAURoverEnvCamera-v0"), wrapper="isaac-orbit")

    # Define the trainer with improved configuration
    trainer_config = {
        "batch_size": 100,  # Reduced from 100 for more stable gradients
        "epochs": 10,      # Increased for better convergence
        "num_workers": 4,
        "shuffle": True,
        "early_stopping_patience": 10,
        "save_freq": 2,
        "validation_freq": 1,
        "log_freq": 50,
        "mixed_precision": False,
    }
    
    trainer = Trainer(cfg=trainer_config,
                      policy=agent,
                      dataset=dataset, 
                      val_dataset=dataset_val)

    # Start training
    trainer.train()

    return trainer


def eval(trainer: Trainer):
    # AppLauncher.add_app_launcher_args(parser)
    # args_cli, hydra_args = parser.parse_known_args()
    # sys.argv = [sys.argv[0]] + hydra_args
    # app_launcher = AppLauncher(args_cli)
    # simulation_app = app_launcher.app
    # from isaaclab_tasks.utils import parse_env_cfg  # noqa: F401, E402
    # import rover_envs  # noqa: F401
    # import rover_envs.envs.navigation.robots  # noqa: F401
    # env_cfg = parse_env_cfg(
    #     args_cli.task, device="cuda:0" if not args_cli.cpu else "cpu", num_envs=args_cli.num_envs
    # )
    # env = gym.make(args_cli.task, cfg=env_cfg)
    # env.reset()
    # #env = load_isaaclab_env(task_name="AAURoverEnvRGBDRaw-v0")
    # #env = wrap_env(env, wrapper="isaaclab")
    # trainer.evaluate(env, num_steps=10000)
    env = load_isaaclab_env(task_name="AAURoverEnvRGBDRaw-v0")
    #env = load_isaaclab_env(task_name="AAURoverEnvRGBDRawTemp-v0")
    #env = wrap_env(env)
    trainer.evaluate(env, num_steps=10000)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    trainer = train_iql()
    print(trainer)
    eval(trainer)
