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
from models import TwinQ_image, actor_gaussian, actor_gaussian_image, v_image
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

def setup_datasets(data_cfg: Dict):
    """Sets up the training and validation datasets."""
    print("Setting up datasets...")
    # Define what data to use, typically "observations" and "actions"
    # This is defined in the config, but we can leave this here for context
    HDF_DEFAULT_ORL_MAPPER = data_cfg["mapper"]

    dataset = HDF5DictDatasetRandom(data_cfg["path"], 
                                    min_idx=data_cfg["train_min_idx"], 
                                    total_samples=data_cfg["train_total_samples"])
    dataset_val = HDF5DictDatasetRandom(data_cfg["path"], 
                                        min_idx=data_cfg["val_min_idx"], 
                                        max_idx=data_cfg["val_max_idx"], 
                                        total_samples=data_cfg["val_total_samples"])
    print("Datasets ready.")
    return dataset, dataset_val

def setup_models_and_agent(model_cfg: Dict, agent_cfg: Dict):
    """Initializes the models and the IQL agent."""
    print("Initializing models and agent...")
    device = model_cfg.get("device", "cuda:0")
    actor = actor_gaussian_image(proprioception_channels=model_cfg["proprioception_channels"], 
                                 image_channels=model_cfg["image_channels"]).to(device)
    critic = TwinQ_image(proprioception_channels=model_cfg["proprioception_channels"], 
                         image_channels=model_cfg["image_channels"]).to(device)
    value = v_image(proprioception_channels=model_cfg["proprioception_channels"], 
                    image_channels=model_cfg["image_channels"]).to(device)

    agent = IQL(actor_policy=actor,
                value_policy=value,
                critic_policy=critic,
                cfg=agent_cfg)
    print("Models and agent ready.")
    return agent

def train_iql(cfg: Dict):
    """
    Trains the IQL agent based on the provided configuration.

    :param cfg: The configuration dictionary.
    :return: The trained trainer instance.
    """
    # Define the dataset and validation dataset
    dataset, dataset_val = setup_datasets(cfg["data"])

    # Define model and agent
    agent = setup_models_and_agent(cfg["model"], cfg.get("agent", {}))

    # Define the trainer
    trainer = Trainer(cfg=cfg["trainer"],
                      policy=agent,
                      dataset=dataset, 
                      val_dataset=dataset_val)

    # Start training
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    return trainer


def eval(trainer: Trainer, eval_cfg: Dict):
    """
    Evaluates the trained agent.

    :param trainer: The trainer instance with the trained policy.
    :param eval_cfg: The evaluation configuration dictionary.
    """
    print("Starting evaluation...")
    env = load_isaaclab_env(task_name=eval_cfg["task_name"])
    trainer.evaluate(env, num_steps=eval_cfg["num_steps"])
    print("Evaluation finished.")


def main():
    """Main function to run training and evaluation."""
    # Centralized configuration
    config = {
        "data": {
            "path": "/home/robotlab/Documents/datasets/dataset_new2.hdf5",
            "mapper": {
                "observations": "observations",
                "actions": "actions",
                "rewards": "rewards",
                "next_observation": "observations",
                "dones": "terminated",
                "depth": "depth",
                "rgb": "rgb"
            },
            "train_min_idx": 100,
            "train_total_samples": 120000,
            "val_min_idx": 1,
            "val_max_idx": 100,
            "val_total_samples": 10000,
        },
        "model": {
            "proprioception_channels": 3,
            "image_channels": 1,
            "device": "cuda:0",
        },
        "agent": {},
        "trainer": {
            "batch_size": 100,
            "epochs": 10,
            "num_workers": 4,
            "shuffle": True,
        },
        "eval": {
            "task_name": "AAURoverEnvRGBDRawTemp-v0",
            "num_steps": 1000000,
        }
    }
    # In a more advanced setup, you could load this config from a file
    # or override it with command-line arguments.

    trainer = train_iql(config)
    eval(trainer, config["eval"])


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
