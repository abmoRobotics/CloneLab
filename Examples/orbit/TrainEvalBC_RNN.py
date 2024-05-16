from typing import Dict

import rover_envs
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import GRUActor, actor_deterministic, actor_deterministic_image
from skrl.envs.loaders.torch import load_isaac_orbit_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils import set_seed

from CloneRL.algorithms.torch.imitation_learning.bc import \
    BehaviourCloning as BC
from CloneRL.algorithms.torch.imitation_learning.bc import \
    BehaviourCloningRNN as BC_RNN
from CloneRL.dataloader.hdf import (HDF_DEFAULT_IL_MAPPER,
                                    HDF_DEFAULT_ORL_MAPPER, HDF5Dataset)
from CloneRL.dataloader.hdf.hdf_loader import (
    CloneLabDataset, OfflineReinforcementLearningDataset)
from CloneRL.trainers.torch.sequential import SequentialTrainer as Trainer

# set_seed(12345)


def train_bc():

    # Define path to HDF5 file and mapper for training and validation
    # data = "/media/anton/T7 Shield/University/1. Master/Datasets/1. Simulation/dataset1/with_rgb_and_depth_0.hdf5"
    # data = "/media/anton/T7 Shield/University/1. Master/Datasets/1. Simulation/dataset4_180_320/with_rgb_and_depth_0.hdf5"
    data = "/media/anton/T7 Shield/University/1. Master/Datasets/1. Simulation/dataset3/with_rgb_and_depth_0.hdf5"
    # Define what data to use typically "observations" and "actions", but for this example we train on depth aswell
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
    dataset = CloneLabDataset(data, HDF_DEFAULT_ORL_MAPPER, min_idx=0, max_idx=50_000, episodic=True)
    dataset_val = CloneLabDataset(
        data, HDF_DEFAULT_ORL_MAPPER, min_idx=60_000, max_idx=68_000, episodic=True)

    # Define model
    policy = GRUActor().to("cuda:0")

    # Choose the algorithm to train with
    agent = BC_RNN(policy=policy, cfg={})

    # Define the trainer
    def env_loader(): return wrap_env(load_isaac_orbit_env(task_name="AAURoverEnvCamera-v0"), wrapper="isaac-orbit")
    # env_loader = None

    cfg = {"batch_size": 2, "epochs": 10, "num_workers": 2, "prefetch_factor": 2}

    # Define the trainer
    trainer = Trainer(cfg=cfg, policy=agent,  # env_loader=env_loader,
                      dataset=dataset, val_dataset=dataset_val)

    # Start training
    trainer.train()

    return trainer


def eval(trainer: Trainer):
    env = load_isaac_orbit_env(task_name="AAURoverEnvCamera-v0")
    env = wrap_env(env, wrapper="isaac-orbit")
    trainer.evaluate(env, num_steps=10000)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    trainer = train_bc()
    eval(trainer)
