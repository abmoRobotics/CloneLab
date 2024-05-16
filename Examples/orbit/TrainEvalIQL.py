from typing import Dict

import rover_envs
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import TwinQ_image, actor_gaussian, actor_gaussian_image, v_image
from skrl.envs.loaders.torch import load_isaac_orbit_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils import set_seed

from CloneRL.algorithms.torch.offline_rl.iql import IQL
from CloneRL.dataloader.hdf import (HDF_DEFAULT_IL_MAPPER,
                                    HDF_DEFAULT_ORL_MAPPER, HDF5Dataset)
from CloneRL.dataloader.hdf.hdf_loader2 import (
    CloneLabDataset, OfflineReinforcementLearningDataset)
from CloneRL.trainers.torch.sequential2 import SequentialTrainer as Trainer

set_seed(12345)


def train_bc():

    # Define path to HDF5 file and mapper for training and validation
    # data = "/media/anton/T7 Shield/University/1. Master/Datasets/1. Simulation/dataset1/with_rgb_and_depth_2.hdf5"
    # data = "/media/anton/T7 Shield/University/1. Master/Datasets/1. Simulation/dataset4_180_320/with_rgb_and_depth_0.hdf5"
    data = "/media/anton/T7 Shield/University/1. Master/Datasets/1. Simulation/dataset3/with_rgb_and_depth_0.hdf5"
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
    dataset = OfflineReinforcementLearningDataset(data, HDF_DEFAULT_ORL_MAPPER, min_idx=0, max_idx=30_000)
    dataset_val = OfflineReinforcementLearningDataset(
        data, HDF_DEFAULT_ORL_MAPPER, min_idx=80_000, max_idx=85_000)

    # Define model
    actor = actor_gaussian_image().to("cuda:0")
    critic = TwinQ_image().to("cuda:0")
    value = v_image().to("cuda:0")

    # Choose the algorithm to train with
    agent = IQL(actor_policy=actor,
                value_policy=value,
                critic_policy=critic,
                cfg={})

    def env_loader(): return wrap_env(load_isaac_orbit_env(task_name="AAURoverEnvCamera-v0"), wrapper="isaac-orbit")

    # Define the trainer
    trainer = Trainer(cfg={"batch_size": 150}, policy=agent,
                      dataset=dataset, val_dataset=dataset_val)

    # Start training
    trainer.train(epoch=6)

    return trainer


def eval(trainer: Trainer):
    env = load_isaac_orbit_env(task_name="AAURoverEnvCamera-v0")
    env = wrap_env(env, wrapper="isaac-orbit")
    trainer.evaluate(env, num_steps=10000)


if __name__ == "__main__":
    trainer = train_bc()
    eval(trainer)
