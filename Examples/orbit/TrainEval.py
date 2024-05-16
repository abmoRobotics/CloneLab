from typing import Dict


import rover_envs
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DepthPolicy
from skrl.envs.loaders.torch import load_isaac_orbit_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils import set_seed
from CloneRL.algorithms.torch.imitation_learning.bc import \
    BehaviourCloning as BC
from CloneRL.dataloader.hdf import (HDF_DEFAULT_IL_MAPPER,
                                    HDF_DEFAULT_ORL_MAPPER, HDF5Dataset)
from CloneRL.trainers.torch.sequential import SequentialTrainer as Trainer

set_seed(42)


def train_bc():

    # Define path to HDF5 file and mapper for training and validation
    # data = "/media/anton/T7 Shield/University/1. Master/Datasets/1. Simulation/dataset1/with_rgb_and_depth_0.hdf5"
    data = "/media/anton/T7 Shield/University/1. Master/Datasets/1. Simulation/dataset4_180_320/with_rgb_and_depth_0.hdf5"
    # Define what data to use typically "observations" and "actions", but for this example we train on depth aswell
    IL_MAPPER = {
        "observations": "observations",
        "actions": "actions",
        "depth": "depth"
    }

    # Define the dataset and validation dataset, we use the same dataset for both here
    dataset = HDF5Dataset(data, IL_MAPPER, min_idx=0, max_idx=10_000)
    dataset_val = HDF5Dataset(
        data, IL_MAPPER, min_idx=10_000, max_idx=15_000)

    # Define model
    policy = DepthPolicy().to("cuda:0")

    # Choose the algorithm to train with
    agent = BC(policy=policy, cfg={})

    # Define the trainer
    trainer = Trainer(cfg={"batch_size": 500}, policy=agent,
                      dataset=dataset, val_dataset=dataset_val)

    # Start training
    trainer.train(epoch=10)

    return trainer


def eval(trainer: Trainer):
    env = load_isaac_orbit_env(task_name="AAURoverEnvCamera-v0")
    env = wrap_env(env, wrapper="isaac-orbit")
    trainer.evaluate(env, num_steps=10000)

if __name__ == "__main__":
    trainer = train_bc()
    eval(trainer)
