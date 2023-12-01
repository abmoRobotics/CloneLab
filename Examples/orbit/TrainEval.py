from typing import Dict

import torch
import torch.nn as nn
from skrl.envs.torch.loaders import load_isaac_orbit_env
from skrl.envs.torch.wrappers import wrap_env
from skrl.utils import set_seed

set_seed(42)
import custom_envs
import torch.nn.functional as F

from CloneRL.algorithms.torch.imitation_learning.bc import \
    BehaviourCloning as BC
from CloneRL.dataloader.hdf import (HDF_DEFAULT_IL_MAPPER,
                                    HDF_DEFAULT_ORL_MAPPER, HDF5Dataset)
from CloneRL.trainers.torch.sequential import SequentialTrainer as Trainer

from .models import DepthPolicy


def train_bc():

    # Define path to HDF5 file and mapper for training and validation
    data = "/media/anton/T7 Shield/University/1. Master/Datasets/1. Simulation/with_rgb_new_0.hdf5"

    # Define what data to use typically "observations" and "actions", but for this example we train on depth aswell
    IL_MAPPER = {
    "observations": "observations",
    "actions": "actions",
    "depth": "depth"
    }

    # Define the dataset and validation dataset, we use the same dataset for both here
    dataset = HDF5Dataset(data, IL_MAPPER, min_idx=0, max_idx=100_000)
    dataset_val = HDF5Dataset(data, IL_MAPPER, min_idx=100_000, max_idx=120_000)

    # Define model
    policy = DepthPolicy().to("cuda:0")

    # Choose the algorithm to train with
    agent = BC(policy = policy, cfg={})

    # Define the trainer
    trainer = Trainer(cfg={"batch_size": 1000}, policy=agent, dataset=dataset, val_dataset=dataset_val)

    # Start training
    trainer.train(epoch=30)

    return trainer

def eval(trainer: Trainer):
    env = load_isaac_orbit_env(task_name="RoverCamera-v0")
    env = wrap_env(env)
    num_envs = env.num_envs

    trainer.evaluate(env, num_steps=10000)





if __name__ == "__main__":
    trainer = train_bc()
    eval(trainer)
