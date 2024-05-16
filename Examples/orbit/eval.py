from typing import Dict

import rover_envs
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import TwinQ_image, actor_gaussian_image, v_image
from skrl.envs.loaders.torch import load_isaac_orbit_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils import set_seed

from CloneRL.algorithms.torch.offline_rl.iql import IQL
from CloneRL.dataloader.hdf import (HDF_DEFAULT_IL_MAPPER,
                                    HDF_DEFAULT_ORL_MAPPER, HDF5Dataset)
from CloneRL.dataloader.hdf.hdf_loader import \
    OfflineReinforcementLearningDataset
from CloneRL.trainers.torch.sequential import SequentialTrainer as Trainer

set_seed(43)


def train_bc():

    # Define path to HDF5 file and mapper for training and validation
    data = "/media/anton/T7 Shield/University/1. Master/Datasets/1. Simulation/dataset1/with_rgb_and_depth_2.hdf5"

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
        "depth": "depth"
    }

    # Define the dataset and validation dataset, we use the same dataset for both here
    dataset = OfflineReinforcementLearningDataset(data, HDF_DEFAULT_ORL_MAPPER, min_idx=0, max_idx=80_000)
    dataset_val = OfflineReinforcementLearningDataset(
        data, HDF_DEFAULT_ORL_MAPPER, min_idx=80_000, max_idx=99_000)

    # Define model
    actor = actor_gaussian_image().to("cuda:0")
    critic = TwinQ_image().to("cuda:0")
    value = v_image().to("cuda:0")

    # Choose the algorithm to train with
    agent = IQL(actor_policy=actor,
                value_policy=value,
                critic_policy=critic,
                cfg={})

    # Define the trainer
    trainer = Trainer(cfg={"batch_size": 256}, policy=agent,
                      dataset=dataset, val_dataset=dataset_val)

    # Start training
    trainer.train(epoch=10)

    return trainer


def eval(trainer: Trainer):

    env = load_isaac_orbit_env(task_name="AAURoverEnvCamera-v0")
    env = wrap_env(env, wrapper="isaac-orbit")
    trainer.evaluate(env, num_steps=10000)


def get_trainer():
    actor = actor_gaussian_image().to("cuda:0")
    critic = TwinQ_image().to("cuda:0")
    value = v_image().to("cuda:0")

    state_dict = torch.load('/home/anton/1._University/0._Master_Project/Workspace/CloneLab/model_8.pt')
    # actor.load_state_dict(state_dict)

    agent = IQL(actor_policy=actor,
                value_policy=value,
                critic_policy=critic,
                cfg={})

    trainer = Trainer(cfg={'batch_size': 256}, policy=agent)

    return trainer


if __name__ == "__main__":
    # trainer = train_bc()
    trainer = get_trainer()
    eval(trainer)
