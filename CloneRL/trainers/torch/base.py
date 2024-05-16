
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from CloneRL.algorithms.torch.imitation_learning.base import BaseAgent


def episodic_collate_fn(batch):
    """ Collate function for episodic data handling variable sequence lengths.

    Args:
        batch: list of batches each of shape (1, num_steps, *obs_shape).

    Returns:
        obs: Dictionary of torch.Tensor with keys corresponding to observation features.
        actions: torch.Tensor of shape (batch_size, max_seq_length, *action_shape)
        rewards: torch.Tensor of shape (batch_size, max_seq_length, 1)
        next_obs: Dictionary of torch.Tensor with keys corresponding to observation features.
        dones: torch.Tensor of shape (batch_size, max_seq_length, 1)
        weights: torch.Tensor of shape (batch_size, max_seq_length, 1)
    """
    # Assuming obs and next_obs are dictionaries containing tensors.
    lengths = [b[1].size(0) for b in batch]
    masks = pad_sequence([torch.ones(l, dtype=torch.bool) for l in lengths], batch_first=True).to(batch[0][1].device)
    # print(f'lengths: {lengths}')
    # exit()
    max_len = max(b[1].size(1) for b in batch)  # Get the maximum sequence length

    # Stack or pad each component appropriately
    obs = {k: pad_sequence([b[0][k].squeeze(0) for b in batch], batch_first=True) for k in batch[0][0].keys()}
    actions = pad_sequence([b[1].squeeze(0) for b in batch], batch_first=True)
    rewards = pad_sequence([b[2].squeeze(0) for b in batch], batch_first=True)
    next_obs = {k: pad_sequence([b[3][k].squeeze(0) for b in batch], batch_first=True) for k in batch[0][0].keys()}
    dones = pad_sequence([b[4].squeeze(0) for b in batch], batch_first=True)
    weights = pad_sequence([b[5].squeeze(0) for b in batch], batch_first=True)

    return obs, actions, rewards, next_obs, dones, weights, masks


class BaseTrainer():
    def __init__(self, cfg, policy, dataset, val_dataset):
        self.cfg = cfg
        self.policy: BaseAgent = policy
        # self.dataset = dataset
        if dataset.episodic:
            # Print warning
            print("NOTE: Episodic dataset detected, batch size corresponds to number of episodes")
            # assert self.cfg["batch_size"] == 1, "Episodic dataset only supports batch size of 1"
            self.train_ds = DataLoader(dataset,
                                       batch_size=self.cfg["batch_size"],
                                       shuffle=False,
                                       num_workers=self.cfg["num_workers"],
                                       prefetch_factor=self.cfg["prefetch_factor"],
                                       collate_fn=episodic_collate_fn)
            self.train_val_ds = DataLoader(val_dataset,
                                           batch_size=self.cfg["batch_size"],
                                           shuffle=False,
                                           num_workers=self.cfg["num_workers"],
                                           prefetch_factor=self.cfg["prefetch_factor"],
                                           collate_fn=episodic_collate_fn)
        else:
            self.train_ds = DataLoader(dataset,
                                       batch_size=self.cfg["batch_size"],
                                       shuffle=self.cfg["shuffle"],
                                       num_workers=self.cfg["num_workers"],
                                       prefetch_factor=self.cfg["prefetch_factor"])
            self.train_val_ds = DataLoader(val_dataset,
                                           batch_size=self.cfg["batch_size"],
                                           shuffle=self.cfg["shuffle"],
                                           num_workers=self.cfg["num_workers"],
                                           prefetch_factor=self.cfg["prefetch_factor"])

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
