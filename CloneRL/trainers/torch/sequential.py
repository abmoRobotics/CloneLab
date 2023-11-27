from typing import List, Tuple, Union, Optional, Dict, Any

import tqdm
import copy

import torch

from CloneRL.trainers.torch.base import BaseTrainer
from CloneRL.algorithms.base import BaseAgent

from torch.utils.data import DataLoader

import wandb

SEQUENTIAL_TRAINER_DEFAULT_CONFIG = {
    "epochs": 100000,
    "simulator": None,
}

class SequentialTrainer(BaseTrainer):
    def __init__(self,
                 policy: BaseAgent,
                 cfg,
                 env: Optional[Any] = None,
                 dataset: DataLoader = None) -> None:

        _cfg = copy.deepcopy(SEQUENTIAL_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(cfg, policy, dataset)

        self.policy.initialize()


    def train(self, epoch: int):

        for epoch in range(epoch):
            total_loss = 0
            for data in tqdm.tqdm(self.train_ds):
                loss = self.policy.train(data)
                total_loss += loss

            average_loss = total_loss / len(self.train_ds)
            wandb.log({"loss": average_loss})