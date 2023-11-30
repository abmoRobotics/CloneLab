from typing import List, Tuple, Union, Optional, Dict, Any

import tqdm
import copy

import torch

from CloneRL.trainers.torch.base import BaseTrainer
from CloneRL.algorithms.torch.imitation_learning.base import BaseAgent

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
                 dataset: DataLoader = None,
                 val_dataset: DataLoader = None) -> None:

        _cfg = copy.deepcopy(SEQUENTIAL_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(cfg, policy, dataset, val_dataset)

        self.policy.initialize()


    def train(self, epoch: int):
        best_val_loss = self.policy.validate(self.train_val_ds)

        

        for epoch in range(epoch):
            total_loss = 0
            for data in tqdm.tqdm(self.train_ds):
                loss = self.policy.train(data)
                total_loss += loss
                #wandb.log({"train loss": loss})
            average_loss = total_loss / len(self.train_ds)
            
            if epoch % 1 == 0:
                self.policy.save_model(f"model_{epoch}.pt")
            


            val_loss = self.policy.validate(self.train_val_ds)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.policy.save_model("best_model.pt")
                #wandb.log({"best train loss": best_val_loss})

            wandb.log({"train loss": average_loss, "val loss": val_loss})

    def evaluate(self, env, num_steps=1000):
        obs, info = env.reset()

        for timestep in tqdm.tqdm(range(num_steps)):
            with torch.no_grad():
                actions = self.policy.policy({"observations": torch.Tensor(obs),
                                              "extras": info})
                next_obs, rewards, terminated, truncated, info = env.step(actions)

                # if not self.headless:
                #     env.render()

            with torch.no_grad():
                if terminated.any() or truncated.any():
                    obs, info = env.reset()
                else:
                    obs = next_obs