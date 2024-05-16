import copy
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import tqdm
from skrl.envs.loaders.torch import load_isaac_orbit_env
from skrl.envs.wrappers.torch import wrap_env
from torch.utils.data import DataLoader

import wandb
from CloneRL.algorithms.torch.imitation_learning.base import BaseAgent
from CloneRL.trainers.torch.base import BaseTrainer

SEQUENTIAL_TRAINER_DEFAULT_CONFIG = {
    "batch_size": 64,
    "num_workers": 4,
    "prefetch_factor": 2,
    "shuffle": False,
    "epochs": 100,
    "simulator": None,
}


class SequentialTrainer(BaseTrainer):
    def __init__(self,
                 policy: BaseAgent,
                 cfg,
                 env: Optional[Any] = None,
                 env_loader: Optional[Any] = None,
                 dataset: DataLoader = None,
                 val_dataset: DataLoader = None) -> None:

        _cfg = copy.deepcopy(SEQUENTIAL_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(_cfg, policy, dataset, val_dataset)

        self.policy.initialize()

        # Start simulation asynchronusly
        if env_loader is not None:
            # self.env = env
            self.simulator_thread = threading.Thread(target=self.simulation_validation, args=(env_loader,))
            self.simulator_thread.start()
            print("Simulation started")

    def simulation_validation(self, env_loader):
        """Run the environment simulation asynchronously"""
        # Start validation environment

        if env_loader is not None:
            self.env = env_loader()

        obs, info = self.env.reset()
        first_iter = True
        num_steps = 100000  # Define the number of steps for the simulation

        for timestep in range(num_steps):
            with torch.no_grad():
                if first_iter:
                    first_iter = False
                    actions = torch.zeros((obs.shape[0], 2))
                else:
                    depth = info['depth'].unsqueeze(1)
                    depth = torch.nan_to_num(depth, nan=10.0)
                    depth = torch.clamp(depth, min=0.0, max=10.0)
                    state = {'proprioceptive': obs[:, :4], 'image': depth}
                    actions = self.policy.act(state)

                next_obs, rewards, terminated, truncated, next_info = self.env.step(actions)
                obs, info = next_obs, next_info

    def train(self):
        """
        Trains and validates the model for a specified number of epochs, saving the model
        when the validation loss improves.

        Args:
        num_epochs (int): The number of epochs to train and validate the model.
        """
        best_val_loss = float('inf')  # Initialize the best validation loss to infinity

        for current_epoch in range(self.cfg["epochs"]):
            # Training phase
            train_loss = self._process_epoch(self.train_ds, current_epoch, train=True)
            # Validation phase
            val_loss = self._process_epoch(self.train_val_ds, current_epoch, train=False)
            wandb.log({"epoch/validation_loss": val_loss}, step=(current_epoch+1)*len(self.train_ds))
            wandb.log({"epoch/train_loss": train_loss}, step=(current_epoch+1)*len(self.train_ds))
            wandb.log({"epoch": current_epoch}, step=(current_epoch+1)*len(self.train_ds))
            # Save the model if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.policy.save_model(f"best_model_{current_epoch}.pt")
                print(f"Model saved with improved validation loss: {best_val_loss}")

            # wandb.log({"train/train_loss": train_loss, "val/val_loss": val_loss},
            #           step=current_epoch * len(self.train_ds))

    def _process_epoch(self, dataset, epoch: int, train: bool):
        """
        Processes an epoch of training or validation, computing the loss for each batch.

        Args:
        dataset (Iterable): The dataset to process.
        epoch (int): The current epoch number.
        train (bool): Flag indicating if the model should be trained or validated.

        Returns:
        float: Average loss for the epoch.
        """
        total_loss = 0.0
        phase = 'Train' if train else 'Val'
        pbar_description = f"{phase} Epoch {epoch}"
        pbar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]"

        with tqdm.tqdm(dataset, desc=pbar_description, bar_format=pbar_format) as pbar:
            for idx, (obs, actions, rewards, next_obs, dones, weights, masks) in enumerate(pbar):
                # print(actions.shape)
                step = epoch * len(dataset) + idx
                if train:
                    loss = self.policy.train(obs, actions, rewards, next_obs, dones, weights, step, masks)

                else:
                    with torch.no_grad():
                        loss = self.policy.validate(obs, actions, rewards, next_obs, dones, weights, step)
                total_loss += loss
                if (idx + 1) == len(dataset):
                    average_loss = total_loss / len(dataset)
                    pbar.set_postfix({f'Avg {phase.lower()} loss': average_loss})
                else:
                    pbar.set_postfix({"Batch loss": loss})
            # wandb.log({"epoch/train_loss": loss}, step=step)
            average_loss = total_loss / len(dataset)
            return average_loss

    def evaluate(self, env, num_steps=1000):
        obs, info = env.reset()
        first_iter = True
        terminated = torch.zeros((obs.shape[0], 1), dtype=torch.bool)
        for timestep in tqdm.tqdm(range(num_steps)):
            with torch.no_grad():
                if first_iter:
                    first_iter = False
                    actions = torch.zeros((obs.shape[0], 2))
                else:
                    rgb = info['rgb']
                    rgb = torch.nan_to_num(rgb, nan=256.0)
                    rgb = torch.clamp(rgb, min=0.0, max=256.0)
                    rgb = rgb.permute(0, 3, 1, 2)
                    depth = info['depth'].unsqueeze(1)
                    depth = torch.nan_to_num(depth, nan=256.0)
                    depth = torch.clamp(depth, min=0.0, max=256.0)
                    state = {'proprioceptive': obs[:, :4], 'image': torch.cat((depth, rgb), dim=1)}
                    actions = self.policy.act(state, terminated)

                next_obs, rewards, terminated, truncated, next_info = env.step(actions)

                obs, info = next_obs, next_info
