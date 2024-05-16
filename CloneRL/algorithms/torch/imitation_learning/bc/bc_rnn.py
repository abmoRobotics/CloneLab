from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.optim.lr_scheduler import StepLR

import wandb
from CloneRL.algorithms.torch.imitation_learning.base import BaseAgent

BC_DEFAULT_CONFIG = {
}


class BehaviourCloningRNN(BaseAgent):

    def __init__(self, policy: nn.Module, cfg, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        super().__init__(cfg, policy, device=device)
        self.policy = policy
        self.loss_fn = nn.MSELoss(reduction='none')
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.scaler = torch.cuda.amp.GradScaler()

        self.initialize()

    def train(self,
              state: torch.Tensor,
              action: torch.Tensor,
              reward: torch.Tensor,
              next_state: torch.Tensor,
              done: torch.Tensor,
              weights: torch.Tensor,
              step: int,
              masks: torch.Tensor) -> float:
        target_actions = action.float()

        def compute_loss(target: torch.Tensor, actions: torch.Tensor, weights: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
            loss = (actions - target).pow(2)  # (batch_size, seq_len, action_dim)
            weighted_loss = loss * weights  # (batch_size, seq_len, action_dim)
            weighted_loss = weighted_loss.sum(dim=2)  # (batch_size, seq_len)
            weighted_loss = weighted_loss * masks  # (batch_size, seq_len)

            # Divide by the number of non-zero elements in the mask
            return weighted_loss.sum() / masks.sum()

        actions = self.policy(state)

        loss = compute_loss(target_actions, actions, weights, masks)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        wandb.log({"train/loss": loss.mean().item()}, step=step)
        return loss.item()

    def validate(self, state, action, reward, next_state, done, weights, step):

        with torch.no_grad():
            target = action.float()
            actions = self.policy(state)
            loss = self.loss_fn(actions, target) * weights
            loss = loss.mean()
        return loss.item()

    def evaluate(self, env, num_episodes=1000):
        pass

    def act(self, state, done, deterministic: bool = False):
        with torch.no_grad():
            action = self.policy.validate(state, done)
            return action
            # return self.policy(state)
