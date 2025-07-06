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


class BehaviourCloning(BaseAgent):

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

        def compute_loss(target: torch.Tensor, actions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # loss = self.loss_fn(actions, target)
            loss = (actions - target).pow(2)
            weighted_loss = loss * weights
            return weighted_loss.mean()

        actions = self.policy(state)
        # actions = torch.zeros_like(action)
        # prev_action = torch.zeros_like(action[0])
        # for i in range(action.shape[0]):
        #     new_proprioceptive = state["proprioceptive"][i].clone()
        #     new_proprioceptive[0:2] = prev_action
        #     temp_state = {
        #         "proprioceptive": new_proprioceptive.unsqueeze(0),
        #         "image": state["image"][i].unsqueeze(0)
        #     }
        #     actions[i] = self.policy(temp_state).squeeze(0)
        #     prev_action = actions[i].detach()
        loss = compute_loss(target_actions, actions, weights)
        # tqdm.write(f'loss: {loss.mean().item()}')
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
            return self.policy(state)
