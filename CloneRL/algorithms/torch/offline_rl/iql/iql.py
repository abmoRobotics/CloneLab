import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import wandb
from CloneRL.algorithms.torch.offline_rl.base import BaseAgent

BC_DEFAULT_CONFIG = {
}


class IQL(BaseAgent):

    def __init__(self,
                 actor_policy: nn.Module,
                 value_policy: nn.Module,
                 critic_policy: nn.Module,
                 cfg,
                 device="cuda:0" if torch.cuda.is_available() else "cpu",
                 actions_lr: float = 1e-3,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.8, # original 0.8
                 temperature: float = 0.1, # original 0.1
                 dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 opt_decay_schedule: Optional[str] = None):
        super().__init__(cfg, actor_policy, device=device)

        # Define the actor, critic and value networks
        self.actor = actor_policy.to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actions_lr)

        self.critic = critic_policy.to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.value = value_policy.to(self.device)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=value_lr)

        self.discount = discount
        self.tau = tau
        self.temperature = temperature

        self.total_steps = 0
        self.expectile = expectile

        self.loss_fn = nn.MSELoss()
        self.scaler = torch.amp.GradScaler()

        self.initialize()

    def expectile_loss(self, diff, expectile=0.8) -> torch.Tensor:
        w = torch.where(diff > 0, expectile, 1 - expectile)
        return torch.mean(w * diff**2)

    def select_action(self, states, deterministic=False):
        pass
        # state = torch.FloatTensor(states).to(self.device)
        # return self.actor.get_action(state, deterministic)

    def train(self, state, action, reward, next_state, done, weights, step, masks):
        def update_value_network(self: IQL, states, actions, logger=None):
            with torch.no_grad():
                q1, q2 = self.critic_target(states, actions)
                q = torch.min(q1, q2)

            value = self.value(states)
            value_loss: torch.Tensor = self.expectile_loss(q - value, self.expectile).mean()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            wandb.log({"train/value_loss": value_loss.mean().item()}, step=step)
            wandb.log({"train/value": value.mean().item()}, step=step)

        def update_q_network(self: IQL, states, actions, rewards, next_states, dones, logger=None):
            with torch.no_grad():
                next_value = self.value(next_states).squeeze(-1)
                target_q = rewards + self.discount * (1 - dones.float()) * next_value
                target_q = target_q.unsqueeze(-1)  # Ensure target_q has the same shape as q1 and q2
            q1, q2 = self.critic(states, actions)
            critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            wandb.log({"train/critic_loss": critic_loss.mean().item()}, step=step)
            wandb.log({"train/q1": q1.mean().item()}, step=step)
            wandb.log({"train/q2": q2.mean().item()}, step=step)

        def update_actor_network(self: IQL, states, actions, logger=None):
            with torch.no_grad():
                value = self.value(states)
                q1, q2 = self.critic(states, actions)
                q = torch.min(q1, q2)
                exp_a = torch.exp((q - value) * self.temperature)
                exp_a = torch.clamp(exp_a, max=100)

            dist: torch.distributions.Normal = self.actor(states)
            log_probs = -dist.log_prob(actions).sum(dim=-1, keepdim=True)  # Sum over action dimensions
            # actor_loss = -(exp_a.unsqueeze(-1) * log_probs).mean()
            actor_loss = torch.mean(exp_a * log_probs)
            # mu = self.actor(states)
            # actor_loss = (exp_a.unsqueeze(-1) * (mu - actions) ** 2).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            wandb.log({"train/actor_loss": actor_loss.mean().item()}, step=step)
            wandb.log({"train/advantage": (q - value).mean().item()}, step=step)
            return actor_loss

        def update_target_network(self: IQL, logger=None):
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Sample a batch of data
        # state, action, reward, next_state, done = data['observations'], data['actions'], data['rewards'], data['next_observation'], data['dones']
        # state = {"observations": state,
        #          "depth": data['depth']}

        # Update the value network
        # state["image"] = state["image"].unsqueeze(1)

        update_value_network(self, state, action)

        # Update the actor network
        actor_loss = update_actor_network(self, state, action)

        # Update the Q network
        update_q_network(self, state, action, reward, next_state, done)

        # Update the target network
        update_target_network(self)

        return actor_loss.item()

    def validate(self, state, action, reward, next_state, done, weights, step, masks) -> torch.Tensor:
        average_loss = 0
        with torch.no_grad():
            q1, q2 = self.critic(state, action)
            q = torch.min(q1, q2)
            exp_adv = torch.exp((q - self.value(state)) * self.temperature)
            exp_adv = torch.clamp(exp_adv, max=100)

            dist: torch.distributions.Normal = self.actor(state)
            log_probs = dist.log_prob(action)
            actor_loss = -(exp_adv.unsqueeze(-1) * log_probs).mean().item()

        return actor_loss

    def evaluate(self, env, num_episodes=1000):
        pass

    def act(self, state, done):
        distribution = self.actor(state)
        return distribution.sample()
        # return self.actor(state)

    def save_model(self, name):
        import os
        base_path = f"runs/{wandb.run.project}/{wandb.run.id}/checkpoints/"
        paths = ["actor/", "critic/", "value/", "optimizer/",
                 "optimizer/actor/", "optimizer/critic/", "optimizer/value/", "critic_target/"]

        # if not os.path.exists(path):
        #     os.makedirs(path)

        for p in paths:
            if not os.path.exists(base_path + p):
                os.makedirs(base_path + p)

        torch.save(self.actor.state_dict(), base_path + "actor/" + name)
        torch.save(self.critic.state_dict(), base_path + "critic/" + name)
        torch.save(self.value.state_dict(), base_path + "value/" + name)

        torch.save(self.actor_optimizer.state_dict(), base_path + "optimizer/actor/" + name)
        torch.save(self.critic_optimizer.state_dict(), base_path + "optimizer/critic/" + name)
        torch.save(self.value_optimizer.state_dict(), base_path + "optimizer/value/" + name)

        torch.save(self.critic_target.state_dict(), base_path + "critic_target/" + name)

    def load_model(self, path, name="best_model.pt"):
        
        self.actor.load_state_dict(torch.load(path + "actor/" + name))
        self.critic.load_state_dict(torch.load(path + "critic/" + name))
        self.value.load_state_dict(torch.load(path + "value/" + name))

        self.actor_optimizer.load_state_dict(torch.load(path + "optimizer/actor/" + name))
        self.critic_optimizer.load_state_dict(torch.load(path + "optimizer/critic/" + name))
        self.value_optimizer.load_state_dict(torch.load(path + "optimizer/value/" + name))

        self.critic_target.load_state_dict(torch.load(path + "critic_target/" + name))
