"""IQL (Implicit Q-Learning) for Recurrent Models.

This module provides an IQL implementation designed for recurrent neural networks
(GRU, LSTM). It handles hidden state management and sequential data processing
for offline reinforcement learning with temporal dependencies.
"""

import copy
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from CloneRL.algorithms.torch.offline_rl.base import BaseAgent


IQL_RECURRENT_DEFAULT_CONFIG = {
    "actions_lr": 1e-3,
    "value_lr": 3e-4,
    "critic_lr": 3e-4,
    "discount": 0.99,
    "tau": 0.005,
    "expectile": 0.8,
    "temperature": 0.1,
    "target_update_freq": 1,
    "grad_clip": 1.0,
    "reset_hidden_on_done": True,
}


class IQLRecurrent(BaseAgent):
    """IQL algorithm for recurrent (GRU/LSTM) models.
    
    This implementation handles:
    - Sequential data with shape (batch, seq_len, ...)
    - Hidden state management across sequences
    - Proper gradient flow through time
    - Episode boundary handling
    
    Args:
        actor_policy: GRU-based actor network.
        value_policy: GRU-based value network.
        critic_policy: GRU-based twin Q network.
        cfg: Configuration dictionary.
        device: Device to use for computation.
        actions_lr: Learning rate for actor.
        value_lr: Learning rate for value network.
        critic_lr: Learning rate for critic.
        discount: Discount factor (gamma).
        tau: Soft update coefficient for target network.
        expectile: Expectile for value function regression.
        temperature: Temperature for advantage weighting.
        target_update_freq: Frequency of target network updates.
        grad_clip: Gradient clipping value (None to disable).
        reset_hidden_on_done: Whether to reset hidden states on episode end.
    """

    def __init__(
        self,
        actor_policy: nn.Module,
        value_policy: nn.Module,
        critic_policy: nn.Module,
        cfg: Dict,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        actions_lr: float = 1e-3,
        value_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.8,
        temperature: float = 0.1,
        target_update_freq: int = 1,
        grad_clip: Optional[float] = 1.0,
        reset_hidden_on_done: bool = True,
    ):
        super().__init__(cfg, actor_policy, device=device)

        # Define the actor, critic and value networks
        self.actor = actor_policy.to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actions_lr)

        self.critic = critic_policy.to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        # Freeze target network
        for param in self.critic_target.parameters():
            param.requires_grad = False
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.value = value_policy.to(self.device)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=value_lr)

        # Algorithm parameters
        self.discount = discount
        self.tau = tau
        self.temperature = temperature
        self.expectile = expectile
        self.grad_clip = grad_clip
        self.target_update_freq = target_update_freq
        self.reset_hidden_on_done = reset_hidden_on_done
        
        # Training state
        self.total_steps = 0
        self.update_count = 0

        # Hidden states for actor, critic, value (used during evaluation)
        self.actor_hidden = None
        self.critic_hidden = None
        self.value_hidden = None

        self.initialize()

    def expectile_loss(self, diff: torch.Tensor, expectile: float = 0.8) -> torch.Tensor:
        """Compute expectile loss.
        
        Args:
            diff: Difference tensor (target - prediction).
            expectile: Expectile value (0.5 = MSE, >0.5 = upper expectile).
            
        Returns:
            Expectile loss value.
        """
        w = torch.where(diff > 0, expectile, 1 - expectile)
        return torch.mean(w * diff ** 2)

    def _apply_mask(self, tensor: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Apply mask to tensor for valid timesteps only.
        
        Args:
            tensor: Input tensor of shape (batch, seq_len, ...).
            masks: Mask tensor of shape (batch, seq_len, 1).
            
        Returns:
            Masked tensor (values at invalid timesteps are zeroed).
        """
        return tensor * masks

    def _masked_mean(self, tensor: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Compute mean over valid (masked) timesteps only.
        
        Args:
            tensor: Input tensor.
            masks: Mask tensor.
            
        Returns:
            Mean over valid timesteps.
        """
        masked = self._apply_mask(tensor, masks)
        return masked.sum() / (masks.sum() + 1e-8)

    def train(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: Dict[str, torch.Tensor],
        done: torch.Tensor,
        weights: torch.Tensor,
        step: int,
        masks: torch.Tensor,
        hidden_reset: Optional[torch.Tensor] = None,
    ) -> float:
        """Training step for recurrent IQL.
        
        Args:
            state: Dictionary of observation tensors (batch, seq_len, ...).
            action: Action tensor (batch, seq_len, action_dim).
            reward: Reward tensor (batch, seq_len, 1).
            next_state: Dictionary of next observation tensors.
            done: Done flags (batch, seq_len, 1).
            weights: Sample weights (batch, seq_len, 1).
            step: Current training step.
            masks: Valid timestep masks (batch, seq_len, 1).
            hidden_reset: Optional mask indicating when to reset hidden states.
            
        Returns:
            Actor loss value.
        """
        self.update_count += 1
        
        # Ensure proper shapes
        if reward.dim() == 2:
            reward = reward.unsqueeze(-1)
        if done.dim() == 2:
            done = done.unsqueeze(-1)
        if masks.dim() == 2:
            masks = masks.unsqueeze(-1)

        # Update networks
        value_loss = self._update_value_network(state, action, masks, step)
        actor_loss = self._update_actor_network(state, action, masks, step)
        critic_loss = self._update_critic_network(state, action, reward, next_state, done, masks, step)

        # Update target network
        if self.update_count % self.target_update_freq == 0:
            self._update_target_network()

        return actor_loss

    def _update_value_network(
        self,
        states: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        masks: torch.Tensor,
        step: int,
    ) -> float:
        """Update value network with expectile regression.
        
        Args:
            states: State dictionary.
            actions: Actions tensor.
            masks: Valid timestep masks.
            step: Current step for logging.
            
        Returns:
            Value loss.
        """
        with torch.no_grad():
            # Get Q-values from target critic
            if hasattr(self.critic_target, 'forward'):
                q1, q2, _, _ = self.critic_target(states, actions)
            else:
                q1, q2 = self.critic_target(states, actions)
            q = torch.min(q1, q2)

        # Get value predictions
        if hasattr(self.value, 'forward'):
            value, _ = self.value(states)
        else:
            value = self.value(states)

        # Compute expectile loss with masking
        diff = q - value
        masked_diff = self._apply_mask(diff, masks)
        
        w = torch.where(masked_diff > 0, self.expectile, 1 - self.expectile)
        value_loss = (w * masked_diff ** 2).sum() / (masks.sum() + 1e-8)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.grad_clip)
        self.value_optimizer.step()
        
        # Logging
        wandb.log({
            "train/value_loss": value_loss.item(),
            "train/value_mean": self._masked_mean(value, masks).item(),
            "train/q_target_mean": self._masked_mean(q, masks).item(),
        }, step=step)
        
        return value_loss.item()

    def _update_critic_network(
        self,
        states: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: Dict[str, torch.Tensor],
        dones: torch.Tensor,
        masks: torch.Tensor,
        step: int,
    ) -> float:
        """Update Q-networks with temporal difference learning.
        
        Args:
            states: State dictionary.
            actions: Actions tensor.
            rewards: Rewards tensor.
            next_states: Next state dictionary.
            dones: Done flags.
            masks: Valid timestep masks.
            step: Current step for logging.
            
        Returns:
            Critic loss.
        """
        with torch.no_grad():
            # Get next value
            if hasattr(self.value, 'forward'):
                next_value, _ = self.value(next_states)
            else:
                next_value = self.value(next_states)
            
            # Compute TD target
            target_q = rewards + self.discount * (1 - dones.float()) * next_value

        # Get Q-value predictions
        if hasattr(self.critic, 'forward'):
            q1, q2, _, _ = self.critic(states, actions)
        else:
            q1, q2 = self.critic(states, actions)

        # Compute critic loss with masking
        critic_loss = self._masked_mean((q1 - target_q) ** 2, masks) + \
                      self._masked_mean((q2 - target_q) ** 2, masks)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # Logging
        wandb.log({
            "train/critic_loss": critic_loss.item(),
            "train/q1_mean": self._masked_mean(q1, masks).item(),
            "train/q2_mean": self._masked_mean(q2, masks).item(),
            "train/target_q_mean": self._masked_mean(target_q, masks).item(),
        }, step=step)
        
        return critic_loss.item()

    def _update_actor_network(
        self,
        states: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        masks: torch.Tensor,
        step: int,
    ) -> float:
        """Update actor network with AWR-style weighting.
        
        Args:
            states: State dictionary.
            actions: Actions tensor.
            masks: Valid timestep masks.
            step: Current step for logging.
            
        Returns:
            Actor loss.
        """
        with torch.no_grad():
            # Get value and Q-values
            if hasattr(self.value, 'forward'):
                value, _ = self.value(states)
            else:
                value = self.value(states)
                
            if hasattr(self.critic, 'forward'):
                q1, q2, _, _ = self.critic(states, actions)
            else:
                q1, q2 = self.critic(states, actions)
            q = torch.min(q1, q2)
            
            # Compute advantage and weights
            advantage = q - value
            exp_weights = torch.exp(advantage * self.temperature)
            exp_weights = torch.clamp(exp_weights, max=100.0)

        # Get log probabilities from actor
        if hasattr(self.actor, 'forward'):
            dist, _ = self.actor(states)
        else:
            dist = self.actor(states)
            
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        
        # AWR loss: maximize log probability weighted by advantage
        actor_loss = -self._masked_mean(exp_weights * log_probs, masks)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        # Logging
        wandb.log({
            "train/actor_loss": actor_loss.item(),
            "train/advantage_mean": self._masked_mean(advantage, masks).item(),
            "train/advantage_std": advantage[masks.bool().expand_as(advantage)].std().item() if masks.sum() > 0 else 0,
            "train/exp_weights_mean": self._masked_mean(exp_weights, masks).item(),
            "train/log_probs_mean": self._masked_mean(log_probs, masks).item(),
        }, step=step)
        
        return actor_loss.item()

    def _update_target_network(self):
        """Soft update of target network."""
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def validate(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: Dict[str, torch.Tensor],
        done: torch.Tensor,
        weights: torch.Tensor,
        step: int,
        masks: torch.Tensor,
        hidden_reset: Optional[torch.Tensor] = None,
    ) -> float:
        """Validation step for recurrent IQL.
        
        Args:
            state: Dictionary of observation tensors.
            action: Action tensor.
            reward: Reward tensor.
            next_state: Dictionary of next observation tensors.
            done: Done flags.
            weights: Sample weights.
            step: Current step.
            masks: Valid timestep masks.
            hidden_reset: Optional hidden state reset mask.
            
        Returns:
            Validation loss (actor loss).
        """
        # Ensure proper shapes
        if masks.dim() == 2:
            masks = masks.unsqueeze(-1)
            
        with torch.no_grad():
            # Get value and Q-values
            if hasattr(self.value, 'forward'):
                value, _ = self.value(state)
            else:
                value = self.value(state)
                
            if hasattr(self.critic, 'forward'):
                q1, q2, _, _ = self.critic(state, action)
            else:
                q1, q2 = self.critic(state, action)
            q = torch.min(q1, q2)
            
            # Compute advantage and weights
            advantage = q - value
            exp_adv = torch.exp(advantage * self.temperature)
            exp_adv = torch.clamp(exp_adv, max=100)

            # Get log probabilities
            if hasattr(self.actor, 'forward'):
                dist, _ = self.actor(state)
            else:
                dist = self.actor(state)
                
            log_probs = dist.log_prob(action).sum(dim=-1, keepdim=True)
            actor_loss = -self._masked_mean(exp_adv * log_probs, masks).item()

        return actor_loss

    def act(
        self, 
        state: Dict[str, torch.Tensor], 
        done: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action for deployment with hidden state management.
        
        Args:
            state: State dictionary.
            done: Done flags for hidden state reset.
            deterministic: Whether to use deterministic actions.
            
        Returns:
            Action tensor.
        """
        with torch.no_grad():
            # Use actor's validate method if available (handles hidden state)
            if hasattr(self.actor, 'validate') and done is not None:
                return self.actor.validate(state, done)
            elif hasattr(self.actor, 'get_action'):
                return self.actor.get_action(state, deterministic=deterministic)
            else:
                if hasattr(self.actor, 'forward'):
                    dist, self.actor_hidden = self.actor(state, self.actor_hidden)
                else:
                    dist = self.actor(state)
                    
                if deterministic:
                    action = dist.mean
                else:
                    action = dist.sample()
                    
                # Squeeze if single timestep
                if action.dim() == 3 and action.shape[1] == 1:
                    action = action.squeeze(1)
                    
                return action

    def reset_hidden(self, batch_size: int = 1):
        """Reset hidden states for all networks.
        
        Args:
            batch_size: Batch size for hidden state initialization.
        """
        if hasattr(self.actor, 'reset_hidden'):
            self.actor.reset_hidden(batch_size)
        if hasattr(self.critic, 'reset_hidden'):
            self.critic.reset_hidden(batch_size)
        if hasattr(self.value, 'reset_hidden'):
            self.value.reset_hidden(batch_size)
        
        self.actor_hidden = None
        self.critic_hidden = None
        self.value_hidden = None

    def save_model(self, name: str):
        """Save model checkpoints.
        
        Args:
            name: Checkpoint name.
        """
        base_path = f"runs/{wandb.run.project}/{wandb.run.id}/checkpoints/"
        paths = [
            "actor/", "critic/", "value/", 
            "optimizer/", "optimizer/actor/", "optimizer/critic/", "optimizer/value/",
            "critic_target/"
        ]

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

    def load_model(self, path: str, name: str = "best_model.pt"):
        """Load model checkpoints.
        
        Args:
            path: Path to checkpoint directory.
            name: Checkpoint name.
        """
        self.actor.load_state_dict(torch.load(path + "actor/" + name))
        self.critic.load_state_dict(torch.load(path + "critic/" + name))
        self.value.load_state_dict(torch.load(path + "value/" + name))

        self.actor_optimizer.load_state_dict(torch.load(path + "optimizer/actor/" + name))
        self.critic_optimizer.load_state_dict(torch.load(path + "optimizer/critic/" + name))
        self.value_optimizer.load_state_dict(torch.load(path + "optimizer/value/" + name))

        self.critic_target.load_state_dict(torch.load(path + "critic_target/" + name))
