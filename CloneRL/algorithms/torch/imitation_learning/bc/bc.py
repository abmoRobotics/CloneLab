"""Behavior Cloning (BC) Algorithm.

This module implements standard Behavior Cloning for imitation learning.
BC directly learns a policy by supervised learning on expert demonstrations,
minimizing the negative log-likelihood of expert actions given states.
"""

import os
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from CloneRL.algorithms.torch.offline_rl.base import BaseAgent


BC_DEFAULT_CONFIG = {
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "grad_clip": 1.0,
    "lr_scheduler": None,  # "step", "cosine", or None
    "lr_decay_steps": 1000,
    "lr_decay_rate": 0.99,
}


class BehaviourCloning(BaseAgent):
    """Behavior Cloning algorithm for imitation learning.
    
    BC learns a policy by maximizing the log-likelihood of expert actions
    given expert states. This is a simple but effective baseline for
    imitation learning.
    
    Supports:
    - Gaussian policies (continuous actions)
    - Deterministic policies
    - Mixed precision training
    - Gradient clipping
    
    Args:
        actor_policy: Policy network that outputs action distribution.
        cfg: Configuration dictionary.
        device: Device to use for computation.
        lr: Learning rate.
        weight_decay: L2 regularization weight.
        grad_clip: Gradient clipping value (None to disable).
        lr_scheduler: Learning rate scheduler type ("step", "cosine", or None).
        lr_decay_steps: Steps between LR decay (for step scheduler).
        lr_decay_rate: LR decay rate (for step scheduler).
    """

    def __init__(
        self,
        actor_policy: nn.Module,
        cfg: Dict,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        grad_clip: Optional[float] = 1.0,
        lr_scheduler: Optional[str] = None,
        lr_decay_steps: int = 1000,
        lr_decay_rate: float = 0.99,
    ):
        # Merge config with defaults
        _cfg = BC_DEFAULT_CONFIG.copy()
        _cfg.update(cfg if cfg is not None else {})
        
        super().__init__(_cfg, actor_policy, device=device)

        # Actor network
        self.actor = actor_policy.to(self.device)
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(), 
            lr=_cfg.get("lr", lr),
            weight_decay=_cfg.get("weight_decay", weight_decay),
        )
        
        # Gradient clipping
        self.grad_clip = _cfg.get("grad_clip", grad_clip)
        
        # Learning rate scheduler
        self.lr_scheduler = None
        scheduler_type = _cfg.get("lr_scheduler", lr_scheduler)
        if scheduler_type == "step":
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.actor_optimizer,
                step_size=_cfg.get("lr_decay_steps", lr_decay_steps),
                gamma=_cfg.get("lr_decay_rate", lr_decay_rate),
            )
        elif scheduler_type == "cosine":
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.actor_optimizer,
                T_max=10000,  # Will be updated based on training
            )
        
        # Training state
        self.total_steps = 0
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        
        self.initialize()

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
    ) -> float:
        """Training step for Behavior Cloning.
        
        Computes the negative log-likelihood loss and updates the actor.
        
        Args:
            state: Dictionary of state tensors.
            action: Expert actions tensor.
            reward: Rewards (not used in BC).
            next_state: Next states (not used in BC).
            done: Done flags (not used in BC).
            weights: Sample weights (optional weighting).
            step: Current training step.
            masks: Valid sample masks.
            
        Returns:
            Actor loss value.
        """
        self.actor.train()
        self.total_steps += 1
        
        # Get action distribution from policy
        dist = self.actor(state)
        
        # Compute negative log-likelihood loss
        if hasattr(dist, 'log_prob'):
            # Gaussian policy
            log_probs = dist.log_prob(action)
            if log_probs.dim() > 1:
                log_probs = log_probs.sum(dim=-1, keepdim=True)
            
            # Apply sample weights if provided
            if weights is not None:
                nll_loss = -(weights * log_probs).mean()
            else:
                nll_loss = -log_probs.mean()
        else:
            # Deterministic policy - use MSE loss
            predicted_actions = dist
            nll_loss = self.mse_loss(predicted_actions, action)
        
        # Backward pass
        self.actor_optimizer.zero_grad()
        nll_loss.backward()
        
        # Gradient clipping
        if self.grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), 
                self.grad_clip
            )
        else:
            grad_norm = 0.0
        
        self.actor_optimizer.step()
        
        # Update learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        # Logging
        wandb.log({
            "train/bc_loss": nll_loss.item(),
            "train/log_probs_mean": -nll_loss.item(),
            "train/grad_norm": grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
            "train/learning_rate": self.actor_optimizer.param_groups[0]['lr'],
        }, step=step)
        
        return nll_loss.item()

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
    ) -> float:
        """Validation step for Behavior Cloning.
        
        Computes the validation loss without updating parameters.
        
        Args:
            state: Dictionary of state tensors.
            action: Expert actions tensor.
            reward: Rewards (not used in BC).
            next_state: Next states (not used in BC).
            done: Done flags (not used in BC).
            weights: Sample weights.
            step: Current validation step.
            masks: Valid sample masks.
            
        Returns:
            Validation loss value.
        """
        self.actor.eval()
        
        with torch.no_grad():
            # Get action distribution from policy
            dist = self.actor(state)
            
            # Compute negative log-likelihood loss
            if hasattr(dist, 'log_prob'):
                log_probs = dist.log_prob(action)
                if log_probs.dim() > 1:
                    log_probs = log_probs.sum(dim=-1, keepdim=True)
                nll_loss = -log_probs.mean()
            else:
                # Deterministic policy
                predicted_actions = dist
                nll_loss = self.mse_loss(predicted_actions, action)
        
        return nll_loss.item()

    def act(self, state: Dict[str, torch.Tensor], done: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get action from policy.
        
        Args:
            state: Dictionary of state tensors.
            done: Done flags (not used in standard BC).
            
        Returns:
            Sampled actions tensor.
        """
        self.actor.eval()
        with torch.no_grad():
            dist = self.actor(state)
            if hasattr(dist, 'sample'):
                return dist.sample()
            else:
                return dist

    def get_action(self, state: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        """Get action from policy with optional deterministic mode.
        
        Args:
            state: Dictionary of state tensors.
            deterministic: If True, return mean action instead of sampling.
            
        Returns:
            Actions tensor.
        """
        self.actor.eval()
        with torch.no_grad():
            dist = self.actor(state)
            if hasattr(dist, 'mean'):
                if deterministic:
                    return dist.mean
                else:
                    return dist.sample()
            else:
                return dist

    def save_model(self, name: str) -> None:
        """Save model checkpoint.
        
        Args:
            name: Checkpoint filename.
        """
        base_path = f"runs/{wandb.run.project}/{wandb.run.id}/checkpoints/"
        paths = ["actor/", "optimizer/actor/"]

        for p in paths:
            full_path = base_path + p
            if not os.path.exists(full_path):
                os.makedirs(full_path)

        torch.save(self.actor.state_dict(), base_path + "actor/" + name)
        torch.save(self.actor_optimizer.state_dict(), base_path + "optimizer/actor/" + name)
        
        # Save scheduler state if exists
        if self.lr_scheduler is not None:
            scheduler_path = base_path + "scheduler/"
            if not os.path.exists(scheduler_path):
                os.makedirs(scheduler_path)
            torch.save(self.lr_scheduler.state_dict(), scheduler_path + name)

    def load_model(self, path: str, name: str = "best_model.pt") -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint directory.
            name: Checkpoint filename.
        """
        self.actor.load_state_dict(torch.load(path + "actor/" + name, weights_only=True))
        
        optimizer_path = path + "optimizer/actor/" + name
        if os.path.exists(optimizer_path):
            self.actor_optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True))
        
        scheduler_path = path + "scheduler/" + name
        if self.lr_scheduler is not None and os.path.exists(scheduler_path):
            self.lr_scheduler.load_state_dict(torch.load(scheduler_path, weights_only=True))
