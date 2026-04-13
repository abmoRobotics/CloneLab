"""Behavior Cloning with Recurrent Neural Networks (BC-RNN).

This module implements Behavior Cloning for recurrent models (GRU/LSTM).
It handles sequential data and hidden state management for learning
policies with temporal dependencies from expert demonstrations.
"""

import os
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from CloneRL.algorithms.torch.offline_rl.base import BaseAgent


BC_RNN_DEFAULT_CONFIG = {
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "grad_clip": 1.0,
    "lr_scheduler": None,
    "lr_decay_steps": 1000,
    "lr_decay_rate": 0.99,
    "reset_hidden_on_done": True,
}


class BehaviourCloningRNN(BaseAgent):
    """Behavior Cloning algorithm for recurrent (GRU/LSTM) models.
    
    This implementation handles:
    - Sequential data with shape (batch, seq_len, ...)
    - Hidden state management across sequences
    - Proper gradient flow through time
    - Episode boundary handling
    
    Args:
        actor_policy: GRU/LSTM-based policy network.
        cfg: Configuration dictionary.
        device: Device to use for computation.
        lr: Learning rate.
        weight_decay: L2 regularization weight.
        grad_clip: Gradient clipping value (None to disable).
        lr_scheduler: Learning rate scheduler type.
        lr_decay_steps: Steps between LR decay.
        lr_decay_rate: LR decay rate.
        reset_hidden_on_done: Whether to reset hidden states on episode end.
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
        reset_hidden_on_done: bool = True,
    ):
        # Merge config with defaults
        _cfg = BC_RNN_DEFAULT_CONFIG.copy()
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
        
        # Hidden state reset on done
        self.reset_hidden_on_done = _cfg.get("reset_hidden_on_done", reset_hidden_on_done)
        
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
                T_max=10000,
            )
        
        # Training state
        self.total_steps = 0
        
        # Hidden state for evaluation
        self.actor_hidden = None
        
        # Loss function
        self.mse_loss = nn.MSELoss(reduction='none')
        
        self.initialize()

    def _apply_mask(self, tensor: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Apply mask to tensor for valid timesteps only.
        
        Args:
            tensor: Input tensor of shape (batch, seq_len, ...).
            masks: Mask tensor of shape (batch, seq_len, 1).
            
        Returns:
            Masked tensor.
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

    def _prepare_sequential_obs(
        self, 
        state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Prepare observation dict for sequential processing.
        
        Ensures all tensors have shape (batch, seq_len, ...).
        
        Args:
            state: Dictionary of state tensors.
            
        Returns:
            Prepared state dictionary.
        """
        prepared = {}
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                # Ensure tensor is on correct device
                value = value.to(self.device)
                prepared[key] = value
        return prepared

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
        """Training step for BC-RNN.
        
        Processes sequences and computes masked negative log-likelihood loss.
        
        Args:
            state: Dictionary of state tensors with shape (batch, seq_len, ...).
            action: Expert actions tensor (batch, seq_len, action_dim).
            reward: Rewards (not used in BC).
            next_state: Next states (not used in BC).
            done: Done flags for hidden state reset.
            weights: Sample weights.
            step: Current training step.
            masks: Valid sample masks (batch, seq_len, 1).
            hidden_reset: Optional mask for hidden state resets (not used in BC).
            
        Returns:
            Actor loss value.
        """
        self.actor.train()
        self.total_steps += 1
        
        # Prepare state for sequential processing
        state = self._prepare_sequential_obs(state)
        action = action.to(self.device)
        masks = masks.to(self.device)
        done = done.to(self.device) if done is not None else None
        
        # Get batch and sequence dimensions
        batch_size = action.shape[0]
        seq_len = action.shape[1]
        
        # Initialize hidden state
        hidden = None
        
        # Forward pass through sequence
        # The GRU actor should handle the full sequence
        if hasattr(self.actor, 'forward_sequence'):
            # Use sequence-aware forward if available
            dist, _ = self.actor.forward_sequence(state, hidden, done)
        else:
            # Standard forward - actor should handle sequences
            dist = self.actor(state, hidden)
            if isinstance(dist, tuple):
                dist, _ = dist
        
        # Compute negative log-likelihood loss
        if hasattr(dist, 'log_prob'):
            # Gaussian policy
            log_probs = dist.log_prob(action)  # (batch, seq_len, action_dim)
            if log_probs.dim() > 2:
                log_probs = log_probs.sum(dim=-1, keepdim=True)  # (batch, seq_len, 1)
            
            # Apply masking for valid timesteps
            nll_loss = self._masked_mean(-log_probs, masks)
        else:
            # Deterministic policy - use MSE loss
            predicted_actions = dist
            mse = self.mse_loss(predicted_actions, action)
            if mse.dim() > 2:
                mse = mse.sum(dim=-1, keepdim=True)
            nll_loss = self._masked_mean(mse, masks)
        
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
            "train/bc_rnn_loss": nll_loss.item(),
            "train/grad_norm": grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
            "train/learning_rate": self.actor_optimizer.param_groups[0]['lr'],
            "train/valid_timesteps": masks.sum().item(),
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
        hidden_reset: Optional[torch.Tensor] = None,
    ) -> float:
        """Validation step for BC-RNN.
        
        Computes validation loss without updating parameters.
        
        Args:
            state: Dictionary of state tensors.
            action: Expert actions tensor.
            reward: Rewards (not used in BC).
            next_state: Next states (not used in BC).
            done: Done flags.
            weights: Sample weights.
            step: Current validation step.
            masks: Valid sample masks.
            hidden_reset: Optional mask for hidden state resets (not used in BC).
            
        Returns:
            Validation loss value.
        """
        self.actor.eval()
        
        with torch.no_grad():
            # Prepare state for sequential processing
            state = self._prepare_sequential_obs(state)
            action = action.to(self.device)
            masks = masks.to(self.device)
            done = done.to(self.device) if done is not None else None
            
            # Forward pass
            hidden = None
            if hasattr(self.actor, 'forward_sequence'):
                dist, _ = self.actor.forward_sequence(state, hidden, done)
            else:
                dist = self.actor(state, hidden)
                if isinstance(dist, tuple):
                    dist, _ = dist
            
            # Compute loss
            if hasattr(dist, 'log_prob'):
                log_probs = dist.log_prob(action)
                if log_probs.dim() > 2:
                    log_probs = log_probs.sum(dim=-1, keepdim=True)
                nll_loss = self._masked_mean(-log_probs, masks)
            else:
                predicted_actions = dist
                mse = self.mse_loss(predicted_actions, action)
                if mse.dim() > 2:
                    mse = mse.sum(dim=-1, keepdim=True)
                nll_loss = self._masked_mean(mse, masks)
        
        return nll_loss.item()

    def reset_hidden_states(self, batch_size: int) -> None:
        """Reset hidden states for evaluation.
        
        Args:
            batch_size: Number of environments.
        """
        self.actor_hidden = None

    def act(self, state: Dict[str, torch.Tensor], done: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get action from recurrent policy.
        
        Maintains hidden state across calls for temporal consistency.
        
        Args:
            state: Dictionary of state tensors (batch, ...) - single timestep.
            done: Done flags to reset hidden states.
            
        Returns:
            Sampled actions tensor.
        """
        self.actor.eval()
        
        with torch.no_grad():
            # Reset hidden states for done environments
            if done is not None and self.reset_hidden_on_done and self.actor_hidden is not None:
                done_mask = done.squeeze(-1) if done.dim() > 1 else done
                if done_mask.any():
                    # Reset hidden states for done environments
                    if isinstance(self.actor_hidden, tuple):
                        h, c = self.actor_hidden
                        h = h.clone()
                        c = c.clone()
                        h[:, done_mask, :] = 0
                        c[:, done_mask, :] = 0
                        self.actor_hidden = (h, c)
                    else:
                        self.actor_hidden = self.actor_hidden.clone()
                        self.actor_hidden[:, done_mask, :] = 0
            
            # Prepare single-step observation
            # Add sequence dimension if not present
            state_seq = {}
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)
                    if value.dim() == 2:  # (batch, features) -> (batch, 1, features)
                        value = value.unsqueeze(1)
                    elif value.dim() == 4:  # (batch, C, H, W) -> (batch, 1, C, H, W)
                        value = value.unsqueeze(1)
                    state_seq[key] = value
            
            # Forward pass with hidden state
            if hasattr(self.actor, 'forward_step'):
                # Use single-step forward if available
                dist, self.actor_hidden = self.actor.forward_step(state_seq, self.actor_hidden)
            else:
                dist = self.actor(state_seq, self.actor_hidden)
                if isinstance(dist, tuple):
                    dist, self.actor_hidden = dist
            
            # Sample action
            if hasattr(dist, 'sample'):
                action = dist.sample()
            else:
                action = dist
            
            # Remove sequence dimension if added
            if action.dim() == 3 and action.shape[1] == 1:
                action = action.squeeze(1)
            
            return action

    def get_action(self, state: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        """Get action from policy with optional deterministic mode.
        
        Args:
            state: Dictionary of state tensors.
            deterministic: If True, return mean action.
            
        Returns:
            Actions tensor.
        """
        self.actor.eval()
        
        with torch.no_grad():
            # Prepare single-step observation
            state_seq = {}
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)
                    if value.dim() == 2:
                        value = value.unsqueeze(1)
                    elif value.dim() == 4:
                        value = value.unsqueeze(1)
                    state_seq[key] = value
            
            # Forward pass
            if hasattr(self.actor, 'forward_step'):
                dist, self.actor_hidden = self.actor.forward_step(state_seq, self.actor_hidden)
            else:
                dist = self.actor(state_seq, self.actor_hidden)
                if isinstance(dist, tuple):
                    dist, self.actor_hidden = dist
            
            # Get action
            if hasattr(dist, 'mean'):
                action = dist.mean if deterministic else dist.sample()
            else:
                action = dist
            
            # Remove sequence dimension
            if action.dim() == 3 and action.shape[1] == 1:
                action = action.squeeze(1)
            
            return action

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
