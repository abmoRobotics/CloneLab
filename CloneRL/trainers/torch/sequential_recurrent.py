"""Sequential Recurrent Trainer for GRU/RNN-based models.

This trainer is designed for training recurrent models (GRU, LSTM) with sequential data.
It handles hidden state management across batches and episodes.
"""

import copy
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

import wandb
from CloneRL.algorithms.torch.imitation_learning.base import BaseAgent
from CloneRL.trainers.torch.base import BaseTrainer


SEQUENTIAL_RECURRENT_TRAINER_DEFAULT_CONFIG = {
    "batch_size": 8,
    "num_workers": 2,
    "prefetch_factor": 2,
    "shuffle": True,
    "epochs": 1,
    "simulator": None,
    "early_stopping_patience": 10,
    "save_freq": 2,
    "validation_freq": 1,
    "log_freq": 100,
    "mixed_precision": True,
    "sequence_length": 16,  # Default sequence length for GRU
    "reset_hidden_on_done": True,  # Reset hidden state when episode ends
    "truncated_bptt_steps": None,  # If set, use truncated backprop through time
    "image_mode": "rgb",  # "rgb", "depth", or None/"none" for no image encoder
}


def recurrent_collate_fn(batch):
    """Collate function for recurrent data with sequence handling.
    
    Args:
        batch: List of tuples (obs, actions, rewards, next_obs, dones, weights, masks).
               Each element has shape (seq_len, ...).
    
    Returns:
        Batched tensors with shape (batch_size, seq_len, ...).
    """
    # Check if batch contains hidden_reset_mask (8 elements vs 7)
    has_hidden_reset = len(batch[0]) == 8
    
    # Stack observations (dictionaries)
    obs = {k: torch.stack([b[0][k] for b in batch], dim=0) for k in batch[0][0].keys()}
    actions = torch.stack([b[1] for b in batch], dim=0)
    rewards = torch.stack([b[2] for b in batch], dim=0)
    next_obs = {k: torch.stack([b[3][k] for b in batch], dim=0) for k in batch[0][3].keys()}
    dones = torch.stack([b[4] for b in batch], dim=0)
    weights = torch.stack([b[5] for b in batch], dim=0)
    masks = torch.stack([b[6] for b in batch], dim=0)
    
    if has_hidden_reset:
        hidden_reset = torch.stack([b[7] for b in batch], dim=0)
        return obs, actions, rewards, next_obs, dones, weights, masks, hidden_reset
    
    return obs, actions, rewards, next_obs, dones, weights, masks


class SequentialRecurrentTrainer(BaseTrainer):
    """Trainer for sequential recurrent models (GRU, LSTM).
    
    This trainer handles:
    - Hidden state management across sequences
    - Truncated backpropagation through time (optional)
    - Proper gradient flow for recurrent models
    - Episode boundary handling for hidden state resets
    """
    
    def __init__(
        self,
        policy: BaseAgent,
        cfg: Dict[str, Any],
        env: Optional[Any] = None,
        env_loader: Optional[Any] = None,
        dataset: DataLoader = None,
        val_dataset: DataLoader = None,
    ) -> None:
        
        _cfg = copy.deepcopy(SEQUENTIAL_RECURRENT_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        
        # Don't call parent __init__ directly, we'll set up dataloaders ourselves
        self.cfg = _cfg
        self.policy = policy
        loader_kwargs = {
            "batch_size": self.cfg["batch_size"],
            "num_workers": self.cfg["num_workers"],
            "collate_fn": recurrent_collate_fn,
            "drop_last": True,
        }
        if self.cfg["num_workers"] > 0:
            loader_kwargs["prefetch_factor"] = self.cfg["prefetch_factor"]
        
        # Create dataloaders with recurrent collate function
        self.train_ds = DataLoader(
            dataset,
            **loader_kwargs,
            shuffle=self.cfg["shuffle"],
        )
        self.train_val_ds = DataLoader(
            val_dataset,
            **loader_kwargs,
            shuffle=False,
        )
        
        self.policy.initialize()
        
        # Start simulation asynchronously if env_loader provided
        if env_loader is not None:
            self.simulator_thread = threading.Thread(
                target=self.simulation_validation, 
                args=(env_loader,)
            )
            self.simulator_thread.start()
            print("Simulation started")

    def simulation_validation(self, env_loader):
        """Run the environment simulation asynchronously with recurrent policy."""
        if env_loader is not None:
            self.env = env_loader()

        obs, info = self.env.reset()
        first_iter = True
        num_steps = 100000
        
        # Get image_mode from config
        image_mode = self.cfg.get("image_mode", "rgb")
        if isinstance(image_mode, str):
            image_mode = image_mode.lower()

        # Reset hidden states for all environments
        if hasattr(self.policy, 'reset_hidden'):
            self.policy.reset_hidden(batch_size=obs.shape[0] if not isinstance(obs, dict) else obs[list(obs.keys())[0]].shape[0])

        for timestep in range(num_steps):
            with torch.no_grad():
                if first_iter:
                    first_iter = False
                    actions = torch.zeros((obs.shape[0], 2))
                else:
                    # Process depth image (always needed)
                    depth = info['depth'].unsqueeze(1)
                    depth = torch.nan_to_num(depth, nan=6.0)
                    depth = torch.clamp(depth, min=0.0, max=6.0)
                    
                    # Build state dict based on image_mode
                    state = {
                        'proprioceptive': obs[:, :4], 
                        'depth': depth,       # depth only (1 channel)
                    }
                    
                    # Helper function to compute grayscale
                    def compute_grayscale(rgb_tensor):
                        """Convert RGB to grayscale using standard weights."""
                        grayscale = rgb_tensor[:, 0] * 0.2989 + rgb_tensor[:, 1] * 0.5870 + rgb_tensor[:, 2] * 0.1140
                        return grayscale.unsqueeze(1)  # (B, 1, H, W)
                    
                    # Helper to get and process RGB
                    def get_rgb():
                        rgb = info.get('rgb', info.get('rgb_image'))
                        if rgb is not None:
                            if rgb.dim() == 4 and rgb.shape[-1] in [3, 4]:  # (B, H, W, C)
                                rgb = rgb.permute(0, 3, 1, 2)  # (B, C, H, W)
                            return rgb[:, :3]
                        return None
                    
                    # Add image based on mode
                    if image_mode == "rgb":
                        # RGB only (3 channels)
                        rgb = get_rgb()
                        if rgb is not None:
                            image = rgb / 255.0
                        else:
                            image = depth.repeat(1, 3, 1, 1)  # Fallback
                        state['image'] = image
                    elif image_mode == "grayscale":
                        # Grayscale only (1 channel)
                        rgb = get_rgb()
                        if rgb is not None:
                            image = compute_grayscale(rgb / 255.0)
                        else:
                            image = depth.clone()  # Fallback
                        state['image'] = image
                    elif image_mode == "depth":
                        # Depth only (1 channel)
                        state['image'] = depth.clone()
                    elif image_mode == "rgbd":
                        # RGB + Depth (4 channels)
                        rgb = get_rgb()
                        if rgb is not None:
                            rgb_normalized = rgb / 255.0
                            image = torch.cat([rgb_normalized, depth], dim=1)
                        else:
                            image = depth.repeat(1, 4, 1, 1)  # Fallback
                        state['image'] = image
                    elif image_mode == "grayscale_depth":
                        # Grayscale + Depth (2 channels)
                        rgb = get_rgb()
                        if rgb is not None:
                            grayscale = compute_grayscale(rgb / 255.0)
                            image = torch.cat([grayscale, depth], dim=1)
                        else:
                            image = depth.repeat(1, 2, 1, 1)  # Fallback
                        state['image'] = image
                    # else: image_mode is None or "none" - no image key added
                    
                    # # --- OLD CODE (always RGB) ---
                    # # Process RGB and normalize to [0, 1] for image encoder
                    # rgb = info.get('rgb', info.get('rgb_image'))
                    # if rgb is not None:
                    #     if rgb.dim() == 4 and rgb.shape[-1] in [3, 4]:  # (B, H, W, C)
                    #         rgb = rgb.permute(0, 3, 1, 2)  # (B, C, H, W)
                    #     image = rgb[:, :3] / 255.0  # Normalize RGB to [0, 1]
                    # else:
                    #     # Fallback if RGB not available
                    #     image = depth.repeat(1, 3, 1, 1)  # Use depth as grayscale fallback
                    # state = {
                    #     'proprioceptive': obs[:, :4], 
                    #     'image': image,       # RGB normalized (3 channels)
                    #     'depth': depth,       # depth only (1 channel)
                    # }
                    # # --- END OLD CODE ---
                    
                    # Get action from recurrent policy
                    if hasattr(self.policy, 'act'):
                        actions = self.policy.act(state, terminated if 'terminated' in dir() else None)
                    else:
                        actions = self.policy.actor.get_action(state, deterministic=True)

                next_obs, rewards, terminated, truncated, next_info = self.env.step(actions)
                
                # Log reward metrics if supported
                if hasattr(self.policy, "record_transitions"):
                    self.policy.record_transitions(rewards, terminated, truncated, next_info, step=timestep)
                
                obs, info = next_obs, next_info

        if hasattr(self.policy, "finalize_summary"):
            try:
                self.policy.finalize_summary()
            except Exception:
                pass

    def train(self):
        """Train the recurrent model for specified epochs."""
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        print("Starting recurrent training...")
        print(f"Training for {self.cfg['epochs']} epochs with batch size {self.cfg['batch_size']}...")
        print(f"Sequence length: {self.cfg.get('sequence_length', 'variable')}")
        
        # Setup mixed precision training
        scaler = torch.amp.GradScaler() if self.cfg.get('mixed_precision', True) else None
        
        for current_epoch in range(self.cfg["epochs"]):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss = self._process_epoch(
                self.train_ds, 
                current_epoch, 
                train=True, 
                scaler=scaler
            )
            train_losses.append(train_loss)
            
            # Validation phase
            if current_epoch % self.cfg.get('validation_freq', 1) == 0:
                val_loss = self._process_epoch(
                    self.train_val_ds, 
                    current_epoch, 
                    train=False, 
                    scaler=None
                )
                val_losses.append(val_loss)
                
                # Log epoch metrics
                epoch_time = time.time() - epoch_start_time
                wandb.log({
                    "epoch/validation_loss": val_loss,
                    "epoch/train_loss": train_loss,
                    "epoch/epoch": current_epoch,
                    "epoch/epoch_time": epoch_time,
                    "epoch/learning_rate": self._get_learning_rate(),
                }, step=(current_epoch + 1) * len(self.train_ds))
                
                # Early stopping and model saving
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.policy.save_model(f"best_model_epoch_{current_epoch}.pt")
                    print(f"✓ Model saved with improved validation loss: {best_val_loss:.6f}")
                else:
                    patience_counter += 1
                    
                # Early stopping check
                if patience_counter >= self.cfg.get('early_stopping_patience', float('inf')):
                    print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                    break
                    
            # Regular checkpointing
            if current_epoch % self.cfg.get('save_freq', 5) == 0:
                self.policy.save_model(f"checkpoint_epoch_{current_epoch}.pt")
                
            # Progress reporting
            print(f"Epoch {current_epoch:3d}/{self.cfg['epochs']:3d} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Best Val: {best_val_loss:.6f} | "
                  f"Patience: {patience_counter}/{self.cfg.get('early_stopping_patience', 'inf')}")
        
        # Save final model
        self.policy.save_model("final_model.pt")
        print(f"Training completed! Best validation loss: {best_val_loss:.6f}")
        
        return {
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'total_epochs': current_epoch + 1
        }

    def _get_learning_rate(self) -> float:
        """Get current learning rate from optimizer."""
        if hasattr(self.policy, 'actor_optimizer'):
            return self.policy.actor_optimizer.param_groups[0]['lr']
        return 0.0

    def _process_epoch(self, dataset, epoch: int, train: bool, scaler=None):
        """Process one epoch with recurrent model handling."""
        total_loss = 0.0
        num_batches = 0
        phase = 'Train' if train else 'Val'
        pbar_description = f"{phase} Epoch {epoch}"
        pbar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]"

        # Set model to appropriate mode
        self._set_model_mode(train)

        with tqdm.tqdm(dataset, desc=pbar_description, bar_format=pbar_format) as pbar:
            for idx, batch_data in enumerate(pbar):
                # Unpack batch data (may include hidden_reset_mask)
                if len(batch_data) == 8:
                    obs, actions, rewards, next_obs, dones, weights, masks, hidden_reset = batch_data
                else:
                    obs, actions, rewards, next_obs, dones, weights, masks = batch_data
                    hidden_reset = None
                
                # Move data to device
                obs = self._move_to_device(obs)
                next_obs = self._move_to_device(next_obs)
                actions = actions.to(self.policy.device)
                rewards = rewards.to(self.policy.device)
                dones = dones.to(self.policy.device)
                weights = weights.to(self.policy.device)
                masks = masks.to(self.policy.device)
                if hidden_reset is not None:
                    hidden_reset = hidden_reset.to(self.policy.device)
                
                step = epoch * len(dataset) + idx
                
                # Training or validation step
                if train:
                    if scaler is not None:
                        # Mixed precision training
                        with torch.amp.autocast(device_type='cuda' if 'cuda' in str(self.policy.device) else 'cpu'):
                            loss = self.policy.train(
                                obs, actions, rewards, next_obs, dones, weights, step, masks,
                                hidden_reset=hidden_reset
                            )
                    else:
                        # Regular precision training
                        loss = self.policy.train(
                            obs, actions, rewards, next_obs, dones, weights, step, masks,
                            hidden_reset=hidden_reset
                        )
                else:
                    # Validation
                    with torch.no_grad():
                        loss = self.policy.validate(
                            obs, actions, rewards, next_obs, dones, weights, step, masks,
                            hidden_reset=hidden_reset
                        )
                
                loss_val = loss
                total_loss += loss_val
                num_batches += 1
                
                # Update progress bar
                if (idx + 1) == len(dataset):
                    average_loss = total_loss / num_batches if num_batches > 0 else 0.0
                    pbar.set_postfix({f'Avg {phase.lower()} loss': f"{average_loss:.6f}"})
                else:
                    pbar.set_postfix({"Batch loss": f"{loss_val:.6f}"})
                
                # Log batch-level metrics occasionally
                if train and idx % self.cfg.get('log_freq', 100) == 0:
                    wandb.log({
                        f"batch/{phase.lower()}_loss": loss_val,
                        "batch/step": step,
                    }, step=step)

        average_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return average_loss

    def _set_model_mode(self, train: bool):
        """Set model to train or eval mode."""
        mode = 'train' if train else 'eval'
        for attr in ['actor', 'critic', 'value']:
            if hasattr(self.policy, attr):
                model = getattr(self.policy, attr)
                if train:
                    model.train()
                else:
                    model.eval()

    def _move_to_device(self, data):
        """Move data to device, handling both tensors and dicts."""
        if isinstance(data, dict):
            return {k: v.to(self.policy.device) for k, v in data.items()}
        return data.to(self.policy.device)

    def evaluate(
        self, 
        env, 
        num_steps: int = 1000,
        deterministic: bool = True,
    ):
        """Evaluate the recurrent policy in the environment.
        
        Args:
            env: The environment to evaluate in.
            num_steps: Number of steps to run.
            deterministic: Whether to use deterministic actions.
        """
        obs, info = env.reset()
        first_iter = True
        
        if isinstance(obs, dict):
            obs = obs.get("policy", obs)
            num_envs = obs["depth_image"].shape[0] if "depth_image" in obs else obs[list(obs.keys())[0]].shape[0]
            terminated = torch.zeros((num_envs, 1), dtype=torch.bool)
        else:
            num_envs = obs.shape[0]
            terminated = torch.zeros((num_envs, 1), dtype=torch.bool)
        
        # Reset hidden states for all environments
        if hasattr(self.policy, 'reset_hidden'):
            self.policy.reset_hidden(batch_size=num_envs)
        elif hasattr(self.policy, 'actor') and hasattr(self.policy.actor, 'reset_hidden'):
            self.policy.actor.reset_hidden(batch_size=num_envs)
        
        print(f"Starting recurrent evaluation with {num_envs} environments...")
        
        # Get image_mode from config
        image_mode = self.cfg.get("image_mode", "rgb")
        if isinstance(image_mode, str):
            image_mode = image_mode.lower()
        print(f"Image mode: {image_mode}")
        
        for timestep in tqdm.tqdm(range(num_steps)):
            with torch.inference_mode():
                if first_iter:
                    first_iter = False
                    actions = torch.zeros((num_envs, 2))
                else:
                    obs = obs.get("policy", obs) if isinstance(obs, dict) and "policy" in obs else obs
                    
                    # Process depth image (always needed)
                    depth_image = obs["depth_image"].permute(0, 3, 1, 2)  # (B, C, H, W)
                    depth_image = torch.nan_to_num(depth_image, nan=6.0)
                    depth_image = torch.where(torch.isinf(depth_image), torch.tensor(6.0, device=depth_image.device), depth_image)
                    depth_image = torch.clip(depth_image, 0.0, 6.0)
                    
                    # Get proprioceptive data
                    angle_diff_obs = obs.get("angle_diff", torch.zeros(num_envs, 1, device=depth_image.device))
                    distance_obs = obs.get("distance", torch.zeros(num_envs, 1, device=depth_image.device))
                    heading_obs = obs.get("heading", torch.zeros(num_envs, 1, device=depth_image.device))
                    proprioceptive_obs = torch.cat((angle_diff_obs, distance_obs, heading_obs), dim=1)
                    
                    # Build state dict based on image_mode
                    state = {
                        'proprioceptive': proprioceptive_obs,
                        'depth': depth_image,  # depth only (1 channel)
                    }
                    
                    # Helper function to compute grayscale
                    def compute_grayscale(rgb_tensor):
                        """Convert RGB to grayscale using standard weights."""
                        grayscale = rgb_tensor[:, 0] * 0.2989 + rgb_tensor[:, 1] * 0.5870 + rgb_tensor[:, 2] * 0.1140
                        return grayscale.unsqueeze(1)  # (B, 1, H, W)
                    
                    # Add image based on mode
                    if image_mode == "rgb":
                        # RGB only (3 channels)
                        rgb = obs["rgb_image"].permute(0, 3, 1, 2)  # (B, C, H, W)
                        image = rgb / 255.0
                        state['image'] = image
                    elif image_mode == "grayscale":
                        # Grayscale only (1 channel)
                        rgb = obs["rgb_image"].permute(0, 3, 1, 2)
                        image = compute_grayscale(rgb / 255.0)
                        state['image'] = image
                    elif image_mode == "depth":
                        # Depth only (1 channel)
                        state['image'] = depth_image.clone()
                    elif image_mode == "rgbd":
                        # RGB + Depth (4 channels)
                        rgb = obs["rgb_image"].permute(0, 3, 1, 2)
                        rgb_normalized = rgb / 255.0
                        image = torch.cat([rgb_normalized, depth_image], dim=1)
                        state['image'] = image
                    elif image_mode == "grayscale_depth":
                        # Grayscale + Depth (2 channels)
                        rgb = obs["rgb_image"].permute(0, 3, 1, 2)
                        grayscale = compute_grayscale(rgb / 255.0)
                        image = torch.cat([grayscale, depth_image], dim=1)
                        state['image'] = image
                    # else: image_mode is None or "none" - no image key added
                    
                    # # --- OLD CODE (always RGB) ---
                    # # Process RGB and normalize to [0, 1] for image encoder
                    # rgb = obs["rgb_image"].permute(0, 3, 1, 2)  # (B, C, H, W)
                    # image = rgb / 255.0  # Normalize RGB to [0, 1] (B, 3, H, W)
                    # state = {
                    #     'proprioceptive': proprioceptive_obs,
                    #     'image': image,        # RGB normalized (3 channels)
                    #     'depth': depth_image,  # depth only (1 channel)
                    # }
                    # # --- END OLD CODE ---
                    
                    # Get action from recurrent policy
                    if hasattr(self.policy, 'act'):
                        actions = self.policy.act(state, terminated)
                    elif hasattr(self.policy, 'actor'):
                        if hasattr(self.policy.actor, 'validate'):
                            actions = self.policy.actor.validate(state, terminated.squeeze(-1))
                        else:
                            actions = self.policy.actor.get_action(state, deterministic=deterministic)
                    else:
                        raise ValueError("Policy must have 'act' method or 'actor' attribute")

                next_obs, rewards, terminated, truncated, next_info = env.step(actions)
                
                # Log reward metrics if supported
                if hasattr(self.policy, "record_transitions"):
                    self.policy.record_transitions(rewards, terminated, truncated, next_info, step=timestep)
                
                obs, info = next_obs, next_info

        print("Evaluation completed!")
