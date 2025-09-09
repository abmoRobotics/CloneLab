import copy
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch.nn.functional as F
import torch
import tqdm
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils.spaces.torch import unflatten_tensorized_space
from torch.utils.data import DataLoader

import wandb
from CloneRL.algorithms.torch.imitation_learning.base import BaseAgent
from CloneRL.trainers.torch.base import BaseTrainer



SEQUENTIAL_TRAINER_DEFAULT_CONFIG = {
    "batch_size": 64,
    "num_workers": 2,
    "prefetch_factor": 2,
    "shuffle": False,
    "epochs": 10,
    "simulator": None,
    "early_stopping_patience": 10,
    "save_freq": 2,
    "validation_freq": 1,
    "log_freq": 100,
    "mixed_precision": True,
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
                # Log reward metrics if supported by the policy
                if hasattr(self.policy, "record_transitions"):
                    self.policy.record_transitions(rewards, terminated, truncated, next_info, step=timestep)
                obs, info = next_obs, next_info

        # Ensure run-level summary (e.g., success rate) is persisted to W&B
        if hasattr(self.policy, "finalize_summary"):
            try:
                self.policy.finalize_summary()
            except Exception:
                pass

    def train(self):
        """
        Trains and validates the model for a specified number of epochs, saving the model
        when the validation loss improves.

        Args:
        num_epochs (int): The number of epochs to train and validate the model.
        """
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        print("Starting training...")
        print(f"Training for {self.cfg['epochs']} epochs with batch size {self.cfg['batch_size']}...")
        
        # Setup mixed precision training
        scaler = torch.amp.GradScaler() if self.cfg.get('mixed_precision', True) else None
        
        for current_epoch in range(self.cfg["epochs"]):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss = self._process_epoch(self.train_ds, current_epoch, train=True, scaler=scaler)
            train_losses.append(train_loss)
            
            # Validation phase
            if current_epoch % self.cfg.get('validation_freq', 1) == 0:
                val_loss = self._process_epoch(self.train_val_ds, current_epoch, train=False, scaler=None)
                val_losses.append(val_loss)
                
                # Log epoch metrics
                epoch_time = time.time() - epoch_start_time
                wandb.log({
                    "epoch/validation_loss": val_loss,
                    "epoch/train_loss": train_loss,
                    "epoch/epoch": current_epoch,
                    "epoch/epoch_time": epoch_time,
                    "epoch/learning_rate": self.policy.actor_optimizer.param_groups[0]['lr'] if hasattr(self.policy, 'actor_optimizer') else 0,
                }, step=(current_epoch+1)*len(self.train_ds))
                
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

    def _process_epoch(self, dataset, epoch: int, train: bool, scaler=None):
        """
        Enhanced epoch processing with mixed precision support and better error handling.
        """
        total_loss = 0.0
        num_batches = 0
        phase = 'Train' if train else 'Val'
        pbar_description = f"{phase} Epoch {epoch}"
        pbar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]"

        # Set model to appropriate mode
        if train:
            if hasattr(self.policy, 'actor'):
                self.policy.actor.train()
            if hasattr(self.policy, 'critic'):
                self.policy.critic.train()
            if hasattr(self.policy, 'value'):
                self.policy.value.train()
        else:
            if hasattr(self.policy, 'actor'):
                self.policy.actor.eval()
            if hasattr(self.policy, 'critic'):
                self.policy.critic.eval()
            if hasattr(self.policy, 'value'):
                self.policy.value.eval()

        with tqdm.tqdm(dataset, desc=pbar_description, bar_format=pbar_format) as pbar:
            for idx, batch_data in enumerate(pbar):
                # Unpack batch data
                obs, actions, rewards, next_obs, dones, weights, masks = batch_data
                
                # Move data to device
                if not isinstance(obs, dict):
                    obs = obs.to(self.policy.device)
                else:
                    obs = {k: v.to(self.policy.device) for k, v in obs.items()}
                
                if not isinstance(next_obs, dict):
                    next_obs = next_obs.to(self.policy.device)
                else:
                    next_obs = {k: v.to(self.policy.device) for k, v in next_obs.items()}
                
                actions = actions.to(self.policy.device)
                rewards = rewards.to(self.policy.device)
                dones = dones.to(self.policy.device)
                weights = weights.to(self.policy.device)
                masks = masks.to(self.policy.device)
                
                step = epoch * len(dataset) + idx
                
                # Training or validation step
                if train and scaler is not None:
                    # Mixed precision training
                    with torch.amp.autocast(device_type='cuda' if 'cuda' in str(self.policy.device) else 'cpu'):
                        loss = self.policy.train(obs, actions, rewards, next_obs, dones, weights, step, masks)
                elif train:
                    # Regular precision training
                    loss = self.policy.train(obs, actions, rewards, next_obs, dones, weights, step, masks)
                else:
                    # Validation
                    with torch.no_grad():
                        loss = self.policy.validate(obs, actions, rewards, next_obs, dones, weights, step, masks)
                
                # Handle loss (could be tensor or float)
                # if torch.is_tensor(loss):
                #     loss_val = loss.item()
                # else:
                #     loss_val = float(loss)
                loss_val = loss # TODO: Consider checking type as above
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
                        
                # except Exception as e:
                #     print(f"Error processing batch {idx}: {str(e)}")
                #     continue

        average_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return average_loss

    def evaluate(self, env, num_steps=1000):
        obs, info = env.reset()
        first_iter = True
        #print(f'num envs: {env.num_envs}')
        #terminated = torch.zeros((obs.shape[0], 1), dtype=torch.bool)
        if isinstance(obs, dict):
            obs = obs["policy"]
            #print(f'obs keys: {obs["policy"].keys()}')
            print(f'number of envs: {obs["depth_image"].shape[0]}')
            terminated = torch.zeros((obs["depth_image"].shape[0], 1), dtype=torch.bool)
        print("Starting evaluation...")
        for timestep in tqdm.tqdm(range(num_steps)):
            with torch.inference_mode():
                if first_iter:
                    first_iter = False
                    actions = torch.zeros((obs["depth_image"].shape[0], 2))
                else:
                    ##### OOOOLLLLDDD IMPLEMENTATION #####
                    # rgb = info['rgb']
                    # rgb = torch.nan_to_num(rgb, nan=256.0)
                    # rgb = torch.clamp(rgb, min=0.0, max=256.0)
                    # rgb = rgb.permute(0, 3, 1, 2)
                    # depth = info['depth'].unsqueeze(1)
                    # depth = torch.nan_to_num(depth, nan=256.0)
                    # depth = torch.clamp(depth, min=0.0, max=256.0)
                    # state = {'proprioceptive': obs[:, 2:4], 'image': torch.cat((depth, rgb), dim=1)}
                    # actions = self.policy.act(state, terminated)
                    #print(f'obs tensor shape: {obs.shape}')
                    #print(env.observation_space)
                    #obs = unflatten_tensorized_space(obs, env.observation_space)
                    #print(f"obs keys: {obs.keys()}")
                    obs = obs["policy"]
                    ##### NEW IMPLEMENTATION #####
                    depth_image = obs["depth_image"].permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
                    depth_image = torch.clip(depth_image, 0.0, 6.0)
                    #depth_image[:, 50:90, 30:130] = 0.0
                    depth_image[:, 200:] = 0.0
                    rgb = obs["rgb_image"].permute(0,3,1,2)
                    rgb[:, 200:] = 0.0
                    #rgb[:, 50:90,30:130] = 0.0
                    grayscale = rgb[:, 0] * 0.2989 + rgb[:, 1] * 0.5870 + rgb[:, 2] * 0.1140
                    grayscale = grayscale / 255
                    grayscale = grayscale.unsqueeze(1)
                    #grayscale.unsqueeze()
                    #image = torch.cat([depth, rgb], dim=1)
                    image = torch.cat([depth_image, grayscale], dim=1)
                    #image = depth_image
                    #depth_image = F.interpolate(depth_image, size=(90, 160), mode='bilinear', align_corners=False)
                    actions = obs["actions"]
                    distance_obs = obs["distance"]
                    heading_obs = obs["heading"]
                    angle_diff_obs = obs["angle_diff"]
                    #proprioceptive_obs = torch.cat((distance_obs, heading_obs, angle_diff_obs), dim=1)
                    proprioceptive_obs = torch.cat((actions, distance_obs, heading_obs, angle_diff_obs), dim=1)
                    state = {
                        'proprioceptive': proprioceptive_obs,
                        'image': image#depth_image
                    
                    }
                    actions = self.policy.act(state, terminated)

                next_obs, rewards, terminated, truncated, next_info = env.step(actions)
                # Log reward metrics if supported by the policy
                if hasattr(self.policy, "record_transitions"):
                    self.policy.record_transitions(rewards, terminated, truncated, next_info, step=timestep)
                # print(f'log keys {next_info["log"].keys()}')
                # print(f'episode keys {next_info["episode"].keys()}')
                obs, info = next_obs, next_info
