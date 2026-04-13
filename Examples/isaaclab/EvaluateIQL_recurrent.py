"""Evaluation script for IQL with Recurrent (GRU) models.

This script loads a pre-trained IQLRecurrent model and evaluates it in the
Isaac Lab environment.
"""

import torch
import multiprocessing as mp
from typing import Optional, Sequence
import sys

from models_cai import GRUActorGaussian, GRUValue, GRUTwinQ
from CloneRL.algorithms.torch.offline_rl.iql import IQLRecurrent
from CloneRL.trainers.torch.sequential_recurrent import SequentialRecurrentTrainer as Trainer
from CloneRL.dataloader.hdf.hdf_loader_gru import HDF5RandomSequenceGRUDataset
from TrainEvalIQL_RNN import load_isaaclab_env

# Dataset path
DATA_PATH = "/home/robotlab/Documents/datasets/dataset_new2.hdf5"

def evaluate_model(checkpoint_path: str, model_name: str):
    """
    Loads a pre-trained recurrent model and evaluates it.

    Args:
        checkpoint_path: Path to the directory containing the saved model weights.
        model_name: Name of the model checkpoint file.
    """
    
    # Sequence length (should match training)
    sequence_length = 16

    # Base GRU Model configuration (shared params)
    # Note: The GRU dataloader provides:
    #   - 'image': RGB (3 channels) - used by image_encoder
    #   - 'depth': depth only (1 channel) - used by depth_encoder
    #   - 'rgb': full RGB (3 channels) - same as 'image'
    gru_base_config = {
        "image_channels": 3,      # RGB from dataloader's 'image' key
        "depth_channels": 1,      # Depth channel from dataloader's 'depth' key
        "image_size": [160, 90],  # Image dimensions [W, H]
        "proprioception_channels": 3,  # angle_diff, distance, heading
        "hidden_size": 128,       # GRU hidden state size
        "num_layers": 2,          # Number of GRU layers
        "image_encoder_features": [8, 16, 32, 64],
        "image_fc_features": [120, 60],
        "mlp_features": [512, 256, 160, 128],
        "device": "cuda:0",
    }
    
    # Actor-specific config (includes action_dim and Gaussian output params)
    actor_config = {
        **gru_base_config,
        "action_dim": 2,
        "min_log_std": -20.0,
        "max_log_std": 2.0,
        "state_independent_log_std": True,
    }
    
    # Critic-specific config (includes action_dim)
    critic_config = {
        **gru_base_config,
        "action_dim": 2,
    }
    
    # Value config (no action_dim needed)
    value_config = {
        **gru_base_config,
    }

    # Create GRU-based models
    actor = GRUActorGaussian(**actor_config)
    critic = GRUTwinQ(**critic_config)
    value = GRUValue(**value_config)

    # IQL algorithm configuration (minimal for evaluation)
    iql_config = {
        "actions_lr": 3e-4,
        "value_lr": 3e-4,
        "critic_lr": 3e-4,
        "discount": 0.99,
        "tau": 0.01,
        "expectile": 0.8,
        "temperature": 0.1,
        "target_update_freq": 1,
        "grad_clip": 1.0,
        "reset_hidden_on_done": True,
    }

    # Create IQLRecurrent agent
    agent = IQLRecurrent(
        actor_policy=actor,
        value_policy=value,
        critic_policy=critic,
        cfg=iql_config,
    )

    # Load the trained model weights
    print(f"Loading model from: {checkpoint_path}{model_name}")
    agent.load_model(checkpoint_path, model_name)

    # Create dummy datasets for trainer (required but not used during evaluation)
    dataset = HDF5RandomSequenceGRUDataset(
        file_path=DATA_PATH,
        sequence_length=sequence_length,
        min_idx=0,
        max_idx=1000,
        total_samples=1000,
        proprioceptive_keys=['angle_diff', 'distance', 'heading'],
    )
    
    dataset_val = HDF5RandomSequenceGRUDataset(
        file_path=DATA_PATH,
        sequence_length=sequence_length,
        min_idx=2000,
        max_idx=2100,
        total_samples=100,
        proprioceptive_keys=['angle_diff', 'distance', 'heading'],
    )

    # Define the trainer
    trainer = Trainer(
        cfg={},
        policy=agent,
        dataset=dataset,
        val_dataset=dataset_val,
    )

    # Load the environment and evaluate
    env = load_isaaclab_env(task_name="AAURoverEnvRGBDRaw-v0")
    trainer.evaluate(env, num_steps=1000)


if __name__ == "__main__":
    # Set the start method for multiprocessing
    mp.set_start_method('spawn', force=True)

    # Define the path to the checkpoint
    checkpoint_path = "runs/CloneLab-Examples_isaaclab/IQL_RGBD_88/checkpoints/"
    model_name = "final_model.pt"
    #model_name = "final_model.pt"

    # Run the evaluation
    evaluate_model(checkpoint_path, model_name=model_name)
