import torch
from models import actor_gaussian_image, TwinQ_image, v_image
from CloneRL.algorithms.torch.offline_rl.iql import IQL
from CloneRL.trainers.torch.sequential import SequentialTrainer as Trainer
import multiprocessing as mp
from TrainEvalIQL import load_isaaclab_env
from CloneRL.dataloader.hdf.hdf_loader import HDF5DictDataset
data = "/home/robotlab/Documents/datasets/dataset_new2.hdf5"
data2 = "/home/robotlab/Documents/datasets/dataset_new2_smallerer.hdf5"
def evaluate_model(checkpoint_path, model_name):
    """
    Loads a pre-trained model and evaluates it.

    :param checkpoint_path: Path to the directory containing the saved model weights.
    """

    model_config2 = {
        "proprioception_channels": 3,
        "image_channels": 2,
        "action_dim": 2,
        "mlp_features": [512, 256, 128, 64],
        "image_input_dim": [160, 90],
        "image_encoder_features": [8, 16, 32, 64],
        "image_fc_features": [120, 60],
        "activation": "leaky_relu",
        "dropout_rate": 0,
        "use_batch_norm": False
    }


    model_config = {
        "proprioception_channels": 3,
        "image_channels": 2,
        "action_dim": 2,
        "mlp_features": [256, 160, 128],
        "image_input_dim": [160, 90],
        "image_encoder_features": [8, 16, 32, 64],
        "image_fc_features": [120, 60],
        "activation": "leaky_relu",
        "dropout_rate": 0,
        "use_batch_norm": False
    }

    # Define model
    actor = actor_gaussian_image(**model_config2).to("cuda:0")
    critic = TwinQ_image(**model_config).to("cuda:0")
    value = v_image(**model_config).to("cuda:0")

    # Choose the algorithm
    agent = IQL(actor_policy=actor,
                value_policy=value,
                critic_policy=critic,
                cfg={})

    # Load the trained model weights
    agent.load_model(checkpoint_path, model_name)
    
    dataset = HDF5DictDataset(data, min_idx=0, max_idx=1000)
    dataset_val = HDF5DictDataset(data, min_idx=2000, max_idx=2100)
    # Define the trainer
    trainer = Trainer(cfg={}, policy=agent, dataset=dataset, val_dataset=dataset_val)

    # Evaluate the model
    env = load_isaaclab_env(task_name="AAURoverEnvRGBDRaw-v0")
    #env = load_isaaclab_env(task_name="AAURoverEnvRGBDRawTemp-v0")
    trainer.evaluate(env, num_steps=1000)


if __name__ == "__main__":
    # Set the start method for multiprocessing
    mp.set_start_method('spawn', force=True)

    # Define the path to the checkpoint
    # Make sure to replace this with the actual path to your checkpoint
    checkpoint_path = "runs/CloneLab-Examples_isaaclab/2025-08-19_17-35-24/checkpoints/"
    checkpoint_path = "runs/CloneLab-Examples_orbit/2025-08-12_10-12-20/checkpoints/" # old random
    checkpoint_path = "runs/CloneLab-Examples_isaaclab/2025-09-12_10-11-58/checkpoints/" # New camera position
    checkpoint_path = "runs/CloneLab-Examples_isaaclab/2025-11-21_15-22-57/checkpoints/" # ~80 % SR after 1000 steps 120 envs
    checkpoint_path = "runs/CloneLab-Examples_isaaclab/2025-11-21_21-23-38/checkpoints/" # ~81.5 % SR after 1000 steps 120 envs
    checkpoint_path = "runs/CloneLab-Examples_isaaclab/2025-11-22_18-24-00/checkpoints/" # ~78.75 % SR after 1000 steps 120 envs
    model_name = "best_model_8.pt"
    model_name = "best_model_9.pt"
    model_name = "best_model_epoch_9.pt"
    model_name = "best_model_epoch_19.pt" # ~80% SR
    model_name = "best_model_epoch_19.pt" # ~81.5% SR
    model_name = "best_model_epoch_19.pt" # ~78.75% SR


    # Run the evaluation
    evaluate_model(checkpoint_path, model_name=model_name)
