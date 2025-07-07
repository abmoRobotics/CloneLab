
import torch
from models import actor_gaussian_image, TwinQ_image, v_image
from CloneRL.algorithms.torch.offline_rl.iql import IQL
from CloneRL.trainers.torch.sequential import SequentialTrainer as Trainer
import multiprocessing as mp
from TrainEvalIQLtest import load_isaaclab_env
from CloneRL.dataloader.hdf.hdf_loader import HDF5DictDataset
data = "/media/anton/T7 Shield/University/PHD/rover_simulation_datasets/dataset_new2.hdf5"
def evaluate_model(checkpoint_path, model_name):
    """
    Loads a pre-trained model and evaluates it.

    :param checkpoint_path: Path to the directory containing the saved model weights.
    """
    # Define model
    actor = actor_gaussian_image(proprioception_channels=3).to("cuda:0")
    critic = TwinQ_image(proprioception_channels=3).to("cuda:0")
    value = v_image(proprioception_channels=3).to("cuda:0")

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
    trainer.evaluate(env, num_steps=1000000)


if __name__ == "__main__":
    # Set the start method for multiprocessing
    mp.set_start_method('spawn', force=True)

    # Define the path to the checkpoint
    # Make sure to replace this with the actual path to your checkpoint
    checkpoint_path = "runs/CloneLab-Examples_orbit/run_1/checkpoints/"
    model_name = "best_model_9.pt"

    # Run the evaluation
    evaluate_model(checkpoint_path, model_name=model_name)
