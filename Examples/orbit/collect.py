import math
import os

import gymnasium as gym
import numpy as np
import rover_envs.envs
from models import GaussianNeuralNetwork
from omni.isaac.orbit.app import AppLauncher
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaac_orbit_env
from skrl.envs.wrappers.torch import wrap_env
from CloneRL.collectors.torch import SequentialCollectorOrbit
from CloneRL.collectors.torch.data_recorder import HDF5DataRecorder
from CloneRL.utils import skrl_get_actions
# import rover_envs.envs.navigation.robots  # noqa F401


def main():
    task_name = "AAURoverEnvCamera-v0"
    env = load_isaac_orbit_env(task_name)
    env = wrap_env(env, wrapper="isaac-orbit")
    print(env.observation_space)
    print(env.action_space)

    num_obs = env.unwrapped.observation_manager.group_obs_dim["policy"][0]
    num_actions = env.unwrapped.action_manager.action_term_dim[0]
    observation_space = gym.spaces.Box(
        low=-math.inf, high=math.inf, shape=(num_obs,))
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actions,))
    # Create models (skrl specific)
    models = {}
    models["policy"] = GaussianNeuralNetwork(observation_space=observation_space,
                                             action_space=action_space,
                                             device=env.device,
                                             encoder_input_size=env.observation_manager.group_obs_term_dim["policy"][-1][0]).cuda()

    # Create agent, load weights and initialize (skrl specific)
    agent = PPO(models=models,
                memory=None,
                cfg=PPO_DEFAULT_CONFIG.copy(),
                observation_space=observation_space,
                action_space=action_space,
                device=env.device)

    # path_to_weights = "data/best_agents/agent1_nov26/checkpoints/best_agent.pt"
    path_to_weights = gym.spec(task_name).kwargs.pop("best_model_path")
    agent.load(path_to_weights)
    agent.init()

    # Define path to HDF5 file for recording
    file_path = "/media/anton/T7 Shield/University/1. Master/Datasets/1. Simulation/dataset4_180_320/with_rgb_and_depth"

    # Here we define the extras that we want to record, can be omitted if not needed
    extras = {
        "rgb": {"shape": (180, 320, 3), "dtype": np.uint8},
        "depth": {"shape": (180, 320), "dtype": np.float32},
    }

    # Define the recorder as HDF5DataRecorder
    recorder = HDF5DataRecorder(
        base_filename=file_path,
        num_obs=num_obs,
        num_actions=num_actions,
        num_envs=4,
        env=env,
        max_rows=125000,
        extras=extras)

    # Define the collector as SequentialCollectorOrbit
    collector = SequentialCollectorOrbit(
        env=env,
        model=agent,
        recorder=recorder,
        predict_fn=skrl_get_actions,
        num_episodes=33000)

    # Start collecting data
    collector.collect()

if __name__ == "__main__":
    main()
