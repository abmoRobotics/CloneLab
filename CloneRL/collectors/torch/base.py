
import gymnasium as gym

from CloneRL.algorithms.torch.imitation_learning.base import BaseAgent

from CloneRL.collectors.torch.data_recorder import DataRecorderBase

import torch.nn as nn

class SequentialCollectorBase:
    def __init__(self, 
                 env: gym.Env,
                 model: nn.Module,
                 recorder: DataRecorderBase,
                 predict_fn = None,
                 num_episodes: int = 1000):
        self.env = env
        self.model = model
        self.recorder = recorder
        self.predict_fn = predict_fn
        self.num_episodes = num_episodes

    def collect(self):
        """ Collects data from the environment using the policy and the predict_fn."""
        raise NotImplementedError("This method should be implemented in a subclass.")

