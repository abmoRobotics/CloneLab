
import gymnasium as gym
import torch.nn as nn
import tqdm

from CloneRL.algorithms.torch.imitation_learning.base import BaseAgent
from CloneRL.collectors.torch.data_recorder import DataRecorderBase

from .base import SequentialCollectorBase


class SequentialCollectorGym(SequentialCollectorBase):
    def __init__(self,
                 env: gym.Env,
                 model: nn.Module,
                 recorder: DataRecorderBase,
                 predict_fn = None,
                 num_episodes: int = 1000):
        super().__init__(env, model, recorder, predict_fn, num_episodes)
        """ Collects data from the environment using the policy and the predict_fn."""

    def collect(self):
        obs, info = self.env.reset()
        for episode in tqdm.tqdm(range(self.num_episodes)):
            action = self.predict_fn(self.model, obs)
            next_obs, reward, done, truncated, info = self.env.step(action)
            self.recorder.append_to_buffer([obs], [[action]], [[reward]], [[done]], [info])
            if done:
                obs, info = self.env.reset()
            else:
                obs = next_obs
