
import gymnasium as gym
import torch
import torch.nn as nn
import tqdm
from skrl.envs.torch.wrappers import Wrapper

from CloneRL.algorithms.torch.imitation_learning.base import BaseAgent
from CloneRL.collectors.torch.data_recorder import DataRecorderBase

from .base import SequentialCollectorBase


class SequentialCollectorOrbit(SequentialCollectorBase):
    def __init__(self,
                 env: Wrapper,
                 model: nn.Module,
                 recorder: DataRecorderBase,
                 predict_fn = None,
                 num_episodes: int = 1000):
        super().__init__(env, model, recorder, predict_fn, num_episodes)
        """ Collects data from the environment using the policy and the predict_fn."""

    def collect(self):
        with torch.no_grad():
            obs, info = self.env.reset()
            for episode in tqdm.tqdm(range(self.num_episodes)):
                action = self.predict_fn(self.model, obs)
                next_obs, reward, done, truncated, info = self.env.step(action)
                #print(f'obs: {obs}, action: {action}, reward: {reward}, done: {done}, truncated: {truncated}, info: {info}')
                self.recorder.append_to_buffer(obs, action, reward, done, info)
                if done.any() or truncated.any():
                    obs, info = self.env.reset()
                else:
                    obs = next_obs

    def preprocess_tensors(self, obs, actions, rewards, done, info):
        return obs, actions, rewards, done, info
