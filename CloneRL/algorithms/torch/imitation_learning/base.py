from typing import Optional, Union, Tuple, Dict, Any, Callable, List

import gymnasium as gym

import torch
import numpy as np

import wandb
import os
class BaseAgent:
    def __init__(
            self,
            cfg: Optional[Dict[str, Any]],
            policy: torch.nn.Module,
            device: Union[str, torch.device] = "cuda:0" if torch.cuda.is_available() else "cpu"):

        self.device = device
        self.policy = policy
        self.cfg = cfg

    def initialize(self):
        import wandb
        import time
        run_name = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        wandb.init(name=run_name)

    def train(self):
        raise NotImplementedError
    
    def save_model(self, name):
        path = f"runs/{wandb.run.project}/{wandb.run.id}/checkpoints/"
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.policy.state_dict(), path + name)
    
    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path))

    def validate(self):
        raise NotImplementedError
    
    def act(self, observation: Dict[str, torch.Tensor], deterministic: bool = False):
        raise NotImplementedError