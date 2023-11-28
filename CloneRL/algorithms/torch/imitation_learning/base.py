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
        # if self.cfg.get("expertiment", {}).get("wandb", False):
        #     import wandb
        #     wandb.init()
        import wandb
        wandb.init()

    def train(self):
        raise NotImplementedError
    
    def save_model(self, name):
        path = f"runs/CloneRL/{wandb.run.project}/{wandb.run.id}/checkpoints/"
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.policy.state_dict(), path + name)
    
    def validate(self):
        raise NotImplementedError

    # def test(self):
    #     raise NotImplementedError

    # def save_model(self):
    #     raise NotImplementedError

    # def load_model(self):
    #     raise NotImplementedError