from typing import Dict

from CloneRL.algorithms.base import BaseAgent
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


BC_DEFAULT_CONFIG = {
}

class BehaviourCloning(BaseAgent):
    
    def __init__(self, policy, cfg, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        super().__init__(cfg, policy, device=device)
        self.policy = policy
        self.loss_fn = nn.MSELoss(reduction="mean")
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scaler = torch.cuda.amp.GradScaler()
    
        self.initialize()

    def train(self, data: Dict[str, torch.Tensor]):
        
        def compute_loss(target, actions) -> torch.Tensor:
            return self.loss_fn(actions, target)

        
        #observations = data['observation']
        target = data['action'].to(self.device)
        
        actions = self.policy(data)

        loss = compute_loss(target, actions)


        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
    
        return loss.item()
        # print(loss.item())





    