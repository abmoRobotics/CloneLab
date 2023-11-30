from typing import Dict

from CloneRL.algorithms.torch.imitation_learning.base import BaseAgent
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
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.scaler = torch.cuda.amp.GradScaler()
    
        self.initialize()

    def train(self, data: Dict[str, torch.Tensor]):
        
        def compute_loss(target, actions) -> torch.Tensor:
            #print(f'target dtype: {target.dtype}, actions dtype: {actions.dtype}')
            return self.loss_fn(actions, target.float())

        
        #observations = data['observation']
        target = data['actions'].to(self.device)
        
        actions = self.policy(data)

        loss = compute_loss(target, actions)


        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
    
        return loss.item()
        # print(loss.item())

    def validate(self, data_val: Dict[str, torch.Tensor]):
        average_loss = 0
        with torch.no_grad():
            for data in data_val:

                target = data['actions'].to(self.device)
                
                actions = self.policy(data)

                loss = self.loss_fn(actions, target.float())
                average_loss += loss.item()
            return average_loss / len(data_val)



    def evaluate(self, env, num_episodes=1000):
      pass

    def act(self, observation: Dict[str, torch.Tensor], deterministic: bool = False):
        pass


    