
from CloneRL.algorithms.base import BaseAgent

from torch.utils.data import DataLoader

class BaseTrainer():
    def __init__(self, cfg, policy, dataset):
        self.cfg = cfg
        self.policy: BaseAgent = policy
        #self.dataset = dataset
        self.train_ds = DataLoader(dataset, batch_size=self.cfg["batch_size"], shuffle=False, num_workers=0)

    def train(self):
        raise NotImplementedError
    
    def evaluate(self):
        raise NotImplementedError
    