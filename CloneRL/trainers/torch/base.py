
from torch.utils.data import DataLoader

from CloneRL.algorithms.torch.imitation_learning.base import BaseAgent


class BaseTrainer():
    def __init__(self, cfg, policy, dataset, val_dataset):
        self.cfg = cfg
        self.policy: BaseAgent = policy
        #self.dataset = dataset
        self.train_ds = DataLoader(dataset, batch_size=self.cfg["batch_size"], shuffle=True, num_workers=0)
        self.train_val_ds = DataLoader(val_dataset, batch_size=self.cfg["batch_size"], shuffle=True, num_workers=0)
    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
