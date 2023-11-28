from typing import Dict

import h5py

import numpy as np

from CloneRL.dataloader.hdf import HDF5Dataset, HDF_DEFAULT_IL_MAPPER, HDF_DEFAULT_ORL_MAPPER
from CloneRL.algorithms.torch.bc import BehaviourCloning as BC
from CloneRL.trainers.torch.sequential import SequentialTrainer as Trainer

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
mock_h5_file_path = 'data/mock_data.h5'

# Create mock data for each dataset
observations = np.random.rand(100, 4).astype(float)  # 100 observations of size 4
actions = np.random.rand(100, 1).astype(float)  # 100 actions of size 1
rewards = np.arange(100)  # 100 rewards
next_observations = np.random.rand(100, 4)  # 100 next observations of size 4
dones = np.random.randint(0, 2, size=(100, 1))  # 100 done flags (0 or 1)

class policy(nn.Module):
    def __init__(self, device = "cuda:0" if torch.cuda.is_available() else "cpu"):
        super(policy, self).__init__()
        self.device = device
        self.policy = nn.Sequential(
            nn.Linear(4, 32,dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(32, 32, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(32, 1, dtype=torch.float64),
            nn.Sigmoid()
        )

    def forward(self, x: Dict[str, torch.Tensor]):
        x = x["observation"].to(self.device)
        for layer in self.policy:
            x = layer(x)
        return x



# Write the mock data to an HDF5 file
with h5py.File(mock_h5_file_path, 'w') as h5file:
    h5file.create_dataset('observation', data=observations)
    h5file.create_dataset('action', data=actions)
    h5file.create_dataset('reward', data=rewards)
    h5file.create_dataset('next_observation', data=next_observations)
    h5file.create_dataset('done', data=dones)

dataset = HDF5Dataset(mock_h5_file_path, HDF_DEFAULT_ORL_MAPPER)

loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

model = policy().to("cuda:0")


agent = BC(model = model, cfg={})

trainer = Trainer(cfg={"batch_size": 30}, policy=agent, dataset=dataset)

trainer.train(epoch=100)




#print(loader)
# for i, data in enumerate(loader):
#     print(data["reward"])
# for i in range(100):
#     data_item = dataset[i]


# dataset.close()