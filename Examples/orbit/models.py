from typing import Dict

import torch
import torch.nn as nn
from skrl.models.torch.base import Model as BaseModel
from skrl.models.torch.gaussian import GaussianMixin


def get_activation(activation_name):
    activation_fns = {
        "leaky_relu": nn.LeakyReLU(),
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "elu": nn.ELU(),
        "relu6": nn.ReLU6(),
        "selu": nn.SELU(),
    }
    if activation_name not in activation_fns:
        raise ValueError(f"Activation function {activation_name} not supported.")
    return activation_fns[activation_name]

class HeightmapEncoder(nn.Module):
    def __init__(self, in_channels, encoder_features=[80, 60], encoder_activation="leaky_relu"):
        super().__init__()
        self.encoder_layers = nn.ModuleList()
        for feature in encoder_features:
            self.encoder_layers.append(nn.Linear(in_channels, feature))
            self.encoder_layers.append(get_activation(encoder_activation))
            in_channels = feature

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x

class GaussianNeuralNetwork(GaussianMixin, BaseModel):
    """Gaussian neural network model."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        encoder_features=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the Gaussian neural network model.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_features (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        BaseModel.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=True, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, reduction="sum")

        self.proprioception_channels = 4
        self.dense_channels = 634
        self.sparse_channels = 1112

        self.dense_encoder = HeightmapEncoder(self.dense_channels, encoder_features, encoder_activation)
        self.sparse_encoder = HeightmapEncoder(self.sparse_channels, encoder_features, encoder_activation)

        self.mlp = nn.ModuleList()


        in_channels = self.proprioception_channels + encoder_features[-1] + encoder_features[-1]
        action_space = action_space.shape[0]
        mlp_features = [256, 160, 128]
        for feature in mlp_features:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(encoder_activation))
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, action_space))
        self.mlp.append(nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(action_space))


    def compute(self, states, role="actor"):
        dense_start = self.proprioception_channels
        dense_end = dense_start + self.dense_channels
        sparse_end = dense_end + self.sparse_channels
        x = states["states"]
        x0 = x[:, 0:4]
        x1 = self.dense_encoder(x[:, dense_start:dense_end])
        x2 = self.sparse_encoder(x[:, dense_end:sparse_end])

        x = torch.cat([x0, x1, x2], dim=1)
        for layer in self.mlp:
            x = layer(x)

        return x, self.log_std_parameter, {}


import torch.nn.functional as F


class HeightmapEncoder(nn.Module):
    def __init__(self, in_channels, encoder_features=[80, 60], encoder_activation="leaky_relu"):
        super().__init__()
        self.encoder_layers = nn.ModuleList()
        for feature in encoder_features:
            self.encoder_layers.append(nn.Linear(in_channels, feature))
            self.encoder_layers.append(nn.LeakyReLU())
            in_channels = feature

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x

class DepthImageEncoder(nn.Module):
    def __init__(self):
        super(DepthImageEncoder, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)

        # Define the fully connected layers
        self.fc1 = nn.Linear(64 * 12 * 20, 120)
        self.fc2 = nn.Linear(120, 60)

    def forward(self, x):
        # Apply convolutional layers with ReLU and max pooling
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Apply the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DepthPolicy(nn.Module):
    def __init__(self, device = "cuda:0" if torch.cuda.is_available() else "cpu"):
        super(DepthPolicy, self).__init__()
        self.device = device
        self.proprioception_channels = 4
        self.dense_channels = 60
        self.sparse_channels = 120

        self.dense_encoder = DepthImageEncoder()

        self.mlp = nn.ModuleList()


        in_channels = self.proprioception_channels + 60
        #action_space = action_space.shape[0]
        mlp_features = [256, 160, 128]
        for feature in mlp_features:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(nn.LeakyReLU())
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 2))
        self.mlp.append(nn.Tanh())

    def forward(self, x: Dict[str, torch.Tensor]):
        x0 = x["observations"].to(self.device)[:, 0:4]
        #depth = x["depth"].to(self.device).reshape(-1, 1, 90, 160)
        # Change inf to 100

        if "depth" in x.keys():
            depth = x["depth"].to(self.device).reshape(-1, 1, 90, 160)
        elif "depth" in x["extras"].keys():
            depth2 = x["extras"]["depth"].detach().clone().unsqueeze(1)
            depth3 = depth2.detach().clone().float()
            depth = depth3
        else:
            depth = torch.zeros((x0.shape[0], 1, 90, 160)).to(self.device)

        depth = torch.where(torch.isinf(depth), torch.full_like(depth, 100), depth)
        x0 = x0[:, 0:4]
        x1 = self.dense_encoder(depth)

        x = torch.cat([x0, x1], dim=1)
        for layer in self.mlp:
            x = layer(x)

        return x

class MlpPolicy(nn.Module):
    def __init__(self, device = "cuda:0" if torch.cuda.is_available() else "cpu"):
        super(MlpPolicy, self).__init__()
        self.device = device
        self.proprioception_channels = 4
        self.dense_channels = 634
        self.sparse_channels = 1112

        self.dense_encoder = HeightmapEncoder(self.dense_channels)
        self.sparse_encoder = HeightmapEncoder(self.sparse_channels)

        self.mlp = nn.ModuleList()


        in_channels = self.proprioception_channels + 120
        #action_space = action_space.shape[0]
        mlp_features = [256, 160, 128]
        for feature in mlp_features:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(nn.LeakyReLU())
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 2))
        self.mlp.append(nn.Tanh())

    def forward(self, x: Dict[str, torch.Tensor]):
        x = x["observations"].to(self.device)

        dense_start = self.proprioception_channels
        dense_end = dense_start + self.dense_channels
        sparse_end = dense_end + self.sparse_channels
        x0 = x[:, 0:4]
        x1 = self.dense_encoder(x[:, dense_start:dense_end])
        x2 = self.sparse_encoder(x[:, dense_end:sparse_end])

        x = torch.cat([x0, x1, x2], dim=1)
        for layer in self.mlp:
            x = layer(x)

        return x
