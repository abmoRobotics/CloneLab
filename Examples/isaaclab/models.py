from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        raise ValueError(
            f"Activation function {activation_name} not supported.")
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
        mlp_input_size=4,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=None,
        encoder_layers=[80, 60],
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
        GaussianMixin.__init__(
            self, clip_actions=True, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, reduction="sum"
        )

        self.mlp_input_size = mlp_input_size
        self.encoder_input_size = encoder_input_size

        in_channels = self.mlp_input_size
        if self.encoder_input_size is not None:
            self.dense_encoder = HeightmapEncoder(
                self.encoder_input_size, encoder_layers, encoder_activation)
            in_channels += encoder_layers[-1]

        self.mlp = nn.ModuleList()

        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        action_space = action_space.shape[0]
        self.mlp.append(nn.Linear(in_channels, action_space))
        self.mlp.append(nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(action_space))

    def compute(self, states, role="actor"):
        # Split the states into proprioception and heightmap if the heightmap is used.
        if self.encoder_input_size is None:
            x = states["states"]
        else:
            x = states["states"][:, 0:self.mlp_input_size]
            encoder_output = self.dense_encoder(
                states["states"][:, self.mlp_input_size - 1:-1])
            x = torch.cat([x, encoder_output], dim=1)

        # Compute the output of the MLP.
        for layer in self.mlp:
            x = layer(x)

        return x, self.log_std_parameter, {}

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
    def __init__(self, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        super(DepthPolicy, self).__init__()
        self.device = device
        self.proprioception_channels = 4
        self.dense_channels = 60
        self.sparse_channels = 120

        self.dense_encoder = DepthImageEncoder()

        self.mlp = nn.ModuleList()

        in_channels = self.proprioception_channels + 60
        # action_space = action_space.shape[0]
        mlp_features = [256, 160, 128]
        for feature in mlp_features:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(nn.LeakyReLU())
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 2))
        self.mlp.append(nn.Tanh())

    def forward(self, x: Dict[str, torch.Tensor]):
        x0 = x["observations"].to(self.device)[:, 0:4]

        if "depth" in x.keys():
            depth = x["depth"].to(self.device).reshape(-1, 1, 180, 360)
        elif "depth" in x["extras"].keys():
            depth2 = x["extras"]["depth"].detach().clone().unsqueeze(1)
            depth3 = depth2.detach().clone().float()
            depth = depth3
        else:
            depth = torch.zeros((x0.shape[0], 1, 180, 360)).to(self.device)

        depth = torch.where(torch.isinf(
            depth), torch.full_like(depth, 100), depth)
        x0 = x0[:, 0:4]
        x1 = self.dense_encoder(depth)

        x = torch.cat([x0, x1], dim=1)
        for layer in self.mlp:
            x = layer(x)

        return x


class GaussianDepthNetwork(nn.Module):
    def __init__(self, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        super(GaussianDepthNetwork, self).__init__()
        self.device = device
        self.proprioception_channels = 4
        self.dense_channels = 60
        self.sparse_channels = 120

        self.dense_encoder = DepthImageEncoder()

        self.mlp = nn.ModuleList()

        in_channels = self.proprioception_channels + 60
        # action_space = action_space.shape[0]
        mlp_features = [256, 160, 128]
        for feature in mlp_features:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(nn.LeakyReLU())
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 2))
        self.mlp.append(nn.Tanh())
        self.log_std = nn.Parameter(torch.zeros(2))

    def forward(self, x: Dict[str, torch.Tensor]):
        x0 = x["observations"].to(self.device)[:, 0:4]

        if "depth" in x.keys():
            depth = x["depth"].to(self.device).reshape(-1, 1, 160, 90)
        elif "depth" in x["extras"].keys():
            depth2 = x["extras"]["depth"].detach().clone().unsqueeze(1)
            depth3 = depth2.detach().clone().float()
            depth = depth3
        else:
            depth = torch.zeros((x0.shape[0], 1, 90, 160)).to(self.device)

        depth = torch.where(torch.isinf(
            depth), torch.full_like(depth, 100), depth)
        x0 = x0[:, 0:4]
        x1 = self.dense_encoder(depth)

        x = torch.cat([x0, x1], dim=1)
        for layer in self.mlp:
            x = layer(x)

        mu = x

        log_std = torch.clamp(self.log_std, min=-20, max=2)
        return torch.distributions.Normal(mu, torch.exp(log_std))


class actor_gaussian_image(nn.Module):
    def __init__(self, device="cuda:0" if torch.cuda.is_available() else "cpu", proprioception_channels=3, encoder_channels=60, image_channels=1):
        super(actor_gaussian_image, self).__init__()
        self.device = device
        self.proprioception_channels = proprioception_channels
        self.encoder_channels = encoder_channels

        self.encoder = conv_encoder(in_channels=image_channels)

        self.mlp = nn.ModuleList()

        in_channels = self.proprioception_channels + self.encoder_channels
        # action_space = action_space.shape[0]
        mlp_features = [256, 160, 128]
        for feature in mlp_features:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(nn.LeakyReLU())
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 2))
        self.mlp.append(nn.Tanh())
        self.log_std = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def forward(self, state: Dict[str, torch.Tensor]):
        x = self.encoder(state["image"])
        is_sequence = x.ndim == 3
        if is_sequence:
            x = torch.cat([state["proprioceptive"][..., 2:5], x], dim=2)
        else:
            x = torch.cat([state["proprioceptive"][..., 2:5], x], dim=1)
        #x = torch.cat([state["proprioceptive"][:, 2:4], x], dim=1)
        for layer in self.mlp:
            x = layer(x)
        mu = x
        log_std = torch.clamp(self.log_std, min=-20, max=2)
        return torch.distributions.Normal(mu, log_std.exp())


class actor_deterministic_image(nn.Module):
    def __init__(self, device="cuda:0" if torch.cuda.is_available() else "cpu", proprioception_channels=3, encoder_channels=60, image_channels=4):
        super(actor_deterministic_image, self).__init__()
        self.device = device
        self.proprioception_channels = proprioception_channels
        self.encoder_channels = encoder_channels

        self.encoder = conv_encoder(in_channels=image_channels)

        self.mlp = nn.ModuleList()

        in_channels = self.proprioception_channels + self.encoder_channels
        # action_space = action_space.shape[0]
        mlp_features = [256, 160, 128]
        for feature in mlp_features:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(nn.LeakyReLU())
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 2))
        self.mlp.append(nn.Tanh())

    def forward(self, state: Dict[str, torch.Tensor]):
        x = self.encoder(state["image"])
        is_sequence = x.ndim == 3
        if is_sequence:
            x = torch.cat([state["proprioceptive"][..., 2:5], x], dim=2)
        else:
            x = torch.cat([state["proprioceptive"][..., 2:5], x], dim=1)
        #x = torch.cat([state["proprioceptive"][..., 2:5], x], dim=1)
        for layer in self.mlp:
            x = layer(x)
        return x


class actor_deterministic(nn.Module):
    def __init__(self, device="cuda:0" if torch.cuda.is_available() else "cpu", proprioception_channels=4, encoder_channels=60, image_channels=1):
        super(actor_deterministic, self).__init__()
        self.device = device
        self.proprioception_channels = proprioception_channels

        self.mlp = nn.ModuleList()

        in_channels = self.proprioception_channels
        # action_space = action_space.shape[0]
        mlp_features = [256, 160, 128]
        for feature in mlp_features:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(nn.LeakyReLU())
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 2))
        self.mlp.append(nn.Tanh())

    def forward(self, state: Dict[str, torch.Tensor]):
        x = state["proprioceptive"][:, :4]
        for layer in self.mlp:
            x = layer(x)
        return x


class actor_gaussian(nn.Module):
    def __init__(self, device="cuda:0" if torch.cuda.is_available() else "cpu", proprioception_channels=3, encoder_channels=60, image_channels=1):
        super(actor_gaussian, self).__init__()
        self.device = device
        self.proprioception_channels = proprioception_channels

        self.mlp = nn.ModuleList()

        in_channels = self.proprioception_channels
        # action_space = action_space.shape[0]
        mlp_features = [256, 160, 128]
        for feature in mlp_features:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(nn.LeakyReLU())
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 2))
        self.mlp.append(nn.Tanh())
        self.log_std = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def forward(self, state: Dict[str, torch.Tensor]):
        x = state["proprioceptive"][:, 2:4]
        for layer in self.mlp:
            x = layer(x)
        mu = x
        log_std = torch.clamp(self.log_std, min=-20, max=2)
        return torch.distributions.Normal(mu, log_std.exp())


class conv_encoder(nn.Module):
    def __init__(self, in_channels, input_dim=[224, 224], encoder_features=[8, 16, 32, 64], fc_features=[120, 60], encoder_activation="leaky_relu"):
        super().__init__()
        padding = 1
        stride = 1
        kernel_size = 3
        self.encoder_layers = nn.ModuleList()
        for feature in encoder_features:
            self.encoder_layers.append(
                nn.Conv2d(in_channels, feature, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
            self.encoder_layers.append(nn.BatchNorm2d(feature))
            self.encoder_layers.append(nn.LeakyReLU(inplace=True))
            self.encoder_layers.append(nn.Conv2d(feature, feature, kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=False))
            self.encoder_layers.append(nn.BatchNorm2d(feature))
            self.encoder_layers.append(nn.LeakyReLU(inplace=True))
            self.encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feature

        self.fc_layers = nn.ModuleList()

        flattened_features = input_dim

        for _ in encoder_features:
            width = (flattened_features[0] - kernel_size + 2 * padding) // stride + 1
            height = (flattened_features[1] - kernel_size + 2 * padding) // stride + 1
            width = (width - 2) // 2 + 1
            height = (height - 2) // 2 + 1
            flattened_features = [width, height]

        in_channels = flattened_features[0] * flattened_features[1] * in_channels
        # Connect the convolutional layers to a fully connected layer
        for feature in fc_features:
            self.fc_layers.append(nn.Linear(in_channels, feature))
            self.fc_layers.append(nn.LeakyReLU())
            in_channels = feature

    def forward(self, x: torch.Tensor):

        is_sequence = x.ndim == 5 # Check if input is a sequence (batch_size, seq_len, channels, height, width)

        if is_sequence:
            # Input shape will be (B, T, C, H, W)
            batch_size, seq_len, channels, height, width = x.shape
            # Reshape to (B * T, C, H, W) for convolutional layers
            x = x.view(batch_size * seq_len, channels, height, width)

        # Apply the convolutional layers
        for layer in self.encoder_layers:
            x = layer(x)

        # Flatten the output for the fully connected layers
        x = x.reshape(x.size(0), -1)

        # Apply the fully connected layers
        for layer in self.fc_layers:
            x = layer(x)

        if is_sequence:
            # Reshape back to (B, T, features) if input was a sequence
            final_features = x.shape[1]
            x = x.view(batch_size, seq_len, final_features)
        return x


class v_image(nn.Module):
    def __init__(self, device="cuda:0" if torch.cuda.is_available() else "cpu", proprioception_channels=3, encoder_channels=60, image_channels=1):
        super(v_image, self).__init__()
        self.device = device
        self.proprioception_channels = proprioception_channels
        self.encoder_channels = encoder_channels

        self.encoder = conv_encoder(in_channels=image_channels)

        self.mlp = nn.ModuleList()

        in_channels = self.proprioception_channels + self.encoder_channels
        # action_space = action_space.shape[0]
        mlp_features = [256, 160, 128]
        for feature in mlp_features:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(nn.LeakyReLU())
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 1))

    def forward(self, state: Dict[str, torch.Tensor]):
        x = self.encoder(state["image"])
        is_sequence = x.ndim == 3
        if is_sequence:
            x = torch.cat([state["proprioceptive"][..., 2:5], x], dim=2)
        else:
            x = torch.cat([state["proprioceptive"][..., 2:5], x], dim=1)
        #x = torch.cat([state["proprioceptive"], x], dim=1)
        for layer in self.mlp:
            x = layer(x)

        return x

        x0 = x["observations"].to(self.device)[:, 0:4]

        if "depth" in x.keys():
            depth = x["depth"].to(self.device).reshape(-1, 1, 160, 90)
        elif "depth" in x["extras"].keys():
            depth2 = x["extras"]["depth"].detach().clone().unsqueeze(1)
            depth3 = depth2.detach().clone().float()
            depth = depth3
        else:
            depth = torch.zeros((x0.shape[0], 1, 90, 160)).to(self.device)

        depth = torch.where(torch.isinf(depth), torch.full_like(depth, 100), depth)

        x0 = x0[:, 0:4]
        x1 = self.encoder(depth)

        x = torch.cat([x0, x1], dim=1)
        for layer in self.mlp:
            x = layer(x)

        return x

class q_image(nn.Module):
    def __init__(self, device="cuda:0" if torch.cuda.is_available() else "cpu", proprioception_channels=3, encoder_channels=60, image_channels=1):
        super(q_image, self).__init__()
        self.device = device
        self.proprioception_channels = proprioception_channels
        self.encoder_channels = encoder_channels

        self.encoder = conv_encoder(in_channels=image_channels)

        self.mlp = nn.ModuleList()

        in_channels = self.proprioception_channels + self.encoder_channels
        # action_space = action_space.shape[0]
        mlp_features = [256, 160, 128]
        for feature in mlp_features:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(nn.LeakyReLU())
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 1))

    def forward(self, state: Dict[str, torch.Tensor], action: torch.Tensor):

        x = self.encoder(state["image"])
        is_sequence = x.ndim == 3
        if is_sequence:
            x = torch.cat([state["proprioceptive"][..., 2:5], x], dim=2)
        else:
            x = torch.cat([state["proprioceptive"][..., 2:5], x], dim=1)
        #x = torch.cat([state["proprioceptive"], x], dim=1)
        for layer in self.mlp:
            x = layer(x)

        return x


class TwinQ_image(nn.Module):
    def __init__(self, device="cuda:0" if torch.cuda.is_available() else "cpu", proprioception_channels=3, encoder_channels=60, image_channels=1):
        super(TwinQ_image, self).__init__()
        self.device = device
        self.q1 = q_image(proprioception_channels=proprioception_channels, encoder_channels=encoder_channels, image_channels=image_channels).to(device)
        self.q2 = q_image(proprioception_channels=proprioception_channels, encoder_channels=encoder_channels, image_channels=image_channels).to(device)

    def forward(self, state: Dict[str, torch.Tensor], action: torch.Tensor):
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        return q1, q2


class MlpPolicy(nn.Module):
    def __init__(self, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        super(MlpPolicy, self).__init__()
        self.device = device
        self.proprioception_channels = 4
        self.dense_channels = 634
        self.sparse_channels = 1112

        self.dense_encoder = HeightmapEncoder(self.dense_channels)
        self.sparse_encoder = HeightmapEncoder(self.sparse_channels)

        self.mlp = nn.ModuleList()
        
        in_channels = self.proprioception_channels + 120
        # action_space = action_space.shape[0]
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

class GRUActor(nn.Module):
    def __init__(self, image_channels=4, image_size=[90, 160], proprioception_channels=3, encoder_channels=60, device=None):
        super(GRUActor, self).__init__()
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        # Assuming conv_encoder is defined elsewhere
        self.encoder = conv_encoder(in_channels=image_channels, input_dim=image_size)
        self.hidden_size = 80
        self.num_layers = 2
        self.hidden_val = None
        self.gru = nn.GRU(input_size=proprioception_channels + encoder_channels,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)

        # Output layers
        mlp_features = [256, 160, 128]
        self.mlp = nn.ModuleList()
        in_features = self.hidden_size
        for out_features in mlp_features:
            self.mlp.append(nn.Linear(in_features, out_features))
            in_features = out_features

        self.mlp.append(nn.Linear(in_features, 2))
        self.mlp.append(nn.Tanh())

    def forward(self, state, hidden=None):
        # print(f'state shape: {state["image"].shape}')
        x_img = state["image"]
        batch, seq_len, channels, height, width = x_img.shape
        x_img = x_img.view(batch * seq_len, channels, height, width)
        x_img = self.encoder(x_img)
        x_img = x_img.view(batch, seq_len, -1)
        x = torch.cat([state["proprioceptive"][:, :, 2:4], x_img], dim=2)  # Combine image and proprioceptive inputs-
        if hidden is None:
            hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, hidden = self.gru(x, hidden)  # OUT: (batch, seq, feature)
        # out = out[:, -1, :]  # Get the last output
        results = []
        for t in range(seq_len):
            step_out = out[:, t, :]
            for layer in self.mlp:
                step_out = layer(step_out)
            results.append(step_out)

        # for layer in self.mlp:
        #     out = layer(out)
        out = torch.stack(results, dim=1)
        return out

    def validate(self, state, done):
        with torch.no_grad():
            x_img = state["image"]
            if x_img.dim() == 4:  # Same check as in forward
                x_img = x_img.unsqueeze(1)
            batch, seq_len, channels, height, width = x_img.shape

            x_img = x_img.view(batch * seq_len, channels, height, width)
            x_img = self.encoder(x_img)
            x_img = x_img.view(batch, seq_len, -1)
            x_proprioceptive = state["proprioceptive"]
            if x_proprioceptive.dim() == 2:
                x_proprioceptive = x_proprioceptive.unsqueeze(1)
            x = torch.cat([x_proprioceptive[:, :, 2:4], x_img], dim=2)
            if self.hidden_val is None:
                self.hidden_val = torch.zeros(self.num_layers, batch, self.hidden_size).to(self.device)

            out, new_hidden = self.gru(x, self.hidden_val)
            not_done = ~done
            self.hidden_val = new_hidden * not_done.transpose(0, 1).unsqueeze(2).float()

            results = []
            for t in range(seq_len):
                step_out = out[:, t, :]
                for layer in self.mlp:
                    step_out = layer(step_out)
                results.append(step_out)

            out = torch.stack(results, dim=1)
            if out.dim() == 3:
                out = out.squeeze(1)
            return out