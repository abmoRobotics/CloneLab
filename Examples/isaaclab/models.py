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


class actor_gaussian_image(nn.Module):
    def __init__(self, 
                 device="cuda:0" if torch.cuda.is_available() else "cpu", 
                 proprioception_channels=3, 
                 encoder_channels=60, 
                 image_channels=1,
                 action_dim=2,
                 mlp_features=[256, 160, 128],
                 image_input_dim=[224, 224],
                 image_encoder_features=[8, 16, 32, 64],
                 image_fc_features=[120, 60],
                 activation="leaky_relu",
                 dropout_rate=0,
                 use_batch_norm=False,
                 min_log_std=-20.0,
                 max_log_std=2.0,
                 state_independent_log_std=True):
        super(actor_gaussian_image, self).__init__()
        self.device = device
        self.proprioception_channels = proprioception_channels
        self.encoder_channels = encoder_channels
        self.action_dim = action_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.state_independent_log_std = state_independent_log_std

        self.encoder = conv_encoder(in_channels=image_channels, input_dim=image_input_dim, encoder_features=image_encoder_features, fc_features=image_fc_features, encoder_activation=activation)

        self.mlp = nn.ModuleList()
        in_channels = self.proprioception_channels + self.encoder_channels
        
        for feature in mlp_features:
            self.mlp.append(nn.Linear(in_channels, feature))
            if use_batch_norm:
                self.mlp.append(nn.BatchNorm1d(feature))
            self.mlp.append(get_activation(activation))
            if dropout_rate > 0:
                self.mlp.append(nn.Dropout(dropout_rate))
            in_channels = feature

        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(in_channels, self.action_dim)
        if state_independent_log_std:
            self.log_std = nn.Parameter(torch.zeros(self.action_dim, dtype=torch.float32))
        else:
            self.log_std_head = nn.Linear(in_channels, self.action_dim)

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state: Dict[str, torch.Tensor]):
        # Encode image
        x_img = self.encoder(state["image"])
        
        # Handle sequential vs batch data
        is_sequence = x_img.ndim == 3
        proprioceptive = state["proprioceptive"]
        
        if is_sequence:
            # For sequential data: (batch, seq_len, features)
            x = torch.cat([proprioceptive[..., :self.proprioception_channels], x_img], dim=2)
        else:
            # For batch data: (batch, features)
            x = torch.cat([proprioceptive[..., :self.proprioception_channels], x_img], dim=1)
        
        # Pass through MLP
        for layer in self.mlp:
            x = layer(x)
        
        # Get mean and log_std
        mu = torch.tanh(self.mean_head(x))  # Bound actions to [-1, 1]
        if self.state_independent_log_std:
            log_std = torch.clamp(self.log_std, min=self.min_log_std, max=self.max_log_std)
        else:
            log_std = self.log_std_head(x)
            log_std = torch.clamp(log_std, min=self.min_log_std, max=self.max_log_std)

        return torch.distributions.Normal(mu, log_std.exp())
    
    def get_action(self, state: Dict[str, torch.Tensor], deterministic=True):
        """Get action for deployment"""
        with torch.no_grad():
            dist = self.forward(state)
            if deterministic:
                return dist.mean
            else:
                return dist.sample()

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
    def __init__(self, 
                 device="cuda:0" if torch.cuda.is_available() else "cpu", 
                 proprioception_channels=3, 
                 encoder_channels=60, 
                 image_channels=1,
                 mlp_features=[256, 160, 128],
                 image_input_dim=[224, 224],
                 image_encoder_features=[8, 16, 32, 64],
                 image_fc_features=[120, 60],
                 activation="leaky_relu",
                 dropout_rate=0,
                 use_batch_norm=False,
                 **kwargs):
        super(v_image, self).__init__()
        self.device = device
        self.proprioception_channels = proprioception_channels
        self.encoder_channels = encoder_channels

        self.encoder = conv_encoder(in_channels=image_channels, input_dim=image_input_dim, encoder_features=image_encoder_features, fc_features=image_fc_features, encoder_activation=activation)

        self.mlp = nn.ModuleList()
        in_channels = self.proprioception_channels + self.encoder_channels
        
        for feature in mlp_features:
            self.mlp.append(nn.Linear(in_channels, feature))
            if use_batch_norm:
                self.mlp.append(nn.BatchNorm1d(feature))
            self.mlp.append(get_activation(activation))
            if dropout_rate > 0:
                self.mlp.append(nn.Dropout(dropout_rate))
            in_channels = feature

        self.value_head = nn.Linear(in_channels, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state: Dict[str, torch.Tensor]):
        # Encode image
        x_img = self.encoder(state["image"])
        
        # Handle sequential vs batch data
        is_sequence = x_img.ndim == 3
        proprioceptive = state["proprioceptive"]
        
        if is_sequence:
            # For sequential data: (batch, seq_len, features)
            x = torch.cat([proprioceptive[..., :self.proprioception_channels], x_img], dim=2)
        else:
            # For batch data: (batch, features)
            x = torch.cat([proprioceptive[..., :self.proprioception_channels], x_img], dim=1)
        
        # Pass through MLP
        for layer in self.mlp:
            x = layer(x)
        
        return self.value_head(x)

class q_image(nn.Module):
    def __init__(self, 
                 device="cuda:0" if torch.cuda.is_available() else "cpu", 
                 proprioception_channels=3, 
                 encoder_channels=60, 
                 image_channels=1,
                 action_dim=2,
                 mlp_features=[256, 160, 128],
                 image_input_dim=[224, 224],
                 image_encoder_features=[8, 16, 32, 64],
                 image_fc_features=[120, 60],
                 activation="leaky_relu",
                 dropout_rate=0,
                 use_batch_norm=False):
        super(q_image, self).__init__()
        self.device = device
        self.proprioception_channels = proprioception_channels
        self.encoder_channels = encoder_channels
        self.action_dim = action_dim

        self.encoder = conv_encoder(in_channels=image_channels, input_dim=image_input_dim, encoder_features=image_encoder_features, fc_features=image_fc_features, encoder_activation=activation)

        self.mlp = nn.ModuleList()
        # Include action dimensions in input
        in_channels = self.proprioception_channels + self.encoder_channels + self.action_dim
        
        for feature in mlp_features:
            self.mlp.append(nn.Linear(in_channels, feature))
            if use_batch_norm:
                self.mlp.append(nn.BatchNorm1d(feature))
            self.mlp.append(get_activation(activation))
            if dropout_rate > 0:
                self.mlp.append(nn.Dropout(dropout_rate))
            in_channels = feature

        self.q_head = nn.Linear(in_channels, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state: Dict[str, torch.Tensor], action: torch.Tensor):
        # Encode image
        x_img = self.encoder(state["image"])
        
        # Handle sequential vs batch data
        is_sequence = x_img.ndim == 3
        proprioceptive = state["proprioceptive"]

        if is_sequence:
            # For sequential data: (batch, seq_len, features)
            x = torch.cat([
                proprioceptive[..., :self.proprioception_channels], 
                x_img, 
                action
            ], dim=2)
        else:
            # For batch data: (batch, features)
            x = torch.cat([
                proprioceptive[..., :self.proprioception_channels], 
                x_img, 
                action
            ], dim=1)
        
        # Pass through MLP
        for layer in self.mlp:
            x = layer(x)
        
        return self.q_head(x)

class TwinQ_image(nn.Module):
    def __init__(self, 
                 device="cuda:0" if torch.cuda.is_available() else "cpu", 
                 proprioception_channels=3, 
                 encoder_channels=60, 
                 image_channels=1,
                 action_dim=2,
                 **kwargs):
        super(TwinQ_image, self).__init__()
        self.device = device
        
        # Create two Q-networks with shared parameters for architecture
        q_kwargs = {
            'device': device,
            'proprioception_channels': proprioception_channels,
            'encoder_channels': encoder_channels,
            'image_channels': image_channels,
            'action_dim': action_dim,
            **kwargs
        }
        
        self.q1 = q_image(**q_kwargs).to(device)
        self.q2 = q_image(**q_kwargs).to(device)

    def forward(self, state: Dict[str, torch.Tensor], action: torch.Tensor):
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        return q1, q2
    
    def q1_forward(self, state: Dict[str, torch.Tensor], action: torch.Tensor):
        """Forward pass through Q1 only (useful for policy updates)"""
        return self.q1(state, action)

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