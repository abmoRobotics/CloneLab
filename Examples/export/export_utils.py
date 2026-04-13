"""ONNX Export Utilities for CloneLab.

This module provides functions to export trained policies to ONNX format
that can be used directly from training scripts.

Example usage in training script:
---------------------------------
from export_utils import export_policy_to_onnx

# After training
trainer = train_bc()

# Export the trained policy
export_policy_to_onnx(
    policy=trainer.policy.actor,
    output_path="my_policy.onnx",
    model_type="normal",  # or "stacked" or "rnn"
    image_size=[160, 90],
    image_channels=3,
    depth_channels=1,
    proprio_dim=3,
)

# For RNN with optional encoders (image_channels=0 or depth_channels=0):
export_policy_to_onnx(
    policy=trainer.policy.actor,
    output_path="rnn_depth_only.onnx",
    model_type="rnn",
    image_size=[160, 90],
    image_channels=0,      # No image encoder
    depth_channels=1,      # Only depth encoder
    proprio_dim=3,
)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Union


class FeedforwardOnnxWrapper(nn.Module):
    """Wrapper for feedforward policies."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, depth, proprioceptive):
        state = {"image": image, "depth": depth, "proprioceptive": proprioceptive}
        dist = self.model(state)
        return dist.mean


class RNNStatefulOnnxWrapper(nn.Module):
    """Wrapper for GRU policies with explicit hidden state.
    
    Handles models with optional image/depth encoders (channels can be 0).
    
    IMPORTANT: Hidden states are transposed to batch-first format for Triton compatibility.
    The GRU internally uses (num_layers, batch, hidden_size), but we expose
    (batch, num_layers, hidden_size) to avoid dynamic dims in non-leading positions.
    """
    def __init__(self, model, action_dim: int = 2):
        super().__init__()
        self.model = model
        self.image_channels = getattr(model, 'image_channels', 0)
        self.depth_channels = getattr(model, 'depth_channels', 0)
        self.action_dim = action_dim

    def forward(self, *args):
        # Dynamically handle inputs based on which encoders are present
        # Order: [image], [depth], proprioceptive, hidden
        idx = 0
        state = {}
        
        if self.image_channels > 0:
            state["image"] = args[idx]
            idx += 1
        if self.depth_channels > 0:
            state["depth"] = args[idx]
            idx += 1
        
        state["proprioceptive"] = args[idx]
        hidden_in = args[idx + 1]
        
        # Transpose hidden from batch-first (batch, num_layers, hidden) 
        # to GRU format (num_layers, batch, hidden)
        hidden = hidden_in.transpose(0, 1).contiguous()
        
        dist, hidden_out = self.model(state, hidden)
        
        # Transpose hidden_out back to batch-first (batch, num_layers, hidden)
        hidden_out = hidden_out.transpose(0, 1).contiguous()
        
        # Get mean action and reshape to ensure static dimensions
        action_mean = dist.mean
        batch_size = action_mean.shape[0]
        action = action_mean.reshape(batch_size, self.action_dim)
        
        return action, hidden_out


class RNNStatelessOnnxWrapper(nn.Module):
    """Wrapper for GRU policies without explicit hidden state.
    
    Handles models with optional image/depth encoders (channels can be 0).
    """
    def __init__(self, model, action_dim: int = 2):
        super().__init__()
        self.model = model
        self.num_layers = model.num_layers
        self.hidden_size = model.hidden_size
        self.image_channels = getattr(model, 'image_channels', 0)
        self.depth_channels = getattr(model, 'depth_channels', 0)
        self.action_dim = action_dim

    def forward(self, *args):
        # Dynamically handle inputs based on which encoders are present
        # Order: [image], [depth], proprioceptive
        idx = 0
        state = {}
        
        if self.image_channels > 0:
            state["image"] = args[idx].unsqueeze(1)  # Add sequence dim
            idx += 1
        if self.depth_channels > 0:
            state["depth"] = args[idx].unsqueeze(1)  # Add sequence dim
            idx += 1
        
        proprioceptive = args[idx]
        state["proprioceptive"] = proprioceptive.unsqueeze(1)  # Add sequence dim
        
        batch_size = proprioceptive.shape[0]
        hidden = torch.zeros(
            self.num_layers, batch_size, self.hidden_size,
            device=proprioceptive.device, dtype=proprioceptive.dtype
        )
        dist, _ = self.model(state, hidden)
        # Reshape to ensure static dimensions
        action_mean = dist.mean
        action = action_mean.reshape(batch_size, self.action_dim)
        return action


def export_policy_to_onnx(
    policy: nn.Module,
    output_path: str,
    model_type: str = "normal",
    image_size: List[int] = [160, 90],
    image_channels: int = 3,
    depth_channels: int = 1,
    proprio_dim: int = 3,
    action_dim: int = 2,
    num_stacked_frames: int = 1,
    hidden_size: int = 128,
    num_layers: int = 2,
    stateful_rnn: bool = True,
    device: str = "cuda:0",
    opset_version: int = 11,
) -> str:
    """Export a trained policy to ONNX format.
    
    Args:
        policy: The trained policy model (actor network)
        output_path: Path for the output ONNX file
        model_type: One of "normal", "stacked", or "rnn"
        image_size: [height, width] of input images
        image_channels: Number of base image channels (before stacking)
        depth_channels: Number of base depth channels (before stacking)
        proprio_dim: Dimension of proprioceptive input
        action_dim: Dimension of action output
        num_stacked_frames: Number of stacked frames (for "stacked" type)
        hidden_size: GRU hidden size (for "rnn" type)
        num_layers: Number of GRU layers (for "rnn" type)
        stateful_rnn: Whether to export RNN with hidden state I/O
        device: Device for export
        opset_version: ONNX opset version
        
    Returns:
        Path to the exported ONNX file
    """
    policy.eval()
    
    # Ensure output has .onnx extension
    if not output_path.endswith(".onnx"):
        output_path += ".onnx"
    
    h, w = image_size
    batch_size = 1
    
    if model_type == "normal":
        wrapper = FeedforwardOnnxWrapper(policy)
        dummy_image = torch.randn(batch_size, image_channels, h, w).to(device)
        dummy_depth = torch.randn(batch_size, depth_channels, h, w).to(device)
        dummy_proprio = torch.randn(batch_size, proprio_dim).to(device)
        
        torch.onnx.export(
            wrapper,
            (dummy_image, dummy_depth, dummy_proprio),
            output_path,
            input_names=["image", "depth", "proprioceptive"],
            output_names=["action"],
            dynamic_axes={
                "image": {0: "batch_size"},
                "depth": {0: "batch_size"},
                "proprioceptive": {0: "batch_size"},
                "action": {0: "batch_size"},
            },
            opset_version=opset_version,
        )
        print(f"✓ Exported normal policy to {output_path}")
        
    elif model_type == "stacked":
        wrapper = FeedforwardOnnxWrapper(policy)
        stacked_img_channels = image_channels * num_stacked_frames
        stacked_depth_channels = depth_channels * num_stacked_frames
        
        dummy_image = torch.randn(batch_size, stacked_img_channels, h, w).to(device)
        dummy_depth = torch.randn(batch_size, stacked_depth_channels, h, w).to(device)
        dummy_proprio = torch.randn(batch_size, proprio_dim).to(device)
        
        torch.onnx.export(
            wrapper,
            (dummy_image, dummy_depth, dummy_proprio),
            output_path,
            input_names=["image", "depth", "proprioceptive"],
            output_names=["action"],
            dynamic_axes={
                "image": {0: "batch_size"},
                "depth": {0: "batch_size"},
                "proprioceptive": {0: "batch_size"},
                "action": {0: "batch_size"},
            },
            opset_version=opset_version,
        )
        print(f"✓ Exported stacked policy to {output_path}")
        print(f"  Image channels: {stacked_img_channels}, Depth channels: {stacked_depth_channels}")
        
    elif model_type == "rnn":
        # Build inputs dynamically based on which encoders are present
        dummy_inputs = []
        input_names = []
        dynamic_axes = {}
        
        if image_channels > 0:
            if stateful_rnn:
                dummy_inputs.append(torch.randn(batch_size, 1, image_channels, h, w).to(device))
            else:
                dummy_inputs.append(torch.randn(batch_size, image_channels, h, w).to(device))
            input_names.append("image")
            dynamic_axes["image"] = {0: "batch_size"}
            
        if depth_channels > 0:
            if stateful_rnn:
                dummy_inputs.append(torch.randn(batch_size, 1, depth_channels, h, w).to(device))
            else:
                dummy_inputs.append(torch.randn(batch_size, depth_channels, h, w).to(device))
            input_names.append("depth")
            dynamic_axes["depth"] = {0: "batch_size"}
        
        if stateful_rnn:
            dummy_inputs.append(torch.randn(batch_size, 1, proprio_dim).to(device))
        else:
            dummy_inputs.append(torch.randn(batch_size, proprio_dim).to(device))
        input_names.append("proprioceptive")
        dynamic_axes["proprioceptive"] = {0: "batch_size"}
        
        if stateful_rnn:
            wrapper = RNNStatefulOnnxWrapper(policy)
            # NOTE: batch-first hidden state (batch, num_layers, hidden_size) for Triton compatibility
            dummy_inputs.append(torch.zeros(batch_size, num_layers, hidden_size).to(device))
            input_names.append("hidden_in")
            dynamic_axes["hidden_in"] = {0: "batch_size"}  # batch is now first dimension
            dynamic_axes["action"] = {0: "batch_size"}
            dynamic_axes["hidden_out"] = {0: "batch_size"}  # batch is now first dimension
            output_names = ["action", "hidden_out"]
        else:
            wrapper = RNNStatelessOnnxWrapper(policy)
            dynamic_axes["action"] = {0: "batch_size"}
            output_names = ["action"]
        
        torch.onnx.export(
            wrapper,
            tuple(dummy_inputs),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
        )
        
        # Print summary
        encoder_info = []
        if image_channels > 0:
            encoder_info.append(f"image ({image_channels} ch)")
        if depth_channels > 0:
            encoder_info.append(f"depth ({depth_channels} ch)")
        encoder_str = " + ".join(encoder_info) if encoder_info else "none"
        
        if stateful_rnn:
            print(f"✓ Exported stateful RNN policy to {output_path}")
            print(f"  Inputs: {', '.join(input_names)}")
            print(f"  Encoders: {encoder_str}")
            print(f"  Hidden state shape: (batch, {num_layers}, {hidden_size}) - BATCH FIRST!")
        else:
            print(f"✓ Exported stateless RNN policy to {output_path}")
            print(f"  Inputs: {', '.join(input_names)}")
            print(f"  Encoders: {encoder_str}")
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'normal', 'stacked', or 'rnn'")
    
    return output_path


def export_from_checkpoint(
    checkpoint_path: str,
    output_path: str,
    model_type: str = "normal",
    model_config: Dict = None,
    device: str = "cuda:0",
) -> str:
    """Export a policy from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        output_path: Path for the output ONNX file
        model_type: One of "normal", "stacked", or "rnn"
        model_config: Model configuration dictionary
        device: Device for export
        
    Returns:
        Path to the exported ONNX file
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from isaaclab.models_cai import actor_gaussian_image, GRUActorGaussian
    
    if model_config is None:
        model_config = {
            "image_size": [160, 90],
            "image_channels": 3,
            "depth_channels": 1,
            "proprio_dim": 3,
            "action_dim": 2,
        }
    
    # Create model based on type
    if model_type in ["normal", "stacked"]:
        num_frames = model_config.get("num_stacked_frames", 1) if model_type == "stacked" else 1
        model = actor_gaussian_image(
            device=device,
            proprioception_channels=model_config["proprio_dim"],
            image_channels=model_config["image_channels"] * num_frames,
            depth_channels=model_config["depth_channels"] * num_frames,
            action_dim=model_config.get("action_dim", 2),
            image_input_dim=model_config["image_size"],
        ).to(device)
    else:  # rnn
        model = GRUActorGaussian(
            device=device,
            proprioception_channels=model_config["proprio_dim"],
            image_channels=model_config["image_channels"],
            depth_channels=model_config["depth_channels"],
            action_dim=model_config.get("action_dim", 2),
            image_size=model_config["image_size"],
            hidden_size=model_config.get("hidden_size", 128),
            num_layers=model_config.get("num_layers", 2),
        ).to(device)
    
    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    if "actor" in state_dict:
        state_dict = state_dict["actor"]
    elif "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    
    # Export
    return export_policy_to_onnx(
        policy=model,
        output_path=output_path,
        model_type=model_type,
        **model_config,
        device=device,
    )
