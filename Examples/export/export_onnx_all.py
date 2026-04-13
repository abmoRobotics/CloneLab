"""ONNX Export Script for CloneLab Trained Policies.

This script provides utilities to export trained policies to ONNX format for:
1. Normal (feedforward) policies from BC and IQL training
2. Stacked frame policies (with frame stacking enabled)
3. Recurrent (GRU) policies from BC-RNN and IQL-RNN training

The RNN export supports optional image/depth encoders - set channels to 0 to disable.

Usage Examples:
--------------
# Export normal BC/IQL policy:
python export_onnx_all.py --model_type normal --model_path path/to/model.pt --output policy.onnx

# Export stacked frame policy:
python export_onnx_all.py --model_type stacked --model_path path/to/model.pt --output policy_stacked.onnx --num_stacked_frames 3

# Export GRU-based RNN policy (with both image and depth encoders):
python export_onnx_all.py --model_type rnn --model_path path/to/model.pt --output policy_rnn.onnx

# Export GRU-based RNN policy (depth only, no image encoder):
python export_onnx_all.py --model_type rnn --model_path path/to/model.pt --output policy_rnn.onnx \
    --image_channels 0 --depth_channels 1

# Export GRU-based RNN policy (grayscale image + depth):
python export_onnx_all.py --model_type rnn --model_path path/to/model.pt --output policy_rnn.onnx \
    --image_channels 1 --depth_channels 1

# Full example with all parameters:
python export_onnx_all.py --model_type rnn \
    --model_path runs/experiment/checkpoints/best_model.pt \
    --output exported_policy.onnx \
    --image_size 160 90 \
    --image_channels 1 \
    --depth_channels 1 \
    --proprio_dim 3 \
    --hidden_size 128 \
    --num_layers 2
"""

import torch
import torch.nn as nn
import argparse
import os
import sys

# Add project paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "isaaclab"))

from Examples.isaaclab.models_cai import (
    actor_gaussian_image, 
    GRUActorGaussian,
)


# =============================================================================
# ONNX Wrapper Classes
# =============================================================================

class NormalPolicyOnnxWrapper(nn.Module):
    """Wrapper for feedforward (normal) policies to make them ONNX-compatible.
    
    The original model returns a distribution, which is not supported by ONNX.
    This wrapper extracts the mean of the distribution (deterministic action).
    
    Input format for ONNX:
        - image: (batch, channels, height, width) - RGB image
        - depth: (batch, channels, height, width) - Depth image
        - proprioceptive: (batch, proprio_dim) - Proprioceptive features
    
    Output:
        - action: (batch, action_dim) - Deterministic action
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, depth: torch.Tensor, proprioceptive: torch.Tensor):
        state = {
            "image": image, 
            "depth": depth, 
            "proprioceptive": proprioceptive
        }
        distribution = self.model(state)
        return distribution.mean


class StackedPolicyOnnxWrapper(nn.Module):
    """Wrapper for stacked frame policies to make them ONNX-compatible.
    
    This is essentially the same as NormalPolicyOnnxWrapper, but the input
    image and depth tensors have more channels (stacked frames along channel dim).
    
    Input format for ONNX (with num_stacked_frames=3):
        - image: (batch, 3*3=9, height, width) - Stacked RGB images
        - depth: (batch, 1*3=3, height, width) - Stacked depth images
        - proprioceptive: (batch, proprio_dim) - Proprioceptive features
    
    Output:
        - action: (batch, action_dim) - Deterministic action
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, depth: torch.Tensor, proprioceptive: torch.Tensor):
        state = {
            "image": image,
            "depth": depth,
            "proprioceptive": proprioceptive
        }
        distribution = self.model(state)
        return distribution.mean


class GRUPolicyOnnxWrapper(nn.Module):
    """Wrapper for GRU-based recurrent policies to make them ONNX-compatible.
    
    For recurrent policies, we need to handle the hidden state. This wrapper
    takes the hidden state as an input and returns the updated hidden state
    along with the action.
    
    Handles models with optional image/depth encoders (channels can be 0).
    
    IMPORTANT: Hidden states are transposed to batch-first format for Triton compatibility.
    The GRU internally uses (num_layers, batch, hidden_size), but we expose
    (batch, num_layers, hidden_size) to avoid dynamic dims in non-leading positions.
    
    Input format for ONNX (inputs vary based on which encoders are enabled):
        - image: (batch, 1, channels, height, width) - Single frame RGB (if image_channels > 0)
        - depth: (batch, 1, channels, height, width) - Single frame depth (if depth_channels > 0)
        - proprioceptive: (batch, 1, proprio_dim) - Proprioceptive features
        - hidden_in: (batch, num_layers, hidden_size) - GRU hidden state (BATCH FIRST!)
    
    Output:
        - action: (batch, action_dim) - Deterministic action
        - hidden_out: (batch, num_layers, hidden_size) - Updated hidden state (BATCH FIRST!)
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
        
        distribution, hidden_out = self.model(state, hidden)
        
        # Transpose hidden_out back to batch-first (batch, num_layers, hidden)
        hidden_out = hidden_out.transpose(0, 1).contiguous()
        
        # Get mean action and reshape to ensure static dimensions
        # Input is (batch, seq=1, action_dim), we want (batch, action_dim)
        action_mean = distribution.mean
        batch_size = action_mean.shape[0]
        action = action_mean.reshape(batch_size, self.action_dim)
        
        return action, hidden_out


class GRUPolicyOnnxWrapperStateless(nn.Module):
    """Stateless wrapper for GRU policies - initializes hidden state internally.
    
    This is a simpler version that doesn't require passing hidden state,
    useful for single-step inference where you always start fresh.
    
    Handles models with optional image/depth encoders (channels can be 0).
    
    WARNING: This loses temporal information between calls. Use the stateful
    version (GRUPolicyOnnxWrapper) for proper recurrent inference.
    
    Input format for ONNX (inputs vary based on which encoders are enabled):
        - image: (batch, channels, height, width) - Single frame RGB (if image_channels > 0)
        - depth: (batch, channels, height, width) - Single frame depth (if depth_channels > 0)
        - proprioceptive: (batch, proprio_dim) - Proprioceptive features
    
    Output:
        - action: (batch, action_dim) - Deterministic action
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
        
        # Initialize hidden state with zeros
        batch_size = proprioceptive.shape[0]
        hidden = torch.zeros(
            self.num_layers, batch_size, self.hidden_size, 
            device=proprioceptive.device, dtype=proprioceptive.dtype
        )
        
        distribution, _ = self.model(state, hidden)
        # Reshape to ensure static dimensions
        action_mean = distribution.mean
        action = action_mean.reshape(batch_size, self.action_dim)
        return action


# =============================================================================
# Model Creation Functions
# =============================================================================

def create_normal_model(config: dict, device: str = "cuda:0"):
    """Create a normal (feedforward) actor model."""
    model = actor_gaussian_image(
        device=device,
        proprioception_channels=config["proprio_dim"],
        image_channels=config["image_channels"],
        depth_channels=config["depth_channels"],
        action_dim=config.get("action_dim", 2),
        mlp_features=config.get("mlp_features", [256, 160, 128]),
        image_input_dim=config["image_size"],
        image_encoder_features=config.get("image_encoder_features", [8, 16, 32, 64]),
        image_fc_features=config.get("image_fc_features", [120, 60]),
    ).to(device)
    return model


def create_stacked_model(config: dict, device: str = "cuda:0"):
    """Create a stacked frame actor model.
    
    Same architecture as normal model, but with more input channels.
    """
    num_frames = config.get("num_stacked_frames", 3)
    model = actor_gaussian_image(
        device=device,
        proprioception_channels=config["proprio_dim"],
        image_channels=config["image_channels"] * num_frames,
        depth_channels=config["depth_channels"] * num_frames,
        action_dim=config.get("action_dim", 2),
        mlp_features=config.get("mlp_features", [256, 160, 128]),
        image_input_dim=config["image_size"],
        image_encoder_features=config.get("image_encoder_features", [8, 16, 32, 64]),
        image_fc_features=config.get("image_fc_features", [120, 60]),
    ).to(device)
    return model


def create_rnn_model(config: dict, device: str = "cuda:0"):
    """Create a GRU-based recurrent actor model."""
    model = GRUActorGaussian(
        device=device,
        proprioception_channels=config["proprio_dim"],
        image_channels=config["image_channels"],
        depth_channels=config["depth_channels"],
        action_dim=config.get("action_dim", 2),
        image_size=config["image_size"],
        hidden_size=config.get("hidden_size", 128),
        num_layers=config.get("num_layers", 2),
        mlp_features=config.get("mlp_features", [256, 160, 128]),
        image_encoder_features=config.get("image_encoder_features", [8, 16, 32, 64]),
        image_fc_features=config.get("image_fc_features", [120, 60]),
    ).to(device)
    return model


# =============================================================================
# Export Functions
# =============================================================================

def export_normal_policy(
    model_path: str,
    output_path: str,
    config: dict,
    device: str = "cuda:0"
):
    """Export a normal (feedforward) policy to ONNX.
    
    Args:
        model_path: Path to the saved model weights (.pt file)
        output_path: Path for the output ONNX file
        config: Model configuration dictionary
        device: Device to use for export
    """
    print(f"Creating normal policy model...")
    model = create_normal_model(config, device)
    
    print(f"Loading weights from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    # Handle different checkpoint formats
    if "actor" in state_dict:
        state_dict = state_dict["actor"]
    elif "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    
    # Wrap for ONNX export
    onnx_model = NormalPolicyOnnxWrapper(model)
    
    # Create dummy inputs
    batch_size = 1
    h, w = config["image_size"]
    dummy_image = torch.randn(batch_size, config["image_channels"], h, w).to(device)
    dummy_depth = torch.randn(batch_size, config["depth_channels"], h, w).to(device)
    dummy_proprio = torch.randn(batch_size, config["proprio_dim"]).to(device)
    
    # Export
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        onnx_model,
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
        opset_version=11,
        verbose=False
    )
    print(f"✓ Export complete: {output_path}")


def export_stacked_policy(
    model_path: str,
    output_path: str,
    config: dict,
    device: str = "cuda:0"
):
    """Export a stacked frame policy to ONNX.
    
    Args:
        model_path: Path to the saved model weights (.pt file)
        output_path: Path for the output ONNX file
        config: Model configuration dictionary (must include num_stacked_frames)
        device: Device to use for export
    """
    num_frames = config.get("num_stacked_frames", 3)
    print(f"Creating stacked policy model with {num_frames} stacked frames...")
    model = create_stacked_model(config, device)
    
    print(f"Loading weights from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    if "actor" in state_dict:
        state_dict = state_dict["actor"]
    elif "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    
    # Wrap for ONNX export
    onnx_model = StackedPolicyOnnxWrapper(model)
    
    # Create dummy inputs with stacked channels
    batch_size = 1
    h, w = config["image_size"]
    image_channels = config["image_channels"] * num_frames
    depth_channels = config["depth_channels"] * num_frames
    
    dummy_image = torch.randn(batch_size, image_channels, h, w).to(device)
    dummy_depth = torch.randn(batch_size, depth_channels, h, w).to(device)
    dummy_proprio = torch.randn(batch_size, config["proprio_dim"]).to(device)
    
    # Export
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        onnx_model,
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
        opset_version=11,
        verbose=False
    )
    print(f"✓ Export complete: {output_path}")
    print(f"  Note: Image input has {image_channels} channels (RGB×{num_frames})")
    print(f"  Note: Depth input has {depth_channels} channels (Depth×{num_frames})")


def export_rnn_policy(
    model_path: str,
    output_path: str,
    config: dict,
    device: str = "cuda:0",
    stateful: bool = True
):
    """Export a GRU-based recurrent policy to ONNX.
    
    Handles models with optional image/depth encoders (image_channels or depth_channels can be 0).
    
    Args:
        model_path: Path to the saved model weights (.pt file)
        output_path: Path for the output ONNX file
        config: Model configuration dictionary
        device: Device to use for export
        stateful: If True, export with hidden state as input/output (recommended).
                  If False, export stateless version (hidden initialized to zeros).
    """
    image_channels = config.get("image_channels", 0)
    depth_channels = config.get("depth_channels", 0)
    
    print(f"Creating RNN (GRU) policy model...")
    print(f"  Image encoder: {'enabled' if image_channels > 0 else 'disabled'} ({image_channels} channels)")
    print(f"  Depth encoder: {'enabled' if depth_channels > 0 else 'disabled'} ({depth_channels} channels)")
    
    model = create_rnn_model(config, device)
    
    print(f"Loading weights from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    if "actor" in state_dict:
        state_dict = state_dict["actor"]
    elif "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create dummy inputs dynamically based on which encoders are present
    batch_size = 1
    h, w = config["image_size"]
    hidden_size = config.get("hidden_size", 128)
    num_layers = config.get("num_layers", 2)
    
    # Build inputs, input_names, and dynamic_axes based on which encoders exist
    dummy_inputs = []
    input_names = []
    dynamic_axes = {}
    
    if image_channels > 0:
        if stateful:
            dummy_inputs.append(torch.randn(batch_size, 1, image_channels, h, w).to(device))
        else:
            dummy_inputs.append(torch.randn(batch_size, image_channels, h, w).to(device))
        input_names.append("image")
        dynamic_axes["image"] = {0: "batch_size"}
    
    if depth_channels > 0:
        if stateful:
            dummy_inputs.append(torch.randn(batch_size, 1, depth_channels, h, w).to(device))
        else:
            dummy_inputs.append(torch.randn(batch_size, depth_channels, h, w).to(device))
        input_names.append("depth")
        dynamic_axes["depth"] = {0: "batch_size"}
    
    if stateful:
        dummy_inputs.append(torch.randn(batch_size, 1, config["proprio_dim"]).to(device))
    else:
        dummy_inputs.append(torch.randn(batch_size, config["proprio_dim"]).to(device))
    input_names.append("proprioceptive")
    dynamic_axes["proprioceptive"] = {0: "batch_size"}
    
    action_dim = config.get("action_dim", 2)
    
    if stateful:
        # Stateful export with hidden state as input/output
        # Hidden state is BATCH-FIRST: (batch, num_layers, hidden_size) for Triton compatibility
        onnx_model = GRUPolicyOnnxWrapper(model, action_dim=action_dim)
        
        # NOTE: batch-first hidden state (batch, num_layers, hidden_size)
        dummy_inputs.append(torch.zeros(batch_size, num_layers, hidden_size).to(device))
        input_names.append("hidden_in")
        dynamic_axes["hidden_in"] = {0: "batch_size"}  # batch is now first dimension
        dynamic_axes["action"] = {0: "batch_size"}
        dynamic_axes["hidden_out"] = {0: "batch_size"}  # batch is now first dimension
        output_names = ["action", "hidden_out"]
        
        print(f"Exporting stateful RNN to {output_path}...")
        torch.onnx.export(
            onnx_model,
            tuple(dummy_inputs),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=11,
            verbose=False
        )
        print(f"✓ Export complete: {output_path}")
        print(f"  Inputs: {', '.join(input_names)}")
        print(f"  Hidden state shape: (batch_size, {num_layers}, {hidden_size}) - BATCH FIRST!")
        print(f"  Initialize hidden_in with zeros for new episodes")
    else:
        # Stateless export (hidden initialized internally)
        onnx_model = GRUPolicyOnnxWrapperStateless(model, action_dim=action_dim)
        dynamic_axes["action"] = {0: "batch_size"}
        output_names = ["action"]
        
        print(f"Exporting stateless RNN to {output_path}...")
        torch.onnx.export(
            onnx_model,
            tuple(dummy_inputs),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=11,
            verbose=False
        )
        print(f"✓ Export complete: {output_path}")
        print(f"  Inputs: {', '.join(input_names)}")
        print(f"  WARNING: Stateless mode - hidden state reset each call")


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export CloneLab trained policies to ONNX format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
# Export normal BC/IQL policy:
python export_onnx_all.py --model_type normal \\
    --model_path runs/experiment/checkpoints/best_actor.pt \\
    --output policy.onnx

# Export stacked frame policy:
python export_onnx_all.py --model_type stacked \\
    --model_path runs/experiment/checkpoints/best_actor.pt \\
    --output policy_stacked.onnx \\
    --num_stacked_frames 3

# Export RNN policy (stateful):
python export_onnx_all.py --model_type rnn \\
    --model_path runs/experiment/checkpoints/best_actor.pt \\
    --output policy_rnn.onnx

# Export RNN policy (stateless):
python export_onnx_all.py --model_type rnn \\
    --model_path runs/experiment/checkpoints/best_actor.pt \\
    --output policy_rnn_stateless.onnx \\
    --stateless
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model_type", 
        type=str, 
        required=True,
        choices=["normal", "stacked", "rnn"],
        help="Type of policy: 'normal' (feedforward), 'stacked' (frame stacking), 'rnn' (GRU-based)"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the saved model weights (.pt file)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Output path for the ONNX file"
    )
    
    # Model architecture arguments
    parser.add_argument(
        "--image_size", 
        type=int, 
        nargs=2, 
        default=[160, 90],
        help="Image size [height, width] (default: 160 90)"
    )
    parser.add_argument(
        "--image_channels", 
        type=int, 
        default=3,
        help="Number of image channels (default: 3 for RGB)"
    )
    parser.add_argument(
        "--depth_channels", 
        type=int, 
        default=1,
        help="Number of depth channels (default: 1)"
    )
    parser.add_argument(
        "--proprio_dim", 
        type=int, 
        default=3,
        help="Proprioceptive input dimension (default: 3)"
    )
    parser.add_argument(
        "--action_dim", 
        type=int, 
        default=2,
        help="Action dimension (default: 2)"
    )
    
    # Frame stacking arguments
    parser.add_argument(
        "--num_stacked_frames", 
        type=int, 
        default=3,
        help="Number of stacked frames for 'stacked' model type (default: 3)"
    )
    
    # RNN arguments
    parser.add_argument(
        "--hidden_size", 
        type=int, 
        default=128,
        help="GRU hidden state size for 'rnn' model type (default: 128)"
    )
    parser.add_argument(
        "--num_layers", 
        type=int, 
        default=2,
        help="Number of GRU layers for 'rnn' model type (default: 2)"
    )
    parser.add_argument(
        "--stateless", 
        action="store_true",
        help="Export RNN in stateless mode (hidden reset each call)"
    )
    
    # Network architecture arguments
    parser.add_argument(
        "--image_encoder_features",
        type=int,
        nargs='+',
        default=[8, 16, 32, 64],
        help="Conv encoder feature sizes (default: 8 16 32 64)"
    )
    parser.add_argument(
        "--image_fc_features",
        type=int,
        nargs='+',
        default=[120, 60],
        help="Image encoder FC layer sizes (default: 120 60)"
    )
    parser.add_argument(
        "--mlp_features",
        type=int,
        nargs='+',
        default=[256, 160, 128],
        help="MLP layer sizes (default: 256 160 128)"
    )
    
    # Other arguments
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0",
        help="Device for export (default: cuda:0)"
    )
    
    args = parser.parse_args()
    
    # Build config
    config = {
        "image_size": args.image_size,
        "image_channels": args.image_channels,
        "depth_channels": args.depth_channels,
        "proprio_dim": args.proprio_dim,
        "action_dim": args.action_dim,
        "num_stacked_frames": args.num_stacked_frames,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "image_encoder_features": args.image_encoder_features,
        "image_fc_features": args.image_fc_features,
        "mlp_features": args.mlp_features,
    }
    
    # Ensure output path has .onnx extension
    output_path = args.output
    if not output_path.endswith(".onnx"):
        output_path += ".onnx"
    
    # Export based on model type
    print("="*60)
    print(f"ONNX Export - Model Type: {args.model_type}")
    print("="*60)
    
    if args.model_type == "normal":
        export_normal_policy(args.model_path, output_path, config, args.device)
    elif args.model_type == "stacked":
        export_stacked_policy(args.model_path, output_path, config, args.device)
    elif args.model_type == "rnn":
        export_rnn_policy(
            args.model_path, output_path, config, args.device, 
            stateful=not args.stateless
        )


if __name__ == "__main__":
    main()
