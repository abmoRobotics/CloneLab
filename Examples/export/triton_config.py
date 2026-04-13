"""Triton Inference Server Configuration Generator for CloneLab Models.

This module generates Triton config.pbtxt files for exported ONNX models.

IMPORTANT: For RNN models with hidden states, Triton has issues with dynamic 
dimensions (-1) in non-leading positions. Use --batch_size to fix the batch 
dimension, especially for hidden_in/hidden_out tensors.

Usage:
------
# Generate config for normal/stacked model:
python triton_config.py --model_name my_policy --model_type normal \
    --image_size 160 90 --image_channels 3 --depth_channels 1 --proprio_dim 3

# Generate config for RNN model (stateful) with fixed batch size (RECOMMENDED):
python triton_config.py --model_name my_rnn_policy --model_type rnn \
    --image_size 160 90 --image_channels 1 --depth_channels 1 --proprio_dim 3 \
    --hidden_size 128 --num_layers 2 --batch_size 1

# Generate config for RNN model (depth only):
python triton_config.py --model_name depth_only_rnn --model_type rnn \
    --image_size 160 90 --image_channels 0 --depth_channels 1 --proprio_dim 3 \
    --hidden_size 128 --num_layers 2 --batch_size 1
"""

import argparse
import os
from typing import List, Optional


def generate_triton_config(
    model_name: str,
    model_type: str = "normal",
    image_size: List[int] = [160, 90],
    image_channels: int = 3,
    depth_channels: int = 1,
    proprio_dim: int = 3,
    action_dim: int = 2,
    hidden_size: int = 128,
    num_layers: int = 2,
    stateful_rnn: bool = True,
    max_batch_size: int = 0,
    batch_size: Optional[int] = None,
    platform: str = "onnxruntime_onnx",
) -> str:
    """Generate Triton Inference Server config.pbtxt content.
    
    Args:
        model_name: Name of the model (used in config)
        model_type: One of "normal", "stacked", or "rnn"
        image_size: [height, width] of input images
        image_channels: Number of image channels (0 to disable)
        depth_channels: Number of depth channels (0 to disable)
        proprio_dim: Dimension of proprioceptive input
        action_dim: Dimension of action output
        hidden_size: GRU hidden size (for RNN)
        num_layers: Number of GRU layers (for RNN)
        stateful_rnn: Whether RNN is stateful (has hidden state I/O)
        max_batch_size: Max batch size (0 for model to handle batching)
        batch_size: Fixed batch size. If None, uses -1 (dynamic). 
                   For RNN models, setting this is RECOMMENDED to avoid 
                   Triton issues with dynamic dims in hidden state tensors.
        platform: Triton platform string
        
    Returns:
        String content of config.pbtxt file
    """
    h, w = image_size
    
    # Use fixed batch size or dynamic (-1)
    batch_dim = str(batch_size) if batch_size is not None else "-1"
    
    config_lines = [
        f'name: "{model_name}"',
        f'platform: "{platform}"',
        f'max_batch_size: {max_batch_size}',
        'input ['
    ]
    
    if model_type in ["normal", "stacked"]:
        # Normal/stacked models always have image, depth, proprioceptive
        config_lines.extend([
            '  {',
            '    name: "image"',
            '    data_type: TYPE_FP32',
            f'    dims: [ {batch_dim}, {image_channels}, {h}, {w} ]',
            '  },',
            '  {',
            '    name: "depth"',
            '    data_type: TYPE_FP32',
            f'    dims: [ {batch_dim}, {depth_channels}, {h}, {w} ]',
            '  },',
            '  {',
            '    name: "proprioceptive"',
            '    data_type: TYPE_FP32',
            f'    dims: [ {batch_dim}, {proprio_dim} ]',
            '  }',
        ])
        
    elif model_type == "rnn":
        # RNN models have optional image/depth based on channels
        inputs_added = False
        
        if image_channels > 0:
            if stateful_rnn:
                # Stateful: (batch, seq=1, channels, h, w)
                config_lines.extend([
                    '  {',
                    '    name: "image"',
                    '    data_type: TYPE_FP32',
                    f'    dims: [ {batch_dim}, 1, {image_channels}, {h}, {w} ]',
                    '  },',
                ])
            else:
                # Stateless: (batch, channels, h, w)
                config_lines.extend([
                    '  {',
                    '    name: "image"',
                    '    data_type: TYPE_FP32',
                    f'    dims: [ {batch_dim}, {image_channels}, {h}, {w} ]',
                    '  },',
                ])
            inputs_added = True
        
        if depth_channels > 0:
            if stateful_rnn:
                config_lines.extend([
                    '  {',
                    '    name: "depth"',
                    '    data_type: TYPE_FP32',
                    f'    dims: [ {batch_dim}, 1, {depth_channels}, {h}, {w} ]',
                    '  },',
                ])
            else:
                config_lines.extend([
                    '  {',
                    '    name: "depth"',
                    '    data_type: TYPE_FP32',
                    f'    dims: [ {batch_dim}, {depth_channels}, {h}, {w} ]',
                    '  },',
                ])
            inputs_added = True
        
        # Proprioceptive input
        if stateful_rnn:
            config_lines.extend([
                '  {',
                '    name: "proprioceptive"',
                '    data_type: TYPE_FP32',
                f'    dims: [ {batch_dim}, 1, {proprio_dim} ]',
                '  },',
            ])
        else:
            config_lines.extend([
                '  {',
                '    name: "proprioceptive"',
                '    data_type: TYPE_FP32',
                f'    dims: [ {batch_dim}, {proprio_dim} ]',
                '  }',
            ])
        
        # Hidden state input for stateful RNN
        # IMPORTANT: Hidden state is now BATCH-FIRST: (batch, num_layers, hidden_size)
        # This matches the updated ONNX export format for Triton compatibility
        if stateful_rnn:
            # Hidden state: (batch, num_layers, hidden_size) - BATCH FIRST
            config_lines.extend([
                '  {',
                '    name: "hidden_in"',
                '    data_type: TYPE_FP32',
                f'    dims: [ {batch_dim}, {num_layers}, {hidden_size} ]',
                '  }',
            ])
        else:
            # Remove trailing comma from proprioceptive
            config_lines[-1] = '  }'
    
    config_lines.append(']')
    
    # Outputs
    config_lines.append('output [')
    config_lines.extend([
        '  {',
        '    name: "action"',
        '    data_type: TYPE_FP32',
        f'    dims: [ {batch_dim}, {action_dim} ]',
        '  }',
    ])
    
    # Hidden state output for stateful RNN
    if model_type == "rnn" and stateful_rnn:
        # Need to add comma after action
        config_lines[-1] = '  },'
        # Hidden state output: (batch, num_layers, hidden_size) - BATCH FIRST
        config_lines.extend([
            '  {',
            '    name: "hidden_out"',
            '    data_type: TYPE_FP32',
            f'    dims: [ {batch_dim}, {num_layers}, {hidden_size} ]',
            '  }',
        ])
    
    config_lines.append(']')
    
    return '\n'.join(config_lines)


def save_triton_config(
    output_path: str,
    model_name: str,
    **kwargs
) -> str:
    """Generate and save Triton config.pbtxt file.
    
    Args:
        output_path: Path to save config.pbtxt (or directory)
        model_name: Name of the model
        **kwargs: Arguments passed to generate_triton_config
        
    Returns:
        Path to saved config file
    """
    config_content = generate_triton_config(model_name=model_name, **kwargs)
    
    # If output_path is a directory, save as config.pbtxt inside it
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "config.pbtxt")
    elif not output_path.endswith(".pbtxt"):
        output_path = output_path + ".pbtxt"
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    print(f"✓ Saved Triton config to {output_path}")
    return output_path


def create_triton_model_repository(
    repo_path: str,
    model_name: str,
    onnx_path: str,
    model_type: str = "normal",
    version: int = 1,
    **kwargs
) -> str:
    """Create a complete Triton model repository structure.
    
    Creates:
        repo_path/
            model_name/
                config.pbtxt
                1/
                    model.onnx
    
    Args:
        repo_path: Path to model repository root
        model_name: Name of the model
        onnx_path: Path to the ONNX model file
        model_type: One of "normal", "stacked", or "rnn"
        version: Model version number
        **kwargs: Arguments passed to generate_triton_config
        
    Returns:
        Path to the model directory
    """
    import shutil
    
    # Create directory structure
    model_dir = os.path.join(repo_path, model_name)
    version_dir = os.path.join(model_dir, str(version))
    os.makedirs(version_dir, exist_ok=True)
    
    # Copy ONNX model
    onnx_dest = os.path.join(version_dir, "model.onnx")
    shutil.copy2(onnx_path, onnx_dest)
    print(f"✓ Copied ONNX model to {onnx_dest}")
    
    # Generate and save config
    config_path = os.path.join(model_dir, "config.pbtxt")
    save_triton_config(
        output_path=config_path,
        model_name=model_name,
        model_type=model_type,
        **kwargs
    )
    
    return model_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate Triton Inference Server config.pbtxt for CloneLab models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
# Normal model config:
python triton_config.py --model_name policy --model_type normal \\
    --image_size 160 90 --image_channels 3 --depth_channels 1

# RNN model with depth only (stateful, fixed batch size - RECOMMENDED):
python triton_config.py --model_name rnn_depth --model_type rnn \\
    --image_size 160 90 --image_channels 0 --depth_channels 1 \\
    --hidden_size 128 --num_layers 2 --batch_size 1

# RNN model (stateless):
python triton_config.py --model_name rnn_stateless --model_type rnn \\
    --image_size 160 90 --image_channels 1 --depth_channels 1 \\
    --stateless

# Create full Triton model repository:
python triton_config.py --model_name my_policy --model_type rnn \\
    --onnx_path policy.onnx --repo_path ./model_repository \\
    --image_channels 0 --depth_channels 1 --batch_size 1

NOTE: For RNN models, using --batch_size 1 is RECOMMENDED to avoid Triton
issues with dynamic dimensions in hidden state tensors.
        """
    )
    
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--model_type", type=str, required=True, 
                        choices=["normal", "stacked", "rnn"], help="Model type")
    parser.add_argument("--output", type=str, default=None, 
                        help="Output path for config.pbtxt (default: <model_name>.pbtxt)")
    
    # Optional: create full repository
    parser.add_argument("--onnx_path", type=str, default=None,
                        help="Path to ONNX model (if provided, creates full repository)")
    parser.add_argument("--repo_path", type=str, default="./model_repository",
                        help="Path to Triton model repository (default: ./model_repository)")
    parser.add_argument("--version", type=int, default=1, help="Model version (default: 1)")
    
    # Model architecture
    parser.add_argument("--image_size", type=int, nargs=2, default=[160, 90],
                        help="Image size [height, width] (default: 160 90)")
    parser.add_argument("--image_channels", type=int, default=3,
                        help="Number of image channels, 0 to disable (default: 3)")
    parser.add_argument("--depth_channels", type=int, default=1,
                        help="Number of depth channels, 0 to disable (default: 1)")
    parser.add_argument("--proprio_dim", type=int, default=3,
                        help="Proprioceptive input dimension (default: 3)")
    parser.add_argument("--action_dim", type=int, default=2,
                        help="Action dimension (default: 2)")
    
    # RNN specific
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="GRU hidden size (default: 128)")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of GRU layers (default: 2)")
    parser.add_argument("--stateless", action="store_true",
                        help="RNN is stateless (no hidden state I/O)")
    
    # Triton specific
    parser.add_argument("--max_batch_size", type=int, default=0,
                        help="Max batch size, 0 for model to handle batching (default: 0)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Fixed batch size. RECOMMENDED for RNN models to avoid Triton issues (default: dynamic)")
    parser.add_argument("--platform", type=str, default="onnxruntime_onnx",
                        help="Triton platform (default: onnxruntime_onnx)")
    
    args = parser.parse_args()
    
    config_kwargs = {
        "model_type": args.model_type,
        "image_size": args.image_size,
        "image_channels": args.image_channels,
        "depth_channels": args.depth_channels,
        "proprio_dim": args.proprio_dim,
        "action_dim": args.action_dim,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "stateful_rnn": not args.stateless,
        "max_batch_size": args.max_batch_size,
        "batch_size": args.batch_size,
        "platform": args.platform,
    }
    
    if args.onnx_path:
        # Create full repository
        create_triton_model_repository(
            repo_path=args.repo_path,
            model_name=args.model_name,
            onnx_path=args.onnx_path,
            **config_kwargs
        )
    else:
        # Just generate config file
        output_path = args.output or f"{args.model_name}.pbtxt"
        save_triton_config(
            output_path=output_path,
            model_name=args.model_name,
            **config_kwargs
        )
        
        # Print the generated config
        print("\nGenerated config:")
        print("-" * 40)
        print(generate_triton_config(model_name=args.model_name, **config_kwargs))


if __name__ == "__main__":
    main()
