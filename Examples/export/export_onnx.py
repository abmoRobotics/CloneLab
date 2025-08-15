import torch
import torch.nn as nn
import argparse
import os
import sys

# Add project root to path to allow importing from Examples.
# This is needed if the project is not installed as a package.
# It assumes the script is in Examples/export/
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Examples.orbit.models import actor_gaussian_image

class OnnxWrapper(nn.Module):
    """
    A wrapper for the actor_gaussian_image model to make it compatible with ONNX export.
    The original model returns a distribution, which is not supported by ONNX.
    This wrapper extracts the mean of the distribution and returns it as a tensor.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, proprioceptive):
        """
        Forward pass for the wrapper.
        
        Args:
            image (torch.Tensor): The image input tensor.
            proprioceptive (torch.Tensor): The proprioceptive input tensor.
            
        Returns:
            torch.Tensor: The mean of the action distribution.
        """
        state = {"image": image, "proprioceptive": proprioceptive}
        distribution = self.model(state)
        return distribution.mean # In Normal distribution, mean is mu

def main():
    """
    Main function to export the model.
    """
    parser = argparse.ArgumentParser(description="Export a PyTorch model to ONNX.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model weights (.pt file). E.g. runs/some_run/checkpoints/best_model.pt")
    parser.add_argument("--output_name", type=str, required=True, help="Name of the output ONNX file. E.g. policy.onnx")
    parser.add_argument("--image_channels", type=int, default=1, help="Number of channels in the input image.")
    parser.add_argument("--proprio_dim", type=int, default=5, help="Dimension of the proprioceptive input.")

    args = parser.parse_args()

    # --- Model Loading ---
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # The model from the user's file is actor_gaussian_image
    # We need to instantiate it with the correct parameters.
    model = actor_gaussian_image(
        device=device,
        image_channels=args.image_channels
    ).to(device)

    # Load the weights
    print(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Wrap the model for ONNX export
    onnx_model = OnnxWrapper(model)

    # --- Dummy Input Creation ---
    # The conv_encoder in actor_gaussian_image expects a 224x224 input
    image_size = (224, 224)
    batch_size = 1
    dummy_image = torch.randn(batch_size, args.image_channels, image_size[0], image_size[1]).to(device)
    dummy_proprio = torch.randn(batch_size, args.proprio_dim).to(device)
    
    # --- ONNX Export ---
    output_path = args.output_name
    if not output_path.endswith(".onnx"):
        output_path += ".onnx"
        
    print(f"Exporting model to {output_path}...")

    torch.onnx.export(
        onnx_model,
        (dummy_image, dummy_proprio),
        output_path,
        input_names=["image", "proprioceptive"],
        output_names=["action"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "proprioceptive": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
        opset_version=11,
        verbose=False
    )

    print(f"Export complete. Model saved to {output_path}")

if __name__ == "__main__":
    main()
