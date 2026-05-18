from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from CloneRL.models.factory import load_model_factory


EXPORT_CONFIG_NAME = "export_config.json"


def load_export_config(checkpoint: str | Path, export_config: str | Path | None = None) -> dict[str, Any]:
    path = _resolve_export_config_path(checkpoint, export_config)
    with open(path, encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(config).__name__}")
    return config


def write_export_config(
    checkpoint_dir: str | Path,
    policy_type: str,
    model_factory: str,
    model_config: dict[str, Any],
    checkpoint_name: str = "final_model.pt",
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / EXPORT_CONFIG_NAME
    payload = {
        "format_version": 1,
        "policy_type": normalize_policy_type(policy_type),
        "model_factory": model_factory,
        "model_config": model_config,
        "checkpoint_name": checkpoint_name,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
    return path


def load_actor(
    checkpoint: str | Path,
    export_config: dict[str, Any],
    device: str,
    checkpoint_name: str | None = None,
):
    model_config = export_config.get("model_config", {})
    factory = load_model_factory(export_config["model_factory"])
    model = factory(**model_config).to(device).eval()
    checkpoint_path = resolve_checkpoint(
        checkpoint,
        checkpoint_name or export_config.get("checkpoint_name", "final_model.pt"),
    )
    state = _actor_state_dict(_torch_load(checkpoint_path, device))
    model.load_state_dict(state)
    return model, checkpoint_path


def make_wrapper(model: nn.Module, export_config: dict[str, Any]) -> nn.Module:
    policy_type = normalize_policy_type(export_config.get("policy_type", "feedforward"))
    if policy_type == "feedforward":
        return FeedForwardWrapper(model, export_config).eval()
    if policy_type == "recurrent":
        return RecurrentWrapper(model, export_config).eval()
    raise ValueError(f"Unsupported policy_type: {policy_type}")


def input_specs(export_config: dict[str, Any]) -> list[dict[str, Any]]:
    model_config = export_config.get("model_config", {})
    policy_type = normalize_policy_type(export_config.get("policy_type", "feedforward"))
    batch_size = 1
    sequence_length = 1
    image_channels = int(model_config.get("image_channels", 3))
    depth_channels = int(model_config.get("depth_channels", 1))
    proprio_dim = int(model_config.get("proprioception_channels", 3))
    hidden_size = int(model_config.get("hidden_size", 128))
    num_layers = int(model_config.get("num_layers", 2))
    h, w = _image_size(model_config)
    specs = []

    if policy_type == "feedforward":
        if image_channels > 0:
            specs.append(_spec("image", [batch_size, image_channels, h, w]))
        if depth_channels > 0:
            specs.append(_spec("depth", [batch_size, depth_channels, h, w]))
        specs.append(_spec("proprioceptive", [batch_size, proprio_dim]))
        return specs

    if image_channels > 0:
        specs.append(_spec("image", [batch_size, sequence_length, image_channels, h, w]))
    if depth_channels > 0:
        specs.append(_spec("depth", [batch_size, sequence_length, depth_channels, h, w]))
    specs.append(_spec("proprioceptive", [batch_size, sequence_length, proprio_dim]))
    specs.append(_spec("hidden_in", [batch_size, num_layers, hidden_size]))
    return specs


def output_specs(export_config: dict[str, Any]) -> list[dict[str, Any]]:
    model_config = export_config.get("model_config", {})
    action_dim = int(model_config.get("action_dim", 2))
    specs = [_spec("action", [1, action_dim])]
    if normalize_policy_type(export_config.get("policy_type", "feedforward")) == "recurrent":
        hidden_size = int(model_config.get("hidden_size", 128))
        num_layers = int(model_config.get("num_layers", 2))
        specs.append(_spec("hidden_out", [1, num_layers, hidden_size]))
    return specs


def normalize_policy_type(policy_type: str) -> str:
    aliases = {"normal": "feedforward", "ff": "feedforward", "rnn": "recurrent"}
    return aliases.get(policy_type.lower(), policy_type.lower())


class FeedForwardWrapper(nn.Module):
    def __init__(self, model: nn.Module, export_config: dict[str, Any]):
        super().__init__()
        model_config = export_config.get("model_config", {})
        self.model = model
        self.has_image = int(model_config.get("image_channels", 3)) > 0
        self.has_depth = int(model_config.get("depth_channels", 1)) > 0

    def forward(self, *inputs):
        state, _ = _state_from_inputs(inputs, self.has_image, self.has_depth)
        return _action_from_output(self.model(state))


class RecurrentWrapper(nn.Module):
    def __init__(self, model: nn.Module, export_config: dict[str, Any]):
        super().__init__()
        model_config = export_config.get("model_config", {})
        self.model = model
        self.has_image = int(model_config.get("image_channels", 3)) > 0
        self.has_depth = int(model_config.get("depth_channels", 1)) > 0

    def forward(self, *inputs):
        state, idx = _state_from_inputs(inputs, self.has_image, self.has_depth)
        hidden = inputs[idx].transpose(0, 1).contiguous()
        output = self.model(state, hidden)
        dist, hidden_out = output if isinstance(output, tuple) else (output, hidden)
        action = _action_from_output(dist)
        return action, hidden_out.transpose(0, 1).contiguous()


def resolve_checkpoint(checkpoint: str | Path, checkpoint_name: str) -> Path:
    checkpoint = Path(checkpoint)
    candidates = [checkpoint]
    if checkpoint.is_dir():
        candidates = [checkpoint / "actor" / checkpoint_name, checkpoint / checkpoint_name]
    resolved = next((path for path in candidates if path.exists()), None)
    if resolved is None:
        raise FileNotFoundError("Could not find checkpoint. Tried: " + ", ".join(str(path) for path in candidates))
    return resolved


def _resolve_export_config_path(checkpoint: str | Path, export_config: str | Path | None) -> Path:
    if export_config is not None:
        return Path(export_config)
    checkpoint = Path(checkpoint)
    candidates = [checkpoint / EXPORT_CONFIG_NAME] if checkpoint.is_dir() else [
        checkpoint.parent / EXPORT_CONFIG_NAME,
        checkpoint.parent.parent / EXPORT_CONFIG_NAME,
    ]
    resolved = next((path for path in candidates if path.exists()), None)
    if resolved is None:
        tried = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Could not find export_config.json. Tried: {tried}")
    return resolved


def _torch_load(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _actor_state_dict(raw_state: Any) -> dict[str, torch.Tensor]:
    state = raw_state
    if isinstance(state, dict):
        for key in ("state_dict", "actor", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"Expected checkpoint state dict, got {type(state).__name__}")
    if state and all(isinstance(key, str) and key.startswith("actor.") for key in state):
        state = {key.removeprefix("actor."): value for key, value in state.items()}
    return state


def _state_from_inputs(inputs, has_image: bool, has_depth: bool):
    state = {}
    idx = 0
    if has_image:
        state["image"] = inputs[idx]
        idx += 1
    if has_depth:
        state["depth"] = inputs[idx]
        idx += 1
    state["proprioceptive"] = inputs[idx]
    return state, idx + 1


def _action_from_output(output) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if hasattr(output, "mean"):
        output = output.mean
    if output.ndim == 3:
        output = output[:, -1, :]
    return output


def _image_size(model_config: dict[str, Any]) -> tuple[int, int]:
    size = model_config.get("image_size") or model_config.get("image_input_dim") or [160, 90]
    if len(size) != 2:
        raise ValueError(f"Expected image size with two values, got {size}")
    return int(size[0]), int(size[1])


def _spec(name: str, shape: list[int]) -> dict[str, Any]:
    return {"name": name, "shape": shape, "dtype": "float32"}
