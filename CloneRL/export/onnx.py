from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from CloneRL.export.policy import input_specs, load_actor, make_wrapper, output_specs


def export_onnx(
    checkpoint: str | Path,
    export_config: dict,
    output_dir: str | Path,
    checkpoint_name: str | None = None,
    device: str = "cpu",
    onnx_name: str = "policy.onnx",
    opset_version: int = 17,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / onnx_name
    sample_inputs_path = output_dir / "sample_inputs.npz"

    model, resolved_checkpoint = load_actor(checkpoint, export_config, device, checkpoint_name)
    wrapper = make_wrapper(model, export_config).to(device).eval()
    inputs = _sample_inputs(export_config, device)
    in_specs = input_specs(export_config)
    out_specs = output_specs(export_config)
    input_names = [spec["name"] for spec in in_specs]
    output_names = [spec["name"] for spec in out_specs]
    export_kwargs = {
        "input_names": input_names,
        "output_names": output_names,
        "opset_version": opset_version,
        "export_params": True,
        "do_constant_folding": True,
        "dynamo": False,
    }
    with torch.no_grad():
        try:
            torch.onnx.export(wrapper, tuple(inputs), str(onnx_path), **export_kwargs)
        except TypeError as exc:
            if "dynamo" not in str(exc):
                raise
            export_kwargs.pop("dynamo", None)
            torch.onnx.export(wrapper, tuple(inputs), str(onnx_path), **export_kwargs)

    _save_sample_inputs(sample_inputs_path, input_names, inputs)
    _check_onnx_model(onnx_path)
    _write_manifest(
        output_dir / "manifest.json",
        export_config,
        resolved_checkpoint,
        in_specs,
        out_specs,
        onnx_path,
        sample_inputs_path,
    )

    return onnx_path


def _sample_inputs(export_config: dict, device: str) -> list[torch.Tensor]:
    return [
        torch.randn(*spec["shape"], device=device, dtype=torch.float32)
        for spec in input_specs(export_config)
    ]


def _check_onnx_model(path: Path) -> None:
    try:
        import onnx
    except ImportError:
        return
    model = onnx.load(str(path))
    onnx.checker.check_model(model)


def _save_sample_inputs(path: Path, input_names: list[str], inputs: list[torch.Tensor]) -> None:
    arrays = {
        name: tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        for name, tensor in zip(input_names, inputs)
    }
    np.savez(path, **arrays)


def _write_manifest(
    path: Path,
    export_config: dict,
    resolved_checkpoint: Path,
    in_specs: list[dict],
    out_specs: list[dict],
    onnx_path: Path,
    sample_inputs_path: Path,
) -> None:
    payload = {
        "format_version": 1,
        "policy_type": export_config.get("policy_type", "feedforward"),
        "model_factory": export_config["model_factory"],
        "model_config": export_config.get("model_config", {}),
        "resolved_checkpoint": str(resolved_checkpoint),
        "inputs": in_specs,
        "outputs": out_specs,
        "onnx": onnx_path.name,
        "sample_inputs": sample_inputs_path.name,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
