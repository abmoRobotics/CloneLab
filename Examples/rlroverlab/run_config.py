from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


CONFIG_SECTIONS = {
    "algorithm",
    "data",
    "dataset",
    "eval",
    "frame_stack",
    "frame_stacking",
    "models",
    "trainer",
    "wandb",
}


def config_path(name: str) -> Path:
    return Path(__file__).with_name("configs") / name


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    default_config: str | Path | None = None,
    required: tuple[str, ...] = (),
) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args()

    if not any("--config" in action.option_strings for action in parser._actions):
        parser.add_argument(
            "--config",
            type=str,
            default=config_args.config,
            help="Optional YAML run config. Values override the script default config.",
        )

    config: dict[str, Any] = {}
    if default_config is not None:
        config = _deep_update(config, load_yaml_config(default_config))
    if config_args.config is not None:
        config = _deep_update(config, load_yaml_config(config_args.config))
    if config:
        parser.set_defaults(**flatten_run_config(config))

    args = parser.parse_args()
    for name in required:
        if getattr(args, name, None) is None:
            parser.error(f"--{name} is required unless set in --config")
    return args


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML configs require PyYAML. Install CloneRL with the updated requirements.") from exc

    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Expected YAML object in {path}, got {type(config).__name__}")
    return config


def flatten_run_config(config: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    _merge_dataset_config(flat, config.get("dataset", config.get("data")))
    _merge_section(flat, config.get("trainer"))
    _merge_section(flat, config.get("algorithm"))
    _merge_section(flat, config.get("eval"))
    _merge_frame_stack_config(flat, config.get("frame_stack", config.get("frame_stacking")))
    _merge_wandb_config(flat, config.get("wandb"))
    _merge_models_config(flat, config.get("models"))

    for key, value in config.items():
        if key not in CONFIG_SECTIONS:
            flat[key] = value
    return flat


def load_model_config(value: str | dict[str, Any] | None, defaults: dict[str, Any]) -> dict[str, Any]:
    config = dict(defaults)
    if value is None:
        return config
    if isinstance(value, dict):
        config.update(value)
        return config

    path = Path(value)
    if path.suffix.lower() in {".yaml", ".yml"}:
        user_config = load_yaml_config(path)
    else:
        with open(path, encoding="utf-8") as file:
            user_config = json.load(file)
    if not isinstance(user_config, dict):
        raise TypeError(f"Expected config object in {path}, got {type(user_config).__name__}")
    config.update(user_config)
    return config


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _merge_section(flat: dict[str, Any], section: Any) -> None:
    if section is None:
        return
    if not isinstance(section, dict):
        raise TypeError(f"Expected config section to be an object, got {type(section).__name__}")
    flat.update(section)


def _merge_dataset_config(flat: dict[str, Any], section: Any) -> None:
    if section is None:
        return
    if isinstance(section, str):
        flat["dataset"] = section
        return
    if not isinstance(section, dict):
        raise TypeError(f"Expected dataset config to be string or object, got {type(section).__name__}")

    aliases = {
        "path": "dataset",
        "train": "dataset",
        "train_path": "dataset",
        "val": "val_dataset",
        "validation": "val_dataset",
        "val_path": "val_dataset",
    }
    for key, value in section.items():
        flat[aliases.get(key, key)] = value


def _merge_frame_stack_config(flat: dict[str, Any], section: Any) -> None:
    if section is None:
        return
    if isinstance(section, bool):
        flat["frame_stacking"] = section
        return
    if not isinstance(section, dict):
        raise TypeError(f"Expected frame_stack config to be bool or object, got {type(section).__name__}")

    aliases = {"enabled": "frame_stacking"}
    for key, value in section.items():
        flat[aliases.get(key, key)] = value


def _merge_wandb_config(flat: dict[str, Any], section: Any) -> None:
    if section is None:
        return
    if not isinstance(section, dict):
        raise TypeError(f"Expected wandb config to be an object, got {type(section).__name__}")

    aliases = {"project": "wandb_project", "mode": "wandb_mode"}
    for key, value in section.items():
        flat[aliases.get(key, key)] = value


def _merge_models_config(flat: dict[str, Any], section: Any) -> None:
    if section is None:
        return
    if not isinstance(section, dict):
        raise TypeError(f"Expected models config to be an object, got {type(section).__name__}")

    for name in ("actor", "critic", "value"):
        model_config = section.get(name)
        if model_config is None:
            continue
        if not isinstance(model_config, dict):
            raise TypeError(f"Expected models.{name} to be an object, got {type(model_config).__name__}")
        if "factory" in model_config:
            flat[f"{name}_factory"] = model_config["factory"]
        if "config" in model_config:
            flat[f"{name}_config"] = model_config["config"]
