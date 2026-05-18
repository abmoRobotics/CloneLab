from __future__ import annotations

import importlib
from typing import Any


def load_model_factory(spec: str):
    if ":" in spec:
        module_name, object_name = spec.split(":", 1)
    else:
        module_name, object_name = spec.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def build_model(factory_spec: str, config: dict[str, Any], device: str | None = None):
    model = load_model_factory(factory_spec)(**config)
    return model.to(device) if device is not None else model
