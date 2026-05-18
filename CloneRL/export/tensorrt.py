from __future__ import annotations

from pathlib import Path
from typing import Any


def build_tensorrt_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    fp16: bool = False,
    workspace_size_gb: float = 2.0,
) -> dict[str, Any]:
    """Build a TensorRT engine from an ONNX model.

    TensorRT engines are target-specific. Build them in the same CUDA/TensorRT
    runtime family that will run the exported policy.
    """

    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT Python bindings are not installed. Install TensorRT in "
            "the target runtime or export ONNX only."
        ) from exc

    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as file:
        parsed = parser.parse(file.read())
    if not parsed:
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parsing failed:\n" + "\n".join(errors))

    builder_config = builder.create_builder_config()
    _set_workspace_limit(trt, builder_config, int(workspace_size_gb * (1024**3)))

    fp16_enabled = False
    if fp16:
        if getattr(builder, "platform_has_fast_fp16", False):
            builder_config.set_flag(trt.BuilderFlag.FP16)
            fp16_enabled = True
        else:
            raise RuntimeError("Requested FP16 TensorRT export, but this platform does not report fast FP16 support.")

    serialized_engine = _build_serialized_engine(builder, network, builder_config)

    with open(engine_path, "wb") as file:
        file.write(bytes(serialized_engine))

    return {
        "engine_path": engine_path,
        "fp16": fp16_enabled,
        "workspace_size_gb": workspace_size_gb,
    }


def _set_workspace_limit(trt, builder_config, workspace_bytes: int) -> None:
    if hasattr(builder_config, "set_memory_pool_limit"):
        builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        builder_config.max_workspace_size = workspace_bytes


def _build_serialized_engine(builder, network, builder_config):
    if hasattr(builder, "build_serialized_network"):
        engine = builder.build_serialized_network(network, builder_config)
        if engine is None:
            raise RuntimeError("TensorRT failed to build a serialized engine.")
        return engine

    engine = builder.build_engine(network, builder_config)
    if engine is None:
        raise RuntimeError("TensorRT failed to build an engine.")
    serialized = engine.serialize()
    if serialized is None:
        raise RuntimeError("TensorRT failed to serialize the engine.")
    return serialized
