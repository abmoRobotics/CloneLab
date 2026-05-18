"""Simple policy export helpers for CloneRL."""

__all__ = ["build_tensorrt_engine", "export_onnx", "write_export_config"]


def __getattr__(name: str):
    if name == "export_onnx":
        from CloneRL.export.onnx import export_onnx

        return export_onnx
    if name == "build_tensorrt_engine":
        from CloneRL.export.tensorrt import build_tensorrt_engine

        return build_tensorrt_engine
    if name == "write_export_config":
        from CloneRL.export.policy import write_export_config

        return write_export_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
