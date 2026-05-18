from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Export CloneRL policies.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    onnx = subparsers.add_parser("onnx", help="Export a PyTorch actor checkpoint to ONNX.")
    onnx.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint directory or actor checkpoint file.")
    onnx.add_argument("--output", type=Path, required=True, help="Output artifact directory.")
    onnx.add_argument("--export-config", "--export_config", dest="export_config", type=Path, default=None)
    onnx.add_argument("--checkpoint-name", "--checkpoint_name", dest="checkpoint_name", default=None)
    onnx.add_argument("--device", default=None)
    onnx.add_argument("--opset-version", "--opset_version", dest="opset_version", type=int, default=17)

    trt = subparsers.add_parser("tensorrt", help="Build a TensorRT engine from an ONNX model.")
    trt.add_argument("--onnx", type=Path, required=True)
    trt.add_argument("--output", type=Path, required=True, help="Output .engine path.")
    trt.add_argument("--fp16", action="store_true")
    trt.add_argument("--workspace-gb", "--workspace_gb", dest="workspace_gb", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    from CloneRL.export.onnx import export_onnx
    from CloneRL.export.policy import load_export_config
    from CloneRL.export.tensorrt import build_tensorrt_engine

    if args.command == "onnx":
        export_config = load_export_config(args.checkpoint, args.export_config)

        onnx_path = export_onnx(
            checkpoint=args.checkpoint,
            export_config=export_config,
            output_dir=args.output,
            checkpoint_name=args.checkpoint_name,
            device=args.device or _default_device(),
            opset_version=args.opset_version,
        )
        print(f"[INFO] Exported ONNX: {onnx_path}")
        return onnx_path

    try:
        engine_path = build_tensorrt_engine(
            onnx_path=args.onnx,
            engine_path=args.output,
            fp16=args.fp16,
            workspace_size_gb=args.workspace_gb,
        )["engine_path"]
    except RuntimeError as exc:
        raise SystemExit(f"[ERROR] {exc}") from None
    print(f"[INFO] Exported TensorRT engine: {engine_path}")
    return engine_path


def _default_device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"
