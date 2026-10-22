#!/usr/bin/env python3
"""Export a static FP32 Ultralytics model to ONNX for TensorRT conversion."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to trained .pt file")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--output", required=True, help="Output .onnx path")
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable graph simplification",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    output.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights), task="detect")
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
        dynamic=False,
        half=False,
        int8=False,
        batch=1,
    )

    exported_path = Path(exported).resolve()
    if exported_path != output:
        output.write_bytes(exported_path.read_bytes())
    print(output)


if __name__ == "__main__":
    main()

