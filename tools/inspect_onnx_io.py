#!/usr/bin/env python3
"""Print ONNX Runtime input/output contracts and Ultralytics metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    print(f"model: {model_path}")
    for item in session.get_inputs():
        print(f"input:  {item.name} {item.shape} {item.type}")
    for item in session.get_outputs():
        print(f"output: {item.name} {item.shape} {item.type}")

    metadata = session.get_modelmeta().custom_metadata_map
    for key in ("description", "names", "task", "imgsz", "batch", "args", "date"):
        if key in metadata:
            print(f"{key}: {metadata[key]}")


if __name__ == "__main__":
    main()

