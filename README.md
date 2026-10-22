# Manifold 3 YOLO Quantization and Deployment

This repository packages the relevant code and experiment records for deploying
defect-detection YOLO models on a DJI Manifold 3 payload computer.

The primary questions covered here are:

1. How the current `best.pt` model becomes ONNX and a TensorRT engine.
2. How INT8 post-training quantization should be added to the PSDK path.
3. Which metrics should decide whether a quantized model is acceptable.
4. What to tune when INT8 accuracy or latency is unsatisfactory.

## Current project findings

- The current defect model is YOLO26n, one class (`loose_brand` / 导线散股),
  trained at 1280x1280.
- The current ONNX is FP32, shape `[1, 3, 1280, 1280]`, and outputs
  `[1, 300, 6]` with end-to-end NMS in the graph.
- The existing PSDK TensorRT engine is built through code that enables FP16.
  No INT8 calibrator is connected in the current PSDK path.
- The current PSDK C++ postprocessor expects the older YOLO11-style
  `[1, 4+nc, 8400]` layout. It must be adapted before quantized YOLO26 models
  can be evaluated correctly.
- A complete INT8 reference implementation exists in
  `source/int8_reference_cpp`; it is a generic YOLO11 implementation and needs
  model-interface adaptation.

Read the full analysis in
[docs/quantization_and_deployment.md](docs/quantization_and_deployment.md).

## Repository layout

```text
docs/                          Analysis, metrics, tuning and source map
training/                      Exact YOLO26n training arguments and results
tools/                         ONNX export and inspection helpers
source/standalone_trt_runner/  PSDK-side TensorRT runner snapshot
source/psdk_ros2_yolo_detect/  ROS2 liveview inference integration snapshot
source/int8_reference_cpp/     INT8 EntropyCalibrator2 reference implementation
source/yolov8_clip/            SPAR/YOLOv8 + CLIP re-ranking source snapshots
artifacts/                     Hashes and provenance; binaries are not committed
```

## Important limitation

TensorRT engines are target-specific. Build and evaluate the engine on the
actual Manifold 3 device with the same TensorRT, CUDA and GPU architecture used
in production. An engine built on the x86 development host cannot be copied to
the Manifold 3.

## Reproducible entry points

Export a static FP32 ONNX model for conversion:

```bash
python tools/export_yolo26_onnx.py \
  --weights /path/to/best.pt \
  --imgsz 1280 \
  --output /path/to/best-yolo26.onnx
```

Inspect ONNX input/output contracts:

```bash
python tools/inspect_onnx_io.py /path/to/best-yolo26.onnx
```

See [source/int8_reference_cpp/ADAPTATION_NOTES.md](source/int8_reference_cpp/ADAPTATION_NOTES.md)
for the changes required before enabling its INT8 path on Manifold 3.

The separate YOLOv8 + CLIP analysis is in
[docs/yolov8_clip_quantization.md](docs/yolov8_clip_quantization.md).

## Provenance

The package copies selected source snapshots from the user's workspace. Original
paths and hashes are recorded in
[docs/source_map.md](docs/source_map.md). Large model files, datasets, DJI
binaries and TensorRT engines are intentionally excluded.

