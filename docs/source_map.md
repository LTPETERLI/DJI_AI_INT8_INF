# Source Map

This file records where each packaged snapshot came from. The source tree was
read-only during packaging.

## Training records

| Packaged file | Original path |
|---|---|
| `training/yolo26n_train46_args.yaml` | `/home/peter/dji/train46-yolo26n-sangu/args.yaml` |
| `training/yolo26n_train46_results.csv` | `/home/peter/dji/train46-yolo26n-sangu/results.csv` |

The training script and dataset were not present in the original workspace.
The exact training configuration is recoverable from `args.yaml`.

## Current PSDK TensorRT runner

| Packaged location | Original source |
|---|---|
| `source/standalone_trt_runner/` | `/home/peter/dji/LZP_PSDK/yolov11-tensorrt/` |

Copied files include the build script, runner, export example, TensorRT
implementation and CUDA preprocessing. Build outputs, duplicate backups and
model binaries were excluded.

## ROS2/PSDK liveview integration

| Packaged location | Original source |
|---|---|
| `source/psdk_ros2_yolo_detect/CMakeLists.txt` | `/home/peter/dji/LZP_PSDK/ros2_workplace/src/yolo_detect/CMakeLists.txt` |
| `source/psdk_ros2_yolo_detect/ai_detect.h` | `/home/peter/dji/LZP_PSDK/ros2_workplace/src/yolo_detect/include/ai_detect.h` |
| `source/psdk_ros2_yolo_detect/main.cpp` | `/home/peter/dji/LZP_PSDK/ros2_workplace/src/yolo_detect/src/main.cpp` |
| `source/psdk_ros2_yolo_detect/yolo/` | `/home/peter/dji/LZP_PSDK/ros2_workplace/src/yolo_detect/yolo/` |

The DJI Payload SDK library, platform HAL and proprietary binaries are not
included. Restore those dependencies from the licensed Payload-SDK package to
build this integration.

## INT8 reference implementation

| Packaged location | Original source |
|---|---|
| `source/int8_reference_cpp/` | `/home/peter/PycharmProjects/TensorRT-YOLO11-main/C++/detect/` |

Only source, headers, README, CMake and license are copied. Images, ONNX models
and generated output were excluded.

## Model artifacts

| Artifact | Original path |
|---|---|
| YOLO26n weights | `/home/peter/dji/train46-yolo26n-sangu/weights/best.pt` |
| YOLO26n ONNX | `/home/peter/dji/sangu-weight/best-yolo26.onnx` |
| YOLO26n TensorRT engine | `/home/peter/dji/LZP_PSDK/customed_yolo_model/best-yolo26.engine` |

Exact sizes and SHA-256 hashes are in `artifacts/MODEL_ARTIFACTS.md`.

