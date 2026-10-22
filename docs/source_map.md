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

## YOLOv8 + CLIP snapshots

| Packaged location | Original source |
|---|---|
| `source/yolov8_clip/roi_clip_branch.py` | `/home/peter/PycharmProjects/analogical_exclude-1/ultralytics/nn/modules/roi_clip_branch.py` |
| `source/yolov8_clip/clip_detect_enhanced.py` | `/home/peter/PycharmProjects/analogical_exclude-1/ultralytics/nn/modules/clip_detect_enhanced.py` |
| `source/yolov8_clip/yolov8-spar-fcm-pzc-simple.yaml` | `/home/peter/PycharmProjects/analogical_exclude-1/ultralytics/cfg/models/v8/yolov8-spar-fcm-pzc-simple.yaml` |
| `source/yolov8_clip/export_spar_onnx.py` | `/home/peter/PycharmProjects/analogical_exclude/ONNX-OTPUT.py` |
| `source/yolov8_clip/spar_onnx_inference.py` | `/home/peter/PycharmProjects/analogical_exclude/ONNX-tuili.py` |
| `source/yolov8_clip/onnx_cuda_inference.py` | `/home/peter/PycharmProjects/analogical_exclude/pt-tuili-1.py` |
| `source/yolov8_clip/export_error_adaptive_max_pool.log` | `/home/peter/PycharmProjects/analogical_exclude/export_err.log` |
| `source/yolov8_clip/yolo_clip_video_inference.py` | `/home/peter/PycharmProjects/Analogical-Reasoning-main/Analogical/yolo_clip_video_inference.py` |
| `source/yolov8_clip/clip_rerank.py` | `/home/peter/PycharmProjects/Analogical-Reasoning-main/Analogical/clip_rerank.py` |
| `source/yolov8_clip/spar_regionclip_fully_integrated.py` | `/home/peter/PycharmProjects/Analogical-Reasoning-main/Analogical_serve/spar_regionclip_fully_integrated.py` |
| `source/yolov8_clip/spar_regionclip_model.py` | `/home/peter/PycharmProjects/Analogical-Reasoning-main/Analogical_serve/analogical/spar_regionclip_model.py` |
| `source/yolov8_clip/clip_text.py` | `/home/peter/PycharmProjects/Analogical-Reasoning-main/Analogical_serve/analogical/models/clip_text.py` |
| `source/yolov8_clip/train_spar_regionclip_deployment_package.py` | `/home/peter/PycharmProjects/Analogical-Reasoning-main/Analogical_serve/train_comprehensive_spar_regionclip_1536_optimized_updated4_deployment_package.py` |

## Model artifacts

| Artifact | Original path |
|---|---|
| YOLO26n weights | `/home/peter/dji/train46-yolo26n-sangu/weights/best.pt` |
| YOLO26n ONNX | `/home/peter/dji/sangu-weight/best-yolo26.onnx` |
| YOLO26n TensorRT engine | `/home/peter/dji/LZP_PSDK/customed_yolo_model/best-yolo26.engine` |

Exact sizes and SHA-256 hashes are in `artifacts/MODEL_ARTIFACTS.md`.

