# YOLOv8 + CLIP Source Snapshot

This directory contains code from two related but different deployment
architectures found in the workspace.

## 1. SPAR / text-conditioned YOLOv8

Files:

- `clip_detect_enhanced.py`
- `roi_clip_branch.py`
- `yolov8-spar-fcm-pzc-simple.yaml`
- `export_spar_onnx.py`
- `spar_onnx_inference.py`
- `onnx_cuda_inference.py`
- `export_error_adaptive_max_pool.log`

SPAR uses CLIP-derived text embeddings in the detection head. The text encoder
can run once and its embeddings can be frozen before ONNX export. The resulting
ONNX normally takes only an image input.

Two successful SPAR ONNX exports were found:

- 4-class defect model: input `[1,3,1024,1024]`, output `[1,8,87040]`.
- 10-class VisDrone model: input `[1,3,1024,1024]`, output `[1,14,21504]`.

An earlier export failed because `adaptive_max_pool2d` did not have an
accessible input size. The successful exports are after that failure; the log is
retained as evidence of the operator compatibility problem.

## 2. YOLO detection plus CLIP image re-ranking

Files:

- `yolo_clip_video_inference.py`
- `clip_rerank.py`
- `spar_regionclip_model.py`
- `spar_regionclip_fully_integrated.py`
- `clip_text.py`
- `train_spar_regionclip_deployment_package.py`

This path detects boxes with YOLO, crops each region, runs the CLIP image encoder,
compares it with text embeddings, and fuses/re-ranks the result. The CLIP image
encoder is a separate model and is not embedded in the successful SPAR ONNX.

`roi_clip_branch.py` is a third variant that attempts to run ROI-CLIP inside the
YOLO head. It uses `torchvision.ops.roi_align`, which requires ONNX/TensorRT
compatibility work and was disabled in the packaged simple model configuration.

## Missing deployment artifacts

No CLIP-specific ONNX model, TensorRT engine, INT8 calibration cache or complete
quantization evaluation was found in the original project directories.

Read `../../docs/yolov8_clip_quantization.md` for the proposed deployment split,
quantization strategy and evaluation metrics.

