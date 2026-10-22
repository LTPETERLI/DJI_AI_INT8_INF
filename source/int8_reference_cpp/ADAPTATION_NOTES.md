# INT8 Reference Adaptation Notes

This snapshot is a generic YOLO11 detector copied from
`TensorRT-YOLO11-main/C++/detect`. It demonstrates:

- `IInt8EntropyCalibrator2`;
- calibration image preprocessing;
- `BuilderFlag::kINT8`;
- serialized engine creation;
- CUDA preprocessing, decode and NMS.

It is not yet configured for the current YOLO26 defect model.

Required changes:

1. Set `kNumClass` to the defect class count.
2. Set `kInputH` and `kInputW` to the production input size.
3. Replace COCO class names.
4. Point `onnxFile` to the new ONNX model.
5. Create a representative calibration directory and set
   `calibrationDataPath`.
6. Set `bINT8Mode = true`.
7. Match calibration preprocessing exactly to runtime preprocessing.
8. Adapt the output decoder to YOLO26 `[1,300,6]` if the model is exported in
   end-to-end mode.
9. Port TensorRT 8 calls to the target TensorRT version if needed.
10. Build and evaluate only on Manifold 3.

The current CUDA preprocess allocates and frees device memory every frame and
calls `cudaDeviceSynchronize()`. For production latency, move allocations to
initialization and remove unnecessary global synchronizations.

