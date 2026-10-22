# PSDK Integration Notes

This snapshot preserves the current ROS2 liveview integration.

Known issues to address before performance measurement:

1. `ai_detect.h` hardcodes an obsolete `/home/dji/...` engine path.
2. The current postprocessor assumes an old `[1,4+nc,8400]` layout.
3. YOLO26 ONNX currently outputs `[1,300,6]` with NMS included.
4. The CUDA preprocess pads with `128`; verify the value used during training,
   ONNX validation and calibration.
5. The preprocess synchronizes after every frame.
6. Payload-SDK proprietary headers/libraries are not included in this package.
7. The detector and camera/gimbal consumers currently use different topic and
   message contracts in this workspace.

Do not use accuracy or latency results from this integration until item 2 is
fixed and verified against the ONNX output.

