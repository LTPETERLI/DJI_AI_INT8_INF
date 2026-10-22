# Standalone Runner Notes

This is a source snapshot of the PSDK-side TensorRT runner.

Current behavior:

- loads an existing `.engine` when passed an engine path;
- builds and saves an engine when passed an `.onnx` path;
- enables FP16 through `isFP16 = true`;
- warms up the engine 10 times;
- runs CUDA preprocessing and CUDA copy/synchronization.

Current limitations:

- no INT8 builder flag or calibrator;
- old YOLO output decoding;
- hardcoded defaults in several entry points;
- no portable configuration file;
- no checksum or TensorRT version check for an existing engine.

The legacy `export_legacy.py` contains obsolete absolute paths and should not be
used unchanged. Prefer `tools/export_yolo26_onnx.py`.

