# Model Artifact Manifest

Large binaries are intentionally excluded from Git. Their source paths, sizes
and SHA-256 hashes are recorded for traceability.

| Artifact | Size (bytes) | SHA-256 |
|---|---:|---|
| YOLO26n `best.pt` | 5,455,934 | `09867373c3a7cc32c7ab88c343296185ad2e06b0d0110c81319a8f943af2b779` |
| YOLO26n FP32 ONNX | 10,309,354 | `710c32cb3f212efc259a8b8a4836136617a900fc1059320b4f11330a94546fd7` |
| Current PSDK TensorRT engine | 7,407,617 | `2d66fba34d6031e9c154e3bb826618882fb0761475ad0726c3de52011f861952` |

ONNX contract:

```text
input  images  float32  [1, 3, 1280, 1280]
output output0 float32  [1, 300, 6]
```

The engine is device-specific and must not be treated as portable. Rebuild it on
Manifold 3 using the production TensorRT environment.

## YOLOv8 + CLIP related artifacts

The following files are not committed:

| Artifact | Size (bytes) | Contract |
|---|---:|---|
| Four-class SPAR ONNX | 262,116,319 | input `[1,3,1024,1024]`, output `[1,8,87040]` |
| VisDrone SPAR ONNX | 275,531,101 | input `[1,3,1024,1024]`, output `[1,14,21504]` |

Both are FP32 static ONNX files. No corresponding TensorRT engines or INT8
calibration caches were found.

