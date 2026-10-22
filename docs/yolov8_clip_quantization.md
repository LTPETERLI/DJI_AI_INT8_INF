# YOLOv8 + CLIP 量化与推理部署梳理

## 1. 结论

YOLOv8 + CLIP 不能简单理解为“一个 YOLO engine 就能完成的模型”。工作区中实际
存在三种相关结构：

1. **SPAR / 文本条件 YOLOv8**：CLIP 只负责生成类别文本嵌入，嵌入冻结后作为
   YOLO 检测头的一部分使用。
2. **YOLO 检测 + CLIP 裁剪复筛**：YOLO 产生检测框，再将每个 ROI 裁剪出来送入
   CLIP 图像编码器，与文本嵌入比较后融合或重排。
3. **ROI-CLIP 内嵌检测头**：在 YOLO Head 内部做 RoIAlign、CLIP 空间投影与融合。

当前工作区已经有第 1 类的成功 ONNX，但没有发现完整的 CLIP TensorRT engine、
INT8 校准缓存或端到端量化报告。第 2、3 类仍以 PyTorch 推理为主。

因此不能把已有的 SPAR ONNX 直接称为“完整 YOLOv8 + CLIP 量化模型”。它更准确
的名称是“带冻结文本嵌入的 SPAR YOLOv8 检测模型”。

## 2. 已找到的实际产物

### 2.1 四缺陷 SPAR ONNX

文件：

```text
/home/peter/PycharmProjects/analogical_exclude/ultralytics/best.onnx
```

接口：

```text
input:  images  [1, 3, 1024, 1024]  float32
output: output0 [1, 8, 87040]       float32
```

元数据：

```text
Ultralytics YOLOv8x-spar model
4 classes:
  Broken_strands
  loose_strands
  foreign_body_entanglement
  rust
```

`8 = 4 个框参数 + 4 个类别`。该模型只有一个图像输入，说明 CLIP 文本编码和
文本嵌入准备已经发生在导出前，导出的计算图不包含 CLIP 图像编码器。

文件约 262 MB，FP32，静态输入。目前没有对应的 TensorRT engine 或 INT8 版本。

### 2.2 VisDrone SPAR ONNX

文件：

```text
/home/peter/PycharmProjects/Analogical-Reasoning-main/Analogical/yolov8x-spar.onnx
```

接口：

```text
input:  images  [1, 3, 1024, 1024]  float32
output: output0 [1, 14, 21504]      float32
```

这是 10 类 VisDrone SPAR 检测输出，同样没有 CLIP 图像编码器输入，也没有对应
engine。

### 2.3 导出失败记录

`source/yolov8_clip/export_error_adaptive_max_pool.log` 记录了早期 ONNX 导出
失败：

```text
Unsupported: ONNX export of operator adaptive_max_pool2d,
input size not accessible.
```

问题位于 `ImagePoolingAttn` 路径。成功 ONNX 的日期晚于该失败记录，说明后来
找到了一种固定输入尺寸或改写模块的方式，但当前快照中没有一条清晰的最终
导出脚本记录所有修复。

## 3. 三条部署路线

### 3.1 只部署 SPAR 文本条件 YOLOv8

这是当前最容易进入 TensorRT 的路线，因为已有成功 ONNX。

流程：

```text
best.pt
  -> set_classes(classes)
  -> 固定输入尺寸，导出 FP32 ONNX
  -> TensorRT FP16 engine
  -> TensorRT INT8 engine
```

优点：

- CLIP 文本编码只需离线运行一次；
- 部署 engine 只包含视觉检测网络；
- 输入接口简单，只有一个图像 tensor。

限制：

- 部署后不能任意改类别文本；
- 类别或 prompt 改变后必须重新生成文本嵌入并重新导出；
- 它不包含 CLIP 对每个候选框的图像级复核。

这是当前应优先量化的目标模型。

### 3.2 两阶段 YOLO + CLIP 复筛

当前复筛代码大致执行：

```text
YOLO image
  -> boxes
  -> crop with context
  -> CLIP image encoder per crop
  -> compare with precomputed text embeddings
  -> fuse/rerank/select
```

当前代码的问题：

- CLIP 以 PyTorch 运行；
- 每个检测框单独处理，batch 利用率低；
- 部分脚本在每个框的循环内重复初始化 CLIP 模型；
- 没有 TensorRT engine；
- 没有对 CLIP 图像编码器做 INT8；
- 没有评估 CLIP 复筛后的端到端漏检和误检。

部署时应拆成：

```text
Engine A: YOLO / SPAR detector
Engine B: CLIP image encoder
CPU/GPU postprocess: crop batching + matrix multiply + fusion
Offline: CLIP text embeddings
```

文本嵌入可以预计算后保存为 `.npy` 或编译进辅助模块，不需要每次推理重新运行
文本 Transformer。

### 3.3 ROI-CLIP 内嵌 YOLO Head

该路线使用：

- `torchvision.ops.roi_align`
- 动态候选框数量；
- 多尺度 ROI 特征；
- 图像嵌入和文本嵌入的点积融合。

该分支在当前简单配置中被关闭：

```yaml
use_roi_clip_branch: false
```

它不适合直接交给普通 TensorRT 导出，原因包括：

1. RoIAlign 是算子兼容点；
2. ROI 数量动态变化；
3. TensorRT 对动态 shape 需要 optimization profile；
4. 需固定 top-K 或 padding/mask；
5. 训练态和推理态计算路径不同；
6. CLIP 文本嵌入仍可能来自图外。

如必须使用内嵌 ROI 分支，需要：

- 用 ONNX opset 支持的 ROIAlign 或 TensorRT 插件替换；
- 固定每帧最多 `K` 个 ROI；
- 不足 `K` 个时补零并保存有效 mask；
- 对多尺度 feature 使用固定 tensor shape；
- 单独验证 ROI-CLIP 融合层 INT8 后是否稳定。

## 4. CLIP 本身怎么量化

CLIP 不是一个普通卷积检测头。部署时必须分开评估：

### 4.1 文本编码器

通常可以离线运行：

```text
class prompts -> CLIP text encoder -> normalized text embeddings
```

如果类别固定，推荐：

- 在训练/导出阶段计算；
- 保存文本嵌入；
- 部署时不再运行文本 Transformer。

这样可以完全消除文本编码器的推理延迟和文本侧量化误差。

### 4.2 图像编码器

两阶段方案中，CLIP 图像编码器通常是主要耗时项，尤其是 ViT-L/14@336px。

建议：

- 将所有 ROI 批量预处理为固定 shape；
- 一次 batch 送入 CLIP engine；
- 对比 FP32/FP16/INT8 的相似度排名和端到端检测结果；
- 如果 INT8 导致相似度 margin 漂移，应优先保留 LayerNorm、attention 输出和
  最终 projection 为 FP16。

### 4.3 融合与阈值

当前复筛依赖：

- `clip_margin`
- `fuse_alpha`
- `margin_tau`
- YOLO confidence

量化后即使框完全一致，CLIP margin 分布也可能整体平移。要先重新统计
FP32/FP16/INT8 的 margin 分布，再决定是否调整阈值。不能只沿用 FP32 的
`margin_tau` 后就判断 INT8 效果差。

## 5. 推荐量化顺序

1. 先量化 SPAR/YOLO 检测 engine。
2. 固定类别文本和 prompt 版本。
3. 对 YOLO 输出做 FP32/FP16/INT8 一致性评估。
4. 若使用两阶段 CLIP，再单独导出 CLIP image encoder。
5. CLIP 图像编码器先 FP16，再尝试 INT8。
6. 最后比较端到端 rerank 结果。

不要一开始把 YOLO、CLIP image encoder、RoIAlign 和融合逻辑全部塞进一个
动态 ONNX。拆图更容易量化、排查和做混合精度。

## 6. 量化指标

### 6.1 YOLO/SPAR 检测质量

- mAP50
- mAP50-95
- Precision
- Recall
- 每类 AP
- 固定工作点漏检率
- 每帧误检数
- 检测框中心误差

### 6.2 CLIP 复筛质量

- CLIP 相似度排序一致性
- Top-1 分类一致率
- `drone - max(distractor, background)` margin 分布变化
- 正样本/负样本分数 AUC
- 复筛造成的额外漏检率
- 复筛消除的误检数
- 最终保留框数量变化

### 6.3 端到端质量

- 最终 mAP
- 最终 Recall
- 最终每帧误检数
- 同一工作点下相比 FP32 的漏检增量

### 6.4 性能

必须分别记录：

- YOLO engine 延迟；
- CLIP image encoder 延迟；
- crop/resize 耗时；
- 文本嵌入是否已离线；
- 后处理/融合耗时；
- P50/P95/P99 和热稳定后的 FPS。

量化 YOLO 后，如果 CLIP 占端到端 70%-90% 时间，整体加速可能很小。必须看
分阶段 profile，而不能只报告 YOLO engine 的毫秒数。

## 7. KL/JS 散度在 CLIP 模型中怎么用

可比较：

- YOLO 特征图和检测头输出；
- CLIP 图像编码器中间 activation；
- CLIP 图像 embedding；
- 最终 text-image similarity 分布；
- margin 分布。

但 CLIP 最终关心的是排序和阈值，KL 只能帮助定位数值偏差。必须同时报告：

- Top-1 一致率；
- margin 位移；
- 漏检/误检变化；
- 端到端 mAP。

## 8. 当前缺失项

原始工作区没有找到：

- 完整 YOLOv8+CLIP 图像编码器的 TensorRT engine；
- CLIP image encoder 的 INT8 cache；
- 四缺陷 SPAR ONNX 的 engine；
- CLIP 复筛前后的量化评估表；
- 固定且可复现的单体导出脚本。

因此当前能做的是“基于已有 SPAR ONNX 和 CLIP 源码构建部署链”，而不是宣称
已经有一份可直接运行的 INT8 单体模型。

