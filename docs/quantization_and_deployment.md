# Manifold 3 缺陷检测模型量化与部署梳理

## 1. 结论先行

当前工程已经具备以下能力：

- 训练 YOLO26/YOLO11 缺陷检测模型并得到 `best.pt`。
- 将 PyTorch 权重导出为静态 ONNX。
- 在 C++ 中从 ONNX 构建 TensorRT engine。
- 在 Manifold 3/ROS2/PSDK liveview 路径中执行 CUDA 预处理和 TensorRT 推理。
- 通过代码开启 TensorRT FP16。

当前工程还没有在 PSDK 主路径中真正启用 INT8：

- 当前 PSDK 的 `YOLOv11.cpp` 只调用 `BuilderFlag::kFP16`。
- 没有接入 `IInt8Calibrator`。
- 没有校准集清单和 `int8.cache`。
- 没有完整的 FP32/FP16/INT8 对比报告。

因此，“生成量化模型”的准确流程应是：

```text
best.pt
  -> FP32 static ONNX
  -> 在 Manifold 3 上用代表性图像校准
  -> TensorRT INT8 engine（必要时混合 FP16）
  -> 在目标设备上做质量、延迟和稳定性评估
```

不要把 ONNX 文件本身称为 INT8 模型。ONNX 是中间表示；INT8 精度主要是在
TensorRT 构建 engine 时确定的。当前 ONNX 是 FP32，engine 才包含目标设备的
层精度、kernel tactic 和序列化计划。

## 2. 当前模型与产物

### 2.1 当前已训练模型

当前可追溯的训练运行是 `train46-yolo26n-sangu`：

| 项目 | 值 |
|---|---|
| 模型 | YOLO26n |
| 任务 | detect |
| 类别 | 1 类：`loose_brand` / 导线散股 |
| 输入尺寸 | 1280x1280 |
| Epochs | 100 |
| Batch | 2 |
| AMP | true |
| 最佳 epoch | 74 |
| mAP50 | 0.93801 |
| mAP50-95 | 0.66364 |
| Precision | 0.91737 |
| Recall | 0.92661 |

训练参数在 `training/yolo26n_train46_args.yaml`，完整曲线数据在
`training/yolo26n_train46_results.csv`。

训练数据和训练启动脚本没有随当前工作区保存，`args.yaml` 中记录的数据路径是
外部机器路径 `/mnt/SDB/.../loose_brand_yolo/dataset.yaml`。因此当前仓库可以
复现导出和部署，但不能单独复现训练数据准备。

### 2.2 当前 ONNX 契约

当前 YOLO26 ONNX：

```text
input:  images  [1, 3, 1280, 1280]  float32
output: output0 [1, 300, 6]         float32
```

每一行输出为：

```text
[x1, y1, x2, y2, confidence, class_id]
```

这是端到端输出格式，NMS 已经在导出图内完成。

### 2.3 当前 PSDK engine

当前 PSDK 代码加载：

```text
/home/dji/DJI_workshop/LZP_PSDK/customed_yolo_model/best-yolo26.engine
```

该路径来自旧机器，当前工作区中并不存在 `/home/dji`。实际工作区中的 engine
位于：

```text
/home/peter/dji/LZP_PSDK/customed_yolo_model/best-yolo26.engine
```

部署前必须把加载路径改成可配置参数，不能继续写死绝对路径。

根据当前构建源码，现有 engine 路径使用 FP16。由于 engine 是二进制且当前开发
机没有安装 TensorRT，无法在主机上进一步检查每一层的实际精度。最终应以
Manifold 3 上的 TensorRT inspector 或构建日志为准。

## 3. 模型大小选择

候选模型包括 YOLO26n/s/x、YOLO11n/s/x、YOLOv8n/s/x。`x` 通常参数最多、
精度潜力最高，也最依赖算力、显存和散热。

不要只根据模型名判断最终性能。Manifold 3 上应至少比较：

| 指标 | 为什么重要 |
|---|---|
| mAP50 / mAP50-95 | 总体检测质量 |
| 缺陷召回率 | 漏检通常比误检代价更高 |
| P50/P95/P99 延迟 | 平均值不能代表无人机链路抖动 |
| 峰值显存 | 影响多路视频/其他 PSDK 模块 |
| 功耗与温度 | 影响长时间飞行稳定性 |
| engine 大小 | 影响部署包和加载时间 |

经验上，`x + INT8` 不一定优于 `s + INT8/FP16`。如果 `x` 量化后仍达不到实时
要求，应把 `s` 或更高分辨率的 `n/s` 作为并行候选，而不是只继续压 `x`。

## 4. ONNX 导出

推荐先导出静态、batch=1、FP32 ONNX：

```bash
python tools/export_yolo26_onnx.py \
  --weights train46-yolo26n-sangu/weights/best.pt \
  --imgsz 1280 \
  --opset 12 \
  --output exported/best-yolo26.onnx
```

导出后检查：

```bash
python tools/inspect_onnx_io.py exported/best-yolo26.onnx
```

必须记录并验证：

- 输入名称、shape、dtype。
- 输出数量、shape 和语义。
- 是否包含端到端 NMS。
- 类别数是否与模型一致。
- `simplify=True` 是否成功。
- PyTorch 与 ONNX 在同一批验证图上的检测结果是否一致。

当前 PSDK 的 `postprocess()` 按旧格式 `[1, 4+nc, 8400]` 解析。YOLO26
`[1, 300, 6]` 模型接入前需要实现新的 decoder。量化评估也必须建立在正确的
后处理之上，否则 mAP 和延迟数据都没有意义。

## 5. TensorRT INT8 生成步骤

### 5.1 必须在 Manifold 3 上构建

Engine 与以下条件绑定：

- GPU 架构；
- TensorRT 版本；
- CUDA/cuDNN 环境；
- 输入 shape 和 optimization profile；
- 构建时的 tactic 选择；
- 量化校准数据。

在 x86 开发机或另一块 Jetson 上构建的 engine 不应直接拷贝使用。

建议先记录目标机环境：

```bash
uname -m
nvidia-smi
dpkg -l | grep -i tensorrt
trtexec --version
```

### 5.2 准备校准集

校准集建议从训练集或独立生产采集集中抽取，不能使用最终测试集：

- 200-500 张有代表性的图片作为起点；
- 覆盖不同线路、塔型、背景、光照、焦距、运动模糊；
- 同时包含目标样本和困难负样本；
- 覆盖全部缺陷类别；
- 如果模型同时处理变焦/广角或可见光/红外，必须按线上数据比例采样；
- 不做随机训练增强，只做与线上完全相同的确定预处理。

校准与推理的预处理必须一致：

```text
letterbox
BGR -> RGB
HWC -> CHW
/255
float32
static batch 1
```

当前 PSDK CUDA letterbox 使用填充值 `128`，而 Ultralytics/报告工具通常使用
`114`。需要确认训练、ONNX 验证和校准数据对应的真实值，并统一。该差异通常
不会导致程序报错，但可能降低检测质量和校准有效性。

### 5.3 使用 EntropyCalibrator2

参考实现位于：

```text
source/int8_reference_cpp/src/calibrator.cpp
source/int8_reference_cpp/include/calibrator.h
```

构建配置应包含：

```cpp
config->setFlag(nvinfer1::BuilderFlag::kINT8);
config->setInt8Calibrator(&calibrator);
```

如希望允许 TensorRT 对不适合 INT8 的层回退：

```cpp
config->setFlag(nvinfer1::BuilderFlag::kFP16);
```

注意：同时打开 INT8 和 FP16 不等于“所有层固定为 FP16”。TensorRT 会选择
满足精度约束的层精度，实际结果必须通过构建日志和层敏感度分析确认。

### 5.4 校准缓存

校准缓存应包含以下信息作为版本键：

- 模型哈希；
- 校准集清单哈希；
- TensorRT 版本；
- GPU 架构；
- 输入 shape；
- 预处理版本。

缓存命中可以缩短构建时间；缓存失效时不能继续复用旧阈值。

### 5.5 当前 PSDK 代码需要补的内容

当前 `source/standalone_trt_runner/YOLOv11.cpp` 只支持 FP16。最小改造包括：

1. 增加 INT8 build 开关。
2. 接入 `IInt8EntropyCalibrator2`。
3. 校准数据预处理与 CUDA runtime 预处理保持一致。
4. 保存 `int8.cache` 和 engine。
5. 把模型路径、engine 路径、输入尺寸、精度模式改为配置项。
6. 为 YOLO26 `[1,300,6]` 增加输出解析器。
7. 在 engine 反序列化失败或输出维度不匹配时快速失败并打印诊断信息。

不能只把 FP16 标志替换成 INT8 标志，因为 INT8 必须有校准器。

## 6. 量化效果评估

### 6.1 质量指标

在完全相同的测试集、输入尺寸、confidence/NMS 参数和后处理下比较：

| 指标 | 说明 |
|---|---|
| mAP50 | 主要总体指标 |
| mAP50-95 | 框定位质量 |
| Precision | 误检水平 |
| Recall | 漏检水平 |
| F1 | P/R 平衡 |
| 每类 mAP/Recall | 防止平均值掩盖某类退化 |
| 固定工作点漏检率 | 比全 range mAP 更接近实际 |
| 每帧误检数 | 直接影响告警质量 |
| 检测数差异 | 快速发现阈值漂移 |

建议最少保留三组结果：

```text
FP32 engine baseline
FP16 engine
INT8 engine
```

基线应是同一 ONNX 在目标设备上构建的 TensorRT engine，而不是开发机上的
PyTorch 速度。

### 6.2 输出一致性指标

对相同输入比较 FP32 和 INT8 的模型输出：

- 置信度平均绝对误差；
- 检测框坐标平均绝对误差；
- 匹配框 IoU 分布；
- 检测数量差异；
- 类别混淆变化；
- 置信度校准误差。

### 6.3 KL 散度怎么用

KL 散度定义为：

```text
D_KL(P || Q) = sum_i P(i) * log(P(i) / Q(i))
```

它衡量两个概率分布的差异。量化评估中可以用于：

1. 比较同一层 FP32 与 INT8 激活值直方图；
2. 比较 FP32 与 INT8 输出的类别置信度分布；
3. 定位敏感层，而不仅是看最终 mAP。

实际计算建议：

- 固定 histogram bin 范围；
- 对两个分布加极小 epsilon，避免 `log(0)`；
- 使用同一批图像和同一层/同一输出节点；
- 同时报告 KL 和 JS 散度，JS 更对称且数值更稳定。

但 KL 不能替代 mAP 或召回率。KL 小幅变化不一定导致检测退化，KL 很大也不
必然意味着该层必须回退 FP16。应把 KL 当作敏感层定位工具，最终决策仍以
验证集和业务漏检指标为准。

### 6.4 信噪比类指标

还可以使用：

```text
SQNR = 10 * log10(sum(x_fp32^2) / sum((x_fp32 - x_int8)^2))
```

以及余弦相似度、最大绝对误差、P95 相对误差。这些指标适合检查单个 tensor
是否因量化发生严重数值失真。

### 6.5 性能指标

建议统一采集：

- 端到端延迟 P50/P95/P99；
- pure inference 延迟；
- 预处理和后处理耗时；
- 稳定 FPS；
- 峰值 GPU/CPU 内存；
- 功耗、温度和降频；
- 连续运行 30-60 分钟后的性能漂移；
- engine 加载时间。

### 6.6 建议验收线

以下是工程起始线，不是行业通用标准，最终应结合漏检成本调整：

- mAP50 相对 FP16 下降不超过 1 个绝对点；
- mAP50-95 下降不超过 1-2 个绝对点；
- 实际工作 confidence 下 Recall 不得明显下降；
- 关键缺陷类别不能出现集中漏检；
- P95/P99 延迟明显改善且无长时间抖动；
- 无 NaN、定位异常、engine 崩溃；
- 峰值显存和温度满足机载约束。

## 7. 量化效果不好时怎么调

按以下顺序排查，避免直接进入 QAT：

1. 先验证 FP32 TensorRT 与 PyTorch/ONNX 的结果一致。
2. 检查模型输出解析是否匹配 `[1,300,6]`。
3. 检查 letterbox、RGB/BGR 和归一化是否一致。
4. 检查校准集是否与线上相机域一致。
5. 检查校准缓存是否过期或被错误复用。
6. 用层敏感度分析找出误差最大的层。
7. 对敏感层保留 FP16，形成 INT8/FP16 混合精度 engine。
8. 增加困难样本和负样本重新校准。
9. 对比 `n/s/x` 和不同输入分辨率。
10. 仍不满足要求时再做 QAT。

### 7.1 校准集调整

如果 INT8 对暗光、小目标、远景导线或模糊画面退化明显，应该把这些条件按
训练分布比例加入校准集。不要只增加“容易图片”，校准集要覆盖决定阈值的尾
部激活。

### 7.2 混合精度

利用 Polygraphy 或 TensorRT layer precision API 做逐层回退。优先把敏感层、
小目标特征层和检测头回退到 FP16，保留 backbone 的 INT8 加速。

### 7.3 QAT

只有 PTQ 不能接受时再做 QAT：

- 使用训练集做伪量化训练；
- 冻结 BN 统计方式要与部署一致；
- QAT 后重新导出 ONNX；
- 将 fake-quant 信息正确映射到 TensorRT；
- 仍然在 Manifold 3 上做最终 mAP 和延迟验证。

### 7.4 模型重选

量化不是解决所有性能问题的唯一办法。如果 `x` 在机载设备上仍然过慢：

- 比较同数据集训练的 `s`/`n`；
- 降低输入分辨率但重新训练/验证小目标召回；
- 考虑剪枝或蒸馏；
- 根据变焦链路只处理 ROI。

## 8. 推荐实验矩阵

每个候选模型至少生成以下结果：

| 模型 | 精度 | 输入 | mAP50 | mAP50-95 | R@工作点 | P95 ms | 峰值显存 |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLO26n | FP16 | 1280 | 待测 | 待测 | 待测 | 待测 | 待测 |
| YOLO26n | INT8 | 1280 | 待测 | 待测 | 待测 | 待测 | 待测 |
| YOLO26s | FP16 | 1280 | 待测 | 待测 | 待测 | 待测 | 待测 |
| YOLO26s | INT8 | 1280 | 待测 | 待测 | 待测 | 待测 | 待测 |
| YOLO26x | FP16 | 1280 | 待测 | 待测 | 待测 | 待测 | 待测 |
| YOLO26x | INT8 | 1280 | 待测 | 待测 | 待测 | 待测 | 待测 |

只有同一目标机、同一数据集和同一后处理下的结果才可直接比较。

## 9. 当前部署风险

1. `ai_detect.h` 写死旧机器的模型绝对路径。
2. YOLO26 端到端输出与当前旧式后处理不兼容。
3. CUDA letterbox 填充值与部分导出/验证管线不一致。
4. 当前 PSDK 路径没有 INT8 calibrator。
5. CUDA 预处理中存在每帧分配和同步，仍有优化空间。
6. 当前视觉节点与云台节点的消息类型/话题未直接接通。
7. 现成 engine 不能跨设备直接复用。

## 10. 下一步落地顺序

1. 在 Manifold 3 上固定并记录 CUDA/TensorRT/GPU 版本。
2. 修复 YOLO26 输出 parser 和模型路径配置。
3. 建立 FP32 TensorRT 基线并验证后处理一致性。
4. 生成 FP16 engine 并记录 mAP 和延迟。
5. 固定校准集清单，接入 EntropyCalibrator2。
6. 生成 INT8 engine 和校准缓存。
7. 完成 FP32/FP16/INT8 指标表。
8. 对失败层做 KL、JS、SQNR 和 layer sensitivity 分析。
9. 按结果采用混合精度、重新校准、QAT 或换模型。

