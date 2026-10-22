from ultralytics import SPARYOLO
w = "/home/peter/服务器文件上传/真实训练文件/train4种缺陷112/weights/best.pt"

# 1) 直接用权重构建模型（内部会加载正确的 YAML 和 nc=4）
model = SPARYOLO(w)
# 2) 加载你自己训练好的权重
model.load(w)
# 2) 直接导出 ONNX
model.export(
    format="onnx",
    imgsz=1024,
    opset=12,
    dynamic=False,   # 先固定尺寸，减少导出出错概率
    simplify=True,   # 环境装了 onnxsim 的话，可以简化图
    device="cpu",
    # nms=False,     # 看你部署需求：要裸输出就关 NMS
)