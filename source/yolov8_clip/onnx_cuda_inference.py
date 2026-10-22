#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import onnxruntime as ort
import cv2
import numpy as np
import os

# ==== 配置部分 ====
ONNX_PATH = "/home/peter/服务器文件上传/真实训练文件/train4种缺陷112/weights/best.onnx"
IMG_PATH  = "/home/peter/PycharmProjects/analogical_exclude/.ultralytics/Broken_strands#0473.jpg"
OUT_PATH  = "onnx_out_0473.jpg"

IMG_SIZE = 1024               # 导出 onnx 时用的 imgsz（保持一致）
CONF_THRES = 0.25
IOU_THRES = 0.45

CLASS_NAMES = [
    "Broken_strands",
    "loose_strands",
    "foreign_body_entanglement",
    "rust",
]

# ==== 工具函数 ====

def letterbox(img, new_shape=IMG_SIZE, color=(114, 114, 114)):
    """仿 YOLOv8 的 letterbox：保持比例缩放 + 填充到 new_shape x new_shape。"""
    shape = img.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    h0, w0 = shape
    h, w = new_shape

    r = min(h / h0, w / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = w - new_unpad[0], h - new_unpad[1]  # 剩余填充

    dw /= 2
    dh /= 2

    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)

    return img, r, (dw, dh)


def xywh2xyxy(x):
    """(x, y, w, h) -> (x1, y1, x2, y2)"""
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def box_iou(box1, box2):
    """计算 IoU，box: [N, 4], [M, 4] in xyxy"""
    # 交集
    inter_x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])

    inter_w = np.maximum(inter_x2 - inter_x1, 0)
    inter_h = np.maximum(inter_y2 - inter_y1, 0)
    inter_area = inter_w * inter_h

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union = area1[:, None] + area2[None, :] - inter_area
    return inter_area / np.clip(union, 1e-6, None)


def nms(boxes, scores, iou_thres=0.45):
    """最基础的 NMS，单类别版。"""
    idxs = scores.argsort()[::-1]
    keep = []

    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = box_iou(boxes[i:i+1], boxes[idxs[1:]])[0]
        idxs = idxs[1:][ious < iou_thres]

    return keep


# ==== 1. 加载 ONNX 模型 ====
session = ort.InferenceSession(ONNX_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("ONNX input:", input_name)
print("ONNX output:", output_name)

# ==== 2. 读取 & 预处理图片 ====
img0 = cv2.imread(IMG_PATH)          # BGR
assert img0 is not None, f"Image not found: {IMG_PATH}"
h0, w0 = img0.shape[:2]

img, r, (dw, dh) = letterbox(img0, IMG_SIZE)
# BGR -> RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))   # HWC -> CHW
img = np.expand_dims(img, 0)         # [1,3,H,W]

# ==== 3. 前向推理 ====
outputs = session.run(None, {input_name: img})
pred = outputs[0]    # [1, 8, 87040]
print("raw pred shape:", pred.shape)

# ==== 4. 解析输出 ====
# 去掉 batch 维度 -> [8, 87040] 再转成 [87040, 8]
pred = np.squeeze(pred, 0)           # [8, 87040]
pred = pred.transpose(1, 0)          # [87040, 8]

boxes = pred[:, :4]                  # xywh (在 letterbox 后的 1024x1024 坐标系上)
cls_scores = pred[:, 4:]             # [87040, 4]，已经是概率形式（0~1）

# 每个位置取最大类别置信度
cls_ids = np.argmax(cls_scores, axis=1)
conf = np.max(cls_scores, axis=1)

# 置信度阈值
mask = conf > CONF_THRES
boxes = boxes[mask]
cls_ids = cls_ids[mask]
conf = conf[mask]

if boxes.shape[0] == 0:
    print("No boxes after conf thresh.")
else:
    # xywh -> xyxy
    boxes_xyxy = xywh2xyxy(boxes)

    # 反 letterbox 到原图坐标
    # 注意 dw, dh 是左右/上下总 padding 的一半
    dw, dh = dw, dh
    boxes_xyxy[:, [0, 2]] -= dw
    boxes_xyxy[:, [1, 3]] -= dh
    boxes_xyxy /= r

    # 限制在图像边界内
    boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clip(0, w0 - 1)
    boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clip(0, h0 - 1)

    # ==== 5. 按类别做 NMS ====
    final_boxes = []
    final_scores = []
    final_cls_ids = []

    for cls in range(len(CLASS_NAMES)):
        idxs = np.where(cls_ids == cls)[0]
        if idxs.size == 0:
            continue
        boxes_c = boxes_xyxy[idxs]
        scores_c = conf[idxs]
        keep = nms(boxes_c, scores_c, IOU_THRES)

        final_boxes.append(boxes_c[keep])
        final_scores.append(scores_c[keep])
        final_cls_ids.append(np.full(len(keep), cls))

    if len(final_boxes) == 0:
        print("No boxes after NMS.")
    else:
        final_boxes = np.concatenate(final_boxes, axis=0)
        final_scores = np.concatenate(final_scores, axis=0)
        final_cls_ids = np.concatenate(final_cls_ids, axis=0)

        # ==== 6. 画框并保存 ====
        for box, score, cls in zip(final_boxes, final_scores, final_cls_ids):
            x1, y1, x2, y2 = box.astype(int)
            label = f"{CLASS_NAMES[cls]} {score:.2f}"
            cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img0, label, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imwrite(OUT_PATH, img0)
        print(f"Saved ONNX detection result to: {os.path.abspath(OUT_PATH)}")

        print("Detections:")
        for box, score, cls in zip(final_boxes, final_scores, final_cls_ids):
            print(CLASS_NAMES[cls], "conf:", float(score), "box:", box.tolist())
