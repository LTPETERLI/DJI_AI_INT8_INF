# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
增强版CLIPDetect模块
集成ROI-CLIP分支和多模态融合机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER
from .roi_clip_branch import RoiClipBranch, MultiBranchFusion, RoICLIPLoss
from .block import ContrastiveHead, BNContrastiveHead
from .conv import Conv
from ultralytics.utils.tal import make_anchors, dist2bbox


class OriginalCLIPDetect(torch.nn.Module):
    """
    原始的 CLIPDetect 兼容头（无 ROI 分支），用于 use_roi_clip_branch=False 时退回。
    定义为顶层类以便可 pickle。
    """

    def __init__(self, nc, embed, with_bn, ch):
        from .head import Detect  # 延迟导入避免循环

        super().__init__()
        self.detect = Detect(nc, ch)
        c3 = max(ch[0], min(nc, 100))
        self.cv3 = torch.nn.ModuleList(
            torch.nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), torch.nn.Conv2d(c3, embed, 1)) for x in ch
        )
        Head = BNContrastiveHead if with_bn else ContrastiveHead
        self.cv4 = torch.nn.ModuleList(Head(embed) for _ in ch)
        self.stride = self.detect.stride
        self.nc = nc
        self.nl = self.detect.nl
        self.reg_max = self.detect.reg_max
        self.no = self.detect.no
        self.export = self.detect.export
        self.dynamic = self.detect.dynamic
        self.end2end = getattr(self.detect, "end2end", False)

    def forward(self, x, text=None):
        for i in range(self.nl):
            if text is None:
                x[i] = torch.cat((self.detect.cv2[i](x[i]), self.detect.cv3[i](x[i])), 1)
            else:
                x[i] = torch.cat((self.detect.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x
        self.no = self.nc + self.reg_max * 4
        y = self.detect._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        # 与 Detect 相同的偏置初始化
        self.detect.bias_init()


class CLIPDetectEnhanced(nn.Module):
    """
    增强版CLIPDetect检测头
    集成ROI-CLIP分支和多分支融合机制
    """
    
    def __init__(self, 
                 nc=80, 
                 embed=512, 
                 with_bn=False, 
                 ch=(),
                 use_roi_clip_branch=False,
                 roi_clip_loss_weight=0.3,
                 roi_size=7,
                 **kwargs):
        super().__init__()
        
        # 基础参数
        self.nc = nc
        self.embed = embed
        self.ch = ch
        self.nl = len(ch)  # 检测层数量
        self.reg_max = 16  # DFL通道数
        
        # ROI-CLIP分支配置
        self.use_roi_clip_branch = use_roi_clip_branch
        self.roi_clip_loss_weight = roi_clip_loss_weight
        self.roi_size = roi_size
        # ROI 过滤上限，避免全量 anchor 触发 OOM
        self.roi_max_rois = kwargs.get("roi_max_rois", 300)
        
        # 基础检测头（继承自原始CLIPDetect）
        c3 = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(  # 回归头
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, 4 * self.reg_max, 1))
            for x in ch
        )
        
        # SPAR分类头（原始CLIPDetect的分类路径）
        self.cv3 = nn.ModuleList(  # 特征提取
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1))
            for x in ch
        )
        
        self.cv4 = nn.ModuleList(  # 对比头
            BNContrastiveHead(embed) if with_bn else ContrastiveHead()
            for _ in ch
        )
        # 构建时先占位 stride，后续模型 build 时会写入真实 stride
        self.stride = torch.zeros(self.nl, dtype=torch.float)
        
        # ROI-CLIP分支
        if self.use_roi_clip_branch:
            self.roi_clip_branch = RoiClipBranch(
                in_channels=c3,  # 使用与分类头相同的通道数
                roi_size=roi_size,
                embed_dim=embed,
                num_classes=nc,
                level_channels=ch,
                spatial_scales=[1.0, 0.5, 0.25, 0.125][:self.nl],  # 根据检测层数量调整
                use_multi_scale=True
            )
            
            # 多分支融合模块
            self.fusion_module = MultiBranchFusion(
                num_classes=nc,
                embed_dim=embed
            )
            
            # ROI-CLIP损失函数
            self.roi_clip_loss = RoICLIPLoss(loss_weight=roi_clip_loss_weight)
        
        # DFL模块
        from .block import DFL
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        
        # 输出维度
        self.no = nc + self.reg_max * 4  # 每个anchor的输出维度

    def _prepare_text_embeddings(self, text_embeddings, batch_size, device, dtype):
        """确保文本嵌入张量形状为 [B, num_classes, embed_dim]。"""
        if text_embeddings is None:
            return None
        if text_embeddings.ndim == 2:
            text_embeddings = text_embeddings.unsqueeze(0)
        if text_embeddings.shape[0] != batch_size:
            text_embeddings = text_embeddings.expand(batch_size, -1, -1)
        return text_embeddings.to(device=device, dtype=dtype)

    def _decode_predictions(self, feats):
        """将头部输出展开为预测框和分类 logits（展平 anchor 维度）。"""
        x_cat = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2)
        pred_dist, spar_cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        pred_dist = pred_dist.permute(0, 2, 1).contiguous()  # [B, N, reg]
        spar_cls = spar_cls.permute(0, 2, 1).contiguous()    # [B, N, C]
        if (self.stride <= 0).any():
            raise RuntimeError(f"CLIPDetectEnhanced stride 非法: {self.stride}")
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        if self.reg_max > 1:
            proj = torch.arange(self.reg_max, device=pred_dist.device, dtype=pred_dist.dtype)
            pred_dist_xyxy = pred_dist.view(pred_dist.shape[0], pred_dist.shape[1], 4, self.reg_max).softmax(-1).matmul(proj)
        else:
            pred_dist_xyxy = pred_dist
        pred_bboxes = dist2bbox(pred_dist_xyxy, anchor_points, xywh=False) * stride_tensor
        return pred_bboxes, pred_dist_xyxy, spar_cls, anchor_points, stride_tensor

    def _compute_roi_logits(self, pyramids, pred_bboxes, text_embeddings, stride_tensor, scores=None):
        """
        基于预测框为每个 anchor 生成 ROI logits，保持与 anchor 展平后的顺序一致。
        """
        device = pred_bboxes.device
        B, N, _ = pred_bboxes.shape
        max_rois = min(self.roi_max_rois, N)

        # 记录每个尺度的 anchor 数，以便拆分与重组
        counts = [p.shape[2] * p.shape[3] for p in pyramids]
        level_spans = []
        start = 0
        for c in counts:
            level_spans.append((start, start + c))
            start += c

        roi_logits_full = pred_bboxes.new_zeros(B, N, self.nc)

        # 每个样本分别计算，避免 batch 交叉
        for b in range(B):
            rois_by_level = []
            split_sizes = []
            if scores is not None:
                keep_idx = scores[b].topk(max_rois).indices
            else:
                keep_idx = torch.arange(max_rois, device=device)

            # 按层划分 ROI
            for (s, e), stride in zip(level_spans, self.stride):
                mask = (keep_idx >= s) & (keep_idx < e)
                idx = keep_idx[mask] - s
                rois_lvl = pred_bboxes[b, s:e]
                rois_lvl = rois_lvl[idx] if idx.numel() else rois_lvl.new_zeros(0, 4)
                rois_by_level.append(rois_lvl)
                split_sizes.append(rois_lvl.shape[0])

            rois_text = text_embeddings[b] if text_embeddings is not None else None
            roi_logits_b, _ = self.roi_clip_branch(
                [p[b:b + 1] for p in pyramids],
                rois_by_level,
                rois_text,
                feature_strides=[float(s) for s in self.stride],
            )

            if roi_logits_b.numel() == 0:
                continue

            # 按层切回 anchor 顺序，仅填充被选中的 anchor
            offset = 0
            for (s, e), size in zip(level_spans, split_sizes):
                if size == 0:
                    continue
                mask = (keep_idx >= s) & (keep_idx < e)
                idx = keep_idx[mask]
                roi_logits_full[b, idx] = roi_logits_b[offset: offset + size]
                offset += size

        return roi_logits_full, level_spans

    def _fuse_logits(self, spar_cls, roi_logits):
        """将 SPAR 分类 logits 与 ROI-CLIP logits 融合。"""
        if not hasattr(self, "fusion_module") or roi_logits is None or roi_logits.numel() == 0:
            return spar_cls
        fused = self.fusion_module(
            spar_cls.reshape(-1, self.nc),
            roi_logits.reshape(-1, self.nc),
        )
        return fused.view_as(spar_cls)

    def _reconstruct_outputs(self, feats, fused_cls, level_spans):
        """用融合后的分类 logits 替换 head 输出中的分类部分，保持原始张量形状。"""
        fused_outputs = []
        start = 0
        for feat, (s, e) in zip(feats, level_spans):
            b, _, h, w = feat.shape
            count = e - s
            reg = feat[:, : self.reg_max * 4]
            cls_map = fused_cls[:, start: start + count].transpose(1, 2).reshape(b, self.nc, h, w)
            fused_outputs.append(torch.cat([reg, cls_map], dim=1))
            start += count
        return fused_outputs
    
    def _decode_dfl(self, box_feats):
        """将 reg_max 分布解码为距离（借鉴 Detect/_inference 逻辑）。"""
        b, c, a = box_feats.shape  # [B, reg*4, anchors]
        if self.reg_max > 1:
            proj = torch.arange(self.reg_max, device=box_feats.device, dtype=box_feats.dtype)
            box_feats = box_feats.permute(0, 2, 1).contiguous()  # [B, anchors, reg*4]
            box_feats = box_feats.view(b, a, 4, self.reg_max)  # [B, anchors, 4, reg]
            return box_feats.softmax(-1).matmul(proj)  # [B, anchors, 4]
        else:
            return box_feats.permute(0, 2, 1).contiguous()

    def forward(self, x, text_embeddings=None, targets=None, img_meta=None):
        """
        前向传播
        
        Args:
            x: List of [B, C, H, W] 多尺度特征图
            text_embeddings: [num_classes, embed_dim] 文本嵌入
            targets: 训练目标（包含边界框和类别）
            img_meta: 图像元信息
            
        Returns:
            如果是训练模式：
                - 原始输出列表
                - ROI-CLIP损失（如果启用）
            如果是推理模式：
                - 融合后的检测结果
        """
        if self.training:
            return self.forward_train(x, text_embeddings, targets, img_meta)
        else:
            return self.forward_inference(x, text_embeddings, img_meta)
    
    def forward_train(self, x, text_embeddings, targets, img_meta):
        """训练模式前向传播"""
        outputs = []
        cls_logits_list = []

        # 1. 计算SPAR分支输出
        for i in range(self.nl):
            reg_feat = self.cv2[i](x[i])
            cls_feat = self.cv3[i](x[i])
            cls_logits = self.cv4[i](cls_feat, text_embeddings) if text_embeddings is not None else cls_feat
            outputs.append(torch.cat((reg_feat, cls_logits), 1))
            cls_logits_list.append(cls_logits)

        # 2. ROI-CLIP 分支与融合
        use_roi = self.use_roi_clip_branch and self.roi_clip_branch is not None
        has_stride = hasattr(self, "stride") and isinstance(self.stride, torch.Tensor) and (self.stride > 0).all()
        text_embeddings = self._prepare_text_embeddings(
            text_embeddings, outputs[0].shape[0], outputs[0].device, outputs[0].dtype
        )

        if use_roi and has_stride and text_embeddings is not None:
            pred_bboxes, pred_dist, spar_cls, anchor_points, stride_tensor = self._decode_predictions(outputs)
            if (stride_tensor <= 0).any():
                LOGGER.warning(f"[ROI-CLIP] stride_tensor 存在非正值: {stride_tensor.unique()}")
            scores = spar_cls.sigmoid().amax(dim=2)  # [B, N]
            roi_logits, level_spans = self._compute_roi_logits(
                x, pred_bboxes, text_embeddings, stride_tensor, scores
            )
            fused_cls = self._fuse_logits(spar_cls, roi_logits)
            fused_outputs = self._reconstruct_outputs(outputs, fused_cls, level_spans)

            aux = {
                "roi_logits": roi_logits,
                "spar_cls": spar_cls,
                "level_spans": level_spans,
                "text_embeddings": text_embeddings,
                "features": x,
            }
            return fused_outputs, aux

        return outputs
    
    def forward_inference(self, x, text_embeddings, img_meta):
        """推理模式前向传播"""
        spar_outputs = []
        for i in range(self.nl):
            reg_feat = self.cv2[i](x[i])
            cls_feat = self.cv3[i](x[i])
            cls_logits = self.cv4[i](cls_feat, text_embeddings) if text_embeddings is not None else cls_feat
            spar_outputs.append(torch.cat((reg_feat, cls_logits), 1))

        use_roi = self.use_roi_clip_branch and self.roi_clip_branch is not None
        has_stride = hasattr(self, "stride") and isinstance(self.stride, torch.Tensor) and (self.stride > 0).all()
        text_embeddings = self._prepare_text_embeddings(
            text_embeddings, spar_outputs[0].shape[0], spar_outputs[0].device, spar_outputs[0].dtype
        )

        if not use_roi or text_embeddings is None or not has_stride:
            return self._postprocess_spar_outputs(spar_outputs)

        # 通过预测框构建 ROI logits 并融合
        pred_bboxes, pred_dist, spar_cls, anchor_points, stride_tensor = self._decode_predictions(spar_outputs)
        scores = spar_cls.sigmoid().amax(dim=2)  # [B, N]
        roi_logits, level_spans = self._compute_roi_logits(
            x, pred_bboxes, text_embeddings, stride_tensor, scores
        )
        fused_cls = self._fuse_logits(spar_cls, roi_logits)
        fused_outputs = self._reconstruct_outputs(spar_outputs, fused_cls, level_spans)

        return self._postprocess_spar_outputs(fused_outputs)
    def _postprocess_spar_outputs(self, spar_outputs):
        """后处理SPAR分支输出"""
        shape = spar_outputs[0].shape  # BCHW
        b = shape[0]
        x_cat = torch.cat([xi.view(b, self.no, -1) for xi in spar_outputs], 2)  # [B, no, anchors]
        box_dist, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        # 解码回归分布到 xyxy
        box = self._decode_dfl(box_dist)  # [B, anchors, 4]
        anchor_points, stride_tensor = make_anchors(spar_outputs, self.stride, 0.5)
        dbox = dist2bbox(box, anchor_points, xywh=False) * stride_tensor  # [B, anchors, 4]
        # 保障坐标有序，避免后处理绘图报错
        x1 = torch.minimum(dbox[..., 0], dbox[..., 2])
        y1 = torch.minimum(dbox[..., 1], dbox[..., 3])
        x2 = torch.maximum(dbox[..., 0], dbox[..., 2])
        y2 = torch.maximum(dbox[..., 1], dbox[..., 3])
        dbox = torch.stack((x1, y1, x2, y2), dim=-1)
        dbox = torch.nan_to_num(dbox, nan=0.0, posinf=0.0, neginf=0.0)
        dbox = torch.clamp(dbox, min=0.0)
        cls = cls.sigmoid().permute(0, 2, 1)  # [B, anchors, nc]

        out = torch.cat((dbox, cls), 2)  # [B, anchors, 4+nc]
        return out if getattr(self, "export", False) else (out, spar_outputs)
    
    def bias_init(self):
        """初始化偏置"""
        # 继承自原始CLIPDetect的偏置初始化
        m = self
        for a, b, s in zip(m.cv2, m.cv3, getattr(m, 'stride', [1.0] * self.nl)):
            a[-1].bias.data[:] = 1.0  # box
    
    def set_roi_clip_enabled(self, enabled=True):
        """设置ROI-CLIP分支是否启用"""
        self.use_roi_clip_branch = enabled
    
    def get_fusion_weights(self):
        """获取融合权重"""
        if hasattr(self, 'fusion_module'):
            return self.fusion_module.get_weights()
        else:
            return 1.0, 0.0  # 只有SPAR分支


# 兼容性包装类
class CLIPDetect(nn.Module):
    """
    兼容性CLIPDetect类
    根据配置决定是否使用增强版本
    """

    def __init__(self, nc=80, embed=512, with_bn=False, ch=(), **kwargs):
        super().__init__()

        # 检查是否启用ROI-CLIP分支
        use_roi_clip = kwargs.get('use_roi_clip_branch', False)
        LOGGER.info(f"[CLIPDetect] use_roi_clip_branch={use_roi_clip}")

        if use_roi_clip:
            # 使用增强版本
            self.impl = CLIPDetectEnhanced(nc, embed, with_bn, ch, **kwargs)
        else:
            # 使用原始版本 - 需要实现原始版本
            self.impl = self._create_original_clipdetect(nc, embed, with_bn, ch)

        # 为兼容性添加必要的属性
        self.stride = self.impl.stride if hasattr(self.impl, 'stride') else torch.zeros(len(ch))
        self.nc = self.impl.nc if hasattr(self.impl, 'nc') else nc
        self.nl = self.impl.nl if hasattr(self.impl, 'nl') else len(ch)
        self.reg_max = self.impl.reg_max if hasattr(self.impl, 'reg_max') else 16
        self.no = self.impl.no if hasattr(self.impl, 'no') else nc + 16 * 4
        self.use_roi_clip_branch = getattr(self.impl, "use_roi_clip_branch", False)
        self.roi_clip_loss_weight = getattr(self.impl, "roi_clip_loss_weight", 0.0)
        self.export = getattr(self.impl, 'export', False)
        self.dynamic = getattr(self.impl, 'dynamic', False)
        self.end2end = getattr(self.impl, 'end2end', False)
        self.max_det = getattr(self.impl, 'max_det', 300)
        self.shape = getattr(self.impl, 'shape', None)
        self.anchors = getattr(self.impl, 'anchors', torch.empty(0))
        self.strides = getattr(self.impl, 'strides', torch.empty(0))
        self.legacy = getattr(self.impl, 'legacy', False)

    def _create_original_clipdetect(self, nc, embed, with_bn, ch):
        """创建原始CLIPDetect实现"""
        return OriginalCLIPDetect(nc, embed, with_bn, ch)

    def forward(self, x, text_embeddings=None, **kwargs):
        return self.impl(x, text_embeddings, **kwargs)

    def bias_init(self):
        if hasattr(self.impl, 'bias_init'):
            self.impl.bias_init()
    # NOTE: Detect/Trainer 依赖 stride/shape/anchors 等属性，确保这些属性沿用 impl
    @property
    def stride(self):
        return getattr(self.impl, "stride", torch.zeros(self.impl.nl if hasattr(self.impl, "nl") else 0))

    @stride.setter
    def stride(self, v):
        if hasattr(self.impl, "stride"):
            self.impl.stride = v

    def to(self, *args, **kwargs):
        """确保 impl 对象也被迁移到指定设备"""
        super().to(*args, **kwargs)
        self.impl.to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        """确保 impl 对象也被迁移到 CUDA 设备"""
        super().cuda(device)
        self.impl.cuda(device)
        return self

    def cpu(self):
        """确保 impl 对象也被迁移到 CPU"""
        super().cpu()
        self.impl.cpu()
        return self

    def _apply(self, fn):
        """确保 impl 对象也应用相同的转换函数"""
        super()._apply(fn)
        self.impl._apply(fn)
        return self
