# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
ROI-CLIP分支模块
用于在CLIPDetect内部实现ROI级别的多模态分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ultralytics.utils.ops import xyxy2xywh
from torchvision.ops import roi_align as tv_roi_align


class RoIAlign(nn.Module):
    """RoI对齐操作"""
    
    def __init__(self, output_size=7, spatial_scale=1.0, sampling_ratio=-1):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
    
    def forward(self, features, rois):
        """
        Args:
            features: [B, C, H, W] 特征图
            rois: [N, 4] 边界框 [x1, y1, x2, y2]（与特征图同尺度）
        Returns:
            pooled_features: [N, C, output_size, output_size]
        """
        device = features.device
        rois = rois.to(device)
        # 拼接 batch_idx，当前每层特征是单张图，batch_idx=0
        rois_with_batch = torch.cat(
            [torch.zeros((rois.shape[0], 1), device=device, dtype=rois.dtype), rois], dim=1
        )
        return tv_roi_align(
            features,
            rois_with_batch,
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
            aligned=False,
        )


class RoiClipBranch(nn.Module):
    """
    ROI-CLIP分支模块
    负责从多尺度特征中提取ROI特征并进行多模态分类
    """
    
    def __init__(self, 
                 in_channels=256,
                 roi_size=7,
                 embed_dim=512,
                 num_classes=1,
                 spatial_scales=[1.0, 0.5, 0.25, 0.125],  # 对应P2,P3,P4,P5
                 use_multi_scale=True,
                 level_channels=None):
        super().__init__()
        
        self.in_channels = in_channels
        self.roi_size = roi_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.spatial_scales = spatial_scales
        self.use_multi_scale = use_multi_scale
        level_channels = level_channels or [in_channels for _ in spatial_scales]
        
        # RoIAlign模块
        self.roi_aligns = nn.ModuleList([
            RoIAlign(output_size=roi_size, spatial_scale=scale)
            for scale in spatial_scales
        ])
        # 各尺度通道对齐到统一 in_channels
        self.level_proj = nn.ModuleList([
            nn.Conv2d(c, in_channels, kernel_size=1) for c in level_channels
        ])
        
        # 特征投影层 - 将ROI特征投影到CLIP嵌入空间
        self.feature_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [N, C, 1, 1]
            nn.Flatten(),  # [N, C]
            nn.Linear(in_channels, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 温度参数 - 用于控制对比学习的温度
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # 融合权重 - 可学习参数
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, multi_scale_features, rois_by_level, text_embeddings, feature_strides=None, img_meta=None):
        """
        前向传播
        
        Args:
            multi_scale_features: List[[B, C, H, W]] 多尺度特征图
            rois_by_level: list，按尺度组织的 ROI，元素可为 None 或 [n,4]（输入坐标为原图尺度）
            text_embeddings: [num_classes, embed_dim] 文本嵌入
            feature_strides: list，可选，每个特征层对应的 stride
            img_meta: 图像元信息
            
        Returns:
            roi_logits: [N, num_classes] ROI分类logits
            roi_features: [N, embed_dim] ROI特征嵌入
        """
        if not rois_by_level or all((r is None or len(r) == 0) for r in rois_by_level):
            device = multi_scale_features[0].device if multi_scale_features else text_embeddings.device
            return (
                torch.zeros(0, self.num_classes, device=device),
                torch.zeros(0, self.embed_dim, device=device),
            )
        
        # 1. 从多尺度特征中提取ROI特征
        roi_features_list = []
        stride_list = feature_strides or [1.0 for _ in multi_scale_features]

        for i, (features, roi_align, stride, proj) in enumerate(
            zip(multi_scale_features, self.roi_aligns, stride_list, self.level_proj)
        ):
            if i >= len(multi_scale_features):
                break
            rois = None if rois_by_level is None else rois_by_level[i]
            if rois is None or len(rois) == 0:
                continue
            features = proj(features)
            scaled_rois = rois / float(stride)
            roi_feat = roi_align(features, scaled_rois)
            roi_features_list.append(roi_feat)

        if roi_features_list:
            roi_features = torch.cat(roi_features_list, dim=0)
        else:
            device = multi_scale_features[0].device
            roi_features = torch.zeros(0, self.in_channels, self.roi_size, self.roi_size, device=device)
        
        # 2. 将ROI特征投影到CLIP嵌入空间
        roi_embed = self.feature_projector(roi_features)  # [N, embed_dim]
        
        # L2归一化
        roi_embed = F.normalize(roi_embed, p=2, dim=1)
        
        # 3. 计算与文本嵌入的相似度
        # text_embeddings: [num_classes, embed_dim]
        # 需要转置以便矩阵乘法: [embed_dim, num_classes]
        text_embed_norm = F.normalize(text_embeddings, p=2, dim=1)
        
        # 计算余弦相似度
        logit_scale = self.logit_scale.exp()
        roi_logits = logit_scale * torch.matmul(roi_embed, text_embed_norm.t())  # [N, num_classes]
        
        return roi_logits, roi_embed
    
    def get_fusion_weight(self):
        """获取融合权重（经过sigmoid约束到[0,1]）"""
        return torch.sigmoid(self.fusion_weight)


class MultiBranchFusion(nn.Module):
    """
    多分支融合模块
    负责融合SPAR分支和ROI-CLIP分支的输出
    """
    
    def __init__(self, num_classes=1, embed_dim=512):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # 可学习融合权重
        self.spar_weight = nn.Parameter(torch.tensor(0.7))  # SPAR分支权重
        self.roi_weight = nn.Parameter(torch.tensor(0.3))   # ROI-CLIP分支权重
        
        # 可选：使用MLP进行更复杂的融合
        self.fusion_mlp = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes * 4),
            nn.ReLU(inplace=True),
            nn.Linear(num_classes * 4, num_classes)
        )
        self.use_mlp_fusion = False  # 默认使用加权融合
        
    def forward(self, spar_logits, roi_logits, use_mlp=False):
        """
        融合两个分支的logits
        
        Args:
            spar_logits: [N, num_classes] SPAR分支logits
            roi_logits: [N, num_classes] ROI-CLIP分支logits
            use_mlp: 是否使用MLP融合
            
        Returns:
            fused_logits: [N, num_classes] 融合后的logits
        """
        if spar_logits.shape != roi_logits.shape:
            # 如果形状不匹配，调整ROI logits形状
            if roi_logits.shape[0] == 0:
                # ROI分支没有输出，直接使用SPAR分支
                return spar_logits
            
            # 尝试匹配形状
            min_len = min(spar_logits.shape[0], roi_logits.shape[0])
            spar_logits = spar_logits[:min_len]
            roi_logits = roi_logits[:min_len]
        
        if use_mlp and self.use_mlp_fusion:
            # 使用MLP融合
            concat_logits = torch.cat([spar_logits, roi_logits], dim=1)  # [N, num_classes*2]
            fused_logits = self.fusion_mlp(concat_logits)
        else:
            # 使用加权融合
            spar_w = torch.sigmoid(self.spar_weight)
            roi_w = torch.sigmoid(self.roi_weight)
            
            # 归一化权重
            total_w = spar_w + roi_w
            spar_w = spar_w / total_w
            roi_w = roi_w / total_w
            
            fused_logits = spar_w * spar_logits + roi_w * roi_logits
        
        return fused_logits
    
    def get_weights(self):
        """获取当前融合权重"""
        spar_w = torch.sigmoid(self.spar_weight)
        roi_w = torch.sigmoid(self.roi_weight)
        total_w = spar_w + roi_w
        return spar_w / total_w, roi_w / total_w


class RoICLIPLoss(nn.Module):
    """ROI-CLIP分支的损失函数"""
    
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, roi_logits, targets, pos_mask=None):
        """
        计算ROI-CLIP分支的损失
        
        Args:
            roi_logits: [N, num_classes] ROI分类logits
            targets: [N] 目标类别
            pos_mask: [N] 正样本掩码
            
        Returns:
            loss: 标量损失值
        """
        if roi_logits.shape[0] == 0:
            # 如果没有ROI预测，返回0损失
            return torch.tensor(0.0, device=roi_logits.device, requires_grad=True)
        
        if pos_mask is not None:
            # 只计算正样本的损失
            roi_logits = roi_logits[pos_mask]
            targets = targets[pos_mask]
        
        if roi_logits.shape[0] == 0:
            return torch.tensor(0.0, device=roi_logits.device, requires_grad=True)
        
        # 计算交叉熵损失
        loss = self.ce_loss(roi_logits, targets)
        
        return self.loss_weight * loss
