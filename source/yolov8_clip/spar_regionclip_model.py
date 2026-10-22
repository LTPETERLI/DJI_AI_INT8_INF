"""
SPAR-RegionCLIP Model following the original engineering approach.
This model extends the standard YOLO model with RegionCLIP components
that can be integrated separately in the training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel, SPARModel
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")


class RegionFeatureExtractor(nn.Module):
    """Region feature extractor for SPAR-RegionCLIP model."""
    
    def __init__(self, input_dim=256, output_dim=512):
        """Initialize RegionFeatureExtractor.
        
        Args:
            input_dim (int): Input feature dimension
            output_dim (int): Output feature dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Simple feature extraction layers without RoIAlign for now
        # Use adaptive pooling to handle variable input sizes
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        # We'll create the linear layer dynamically based on input
        
    def forward(self, feature_maps, proposals):
        """Extract region features from feature maps and proposals.
        
        Args:
            feature_maps: List of feature maps from backbone at different scales
            proposals: List of proposals for each image in the batch
                    Each element is [x1, y1, x2, y2] format
                    
        Returns:
            region_features: [B, N, output_dim] tensor of region features
        """
        batch_size = len(proposals)
        device = feature_maps[0].device if feature_maps else torch.device("cpu")
        
        # Simple implementation: just use global average pooling
        all_region_features = []
        
        for i in range(batch_size):
            # Get the feature map for this batch item
            feat_map = feature_maps[0]  # [B, C, H, W]
            batch_feat_map = feat_map[i:i+1]  # [1, C, H, W] for this batch
            
            # Extract features using adaptive pooling
            pooled = self.pool(batch_feat_map)  # [1, C, 1, 1]
            flattened = self.flatten(pooled)  # [1, C]
            
            # Create linear layer dynamically if not exists or wrong size
            if not hasattr(self, 'linear') or self.linear.in_features != flattened.shape[1]:
                self.linear = nn.Linear(flattened.shape[1], self.output_dim).to(device)
                self.relu = nn.ReLU(inplace=True)
                self.norm = nn.LayerNorm(self.output_dim).to(device)
            
            # Move all layers to correct device
            self.linear = self.linear.to(device)
            self.relu = self.relu.to(device)
            self.norm = self.norm.to(device)
            
            # Apply linear transformation
            region_features = self.linear(flattened)  # [1, output_dim]
            region_features = self.relu(region_features)
            region_features = self.norm(region_features)
            region_features = region_features.unsqueeze(0)  # [1, 1, output_dim]
            
            all_region_features.append(region_features)
        
        # Stack batch features
        region_features = torch.cat(all_region_features, dim=0)  # [B, 1, output_dim]
        return region_features


# Wrapper class that adds RegionCLIP functionality to existing SPAR models
class SPAR_RegionCLIP_Wrapper:
    """Wrapper to add RegionCLIP functionality to existing SPAR models."""
    
    def __init__(self, model):
        """Initialize wrapper with existing model.
        
        Args:
            model: Existing SPAR model
        """
        self.model = model
        self.region_feature_extractor = RegionFeatureExtractor(input_dim=256, output_dim=512)
        self.region_features = None
        self.region_boxes = None
        self.context_features = None
        
        # Initialize CLIP model if available
        if CLIP_AVAILABLE:
            try:
                device = next(model.parameters()).device
                self.clip_model, _ = clip.load("ViT-B/32", device=device)
                self.clip_model.eval()
            except Exception as e:
                print(f"Could not load CLIP model: {e}")
                self.clip_model = None
        else:
            self.clip_model = None
        
        # Set default class names
        self.class_names = ['pedestrian', 'people', 'bicycle', 'car', 'van', 
                           'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    
    def extract_region_features(self, features, proposals):
        """Extract region features from multi-scale features and proposals.
        
        Args:
            features (List[Tensor]): Multi-scale feature maps
            proposals (List[Tensor]): Region proposals for each image
            
        Returns:
            region_features (Tensor): Region features (batch_size, num_regions, feature_dim)
            region_boxes (Tensor): Normalized region boxes (batch_size, num_regions, 4)
        """
        # Get device from features or use CPU as fallback
        device = features[0].device if features and hasattr(features[0], 'device') else torch.device("cpu")
        
        if self.region_feature_extractor is not None:
            region_features = self.region_feature_extractor(features, proposals)
        else:
            # Fallback implementation
            batch_size = len(proposals)
            region_features = torch.zeros(batch_size, 1, 512, device=device)
        
        # Normalize bounding boxes
        region_boxes = []
        for i, proposal in enumerate(proposals):
            if len(proposal) > 0:
                boxes = proposal.clone() / 1536.0  # Normalize by image size
                # Ensure boxes are on the correct device
                boxes = boxes.to(device)
            else:
                boxes = torch.zeros(1, 4, device=device)
            region_boxes.append(boxes)
        
        # Pad to same number of regions across batch
        max_regions = max([rb.size(0) for rb in region_boxes])
        padded_boxes = []
        for rb in region_boxes:
            if rb.size(0) < max_regions:
                padding = torch.zeros(max_regions - rb.size(0), rb.size(1), 
                                    device=device, dtype=rb.dtype)
                rb = torch.cat([rb, padding], dim=0)
            padded_boxes.append(rb)
        
        region_boxes = torch.stack(padded_boxes, dim=0)  # [B, max_regions, 4]
        
        return region_features, region_boxes
    
    def compute_text_features(self, class_names=None):
        """Compute text features using CLIP model.
        
        Args:
            class_names (list, optional): List of class names
            
        Returns:
            text_features (Tensor): Text features or None if CLIP not available
        """
        if self.clip_model is None:
            return None
            
        if class_names is None:
            class_names = self.class_names
            
        try:
            # Create text descriptions
            text_descriptions = [f"a photo of {name}" for name in class_names]
            text_tokens = clip.tokenize(text_descriptions).to(next(self.clip_model.parameters()).device)
            
            # Encode text features
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = F.normalize(text_features, dim=-1)
            
            return text_features
        except Exception as e:
            print(f"Error computing text features: {e}")
            return None
    
    def get_region_features(self):
        """Get extracted region features.
        
        Returns:
            region_features (Tensor): Region features or None if not available
        """
        return getattr(self, 'region_features', None)


def build_spar_regionclip_model(model="yolov8s.yaml", ch=3, nc=10, verbose=True):
    """Build SPAR-RegionCLIP model.
    
    Args:
        model (str): Model configuration file path
        ch (int): Number of input channels
        nc (int): Number of classes
        verbose (bool): Whether to print verbose information
        
    Returns:
        tuple: (SPARModel, SPAR_RegionCLIP_Wrapper) - Base model and RegionCLIP wrapper
    """
    # Create base SPAR model
    base_model = SPARModel(cfg=model, ch=ch, nc=nc, verbose=verbose)
    
    # Create RegionCLIP wrapper
    regionclip_wrapper = SPAR_RegionCLIP_Wrapper(base_model)
    
    return base_model, regionclip_wrapper