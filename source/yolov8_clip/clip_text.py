"""
CLIP text encoder.
Uses existing CLIP implementations for text encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import open_clip
    _has_open_clip = True
except ImportError:
    _has_open_clip = False
    try:
        import clip
        _has_clip = True
    except ImportError:
        _has_clip = False


class CLIPTextEncoder(nn.Module):
    """CLIP text encoder."""

    def __init__(self, model_name="ViT-B/32", pretrained="openai", device="cuda"):
        super().__init__()
        self.device = device
        
        # Try to load open_clip first, fallback to clip
        if _has_open_clip:
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.use_open_clip = True
        elif _has_clip:
            self.clip_model, self.preprocess = clip.load(model_name, device=device)
            self.tokenizer = clip.tokenize
            self.use_open_clip = False
        else:
            raise ImportError(
                "Neither 'open_clip_torch' nor 'clip' is installed. "
                "Please install one of them to use CLIPTextEncoder."
            )
            
        self.clip_model = self.clip_model.to(device).eval()
        
    def forward(self, texts):
        """
        Encode texts.

        Args:
            texts (List[str]): List of text prompts.

        Returns:
            text_features (Tensor): Encoded text features of shape (batch_size, feature_dim).
        """
        # Tokenize texts
        if self.use_open_clip:
            text_tokens = self.tokenizer(texts)
        else:
            text_tokens = self.tokenizer(texts, truncate=True)
            
        text_tokens = text_tokens.to(self.device)
        
        # Use CLIP's built-in text encoding
        text_features = self.clip_model.encode_text(text_tokens)
        
        return text_features