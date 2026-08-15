import torch
import torch.nn as nn
from .config import SiglipVisionConfig
from .vision_transformer import SiglipVisionTransformer


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, Np, D]
        return self.vision_model(pixel_values=pixel_values)
