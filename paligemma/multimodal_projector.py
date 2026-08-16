import torch
import torch.nn as nn
from paligemma.config import PaliGemmaConfig


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.vision_config.hidden_size,
            config.projection_dim,
            bias=True,
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        # [B, Np, D_image] -> [B, Np, D_proj=D_text ]
        hidden_states = self.linear(image_features)
        return hidden_states
