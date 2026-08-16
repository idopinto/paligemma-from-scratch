import torch
import torch.nn as nn
from .config import SiglipVisionConfig
from .vision_embeddings import SiglipVisionEmbeddings
from .encoder import SiglipEncoder
from paligemma.common.layer_norm import LayerNorm


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = LayerNorm(
            embed_dim=config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)  # [B, C, H, W] -> [B, Np, D]
        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states
        )  # [B, Np, D] -> [B, Np, D]
        last_hidden_state = self.post_layernorm(
            last_hidden_state
        )  # [B, Np, D] -> [B, Np, D]
        return last_hidden_state
