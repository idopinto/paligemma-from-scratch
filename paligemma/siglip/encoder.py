import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SiglipVisionConfig
from paligemma.siglip.mlp import SiglipMLP
from paligemma.siglip.attention import SiglipAttention


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # residual / hidden_states: [B, Np, D]
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)  # [B, Np, D] -> [B, Np, D]
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states
        )  # [B, Np, D] -> [B, Np, D]
        hidden_states = residual + hidden_states  # [B, Np, D] -> [B, Np, D]
        residual = hidden_states  # [B, Np, D]
        hidden_states = self.layer_norm2(hidden_states)  # [B, Np, D] -> [B, Np, D]
        hidden_states = self.mlp(hidden_states)  # [B, Np, D] -> [B, Np, D]
        hidden_states = residual + hidden_states  # [B, Np, D] -> [B, Np, D]
        return hidden_states

    # def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    #     hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states))
    #     hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
    #     return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        # [B, Np, D] -> [B, Np, D]
        hidden_states = input_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states
