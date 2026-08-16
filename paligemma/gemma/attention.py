import torch
import torch.nn as nn
import torch.nn.functional as F
from paligemma.gemma.config import GemmaConfig


class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int | None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0

        self.k_proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.num_heads * self.head_dim,
            bias=config.attentions_bias,
        )
        self.q_proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.num_key_value_heads * self.head_dim,
            bias=config.attentions_bias,
        )
        self.v_proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.num_key_value_heads * self.head_dim,
            bias=config.attentions_bias,
        )
        self.out_proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.num_key_value_heads * self.head_dim,
            bias=config.attentions_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
