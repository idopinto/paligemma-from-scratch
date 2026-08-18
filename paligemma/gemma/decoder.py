from paligemma.gemma.kv_cache import KVCache
import torch
import torch.nn as nn
from paligemma.gemma.config import GemmaConfig
from paligemma.common.rms_norm import RMSNorm
from paligemma.gemma.attention import GemmaAttention
from paligemma.gemma.mlp import GemmaMLP
import torch.nn.functional as F


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = GemmaAttention(config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: KVCache,
    ):
        # [B, S, D_text]->[B,S, D_text]
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
