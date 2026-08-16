from paligemma.kv_cache import KVCache
import torch
import torch.nn as nn
from paligemma.gemma.config import GemmaConfig
from paligemma.common.rms_norm import RMSNorm
from paligemma.gemma.decoder import GemmaDecoderLayer


class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                GemmaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds  # [B, S, D_text]
        normalizer = torch.tensor(
            self.config.hidden_size**0.5, dtype=hidden_states.dtype
        )
        hidden_states = hidden_states * normalizer
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausalLM(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            in_features=config.hidden_size, out_features=config.vocab_size, bias=False
        )

    def forward(
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        kv_cache: KVCache,
    ) -> tuple:
        # input_embeds: [B, S, D_text]
        # outputs: [B, S, D_text]
        outputs = self.model(attention_mask, position_ids, inputs_embeds, kv_cache)
        hidden_states = outputs
        logits = self.lm_head(hidden_states).float()
        return_data = {"logits": logits}
        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache
        return return_data

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight
