import math
from paligemma.gemma.kv_cache import KVCache
import torch
import torch.nn as nn
import torch.nn.functional as F
from paligemma.gemma.config import GemmaConfig


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads so they line up 1:1 with query heads (GQA).

    hidden_states: [B, num_kv_heads, S, head_dim]
    returns:       [B, num_kv_heads * n_rep, S, head_dim]  == [B, num_q_heads, S, head_dim]

    For PaliGemma's Gemma-2B, num_kv_heads=1, num_q_heads=8, so n_rep=8:
    one shared K/V head is copied to each of the 8 query heads.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # [B, kv_heads, S, dh] -> [B, kv_heads, 1, S, dh] -> [B, kv_heads, n_rep, S, dh]
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class GemmaAttention(nn.Module):
    """Grouped-query self-attention with RoPE and a KV cache.

    Unlike SigLIP (MHA: H_q == H_kv, head_dim = D / H), Gemma-2B uses GQA:
    8 query heads share 1 KV head. `head_dim` is explicit (256), not D / H_q.
    `layer_idx` selects which slot in `kv_cache` this layer reads/writes.
    """

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

        self.q_proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.num_heads * self.head_dim,
            bias=config.attentions_bias,
        )
        self.k_proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.num_key_value_heads * self.head_dim,
            bias=config.attentions_bias,
        )
        self.v_proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.num_key_value_heads * self.head_dim,
            bias=config.attentions_bias,
        )
        self.o_proj = nn.Linear(
            in_features=self.num_heads * self.head_dim,
            out_features=self.hidden_size,
            bias=config.attentions_bias,
        )
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        kv_cache: KVCache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: [B, Q, D]  (Q = q_len; 1 during decode)
        batch_size, q_len, _ = hidden_states.size()

        # STEP 1: project tokens into Q, K, V. GQA: Q has more heads than K/V.
        query_states = self.q_proj(hidden_states)  # [B, Q, H_q * dh]
        key_states = self.k_proj(hidden_states)  # [B, Q, H_kv * dh]
        value_states = self.v_proj(hidden_states)  # [B, Q, H_kv * dh]

        # STEP 2: split into heads, then put heads before sequence so the
        # next matmul runs all heads in one batched op.
        query_states = query_states.view(
            batch_size, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # [B, Q, H_q * dh] -> [B, Q, H_q, dh] -> [B, H_q, Q, dh]
        key_states = key_states.view(
            batch_size, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)  # [B, Q, H_kv * dh] -> [B, Q, H_kv, dh] -> [B, H_kv, Q, dh]
        value_states = value_states.view(
            batch_size, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)  # [B, Q, H_kv * dh] -> [B, Q, H_kv, dh] -> [B, H_kv, Q, dh]

        # STEP 3: RoPE rotates Q and K by position so attention is position-aware.
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # STEP 4: append this step's K/V to the cache (prefill: whole prefix;
        # decode: one new token). Cached K/V stay at H_kv heads until repeat.
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(
                key_states, value_states, self.layer_idx
            )  # [B, H_kv, KV, dh]  (KV = cache_len + Q)

        # STEP 5: copy each KV head n_rep times so H_kv matches H_q for the matmul.
        key_states = repeat_kv(
            key_states, self.num_key_value_groups
        )  # [B, H_q, KV, dh]
        value_states = repeat_kv(
            value_states, self.num_key_value_groups
        )  # [B, H_q, KV, dh]

        # STEP 6: scores = Q @ K^T / sqrt(dh)
        # [B, H_q, Q, dh] x [B, H_q, dh, KV] -> [B, H_q, Q, KV]
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        # Additive mask: 0 = attend, large negative = block. Broadcasts over heads.
        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        # Softmax over keys; fp32 for stability, then cast back.
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )  # [B, H_q, Q, KV]
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        # [B, H_q, Q, KV] @ [B, H_q, KV, dh] -> [B, H_q, Q, dh]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # STEP 7: merge heads and project back to model dim.
        # [B, H_q, Q, dh] -> [B, Q, H_q, dh]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [B, Q, H_q, dh] -> [B, Q, H_q * dh]
        attn_output = attn_output.view(batch_size, q_len, -1)
        # W_o: [B, Q, H_q * dh] -> [B, Q, D]
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights
