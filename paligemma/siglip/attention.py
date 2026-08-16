import torch
import torch.nn as nn
import torch.nn.functional as F
from paligemma.siglip.config import SiglipVisionConfig


class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        assert self.embed_dim % self.num_heads == 0, (
            "embedding dimension should be divisible with the number of heads"
        )
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5  # 1 / sqrt(d_k)
        self.dropout_p = self.config.attention_dropout
        self.k_proj = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)
        self.q_proj = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)
        self.v_proj = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)
        self.out_proj = nn.Linear(
            in_features=self.embed_dim, out_features=self.embed_dim
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # hidden_states: [B, Np, D]
        batch_size, seq_len, _ = hidden_states.size()

        # STEP 1: project each patch into Q, K, V (same D, three different W matrices)
        query_states = self.q_proj(hidden_states)  # [B, Np, D]
        key_states = self.k_proj(hidden_states)  # [B, Np, D]
        value_states = self.v_proj(hidden_states)  # [B, Np, D]

        # STEP 2: split D into H heads (dh = D / H), then put heads before sequence
        # so the next matmul runs all heads in one batched op
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # [B, Np, D] -> [B, Np, H, dh] -> [B, H, Np, dh]
        key_states = key_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # [[B, Np, D] -> [B, Np, H, dh] -> [B, H, Np, dh]
        value_states = value_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # [B, Np, H, dh] -> [B, Np, H, dh] -> [B, H, Np, dh]

        # STEP 3: scores = Q @ K^T / sqrt(dh)
        # each query patch gets a similarity with every key patch, per head
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        )  # [B, H, Np,dh] X [B, H, dh, Np]-> [B, H, Np, Np]
        assert attn_weights == (batch_size, self.num_heads, seq_len, seq_len), (
            f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)} but is {attn_weights.size()}"
        )

        # STEP 4: softmax over keys so each query row is a distribution over patches
        # upcast to fp32 for stability, then cast back (fp16 softmax can overflow)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )  # [B, H, Np, Np] -> [B, H, Np, Np]

        # STEP 5: drop some attention connections while training (regularization)
        # PaliGemma sets p=0, so this is a no-op at inference and usually at train too
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)

        # STEP 6: weighted sum of values — each query patch mixes information from all patches
        # row i of attn_weights (a distribution over keys) is dotted with every value vector
        attn_output = torch.matmul(
            attn_weights, value_states
        )  # [B, H, Np, Np] @ [B, H, Np, dh] -> [B, H, Np, dh]
        assert attn_weights == (batch_size, self.num_heads, seq_len, self.head_dim), (
            f"Attention outputs should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)} but is {attn_output.size()}"
        )
        attn_output = (
            attn_output.tranpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.head_dim * self.num_heads)
        )  # # [B, H, Np, dh] -> [B, Np, H, dh] -> [B, Np, H * dh= D]
        attn_output = self.out_proj(attn_output)  # [B, Np, D] X [B, D, D] = [B, Np, D]
        return attn_output, attn_weights
