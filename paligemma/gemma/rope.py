import torch
import torch.nn as nn


class GemmaRotaryEmbedding(nn.Module):
    """Builds the cos/sin tables RoPE rotates queries and keys by.

    Splits a head into dh/2 pairs and rotates pair i by position * theta_i.
    Rotations preserve dot products, so a score between positions m and n
    depends only on m - n.

    `dim` is head_dim (256), not hidden_size. Nothing here is learned, and
    max_position_embeddings is unused -- the tables are built per pass.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: int = 10000,
    ):
        super().__init__()
        self.dim = dim  # set to head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # theta_i = base^(-2i/dim) for i = 0 .. dim/2 - 1, one per rotation pair.
        # Small i rotates fast (short wavelength), large i slowly, so a position
        # ends up encoded at many scales at once.
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids, seq_len) -> torch.Tensor:
        """cos/sin for position_ids [B, seq_len], each returned [B, seq_len, dh].

        `x` supplies device/dtype only; `seq_len` is unused (HF parity). Angles
        are duplicated across the two halves to match `rotate_half`.
        """
        # x: [B, Hkv or Hq, seq_len, head_size]
        self.inv_freq.to(x.device)
        # Copy the inv_freg tensor for batch in the sequence
        # inv_freq_expanded: [B, head_dim // 2, 1]
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        # position_ids_expanded
        position_ids_expanded = position_ids[:, None, :].float()
        # autocast has no mps path, so fall back to cpu rules on Apple silicon.
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        # fp32: cos/sin of a large angle loses most of its precision in bf16.
        with torch.autocast(device_type=device_type, enabled=False):
            # Multiply each theta by the position (which is the argument of the sin and cos functions)
            # freqs [B, head_dim // 2, 1]@ [B, 1, seq_len] -> [B, seq_len, head_dim//2]
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            # emb: [B, seq_len, head_dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [B, seq_len, head_dim]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    (a, b) -> (-b, a), pairing dim i with i + dh/2 -- halves, not neighbours."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate q [B, H_q, S, dh] and k [B, H_kv, S, dh] by their positions.

    cos/sin are [B, S, dh]; unsqueeze_dim=1 adds the head axis to broadcast.
    Keys are rotated before caching, so cached keys are already positioned.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
