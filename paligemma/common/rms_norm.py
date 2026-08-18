import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, embed_dim: int, eps: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(embed_dim), requires_grad=True)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, S, D]
        output = self._norm(x)
        # Llama does x.to(float16) while gemma does (x*w).to(float16)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)
