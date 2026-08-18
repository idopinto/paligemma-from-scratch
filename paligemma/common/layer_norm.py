import torch
import torch.nn as nn

"""
Why?
Big Change in input of a layer
            |
            V
Big change in output of a layer
            |
            V
Big change in loss
            |
            V
Big change in gradient 
            |
            V
Big change in the weights of the networks
            |
            V
model that trains slowly!
"""


class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        # Named `weight`/`bias` (not `gamma`/`beta`) to match HF checkpoint keys.
        self.weight = nn.Parameter(torch.ones(embed_dim), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(embed_dim), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ((x - mean) / sqrt(var + eps)) * weight + bias
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(
            dim=-1,
            unbiased=False,
            keepdim=True,
        )

        x_norm = (x - mean) / (var + self.eps) ** 0.5
        return self.weight * x_norm + self.bias


# Test LayerNorm against nn.LayerNorm for equivalence
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)  # example tensor

    # LayerNorm using our implementation
    embed_dim = x.shape[-1]
    eps = 1e-5
    my_ln = LayerNorm(embed_dim, eps)
    # Patch: our LayerNorm is missing nn.Module init and parameters
    my_ln.eval()

    # LayerNorm using PyTorch
    torch_ln = nn.LayerNorm(embed_dim, eps)
    torch_ln.eval()

    # To make a fair test, align the weights/bias (which our impl doesn't have)
    # Compare only for zero weight/bias

    y_my = my_ln(x)
    y_torch = torch_ln(x)
    # center and scale like nn.LayerNorm, but default weight=1,bias=0

    print("Max abs diff:", (y_my - y_torch).abs().max().item())
    print("Are close:", torch.allclose(y_my, y_torch, atol=1e-5))
