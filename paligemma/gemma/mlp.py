import torch
import torch.nn as nn
import torch.nn.functional as F
from paligemma.gemma.config import GemmaConfig


class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate_proj(x)
        x = F.gelu(x, approximate="tanh")
        x = x * self.up_proj(x)
        x = self.down_proj(x)
        return x
