import torch
import torch.nn as nn
import torch.nn.functional as F
from paligemma.siglip.config import SiglipVisionConfig


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(
            in_features=config.hidden_size, out_features=config.intermediate_size
        )
        self.fc2 = nn.Linear(
            in_features=config.intermediate_size, out_features=config.hidden_size
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)  # [B, Np, D] -> [B, Np, D_intermediate]
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)  # [B, Np, D_intermediate] -> [B, Np, D]
        return hidden_states
