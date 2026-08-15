import torch
import torch.nn as nn

from siglip.common.embedding import num_embeddings

from .config import SiglipVisionConfig
from common import Embedding


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,  # 3 (R,G,B)
            out_channels=config.hidden_dim,  # 768
            kernel_size=config.patch_size,  # 16
            stride=config.patch_size,  # for non-overlapping
            padding="valid",  # This indicates no padding is added
        )  # total params count: patch_size ** 2 * embed_dim = 16 * 16* 768 = 196,608
        # 224 / 16 = 14 => 14 **2 = 196 patches
        self.n_patches = (config.image_size // self.patch_size) ** 2
        self.n_positions = self.num_patchs  # 196
        self.position_embedding = Embedding(
            num_embeddings=self.n_positions, embedding_dim=self.embed_dim
        )
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W]
        # Convolve the `patch_size` kernel over the image,
        # with no overlapping patches since the
        # The output of the convolution wil have shape [B, D, NpH, NpW]
        # where NpH = height // patch_size, NpW= width // patch_size
        _, _, height, width = pixel_values.shape
        patch_embeddings = self.patch_embedding(
            pixel_values
        )  # [B,C,H, W] -> [B,D,NpH,NpW]([B, 768, 224, 224] → [B, 768, 14, 14])
        patch_embeddings = patch_embeddings.flatten(
            start_dim=2, end_dim=-1
        )  # [B,D,NpH,NpW] -> [B,D, Np] ([B, 768, 14, 14] → [B, 768, 196])
        patch_embeddings = patch_embeddings.transpose(
            1, 2
        )  # [B, D, Np] -> [B, Np, D] ([B, 768, 196] -> [B, 196, 768])
        position_embeddings = self.position_embedding(self.position_ids)
        final_embeddings = (
            patch_embeddings + position_embeddings
        )  # # [B, D, Np] -> [B, Np, D] ([B, 768, 196] -> [B, 196, 768])
        return final_embeddings
