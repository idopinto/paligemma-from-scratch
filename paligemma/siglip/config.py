from dataclasses import dataclass


# polygamma comes in different sizes
# 224, 448, 896
@dataclass
class SiglipVisionConfig:
    hidden_size: int = 768
    intermediate_size: int = 3072  # for the mlp layers
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int = 224  # any image is resized to this `image_size`
    patch_size: int = 16  # 16 X 16 patches
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    num_image_tokens: int | None = None


@dataclass
class SiglipSo400mVisionConfig(SiglipVisionConfig):
    """Shape-optimized ViT (~400M). PaliGemma's vision tower.

    Same architecture as google/siglip-so400m-patch14-384.
    PaliGemma later runs this at 224 / 448 / 896 (pos embeds interpolated).
    """

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 384
    patch_size: int = 14
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    num_image_tokens: int = (384 // 14) ** 2  # 27 ** 2 = 729
