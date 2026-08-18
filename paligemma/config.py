from paligemma.siglip.config import SiglipVisionConfig
from paligemma.gemma.config import GemmaConfig


class PaliGemmaConfig:
    def __init__(
        self,
        vision_config: SiglipVisionConfig | dict | None = None,
        text_config: GemmaConfig | dict | None = None,
        ignore_index: int = -100,
        image_token_index: int = 256000,
        vocab_size: int = 257152,
        projection_dim: int = 2048,
        hidden_size: int = 2048,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id
        # Accept an already-built config, a dict from a checkpoint's config.json,
        # or nothing at all.
        if vision_config is None:
            vision_config = SiglipVisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = SiglipVisionConfig.from_dict(vision_config)
        self.vision_config = vision_config
        if text_config is None:
            text_config = GemmaConfig()
        if isinstance(text_config, dict):
            text_config = GemmaConfig.from_dict(
                {"pad_token_id": pad_token_id, **text_config}
            )
        self.text_config = text_config
        self.vocab_size = vocab_size
        self.text_config.num_image_tokens = (
            self.vision_config.image_size // self.vision_config.patch_size
        ) ** 2
        self.vision_config.projection_dim = projection_dim

    def __repr__(self) -> str:
        vc = self.vision_config
        tc = self.text_config
        num_image_tokens = getattr(tc, "num_image_tokens", None)
        return (
            "PaliGemmaConfig(\n"
            f"  ignore_index={self.ignore_index},\n"
            f"  image_token_index={self.image_token_index},\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  projection_dim={self.projection_dim},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  pad_token_id={self.pad_token_id},\n"
            f"  is_encoder_decoder={self.is_encoder_decoder},\n"
            "  vision_config=SiglipVisionConfig(\n"
            f"    hidden_size={vc.hidden_size},\n"
            f"    intermediate_size={vc.intermediate_size},\n"
            f"    num_hidden_layers={vc.num_hidden_layers},\n"
            f"    num_attention_heads={vc.num_attention_heads},\n"
            f"    num_channels={vc.num_channels},\n"
            f"    image_size={vc.image_size},\n"
            f"    patch_size={vc.patch_size},\n"
            f"    layer_norm_eps={vc.layer_norm_eps},\n"
            f"    attention_dropout={vc.attention_dropout},\n"
            f"    num_image_tokens={vc.num_image_tokens},\n"
            f"    projection_dim={getattr(vc, 'projection_dim', None)},\n"
            "  ),\n"
            "  text_config=GemmaConfig(\n"
            f"    vocab_size={tc.vocab_size},\n"
            f"    hidden_size={tc.hidden_size},\n"
            f"    intermediate_size={tc.intermediate_size},\n"
            f"    num_hidden_layers={tc.num_hidden_layers},\n"
            f"    num_attention_heads={tc.num_attention_heads},\n"
            f"    num_key_value_heads={tc.num_key_value_heads},\n"
            f"    head_dim={tc.head_dim},\n"
            f"    max_position_embddings={tc.max_position_embddings},\n"
            f"    rms_norm_eps={tc.rms_norm_eps},\n"
            f"    rope_theta={tc.rope_theta},\n"
            f"    attentions_bias={tc.attentions_bias},\n"
            f"    attention_dropout={tc.attention_dropout},\n"
            f"    pad_token_id={tc.pad_token_id},\n"
            f"    num_image_tokens={num_image_tokens},\n"
            "  ),\n"
            ")"
        )
