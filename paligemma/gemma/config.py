from dataclasses import dataclass, fields

# GEMMA_2B_TEXT_CONFIG = {
#     "vocab_size": 257152,
#     "hidden_size": 2048,
#     "intermediate_size": 16384,
#     "num_hidden_layers": 18,
#     "num_attention_heads": 8,
#     "num_key_value_heads": 1,
#     "head_dim": 256,
# }


@dataclass
class GemmaConfig:
    vocab_size: int = 257152
    hidden_size: int = 2048
    intermediate_size: int = 16384
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    max_position_embddings: int = 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attentions_bias: bool = False
    attention_dropout: float = 0.0
    pad_token_id: int | None = None

    @classmethod
    def from_dict(cls, config: dict) -> GemmaConfig:
        """Build from a checkpoint's config.json.

        HF checkpoints carry keys we don't model (model_type, projector_hidden_act,
        vision_use_head, ...); drop anything that isn't a declared field.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in config.items() if k in known})

    # def __init__(self, **kwargs):
    #     for key, value in kwargs.items():
    #         setattr(self, key, value)

    # def __init__(
    #     self,
    #     vocab_size: int,
    #     hidden_size: int,
    #     intermediate_size: int,
    #     num_hidden_layers: int,
    #     num_attention_heads: int,
    #     num_key_value_heads: int,
    #     head_dim: int = 256,
    #     max_position_embddings: int = 8192,
    #     rms_norm_eps: float = 1e-6,
    #     rope_theta: float = 10000.0,
    #     attentions_bias: bool = False,
    #     attention_dropout: float = 0.0,
    #     pad_token_id: int | None = None,
    #     **kwargs,
    # ):
    #     self.vocab_size = vocab_size
    #     self.hidden_size = hidden_size
    #     self.intermediate_size = intermediate_size
    #     self.num_hidden_layers = num_hidden_layers
    #     self.num_attention_heads = num_attention_heads
    #     self.num_key_value_heads = num_key_value_heads
    #     self.head_dim = head_dim
    #     self.max_position_embddings = max_position_embddings
    #     self.rms_norm_eps = rms_norm_eps
    #     self.rope_theta = rope_theta
    #     self.attentions_bias = attentions_bias
    #     self.attention_dropout = attention_dropout
    #     self.pad_token_id = pad_token_id
