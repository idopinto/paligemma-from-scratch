class GemmaConfig:
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_key_value_heads: int,
        head_dim: int = 256,
        max_position_embddings: int = 8192,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        attentions_bias: bool = False,
        attention_dropout: float = 0.0,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embddings = max_position_embddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attentions_bias = attentions_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id
