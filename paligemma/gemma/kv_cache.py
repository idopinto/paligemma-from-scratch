import torch


class KVCache:
    """Per-layer key/value cache for autoregressive decoding.

    Two parallel lists with one entry per *text* decoder layer, i.e. L =
    text_config.num_hidden_layers (18 for the Gemma-2B in paligemma-3b). The
    vision tower has no cache: SigLIP encodes every patch in a single pass.

        key_cache   = [K_0, K_1, ..., K_L-1]
        value_cache = [V_0, V_1, ..., V_L-1]

    Each entry has shape [B, H_kv, S, h]:
        B     batch
        H_kv  key/value heads (1 in PaliGemma's Gemma-2B, i.e. multi-query)
        S     tokens cached so far -- the only axis that ever grows
        h     head_dim (256)

    Two different "lengths" live in this class, and mixing them up is the main
    source of confusion:

        len(self.key_cache)          -> how many LAYERS have stored anything (0..L)
        self.key_cache[i].shape[-2]  -> how many TOKENS are cached (260, 261, ...)

    The list length climbs 0 -> L during the first forward pass and never
    changes again. The token axis jumps to the prompt length at prefill and
    then grows by one per decode step.

    `GemmaAttention` owns a `layer_idx` and calls `update()` once per forward
    pass, with layers always running in order 0 -> L-1. Nothing here resets, so
    every generation needs a fresh KVCache.
    """

    def __init__(self) -> None:
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def num_items(self):
        """How many tokens are currently cached (not how many layers).

        Reads layer 0 and assumes all layers are in sync, which holds because
        every forward pass updates each layer exactly once, in order. Call it
        before a pass -- as `_merge_input_ids_with_image_features` does to tell
        prefill from decode -- where it means "tokens from previous steps".
        Called midway through a pass it would be ahead, since layers 0..k
        already include the current step while the rest do not.

        Returns 0 when nothing has been cached yet, which is the prefill signal.
        """
        if len(self.key_cache) == 0:
            return 0
        else:  # The shape of the key cache is [B, H_kv, S, h=head_dim]
            first_tensor_from_key_cache = self.key_cache[0]
            num_tokens = first_tensor_from_key_cache.shape[-2]
            return num_tokens

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ):
        """Store this step's keys/values for one layer, return the full history.

        Args:
            key_states: [B, H_kv, q, h] keys for the q new tokens, post-RoPE.
                q is the whole prompt at prefill and 1 during decode.
            value_states: [B, H_kv, q, h] values for those same tokens.
            layer_idx: which slot to write, 0..num_hidden_layers-1.

        Returns:
            (keys, values), each [B, H_kv, S + q, h] -- everything cached for
            this layer, not just the new part. `GemmaAttention` feeds these
            straight into the score matmul once `repeat_kv` has widened H_kv
            up to H_q.

        The branch below asks "does this layer's slot exist yet?", nothing about
        tokens: a list of length 3 has indices 0..2, so index 3 has no tensor to
        concatenate onto. Because layers run in order and the list only ever
        grows by appending, len(key_cache) == layer_idx for every layer of the
        first pass (L appends), and len(key_cache) == L > layer_idx on every
        pass after that (concatenate from then on).
        """
        # Prefill: this layer has never cached anything, so create its slot.
        if len(self.key_cache) <= layer_idx:
            # key_states: [B, H_kv, len(Prompt), h], value states: [B, H_kv, len(Prompt), h]
            self.key_cache.append(
                key_states
            )  # [T0..T_layer_idx-1] -> [T0..T_layer_idx]
            self.value_cache.append(
                value_states
            )  # [T0..T_layer_idx-1] -> [T0..T_layer_idx]
        else:  # Decode: the slot exists, so grow it along the token axis.
            # key_states: [B, H_kv, q=1, h], value states: [B, H_kv, q=1, h]
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )  # [B, H_kv, S, h] -> [B, H_kv, S + 1, h]
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )  # [B, H_kv, S, h] -> [B, H_kv, S + 1, h]
        updated_key_states, updated_value_states = (
            self.key_cache[layer_idx],
            self.value_cache[layer_idx],
        )
        return updated_key_states, updated_value_states

    def get_memory_size(self):
        """Logical size of the cache in bytes: keys plus values, all layers.

        One entry holds B * H_kv * C * hd elements. There are L layers and a
        matching value tensor for each, hence the final * 2.

        This is the size of the tensors, not the memory the process is holding.
        `update` allocates a fresh tensor on every torch.cat and PyTorch's
        caching allocator hangs onto the freed blocks, so
        torch.cuda.memory_reserved() will read noticeably higher.
        """
        if len(self.key_cache) == 0:
            return 0

        L = len(self.key_cache)  # num of layers
        B, Hkv, C, hd = self.key_cache[0].shape  # C is the token axis
        # float16/bfloat16 are 2 bytes and float32 is 4. Narrower or wider
        # dtypes (fp8, float64) would need tensor.element_size() instead.
        dtype = self.key_cache[0].dtype
        if dtype == torch.float16 or dtype == torch.bfloat16:
            bytes_per_elem = 2
        else:
            bytes_per_elem = 4

        return B * L * Hkv * hd * C * 2 * bytes_per_elem
