import time

import torch
import torch.nn as nn
from paligemma.config import PaliGemmaConfig
from paligemma.multimodal_projector import PaliGemmaMultiModalProjector
from paligemma.siglip.modeling_siglip import SiglipVisionModel
from paligemma.gemma.modeling_gemma import GemmaForCausalLM
from paligemma.gemma.kv_cache import KVCache
from paligemma.processor.processing_paligemma import PaliGemmaProcessor


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        self.language_model = GemmaForCausalLM(config.text_config)
        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_features: torch.FloatTensor = None,
        attention_mask: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> tuple:
        """
        Either `pixel_values` (raw image) or `image_features` (pre-encoded patches)
        must be provided. Pass `image_features` on decode steps to skip re-running
        the vision tower, which never changes across the generation loop.
        """
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # Extract the input embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(
            input_ids
        )  # [B, S] -> [B, S, D_text]

        # On the first (prefill) step pixel_values are provided and we encode them.
        # On every decode step the caller passes pre-computed image_features instead,
        # so we skip the 27-layer SigLIP encoder entirely.
        if image_features is None:
            selected_image_feature = self.vision_tower(
                pixel_values.to(inputs_embeds.dtype)
            )  # [B, C, H, W] -> [B, Np, D_image]
            image_features = self.multi_modal_projector(
                selected_image_feature
            )  # [B, Np, D_image] -> [B, Np, D_text]

        # Merge the embeddings of the text tokens and the image tokens
        inputs_embeds, attention_mask, position_ids = (
            self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask, kv_cache
            )
        )  # inputs_embeds: [B, Np + (S - Np), D_text]
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )
        return outputs

    def generate(
        self,
        model_inputs: dict,
        processor: PaliGemmaProcessor,
        max_tokens_to_generate: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ):
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        pixel_values = model_inputs["pixel_values"]

        kv_cache = KVCache()
        # Generate tokens until you see the stop token
        stop_token = processor.tokenizer.eos_token_id
        generated_tokens = []

        # Encode the image once before the loop. The vision tower (27 SigLIP
        # layers) output never changes, so there is no reason to re-run it on
        # every decode step. We pass the cached features on steps 2+.
        with torch.no_grad():
            selected_image_feature = self.vision_tower(
                pixel_values.to(
                    next(self.vision_tower.parameters()).dtype
                )
            )
            cached_image_features = self.multi_modal_projector(selected_image_feature)

        start_time = time.perf_counter()
        for step in range(max_tokens_to_generate):
            # Prefill (step 0): pass image_features so forward() skips the vision tower.
            # Decode (step 1+): same cached features, still skip the vision tower.
            outputs = self.forward(
                input_ids=input_ids,
                image_features=cached_image_features,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
            kv_cache = outputs["kv_cache"]
            next_token_logits = outputs["logits"][:, -1, :]
            # Sample the next token
            if do_sample:
                # Apply temperature
                next_token_logits = torch.softmax(
                    next_token_logits / temperature, dim=-1
                )
                next_token = self._sample_top_p(next_token_logits, top_p)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            assert next_token.size() == (1, 1)
            next_token = next_token.squeeze(0)  # Remove batch dimension
            generated_tokens.append(next_token)
            # Stop if the stop token has been generated
            if next_token.item() == stop_token:
                break
            # Append the next token to the input
            input_ids = next_token.unsqueeze(-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
            )

        generated_tokens = torch.cat(generated_tokens, dim=-1)
        generation_time_s = time.perf_counter() - start_time
        num_tokens_generated = generated_tokens.numel()
        tokens_per_second = (
            num_tokens_generated / generation_time_s if generation_time_s > 0 else 0.0
        )

        # Decode the generated tokens
        decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        response = {
            "response": decoded,
            "generation_time_s": generation_time_s,
            "num_tokens_generated": num_tokens_generated,
            "tokens_per_second": tokens_per_second,
        }
        return response

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: KVCache,
    ):
        """Build the mixed image+text sequence Gemma actually attends over.

        The processor lays tokens out as:
            [<image>] * Np  +  [BOS]  +  text tokens  +  [\\n]
        `input_ids` still holds dummy `<image>` ids. This method swaps those
        positions for the real SigLIP+projector vectors, then builds the
        prefix-LM attention mask and RoPE position ids.

        Args:
            image_features: [B, Np, D] vision embeddings already projected to text dim.
            inputs_embeds: [B, S, D] embeddings of the full sequence (placeholders + text).
            input_ids: [B, S] used only to locate image / text / pad positions.
            attention_mask: [B, S] 1 = real token, 0 = pad.
            kv_cache: empty during prefill; populated during decode.

        Returns:
            final_embedding: [B, S, D] mixed image+text embeddings.
            causal_mask: [B, 1, Q, KV] additive mask (0 = attend, min_dtype = block).
            position_ids: [B, Q] RoPE positions.
        """
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape

        # Keep image vectors from dominating text embeddings (same 1/sqrt(D) idea as attention).
        scaled_image_features = image_features / (self.config.hidden_size**0.5)

        # --- 1. Merge embeddings ------------------------------------------------
        # Start from zeros [B, S, D], then copy text, scatter images, zero pads.
        final_embedding = torch.zeros(
            batch_size,
            sequence_length,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

        # [B, S] boolean maps: each position is exactly one of {text, image, pad}.
        text_mask = (input_ids != self.config.image_token_index) & (
            input_ids != self.pad_token_id
        )
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == self.pad_token_id

        # Expand [B, S] -> [B, S, D] so the masks broadcast over the embedding dim.
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Copy text embeddings into text positions.
        final_embedding = torch.where(
            text_mask_expanded, inputs_embeds, final_embedding
        )
        # Scatter image patches into <image> positions. Cannot use torch.where:
        # scaled_image_features is [B, Np, D], not [B, S, D] — Np < S.
        final_embedding = final_embedding.masked_scatter(
            image_mask_expanded, scaled_image_features
        )
        # Zero padding so pad tokens contribute nothing to attention.
        final_embedding = torch.where(
            pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding
        )

        # --- 2. Attention mask (prefix-LM) --------------------------------------
        # Prefill: image + prompt attend bidirectionally (all zeros = nothing blocked).
        # Decode: the new token (Q=1) attends to the full cached prefix + itself.
        # Additive mask: 0 = attend. This demo also asserts no padding.
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]  # seq_len

        if kv_cache is None or kv_cache.num_items() == 0:
            # Prefill: [B, Q, Q] — every prefix token can see every other prefix token.
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Decode: generating one token, so Q must be 1. KV = cache + the new query.
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )
        # [B, Q, KV] -> [B, 1, Q, KV] so it broadcasts over attention heads.
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Prefill: cumsum of the mask -> 1, 2, 3, ...  Pad positions are forced to 1
            # so they do not get a fake increasing index.
            position_ids = (
                attention_mask.cumsum(-1).masked_fill(attention_mask == 0, 1).to(device)
            )
        return final_embedding, causal_mask, position_ids

    def _sample_top_p(self, probs: torch.Tensor, p: float):
        # (B, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # (B, vocab_size)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # (B, vocab_size)
        # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
        mask = probs_sum - probs_sort > p
        # Zero out all the probabilities of tokens that are not selected by the Top P
        probs_sort[mask] = 0.0
        # Redistribute the probabilities so that they sum up to 1.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # Sample a token (its index) from the top p distribution
        next_token = torch.multinomial(probs_sort, num_samples=1)
        # Get the token position in the vocabulary corresponding to the sampled index
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    def tie_weights(self):
        self.language_model.tie_weights()
