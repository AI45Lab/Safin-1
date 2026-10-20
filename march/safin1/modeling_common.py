"""Shared Safin-1 modeling common for dense and MoE remote code."""

from __future__ import annotations

from typing import Any

import inspect
import torch
from torch import nn
from transformers.masking_utils import create_causal_mask
from .march_gdn import trace_hsd_internal
from .cache_utils import (
    _cache_expanded_length, _cache_expanded_lengths, _cache_has_previous_state,
    _cache_text_lengths, _patch_hsd_cache_get_seq_length, _reorder_hsd_sidecar_cache,
    _RowFullAttentionCache,
    _full_attention_layer_idx,
    _merge_row_full_attention_caches,
    _slice_row_full_attention_cache,
)
from .anchor_layout import _anchor_enabled, _anchor_text_indices, _anchor_has_tokens, _filter_anchor_hidden, _filter_batch_sequence_to_text, _filter_position_embeddings_to_text, _scatter_text_hidden


def _config_return_dict(config) -> bool:
    value = getattr(config, "return_dict", None)
    if value is not None:
        return bool(value)
    return bool(getattr(config, "use_return_dict", True))


def _as_tuple_output(output):
    return tuple(value for value in output.to_tuple() if value is not None)


def _is_topk_router(module: nn.Module) -> bool:
    return module.__class__.__name__ == "Qwen3_5MoeTopKRouter"


def _filter_router_logits(logits: torch.Tensor, anchor_mask: torch.Tensor | None) -> torch.Tensor:
    if anchor_mask is None or not bool(anchor_mask.any().item()):
        return logits
    anchor_mask = anchor_mask.to(device=logits.device, dtype=torch.bool)
    text_mask = ~anchor_mask
    text_counts = text_mask.sum(dim=1)
    if not torch.equal(text_counts, text_counts[:1].expand_as(text_counts)):
        raise NotImplementedError("Safin-1 HSD requires equal text slot counts across batch rows.")
    flat_text_mask = text_mask.reshape(-1)
    if logits.ndim >= 2 and logits.shape[0] == flat_text_mask.numel():
        return logits[flat_text_mask]
    if logits.ndim >= 3 and tuple(logits.shape[:2]) == tuple(anchor_mask.shape):
        return logits[text_mask].view(anchor_mask.shape[0], int(text_counts[0].item()), *logits.shape[2:])
    return logits


def _record_router_logits(model: nn.Module, enabled: bool, anchor_mask: torch.Tensor | None):
    if not enabled:
        return [], []

    router_logits = []
    handles = []

    def hook(_module, _inputs, output):
        if isinstance(output, tuple) and output:
            router_logits.append(_filter_router_logits(output[0], anchor_mask))
        elif torch.is_tensor(output):
            router_logits.append(_filter_router_logits(output, anchor_mask))

    for module in model.modules():
        if _is_topk_router(module):
            handles.append(module.register_forward_hook(hook))
    return router_logits, handles




def _run_layer_mlp_delta(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    anchor_mask: torch.Tensor | None,
    hsd_valid_mask: torch.Tensor | None = None,
    text_valid_mask: torch.Tensor | None = None,
    anchor_valid_mask: torch.Tensor | None = None,
):
    def _run_standard_mlp(states: torch.Tensor) -> torch.Tensor:
        output = layer.mlp(states)
        return output[0] if isinstance(output, tuple) else output

    def _run_standard_mlp_by_mask(mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=hidden_states.device, dtype=torch.bool)
        if tuple(mask.shape) != tuple(hidden_states.shape[:2]):
            raise ValueError(f"MLP valid mask shape {tuple(mask.shape)} must match hidden states {tuple(hidden_states.shape[:2])}.")
        output = hidden_states.new_zeros(hidden_states.shape)
        for batch_idx in range(hidden_states.shape[0]):
            row_mask = mask[batch_idx]
            if not bool(row_mask.any().item()):
                continue
            row_hidden = hidden_states[batch_idx : batch_idx + 1, row_mask]
            row_output = _run_standard_mlp(row_hidden)
            if not torch.is_tensor(row_output):
                raise TypeError("HSD MLP requires the layer MLP to return a tensor.")
            output[batch_idx, row_mask] = row_output.reshape(-1, hidden_states.shape[-1]).to(dtype=output.dtype)
        return output

    if hsd_valid_mask is not None:
        hsd_valid_mask = hsd_valid_mask.to(device=hidden_states.device, dtype=torch.bool)
        if tuple(hsd_valid_mask.shape) != tuple(hidden_states.shape[:2]):
            raise ValueError(
                f"hsd_valid_mask shape {tuple(hsd_valid_mask.shape)} must match hidden states {tuple(hidden_states.shape[:2])}."
            )
        if not bool(hsd_valid_mask.all().item()):
            return _run_standard_mlp_by_mask(hsd_valid_mask)

    return _run_standard_mlp(hidden_states)


def _run_full_attention_padded_fallback(
    layer: nn.Module,
    *,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    past_key_values,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    anchor_mask: torch.Tensor,
    text_valid_mask: torch.Tensor | None,
    **kwargs,
) -> torch.Tensor | None:
    if text_valid_mask is None:
        return None
    text_hidden_states = _filter_anchor_hidden(hidden_states, anchor_mask)
    text_valid_slots = _filter_batch_sequence_to_text(text_valid_mask, anchor_mask).to(
        device=text_hidden_states.device,
        dtype=torch.bool,
    )
    layer_idx = _full_attention_layer_idx(layer)
    cached_per_row = False
    cache_layer = None
    if past_key_values is not None:
        layers = getattr(past_key_values, "layers", None)
        cache_layer = layers[layer_idx] if layers is not None and len(layers) > layer_idx else None
        cached_per_row = torch.is_tensor(getattr(cache_layer, "hsd_text_cache_lengths_per_row", None))
    if bool(text_valid_slots.all().item()) and not cached_per_row:
        return None

    text_position_ids = _filter_batch_sequence_to_text(position_ids, anchor_mask)
    text_position_embeddings = _filter_position_embeddings_to_text(position_embeddings, anchor_mask)
    output = text_hidden_states.new_zeros(text_hidden_states.shape)
    row_caches: list[_RowFullAttentionCache | None] = []
    for batch_idx in range(text_hidden_states.shape[0]):
        valid = text_valid_slots[batch_idx]
        if not bool(valid.any().item()):
            raise ValueError("Safin-1 full-attention fallback requires at least one valid text token.")
        row_hidden = text_hidden_states[batch_idx : batch_idx + 1, valid]
        row_position_ids = None
        if torch.is_tensor(text_position_ids):
            row_position_ids = text_position_ids[batch_idx : batch_idx + 1, valid]
        row_position_embeddings = tuple(item[batch_idx : batch_idx + 1, valid] for item in text_position_embeddings)
        row_cache = _slice_row_full_attention_cache(past_key_values, layer_idx, batch_idx, text_hidden_states.shape[0])
        # Padding/anchors have been removed, and each row has its own KV length.
        # Let Transformers select the backend's mask representation, including
        # explicit causal masks for eager/rectangular SDPA and None for Flash.
        row_past_length = 0 if row_cache is None else row_cache.get_seq_length()
        row_mask = _create_qwen_causal_mask(
            config=layer.self_attn.config,
            inputs_embeds=row_hidden,
            attention_mask=None,
            cache_position=torch.arange(
                row_past_length, row_past_length + row_hidden.shape[1],
                device=row_hidden.device,
            ),
            past_key_values=row_cache,
            position_ids=row_position_ids,
        )
        row_output, _ = layer.self_attn(
            hidden_states=row_hidden,
            attention_mask=row_mask,
            position_ids=row_position_ids,
            past_key_values=row_cache,
            position_embeddings=row_position_embeddings,
            **kwargs,
        )
        output[batch_idx, valid] = row_output.reshape(-1, output.shape[-1]).to(output.dtype)
        row_caches.append(row_cache)
    _merge_row_full_attention_caches(past_key_values, layer_idx, row_caches)
    return _scatter_text_hidden(hidden_states, output, anchor_mask)


def _historical_hsd_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    anchor_mask: torch.Tensor | None = None,
    hsd_valid_mask: torch.Tensor | None = None,
    text_valid_mask: torch.Tensor | None = None,
    anchor_valid_mask: torch.Tensor | None = None,
    persistent_memory_mask: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    layer_idx = int(getattr(self, "_historical_hsd_layer_idx", getattr(self, "layer_idx", 0)))
    trace_anchor_mask = anchor_mask if anchor_mask is not None and tuple(anchor_mask.shape) == tuple(hidden_states.shape[:2]) else None
    trace_hsd_internal(layer_idx, "decoder_input", _filter_anchor_hidden(hidden_states, trace_anchor_mask))
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    trace_hsd_internal(layer_idx, "input_layernorm_out", _filter_anchor_hidden(hidden_states, trace_anchor_mask))

    if self.layer_type == "linear_attention" or getattr(self, "_historical_hsd_self_attn_replaced", False):
        hsd_attn = self.linear_attn if self.layer_type == "linear_attention" else self.self_attn
        hidden_states = hsd_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
            anchor_mask=anchor_mask,
            hsd_valid_mask=hsd_valid_mask,
            text_valid_mask=text_valid_mask,
            anchor_valid_mask=anchor_valid_mask,
            persistent_memory_mask=persistent_memory_mask,
            **kwargs,
        )
    elif self.layer_type == "full_attention":
        padded_full_attention = None
        if anchor_mask is not None:
            padded_full_attention = _run_full_attention_padded_fallback(
                self,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                anchor_mask=anchor_mask,
                text_valid_mask=text_valid_mask,
                **kwargs,
            )
        if padded_full_attention is not None:
            hidden_states = padded_full_attention
        elif _anchor_has_tokens(anchor_mask):
            text_hidden_states = _filter_anchor_hidden(hidden_states, anchor_mask)
            text_position_ids = _filter_batch_sequence_to_text(position_ids, anchor_mask)
            text_position_embeddings = _filter_position_embeddings_to_text(position_embeddings, anchor_mask)
            text_hidden_states, _ = self.self_attn(
                hidden_states=text_hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                position_embeddings=text_position_embeddings,
                **kwargs,
            )
            hidden_states = _scatter_text_hidden(hidden_states, text_hidden_states, anchor_mask)
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )

    trace_hsd_internal(layer_idx, "attention_raw_out", _filter_anchor_hidden(hidden_states, trace_anchor_mask))
    hidden_states = residual + hidden_states
    trace_hsd_internal(layer_idx, "after_attn_residual", _filter_anchor_hidden(hidden_states, trace_anchor_mask))
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    trace_hsd_internal(layer_idx, "post_attention_layernorm_out", _filter_anchor_hidden(hidden_states, trace_anchor_mask))
    hidden_states = _run_layer_mlp_delta(
        self,
        hidden_states,
        anchor_mask,
        hsd_valid_mask=hsd_valid_mask,
        text_valid_mask=text_valid_mask,
        anchor_valid_mask=anchor_valid_mask,
    )
    trace_hsd_internal(layer_idx, "mlp_out", _filter_anchor_hidden(hidden_states, trace_anchor_mask))
    hidden_states = residual + hidden_states
    trace_hsd_internal(layer_idx, "decoder_output", _filter_anchor_hidden(hidden_states, trace_anchor_mask))
    return hidden_states


def _update_linear_attn_mask(model: Any, attention_mask, cache_position, past_key_values):
    try:
        parameters = list(inspect.signature(model._update_linear_attn_mask).parameters)
        if len(parameters) >= 2 and parameters[1] == "cache_position":
            return model._update_linear_attn_mask(attention_mask, cache_position)
        return model._update_linear_attn_mask(attention_mask, past_key_values)
    except (TypeError, ValueError):
        pass
    try:
        return model._update_linear_attn_mask(attention_mask, cache_position)
    except (TypeError, AttributeError):
        return model._update_linear_attn_mask(attention_mask, past_key_values)


def _create_qwen_causal_mask(config, inputs_embeds, attention_mask, cache_position, past_key_values, position_ids):
    try:
        parameters = inspect.signature(create_causal_mask).parameters
    except (TypeError, ValueError):
        parameters = {}
    embed_kwarg = "inputs_embeds" if "inputs_embeds" in parameters else "input_embeds"
    try:
        return create_causal_mask(
            config=config,
            **{embed_kwarg: inputs_embeds},
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
    except TypeError:
        return create_causal_mask(
            config=config,
            **{"input_embeds" if embed_kwarg == "inputs_embeds" else "inputs_embeds": inputs_embeds},
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )


def _decoder_layer_accepts_cache_position(decoder_layer) -> bool:
    try:
        param = inspect.signature(decoder_layer.forward).parameters.get("cache_position")
    except (TypeError, ValueError):
        return False
    return param is not None and param.kind is not inspect.Parameter.VAR_KEYWORD


class Safin1AnchorEmbeddingMixin:
    """Memory-anchor embedding methods shared by dense and MoE text models."""

    def _initialize_anchor_embedding_weight(self) -> None:
        init_from = str(getattr(self.config, "anchor_embedding_init_from", "random"))
        if init_from == "random":
            nn.init.normal_(self.anchor_embedding.weight, mean=0.0, std=self.config.initializer_range)
        else:
            embeddings = self.get_input_embeddings()
            weight = getattr(embeddings, "weight", None)
            if weight is None:
                raise RuntimeError("Cannot initialize HSD anchor embedding because token embeddings are missing.")
            if init_from in {"placeholder", "bos"}:
                if init_from == "bos":
                    bos_token_id = getattr(self.config, "anchor_bos_token_id", None)
                    if bos_token_id is None:
                        bos_token_id = getattr(self.config, "bos_token_id", None)
                    if bos_token_id is None:
                        raise ValueError(
                            "anchor_embedding_init_from='bos' requires config.anchor_bos_token_id "
                            "or config.bos_token_id."
                        )
                    token_id = int(bos_token_id)
                else:
                    token_id = int(getattr(self.config, "anchor_placeholder_token_id", 0) or 0)
                if token_id < 0 or token_id >= weight.shape[0]:
                    raise ValueError(
                        f"anchor {init_from} token id {token_id} is outside the embedding table "
                        f"with vocab size {weight.shape[0]}."
                    )
                vector = weight[token_id].detach()
            elif init_from == "vocab_mean":
                vector = weight.detach().float().mean(dim=0)
            else:
                raise ValueError("anchor_embedding_init_from must be one of random, placeholder, bos, or vocab_mean.")
            vector = vector.to(device=self.anchor_embedding.weight.device, dtype=self.anchor_embedding.weight.dtype)
            self.anchor_embedding.weight.copy_(vector.view(1, -1).expand_as(self.anchor_embedding.weight))

        init_scale = getattr(self.config, "anchor_embedding_init_scale", None)
        if init_scale is not None:
            self.anchor_embedding.weight.mul_(float(init_scale))

    def _apply_anchor_embedding(self, inputs_embeds: torch.Tensor, anchor_mask: torch.Tensor | None) -> torch.Tensor:
        if anchor_mask is None or not bool(anchor_mask.any().item()):
            return inputs_embeds
        anchor_mask = anchor_mask.to(device=inputs_embeds.device, dtype=torch.bool)
        anchor_embedding_type = str(getattr(self.config, "anchor_embedding_type", "learned"))
        if anchor_embedding_type == "placeholder":
            return inputs_embeds
        output = inputs_embeds.clone()
        if anchor_embedding_type == "learned":
            if not hasattr(self, "anchor_embedding"):
                raise ValueError("learned HSD anchor embeddings require anchor_token_num > 0.")
            anchor_ordinals = anchor_mask.long().cumsum(dim=1).sub(1)
            anchor_ids = anchor_ordinals[anchor_mask].remainder(int(self.config.anchor_token_num))
            replacements = self.anchor_embedding(anchor_ids.to(device=self.anchor_embedding.weight.device))
            output[anchor_mask] = replacements.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            return output
        if anchor_embedding_type == "chunk_end":
            anchors = _anchor_text_indices(anchor_mask).to(inputs_embeds.device)
            anchor_embeds = inputs_embeds.gather(
                1,
                anchors.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1]),
            )
            output[anchor_mask] = anchor_embeds.reshape(-1, inputs_embeds.shape[-1])
            return output
        raise ValueError(f"Unsupported anchor_embedding_type: {anchor_embedding_type}")


def _single_token_cached_input(input_ids, past_text_lens, attention_mask, next_sequence_length,
                               is_first_iteration):
    """Validate a cached suffix before selecting its sole new text token.

    Full-prefix callers use text lengths (not expanded anchor/KV lengths).
    Transformers can also pass cropped generated history with an explicit
    one-token selector, e.g. after an inputs_embeds prompt.
    """
    if next_sequence_length is not None and (
        not isinstance(next_sequence_length, int) or isinstance(next_sequence_length, bool)
        or next_sequence_length != 1
    ):
        # HF prefill derives this selector from get_seq_length(), which can
        # include anchors in all-linear caches. Use text counts in that case.
        expanded_prefill_offset = (is_first_iteration and isinstance(next_sequence_length, int)
                                   and not isinstance(next_sequence_length, bool)
                                   and next_sequence_length <= 0)
        if not expanded_prefill_offset:
            raise NotImplementedError("Safin-1 HSD V1 supports one new decode token per step.")
    if input_ids.shape[1] == 1:
        return input_ids
    full_mask = (torch.is_tensor(attention_mask) and attention_mask.ndim == 2
                 and attention_mask.shape[0] == input_ids.shape[0]
                 and attention_mask.shape[1] >= input_ids.shape[1])
    if full_mask:
        new_lengths = attention_mask.to(device=input_ids.device).ne(0).sum(-1) - past_text_lens
    else:
        new_lengths = input_ids.shape[1] - past_text_lens
    if not bool((new_lengths == 1).all().item()):
        selected_history = (not full_mask and next_sequence_length == 1
                            and bool((new_lengths < 0).all().item()))
        if not selected_history:
            raise NotImplementedError(
                "Safin-1 HSD V1 supports one new decode token per step; "
                "pass the full text prefix with one new token, or a single token. "
                f"Uncached text counts: {new_lengths.tolist()}."
            )
    return input_ids[:, -1:]


class Safin1GenerationMixin:
    """Shared anchor-aware generation; super() dispatches to the upstream model API."""

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, generation_mode, batch_size, max_cache_length):
        result = super()._prepare_cache_for_generation(
            generation_config,
            model_kwargs,
            generation_mode,
            batch_size,
            max_cache_length,
        )
        if _anchor_enabled(self.config.text_config):
            cache = model_kwargs.get("past_key_values")
            if cache is not None:
                model_kwargs["past_key_values"] = _patch_hsd_cache_get_seq_length(cache)
        return result

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        logits_to_keep=None,
        cache_position=None,
        **kwargs,
    ):
        if _anchor_enabled(self.config.text_config) and past_key_values is not None and not _cache_has_previous_state(past_key_values):
            model_kwargs = dict(kwargs)
            model_kwargs.pop("next_sequence_length", None)
            mm_token_type_ids = model_kwargs.pop("mm_token_type_ids", None)
            model_inputs = {
                "input_ids": input_ids.contiguous() if input_ids is not None and inputs_embeds is None else None,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": _patch_hsd_cache_get_seq_length(past_key_values),
                "use_cache": use_cache,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "mm_token_type_ids": mm_token_type_ids,
            }
            if cache_position is not None:
                model_inputs["cache_position"] = cache_position
            if logits_to_keep is not None:
                model_inputs["logits_to_keep"] = logits_to_keep
            model_inputs.update(model_kwargs)
            return model_inputs

        if not _anchor_enabled(self.config.text_config) or past_key_values is None or not _cache_has_previous_state(past_key_values):
            model_inputs = super().prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                use_cache=use_cache,
                is_first_iteration=is_first_iteration,
                logits_to_keep=logits_to_keep,
                cache_position=cache_position,
                **kwargs,
            )
            return model_inputs

        if input_ids is None:
            raise ValueError("Safin-1 HSD cached generation requires input_ids during decode.")
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("Safin-1 HSD cached decode requires a nonempty [batch, text] input_ids tensor.")
        batch_size = input_ids.shape[0]
        past_text_lens = _cache_text_lengths(past_key_values, batch_size, input_ids.device)
        past_expanded_lens = _cache_expanded_lengths(past_key_values, batch_size, input_ids.device)
        input_ids = _single_token_cached_input(
            input_ids, past_text_lens, attention_mask,
            kwargs.pop("next_sequence_length", None), is_first_iteration,
        )

        anchor_token_num = int(self.config.text_config.anchor_token_num)
        chunk_size = int(self.config.text_config.anchor_chunk_size)
        append_anchor = (past_text_lens + 1).remainder(chunk_size) == 0
        current_len = 1 + (anchor_token_num if bool(append_anchor.any().item()) else 0)
        if current_len > 1:
            anchor_ids = torch.full(
                (batch_size, anchor_token_num),
                int(getattr(self.config.text_config, "anchor_placeholder_token_id", 0)),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            input_ids = torch.cat([input_ids, anchor_ids], dim=1)
        anchor_mask = torch.zeros(batch_size, input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
        anchor_valid_mask = torch.zeros_like(anchor_mask)
        persistent_memory_mask = torch.zeros_like(anchor_mask)
        text_valid_mask = torch.zeros_like(anchor_mask)
        hsd_valid_mask = torch.zeros_like(anchor_mask)
        text_valid_mask[:, 0] = True
        hsd_valid_mask[:, 0] = True
        if current_len > 1:
            anchor_mask[:, 1:] = True
            anchor_valid_mask[:, 1:] = append_anchor.view(batch_size, 1)
            hsd_valid_mask[:, 1:] = anchor_valid_mask[:, 1:]

        current_position_ids = self._build_decode_position_ids(
            batch_size=batch_size,
            past_text_len=past_text_lens,
            current_len=input_ids.shape[1],
            append_anchor=append_anchor,
            device=input_ids.device,
        )
        past_expanded_len = int(past_expanded_lens.max().item()) if past_expanded_lens.numel() else _cache_expanded_length(past_key_values)
        if cache_position is not None:
            cache_position = torch.arange(
                past_expanded_len,
                past_expanded_len + input_ids.shape[1],
                dtype=cache_position.dtype,
                device=input_ids.device,
            )
        decode_attention_mask = attention_mask
        if torch.is_tensor(decode_attention_mask) and bool(decode_attention_mask.all().item()):
            decode_attention_mask = None

        model_inputs = {
            "input_ids": input_ids.contiguous(),
            "inputs_embeds": None,
            "attention_mask": decode_attention_mask,
            "position_ids": current_position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "anchor_mask": anchor_mask,
            "hsd_valid_mask": hsd_valid_mask,
            "text_valid_mask": text_valid_mask,
            "anchor_valid_mask": anchor_valid_mask,
            "persistent_memory_mask": persistent_memory_mask,
            "pixel_values": None,
            "pixel_values_videos": None,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "mm_token_type_ids": None,
        }
        if cache_position is not None:
            model_inputs["cache_position"] = cache_position
        if logits_to_keep is not None:
            model_inputs["logits_to_keep"] = logits_to_keep
        model_inputs.update(kwargs)
        return model_inputs

    def _build_decode_position_ids(
        self,
        batch_size: int,
        past_text_len: int | torch.Tensor,
        current_len: int,
        append_anchor: bool | torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if torch.is_tensor(past_text_len):
            past_text_tensor = past_text_len.to(device=device, dtype=torch.long).reshape(batch_size)
        else:
            past_text_tensor = torch.full((batch_size,), int(past_text_len), dtype=torch.long, device=device)
        if torch.is_tensor(append_anchor):
            append_anchor_tensor = append_anchor.to(device=device, dtype=torch.bool).reshape(batch_size)
        else:
            append_anchor_tensor = torch.full((batch_size,), bool(append_anchor), dtype=torch.bool, device=device)
        text_pos = past_text_tensor.view(batch_size, 1).expand(batch_size, current_len).clone()
        if current_len > 1:
            anchor_position_type = str(getattr(self.config.text_config, "anchor_position_ids_type", "zeros"))
            if anchor_position_type == "chunk_end":
                text_pos[:, 1:] = past_text_tensor.view(batch_size, 1)
            else:
                text_pos[:, 1:] = 0
            text_pos[:, 1:] = torch.where(
                append_anchor_tensor.view(batch_size, 1),
                text_pos[:, 1:],
                torch.zeros_like(text_pos[:, 1:]),
            )
        if self.model.rope_deltas is not None:
            delta = self.model.rope_deltas.to(device=device, dtype=torch.long)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            if current_len > 1 and str(
                getattr(self.config.text_config, "anchor_position_ids_type", "zeros")
            ) == "zeros":
                text_pos[:, :1] = text_pos[:, :1] + delta
            else:
                text_pos = text_pos + delta
        return text_pos.view(1, batch_size, current_len).expand(4, batch_size, current_len)

    def _reorder_cache(self, past_key_values, beam_idx):
        past_key_values = _patch_hsd_cache_get_seq_length(past_key_values)
        if getattr(past_key_values, "_march_batch_ops_patched", False):
            past_key_values.reorder_cache(beam_idx)
            return past_key_values
        reorder_cache = getattr(super(), "_reorder_cache", None)
        if callable(reorder_cache):
            reordered = reorder_cache(past_key_values, beam_idx)
            if reordered is not None:
                past_key_values = reordered
        elif hasattr(past_key_values, "reorder_cache"):
            past_key_values.reorder_cache(beam_idx)
        return _reorder_hsd_sidecar_cache(past_key_values, beam_idx)
