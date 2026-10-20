"""Model wrappers that replace Qwen3.5-MoE linear attention with historical GDN."""

from __future__ import annotations

import inspect
from types import MethodType

import torch
from torch import nn
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen35_moe_modeling
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeModel,
    Qwen3_5MoeModelOutputWithPast,
    Qwen3_5MoePreTrainedModel,
    Qwen3_5MoeTextModel,
    Qwen3_5MoeVisionModel,
)

from .gdn_cache import Safin1GDNCacheMixin as _remote_gdn_cache  # noqa: F401
from .gdn_recurrence import Safin1GDNRecurrenceMixin as _remote_gdn_recurrence  # noqa: F401
from .configuration_common import Safin1RuntimeFields as _remote_runtime_fields  # noqa: F401
from .config_utils import atomic_write_bytes as _remote_atomic_write  # noqa: F401

# Transformers local remote-code caching copies direct relative imports.
# Declare transitive dependencies here so a fresh cache is self-contained.
from .fused_history_read import fused_prefill_top4_history_read as _remote_history_read  # noqa: F401
from .fused_march_segments import FLA_CHUNK_SIZE as _remote_fla_chunk_size  # noqa: F401
from .configuration_safin1_moe import Safin1MoeConfig, Safin1MoeTextConfig
from .march_gdn import (
    Safin1MoeGatedDeltaNet,
    collect_persistent_memory_metrics,
    load_persistent_memory_artifact,
    save_persistent_memory_artifact,
    trace_hsd_internal,
)
from .ops import route_softmax_weights as _route_softmax_weights  # noqa: F401
from .persistent_memory import save_persistent_memory_artifact as _remote_persistent_memory  # noqa: F401
from .runtime_validation import validate_safin1_runtime as _remote_runtime_validation  # noqa: F401


from .cache_utils import (
    _RowFullAttentionCache,
    _cache_expanded_length,
    _cache_expanded_lengths,
    _cache_has_previous_state,
    _cache_lengths,
    _cache_text_length,
    _cache_text_lengths,
    _full_attention_layer_idx,
    _length_value_to_int,
    _merge_row_full_attention_caches,
    _new_qwen35_cache,
    _patch_hsd_cache_get_seq_length,
    _reorder_batch_tensor,
    _reorder_hsd_sidecar_cache,
    _set_root_cache_lengths,
    _slice_row_full_attention_cache,
)
from .anchor_layout import _anchor_enabled, _anchor_has_tokens, _anchor_text_indices, _build_anchor_layout, _build_anchor_layout_masks, _expand_2d_with_anchor, _expand_embeds_with_anchor, _expand_position_ids, _filter_anchor_hidden, _filter_batch_sequence_to_text, _filter_position_embeddings_to_text, _full_attention_mask_source, _insert_anchor_inputs, _insert_anchor_inputs_with_layout, _insert_left_padded_anchor_inputs, _is_left_padding_mask, _persistent_memory_enabled, _scatter_text_hidden, _text_positions_from_anchor_mask
from .modeling_common import Safin1AnchorEmbeddingMixin, Safin1GenerationMixin, _as_tuple_output, _config_return_dict, _create_qwen_causal_mask, _decoder_layer_accepts_cache_position, _filter_router_logits, _historical_hsd_decoder_layer_forward, _is_topk_router, _record_router_logits, _run_full_attention_padded_fallback, _run_layer_mlp_delta, _update_linear_attn_mask


def replace_linear_attention_layers(language_model: Qwen3_5MoeTextModel, config: Safin1MoeTextConfig) -> None:
    for layer_idx, layer in enumerate(language_model.layers):
        layer_type = config.layer_types[layer_idx]
        layer._historical_hsd_config = config
        layer._historical_hsd_layer_idx = int(layer_idx)
        if config.layer_types[layer_idx] == "linear_attention":
            layer.linear_attn = Safin1MoeGatedDeltaNet(config, layer_idx)
            if _anchor_enabled(config) and not getattr(layer, "_historical_hsd_linear_patched", False):
                layer.forward = MethodType(_historical_hsd_decoder_layer_forward, layer)
                layer._historical_hsd_linear_patched = True
                layer._historical_hsd_accepts_anchor_mask = True
        elif layer_type == "full_attention" and _anchor_enabled(config):
            if not getattr(layer, "_historical_hsd_full_attention_patched", False):
                layer.forward = MethodType(_historical_hsd_decoder_layer_forward, layer)
                layer._historical_hsd_full_attention_patched = True
                layer._historical_hsd_accepts_anchor_mask = True


class Safin1MoeTextModel(Safin1AnchorEmbeddingMixin, Qwen3_5MoeTextModel):
    """Official text model with optional HSD anchor-token support."""

    config: Safin1MoeTextConfig
    config_class = Safin1MoeTextConfig

    def __init__(self, config: Safin1MoeTextConfig):
        super().__init__(config)
        if _anchor_enabled(config) and str(getattr(config, "anchor_embedding_type", "learned")) == "learned":
            self.anchor_embedding = nn.Embedding(int(config.anchor_token_num), config.hidden_size)
            with torch.no_grad():
                self._initialize_anchor_embedding_weight()
        if getattr(config, "hist_replace_linear_attn", True):
            replace_linear_attention_layers(self, config)


    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        anchor_mask: torch.Tensor | None = None,
        hsd_valid_mask: torch.Tensor | None = None,
        text_valid_mask: torch.Tensor | None = None,
        anchor_valid_mask: torch.Tensor | None = None,
        persistent_memory_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> Qwen3_5MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_hidden_states = kwargs.pop("output_hidden_states", None)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else getattr(self.config, "output_hidden_states", False)
        )
        output_hidden_states = bool(output_hidden_states)
        output_attentions = bool(kwargs.pop("output_attentions", False))
        if output_attentions:
            raise NotImplementedError("Safin-1 HSD does not expose attentions for linear layers.")
        output_router_logits = kwargs.pop("output_router_logits", None)
        output_router_logits = (
            output_router_logits if output_router_logits is not None else getattr(self.config, "output_router_logits", False)
        )
        output_router_logits = bool(output_router_logits)
        return_dict = kwargs.pop("return_dict", None)
        return_dict = return_dict if return_dict is not None else _config_return_dict(self.config)
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        cache_position = kwargs.pop("cache_position", cache_position)

        force_layout = bool(use_cache and _anchor_enabled(self.config))
        if anchor_mask is None and _anchor_enabled(self.config) and not _cache_has_previous_state(past_key_values):
            layout = _insert_anchor_inputs_with_layout(
                config=self.config,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                force_layout=force_layout,
            )
            input_ids = layout["input_ids"]
            inputs_embeds = layout["inputs_embeds"]
            attention_mask = layout["attention_mask"]
            position_ids = layout["position_ids"]
            anchor_mask = layout["anchor_mask"]
            hsd_valid_mask = layout["hsd_valid_mask"]
            text_valid_mask = layout["text_valid_mask"]
            anchor_valid_mask = layout["anchor_valid_mask"]
            persistent_memory_mask = layout["persistent_memory_mask"]
            expected_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
            if cache_position is not None and int(cache_position.numel()) != int(expected_len):
                cache_position = None

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if anchor_mask is None and use_cache and _anchor_enabled(self.config) and _cache_has_previous_state(past_key_values):
            anchor_mask = torch.zeros(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
            hsd_valid_mask = torch.ones_like(anchor_mask, dtype=torch.bool)
            text_valid_mask = ~anchor_mask
            anchor_valid_mask = anchor_mask
            persistent_memory_mask = torch.zeros_like(anchor_mask)
        if anchor_mask is not None:
            anchor_mask = anchor_mask.to(device=inputs_embeds.device, dtype=torch.bool)
            if hsd_valid_mask is not None:
                hsd_valid_mask = hsd_valid_mask.to(device=inputs_embeds.device, dtype=torch.bool)
            if text_valid_mask is not None:
                text_valid_mask = text_valid_mask.to(device=inputs_embeds.device, dtype=torch.bool)
            if anchor_valid_mask is not None:
                anchor_valid_mask = anchor_valid_mask.to(device=inputs_embeds.device, dtype=torch.bool)
            if persistent_memory_mask is not None:
                persistent_memory_mask = persistent_memory_mask.to(device=inputs_embeds.device, dtype=torch.bool)
            inputs_embeds = self._apply_anchor_embedding(inputs_embeds, anchor_mask)

        if use_cache and past_key_values is None:
            past_key_values = _new_qwen35_cache(self.config)
        elif use_cache:
            past_key_values = _patch_hsd_cache_get_seq_length(past_key_values)
        batch_size = int(inputs_embeds.shape[0])
        past_text_before_per_row = (
            _cache_text_lengths(past_key_values, batch_size, inputs_embeds.device)
            if use_cache and past_key_values is not None
            else torch.zeros(batch_size, dtype=torch.long, device=inputs_embeds.device)
        )
        past_expanded_before_per_row = (
            _cache_expanded_lengths(past_key_values, batch_size, inputs_embeds.device)
            if use_cache and past_key_values is not None
            else torch.zeros(batch_size, dtype=torch.long, device=inputs_embeds.device)
        )
        past_expanded_before = int(past_expanded_before_per_row.max().item()) if past_expanded_before_per_row.numel() else 0

        if cache_position is None:
            cache_position = torch.arange(
                past_expanded_before,
                past_expanded_before + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids[1:]
        else:
            text_position_ids = None
            rope_position_ids = position_ids

        active_layer_types = self.config.layer_types[: self.config.num_hidden_layers]
        has_full_attention = any(layer_type != "linear_attention" for layer_type in active_layer_types)
        causal_mask = None
        full_attention_anchor_tokens = bool(
            False
        )
        full_attention_inputs_embeds = (
            (_filter_anchor_hidden(inputs_embeds, anchor_mask))
        )
        full_attention_position_ids = (
            (_filter_batch_sequence_to_text(text_position_ids, anchor_mask))
        )
        full_attention_mask = (
            (_full_attention_mask_source(attention_mask, anchor_mask))
        )
        full_attention_cache_position = cache_position
        if (not False and cache_position is not None
                and int(cache_position.numel()) != int(full_attention_inputs_embeds.shape[1])):
            if torch.is_tensor(full_attention_mask) and full_attention_mask.ndim == 2:
                full_start = int(full_attention_mask.shape[-1]) - int(full_attention_inputs_embeds.shape[1])
                full_attention_cache_position = torch.arange(
                    full_start,
                    full_start + int(full_attention_inputs_embeds.shape[1]),
                    dtype=cache_position.dtype,
                    device=cache_position.device,
                )
            elif _anchor_has_tokens(anchor_mask):
                try:
                    text_positions = _text_positions_from_anchor_mask(anchor_mask, cache_position.device)
                    full_attention_cache_position = cache_position.index_select(0, text_positions)
                except NotImplementedError:
                    full_start = int(past_text_before_per_row.max().item()) if past_text_before_per_row.numel() else 0
                    full_attention_cache_position = torch.arange(
                        full_start,
                        full_start + int(full_attention_inputs_embeds.shape[1]),
                        dtype=cache_position.dtype,
                        device=cache_position.device,
                    )
        if has_full_attention:
            causal_mask = _create_qwen_causal_mask(
                config=self.config,
                inputs_embeds=full_attention_inputs_embeds,
                attention_mask=full_attention_mask,
                cache_position=full_attention_cache_position,
                past_key_values=past_key_values,
                position_ids=full_attention_position_ids,
            )
        linear_attention_mask_source = attention_mask
        if (
            torch.is_tensor(linear_attention_mask_source)
            and linear_attention_mask_source.ndim == 2
            and linear_attention_mask_source.shape[-1] != inputs_embeds.shape[1]
        ):
            linear_attention_mask_source = None
        linear_attn_mask = _update_linear_attn_mask(self, linear_attention_mask_source, cache_position, past_key_values)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, rope_position_ids)
        all_hidden_states = () if output_hidden_states else None
        router_logits, router_handles = _record_router_logits(self, output_router_logits, anchor_mask)
        try:
            for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
                if output_hidden_states:
                    all_hidden_states += (_filter_anchor_hidden(hidden_states, anchor_mask),)
                is_linear_attention = self.config.layer_types[i] == "linear_attention"
                layer_mask = linear_attn_mask if is_linear_attention else causal_mask
                layer_position_ids = text_position_ids if is_linear_attention else full_attention_position_ids
                layer_cache_position = cache_position if is_linear_attention else full_attention_cache_position
                layer_kwargs = dict(
                    position_embeddings=position_embeddings,
                    attention_mask=layer_mask,
                    position_ids=layer_position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    **kwargs,
                )
                if is_linear_attention or getattr(decoder_layer, "_historical_hsd_accepts_anchor_mask", False):
                    layer_kwargs["anchor_mask"] = anchor_mask
                    layer_kwargs["hsd_valid_mask"] = hsd_valid_mask
                    layer_kwargs["text_valid_mask"] = text_valid_mask
                    layer_kwargs["anchor_valid_mask"] = anchor_valid_mask
                    layer_kwargs["persistent_memory_mask"] = persistent_memory_mask
                if _decoder_layer_accepts_cache_position(decoder_layer):
                    layer_kwargs["cache_position"] = layer_cache_position
                hidden_states = decoder_layer(hidden_states, **layer_kwargs)
        finally:
            for handle in router_handles:
                handle.remove()

        final_trace_layer = int(self.config.num_hidden_layers)
        trace_hsd_internal(
            final_trace_layer,
            "final_norm_input",
            _filter_anchor_hidden(hidden_states, anchor_mask),
        )
        hidden_states = self.norm(hidden_states)
        hidden_states = _filter_anchor_hidden(hidden_states, anchor_mask)
        trace_hsd_internal(final_trace_layer, "final_norm_out", hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if use_cache and past_key_values is not None and anchor_mask is not None:
            if text_valid_mask is None:
                current_text_per_row = (~anchor_mask).long().sum(dim=1)
            else:
                current_text_per_row = text_valid_mask.long().sum(dim=1)
            if hsd_valid_mask is None:
                current_expanded_per_row = torch.full_like(current_text_per_row, int(inputs_embeds.shape[1]))
            else:
                current_expanded_per_row = hsd_valid_mask.long().sum(dim=1)
            _set_root_cache_lengths(
                past_key_values,
                text_lengths=past_text_before_per_row + current_text_per_row.to(past_text_before_per_row.device),
                expanded_lengths=past_expanded_before_per_row
                + current_expanded_per_row.to(past_expanded_before_per_row.device),
            )

        outputs = Qwen3_5MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=None,
            router_logits=tuple(router_logits) if output_router_logits else None,
        )
        if not return_dict:
            return _as_tuple_output(outputs)
        return outputs


class Safin1MoeModel(Qwen3_5MoeModel):
    """Top-level multimodal model with the historical text backbone."""

    config: Safin1MoeConfig
    config_class = Safin1MoeConfig

    def __init__(self, config: Safin1MoeConfig):
        Qwen3_5MoePreTrainedModel.__init__(self, config)
        self.visual = Qwen3_5MoeVisionModel._from_config(config.vision_config)
        self.language_model = Safin1MoeTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        if getattr(module, "_hsd_zero_init", False):
            nn.init.zeros_(module.weight)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        anchor_mask: torch.Tensor | None = None,
        hsd_valid_mask: torch.Tensor | None = None,
        text_valid_mask: torch.Tensor | None = None,
        anchor_valid_mask: torch.Tensor | None = None,
        persistent_memory_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple | Qwen3_5MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        return_dict = kwargs.pop("return_dict", None)
        return_dict = return_dict if return_dict is not None else _config_return_dict(self.config)
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, image_grid_thw, return_dict=True, **kwargs)
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        if pixel_values_videos is not None:
            video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True, **kwargs)
            video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        use_cache = kwargs.get("use_cache", None)
        force_layout = bool(use_cache and _anchor_enabled(self.config.text_config))
        if anchor_mask is None and _anchor_enabled(self.config.text_config) and not _cache_has_previous_state(past_key_values):
            layout = _insert_anchor_inputs_with_layout(
                config=self.config.text_config,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                mm_token_type_ids=mm_token_type_ids,
                force_layout=force_layout,
            )
            input_ids = layout["input_ids"]
            inputs_embeds = layout["inputs_embeds"]
            attention_mask = layout["attention_mask"]
            position_ids = layout["position_ids"]
            mm_token_type_ids = layout["mm_token_type_ids"]
            anchor_mask = layout["anchor_mask"]
            hsd_valid_mask = layout["hsd_valid_mask"]
            text_valid_mask = layout["text_valid_mask"]
            anchor_valid_mask = layout["anchor_valid_mask"]
            persistent_memory_mask = layout["persistent_memory_mask"]
            expected_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
            if cache_position is not None and int(cache_position.numel()) != int(expected_len):
                cache_position = None

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            anchor_mask=anchor_mask,
            hsd_valid_mask=hsd_valid_mask,
            text_valid_mask=text_valid_mask,
            anchor_valid_mask=anchor_valid_mask,
            persistent_memory_mask=persistent_memory_mask,
            return_dict=True,
            **kwargs,
        )
        output = Qwen3_5MoeModelOutputWithPast(**outputs, rope_deltas=self.rope_deltas)
        if not return_dict:
            return _as_tuple_output(output)
        return output


class Safin1MoeForConditionalGeneration(Safin1GenerationMixin, Qwen3_5MoeForConditionalGeneration):
    """Causal generation wrapper preserving the official Qwen3.5-MoE API."""

    config: Safin1MoeConfig
    config_class = Safin1MoeConfig

    def __init__(self, config: Safin1MoeConfig):
        Qwen3_5MoePreTrainedModel.__init__(self, config)
        self.model = Safin1MoeModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        if getattr(module, "_hsd_zero_init", False):
            nn.init.zeros_(module.weight)
        state = getattr(module, "persistent_memory_state", None)
        if isinstance(state, nn.Parameter):
            nn.init.zeros_(state)

    def save_persistent_memory(self, output_dir):
        return save_persistent_memory_artifact(self, output_dir)

    def load_persistent_memory(self, artifact, *, strict: bool = True):
        return load_persistent_memory_artifact(self, artifact, strict=strict)

    def get_persistent_memory_metrics(
        self, *, cumulative: bool = False, reset: bool = False, cpu: bool = True
    ):
        return collect_persistent_memory_metrics(
            self.model,
            cumulative=cumulative,
            reset=reset,
            cpu=cpu,
        )


__all__ = [
    "Safin1MoeTextModel",
    "Safin1MoeModel",
    "Safin1MoeForConditionalGeneration",
    "replace_linear_attention_layers",
]
