"""Shared Safin-1 cache utils for dense and MoE remote code."""

from __future__ import annotations


from types import MethodType
import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen35_moe_modeling


def _new_qwen35_cache(config):
    cache_cls = getattr(qwen35_moe_modeling, "Qwen3_5MoeDynamicCache", DynamicCache)
    try:
        cache = cache_cls(config=config)
    except TypeError:
        cache = cache_cls()
    return _patch_hsd_cache_get_seq_length(cache)


def _patch_hsd_cache_get_seq_length(cache):
    if cache is None:
        return cache
    _patch_hsd_cache_batch_ops(cache)
    if getattr(cache, "_historical_hsd_get_seq_length_patched", False):
        return cache
    original_get_seq_length = getattr(cache, "get_seq_length", None)
    if not callable(original_get_seq_length):
        return cache

    def get_seq_length(self, *args, **kwargs):
        try:
            return original_get_seq_length(*args, **kwargs)
        except ValueError as exc:
            message = str(exc)
            if "get_seq_length" not in message or "LinearAttention" not in message:
                raise
            value = getattr(self, "hsd_seen_expanded_tokens", None)
            if value is not None:
                return _length_value_to_int(value)
            value = getattr(self, "hsd_seen_text_tokens", None)
            if value is not None:
                return _length_value_to_int(value)
            return 0

    cache.get_seq_length = MethodType(get_seq_length, cache)
    cache._historical_hsd_get_seq_length_patched = True
    return cache


def _cache_has_previous_state(past_key_values) -> bool:
    if past_key_values is None:
        return False
    if _length_value_to_int(getattr(past_key_values, "hsd_seen_expanded_tokens", 0)) > 0:
        return True
    if _length_value_to_int(getattr(past_key_values, "hsd_seen_text_tokens", 0)) > 0:
        return True
    if _cache_text_length(past_key_values) > 0:
        return True
    has_previous_state = getattr(past_key_values, "has_previous_state", None)
    if callable(has_previous_state):
        try:
            return bool(has_previous_state())
        except TypeError:
            return bool(has_previous_state(0))
    if isinstance(has_previous_state, bool):
        return has_previous_state
    try:
        return int(past_key_values.get_seq_length()) > 0
    except Exception:
        return False


def _cache_text_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    value = getattr(past_key_values, "hsd_seen_text_tokens", None)
    if value is not None:
        return _length_value_to_int(value)
    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        for layer in layers:
            value = getattr(layer, "hsd_seen_text_tokens", None)
            if value is not None:
                return _length_value_to_int(value)
    layers = getattr(past_key_values, "_hsd_layer_states", None)
    if layers is not None:
        for layer in layers:
            value = getattr(layer, "hsd_seen_text_tokens", None)
            if value is not None:
                return _length_value_to_int(value)
    try:
        return int(past_key_values.get_seq_length())
    except Exception:
        return 0


def _cache_expanded_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    value = getattr(past_key_values, "hsd_seen_expanded_tokens", None)
    if value is not None:
        return _length_value_to_int(value)
    try:
        value = int(past_key_values.get_seq_length())
        if value > 0:
            return value
    except Exception:
        pass
    text_len = _cache_text_length(past_key_values)
    return text_len


def _length_value_to_int(value) -> int:
    if value is None:
        return 0
    if torch.is_tensor(value):
        if value.numel() == 0:
            return 0
        return int(value.detach().max().item())
    return int(value or 0)


def _cache_lengths(
    past_key_values,
    *,
    base_attr: str,
    per_row_attr: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if past_key_values is None:
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    for source in (past_key_values,):
        value = getattr(source, per_row_attr, None)
        if value is not None:
            tensor = torch.as_tensor(value, dtype=torch.long, device=device).reshape(-1)
            if tensor.numel() == batch_size:
                return tensor
            if tensor.numel() == 1:
                return tensor.expand(batch_size)
    for attr_name in ("layers", "_hsd_layer_states"):
        layers = getattr(past_key_values, attr_name, None)
        if layers is None:
            continue
        for layer in layers:
            value = getattr(layer, per_row_attr, None)
            if value is not None:
                tensor = torch.as_tensor(value, dtype=torch.long, device=device).reshape(-1)
                if tensor.numel() == batch_size:
                    return tensor
                if tensor.numel() == 1:
                    return tensor.expand(batch_size)
            value = getattr(layer, base_attr, None)
            if value is not None:
                scalar = _length_value_to_int(value)
                return torch.full((batch_size,), scalar, dtype=torch.long, device=device)
    value = getattr(past_key_values, base_attr, None)
    if value is not None:
        scalar = _length_value_to_int(value)
        return torch.full((batch_size,), scalar, dtype=torch.long, device=device)
    scalar = _cache_text_length(past_key_values) if base_attr == "hsd_seen_text_tokens" else _cache_expanded_length(past_key_values)
    return torch.full((batch_size,), scalar, dtype=torch.long, device=device)


def _cache_text_lengths(past_key_values, batch_size: int, device: torch.device) -> torch.Tensor:
    return _cache_lengths(
        past_key_values,
        base_attr="hsd_seen_text_tokens",
        per_row_attr="hsd_seen_text_tokens_per_row",
        batch_size=batch_size,
        device=device,
    )


def _cache_expanded_lengths(past_key_values, batch_size: int, device: torch.device) -> torch.Tensor:
    return _cache_lengths(
        past_key_values,
        base_attr="hsd_seen_expanded_tokens",
        per_row_attr="hsd_seen_expanded_tokens_per_row",
        batch_size=batch_size,
        device=device,
    )


def _set_root_cache_lengths(
    past_key_values,
    *,
    text_lengths: torch.Tensor,
    expanded_lengths: torch.Tensor,
) -> None:
    if past_key_values is None:
        return
    text_cpu = text_lengths.detach().to(dtype=torch.long)
    expanded_cpu = expanded_lengths.detach().to(dtype=torch.long)
    past_key_values.hsd_seen_text_tokens_per_row = text_cpu
    past_key_values.hsd_seen_expanded_tokens_per_row = expanded_cpu
    past_key_values.hsd_seen_text_tokens = _length_value_to_int(text_cpu)
    past_key_values.hsd_seen_expanded_tokens = _length_value_to_int(expanded_cpu)


class _RowFullAttentionCache:
    def __init__(self, layer_idx: int, keys: torch.Tensor | None = None, values: torch.Tensor | None = None):
        self.layer_idx = int(layer_idx)
        self.keys = keys
        self.values = values

    def get_seq_length(self, layer_idx: int = 0) -> int:
        # This adapter represents one compacted row of one full-attention layer.
        return int(self.keys.shape[-2]) if torch.is_tensor(self.keys) else 0

    def get_mask_sizes(self, query_length: int | torch.Tensor, layer_idx: int = 0) -> tuple[int, int]:
        length = int(query_length.numel()) if torch.is_tensor(query_length) else int(query_length)
        return self.get_seq_length() + length, 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args, **kwargs):
        if int(layer_idx) != self.layer_idx:
            raise ValueError(f"row full-attention cache received layer {layer_idx}, expected {self.layer_idx}.")
        if self.keys is None or self.values is None or self.keys.numel() == 0:
            self.keys = key_states
            self.values = value_states
        else:
            self.keys = torch.cat([self.keys.to(key_states.device), key_states], dim=-2)
            self.values = torch.cat([self.values.to(value_states.device), value_states], dim=-2)
        return self.keys, self.values


def _full_attention_layer_idx(layer: nn.Module) -> int:
    attn = getattr(layer, "self_attn", None)
    return int(getattr(attn, "layer_idx", getattr(layer, "_historical_hsd_layer_idx", getattr(layer, "layer_idx", 0))))


def _slice_row_full_attention_cache(past_key_values, layer_idx: int, batch_idx: int, batch_size: int) -> _RowFullAttentionCache | None:
    if past_key_values is None:
        return None
    layers = getattr(past_key_values, "layers", None)
    if layers is None or len(layers) <= layer_idx:
        return _RowFullAttentionCache(layer_idx)
    layer_cache = layers[layer_idx]
    keys = getattr(layer_cache, "keys", None)
    values = getattr(layer_cache, "values", None)
    if not torch.is_tensor(keys) or not torch.is_tensor(values) or keys.numel() == 0:
        return _RowFullAttentionCache(layer_idx)
    lengths = getattr(layer_cache, "hsd_text_cache_lengths_per_row", None)
    if torch.is_tensor(lengths) and lengths.numel() > batch_idx:
        row_len = int(lengths.reshape(-1)[batch_idx].item())
    else:
        row_len = int(keys.shape[-2])
    return _RowFullAttentionCache(
        layer_idx,
        keys[batch_idx : batch_idx + 1, :, :row_len].contiguous(),
        values[batch_idx : batch_idx + 1, :, :row_len].contiguous(),
    )


def _merge_row_full_attention_caches(past_key_values, layer_idx: int, row_caches: list[_RowFullAttentionCache | None]) -> None:
    if past_key_values is None:
        return
    layers = getattr(past_key_values, "layers", None)
    if layers is None or len(layers) <= layer_idx:
        return
    keyed = [cache for cache in row_caches if cache is not None and torch.is_tensor(cache.keys)]
    if not keyed:
        return
    sample_keys = keyed[0].keys
    sample_values = keyed[0].values
    max_len = max(int(cache.keys.shape[-2]) if cache is not None and torch.is_tensor(cache.keys) else 0 for cache in row_caches)
    batch_size = len(row_caches)
    merged_keys = sample_keys.new_zeros(batch_size, sample_keys.shape[1], max_len, sample_keys.shape[-1])
    merged_values = sample_values.new_zeros(batch_size, sample_values.shape[1], max_len, sample_values.shape[-1])
    lengths = torch.zeros(batch_size, dtype=torch.long, device=merged_keys.device)
    for batch_idx, cache in enumerate(row_caches):
        if cache is None or not torch.is_tensor(cache.keys) or not torch.is_tensor(cache.values):
            continue
        row_len = int(cache.keys.shape[-2])
        merged_keys[batch_idx : batch_idx + 1, :, :row_len] = cache.keys.to(merged_keys.dtype)
        merged_values[batch_idx : batch_idx + 1, :, :row_len] = cache.values.to(merged_values.dtype)
        lengths[batch_idx] = row_len
    layer_cache = layers[layer_idx]
    layer_cache.keys = merged_keys
    layer_cache.values = merged_values
    layer_cache.is_initialized = True
    layer_cache.dtype = merged_keys.dtype
    layer_cache.device = merged_keys.device
    layer_cache.hsd_text_cache_lengths_per_row = lengths


def _reorder_batch_tensor(value, beam_idx: torch.LongTensor, *, min_ndim=1):
    """Select rows of a declared batch tensor, including expansion and pruning."""
    if torch.is_tensor(value) and value.ndim >= min_ndim:
        return value.index_select(0, beam_idx.to(value.device))
    if isinstance(value, list):
        return [_reorder_batch_tensor(item, beam_idx, min_ndim=min_ndim) for item in value]
    if isinstance(value, tuple):
        return tuple(_reorder_batch_tensor(item, beam_idx, min_ndim=min_ndim) for item in value)
    return value


_PER_ROW_LENGTHS = ("hsd_seen_text_tokens_per_row", "hsd_seen_expanded_tokens_per_row",
                    "hsd_text_cache_lengths_per_row")
_ANCHOR_BATCH_FIELDS = ("hsd_anchor_states", "hsd_anchor_keys", "hsd_anchor_boundaries",
                       "hsd_anchor_valid_mask", "hsd_anchor_persistent_memory_mask")


def _hsd_cache_objects(cache):
    seen = set()
    for obj in [cache, *(getattr(cache, "_hsd_layer_states", None) or []), *(getattr(cache, "layers", None) or [])]:
        if obj is not None and id(obj) not in seen:
            seen.add(id(obj))
            yield obj


def _hsd_cache_batch_size(cache):
    for obj in _hsd_cache_objects(cache):
        for name in (*_PER_ROW_LENGTHS, "hsd_anchor_states", "conv_states", "recurrent_states", "keys"):
            value = getattr(obj, name, None)
            if torch.is_tensor(value) and value.ndim > 0:
                return int(value.shape[0])
    for name in ("_hsd_conv_states", "_hsd_recurrent_states"):
        for value in getattr(cache, name, []) or []:
            if torch.is_tensor(value) and value.ndim > 0:
                return int(value.shape[0])
    return None


def _validate_cache_indices(cache, indices):
    if not torch.is_tensor(indices) or indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("cache row indices must be a one-dimensional integer tensor")
    batch_size = _hsd_cache_batch_size(cache)
    if indices.numel() and (int(indices.min()) < 0 or
                           (batch_size is not None and int(indices.max()) >= batch_size)):
        raise IndexError(f"cache row indices are out of range for batch size {batch_size}")


def _reorder_hsd_sidecar_cache(past_key_values, beam_idx: torch.LongTensor):
    """Reorder MARCH sidecars only; upstream owns native KV/conv/recurrent layers."""
    if past_key_values is None:
        return past_key_values
    _validate_cache_indices(past_key_values, beam_idx)
    for obj in _hsd_cache_objects(past_key_values):
        for name in _PER_ROW_LENGTHS:
            if hasattr(obj, name):
                setattr(obj, name, _reorder_batch_tensor(getattr(obj, name), beam_idx))
        for name in _ANCHOR_BATCH_FIELDS:
            if hasattr(obj, name):
                # One-dimensional boundaries/masks are shared across the batch.
                # Their length can coincide with B; that does not make them batch tensors.
                setattr(obj, name, _reorder_batch_tensor(getattr(obj, name), beam_idx, min_ndim=2))
        for base in ("hsd_seen_text_tokens", "hsd_seen_expanded_tokens"):
            lengths = getattr(obj, base + "_per_row", None)
            if torch.is_tensor(lengths):
                setattr(obj, base, _length_value_to_int(lengths))
    for name in ("_hsd_conv_states", "_hsd_recurrent_states"):
        states = getattr(past_key_values, name, None)
        if states is not None:
            setattr(past_key_values, name, _reorder_batch_tensor(states, beam_idx))
    return past_key_values


def _patch_hsd_cache_batch_ops(cache):
    """Give all HF cache row operations one native reorder and one sidecar reorder.

    LinearAttentionLayer lacks the upstream repeat/select methods. Implement
    those through its native reorder_cache, which also handles size changes.
    """
    if getattr(cache, "_march_batch_ops_patched", False):
        return cache
    original = getattr(cache, "reorder_cache", None)
    if not callable(original):
        return cache

    def reorder_cache(self, indices):
        _validate_cache_indices(self, indices)
        # Select sidecars before native layers change B, so validation sees old B.
        _reorder_hsd_sidecar_cache(self, indices)
        return original(indices)

    def batch_select_indices(self, indices):
        return self.reorder_cache(indices)

    def batch_repeat_interleave(self, repeats):
        if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
            raise ValueError("cache repeats must be a positive integer")
        batch_size = _hsd_cache_batch_size(self)
        if batch_size is None:
            return None
        indices = torch.arange(batch_size, dtype=torch.long).repeat_interleave(repeats)
        return self.reorder_cache(indices)

    cache.reorder_cache = MethodType(reorder_cache, cache)
    cache.batch_select_indices = MethodType(batch_select_indices, cache)
    cache.batch_repeat_interleave = MethodType(batch_repeat_interleave, cache)
    cache._march_batch_ops_patched = True
    return cache
