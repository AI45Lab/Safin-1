"""MARCH GDN HF cache row adapters and anchor history ownership."""

from __future__ import annotations

import torch
from transformers.cache_utils import Cache
from types import SimpleNamespace
from typing import Any


class Safin1GDNCacheMixin:
    """Shared methods; the model entrypoint owns parameters and dispatch."""

    @staticmethod
    def _slice_batch_tensor(value: torch.Tensor | None, batch_idx: int, batch_size: int) -> torch.Tensor | None:
        if not torch.is_tensor(value) or value.ndim == 0 or int(value.shape[0]) != int(batch_size):
            return value
        return value[batch_idx : batch_idx + 1].contiguous()


    @staticmethod
    def _length_for_row(value, batch_idx: int) -> int:
        if value is None:
            return 0
        if torch.is_tensor(value):
            if value.numel() == 0:
                return 0
            flat = value.reshape(-1)
            if flat.numel() == 1:
                return int(flat.item())
            return int(flat[batch_idx].item())
        return int(value or 0)


    def _slice_cache_for_row(self, cache_params: Cache | None, batch_idx: int, batch_size: int) -> Any | None:
        if cache_params is None:
            return None
        row_cache = SimpleNamespace()

        conv_state = self._get_conv_state(cache_params)
        if torch.is_tensor(conv_state):
            row_cache._hsd_conv_states = [None for _ in range(self.layer_idx + 1)]
            row_cache._hsd_conv_states[self.layer_idx] = self._slice_batch_tensor(conv_state, batch_idx, batch_size)
        recurrent_state = self._get_recurrent_state(cache_params)
        if torch.is_tensor(recurrent_state):
            row_cache._hsd_recurrent_states = [None for _ in range(self.layer_idx + 1)]
            row_cache._hsd_recurrent_states[self.layer_idx] = self._slice_batch_tensor(
                recurrent_state,
                batch_idx,
                batch_size,
            )

        source_state = self._get_hsd_state(cache_params)
        row_state = SimpleNamespace()
        if source_state is not None:
            row_state.hsd_seen_text_tokens = self._length_for_row(
                getattr(source_state, "hsd_seen_text_tokens_per_row", getattr(source_state, "hsd_seen_text_tokens", 0)),
                batch_idx,
            )
            old_states = getattr(source_state, "hsd_anchor_states", None)
            old_keys = getattr(source_state, "hsd_anchor_keys", None)
            old_boundaries = getattr(source_state, "hsd_anchor_boundaries", None)
            old_valid = getattr(source_state, "hsd_anchor_valid_mask", None)
            old_memory_mask = getattr(source_state, "hsd_anchor_persistent_memory_mask", None)
            if torch.is_tensor(old_states):
                if torch.is_tensor(old_valid) and old_valid.ndim == 2 and old_valid.shape[0] == batch_size:
                    valid = old_valid[batch_idx].to(device=old_states.device, dtype=torch.bool)
                else:
                    valid = torch.ones(old_states.shape[1], dtype=torch.bool, device=old_states.device)
                row_state.hsd_anchor_states = old_states[batch_idx : batch_idx + 1, valid].contiguous()
                if torch.is_tensor(old_keys):
                    row_state.hsd_anchor_keys = old_keys[batch_idx : batch_idx + 1, valid].contiguous()
                if torch.is_tensor(old_boundaries):
                    if old_boundaries.ndim == 2 and old_boundaries.shape[0] == batch_size:
                        row_state.hsd_anchor_boundaries = old_boundaries[batch_idx, valid].to(old_states.device)
                    else:
                        row_state.hsd_anchor_boundaries = old_boundaries.to(old_states.device)
                if torch.is_tensor(old_memory_mask):
                    if old_memory_mask.ndim == 2 and old_memory_mask.shape[0] == batch_size:
                        row_state.hsd_anchor_persistent_memory_mask = old_memory_mask[
                            batch_idx : batch_idx + 1, valid
                        ].contiguous()
                    else:
                        row_state.hsd_anchor_persistent_memory_mask = old_memory_mask[..., valid].contiguous()

        row_cache._hsd_layer_states = [SimpleNamespace() for _ in range(self.layer_idx + 1)]
        row_cache._hsd_layer_states[self.layer_idx] = row_state
        return row_cache


    def _merge_row_caches(
        self,
        cache_params: Cache,
        row_caches: list[Any | None],
        batch_size: int,
        device: torch.device,
    ) -> None:
        row_states = [self._get_hsd_state(row_cache) for row_cache in row_caches]
        seen = torch.tensor(
            [int(getattr(row_state, "hsd_seen_text_tokens", 0) or 0) for row_state in row_states],
            dtype=torch.long,
            device=device,
        )

        recurrent_rows = [self._get_recurrent_state(row_cache) for row_cache in row_caches]
        if all(torch.is_tensor(item) for item in recurrent_rows):
            self._update_recurrent_state(cache_params, torch.cat(recurrent_rows, dim=0))
        conv_rows = [self._get_conv_state(row_cache) for row_cache in row_caches]
        if all(torch.is_tensor(item) for item in conv_rows):
            self._update_conv_state(cache_params, torch.cat(conv_rows, dim=0))

        hsd_state = self._get_hsd_state(cache_params)
        if hsd_state is None:
            return
        hsd_state.hsd_seen_text_tokens_per_row = seen
        hsd_state.hsd_seen_text_tokens = int(seen.max().item()) if seen.numel() else 0
        if not self.hsd_layer_uses_global_read:
            return
        state_rows = [getattr(row_state, "hsd_anchor_states", None) for row_state in row_states]
        key_rows = [getattr(row_state, "hsd_anchor_keys", None) for row_state in row_states]
        boundary_rows = [getattr(row_state, "hsd_anchor_boundaries", None) for row_state in row_states]
        memory_rows = [
            getattr(row_state, "hsd_anchor_persistent_memory_mask", None) for row_state in row_states
        ]
        tensor_state_rows = [item for item in state_rows if torch.is_tensor(item)]
        if not tensor_state_rows:
            hsd_state.hsd_anchor_states = None
            hsd_state.hsd_anchor_keys = None
            hsd_state.hsd_anchor_boundaries = None
            hsd_state.hsd_anchor_valid_mask = None
            hsd_state.hsd_anchor_persistent_memory_mask = None
            return

        sample_states = tensor_state_rows[0]
        max_anchor = max(int(item.shape[1]) if torch.is_tensor(item) else 0 for item in state_rows)
        states = sample_states.new_zeros(batch_size, max_anchor, *sample_states.shape[2:])
        valid_mask = torch.zeros(batch_size, max_anchor, dtype=torch.bool, device=states.device)
        boundaries = torch.full((batch_size, max_anchor), -1, dtype=torch.long, device=states.device)
        memory_mask = torch.zeros(batch_size, max_anchor, dtype=torch.bool, device=states.device)
        keys = None
        tensor_key_rows = [item for item in key_rows if torch.is_tensor(item)]
        if tensor_key_rows:
            sample_keys = tensor_key_rows[0]
            keys = sample_keys.new_zeros(batch_size, max_anchor, *sample_keys.shape[2:])

        for batch_idx, row_state in enumerate(row_states):
            row_states_tensor = getattr(row_state, "hsd_anchor_states", None)
            if not torch.is_tensor(row_states_tensor):
                continue
            count = int(row_states_tensor.shape[1])
            if count == 0:
                continue
            states[batch_idx : batch_idx + 1, :count] = row_states_tensor.to(states.dtype)
            valid_mask[batch_idx, :count] = True
            row_keys_tensor = getattr(row_state, "hsd_anchor_keys", None)
            if keys is not None and torch.is_tensor(row_keys_tensor):
                keys[batch_idx : batch_idx + 1, :count] = row_keys_tensor.to(keys.dtype)
            row_boundaries_tensor = getattr(row_state, "hsd_anchor_boundaries", None)
            if torch.is_tensor(row_boundaries_tensor):
                boundaries[batch_idx, :count] = row_boundaries_tensor.to(device=boundaries.device, dtype=torch.long)
            row_memory_tensor = memory_rows[batch_idx]
            if torch.is_tensor(row_memory_tensor):
                memory_mask[batch_idx, :count] = row_memory_tensor.reshape(-1)[:count].to(
                    device=memory_mask.device, dtype=torch.bool
                )

        hsd_state.hsd_anchor_states = states
        hsd_state.hsd_anchor_keys = keys
        hsd_state.hsd_anchor_boundaries = boundaries
        hsd_state.hsd_anchor_valid_mask = valid_mask
        hsd_state.hsd_anchor_persistent_memory_mask = memory_mask


    def _merge_anchor_cache(
        self,
        cache_params: Cache | None,
        new_anchor_states: torch.Tensor,
        new_anchor_keys: torch.Tensor | None,
        new_anchor_boundaries: torch.Tensor,
        new_persistent_memory_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        old_states, old_keys, old_boundaries, old_memory_mask = None, None, None, None
        hsd_state = self._get_hsd_state(cache_params)
        if hsd_state is not None:
            old_states = getattr(hsd_state, "hsd_anchor_states", None)
            old_keys = getattr(hsd_state, "hsd_anchor_keys", None)
            old_boundaries = getattr(hsd_state, "hsd_anchor_boundaries", None)
            old_memory_mask = getattr(hsd_state, "hsd_anchor_persistent_memory_mask", None)

        new_memory_mask = new_persistent_memory_mask.to(
            device=new_anchor_states.device, dtype=torch.bool
        )
        if tuple(new_memory_mask.shape) != tuple(new_anchor_states.shape[:2]):
            raise RuntimeError(
                "new persistent memory cache mask shape mismatch: "
                f"got {tuple(new_memory_mask.shape)}, expected {tuple(new_anchor_states.shape[:2])}."
            )

        if old_states is None or old_states.shape[1] == 0:
            return new_anchor_states, new_anchor_keys, new_anchor_boundaries, new_memory_mask
        if new_anchor_states.shape[1] == 0:
            if old_memory_mask is None:
                old_memory_mask = torch.zeros(
                    old_states.shape[:2], dtype=torch.bool, device=old_states.device
                )
            return old_states, old_keys, old_boundaries, old_memory_mask
        if old_keys is None or new_anchor_keys is None:
            merged_keys = None
        else:
            merged_keys = torch.cat([old_keys, new_anchor_keys.to(old_keys.dtype)], dim=1)
        if old_memory_mask is None:
            old_memory_mask = torch.zeros(old_states.shape[:2], dtype=torch.bool, device=old_states.device)
        else:
            old_memory_mask = old_memory_mask.to(device=old_states.device, dtype=torch.bool)
            if old_memory_mask.ndim == 1:
                old_memory_mask = old_memory_mask.view(1, -1).expand(old_states.shape[0], -1)
        return (
            torch.cat([old_states, new_anchor_states.to(old_states.dtype)], dim=1),
            merged_keys,
            torch.cat([old_boundaries.to(new_anchor_boundaries.device), new_anchor_boundaries], dim=0),
            torch.cat([old_memory_mask, new_memory_mask.to(old_memory_mask.device)], dim=1),
        )


    def _get_layer_cache(self, cache_params: Cache | None) -> Any | None:
        if cache_params is None:
            return None
        layers = getattr(cache_params, "layers", None)
        if layers is not None and len(layers) > self.layer_idx:
            return layers[self.layer_idx]
        return cache_params


    def _get_hsd_state(self, cache_params: Cache | None) -> Any | None:
        if cache_params is None:
            return None
        hsd_layers = getattr(cache_params, "_hsd_layer_states", None)
        if hsd_layers is None:
            cache_layers = getattr(cache_params, "layers", None) or []
            num_layers = len(getattr(cache_params, "conv_states", []) or [])
            num_layers = max(num_layers, len(cache_layers))
            num_layers = max(num_layers, self.layer_idx + 1)
            hsd_layers = [SimpleNamespace() for _ in range(num_layers)]
            cache_params._hsd_layer_states = hsd_layers
        while len(hsd_layers) <= self.layer_idx:
            hsd_layers.append(SimpleNamespace())
        return hsd_layers[self.layer_idx]


    def _has_previous_state(self, cache_params: Cache | None) -> bool:
        if cache_params is None:
            return False
        has_previous_state = getattr(cache_params, "has_previous_state", None)
        if callable(has_previous_state):
            try:
                return bool(has_previous_state(self.layer_idx))
            except TypeError:
                return bool(has_previous_state())
        if isinstance(has_previous_state, bool):
            return has_previous_state
        return (
            self._cache_seen_text_tokens(cache_params) > 0
            or self._get_conv_state(cache_params) is not None
            or self._get_recurrent_state(cache_params) is not None
        )


    def _cache_seen_text_tokens(self, cache_params: Cache | None) -> int:
        hsd_state = self._get_hsd_state(cache_params)
        if hsd_state is None:
            return 0
        return int(getattr(hsd_state, "hsd_seen_text_tokens", 0) or 0)


    def _get_conv_state(self, cache_params: Cache | None) -> torch.Tensor | None:
        layer_cache = self._get_layer_cache(cache_params)
        if layer_cache is not None and hasattr(layer_cache, "conv_states"):
            conv_state = layer_cache.conv_states
            if isinstance(conv_state, (list, tuple)):
                return conv_state[self.layer_idx] if len(conv_state) > self.layer_idx else None
            return conv_state
        conv_states = getattr(cache_params, "conv_states", None) if cache_params is not None else None
        if conv_states is not None and len(conv_states) > self.layer_idx:
            return conv_states[self.layer_idx]
        hsd_conv_states = getattr(cache_params, "_hsd_conv_states", None) if cache_params is not None else None
        if hsd_conv_states is not None and len(hsd_conv_states) > self.layer_idx:
            return hsd_conv_states[self.layer_idx]
        return None


    def _get_recurrent_state(self, cache_params: Cache | None) -> torch.Tensor | None:
        hsd_recurrent_states = getattr(cache_params, "_hsd_recurrent_states", None) if cache_params is not None else None
        if hsd_recurrent_states is not None and len(hsd_recurrent_states) > self.layer_idx:
            recurrent_state = hsd_recurrent_states[self.layer_idx]
            if recurrent_state is not None:
                return recurrent_state
        layer_cache = self._get_layer_cache(cache_params)
        if layer_cache is not None and hasattr(layer_cache, "recurrent_states"):
            recurrent_state = layer_cache.recurrent_states
            if isinstance(recurrent_state, (list, tuple)):
                return recurrent_state[self.layer_idx] if len(recurrent_state) > self.layer_idx else None
            return recurrent_state
        recurrent_states = getattr(cache_params, "recurrent_states", None) if cache_params is not None else None
        if recurrent_states is not None and len(recurrent_states) > self.layer_idx:
            return recurrent_states[self.layer_idx]
        return None


    def _update_conv_state(self, cache_params: Cache, conv_state: torch.Tensor) -> None:
        update_conv_state = getattr(cache_params, "update_conv_state", None)
        if callable(update_conv_state):
            try:
                update_conv_state(conv_state, self.layer_idx)
                return
            except AttributeError:
                pass
        conv_states = getattr(cache_params, "conv_states", None)
        if conv_states is not None and len(conv_states) > self.layer_idx:
            conv_states[self.layer_idx] = conv_state
            return
        hsd_conv_states = getattr(cache_params, "_hsd_conv_states", None)
        if hsd_conv_states is None:
            num_layers = len(getattr(cache_params, "layers", None) or [])
            num_layers = max(num_layers, self.layer_idx + 1)
            hsd_conv_states = [None for _ in range(num_layers)]
            cache_params._hsd_conv_states = hsd_conv_states
        while len(hsd_conv_states) <= self.layer_idx:
            hsd_conv_states.append(None)
        hsd_conv_states[self.layer_idx] = conv_state


    def _update_recurrent_state(self, cache_params: Cache, recurrent_state: torch.Tensor) -> None:
        hsd_recurrent_states = getattr(cache_params, "_hsd_recurrent_states", None)
        if hsd_recurrent_states is None:
            num_layers = len(getattr(cache_params, "layers", None) or [])
            num_layers = max(num_layers, self.layer_idx + 1)
            hsd_recurrent_states = [None for _ in range(num_layers)]
            cache_params._hsd_recurrent_states = hsd_recurrent_states
        while len(hsd_recurrent_states) <= self.layer_idx:
            hsd_recurrent_states.append(None)
        hsd_recurrent_states[self.layer_idx] = recurrent_state

        update_recurrent_state = getattr(cache_params, "update_recurrent_state", None)
        if callable(update_recurrent_state):
            try:
                update_recurrent_state(recurrent_state, self.layer_idx)
                return
            except AttributeError:
                pass
        recurrent_states = getattr(cache_params, "recurrent_states", None)
        if recurrent_states is not None and len(recurrent_states) > self.layer_idx:
            recurrent_states[self.layer_idx] = recurrent_state
            return

