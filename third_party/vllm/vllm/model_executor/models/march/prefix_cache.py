# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental prefix snapshot/restore; attention owns all buffers and generations."""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

from .runtime import _advance_hsd_conv_state, _env_truthy, _is_valid_hsd_state_index

logger = init_logger("vllm.model_executor.models.safin1")


class PrefixSnapshotMixin:
    def _hsd_prefix_snapshot_max_bytes_per_layer(self) -> int:
        raw = os.environ.get("VLLM_HSD_PREFIX_CACHE_MAX_BYTES", "").strip()
        if raw:
            try:
                total = int(raw)
            except ValueError:
                logger.warning_once(
                    "Ignoring invalid VLLM_HSD_PREFIX_CACHE_MAX_BYTES=%r",
                    raw,
                )
                total = 2 * 1024**3
        else:
            total = 2 * 1024**3
        if total <= 0:
            return 0
        num_layers = max(1, int(getattr(self.config, "num_hidden_layers", 1) or 1))
        return max(1, total // num_layers)

    @staticmethod
    def _hsd_prefix_snapshot_tensor_bytes(tensor: torch.Tensor | None) -> int:
        if tensor is None:
            return 0
        return int(tensor.numel() * tensor.element_size())

    def _hsd_prefix_snapshot_bytes_for(
        self, snapshot: dict[str, torch.Tensor | int]
    ) -> int:
        total = 0
        for value in snapshot.values():
            if torch.is_tensor(value):
                total += self._hsd_prefix_snapshot_tensor_bytes(value)
        return total

    def _evict_hsd_prefix_snapshots(self) -> None:
        max_bytes = self._hsd_prefix_snapshot_max_bytes_per_layer()
        if max_bytes <= 0:
            self.hsd_prefix_snapshot_step_evicted_keys.update(
                self.hsd_prefix_snapshot_cache
            )
            self.hsd_prefix_snapshot_cache.clear()
            self.hsd_prefix_snapshot_bytes = 0
            return
        while (
            self.hsd_prefix_snapshot_bytes > max_bytes
            and self.hsd_prefix_snapshot_cache
        ):
            key, snapshot = self.hsd_prefix_snapshot_cache.popitem(last=False)
            self.hsd_prefix_snapshot_step_evicted_keys.add(key)
            self.hsd_prefix_snapshot_bytes -= self._hsd_prefix_snapshot_bytes_for(
                snapshot
            )
            self._record_hsd_varlen_event(
                "hsd_prefix_cache_fallback",
                reason="evicted",
                key=key,
                prefix_snapshot_bytes=int(self.hsd_prefix_snapshot_bytes),
            )

    def _materialize_hsd_anchor_history_for_snapshot(
        self,
        state_idx: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        page_cache = getattr(self, "hsd_anchor_page_cache", None)
        if page_cache is not None:
            materialized = self._materialize_anchor_page_history(
                page_cache,
                state_idx,
                device=device,
            )
            if materialized is not None:
                return materialized
        cached = self.hsd_anchor_cache.get(state_idx)
        if cached is None:
            return None
        states = cached.get("states")
        keys = cached.get("keys")
        boundaries = cached.get("boundaries")
        if not (
            torch.is_tensor(states)
            and torch.is_tensor(keys)
            and torch.is_tensor(boundaries)
            and int(states.shape[1]) > 0
        ):
            return None
        return (
            states.to(device=device),
            keys.to(device=device),
            boundaries.to(device=device),
        )

    def _store_hsd_prefix_snapshot_parts(
        self,
        *,
        key: str,
        state_idx: int,
        prefix_len: int,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        anchor_states: torch.Tensor | None = None,
        anchor_keys: torch.Tensor | None = None,
        anchor_boundaries: torch.Tensor | None = None,
        replay_base_len: int | None = None,
        replay_recurrent_state: torch.Tensor | None = None,
        replay_query: torch.Tensor | None = None,
        replay_key: torch.Tensor | None = None,
        replay_value: torch.Tensor | None = None,
        replay_g: torch.Tensor | None = None,
        replay_beta: torch.Tensor | None = None,
    ) -> bool:
        if not key:
            self._record_hsd_varlen_event(
                "hsd_prefix_cache_fallback",
                reason="snapshot_empty_key",
                prefix_len=int(prefix_len),
                state_idx=int(state_idx),
            )
            return False
        if not _is_valid_hsd_state_index(state_idx):
            self._record_hsd_varlen_event(
                "hsd_prefix_cache_fallback",
                reason="snapshot_invalid_state_idx",
                key=key,
                prefix_len=int(prefix_len),
                state_idx=int(state_idx),
            )
            return False
        max_bytes = self._hsd_prefix_snapshot_max_bytes_per_layer()
        if max_bytes <= 0:
            self._record_hsd_varlen_event(
                "hsd_prefix_cache_fallback",
                reason="snapshot_cache_disabled",
                key=key,
                prefix_len=int(prefix_len),
                state_idx=int(state_idx),
            )
            return False
        snapshot: dict[str, torch.Tensor | int] = {
            "cache_generation": int(self.hsd_prefix_cache_generation),
            "prefix_len": int(prefix_len),
            "conv_state": conv_state.detach().clone(),
            "recurrent_state": recurrent_state.detach().clone(),
        }
        if (
            torch.is_tensor(anchor_states)
            and torch.is_tensor(anchor_keys)
            and torch.is_tensor(anchor_boundaries)
            and int(anchor_states.shape[1]) > 0
        ):
            snapshot["anchor_states"] = anchor_states.detach().clone()
            snapshot["anchor_keys"] = anchor_keys.detach().clone()
            snapshot["anchor_boundaries"] = anchor_boundaries.detach().clone()
        replay_tensors = (
            replay_recurrent_state,
            replay_query,
            replay_key,
            replay_value,
            replay_g,
            replay_beta,
        )
        if replay_base_len is not None and all(
            torch.is_tensor(tensor) for tensor in replay_tensors
        ):
            snapshot["replay_base_len"] = int(replay_base_len)
            snapshot["replay_recurrent_state"] = replay_recurrent_state.detach().clone()
            snapshot["replay_query"] = replay_query.detach().clone()
            snapshot["replay_key"] = replay_key.detach().clone()
            snapshot["replay_value"] = replay_value.detach().clone()
            snapshot["replay_g"] = replay_g.detach().clone()
            snapshot["replay_beta"] = replay_beta.detach().clone()

        snapshot_bytes = self._hsd_prefix_snapshot_bytes_for(snapshot)
        if snapshot_bytes > max_bytes:
            self._record_hsd_varlen_event(
                "hsd_prefix_cache_fallback",
                reason="snapshot_too_large",
                key=key,
                snapshot_bytes=int(snapshot_bytes),
                max_bytes=int(max_bytes),
            )
            return False
        old = self.hsd_prefix_snapshot_cache.pop(key, None)
        if old is not None:
            self.hsd_prefix_snapshot_bytes -= self._hsd_prefix_snapshot_bytes_for(old)
        self.hsd_prefix_snapshot_cache[key] = snapshot
        self.hsd_prefix_snapshot_bytes += snapshot_bytes
        self._evict_hsd_prefix_snapshots()
        if key in self.hsd_prefix_snapshot_cache:
            self._record_hsd_varlen_event(
                "hsd_prefix_snapshot_store",
                key=key,
                prefix_len=int(prefix_len),
                state_idx=int(state_idx),
                recurrent_replay_tokens=int(
                    snapshot.get("replay_query", conv_state.new_zeros(1, 0)).shape[1]
                )
                if torch.is_tensor(snapshot.get("replay_query"))
                else 0,
                snapshot_bytes=int(snapshot_bytes),
                prefix_snapshot_bytes=int(self.hsd_prefix_snapshot_bytes),
            )
            ready = getattr(self, "hsd_prefix_snapshot_step_ready_keys", None)
            if ready is None:
                ready = set()
                self.hsd_prefix_snapshot_step_ready_keys = ready
            ready.add(key)
            return True
        return False

    def capture_hsd_prefix_snapshot(
        self,
        *,
        key: str,
        state_idx: int,
        prefix_len: int,
    ) -> bool:
        conv_state = self._conv_state_for_request(state_idx)
        recurrent_state = self._stored_recurrent_state_for_request(state_idx)
        if conv_state is None or recurrent_state is None:
            return False
        anchor_states = anchor_keys = anchor_boundaries = None
        materialized = self._materialize_hsd_anchor_history_for_snapshot(
            state_idx,
            device=conv_state.device,
        )
        if materialized is not None:
            anchor_states, anchor_keys, anchor_boundaries = materialized
        return self._store_hsd_prefix_snapshot_parts(
            key=key,
            state_idx=state_idx,
            prefix_len=prefix_len,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            anchor_states=anchor_states,
            anchor_keys=anchor_keys,
            anchor_boundaries=anchor_boundaries,
        )

    def capture_hsd_prefix_snapshot_from_tensors(
        self,
        *,
        state_idx: int,
        history_state_idx: int,
        past_text_len: int,
        snapshot_requests: list[tuple[str, int, int]],
        conv_state_snapshot: torch.Tensor,
        pre_conv_inputs: torch.Tensor,
        query: torch.Tensor,
        key_tensor: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        final_recurrent_state: torch.Tensor | None,
        new_anchor_states: torch.Tensor | None,
        new_anchor_keys: torch.Tensor | None,
        new_anchor_boundaries: torch.Tensor,
    ) -> None:
        if not snapshot_requests:
            return
        text_len = int(pre_conv_inputs.shape[0])
        requests = sorted(
            {
                (str(key), int(prefix_len), int(local_pos))
                for key, prefix_len, local_pos in snapshot_requests
            },
            key=lambda item: item[2],
        )
        cursor = 0
        state = initial_state.contiguous()
        old_anchor = self._materialize_hsd_anchor_history_for_snapshot(
            history_state_idx,
            device=pre_conv_inputs.device,
        )
        old_states = old_keys = old_boundaries = None
        if old_anchor is not None:
            old_states, old_keys, old_boundaries = old_anchor
            # Grouped paged prefill can append this step's anchors before
            # snapshot capture. Merge only history that predates this step;
            # the new anchors below supply each current boundary once.
            prior = old_boundaries <= int(past_text_len)
            old_states = old_states[:, prior]
            old_keys = old_keys[:, prior]
            old_boundaries = old_boundaries[prior]

        for snapshot_key, prefix_len, local_pos in requests:
            expected_local_pos = int(prefix_len) - int(past_text_len)
            if int(local_pos) != expected_local_pos:
                self._record_hsd_varlen_event(
                    "hsd_prefix_cache_fallback",
                    reason="snapshot_local_pos_mismatch",
                    key=snapshot_key,
                    prefix_len=int(prefix_len),
                    past_text_len=int(past_text_len),
                    local_pos=int(local_pos),
                    expected_local_pos=expected_local_pos,
                )
                continue
            if local_pos <= 0 or local_pos > text_len:
                self._record_hsd_varlen_event(
                    "hsd_prefix_cache_fallback",
                    reason="snapshot_local_pos_out_of_range",
                    key=snapshot_key,
                    prefix_len=int(prefix_len),
                    local_pos=int(local_pos),
                    text_len=text_len,
                )
                continue
            boundary_checkpoint_state = None
            boundary_checkpoint_local_pos = None
            if (
                _env_truthy(
                    "VLLM_HSD_PREFIX_CACHE_USE_BOUNDARY_STATE",
                    default=True,
                )
                and torch.is_tensor(new_anchor_states)
                and int(new_anchor_states.shape[1]) > 0
                and int(new_anchor_boundaries.numel()) > 0
            ):
                eligible_boundaries = new_anchor_boundaries.to(
                    device=pre_conv_inputs.device
                ) <= int(prefix_len)
                checkpoint_indices = eligible_boundaries.nonzero(
                    as_tuple=False
                ).flatten()
                if int(checkpoint_indices.numel()) > 0:
                    checkpoint_idx = int(checkpoint_indices[-1].item())
                    checkpoint_prefix_len = int(
                        new_anchor_boundaries[checkpoint_idx].item()
                    )
                    checkpoint_local_pos = checkpoint_prefix_len - int(past_text_len)
                    if cursor <= checkpoint_local_pos <= local_pos:
                        boundary_checkpoint_state = new_anchor_states[
                            :, checkpoint_idx
                        ].contiguous()
                        boundary_checkpoint_local_pos = checkpoint_local_pos
            if boundary_checkpoint_state is not None:
                assert boundary_checkpoint_local_pos is not None
                state = boundary_checkpoint_state
                cursor = boundary_checkpoint_local_pos
                self._record_hsd_varlen_event(
                    (
                        "hsd_prefix_snapshot_exact_boundary_state_hit"
                        if cursor == local_pos
                        else "hsd_prefix_snapshot_boundary_checkpoint_hit"
                    ),
                    key=snapshot_key,
                    prefix_len=int(prefix_len),
                    local_pos=int(local_pos),
                    checkpoint_local_pos=int(cursor),
                )
            if local_pos > cursor:
                _, state = self._run_delta_segment(
                    query,
                    key_tensor,
                    value,
                    g,
                    beta,
                    cursor,
                    local_pos,
                    state,
                )
                cursor = local_pos

            if (
                local_pos == text_len
                and torch.is_tensor(final_recurrent_state)
                and tuple(final_recurrent_state.shape) == tuple(state.shape)
            ):
                replay_state = state
                state = final_recurrent_state.contiguous()
                cursor = local_pos
                event_payload: dict[str, object] = {
                    "key": snapshot_key,
                    "prefix_len": int(prefix_len),
                    "local_pos": int(local_pos),
                }
                if _env_truthy("VLLM_HSD_PREFIX_CACHE_DEBUG"):
                    event_payload.update(
                        self._hsd_tensor_diff_stats(
                            "replayed_vs_step_final_state",
                            replay_state,
                            state,
                        )
                    )
                self._record_hsd_varlen_event(
                    "hsd_prefix_snapshot_exact_step_final_state_hit",
                    **event_payload,
                )

            conv_state = _advance_hsd_conv_state(
                conv_state_snapshot,
                pre_conv_inputs[:local_pos],
                has_initial_state=past_text_len > 0,
            )
            step_final_conv_state = self._conv_state_for_request(state_idx)
            if (
                local_pos == text_len
                and torch.is_tensor(step_final_conv_state)
                and tuple(step_final_conv_state.shape) == tuple(conv_state.shape)
            ):
                reconstructed_conv_state = conv_state
                conv_state = step_final_conv_state.detach().clone()
                event_payload = {
                    "key": snapshot_key,
                    "prefix_len": int(prefix_len),
                    "local_pos": int(local_pos),
                }
                if _env_truthy("VLLM_HSD_PREFIX_CACHE_DEBUG"):
                    event_payload.update(
                        self._hsd_tensor_diff_stats(
                            "reconstructed_vs_step_final_conv_state",
                            reconstructed_conv_state,
                            conv_state,
                        )
                    )
                self._record_hsd_varlen_event(
                    "hsd_prefix_snapshot_exact_step_final_conv_state_hit",
                    **event_payload,
                )
            if _env_truthy("VLLM_HSD_PREFIX_CACHE_DEBUG"):
                self._record_hsd_varlen_event(
                    "hsd_prefix_snapshot_store_debug",
                    key=snapshot_key,
                    state_idx=int(state_idx),
                    prefix_len=int(prefix_len),
                    past_text_len=int(past_text_len),
                    local_pos=int(local_pos),
                    conv_initial_state_used=bool(past_text_len > 0),
                    recurrent_norm=float(state.detach().float().norm().item()),
                    recurrent_sum=float(state.detach().float().sum().item()),
                    conv_norm=float(conv_state.detach().float().norm().item()),
                    conv_sum=float(conv_state.detach().float().sum().item()),
                )
            anchor_states = old_states
            anchor_keys = old_keys
            anchor_boundaries = old_boundaries
            if (
                torch.is_tensor(new_anchor_states)
                and torch.is_tensor(new_anchor_keys)
                and int(new_anchor_states.shape[1]) > 0
            ):
                mask = new_anchor_boundaries.to(device=pre_conv_inputs.device) <= int(
                    prefix_len
                )
                if bool(mask.any().item()):
                    selected_states = new_anchor_states[:, mask]
                    selected_keys = new_anchor_keys[:, mask]
                    selected_boundaries = new_anchor_boundaries[mask].to(
                        device=pre_conv_inputs.device
                    )
                    if anchor_states is None or anchor_keys is None:
                        anchor_states = selected_states
                        anchor_keys = selected_keys
                        anchor_boundaries = selected_boundaries
                    else:
                        assert anchor_boundaries is not None
                        anchor_states = torch.cat(
                            [anchor_states, selected_states],
                            dim=1,
                        )
                        anchor_keys = torch.cat([anchor_keys, selected_keys], dim=1)
                        anchor_boundaries = torch.cat(
                            [
                                anchor_boundaries.to(device=selected_boundaries.device),
                                selected_boundaries,
                            ],
                            dim=0,
                        )

            replay_kwargs: dict[str, object] = {}
            if _env_truthy("VLLM_HSD_PREFIX_CACHE_RECURRENT_REPLAY"):
                anchor_chunk_size = int(
                    getattr(self.config, "anchor_chunk_size", 0) or 0
                )
                if anchor_chunk_size <= 0:
                    self._record_hsd_varlen_event(
                        "hsd_prefix_cache_fallback",
                        reason="recurrent_replay_bad_anchor_chunk_size",
                        key=snapshot_key,
                        prefix_len=int(prefix_len),
                    )
                    continue
                replay_base_len = (
                    int(prefix_len) // anchor_chunk_size
                ) * anchor_chunk_size
                replay_tokens = int(prefix_len) - replay_base_len
                if replay_tokens > 0:
                    replay_start = replay_base_len - int(past_text_len)
                    if replay_start < 0 or replay_start > local_pos:
                        self._record_hsd_varlen_event(
                            "hsd_prefix_cache_fallback",
                            reason="recurrent_replay_base_not_in_step",
                            key=snapshot_key,
                            prefix_len=int(prefix_len),
                            past_text_len=int(past_text_len),
                            replay_base_len=int(replay_base_len),
                            local_pos=int(local_pos),
                        )
                        continue

                    replay_state = None
                    if replay_start == 0:
                        replay_state = initial_state.contiguous()
                    elif (
                        torch.is_tensor(new_anchor_states)
                        and int(new_anchor_states.shape[1]) > 0
                    ):
                        replay_boundary_matches = (
                            (
                                new_anchor_boundaries.to(device=pre_conv_inputs.device)
                                == int(replay_base_len)
                            )
                            .nonzero(as_tuple=False)
                            .flatten()
                        )
                        if int(replay_boundary_matches.numel()) > 0:
                            replay_idx = int(replay_boundary_matches[-1].item())
                            replay_state = new_anchor_states[:, replay_idx].contiguous()
                    if replay_state is None:
                        self._record_hsd_varlen_event(
                            "hsd_prefix_cache_fallback",
                            reason="recurrent_replay_state_missing",
                            key=snapshot_key,
                            prefix_len=int(prefix_len),
                            past_text_len=int(past_text_len),
                            replay_base_len=int(replay_base_len),
                            replay_start=int(replay_start),
                        )
                        continue

                    replay_end = int(local_pos)
                    if replay_end - replay_start != replay_tokens:
                        self._record_hsd_varlen_event(
                            "hsd_prefix_cache_fallback",
                            reason="recurrent_replay_length_mismatch",
                            key=snapshot_key,
                            prefix_len=int(prefix_len),
                            replay_base_len=int(replay_base_len),
                            replay_start=int(replay_start),
                            replay_end=int(replay_end),
                            replay_tokens=int(replay_tokens),
                        )
                        continue
                    replay_kwargs = {
                        "replay_base_len": int(replay_base_len),
                        "replay_recurrent_state": replay_state,
                        "replay_query": query[:, replay_start:replay_end],
                        "replay_key": key_tensor[:, replay_start:replay_end],
                        "replay_value": value[:, replay_start:replay_end],
                        "replay_g": g[:, replay_start:replay_end],
                        "replay_beta": beta[:, replay_start:replay_end],
                    }

            stored = self._store_hsd_prefix_snapshot_parts(
                key=snapshot_key,
                state_idx=state_idx,
                prefix_len=int(prefix_len),
                conv_state=conv_state,
                recurrent_state=state,
                anchor_states=anchor_states,
                anchor_keys=anchor_keys,
                anchor_boundaries=anchor_boundaries,
                **replay_kwargs,
            )
            if not stored:
                self._record_hsd_varlen_event(
                    "hsd_prefix_cache_fallback",
                    reason="snapshot_store_failed",
                    key=snapshot_key,
                    prefix_len=int(prefix_len),
                )
            elif replay_kwargs and int(local_pos) == text_len:
                stored_snapshot = self.hsd_prefix_snapshot_cache.get(snapshot_key)
                if stored_snapshot is not None:
                    replay_names = (
                        "replay_recurrent_state",
                        "replay_query",
                        "replay_key",
                        "replay_value",
                        "replay_g",
                        "replay_beta",
                    )
                    if all(
                        torch.is_tensor(stored_snapshot.get(name))
                        for name in replay_names
                    ):
                        self.hsd_prefix_recurrent_replay[int(state_idx)] = {
                            "prefix_len": int(prefix_len),
                            "replay_base_len": int(stored_snapshot["replay_base_len"]),
                            **{name: stored_snapshot[name] for name in replay_names},
                        }
                        self._record_hsd_varlen_event(
                            "hsd_prefix_recurrent_replay_armed_after_store",
                            key=snapshot_key,
                            state_idx=int(state_idx),
                            prefix_len=int(prefix_len),
                            replay_tokens=int(stored_snapshot["replay_query"].shape[1]),
                        )

    def _hsd_graph_prefix_storage(self) -> dict[str, torch.Tensor]:
        """Return already allocated buffers read by captured Graph V3 decode."""
        sidecar = self.hsd_persistent_state_sidecar
        buffers = {
            name: sidecar.get(name) for name in ("conv_state", "recurrent_state")
        }
        if self.hsd_layer_uses_global_read:
            pages = self.hsd_anchor_page_cache
            if (
                pages is None
                or not pages.fixed_capacity
                or not pages._metadata_fixed_capacity
            ):
                raise RuntimeError(
                    "Graph prefix restore requires fixed history pages and metadata"
                )
            buffers.update(
                {
                    name: getattr(pages, name, None)
                    for name in (
                        "states_pages",
                        "keys_pages",
                        "boundary_pages",
                        "_lengths_by_state",
                        "_page_table_by_state",
                    )
                }
            )
        if any(not torch.is_tensor(value) for value in buffers.values()):
            raise RuntimeError(
                "Graph prefix restore requires preallocated state buffers"
            )
        return buffers

    def _hsd_prefix_snapshot_restore_preflight(
        self,
        *,
        key: str,
        state_idx: int,
        history_state_idx: int | None,
        prefix_len: int,
    ) -> str | None:
        snapshot = self.hsd_prefix_snapshot_cache.get(key)
        if snapshot is None:
            return "snapshot_missing"
        if int(snapshot.get("cache_generation", -1)) != int(
            self.hsd_prefix_cache_generation
        ):
            return "snapshot_stale_generation"
        if not _is_valid_hsd_state_index(state_idx):
            return "snapshot_invalid_state_idx"
        if int(snapshot.get("prefix_len", -1)) != int(prefix_len):
            return "config_mismatch"
        if not (
            torch.is_tensor(snapshot.get("conv_state"))
            and torch.is_tensor(snapshot.get("recurrent_state"))
        ):
            return "snapshot_state_missing"
        conv_state, recurrent_state = (
            snapshot["conv_state"],
            snapshot["recurrent_state"],
        )
        if (
            conv_state.ndim != 3
            or recurrent_state.ndim != 4
            or conv_state.shape[0] != 1
            or recurrent_state.shape[0] != 1
        ):
            return "snapshot_state_shape"
        anchor_state_idx = (
            int(state_idx) if history_state_idx is None else int(history_state_idx)
        )
        if not _is_valid_hsd_state_index(anchor_state_idx):
            return "snapshot_invalid_history_state_idx"
        if _env_truthy("VLLM_HSD_ENABLE_GRAPH_PREFIX_CACHE"):
            try:
                storage = self._hsd_graph_prefix_storage()
            except RuntimeError:
                return "graph_storage_unready"
            conv, recurrent = storage["conv_state"], storage["recurrent_state"]
            if conv.is_cuda and torch.cuda.is_current_stream_capturing():
                return "graph_restore_inside_capture"
            if state_idx >= min(conv.shape[0], recurrent.shape[0]):
                return "graph_state_capacity"
            recurrent_shape = (
                *recurrent_state.shape[1:-2],
                recurrent_state.shape[-1],
                recurrent_state.shape[-2],
            )
            if (
                tuple(conv.shape[1:]) != tuple(conv_state.shape[1:])
                or tuple(recurrent.shape[1:]) != recurrent_shape
            ):
                return "graph_state_shape"
            lengths = storage.get("_lengths_by_state")
            if lengths is not None and anchor_state_idx >= lengths.shape[0]:
                return "graph_history_capacity"
        states = snapshot.get("anchor_states")
        keys = snapshot.get("anchor_keys")
        boundaries = snapshot.get("anchor_boundaries")
        anchor_tensors = (
            torch.is_tensor(states),
            torch.is_tensor(keys),
            torch.is_tensor(boundaries),
        )
        if any(anchor_tensors) and not all(anchor_tensors):
            return "anchor_snapshot_incomplete"
        if all(anchor_tensors):
            if (
                states.ndim != 5
                or keys.ndim != 4
                or boundaries.ndim != 1
                or states.shape[0] != 1
                or keys.shape[0] != 1
                or states.shape[1] != keys.shape[1]
                or states.shape[1] != boundaries.numel()
                or tuple(states.shape[2:]) != tuple(recurrent_state.shape[1:])
            ):
                return "anchor_snapshot_shape"
            if boundaries.numel() and bool(
                (boundaries < 0).any()
                or (boundaries > int(prefix_len)).any()
                or (boundaries[1:] <= boundaries[:-1]).any()
            ):
                return "anchor_snapshot_boundaries"
        if all(anchor_tensors) and int(states.shape[1]) > 0:
            page_cache = getattr(self, "hsd_anchor_page_cache", None)
            if (
                page_cache is not None
                and getattr(page_cache, "fixed_capacity", False)
                and not page_cache.can_replace(
                    anchor_state_idx,
                    int(states.shape[1]),
                )
            ):
                return "anchor_page_capacity"
        return None

    def restore_hsd_prefix_snapshot(
        self,
        *,
        key: str,
        state_idx: int,
        history_state_idx: int | None = None,
        prefix_len: int,
    ) -> bool:
        preflight_reason = self._hsd_prefix_snapshot_restore_preflight(
            key=key,
            state_idx=state_idx,
            history_state_idx=history_state_idx,
            prefix_len=prefix_len,
        )
        if preflight_reason is not None:
            self._record_hsd_varlen_event(
                "hsd_prefix_cache_fallback",
                reason=preflight_reason,
                key=key,
                prefix_len=int(prefix_len),
                state_idx=int(state_idx),
                history_state_idx=(
                    int(state_idx)
                    if history_state_idx is None
                    else int(history_state_idx)
                ),
            )
            return False
        snapshot = self.hsd_prefix_snapshot_cache.get(key)
        assert snapshot is not None
        if int(snapshot.get("prefix_len", -1)) != int(prefix_len):
            self._record_hsd_varlen_event(
                "hsd_prefix_cache_fallback",
                reason="config_mismatch",
                key=key,
                expected_prefix_len=int(prefix_len),
                snapshot_prefix_len=int(snapshot.get("prefix_len", -1)),
            )
            return False
        conv_state = snapshot.get("conv_state")
        recurrent_state = snapshot.get("recurrent_state")
        if not (torch.is_tensor(conv_state) and torch.is_tensor(recurrent_state)):
            return False
        self.hsd_prefix_recurrent_replay.pop(int(state_idx), None)
        recurrent_replay_tokens = 0
        if _env_truthy("VLLM_HSD_PREFIX_CACHE_RECURRENT_REPLAY"):
            anchor_chunk_size = int(getattr(self.config, "anchor_chunk_size", 0) or 0)
            if anchor_chunk_size <= 0:
                return False
            recurrent_replay_tokens = int(prefix_len) % anchor_chunk_size
            if recurrent_replay_tokens > 0:
                replay_names = (
                    "replay_recurrent_state",
                    "replay_query",
                    "replay_key",
                    "replay_value",
                    "replay_g",
                    "replay_beta",
                )
                replay_tensors = {name: snapshot.get(name) for name in replay_names}
                replay_base_len = snapshot.get("replay_base_len")
                if not isinstance(replay_base_len, int) or any(
                    not torch.is_tensor(tensor) for tensor in replay_tensors.values()
                ):
                    self._record_hsd_varlen_event(
                        "hsd_prefix_cache_fallback",
                        reason="recurrent_replay_snapshot_missing",
                        key=key,
                        prefix_len=int(prefix_len),
                        state_idx=int(state_idx),
                    )
                    return False
                if (
                    int(replay_base_len) + recurrent_replay_tokens != int(prefix_len)
                    or int(replay_base_len) % anchor_chunk_size != 0
                    or any(
                        int(replay_tensors[name].shape[1]) != recurrent_replay_tokens
                        for name in replay_names[1:]
                    )
                    or tuple(replay_tensors["replay_recurrent_state"].shape)
                    != tuple(recurrent_state.shape)
                ):
                    self._record_hsd_varlen_event(
                        "hsd_prefix_cache_fallback",
                        reason="recurrent_replay_snapshot_invalid",
                        key=key,
                        prefix_len=int(prefix_len),
                        replay_base_len=int(replay_base_len),
                        replay_tokens=int(recurrent_replay_tokens),
                        state_idx=int(state_idx),
                    )
                    return False
                self.hsd_prefix_recurrent_replay[int(state_idx)] = {
                    "prefix_len": int(prefix_len),
                    "replay_base_len": int(replay_base_len),
                    **replay_tensors,
                }
        anchor_state_idx = (
            int(state_idx) if history_state_idx is None else int(history_state_idx)
        )
        if not _is_valid_hsd_state_index(anchor_state_idx):
            self._record_hsd_varlen_event(
                "hsd_prefix_cache_fallback",
                reason="snapshot_invalid_history_state_idx",
                key=key,
                prefix_len=int(prefix_len),
                state_idx=int(state_idx),
                history_state_idx=int(anchor_state_idx),
            )
            return False

        graph_storage = (
            {
                name: value.data_ptr()
                for name, value in self._hsd_graph_prefix_storage().items()
            }
            if _env_truthy("VLLM_HSD_ENABLE_GRAPH_PREFIX_CACHE")
            else None
        )
        self._store_conv_state(state_idx, conv_state)
        self._store_recurrent_state(state_idx, recurrent_state)
        self.hsd_text_state_cache[state_idx] = {
            "conv_state": conv_state.detach().clone(),
            "recurrent_state": recurrent_state.detach().clone(),
        }
        if _env_truthy("VLLM_HSD_PREFIX_CACHE_DEBUG"):
            self._record_hsd_varlen_event(
                "hsd_prefix_snapshot_restore_debug",
                key=key,
                prefix_len=int(prefix_len),
                state_idx=int(state_idx),
                restored_recurrent_norm=float(
                    recurrent_state.detach().float().norm().item()
                ),
                restored_recurrent_sum=float(
                    recurrent_state.detach().float().sum().item()
                ),
                restored_conv_norm=float(conv_state.detach().float().norm().item()),
                restored_conv_sum=float(conv_state.detach().float().sum().item()),
            )

        page_cache = getattr(self, "hsd_anchor_page_cache", None)
        if page_cache is not None:
            page_cache.clear({anchor_state_idx})
        self.hsd_anchor_cache.pop(anchor_state_idx, None)
        states = snapshot.get("anchor_states")
        keys = snapshot.get("anchor_keys")
        boundaries = snapshot.get("anchor_boundaries")
        if (
            torch.is_tensor(states)
            and torch.is_tensor(keys)
            and torch.is_tensor(boundaries)
            and int(states.shape[1]) > 0
        ):
            if page_cache is not None:
                appended = (
                    page_cache.try_append(
                        anchor_state_idx,
                        self._anchor_states_to_paged_cache(states),
                        keys,
                        boundaries,
                    )
                    if getattr(page_cache, "fixed_capacity", False)
                    else True
                )
                if not appended:
                    self._record_hsd_varlen_event(
                        "hsd_prefix_cache_fallback",
                        reason="anchor_page_capacity",
                        key=key,
                        prefix_len=int(prefix_len),
                    )
                    return False
                if not getattr(page_cache, "fixed_capacity", False):
                    page_cache.append(
                        anchor_state_idx,
                        self._anchor_states_to_paged_cache(states),
                        keys,
                        boundaries,
                    )
            else:
                self.hsd_anchor_cache[anchor_state_idx] = {
                    "states": states.detach().clone(),
                    "keys": keys.detach().clone(),
                    "boundaries": boundaries.detach().clone(),
                }
        if graph_storage is not None:
            current = {
                name: value.data_ptr()
                for name, value in self._hsd_graph_prefix_storage().items()
            }
            if current != graph_storage:
                raise RuntimeError(
                    "Graph prefix restore replaced captured state storage"
                )
            self._record_hsd_varlen_event(
                "hsd_prefix_graph_restore_ready",
                key=key,
                prefix_len=int(prefix_len),
                storage_preserved=True,
            )
        self.hsd_prefix_snapshot_cache.move_to_end(key)
        self._record_hsd_varlen_event(
            "hsd_prefix_snapshot_restore",
            key=key,
            prefix_len=int(prefix_len),
            state_idx=int(state_idx),
            history_state_idx=int(anchor_state_idx),
            recurrent_replay_tokens=int(recurrent_replay_tokens),
        )
        return True

    def _run_hsd_prefix_recurrent_replay(
        self,
        *,
        state_idx: int,
        past_text_len: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        anchor_q: torch.Tensor,
        boundary_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not _env_truthy("VLLM_HSD_PREFIX_CACHE_RECURRENT_REPLAY"):
            return None
        replay = self.hsd_prefix_recurrent_replay.pop(int(state_idx), None)
        if replay is None:
            return None
        prefix_len = int(replay.get("prefix_len", -1))
        replay_base_len = int(replay.get("replay_base_len", -1))
        replay_query = replay.get("replay_query")
        replay_key = replay.get("replay_key")
        replay_value = replay.get("replay_value")
        replay_g = replay.get("replay_g")
        replay_beta = replay.get("replay_beta")
        replay_state = replay.get("replay_recurrent_state")
        replay_tensors = (
            replay_query,
            replay_key,
            replay_value,
            replay_g,
            replay_beta,
            replay_state,
        )
        if prefix_len != int(past_text_len) or any(
            not torch.is_tensor(tensor) for tensor in replay_tensors
        ):
            raise RuntimeError(
                "hsd_prefix_recurrent_replay_invalid: "
                f"state_idx={state_idx} prefix_len={prefix_len} "
                f"past_text_len={past_text_len}"
            )
        replay_tokens = int(replay_query.shape[1])
        if (
            replay_tokens <= 0
            or replay_base_len + replay_tokens != prefix_len
            or any(
                int(tensor.shape[1]) != replay_tokens
                for tensor in (
                    replay_query,
                    replay_key,
                    replay_value,
                    replay_g,
                    replay_beta,
                )
            )
        ):
            raise RuntimeError(
                "hsd_prefix_recurrent_replay_shape_mismatch: "
                f"state_idx={state_idx} replay_base_len={replay_base_len} "
                f"replay_tokens={replay_tokens} prefix_len={prefix_len}"
            )

        def prepend(
            replay_tensor: torch.Tensor,
            current_tensor: torch.Tensor,
        ) -> torch.Tensor:
            return torch.cat(
                [
                    replay_tensor.to(
                        device=current_tensor.device,
                        dtype=current_tensor.dtype,
                    ),
                    current_tensor,
                ],
                dim=1,
            ).contiguous()

        replayed_query = prepend(replay_query, query)
        replayed_key = prepend(replay_key, key)
        replayed_value = prepend(replay_value, value)
        replayed_g = prepend(replay_g, g)
        replayed_beta = prepend(replay_beta, beta)
        text_o, anchor_o, anchor_states, recurrent_state = self._run_hsd_segments(
            query=replayed_query,
            key=replayed_key,
            value=replayed_value,
            g=replayed_g,
            beta=replayed_beta,
            anchor_q=anchor_q,
            boundary_counts=boundary_counts,
            boundary_start=replay_base_len,
            initial_state=replay_state.to(
                device=query.device,
                dtype=torch.float32,
            ).clone(),
        )
        if int(text_o.shape[1]) != replay_tokens + int(query.shape[1]):
            raise RuntimeError(
                "hsd_prefix_recurrent_replay_output_mismatch: "
                f"state_idx={state_idx} output_tokens={int(text_o.shape[1])} "
                f"replay_tokens={replay_tokens} text_tokens={int(query.shape[1])}"
            )
        self._record_hsd_varlen_event(
            "hsd_prefix_recurrent_replay_hit",
            state_idx=int(state_idx),
            prefix_len=int(prefix_len),
            replay_base_len=int(replay_base_len),
            replay_tokens=int(replay_tokens),
            text_tokens=int(query.shape[1]),
            anchor_tokens=int(anchor_q.shape[1]),
        )
        return (
            text_o[:, replay_tokens:].contiguous(),
            anchor_o,
            anchor_states,
            recurrent_state,
        )
