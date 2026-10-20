# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped paged history reads during prefill.

Uses the attention-owned page cache, preserves append/read ordering and reports
capacity fallbacks. Decode and request-slot lifecycle remain in the model."""

from __future__ import annotations

import os

import torch

from . import backends
from .runtime import _env_truthy


class PagedPrefillMixin:
    def _try_batched_paged_prefill_history_read(
        self,
        *,
        text_o: torch.Tensor,
        query: torch.Tensor,
        router_q: torch.Tensor | None,
        null_logits: torch.Tensor | None,
        anchor_states: torch.Tensor,
        anchor_keys: torch.Tensor | None,
        anchor_offsets: list[int],
        anchor_boundaries: list[torch.Tensor],
        text_offsets: list[int],
        history_state_indices: list[int],
        past_text_lens: list[int],
        trace_requested: bool,
    ) -> bool:
        batched_prefill_history_requested = _env_truthy(
            "VLLM_HSD_ENABLE_BATCHED_PAGED_PREFILL_HISTORY_READ"
        ) or _env_truthy("VLLM_HSD_ENABLE_PAGED_PREFILL_HISTORY_READ")
        allow_paged_prefill_history_read = self._allow_paged_prefill_history_read()
        if batched_prefill_history_requested and not allow_paged_prefill_history_read:
            self._record_hsd_varlen_event(
                "moe_reference_prefill_history_read_dispatch",
                requests=len(history_state_indices),
                text_tokens=int(text_offsets[-1]) if text_offsets else 0,
            )
        batched_prefill_history_enabled = (
            batched_prefill_history_requested and allow_paged_prefill_history_read
        )
        batched_prefill_history_enabled = (
            batched_prefill_history_enabled
            and not _env_truthy("VLLM_HSD_DISABLE_BATCHED_PAGED_PREFILL_HISTORY_READ")
        )
        if (
            not batched_prefill_history_enabled
            or trace_requested
            or not self.hsd_layer_uses_global_read
            or self.hsd_history_out_rmsnorm_weight is not None
            or not self._using_scheduler_history_cache()
            or router_q is None
            or anchor_keys is None
        ):
            return False
        static_history_read = backends._load_hsd_paged_static_history_read()
        page_cache = self._get_hsd_anchor_page_cache()
        if static_history_read is None or page_cache is None:
            return False

        req_count = len(history_state_indices)
        if (
            len(text_offsets) != req_count + 1
            or len(anchor_offsets) != req_count + 1
            or len(anchor_boundaries) != req_count
            or len(past_text_lens) != req_count
        ):
            return False
        for state_idx in set(history_state_indices):
            if not self._migrate_legacy_anchor_cache_to_pages(state_idx):
                return False

        anchor_counts = [
            anchor_offsets[idx + 1] - anchor_offsets[idx] for idx in range(req_count)
        ]
        counts_by_state: dict[int, int] = {}
        for state_idx, count in zip(history_state_indices, anchor_counts, strict=True):
            counts_by_state[state_idx] = counts_by_state.get(state_idx, 0) + count

        required_state_rows = max(history_state_indices, default=-1) + 1
        required_page_cols = (
            max(
                (page_cache.length(state_idx) + count + page_cache.page_m - 1)
                // page_cache.page_m
                for state_idx, count in counts_by_state.items()
            )
            if counts_by_state
            else 0
        )
        metadata_rows = (
            0
            if page_cache._lengths_by_state is None
            else int(page_cache._lengths_by_state.shape[0])
        )
        metadata_cols = (
            0
            if page_cache._page_table_by_state is None
            else int(page_cache._page_table_by_state.shape[1])
        )
        if metadata_rows < required_state_rows or metadata_cols < required_page_cols:
            if _env_truthy("VLLM_HSD_ENABLE_GRAPH_V3"):
                self._record_hsd_varlen_event(
                    "paged_prefill_history_read_fallback",
                    reason="graph_metadata_capacity",
                    requests=req_count,
                    required_state_rows=required_state_rows,
                    metadata_rows=metadata_rows,
                    required_page_cols=required_page_cols,
                    metadata_cols=metadata_cols,
                )
                return False
            metadata_device = page_cache._metadata_device(anchor_states.device)
            if required_state_rows:
                page_cache._ensure_metadata_state_capacity(
                    required_state_rows - 1,
                    device=metadata_device,
                )
            page_cache._ensure_metadata_page_capacity(
                required_page_cols,
                device=metadata_device,
            )
            self._record_hsd_varlen_event(
                "paged_prefill_history_metadata_initialized",
                requests=req_count,
                state_rows=required_state_rows,
                page_cols=required_page_cols,
            )

        if page_cache.fixed_capacity:
            pages_needed = sum(
                page_cache._pages_needed_for_append(state_idx, count)
                for state_idx, count in counts_by_state.items()
            )
            if pages_needed > len(page_cache.free_pages):
                self._record_hsd_varlen_event(
                    "paged_prefill_history_read_fallback",
                    reason="insufficient_pages",
                    requests=req_count,
                    pages_needed=pages_needed,
                    pages_free=len(page_cache.free_pages),
                )
                return False

        total_anchors = int(anchor_offsets[-1])
        packed_boundaries = (
            torch.cat(anchor_boundaries, dim=0).to(
                device=anchor_states.device,
                dtype=torch.long,
            )
            if total_anchors
            else torch.empty(
                0,
                device=anchor_states.device,
                dtype=torch.long,
            )
        )
        paged_anchor_states = self._anchor_states_to_paged_cache(anchor_states)
        batched_append = page_cache.try_append_batch(
            history_state_indices,
            paged_anchor_states,
            anchor_keys,
            packed_boundaries,
            anchor_offsets,
        )
        if batched_append:
            self._record_hsd_varlen_event(
                "batched_paged_prefill_anchor_append_hit",
                requests=req_count,
                anchor_tokens=total_anchors,
            )
        else:
            for req_idx, state_idx in enumerate(history_state_indices):
                anchor_begin = anchor_offsets[req_idx]
                anchor_end = anchor_offsets[req_idx + 1]
                appended = page_cache.try_append(
                    state_idx,
                    paged_anchor_states[anchor_begin:anchor_end]
                    .unsqueeze(0)
                    .contiguous(),
                    anchor_keys[anchor_begin:anchor_end].unsqueeze(0).contiguous(),
                    anchor_boundaries[req_idx],
                )
                if not appended:
                    raise RuntimeError(
                        "HSD batched paged prefill history append failed after "
                        "capacity preflight."
                    )

        total_text = int(text_offsets[-1])
        text_lengths = [
            text_offsets[idx + 1] - text_offsets[idx] for idx in range(req_count)
        ]
        text_lens = torch.tensor(
            text_lengths,
            device=text_o.device,
            dtype=torch.long,
        )
        req_state_indices = torch.tensor(
            history_state_indices,
            device=text_o.device,
            dtype=torch.long,
        )
        req_past_lens = torch.tensor(
            past_text_lens,
            device=text_o.device,
            dtype=torch.long,
        )
        req_text_offsets = torch.tensor(
            text_offsets[:-1],
            device=text_o.device,
            dtype=torch.long,
        )
        token_state_indices = torch.repeat_interleave(
            req_state_indices,
            text_lens,
        )
        token_past_lens = torch.repeat_interleave(req_past_lens, text_lens)
        token_req_offsets = torch.repeat_interleave(
            req_text_offsets,
            text_lens,
        )
        token_start_indices = (
            token_past_lens
            + torch.arange(total_text, device=text_o.device, dtype=torch.long)
            - token_req_offsets
        )
        token_valid_mask = torch.ones(
            total_text,
            device=text_o.device,
            dtype=torch.bool,
        )
        route_scale = float(
            (self.hsd_route_head_dim**-0.5 if self.hsd_use_route_scale else 1.0)
            * self.hsd_route_logit_scale
        )
        read_scale = float(self.head_k_dim**-0.5)
        history_scale = self._history_scale_tensor(text_o.device)
        compare_paged_prefill = _env_truthy(
            "VLLM_HSD_COMPARE_PAGED_PREFILL_HISTORY_READ"
        )
        reference_text_o = text_o.detach().clone() if compare_paged_prefill else None
        blocked_history_applied = False
        max_history_len = max(
            (page_cache.length(state_idx) for state_idx in history_state_indices),
            default=0,
        )
        blocked_prefill_history_enabled = (
            _env_truthy("VLLM_HSD_ENABLE_BLOCKED_PREFILL_HISTORY_READ")
            or _env_truthy("VLLM_HSD_ENABLE_PAGED_PREFILL_HISTORY_READ")
        ) and not _env_truthy("VLLM_HSD_DISABLE_BLOCKED_PREFILL_HISTORY_READ")
        if blocked_prefill_history_enabled and max_history_len <= 4:
            blocked_history_read = (
                backends._load_hsd_paged_blocked_prefill_history_read()
            )
            blocked_result = blocked_history_read(
                cache=page_cache,
                read_q=query,
                router_q=router_q,
                token_state_indices=token_state_indices,
                request_state_indices=req_state_indices,
                text_offsets=torch.tensor(
                    text_offsets,
                    device=text_o.device,
                    dtype=torch.long,
                ),
                start_indices=token_start_indices,
                valid_mask=token_valid_mask,
                max_text_len=max(text_lengths, default=0),
                route_scale=route_scale,
                read_scale=read_scale,
                base_out=text_o,
                history_scale=history_scale,
                null_logits=null_logits,
                block_t=int(
                    os.environ.get(
                        "HSD_PAGED_PREFILL_HISTORY_BLOCK_T",
                        "4",
                    )
                ),
                value_block_size=int(
                    os.environ.get(
                        "HSD_PAGED_PREFILL_HISTORY_VALUE_BLOCK_SIZE",
                        "32",
                    )
                ),
                persistent_memory_enabled=(
                    self.hsd_persistent_memory_mode != "removed"
                ),
                memory_metrics=self._paged_persistent_memory_metric_sink(query),
            )
            blocked_history_applied = blocked_result is not None
            if blocked_history_applied:
                self._record_hsd_varlen_event(
                    "blocked_paged_prefill_history_read_hit",
                    requests=req_count,
                    text_tokens=total_text,
                    anchor_tokens=sum(anchor_counts),
                    block_t=int(
                        os.environ.get(
                            "HSD_PAGED_PREFILL_HISTORY_BLOCK_T",
                            "4",
                        )
                    ),
                )
        if not blocked_history_applied:
            locality_text_offsets = (
                torch.tensor(
                    text_offsets,
                    device=text_o.device,
                    dtype=torch.long,
                )
                if os.environ.get("VLLM_HSD_PREFILL_TOP4_LOCALITY_FILE", "").strip()
                else None
            )
            static_history_read(
                cache=page_cache,
                read_q=query,
                router_q=router_q,
                state_indices=token_state_indices,
                start_indices=token_start_indices,
                valid_mask=token_valid_mask,
                route_scale=route_scale,
                read_scale=read_scale,
                base_out=text_o,
                history_scale=history_scale,
                null_logits=null_logits,
                read_q_is_raw=False,
                normalize_router_q=False,
                value_block_size=int(
                    os.environ.get(
                        "HSD_PAGED_PREFILL_HISTORY_VALUE_BLOCK_SIZE",
                        "32",
                    )
                ),
                locality_text_offsets=locality_text_offsets,
                locality_tag=str(self.prefix),
                locality_max_history_len=max_history_len,
                persistent_memory_enabled=(
                    self.hsd_persistent_memory_mode != "removed"
                ),
                memory_metrics=self._paged_persistent_memory_metric_sink(query),
            )
        if reference_text_o is not None:
            for req_idx, state_idx in enumerate(history_state_indices):
                text_begin = text_offsets[req_idx]
                text_end = text_offsets[req_idx + 1]
                materialized = self._materialize_anchor_page_history(
                    page_cache,
                    state_idx,
                    device=text_o.device,
                )
                if materialized is None:
                    continue
                cached_states, cached_keys, cached_boundaries = materialized
                req_null_logits = None
                if null_logits is not None:
                    req_null_logits = null_logits[text_begin:text_end].view(
                        1,
                        text_end - text_begin,
                        self.num_v_heads // self.tp_size,
                        1,
                    )
                reference_history = self._history_read(
                    read_q=query[text_begin:text_end].unsqueeze(0),
                    router_q=router_q[text_begin:text_end].unsqueeze(0),
                    states=cached_states,
                    anchor_k=cached_keys,
                    boundary_counts=cached_boundaries,
                    text_len=text_end - text_begin,
                    start_index=past_text_lens[req_idx],
                    null_logits=req_null_logits,
                ).squeeze(0)
                reference_text_o[text_begin:text_end].add_(
                    (
                        reference_history
                        * history_scale.to(dtype=reference_history.dtype)
                    ).to(reference_text_o.dtype)
                )
            paged_prefill_diff = (text_o.float() - reference_text_o.float()).abs()
            self._record_hsd_varlen_event(
                "paged_prefill_history_reference_compare",
                requests=req_count,
                text_tokens=total_text,
                max_abs=float(paged_prefill_diff.max().item()),
                mean_abs=float(paged_prefill_diff.mean().item()),
                differing=int(torch.count_nonzero(paged_prefill_diff).item()),
            )
            if _env_truthy("VLLM_HSD_USE_REFERENCE_PAGED_PREFILL_HISTORY_READ"):
                text_o.copy_(reference_text_o)
                self._record_hsd_varlen_event(
                    "paged_prefill_history_reference_hit",
                    requests=req_count,
                    text_tokens=total_text,
                )
        self._record_hsd_varlen_event(
            "batched_paged_prefill_history_read_hit",
            requests=req_count,
            text_tokens=total_text,
            anchor_tokens=sum(anchor_counts),
        )
        return True
