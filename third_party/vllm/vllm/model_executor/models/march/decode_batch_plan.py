# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission, routing/history selection and original text-state restoration.

Methods preserve arithmetic, checks and cache-write order. No model parameters
or request-state lifecycle are owned by this mixin.
"""

from __future__ import annotations

import os
import time
from typing import Literal

import torch

from vllm.forward_context import get_forward_context

from . import backends
from .decode_batch_state import (
    DecodeBatch,
    DecodeHistorySelection,
    DecodeObservers,
    DecodePolicy,
    PreparedDecode,
)
from .diagnostics import _hsd_decode_trace_requested, _hsd_trace_requested
from .layout import HistoricalHSDLayout, _to_cpu_int_list
from .runtime import (
    _compact_all_layers_fast_path_enabled,
    _env_truthy,
    _hsd_history_available,
    _is_valid_hsd_state_index,
)


class BatchDecodePlanMixin:
    def _prepare_hsd_batched_decode(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        layout: HistoricalHSDLayout,
    ) -> PreparedDecode | Literal[False]:

        def fallback(reason: str) -> bool:
            self._record_hsd_decode_prepare_fallback(reason)
            return False

        if self.hsd_decode_prepare_backend != "batched":
            return False
        if torch.is_grad_enabled():
            return fallback("grad_enabled")
        if not hidden_states.is_cuda:
            return fallback("not_cuda")
        if not self._layout_is_cached_decode(layout):
            return fallback("not_cached_decode")
        if self._has_spec_decode_metadata():
            return fallback("spec_decode")

        anchor_token_num = int(getattr(self.config, "anchor_token_num", 0) or 0)
        if anchor_token_num != 1:
            return fallback("anchor_token_num_not_one")
        anchor_chunk_size = int(getattr(self.config, "anchor_chunk_size", 0) or 0)
        if anchor_chunk_size <= 0:
            return fallback("bad_anchor_chunk_size")

        conv_cache = self._conv_cache_for_kernel()
        if conv_cache is None:
            return fallback("missing_conv_cache")
        ssm_state = self._ssm_cache_for_kernel()
        if ssm_state is None:
            return fallback("missing_ssm_cache")

        local_v_heads = self.num_v_heads // self.tp_size
        local_k_heads = self.num_k_heads // self.tp_size
        local_key_dim = self.key_dim // self.tp_size
        local_value_dim = self.value_dim // self.tp_size
        qkv_size = local_key_dim * 2 + local_value_dim
        z_size = local_value_dim

        req_count = len(layout.req_expanded_starts)
        if req_count <= 0:
            return fallback("empty_layout")
        if int(layout.text_to_expanded.numel()) != req_count:
            return fallback("text_index_count_mismatch")
        padded_static_decode = (
            bool(layout.padded_static_decode)
            and layout.decode_valid_mask is not None
            and _env_truthy("VLLM_HSD_ENABLE_PADDED_STATIC_DECODE")
        )
        decode_valid_mask = layout.decode_valid_mask if padded_static_decode else None
        decode_valid_flags = layout.decode_valid_flags if padded_static_decode else None
        use_static_sidecar = (
            padded_static_decode and self._use_hsd_static_text_state_sidecar()
        )
        if padded_static_decode:
            assert decode_valid_mask is not None
            if decode_valid_flags is None or len(decode_valid_flags) != req_count:
                return fallback("bad_decode_valid_flags")
            if int(decode_valid_mask.numel()) != req_count:
                return fallback("bad_decode_valid_mask")
            if not _env_truthy("VLLM_HSD_ENABLE_UNSAFE_PADDED_STATIC_DECODE"):
                return fallback("padded_static_text_state_disabled")
            if (
                _env_truthy("VLLM_HSD_PADDED_STATIC_USE_SIDECAR")
                and not use_static_sidecar
            ):
                return fallback("static_text_state_sidecar_disabled")

        compact_no_anchor_setup = bool(
            self.hsd_use_compact_state_slots
            and not padded_static_decode
            and layout.decode_only_no_anchor_value
            and layout.state_indices is not None
            and layout.past_text_lens_tensor is not None
        )
        if compact_no_anchor_setup:
            anchor_offsets = [0] * (req_count + 1)
            state_indices = layout.state_indices_list
            past_text_lens = layout.past_text_lens
            boundary_counts_by_req: list[torch.Tensor] | None = None
        else:
            anchor_offsets = [0]
            state_indices: list[int] = []
            past_text_lens: list[int] = []
            boundary_counts_by_req = []
            for req_idx, expanded_len in enumerate(layout.req_expanded_lens):
                is_padding_req = False
                if decode_valid_flags is not None:
                    is_padding_req = not decode_valid_flags[req_idx]
                text_len = int(layout.req_text_lens[req_idx])
                anchor_len = int(layout.req_anchor_lens[req_idx])
                if text_len != 1:
                    return fallback("text_len_not_one")
                if int(expanded_len) != text_len + anchor_len:
                    return fallback("layout_len_mismatch")

                past_text_len = int(layout.past_text_lens[req_idx])
                if past_text_len <= 0:
                    return fallback("prefill_request")
                expected_anchor_len = (
                    past_text_len + 1
                ) // anchor_chunk_size - past_text_len // anchor_chunk_size
                if anchor_len != expected_anchor_len:
                    return fallback("anchor_boundary_count_not_regular")

                req_anchor_start = layout.req_anchor_starts[req_idx]
                boundary_counts = layout.anchor_boundaries[
                    req_anchor_start : req_anchor_start + anchor_len
                ]
                if anchor_len:
                    expected_boundary = torch.full_like(
                        boundary_counts, past_text_len + 1
                    )
                    if not torch.equal(boundary_counts, expected_boundary):
                        return fallback("anchor_boundaries_not_current_token")

                state_idx = self._state_index_for_request(layout, req_idx)
                if is_padding_req:
                    if int(state_idx) != 0:
                        return fallback("invalid_padding_state")
                elif not _is_valid_hsd_state_index(state_idx):
                    return fallback("null_state_idx")
                state_indices.append(state_idx)
                past_text_lens.append(past_text_len)
                boundary_counts_by_req.append(boundary_counts)
                anchor_offsets.append(anchor_offsets[-1] + anchor_len)

        if (
            padded_static_decode
            and _env_truthy("VLLM_HSD_DISABLE_PADDED_STATIC_HISTORY_READ")
            and any(
                _hsd_history_available(
                    self.config,
                    past_text_lens[req_idx],
                    anchor_chunk_size,
                )
                for req_idx in range(req_count)
                if decode_valid_mask is None or bool(decode_valid_flags[req_idx])
            )
        ):
            return fallback("history_read_static_disabled")

        if self._use_hsd_persistent_state_sidecar():
            required_rows = (
                self.hsd_persistent_state_sidecar_rows
                if compact_no_anchor_setup
                else max(state_indices) + 1
            )
            conv_cache = self._conv_cache_for_kernel(required_rows)
            ssm_state = self._ssm_cache_for_kernel(required_rows)
            if conv_cache is None or ssm_state is None:
                return fallback("missing_persistent_state_sidecar")
        if not compact_no_anchor_setup:
            if max(state_indices) >= conv_cache.shape[0]:
                return fallback("conv_state_idx_oob")
            if max(state_indices) >= ssm_state.shape[0]:
                return fallback("ssm_state_idx_oob")

        total_anchor = anchor_offsets[-1]
        anchor_positions = layout.anchor_positions
        if int(anchor_positions.numel()) != total_anchor:
            return fallback("anchor_index_count_mismatch")

        trace_requested = _hsd_trace_requested() or _hsd_decode_trace_requested()
        decode_timing_enabled = _env_truthy("VLLM_HSD_DECODE_TIMING")
        decode_timing_sync = _env_truthy("VLLM_HSD_DECODE_TIMING_SYNC")
        decode_timing_last = time.perf_counter()
        decode_fine_timing_enabled = decode_timing_enabled and _env_truthy(
            "VLLM_HSD_DECODE_TIMING_FINE"
        )
        decode_fine_timing_last = decode_timing_last
        decode_timing_payload: dict[str, int] = {}

        def mark_decode_timing(name: str) -> None:
            nonlocal decode_timing_last
            if not decode_timing_enabled:
                return
            if decode_timing_sync:
                torch.cuda.synchronize(hidden_states.device)
            now = time.perf_counter()
            decode_timing_payload[f"{name}_us"] = int(
                max(0.0, now - decode_timing_last) * 1_000_000
            )
            decode_timing_last = now

        def mark_decode_fine_timing(name: str) -> None:
            nonlocal decode_fine_timing_last
            if not decode_fine_timing_enabled:
                return
            if decode_timing_sync:
                torch.cuda.synchronize(hidden_states.device)
            now = time.perf_counter()
            decode_timing_payload[f"{name}_us"] = int(
                max(0.0, now - decode_fine_timing_last) * 1_000_000
            )
            decode_fine_timing_last = now

        sync_stage_config = (
            os.environ.get("VLLM_HSD_DECODE_SYNC_STAGES", "").strip().lower()
        )
        sync_decode_stages = {
            item.strip() for item in sync_stage_config.split(",") if item.strip()
        }

        def maybe_sync_decode_stage(name: str) -> None:
            if not sync_decode_stages:
                return
            if "all" not in sync_decode_stages and name not in sync_decode_stages:
                return
            torch.cuda.synchronize(hidden_states.device)

        graphable_decode_requested = _env_truthy("VLLM_HSD_ENABLE_GRAPHABLE_DECODE")
        no_anchor_base_fast_path = (
            self.hsd_layer_uses_global_read
            and not trace_requested
            and _env_truthy("VLLM_HSD_ENABLE_NO_ANCHOR_FAST_PATH")
            and not _env_truthy("VLLM_HSD_DISABLE_NO_ANCHOR_FAST_PATH")
        )
        no_anchor_multi_allowed = req_count == 1 or _env_truthy(
            "VLLM_HSD_ENABLE_UNSAFE_NO_ANCHOR_MULTI"
        )
        no_anchor_fast_path = no_anchor_base_fast_path and no_anchor_multi_allowed
        if no_anchor_base_fast_path and not no_anchor_multi_allowed:
            self._record_hsd_varlen_event(
                "hsd_no_anchor_fast_path_fallback",
                reason="multi_request_no_anchor_disabled",
                requests=req_count,
            )
        no_anchor_read_only_enabled = no_anchor_fast_path and _env_truthy(
            "VLLM_HSD_ENABLE_NO_ANCHOR_READ_ONLY"
        )
        if compact_no_anchor_setup:
            anchor_counts_by_req = [0] * req_count
            anchor_history_by_req = (
                layout.decode_history_valid_flags or [False] * req_count
            )
            global_read_needed_by_req = (
                anchor_history_by_req
                if self.hsd_layer_uses_global_read
                else [False] * req_count
            )
            no_anchor_requests = req_count
            read_only_requests = (
                int(layout.decode_history_valid_count_value or 0)
                if self.hsd_layer_uses_global_read and no_anchor_read_only_enabled
                else 0
            )
            no_history_skip_requests = (
                req_count - int(layout.decode_history_valid_count_value or 0)
                if self.hsd_layer_uses_global_read
                else 0
            )
        else:
            anchor_counts_by_req = [
                anchor_offsets[req_idx + 1] - anchor_offsets[req_idx]
                for req_idx in range(req_count)
            ]
            anchor_history_by_req = (
                [False] * req_count
                if padded_static_decode
                else [
                    _hsd_history_available(
                        self.config,
                        past_text_lens[req_idx],
                        anchor_chunk_size,
                    )
                    if no_anchor_fast_path
                    else False
                    for req_idx in range(req_count)
                ]
            )
            global_read_needed_by_req = []
            no_anchor_requests = 0
            read_only_requests = 0
            no_history_skip_requests = 0
            for req_idx, anchor_count in enumerate(anchor_counts_by_req):
                if anchor_count == 0:
                    no_anchor_requests += 1
                if not self.hsd_layer_uses_global_read:
                    global_read_needed_by_req.append(False)
                elif padded_static_decode:
                    valid_req = decode_valid_flags is None or bool(
                        decode_valid_flags[req_idx]
                    )
                    has_anchor_history = valid_req and _hsd_history_available(
                        self.config,
                        past_text_lens[req_idx],
                        anchor_chunk_size,
                    )
                    global_read_needed_by_req.append(has_anchor_history)
                elif not no_anchor_fast_path:
                    global_read_needed_by_req.append(True)
                elif anchor_count > 0:
                    global_read_needed_by_req.append(True)
                elif anchor_history_by_req[req_idx]:
                    global_read_needed_by_req.append(True)
                    if no_anchor_read_only_enabled:
                        read_only_requests += 1
                else:
                    global_read_needed_by_req.append(False)
                    no_history_skip_requests += 1
        needs_router_q = self.hsd_layer_uses_global_read and any(
            global_read_needed_by_req
        )

        device = hidden_states.device
        if layout.state_indices is not None:
            state_indices_tensor = layout.state_indices.to(
                device=device,
                dtype=torch.int32,
            )
        else:
            state_indices_tensor = torch.tensor(
                state_indices,
                dtype=torch.int32,
                device=device,
            )
        if layout.history_state_indices is not None:
            history_state_indices_tensor = layout.history_state_indices.to(
                device=device,
                dtype=torch.int32,
            )
            if int(history_state_indices_tensor.numel()) != req_count:
                return fallback("bad_history_state_indices")
        elif padded_static_decode or compact_no_anchor_setup:
            history_state_indices_tensor = None
        else:
            history_state_indices_tensor = state_indices_tensor
        if layout.past_text_lens_tensor is not None:
            past_text_lens_tensor = layout.past_text_lens_tensor
        else:
            past_text_lens_tensor = torch.tensor(
                past_text_lens,
                dtype=torch.long,
                device=device,
            )
        common_parity_debug = _env_truthy("VLLM_HSD_COMMON_PARITY_DEBUG")
        common_parity_debug_layers = {
            int(item)
            for item in os.environ.get(
                "VLLM_HSD_COMMON_PARITY_DEBUG_LAYERS", "0"
            ).split(",")
            if item.strip()
        }

        def record_common_parity_debug(
            stage: str,
            tensor: torch.Tensor | None,
        ) -> None:
            if (
                not common_parity_debug
                or self.layer_idx not in common_parity_debug_layers
                or tensor is None
                or tensor.numel() == 0
            ):
                return
            values = tensor.detach().float().reshape(-1)
            self._record_hsd_varlen_event(
                "hsd_common_parity_debug",
                layer=int(self.layer_idx),
                stage=stage,
                past_text_len_min=min(past_text_lens),
                past_text_len_max=max(past_text_lens),
                numel=int(values.numel()),
                mean=float(values.mean().item()),
                l2=float(torch.linalg.vector_norm(values).item()),
                max_abs=float(values.abs().max().item()),
                sample=values[:8].cpu().tolist(),
            )

        page_cache_for_common = getattr(self, "hsd_anchor_page_cache", None)
        common_history_candidate_indices: list[int] = []
        common_history_candidate_tensor_prepared: torch.Tensor | None = None
        common_history_candidate_count = 0
        common_history_valid_bools: list[bool] | None = None
        common_history_valid_tensor: torch.Tensor | None = None
        common_history_valid_count = 0
        common_history_use_full_batch = False
        common_router_project_full_batch = False
        common_history_candidate_state_key: tuple[int, ...] | None = None
        max_history_batch: int | None = None
        no_anchor_common_fallback_reason: str | None = None

        def record_no_anchor_common_fallback(reason: str, **payload) -> None:
            clean_payload = {
                key: value for key, value in payload.items() if value is not None
            }
            self._record_hsd_varlen_event(
                "hsd_no_anchor_common_fallback",
                reason=reason,
                requests=req_count,
                anchor_tokens=total_anchor,
                padded_static_decode=int(padded_static_decode),
                **clean_payload,
            )

        no_anchor_common_base_fast_path = (
            total_anchor == 0
            and not trace_requested
            and (
                padded_static_decode
                or self.hsd_layer_uses_global_read
                or _env_truthy("VLLM_HSD_ENABLE_NO_ANCHOR_COMMON_ALL_LAYERS")
                or (
                    compact_no_anchor_setup
                    and _compact_all_layers_fast_path_enabled(req_count)
                )
            )
            and self.hsd_decode_backend == "packed"
            and _env_truthy("VLLM_HSD_ENABLE_NO_ANCHOR_COMMON_FAST_PATH")
            and not _env_truthy("VLLM_HSD_DISABLE_BATCHED_RECURRENT_DECODE")
            and not _env_truthy("VLLM_HSD_DISABLE_NO_ANCHOR_COMMON_FAST_PATH")
        )
        no_anchor_common_multi_allowed = req_count == 1 or _env_truthy(
            "VLLM_HSD_ENABLE_UNSAFE_NO_ANCHOR_COMMON_MULTI"
        )
        no_anchor_common_fast_path = (
            no_anchor_common_base_fast_path and no_anchor_common_multi_allowed
        )
        if no_anchor_common_base_fast_path and not no_anchor_common_multi_allowed:
            no_anchor_common_fallback_reason = "multi_request_common_disabled"
            record_no_anchor_common_fallback(
                no_anchor_common_fallback_reason,
                unsafe_multi_env=0,
            )
        if (
            no_anchor_common_fast_path
            and no_history_skip_requests > 0
            and _env_truthy("VLLM_HSD_DISABLE_NO_ANCHOR_COMMON_NO_HISTORY")
        ):
            no_anchor_common_fast_path = False
            no_anchor_common_fallback_reason = "no_history_requests_disabled"
            record_no_anchor_common_fallback(
                no_anchor_common_fallback_reason,
                no_history_skip_requests=no_history_skip_requests,
            )
        if no_anchor_common_fast_path and self.hsd_layer_uses_global_read:
            backend = (
                str(getattr(self, "hsd_history_read_backend", "auto")).strip().lower()
            )
            if not any(global_read_needed_by_req):
                common_history_candidate_indices = []
                common_history_candidate_tensor_prepared = None
                common_history_candidate_count = 0
            elif (
                backend == "loop"
                or page_cache_for_common is None
                or not getattr(page_cache_for_common, "fixed_capacity", False)
                or backends._load_hsd_paged_history_read() is None
            ):
                no_anchor_common_fast_path = False
                no_anchor_common_fallback_reason = "missing_paged_cache"
                record_no_anchor_common_fallback(
                    no_anchor_common_fallback_reason,
                    backend=backend,
                    has_page_cache=page_cache_for_common is not None,
                    fixed_capacity=int(
                        bool(
                            page_cache_for_common is not None
                            and getattr(page_cache_for_common, "fixed_capacity", False)
                        )
                    ),
                )
            elif history_state_indices_tensor is None and any(
                global_read_needed_by_req
            ):
                no_anchor_common_fast_path = False
                no_anchor_common_fallback_reason = "history_state_indices_missing"
                record_no_anchor_common_fallback(
                    no_anchor_common_fallback_reason,
                    needed_requests=sum(
                        1 for needed in global_read_needed_by_req if needed
                    ),
                )
            else:
                history_lookup_state_indices = (
                    history_state_indices_tensor
                    if history_state_indices_tensor is not None
                    else state_indices_tensor
                )
                fast_history_setup = _env_truthy("VLLM_HSD_COMMON_FAST_HISTORY_SETUP")
                if fast_history_setup:
                    if (
                        total_anchor == 0
                        and layout.decode_history_valid_flags is not None
                        and layout.decode_history_valid_mask is not None
                        and layout.decode_history_valid_count_value is not None
                    ):
                        common_history_valid_bools = layout.decode_history_valid_flags
                        common_history_valid_tensor = layout.decode_history_valid_mask
                        common_history_valid_count = int(
                            layout.decode_history_valid_count_value
                        )
                    else:
                        common_history_valid_bools = [
                            bool(global_read_needed_by_req[req_idx])
                            and _hsd_history_available(
                                self.config,
                                past_text_lens[req_idx],
                                anchor_chunk_size,
                            )
                            for req_idx in range(req_count)
                        ]
                        common_history_valid_count = sum(
                            1 for valid in common_history_valid_bools if valid
                        )
                    common_history_use_full_batch = bool(
                        common_history_valid_count
                        and (
                            padded_static_decode
                            or (
                                compact_no_anchor_setup
                                and not _env_truthy(
                                    "VLLM_HSD_DISABLE_COMPACT_FULL_BATCH_HISTORY_READ"
                                )
                            )
                            or _env_truthy("VLLM_HSD_ENABLE_FULL_BATCH_HISTORY_READ")
                        )
                    )
                    if common_history_use_full_batch:
                        common_history_candidate_tensor_prepared = (
                            layout.text_to_expanded
                            if padded_static_decode or compact_no_anchor_setup
                            else torch.arange(
                                req_count, dtype=torch.long, device=device
                            )
                        )
                        if common_history_valid_tensor is None:
                            common_history_valid_tensor = torch.tensor(
                                common_history_valid_bools,
                                dtype=torch.bool,
                                device=device,
                            )
                    else:
                        if layout.decode_history_candidate_indices is not None:
                            common_history_candidate_tensor_prepared = (
                                layout.decode_history_candidate_indices
                            )
                            common_history_candidate_indices = (
                                layout.decode_history_candidate_indices_list or []
                            )
                        else:
                            common_history_candidate_indices = [
                                req_idx
                                for req_idx, valid in enumerate(
                                    common_history_valid_bools
                                )
                                if valid
                            ]
                            common_history_candidate_tensor_prepared = torch.tensor(
                                common_history_candidate_indices,
                                dtype=torch.long,
                                device=device,
                            )
                        common_history_valid_tensor = None
                    common_history_candidate_count = int(
                        common_history_candidate_tensor_prepared.numel()
                    )
                    max_history_len = (
                        int(layout.decode_history_max_anchor_len_value)
                        if layout.decode_history_max_anchor_len_value is not None
                        else max(
                            (
                                int(past_text_lens[req_idx]) // anchor_chunk_size
                                + int(
                                    self.hsd_persistent_memory_mode != "removed"
                                    and int(past_text_lens[req_idx]) > 0
                                )
                                for req_idx, valid in enumerate(
                                    common_history_valid_bools
                                )
                                if valid
                            ),
                            default=0,
                        )
                    )
                    default_history_batch = (
                        str(max(16, req_count)) if padded_static_decode else "16"
                    )
                    max_history_batch = int(
                        os.environ.get(
                            "HSD_PAGED_HISTORY_MAX_BATCH",
                            default_history_batch,
                        )
                    )
                    max_anchor_env = os.environ.get(
                        "HSD_PAGED_HISTORY_MAX_ANCHOR_LEN", "2048"
                    ).strip()
                    max_anchor_len = (
                        None
                        if not max_anchor_env or int(max_anchor_env) <= 0
                        else int(max_anchor_env)
                    )
                    if max_history_batch <= 0:
                        no_anchor_common_fast_path = False
                        no_anchor_common_fallback_reason = (
                            "bad_paged_history_batch_limit"
                        )
                        record_no_anchor_common_fallback(
                            no_anchor_common_fallback_reason,
                            max_history_batch=max_history_batch,
                        )
                        common_history_candidate_indices = []
                        common_history_candidate_tensor_prepared = None
                        common_history_candidate_count = 0
                    elif common_history_candidate_count > max_history_batch:
                        self._record_hsd_varlen_event(
                            "paged_history_read_split",
                            history_candidate_requests=common_history_candidate_count,
                            max_history_batch=max_history_batch,
                            max_history_len=max_history_len,
                            padded_static_decode=int(padded_static_decode),
                        )
                    elif (
                        max_anchor_len is not None and max_history_len > max_anchor_len
                    ):
                        no_anchor_common_fast_path = False
                        no_anchor_common_fallback_reason = "paged_history_anchor_limit"
                        record_no_anchor_common_fallback(
                            no_anchor_common_fallback_reason,
                            history_candidate_requests=common_history_candidate_count,
                            max_anchor_len=max_anchor_len,
                            max_history_len=max_history_len,
                        )
                        common_history_candidate_indices = []
                        common_history_candidate_tensor_prepared = None
                        common_history_candidate_count = 0
                    if no_anchor_common_fast_path:
                        self._record_hsd_varlen_event(
                            "hsd_common_fast_history_setup_hit",
                            requests=req_count,
                            history_candidate_requests=common_history_candidate_count,
                            history_valid_requests=common_history_valid_count,
                            full_batch_history_read=int(common_history_use_full_batch),
                        )
                else:
                    needed_tensor = torch.tensor(
                        global_read_needed_by_req,
                        dtype=torch.bool,
                        device=device,
                    )
                    if _env_truthy("VLLM_HSD_COMMON_TRUST_PAST_LEN_HISTORY"):
                        # Experimental: avoid a redundant page-cache metadata
                        # lookup on every no-anchor decode layer. The paged
                        # read still consults the real page table and returns
                        # zero contribution for states without materialized
                        # history.
                        estimated_lengths = torch.div(
                            past_text_lens_tensor,
                            anchor_chunk_size,
                            rounding_mode="floor",
                        )
                        history_lengths_tensor = torch.where(
                            needed_tensor,
                            estimated_lengths,
                            torch.zeros_like(estimated_lengths),
                        )
                    elif hasattr(page_cache_for_common, "lengths"):
                        history_lengths_tensor = page_cache_for_common.lengths(
                            history_lookup_state_indices,
                            device=device,
                        ).to(device=device, dtype=torch.long)
                    else:
                        history_lookup_state_list = _to_cpu_int_list(
                            history_lookup_state_indices,
                            req_count,
                        )
                        history_lengths = [
                            page_cache_for_common.length(state_idx)
                            for state_idx in history_lookup_state_list
                        ]
                        history_lengths_tensor = torch.tensor(
                            history_lengths,
                            device=device,
                            dtype=torch.long,
                        )
                    missing_history_mask = needed_tensor & (history_lengths_tensor <= 0)
                    missing_history_count = int(missing_history_mask.sum().item())
                    missing_history_req_indices: list[int] = []
                    if missing_history_count:
                        missing_history_req_indices = (
                            missing_history_mask.nonzero(as_tuple=False)
                            .flatten()
                            .detach()
                            .cpu()
                            .tolist()
                        )
                        history_lookup_state_list = _to_cpu_int_list(
                            history_lookup_state_indices,
                            req_count,
                        )
                        legacy_present_requests = 0
                        migrated_requests = 0
                        migrate_failed_requests = 0
                        for req_idx in missing_history_req_indices:
                            state_idx = int(history_lookup_state_list[req_idx])
                            legacy_entry = self._legacy_anchor_history(
                                state_idx,
                                device=device,
                            )
                            if legacy_entry is None:
                                continue
                            legacy_present_requests += 1
                            before_length = int(page_cache_for_common.length(state_idx))
                            try:
                                migrated_ok = (
                                    self._migrate_legacy_anchor_cache_to_pages(
                                        state_idx
                                    )
                                )
                            except Exception as exc:
                                migrated_ok = False
                                self._record_hsd_varlen_event(
                                    "hsd_static_history_legacy_migrate_failed",
                                    state_idx=state_idx,
                                    error=type(exc).__name__,
                                )
                            after_length = int(page_cache_for_common.length(state_idx))
                            if after_length > max(0, before_length):
                                migrated_requests += 1
                            elif not migrated_ok:
                                migrate_failed_requests += 1

                        if legacy_present_requests:
                            if hasattr(page_cache_for_common, "lengths"):
                                history_lengths_tensor = page_cache_for_common.lengths(
                                    history_lookup_state_indices,
                                    device=device,
                                ).to(device=device, dtype=torch.long)
                            else:
                                history_lengths = [
                                    page_cache_for_common.length(state_idx)
                                    for state_idx in history_lookup_state_list
                                ]
                                history_lengths_tensor = torch.tensor(
                                    history_lengths,
                                    device=device,
                                    dtype=torch.long,
                                )
                            still_missing_requests = sum(
                                1
                                for req_idx in missing_history_req_indices
                                if int(history_lengths_tensor[req_idx].item()) <= 0
                            )
                            self._record_hsd_varlen_event(
                                "hsd_static_history_legacy_migrate_hit",
                                requested=len(missing_history_req_indices),
                                legacy_present_requests=legacy_present_requests,
                                migrated_requests=migrated_requests,
                                migrate_failed_requests=migrate_failed_requests,
                                still_missing_requests=still_missing_requests,
                            )
                    common_history_valid_tensor = needed_tensor & (
                        history_lengths_tensor > 0
                    )
                    common_history_valid_count = int(
                        common_history_valid_tensor.sum().item()
                    )
                    common_history_use_full_batch = bool(
                        common_history_valid_count
                        and (
                            padded_static_decode
                            or _env_truthy("VLLM_HSD_ENABLE_FULL_BATCH_HISTORY_READ")
                        )
                    )
                    if common_history_use_full_batch:
                        common_history_candidate_tensor_prepared = torch.arange(
                            req_count,
                            dtype=torch.long,
                            device=device,
                        )
                    else:
                        common_history_candidate_tensor_prepared = (
                            common_history_valid_tensor.nonzero(as_tuple=False)
                            .flatten()
                            .to(dtype=torch.long)
                        )
                    common_history_candidate_count = int(
                        common_history_candidate_tensor_prepared.numel()
                    )
                    if missing_history_req_indices:
                        missing_state_indices = [
                            int(history_lookup_state_list[req_idx])
                            for req_idx in missing_history_req_indices
                        ]
                        missing_past_lens = [
                            int(past_text_lens[req_idx])
                            for req_idx in missing_history_req_indices
                        ]
                        legacy_present_requests = sum(
                            1
                            for state_idx in missing_state_indices
                            if self._legacy_anchor_history(
                                state_idx,
                                device=device,
                            )
                            is not None
                        )
                        self._record_hsd_varlen_event(
                            "hsd_no_anchor_history_missing_zero_read",
                            needed_requests=sum(
                                1 for needed in global_read_needed_by_req if needed
                            ),
                            missing_requests=len(missing_history_req_indices),
                            missing_state_idx_min=(
                                min(missing_state_indices)
                                if missing_state_indices
                                else None
                            ),
                            missing_state_idx_max=(
                                max(missing_state_indices)
                                if missing_state_indices
                                else None
                            ),
                            missing_past_len_min=(
                                min(missing_past_lens) if missing_past_lens else None
                            ),
                            missing_past_len_max=(
                                max(missing_past_lens) if missing_past_lens else None
                            ),
                            legacy_present_requests=legacy_present_requests,
                            padded_static_decode=int(padded_static_decode),
                        )
                    max_history_len = max(
                        int(
                            history_lengths_tensor.masked_select(
                                common_history_valid_tensor
                            )
                            .max()
                            .item()
                        )
                        if common_history_valid_count
                        else 0,
                        0,
                    )
                    default_history_batch = (
                        str(max(16, req_count)) if padded_static_decode else "16"
                    )
                    max_history_batch = int(
                        os.environ.get(
                            "HSD_PAGED_HISTORY_MAX_BATCH",
                            default_history_batch,
                        )
                    )
                    max_anchor_env = os.environ.get(
                        "HSD_PAGED_HISTORY_MAX_ANCHOR_LEN", "2048"
                    ).strip()
                    max_anchor_len = (
                        None
                        if not max_anchor_env or int(max_anchor_env) <= 0
                        else int(max_anchor_env)
                    )
                    if max_history_batch <= 0:
                        no_anchor_common_fast_path = False
                        no_anchor_common_fallback_reason = (
                            "bad_paged_history_batch_limit"
                        )
                        record_no_anchor_common_fallback(
                            no_anchor_common_fallback_reason,
                            max_history_batch=max_history_batch,
                        )
                        common_history_candidate_indices = []
                        common_history_candidate_tensor_prepared = None
                        common_history_candidate_count = 0
                    elif common_history_candidate_count > max_history_batch:
                        self._record_hsd_varlen_event(
                            "paged_history_read_split",
                            history_candidate_requests=common_history_candidate_count,
                            max_history_batch=max_history_batch,
                            max_history_len=max_history_len,
                            padded_static_decode=int(padded_static_decode),
                        )
                    elif (
                        max_anchor_len is not None and max_history_len > max_anchor_len
                    ):
                        no_anchor_common_fast_path = False
                        no_anchor_common_fallback_reason = "paged_history_anchor_limit"
                        record_no_anchor_common_fallback(
                            no_anchor_common_fallback_reason,
                            history_candidate_requests=common_history_candidate_count,
                            max_anchor_len=max_anchor_len,
                            max_history_len=max_history_len,
                        )
                        common_history_candidate_indices = []
                        common_history_candidate_tensor_prepared = None
                        common_history_candidate_count = 0
        elif no_anchor_common_fast_path:
            page_cache_for_common = None
        mark_decode_fine_timing("setup_history")
        if no_anchor_common_fast_path:
            needs_router_q = (
                self.hsd_layer_uses_global_read and common_history_candidate_count > 0
            )
        if graphable_decode_requested:
            reason = None
            if trace_requested:
                reason = "trace"
            elif total_anchor != 0:
                reason = "anchor_boundary"
            elif not no_anchor_common_fast_path:
                reason = no_anchor_common_fallback_reason or (
                    "missing_paged_cache"
                    if self.hsd_layer_uses_global_read
                    else "dynamic_shape"
                )
            if reason is not None:
                expected_eager_dispatch = reason == "anchor_boundary"
                self._record_hsd_varlen_event(
                    (
                        "hsd_static_decode_eager_dispatch"
                        if expected_eager_dispatch
                        else "hsd_static_decode_fallback"
                    ),
                    reason=reason,
                    requests=req_count,
                    anchor_tokens=total_anchor,
                )
                self._record_hsd_varlen_event(
                    (
                        "hsd_graphable_decode_eager_dispatch"
                        if expected_eager_dispatch
                        else "hsd_graphable_decode_fallback"
                    ),
                    reason=reason,
                    requests=req_count,
                    anchor_tokens=total_anchor,
                )
                if padded_static_decode:
                    self._record_hsd_varlen_event(
                        (
                            "hsd_padded_static_decode_eager_dispatch"
                            if expected_eager_dispatch
                            else "hsd_padded_static_decode_fallback"
                        ),
                        reason=reason,
                        requests=req_count,
                        anchor_tokens=total_anchor,
                    )
        if padded_static_decode and not no_anchor_common_fast_path:
            return fallback(
                no_anchor_common_fallback_reason or "static_no_anchor_common_disabled"
            )

        common_history_candidate_tensor = None
        if common_history_candidate_count > 0:
            if common_history_candidate_tensor_prepared is not None:
                common_history_candidate_tensor = (
                    common_history_candidate_tensor_prepared
                )
            else:
                common_history_candidate_tensor = torch.tensor(
                    common_history_candidate_indices,
                    dtype=torch.long,
                    device=device,
                )
            history_state_indices_key = (
                layout.history_state_indices_list
                if layout.history_state_indices_list is not None
                else state_indices
            )
            if common_history_use_full_batch:
                common_history_candidate_state_key = (
                    layout.history_state_indices_key
                    if layout.history_state_indices_key is not None
                    else tuple(int(v) for v in history_state_indices_key)
                )
            elif common_history_candidate_indices:
                common_history_candidate_state_key = tuple(
                    int(history_state_indices_key[req_idx])
                    for req_idx in common_history_candidate_indices
                )
            elif common_history_valid_bools is not None:
                common_history_candidate_state_key = tuple(
                    int(history_state_indices_key[req_idx])
                    for req_idx, valid in enumerate(common_history_valid_bools)
                    if valid
                )
            common_router_project_full_batch = (
                no_anchor_common_fast_path
                and not padded_static_decode
                and req_count > 1
                and not _env_truthy("VLLM_HSD_ENABLE_SUBSET_ROUTER_PROJ")
            )
        mark_decode_fine_timing("setup_candidate_tensor")
        persistent_text_cache = (
            get_forward_context().attn_metadata is not None
            and not _env_truthy("VLLM_HSD_DISABLE_TEXT_STATE_CACHE")
        )
        if persistent_text_cache:
            if use_static_sidecar:
                if not self._restore_hsd_static_text_state_sidecar_batched(
                    state_indices_tensor,
                    decode_valid_mask,
                ):
                    return fallback("static_text_state_sidecar_missing")
            else:
                if not self._restore_hsd_text_state_cache_batched(
                    state_indices_tensor,
                    device,
                ):
                    for state_idx in state_indices:
                        self._restore_hsd_text_state_cache(state_idx, device)
        mark_decode_fine_timing("setup_text_cache_restore")

        return PreparedDecode(
            batch=DecodeBatch(
                hidden_states=hidden_states,
                output=output,
                layout=layout,
                anchor_counts_by_req=anchor_counts_by_req,
                anchor_offsets=anchor_offsets,
                anchor_positions=anchor_positions,
                boundary_counts_by_req=boundary_counts_by_req,
                conv_cache=conv_cache,
                decode_valid_mask=decode_valid_mask,
                device=device,
                history_state_indices_tensor=history_state_indices_tensor,
                local_k_heads=local_k_heads,
                local_key_dim=local_key_dim,
                local_v_heads=local_v_heads,
                local_value_dim=local_value_dim,
                padded_static_decode=padded_static_decode,
                past_text_lens=past_text_lens,
                past_text_lens_tensor=past_text_lens_tensor,
                qkv_size=qkv_size,
                req_count=req_count,
                ssm_state=ssm_state,
                state_indices=state_indices,
                state_indices_tensor=state_indices_tensor,
                total_anchor=total_anchor,
                z_size=z_size,
            ),
            policy=DecodePolicy(
                global_read_needed_by_req=global_read_needed_by_req,
                graphable_decode_requested=graphable_decode_requested,
                needs_router_q=needs_router_q,
                no_anchor_common_fast_path=no_anchor_common_fast_path,
                no_anchor_fast_path=no_anchor_fast_path,
                no_anchor_read_only_enabled=no_anchor_read_only_enabled,
                no_anchor_requests=no_anchor_requests,
                no_history_skip_requests=no_history_skip_requests,
                persistent_text_cache=persistent_text_cache,
                read_only_requests=read_only_requests,
                trace_requested=trace_requested,
                use_static_sidecar=use_static_sidecar,
            ),
            history=DecodeHistorySelection(
                common_history_candidate_count=common_history_candidate_count,
                common_history_candidate_state_key=common_history_candidate_state_key,
                common_history_candidate_tensor=common_history_candidate_tensor,
                common_history_use_full_batch=common_history_use_full_batch,
                common_history_valid_bools=common_history_valid_bools,
                common_history_valid_count=common_history_valid_count,
                common_history_valid_tensor=common_history_valid_tensor,
                common_router_project_full_batch=common_router_project_full_batch,
                max_history_batch=max_history_batch,
                page_cache_for_common=page_cache_for_common,
            ),
            observers=DecodeObservers(
                common_parity_debug=common_parity_debug,
                common_parity_debug_layers=common_parity_debug_layers,
                decode_timing_enabled=decode_timing_enabled,
                decode_timing_payload=decode_timing_payload,
                fallback=fallback,
                mark_decode_fine_timing=mark_decode_fine_timing,
                mark_decode_timing=mark_decode_timing,
                maybe_sync_decode_stage=maybe_sync_decode_stage,
                record_common_parity_debug=record_common_parity_debug,
                record_no_anchor_common_fallback=record_no_anchor_common_fallback,
            ),
        )
