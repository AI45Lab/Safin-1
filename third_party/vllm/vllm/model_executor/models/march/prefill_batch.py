# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed multi-request prefill orchestration.

Stages retain their original order: admission/layout, projection/convolution,
local recurrence/anchors, history read, state/snapshot writes and output scatter.
All mutable tensors and parameters remain owned by the attention module."""

from __future__ import annotations

import os

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

from . import backends
from .backends import reference_causal_conv1d_fn
from .diagnostics import _hsd_trace_requested
from .layout import HistoricalHSDLayout, HSDPreparedBatch
from .runtime import (
    _advance_hsd_conv_state,
    _env_int,
    _env_truthy,
    _is_valid_hsd_state_index,
    _l2norm,
)

logger = init_logger("vllm.model_executor.models.safin1")


class BatchedPrefillMixin:
    def _forward_hsd_batched_varlen(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        layout: HistoricalHSDLayout,
    ) -> bool:
        def fallback(reason: str) -> bool:
            for req_items in layout.hsd_prefix_snapshot_requests or []:
                for key, prefix_len, _ in req_items:
                    self._record_hsd_varlen_event(
                        "hsd_prefix_cache_fallback",
                        reason=f"snapshot_store_{reason}",
                        key=str(key),
                        prefix_len=int(prefix_len),
                    )
            if (
                reason == "missing_conv_cache"
                and _env_truthy("VLLM_HSD_ENABLE_GRAPH_V3")
                and self._native_conv_cache_for_kernel() is None
            ):
                self._record_hsd_varlen_event(
                    "hsd_graph_v3_prefill_warmup_without_conv_cache",
                    requests=len(layout.req_expanded_starts),
                )
                return False
            if reason in {"decode_step", "anchor_boundary_decode_step"} and _env_truthy(
                "VLLM_HSD_ENABLE_GRAPH_V3"
            ):
                self._record_hsd_varlen_event(
                    "hsd_varlen_prefill_not_applicable",
                    reason=reason,
                    requests=len(layout.req_expanded_starts),
                )
                return False
            self._record_hsd_varlen_fallback(f"batched_{reason}")
            return False

        if self.hsd_prefill_backend != "varlen":
            return False
        if torch.is_grad_enabled():
            return fallback("grad_enabled")
        if not hidden_states.is_cuda:
            return fallback("not_cuda")
        if layout.num_expanded_tokens <= 0:
            return fallback("empty_layout")

        trace_backend = "varlen_prefill"
        anchor_chunk_size = int(getattr(self.config, "anchor_chunk_size", 0) or 0)
        anchor_token_num = int(getattr(self.config, "anchor_token_num", 0) or 0)
        arbitrary_anchor_kernel = _env_truthy("VLLM_HSD_ENABLE_ARBITRARY_ANCHOR_KERNEL")
        if anchor_chunk_size <= 0 or anchor_chunk_size % 64 != 0:
            return fallback("bad_anchor_chunk_size")
        if anchor_token_num != 1:
            return fallback("anchor_token_num_not_one")

        short_prefill_threshold = int(
            os.environ.get(
                "VLLM_HSD_VARLEN_MIN_FRESH_PREFILL_TOKENS",
                str(anchor_chunk_size),
            )
        )
        short_prefill_requests = sum(
            1
            for text_len, past_text_len in zip(
                layout.req_text_lens,
                layout.past_text_lens,
            )
            if (int(past_text_len) <= 0 and int(text_len) < short_prefill_threshold)
        )
        if short_prefill_threshold > 0 and short_prefill_requests > 0:
            self._hsd_force_segments_fallback = True
            self._record_hsd_varlen_event(
                "hsd_varlen_short_prefill_segments_dispatch",
                requests=len(layout.req_text_lens),
                short_requests=short_prefill_requests,
                threshold=short_prefill_threshold,
            )
            return False
        require_homogeneous_prefill = _env_truthy(
            "VLLM_HSD_VARLEN_REQUIRE_HOMOGENEOUS_PREFILL"
        )
        distinct_text_lens = len({int(text_len) for text_len in layout.req_text_lens})
        if (
            require_homogeneous_prefill
            and len(layout.req_text_lens) > 1
            and distinct_text_lens > 1
        ):
            self._hsd_force_segments_fallback = True
            self._record_hsd_varlen_event(
                "hsd_varlen_heterogeneous_prefill_segments_dispatch",
                requests=len(layout.req_text_lens),
                distinct_text_lens=distinct_text_lens,
            )
            return False

        varlen_forward = backends._load_hsd_varlen_prefill()
        if varlen_forward is None:
            if backends._HSD_VARLEN_PREFILL_IMPORT_ERROR is not None:
                logger.warning_once(
                    "HSD grouped varlen prefill backend is selected but the "
                    "helper could not be imported: %s. Falling back to segments.",
                    backends._HSD_VARLEN_PREFILL_IMPORT_ERROR,
                )
            return fallback("import_failed")

        conv_cache = self._conv_cache_for_kernel()
        if conv_cache is None:
            return fallback("missing_conv_cache")

        local_v_heads = self.num_v_heads // self.tp_size
        local_k_heads = self.num_k_heads // self.tp_size
        local_key_dim = self.key_dim // self.tp_size
        local_value_dim = self.value_dim // self.tp_size
        qkv_size = local_key_dim * 2 + local_value_dim
        z_size = local_value_dim

        req_count = len(layout.req_expanded_starts)
        if req_count > 1 and not _env_truthy(
            "VLLM_HSD_ENABLE_MULTI_REQUEST_VARLEN_PREFILL"
        ):
            return fallback("multi_request_prefill")
        text_offsets = [0]
        anchor_offsets = [0]
        kernel_anchor_offsets = [0]
        memory_counts_by_req: list[int] = []
        state_indices: list[int] = []
        history_state_indices: list[int] = []
        past_text_lens: list[int] = []
        boundary_counts_by_req: list[torch.Tensor] = []

        for req_idx, expanded_len in enumerate(layout.req_expanded_lens):
            if int(expanded_len) <= 0:
                return fallback("empty_request")
            text_len = int(layout.req_text_lens[req_idx])
            anchor_len = int(layout.req_anchor_lens[req_idx])
            if text_len <= 0:
                return fallback("empty_text")
            if int(expanded_len) != text_len + anchor_len:
                return fallback("layout_len_mismatch")

            past_text_len = int(layout.past_text_lens[req_idx])
            if past_text_len > 0 and text_len == 1:
                return fallback(
                    "anchor_boundary_decode_step" if anchor_len > 0 else "decode_step"
                )
            if past_text_len % anchor_chunk_size != 0 and not arbitrary_anchor_kernel:
                return fallback("boundary_start_not_aligned")

            req_anchor_start = layout.req_anchor_starts[req_idx]
            boundary_counts = layout.anchor_boundaries[
                req_anchor_start : req_anchor_start + anchor_len
            ]
            memory_count = int(anchor_len > 0 and int(boundary_counts[0].item()) == 0)
            if memory_count and (
                self.hsd_persistent_memory_mode == "removed"
                or bool((boundary_counts[1:] == 0).any().item())
            ):
                return fallback("invalid_persistent_memory_boundaries")
            dynamic_boundaries = boundary_counts[memory_count:]
            expected_count = (
                past_text_len + text_len
            ) // anchor_chunk_size - past_text_len // anchor_chunk_size
            if anchor_len != expected_count + memory_count:
                return fallback("anchor_boundary_count_not_regular")
            if expected_count:
                first_boundary = (
                    (past_text_len // anchor_chunk_size) + 1
                ) * anchor_chunk_size
                expected = torch.arange(
                    first_boundary,
                    first_boundary + expected_count * anchor_chunk_size,
                    anchor_chunk_size,
                    dtype=dynamic_boundaries.dtype,
                    device=dynamic_boundaries.device,
                )
                if not torch.equal(dynamic_boundaries, expected):
                    return fallback("anchor_boundaries_not_regular")

            text_offsets.append(text_offsets[-1] + text_len)
            anchor_offsets.append(anchor_offsets[-1] + anchor_len)
            kernel_anchor_offsets.append(kernel_anchor_offsets[-1] + expected_count)
            memory_counts_by_req.append(memory_count)
            state_idx = self._state_index_for_request(layout, req_idx)
            if not _is_valid_hsd_state_index(state_idx):
                return fallback("null_state_idx")
            state_indices.append(state_idx)
            history_state_indices.append(
                self._history_state_index_for_request(layout, req_idx)
            )
            past_text_lens.append(past_text_len)
            boundary_counts_by_req.append(boundary_counts)

        replay_state_indices = [
            state_idx
            for state_idx in state_indices
            if state_idx in self.hsd_prefix_recurrent_replay
        ]
        if replay_state_indices:
            self._record_hsd_varlen_event(
                "hsd_prefix_recurrent_replay_loop_dispatch",
                requests=len(replay_state_indices),
                text_tokens=sum(layout.req_text_lens),
            )
            return False

        total_text = text_offsets[-1]
        total_anchor = anchor_offsets[-1]
        total_kernel_anchor = kernel_anchor_offsets[-1]
        if self._use_hsd_persistent_state_sidecar() and state_indices:
            conv_cache = self._conv_cache_for_kernel(max(state_indices) + 1)
            if conv_cache is None:
                return fallback("missing_persistent_conv_sidecar")
        if int(layout.text_to_expanded.numel()) != total_text:
            return fallback("text_index_count_mismatch")
        anchor_positions = layout.anchor_positions
        if int(anchor_positions.numel()) != total_anchor:
            return fallback("anchor_index_count_mismatch")

        prepared = HSDPreparedBatch(
            text_offsets=text_offsets,
            anchor_offsets=anchor_offsets,
            state_indices=state_indices,
            past_text_lens=past_text_lens,
            boundary_counts=boundary_counts_by_req,
        )

        persistent_text_cache = (
            get_forward_context().attn_metadata is not None
            and not _env_truthy("VLLM_HSD_DISABLE_TEXT_STATE_CACHE")
        )
        device = hidden_states.device
        multi_capture_req_idx = int(
            _env_int("VLLM_HSD_PREFILL_PASSIVE_SIDECAR_REQUEST_INDEX") or 0
        )

        def capture_single_prefill_sidecar(
            stage: str,
            tensor: torch.Tensor | None,
        ) -> None:
            capture_req_idx = 0
            if req_count != 1:
                if not _env_truthy(
                    "VLLM_HSD_PREFILL_PASSIVE_SIDECAR_ALLOW_MULTI_REQUEST"
                ):
                    return
                capture_req_idx = multi_capture_req_idx
                if capture_req_idx < 0 or capture_req_idx >= req_count:
                    return
                if stage in {
                    "initial_recurrent_state",
                    "final_recurrent_state",
                }:
                    if not torch.is_tensor(tensor) or int(tensor.shape[0]) != req_count:
                        return
                    tensor = tensor[capture_req_idx : capture_req_idx + 1]
                elif stage != "conv_state_after_text_conv":
                    # Multi-request diagnostics intentionally stay small. The
                    # token/anchor tensors have different packed dimensions;
                    # existing layer traces cover those stages already.
                    return
            self._capture_hsd_prefill_passive_sidecar(
                stage,
                tensor,
                past_text_len=int(prepared.past_text_lens[capture_req_idx]),
                text_len=int(
                    prepared.text_offsets[capture_req_idx + 1]
                    - prepared.text_offsets[capture_req_idx]
                ),
            )

        state_indices_tensor = torch.tensor(
            prepared.state_indices,
            dtype=torch.int32,
            device=device,
        )
        text_query_start_loc = torch.tensor(
            prepared.text_offsets,
            dtype=torch.int32,
            device=device,
        )
        trace_requested = _hsd_trace_requested()
        cu_seqlens = torch.tensor(
            prepared.text_offsets,
            # Match vLLM GDN's segment reference. Its packed Triton kernels
            # are parity-stable for long HSD inputs only with int32 indices.
            dtype=torch.int32,
            device=device,
        )
        anchor_offsets_tensor = torch.tensor(
            kernel_anchor_offsets,
            dtype=torch.long,
            device=device,
        )
        local_anchor_positions_tensor = None
        partial_anchor_tokens = 0
        needs_arbitrary_anchor_positions = (
            arbitrary_anchor_kernel
            and total_kernel_anchor > 0
            and any(
                int(past_text_len) % anchor_chunk_size != 0
                for past_text_len in prepared.past_text_lens
            )
        )
        if needs_arbitrary_anchor_positions:
            local_positions_by_req = []
            for req_idx, boundary_counts in enumerate(prepared.boundary_counts):
                local_positions = (
                    boundary_counts[memory_counts_by_req[req_idx] :].to(
                        device=device, dtype=torch.long
                    )
                    - int(prepared.past_text_lens[req_idx])
                ).contiguous()
                local_positions_by_req.append(local_positions)
                text_len = int(layout.req_text_lens[req_idx])
                for pos in local_positions.detach().cpu().tolist():
                    pos_int = int(pos)
                    if pos_int != text_len and pos_int % 64 != 0:
                        partial_anchor_tokens += 1
            local_anchor_positions_tensor = (
                torch.cat(local_positions_by_req, dim=0)
                if local_positions_by_req
                else torch.empty(0, dtype=torch.long, device=device)
            )
        has_initial_state = torch.tensor(
            [past > 0 for past in prepared.past_text_lens],
            dtype=torch.bool,
            device=device,
        )

        initial_states = []
        for req_idx, state_idx in enumerate(prepared.state_indices):
            history_state_idx = history_state_indices[req_idx]
            past_text_len = prepared.past_text_lens[req_idx]
            if past_text_len <= 0:
                self.hsd_anchor_cache.pop(history_state_idx, None)
                page_cache = getattr(self, "hsd_anchor_page_cache", None)
                if page_cache is not None:
                    page_cache.clear({history_state_idx})
                self.hsd_text_state_cache.pop(state_idx, None)
                self.hsd_prefix_recurrent_replay.pop(state_idx, None)
                self._clear_native_state_for_request(state_idx)
            elif persistent_text_cache:
                self._restore_hsd_text_state_cache(state_idx, device)
            initial_states.append(
                self._recurrent_state_for_request(
                    state_idx,
                    past_text_len,
                    device,
                )
            )
        initial_state = torch.cat(initial_states, dim=0)
        capture_single_prefill_sidecar("initial_recurrent_state", initial_state)

        expanded_hidden = hidden_states[: layout.num_expanded_tokens]
        text_hidden = None
        if trace_requested or self.hsd_layer_uses_global_read:
            text_hidden = expanded_hidden.index_select(0, layout.text_to_expanded)
        capture_single_prefill_sidecar(
            "hidden_states_text_before_proj",
            (
                text_hidden
                if text_hidden is not None
                else expanded_hidden.index_select(0, layout.text_to_expanded)
            ),
        )
        mixed_qkvz, _ = self.in_proj_qkvz(expanded_hidden)
        ba, _ = self.in_proj_ba(expanded_hidden)
        mixed_qkv, z_all_flat = mixed_qkvz.split([qkv_size, z_size], dim=-1)
        b_all, a_all = ba.chunk(2, dim=-1)

        mixed_qkv_text_before_conv = mixed_qkv.index_select(
            0,
            layout.text_to_expanded,
        ).contiguous()
        capture_single_prefill_sidecar(
            "mixed_qkv_text_before_conv",
            mixed_qkv_text_before_conv,
        )
        b_text = b_all.index_select(0, layout.text_to_expanded).contiguous()
        a_text = a_all.index_select(0, layout.text_to_expanded).contiguous()
        z_all = z_all_flat.view(
            layout.num_expanded_tokens,
            local_v_heads,
            self.head_v_dim,
        )

        conv_state_snapshot = (
            conv_cache.index_select(
                0,
                state_indices_tensor.long(),
            )
            .detach()
            .clone()
        )
        if _env_truthy("VLLM_HSD_PREFIX_CACHE_DEBUG"):
            for req_pos, state_idx in enumerate(prepared.state_indices):
                recurrent_entry = initial_state[req_pos : req_pos + 1]
                conv_entry = conv_state_snapshot[req_pos : req_pos + 1]
                self._record_hsd_varlen_event(
                    "hsd_prefix_suffix_prefill_entry_debug",
                    state_idx=int(state_idx),
                    past_text_len=int(prepared.past_text_lens[req_pos]),
                    initial_recurrent_norm=float(
                        recurrent_entry.detach().float().norm().item()
                    ),
                    initial_recurrent_sum=float(
                        recurrent_entry.detach().float().sum().item()
                    ),
                    initial_conv_norm=float(conv_entry.detach().float().norm().item()),
                    initial_conv_sum=float(conv_entry.detach().float().sum().item()),
                )
        use_request_loop_conv = self._use_request_loop_conv(req_count)
        if (
            req_count == 1
            and int(prepared.past_text_lens[0]) > 0
            and self._use_reference_conv_prefill()
            and self._use_reference_conv_explicit_state_prefix()
        ):
            # Prefix restore used to inherit the conservative request-loop
            # conv path. Route the single restored suffix through the same
            # grouped/reference dispatch as one-shot prefill so the explicit
            # raw-state prefix path below is actually used.
            use_request_loop_conv = False
        if _env_truthy("VLLM_HSD_PREFIX_CACHE_DEBUG") and any(
            int(past) > 0 for past in prepared.past_text_lens
        ):
            self._record_hsd_varlen_event(
                "hsd_prefix_conv_dispatch_debug",
                requests=req_count,
                prepared_requests=len(prepared.state_indices),
                past_text_lens=[int(past) for past in prepared.past_text_lens],
                deterministic_varlen_conv_env=os.environ.get(
                    "VLLM_HSD_DETERMINISTIC_VARLEN_CONV"
                ),
                disable_batched_conv_env=os.environ.get(
                    "VLLM_HSD_DISABLE_BATCHED_CONV"
                ),
                reference_conv_prefill=bool(self._use_reference_conv_prefill()),
                explicit_state_prefix=bool(
                    self._use_reference_conv_explicit_state_prefix()
                ),
                use_request_loop_conv=bool(use_request_loop_conv),
            )
        if trace_requested:
            self._trace_internal("conv1d_weight", self.conv1d.weight)
        if use_request_loop_conv:
            conv_chunks: list[torch.Tensor] = []
            for req_pos, state_idx in enumerate(prepared.state_indices):
                text_begin = prepared.text_offsets[req_pos]
                text_end = prepared.text_offsets[req_pos + 1]
                conv_chunks.append(
                    self._run_text_conv(
                        mixed_qkv_text_before_conv[text_begin:text_end]
                        .unsqueeze(0)
                        .contiguous(),
                        state_idx,
                        prepared.past_text_lens[req_pos],
                        decode_step=False,
                    ).squeeze(0)
                )
            mixed_qkv_text = torch.cat(conv_chunks, dim=0).contiguous()
        else:
            conv_weights = self.conv1d.weight.view(
                self.conv1d.weight.size(0),
                self.conv1d.weight.size(2),
            )
            use_reference_conv_prefill = (
                reference_causal_conv1d_fn is not None
                and self._use_reference_conv_prefill()
            )
            if use_reference_conv_prefill:
                if req_count == 1:
                    use_explicit_state_prefix = (
                        int(prepared.past_text_lens[0]) > 0
                        and self._use_reference_conv_explicit_state_prefix()
                    )
                    if int(prepared.past_text_lens[0]) > 0 and _env_truthy(
                        "VLLM_HSD_PREFIX_CACHE_DEBUG"
                    ):
                        self._record_hsd_varlen_event(
                            "reference_conv_suffix_mode",
                            past_text_len=int(prepared.past_text_lens[0]),
                            text_tokens=total_text,
                            explicit_state_prefix=bool(use_explicit_state_prefix),
                            explicit_state_prefix_env=os.environ.get(
                                "VLLM_HSD_REFERENCE_CONV_EXPLICIT_STATE_PREFIX"
                            ),
                        )
                    if use_explicit_state_prefix:
                        mixed_qkv_text = (
                            self._run_reference_single_conv_prefill_with_explicit_state(
                                token_inputs=mixed_qkv_text_before_conv,
                                conv_state_snapshot=conv_state_snapshot,
                                conv_weights=conv_weights,
                            )
                        )
                        self._record_hsd_varlen_event(
                            "reference_conv_explicit_state_prefix_hit",
                            requests=1,
                            text_tokens=total_text,
                            state_tokens=int(conv_state_snapshot.shape[-1]),
                        )
                    else:
                        conv_input = (
                            mixed_qkv_text_before_conv.contiguous()
                            .unsqueeze(0)
                            .transpose(1, 2)
                        )
                        reference_initial_state = None
                        if int(prepared.past_text_lens[0]) > 0:
                            reference_initial_state = (
                                conv_state_snapshot.transpose(1, 2)
                                .contiguous()
                                .transpose(1, 2)
                            )
                        mixed_qkv_text = reference_causal_conv1d_fn(
                            x=conv_input,
                            weight=conv_weights,
                            bias=self.conv1d.bias,
                            initial_states=reference_initial_state,
                            activation=self.activation,
                        )
                        mixed_qkv_text = (
                            mixed_qkv_text.squeeze(0).transpose(0, 1).contiguous()
                        )
                else:
                    mixed_qkv_text = self._run_reference_varlen_conv_prefill(
                        packed_inputs=mixed_qkv_text_before_conv,
                        token_offsets=prepared.text_offsets,
                        conv_state_snapshot=conv_state_snapshot,
                        past_text_lens=prepared.past_text_lens,
                        conv_weights=conv_weights,
                    )
                self._record_hsd_varlen_event(
                    "reference_conv_prefill_hit",
                    requests=req_count,
                    text_tokens=total_text,
                    has_initial_state=int(prepared.past_text_lens[0]) > 0,
                    source="grouped_varlen",
                )
            else:
                mixed_qkv_text = (
                    causal_conv1d_fn(
                        mixed_qkv_text_before_conv.transpose(0, 1),
                        conv_weights,
                        self.conv1d.bias,
                        conv_cache,
                        query_start_loc=text_query_start_loc,
                        cache_indices=state_indices_tensor,
                        has_initial_state=has_initial_state,
                        activation=self.activation,
                    )
                    .transpose(0, 1)
                    .contiguous()
                )
        capture_single_prefill_sidecar(
            "mixed_qkv_text_after_conv",
            mixed_qkv_text,
        )

        query, key, value, g, beta = self._post_conv_prep_hsd(
            mixed_qkv_text=mixed_qkv_text,
            a=a_text,
            b=b_text,
            local_k_heads=local_k_heads,
            local_v_heads=local_v_heads,
            local_key_dim=local_key_dim,
            local_value_dim=local_value_dim,
        )
        if local_v_heads // max(1, local_k_heads) > 1:
            repeat = local_v_heads // local_k_heads
            query = query.repeat_interleave(repeat, dim=1)
            key = key.repeat_interleave(repeat, dim=1)
        for stage, tensor in (
            ("q_text", query),
            ("k_text", key),
            ("v_text", value),
            ("beta_text", beta),
            ("g_text", g),
        ):
            capture_single_prefill_sidecar(stage, tensor)

        if total_anchor:
            anchor_hidden = expanded_hidden.index_select(0, anchor_positions)
            anchor_q_all, _ = self.anchor_q_proj(anchor_hidden)
            anchor_q_all = anchor_q_all.view(
                total_anchor,
                local_v_heads,
                self.head_k_dim,
            )
            anchor_q_all = _l2norm(anchor_q_all.contiguous())
        else:
            anchor_q_all = expanded_hidden.new_zeros(
                0,
                local_v_heads,
                self.head_k_dim,
            )

        anchor_k_all = None
        router_q_all = None
        null_logits_all = None
        if self.hsd_layer_uses_global_read:
            if self.anchor_k_proj is None:
                raise RuntimeError("anchor_k_proj is required for HSD global read.")
            if self.router_q_proj is None:
                raise RuntimeError("router_q_proj is required for HSD global read.")
            if total_anchor:
                anchor_k_all, _ = self.anchor_k_proj(anchor_hidden)
                anchor_k_all = anchor_k_all.view(
                    total_anchor,
                    self.hsd_local_route_key_heads,
                    self.hsd_route_head_dim,
                )
                anchor_k_all = self._replace_persistent_anchor_route_key(
                    anchor_k_all,
                    layout.persistent_memory_mask[
                        : layout.num_expanded_tokens
                    ].index_select(0, anchor_positions),
                )
                if self.hsd_route_qk_l2norm:
                    anchor_k_all = _l2norm(anchor_k_all.contiguous())
            else:
                anchor_k_all = expanded_hidden.new_zeros(
                    0,
                    self.hsd_local_route_key_heads,
                    self.hsd_route_head_dim,
                )
            assert text_hidden is not None
            router_q_all, _ = self.router_q_proj(text_hidden)
            router_q_all = router_q_all.view(
                total_text,
                self.hsd_local_route_query_heads,
                self.hsd_route_head_dim,
            )
            if self.hsd_route_qk_l2norm:
                router_q_all = _l2norm(router_q_all.contiguous())
            null_logits_all = self._project_null_gate_logits(text_hidden)

        exact_segment_parity = _env_truthy("VLLM_HSD_VARLEN_EXACT_SEGMENT_PARITY")
        if not exact_segment_parity:
            self._maybe_warm_varlen_segment_autotune(
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                first_text_len=(prepared.text_offsets[1] - prepared.text_offsets[0]),
            )

        compare_varlen_segments = _env_truthy("VLLM_HSD_VARLEN_COMPARE_SEGMENTS")
        varlen_compare_inputs = None
        if compare_varlen_segments:
            varlen_compare_inputs = {
                "query": query.detach().clone(),
                "key": key.detach().clone(),
                "value": value.detach().clone(),
                "g": g.detach().clone(),
                "beta": beta.detach().clone(),
                "initial_state": initial_state.detach().clone(),
            }

        result = (
            self._run_hsd_grouped_segment_parity(
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
                text_offsets=prepared.text_offsets,
                anchor_offsets=kernel_anchor_offsets,
                boundary_counts=[
                    boundaries[memory_count:]
                    for boundaries, memory_count in zip(
                        prepared.boundary_counts,
                        memory_counts_by_req,
                        strict=True,
                    )
                ],
                past_text_lens=prepared.past_text_lens,
                initial_state=initial_state.float().contiguous(),
            )
            if exact_segment_parity
            else varlen_forward(
                query=query.unsqueeze(0).contiguous(),
                key=key.unsqueeze(0).contiguous(),
                value=value.unsqueeze(0).contiguous(),
                g=g.unsqueeze(0).contiguous(),
                beta=beta.unsqueeze(0).contiguous(),
                cu_seqlens=cu_seqlens,
                anchor_offsets=anchor_offsets_tensor,
                anchor_chunk_size=anchor_chunk_size,
                initial_state=initial_state.float().contiguous(),
                scale=self.head_k_dim**-0.5,
                local_anchor_positions=local_anchor_positions_tensor,
                use_vllm_reference_kernel=(self._use_vllm_varlen_reference_kernel()),
                use_deterministic_kkt=self._use_deterministic_varlen_kkt(),
            )
        )
        if result is None:
            for req_pos, state_idx in enumerate(prepared.state_indices):
                self._store_conv_state(
                    state_idx,
                    conv_state_snapshot[req_pos : req_pos + 1],
                )
            return fallback("kernel_returned_none")

        logger.info_once("Using HSD grouped varlen packed prefill helper")
        self._record_hsd_varlen_counter("batched_prepare_calls")
        self._record_hsd_varlen_counter("grouped_varlen_calls")
        self._record_hsd_varlen_counter("grouped_varlen_requests", req_count)
        self._record_hsd_varlen_counter("grouped_varlen_text_tokens", total_text)
        self._record_hsd_varlen_counter("grouped_varlen_anchor_tokens", total_anchor)
        self._record_hsd_varlen_event(
            "grouped_varlen_hit",
            requests=req_count,
            text_tokens=total_text,
            anchor_tokens=total_anchor,
        )
        if self._use_deterministic_varlen_kkt():
            self._record_hsd_varlen_event(
                "grouped_varlen_deterministic_kkt_hit",
                requests=req_count,
                text_tokens=total_text,
                anchor_tokens=total_anchor,
            )
        if self._use_deterministic_prefill_autotune():
            self._record_hsd_varlen_event(
                "grouped_varlen_deterministic_autotune_hit",
                requests=req_count,
                text_tokens=total_text,
                anchor_tokens=total_anchor,
            )
        use_vllm_reference_kernel = (
            not exact_segment_parity and self._use_vllm_varlen_reference_kernel()
        )
        if exact_segment_parity:
            self._record_hsd_varlen_event(
                "grouped_varlen_exact_segment_parity_hit",
                requests=req_count,
                text_tokens=total_text,
                anchor_tokens=total_anchor,
            )
        elif local_anchor_positions_tensor is None and use_vllm_reference_kernel:
            self._record_hsd_varlen_event(
                "grouped_varlen_vllm_parity_kernel_hit",
                requests=req_count,
                text_tokens=total_text,
                anchor_tokens=total_anchor,
            )
        elif local_anchor_positions_tensor is None:
            self._record_hsd_varlen_event(
                "grouped_varlen_deterministic_kernel_hit",
                requests=req_count,
                text_tokens=total_text,
                anchor_tokens=total_anchor,
                model_type=str(getattr(self.config, "model_type", "")),
            )
        if local_anchor_positions_tensor is not None:
            self._record_hsd_varlen_event(
                "grouped_varlen_arbitrary_anchor_kernel_hit",
                requests=req_count,
                text_tokens=total_text,
                anchor_tokens=total_anchor,
                partial_anchor_tokens=partial_anchor_tokens,
            )
        if not use_request_loop_conv and req_count > 1:
            self._record_hsd_varlen_counter("batched_conv_prefill_calls")
            self._record_hsd_varlen_counter("batched_conv_prefill_requests", req_count)
            self._record_hsd_varlen_counter(
                "batched_conv_prefill_text_tokens",
                total_text,
            )
            self._record_hsd_varlen_event(
                "batched_conv_prefill_hit",
                requests=req_count,
                text_tokens=total_text,
            )

        text_o_all, anchor_states_flat, final_state_flat = result
        # The HSD varlen helper uses TRANSPOSE_STATE=False and returns both
        # anchor and final states in HSD's native [H, K, V] layout. Applying
        # another K/V transpose here corrupts the state restored by decode.
        anchor_states_flat = anchor_states_flat.contiguous()
        if any(memory_counts_by_req):
            anchor_states_flat = self._restore_persistent_varlen_anchor_states(
                dynamic_anchor_states=anchor_states_flat,
                anchor_q=anchor_q_all,
                anchor_offsets=prepared.anchor_offsets,
                kernel_anchor_offsets=kernel_anchor_offsets,
                memory_counts=memory_counts_by_req,
            )
        final_state_flat = final_state_flat.contiguous()
        capture_single_prefill_sidecar("text_o_before_history", text_o_all)
        capture_single_prefill_sidecar("anchor_states", anchor_states_flat)
        if anchor_k_all is not None:
            capture_single_prefill_sidecar("anchor_keys", anchor_k_all)
        capture_single_prefill_sidecar(
            "final_recurrent_state",
            final_state_flat,
        )
        if compare_varlen_segments:
            assert varlen_compare_inputs is not None
            for name, current in (
                ("query", query),
                ("key", key),
                ("value", value),
                ("g", g),
                ("beta", beta),
                ("initial_state", initial_state),
            ):
                mutation_record = {
                    "tensor": name,
                    "requests": int(req_count),
                    "text_tokens": int(total_text),
                }
                mutation_record.update(
                    self._hsd_tensor_diff_stats(
                        "mutation",
                        current,
                        varlen_compare_inputs[name],
                    )
                )
                self._record_hsd_varlen_event(
                    "hsd_varlen_input_mutation_compare",
                    **mutation_record,
                )
            for req_idx in range(req_count):
                text_begin = prepared.text_offsets[req_idx]
                text_end = prepared.text_offsets[req_idx + 1]
                anchor_begin = prepared.anchor_offsets[req_idx]
                anchor_end = prepared.anchor_offsets[req_idx + 1]
                state_idx = prepared.state_indices[req_idx]
                past_text_len = prepared.past_text_lens[req_idx]
                record: dict[str, object] = {
                    "req_idx": int(req_idx),
                    "state_idx": int(state_idx),
                    "past_text_len": int(past_text_len),
                    "text_len": int(text_end - text_begin),
                    "anchor_count": int(anchor_end - anchor_begin),
                }
                try:
                    ref_text_o, _ref_anchor_o, ref_anchor_states, ref_state = (
                        self._run_hsd_segments(
                            query=varlen_compare_inputs["query"][
                                text_begin:text_end
                            ].unsqueeze(0),
                            key=varlen_compare_inputs["key"][
                                text_begin:text_end
                            ].unsqueeze(0),
                            value=varlen_compare_inputs["value"][
                                text_begin:text_end
                            ].unsqueeze(0),
                            g=varlen_compare_inputs["g"][text_begin:text_end].unsqueeze(
                                0
                            ),
                            beta=varlen_compare_inputs["beta"][
                                text_begin:text_end
                            ].unsqueeze(0),
                            anchor_q=anchor_q_all[anchor_begin:anchor_end].unsqueeze(0),
                            boundary_counts=prepared.boundary_counts[req_idx],
                            boundary_start=past_text_len,
                            initial_state=varlen_compare_inputs["initial_state"][
                                req_idx : req_idx + 1
                            ],
                        )
                    )
                    actual_text_o = text_o_all[:, text_begin:text_end]
                    actual_anchor_states = anchor_states_flat[
                        anchor_begin:anchor_end
                    ].unsqueeze(0)
                    actual_state = final_state_flat[req_idx : req_idx + 1]
                    record.update(
                        self._hsd_tensor_diff_stats(
                            "text_o",
                            actual_text_o,
                            ref_text_o,
                        )
                    )
                    record.update(
                        self._hsd_tensor_diff_stats(
                            "anchor_states",
                            actual_anchor_states,
                            ref_anchor_states,
                        )
                    )
                    record.update(
                        self._hsd_tensor_diff_stats(
                            "final_state",
                            actual_state,
                            ref_state,
                        )
                    )
                    if (
                        actual_state.ndim >= 2
                        and actual_state.shape[-1] == actual_state.shape[-2]
                    ):
                        record.update(
                            self._hsd_tensor_diff_stats(
                                "final_state_transposed",
                                actual_state.transpose(-1, -2).contiguous(),
                                ref_state,
                            )
                        )
                except Exception as exc:
                    record["error"] = repr(exc)
                self._record_hsd_varlen_event(
                    "hsd_varlen_segments_compare",
                    **record,
                )
        if not use_request_loop_conv:
            self._store_batched_conv_states(
                state_indices=prepared.state_indices,
                token_offsets=prepared.text_offsets,
                conv_state_snapshot=conv_state_snapshot,
                token_inputs=mixed_qkv_text_before_conv,
                past_text_lens=prepared.past_text_lens,
            )
            capture_single_prefill_sidecar(
                "conv_state_after_text_conv",
                self._conv_state_for_request(
                    prepared.state_indices[
                        multi_capture_req_idx
                        if 0 <= multi_capture_req_idx < req_count
                        else 0
                    ]
                ),
            )
            if compare_varlen_segments:
                for req_pos, state_idx in enumerate(prepared.state_indices):
                    text_begin = prepared.text_offsets[req_pos]
                    text_end = prepared.text_offsets[req_pos + 1]
                    expected_conv_state = _advance_hsd_conv_state(
                        conv_state_snapshot[req_pos : req_pos + 1],
                        mixed_qkv_text_before_conv[text_begin:text_end],
                        has_initial_state=prepared.past_text_lens[req_pos] > 0,
                    )
                    record = {
                        "req_idx": int(req_pos),
                        "state_idx": int(state_idx),
                        "past_text_len": int(prepared.past_text_lens[req_pos]),
                        "text_len": int(text_end - text_begin),
                    }
                    record.update(
                        self._hsd_tensor_diff_stats(
                            "conv_store",
                            self._conv_state_for_request(state_idx),
                            expected_conv_state,
                        )
                    )
                    self._record_hsd_varlen_event(
                        "hsd_varlen_conv_store_readback",
                        **record,
                    )
        text_o_all = text_o_all.squeeze(0).contiguous()
        deterministic_anchor_read = self._use_deterministic_anchor_read(total_anchor)
        segment_parity_anchor_o = (
            total_anchor > 0
            and not deterministic_anchor_read
            and _env_truthy("VLLM_HSD_VARLEN_SEGMENT_PARITY_ANCHOR_O")
        )
        use_batched_anchor_o = (
            total_anchor > 0
            and not deterministic_anchor_read
            and not segment_parity_anchor_o
            and not _env_truthy("VLLM_HSD_DISABLE_BATCHED_ANCHOR_O")
        )
        if deterministic_anchor_read:
            anchor_read = backends._load_hsd_deterministic_anchor_read()
            anchor_o_all = (
                None
                if anchor_read is None
                else anchor_read(
                    anchor_q_all,
                    anchor_states_flat,
                    scale=self.head_k_dim**-0.5,
                )
            )
            if anchor_o_all is None:
                raise RuntimeError(
                    "VLLM_HSD_DETERMINISTIC_ANCHOR_READ was requested, but "
                    "the fixed-reduction Triton kernel is unavailable."
                )
            anchor_o_all = anchor_o_all.to(text_o_all.dtype)
            self._record_hsd_varlen_event(
                "deterministic_anchor_read_hit",
                requests=req_count,
                anchor_tokens=total_anchor,
            )
        elif segment_parity_anchor_o:
            # The segments reference evaluates one anchor boundary per
            # einsum call. MoE routing can amplify the BF16 difference from a
            # single larger batched einsum, so keep an exact diagnostic/parity
            # path while the fixed-reduction anchor kernel is validated.
            anchor_rows = []
            for anchor_idx in range(total_anchor):
                anchor_rows.append(
                    (
                        torch.einsum(
                            "bmhk,bmhkv->bmhv",
                            anchor_q_all[anchor_idx : anchor_idx + 1]
                            .view(1, 1, local_v_heads, self.head_k_dim)
                            .to(anchor_states_flat.dtype),
                            anchor_states_flat[anchor_idx : anchor_idx + 1].view(
                                1,
                                1,
                                local_v_heads,
                                self.head_k_dim,
                                self.head_v_dim,
                            ),
                        )
                        * (self.head_k_dim**-0.5)
                    )
                    .to(text_o_all.dtype)
                    .view(1, local_v_heads, self.head_v_dim)
                )
            anchor_o_all = torch.cat(anchor_rows, dim=0)
            self._record_hsd_varlen_event(
                "segment_parity_anchor_o_hit",
                requests=req_count,
                anchor_tokens=total_anchor,
            )
        elif use_batched_anchor_o:
            anchor_o_all = (
                torch.einsum(
                    "mhk,mhkv->mhv",
                    anchor_q_all.to(anchor_states_flat.dtype),
                    anchor_states_flat,
                )
                * (self.head_k_dim**-0.5)
            ).to(text_o_all.dtype)
            self._record_hsd_varlen_event(
                "batched_anchor_o_hit",
                requests=req_count,
                anchor_tokens=total_anchor,
            )
        else:
            anchor_o_all = expanded_hidden.new_zeros(
                total_anchor,
                local_v_heads,
                self.head_v_dim,
            )
        core_attn_out = expanded_hidden.new_zeros(
            layout.num_expanded_tokens,
            local_v_heads,
            self.head_v_dim,
        )
        batched_paged_prefill_history_applied = (
            self._try_batched_paged_prefill_history_read(
                text_o=text_o_all,
                query=query,
                router_q=router_q_all,
                null_logits=null_logits_all,
                anchor_states=anchor_states_flat,
                anchor_keys=anchor_k_all,
                anchor_offsets=prepared.anchor_offsets,
                anchor_boundaries=prepared.boundary_counts,
                text_offsets=prepared.text_offsets,
                history_state_indices=history_state_indices,
                past_text_lens=prepared.past_text_lens,
                trace_requested=trace_requested,
            )
        )

        for req_idx in range(req_count):
            text_begin = prepared.text_offsets[req_idx]
            text_end = prepared.text_offsets[req_idx + 1]
            anchor_begin = prepared.anchor_offsets[req_idx]
            anchor_end = prepared.anchor_offsets[req_idx + 1]
            text_len = text_end - text_begin
            anchor_count = anchor_end - anchor_begin
            state_idx = prepared.state_indices[req_idx]
            history_state_idx = history_state_indices[req_idx]
            past_text_len = prepared.past_text_lens[req_idx]
            boundary_counts = prepared.boundary_counts[req_idx]

            req_query = query[text_begin:text_end].unsqueeze(0)
            req_text_o = text_o_all[text_begin:text_end].unsqueeze(0)
            req_recurrent_state = final_state_flat[req_idx : req_idx + 1].contiguous()
            req_anchor_states = (
                anchor_states_flat[anchor_begin:anchor_end].unsqueeze(0).contiguous()
            )
            req_anchor_q = anchor_q_all[anchor_begin:anchor_end].unsqueeze(0)
            if anchor_count:
                if (
                    deterministic_anchor_read
                    or use_batched_anchor_o
                    or segment_parity_anchor_o
                ):
                    req_anchor_o = anchor_o_all[anchor_begin:anchor_end].unsqueeze(0)
                else:
                    req_anchor_o = (
                        torch.einsum(
                            "bmhk,bmhkv->bmhv",
                            req_anchor_q.to(req_anchor_states.dtype),
                            req_anchor_states,
                        )
                        * (self.head_k_dim**-0.5)
                    ).to(req_text_o.dtype)
                    anchor_o_all[anchor_begin:anchor_end].copy_(req_anchor_o.squeeze(0))
            else:
                req_anchor_o = expanded_hidden.new_zeros(
                    1,
                    0,
                    local_v_heads,
                    self.head_v_dim,
                )

            if trace_requested:
                req_key = key[text_begin:text_end].unsqueeze(0)
                req_value = value[text_begin:text_end].unsqueeze(0)
                req_beta = beta[text_begin:text_end].unsqueeze(0)
                req_g = g[text_begin:text_end].unsqueeze(0)
                self._trace_internal(
                    "hsd_past_text_len",
                    expanded_hidden.new_tensor([past_text_len], dtype=torch.int64),
                )
                self._trace_internal(
                    "hsd_state_idx",
                    expanded_hidden.new_tensor([state_idx], dtype=torch.int64),
                )
                self._trace_internal(
                    "conv_state_before_text_conv",
                    conv_state_snapshot[req_idx : req_idx + 1],
                )
                self._trace_internal(
                    "hidden_states_text_before_proj",
                    text_hidden[text_begin:text_end].view(
                        1,
                        text_len,
                        text_hidden.shape[-1],
                    ),
                )
                z_text = z_all_flat.index_select(
                    0,
                    layout.text_to_expanded[text_begin:text_end],
                ).view(1, text_len, z_size)
                qkvzba_text = torch.cat(
                    [
                        mixed_qkv_text_before_conv[text_begin:text_end].view(
                            1,
                            text_len,
                            qkv_size,
                        ),
                        z_text,
                        b_text[text_begin:text_end].view(1, text_len, -1),
                        a_text[text_begin:text_end].view(1, text_len, -1),
                    ],
                    dim=-1,
                )
                if qkvzba_text.shape[-1] % max(1, local_k_heads) == 0:
                    qkvzba_text = qkvzba_text.view(1, text_len, local_k_heads, -1)
                self._trace_internal("qkvzba_text_after_proj", qkvzba_text)
                self._trace_internal(
                    "gate_text",
                    z_all.index_select(
                        0, layout.text_to_expanded[text_begin:text_end]
                    ).view(1, text_len, local_v_heads, self.head_v_dim),
                )
                self._trace_internal(
                    "conv_state_after_text_conv",
                    self._conv_state_for_request(state_idx),
                )
                self._trace_internal(
                    "qkvzba_text_after_conv",
                    mixed_qkv_text[text_begin:text_end].view(1, text_len, qkv_size),
                )
                self._trace_internal("q_text", req_query)
                self._trace_internal("k_text", req_key)
                self._trace_internal("v_text", req_value)
                self._trace_internal("beta_text", req_beta)
                self._trace_internal("g_text", req_g)
                self._trace_internal(
                    "anchor_hidden_pre",
                    anchor_hidden[anchor_begin:anchor_end].unsqueeze(0),
                )
                self._trace_internal("anchor_q", req_anchor_q)
                if self.hsd_layer_uses_global_read and anchor_k_all is not None:
                    self._trace_internal(
                        "anchor_k",
                        anchor_k_all[anchor_begin:anchor_end].unsqueeze(0),
                    )
                self._trace_internal(
                    "anchor_boundaries",
                    boundary_counts.view(1, -1).to(device=device),
                )
                self._trace_internal("text_o_before_history", req_text_o)
                self._trace_internal("anchor_o", req_anchor_o)
                self._trace_internal("anchor_states", req_anchor_states)

            req_anchor_k = None
            if self.hsd_layer_uses_global_read:
                assert anchor_k_all is not None
                req_anchor_k = anchor_k_all[anchor_begin:anchor_end].unsqueeze(0)
            snapshot_requests = (
                layout.hsd_prefix_snapshot_requests[req_idx]
                if req_idx < len(layout.hsd_prefix_snapshot_requests)
                else []
            )
            if snapshot_requests:
                self.capture_hsd_prefix_snapshot_from_tensors(
                    state_idx=state_idx,
                    history_state_idx=history_state_idx,
                    past_text_len=past_text_len,
                    snapshot_requests=snapshot_requests,
                    conv_state_snapshot=conv_state_snapshot[req_idx : req_idx + 1],
                    pre_conv_inputs=mixed_qkv_text_before_conv[text_begin:text_end],
                    query=req_query,
                    key_tensor=key[text_begin:text_end].unsqueeze(0),
                    value=value[text_begin:text_end].unsqueeze(0),
                    g=g[text_begin:text_end].unsqueeze(0),
                    beta=beta[text_begin:text_end].unsqueeze(0),
                    initial_state=initial_state[req_idx : req_idx + 1],
                    final_recurrent_state=req_recurrent_state,
                    new_anchor_states=req_anchor_states,
                    new_anchor_keys=req_anchor_k,
                    new_anchor_boundaries=boundary_counts,
                )

            if (
                self.hsd_layer_uses_global_read
                and not batched_paged_prefill_history_applied
            ):
                assert req_anchor_k is not None
                materialized = None
                paged_prefill_history_applied = False
                if self._using_scheduler_history_cache():
                    appended, page_cache, _ = self._append_new_anchor_to_history_cache(
                        state_idx=history_state_idx,
                        past_text_len=past_text_len,
                        new_anchor_states=req_anchor_states,
                        new_anchor_keys=req_anchor_k,
                        new_anchor_boundaries=boundary_counts,
                        use_paged_history=True,
                    )
                    if (
                        appended
                        and page_cache is not None
                        and page_cache.length(history_state_idx) > 0
                    ):
                        static_history_read = (
                            backends._load_hsd_paged_static_history_read()
                        )
                        use_paged_prefill_read = (
                            _env_truthy("VLLM_HSD_ENABLE_PAGED_PREFILL_HISTORY_READ")
                            and self._allow_paged_prefill_history_read()
                            and not trace_requested
                            and self.hsd_history_out_rmsnorm_weight is None
                            and static_history_read is not None
                        )
                        if use_paged_prefill_read:
                            assert router_q_all is not None
                            router_q = router_q_all[text_begin:text_end]
                            null_logits = None
                            if null_logits_all is not None:
                                null_logits = null_logits_all[text_begin:text_end].view(
                                    text_len, local_v_heads
                                )
                            token_start_indices = torch.arange(
                                past_text_len,
                                past_text_len + text_len,
                                device=device,
                                dtype=torch.long,
                            )
                            token_state_indices = torch.full(
                                (text_len,),
                                history_state_idx,
                                device=device,
                                dtype=torch.long,
                            )
                            token_valid_mask = torch.ones(
                                text_len,
                                device=device,
                                dtype=torch.bool,
                            )
                            assert static_history_read is not None
                            static_history_read(
                                cache=page_cache,
                                read_q=req_query.squeeze(0),
                                router_q=router_q,
                                state_indices=token_state_indices,
                                start_indices=token_start_indices,
                                valid_mask=token_valid_mask,
                                route_scale=float(
                                    (
                                        self.head_k_dim**-0.5
                                        if self.hsd_use_route_scale
                                        else 1.0
                                    )
                                    * self.hsd_route_logit_scale
                                ),
                                read_scale=float(self.head_k_dim**-0.5),
                                base_out=req_text_o.squeeze(0),
                                history_scale=self._history_scale_tensor(device),
                                null_logits=null_logits,
                                read_q_is_raw=False,
                                normalize_router_q=False,
                                value_block_size=int(
                                    os.environ.get(
                                        "HSD_PAGED_PREFILL_HISTORY_VALUE_BLOCK_SIZE",
                                        "32",
                                    )
                                ),
                                persistent_memory_enabled=(
                                    self.hsd_persistent_memory_mode != "removed"
                                ),
                                memory_metrics=(
                                    self._paged_persistent_memory_metric_sink(req_query)
                                ),
                            )
                            paged_prefill_history_applied = True
                            self._record_hsd_varlen_event(
                                "paged_prefill_history_read_hit",
                                requests=1,
                                text_tokens=text_len,
                                anchor_tokens=page_cache.length(history_state_idx),
                            )
                        else:
                            materialized = self._materialize_anchor_page_history(
                                page_cache,
                                history_state_idx,
                                device=device,
                            )
                if not paged_prefill_history_applied and materialized is None:
                    materialized = self._merge_anchor_cache(
                        state_idx=history_state_idx,
                        past_text_len=past_text_len,
                        new_anchor_states=req_anchor_states,
                        new_anchor_keys=req_anchor_k,
                        new_anchor_boundaries=boundary_counts,
                    )
                if not paged_prefill_history_applied:
                    assert materialized is not None
                    anchor_states, anchor_keys, anchor_boundaries = materialized
                if (
                    not paged_prefill_history_applied
                    and anchor_states.shape[1] > 0
                    and anchor_keys is not None
                ):
                    assert router_q_all is not None
                    router_q = router_q_all[text_begin:text_end].unsqueeze(0)
                    null_logits = None
                    if null_logits_all is not None:
                        null_logits = (
                            null_logits_all[text_begin:text_end]
                            .view(1, text_len, local_v_heads, 1)
                            .contiguous()
                        )
                    if trace_requested:
                        self._trace_internal("router_q", router_q)
                        self._trace_internal("read_q", req_query)
                        self._trace_internal("null_gate_logits", null_logits)
                    history_out = self._history_read(
                        read_q=req_query,
                        router_q=router_q,
                        states=anchor_states,
                        anchor_k=anchor_keys,
                        boundary_counts=anchor_boundaries,
                        text_len=text_len,
                        start_index=past_text_len,
                        null_logits=null_logits,
                    )
                    if trace_requested:
                        self._trace_internal("history_out_text", history_out)
                    history_out = self._apply_history_out_rmsnorm(history_out)
                    if (
                        trace_requested
                        and self.hsd_history_out_rmsnorm_weight is not None
                    ):
                        self._trace_internal("history_out_rmsnorm_text", history_out)
                    req_text_o = req_text_o + (
                        history_out
                        * self._history_scale_tensor(history_out.device).to(
                            dtype=history_out.dtype
                        )
                    ).to(req_text_o.dtype)

            if trace_requested:
                self._trace_internal("delta_o_text", req_text_o)
                self._trace_internal("recurrent_state_to_store", req_recurrent_state)
            self._store_recurrent_state(state_idx, req_recurrent_state)
            if compare_varlen_segments:
                record = {
                    "req_idx": int(req_idx),
                    "state_idx": int(state_idx),
                    "past_text_len": int(past_text_len),
                    "text_len": int(text_len),
                    "anchor_count": int(anchor_count),
                }
                record.update(
                    self._hsd_tensor_diff_stats(
                        "recurrent_store",
                        self._stored_recurrent_state_for_request(state_idx),
                        req_recurrent_state,
                    )
                )
                self._record_hsd_varlen_event(
                    "hsd_varlen_recurrent_store_readback",
                    **record,
                )
            if trace_requested:
                self._trace_internal(
                    "recurrent_state_after_store_readback",
                    self._stored_recurrent_state_for_request(state_idx),
                )
            if persistent_text_cache:
                self._update_hsd_text_state_cache(state_idx)
            text_o_all[text_begin:text_end].copy_(req_text_o.squeeze(0))

        capture_single_prefill_sidecar("delta_o_text", text_o_all)

        core_attn_out.index_copy_(0, layout.text_to_expanded, text_o_all)
        if total_anchor:
            core_attn_out.index_copy_(0, anchor_positions, anchor_o_all)
        capture_single_prefill_sidecar(
            "core_attn_out_before_norm",
            core_attn_out.index_select(0, layout.text_to_expanded),
        )
        if trace_requested:
            for req_idx in range(req_count):
                state_idx = state_indices[req_idx]
                past_text_len = past_text_lens[req_idx]
                anchor_count = anchor_offsets[req_idx + 1] - anchor_offsets[req_idx]
                boundary_counts = boundary_counts_by_req[req_idx]
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="core_attn_out_before_norm",
                    tensor=core_attn_out.index_select(
                        0, layout.text_to_expanded[req_idx : req_idx + 1]
                    ).view(1, 1, local_v_heads, self.head_v_dim),
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z_flat = z_all.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z_flat)
        core_attn_out = core_attn_out.reshape(layout.num_expanded_tokens, -1)
        capture_single_prefill_sidecar(
            "core_attn_out_after_norm",
            core_attn_out.index_select(0, layout.text_to_expanded),
        )
        if trace_requested:
            for req_idx in range(req_count):
                state_idx = state_indices[req_idx]
                past_text_len = past_text_lens[req_idx]
                anchor_count = anchor_offsets[req_idx + 1] - anchor_offsets[req_idx]
                boundary_counts = boundary_counts_by_req[req_idx]
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="core_attn_out_after_norm",
                    tensor=core_attn_out.index_select(
                        0, layout.text_to_expanded[req_idx : req_idx + 1]
                    ).view(1, 1, -1),
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
        projected_out, _ = self.out_proj(core_attn_out)
        capture_single_prefill_sidecar(
            "projected_out_text",
            projected_out.index_select(0, layout.text_to_expanded),
        )
        if trace_requested:
            for req_idx in range(req_count):
                state_idx = state_indices[req_idx]
                past_text_len = past_text_lens[req_idx]
                anchor_count = anchor_offsets[req_idx + 1] - anchor_offsets[req_idx]
                boundary_counts = boundary_counts_by_req[req_idx]
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="projected_out_text",
                    tensor=projected_out.index_select(
                        0, layout.text_to_expanded[req_idx : req_idx + 1]
                    ).view(1, 1, -1),
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
        output[: layout.num_expanded_tokens] = projected_out
        return True
