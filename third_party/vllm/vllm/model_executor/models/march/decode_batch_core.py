# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Original projections, convolution, local GDN and boundary append ordering.

Methods preserve arithmetic, checks and cache-write order. No model parameters
or request-state lifecycle are owned by this mixin.
"""

from __future__ import annotations

import torch

from vllm.model_executor.layers.fla.ops import (
    fused_post_conv_read_q,
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

from .decode_batch_state import DecodeBoundary, DecodeCore, PreparedDecode
from .runtime import _env_truthy, _l2norm, _skip_redundant_recurrent_state_store_enabled


class BatchDecodeCoreMixin:
    def _prepare_hsd_batched_decode_core(self, prepared: PreparedDecode) -> DecodeCore:
        anchor_positions = prepared.batch.anchor_positions
        common_history_candidate_tensor = (
            prepared.history.common_history_candidate_tensor
        )
        common_parity_debug = prepared.observers.common_parity_debug
        common_parity_debug_layers = prepared.observers.common_parity_debug_layers
        common_router_project_full_batch = (
            prepared.history.common_router_project_full_batch
        )
        conv_cache = prepared.batch.conv_cache
        hidden_states = prepared.batch.hidden_states
        layout = prepared.batch.layout
        local_k_heads = prepared.batch.local_k_heads
        local_key_dim = prepared.batch.local_key_dim
        local_v_heads = prepared.batch.local_v_heads
        local_value_dim = prepared.batch.local_value_dim
        mark_decode_fine_timing = prepared.observers.mark_decode_fine_timing
        mark_decode_timing = prepared.observers.mark_decode_timing
        maybe_sync_decode_stage = prepared.observers.maybe_sync_decode_stage
        needs_router_q = prepared.policy.needs_router_q
        no_anchor_common_fast_path = prepared.policy.no_anchor_common_fast_path
        padded_static_decode = prepared.batch.padded_static_decode
        past_text_lens = prepared.batch.past_text_lens
        qkv_size = prepared.batch.qkv_size
        record_common_parity_debug = prepared.observers.record_common_parity_debug
        req_count = prepared.batch.req_count
        ssm_state = prepared.batch.ssm_state
        state_indices = prepared.batch.state_indices
        state_indices_tensor = prepared.batch.state_indices_tensor
        total_anchor = prepared.batch.total_anchor
        trace_requested = prepared.policy.trace_requested
        z_size = prepared.batch.z_size

        expanded_hidden = hidden_states[: layout.num_expanded_tokens]
        text_only_identity_layout = (
            total_anchor == 0
            and int(layout.num_expanded_tokens) == req_count
            and int(layout.text_to_expanded.numel()) == req_count
        )
        trace_backend = "batched_decode_light"
        text_hidden = None
        if trace_requested or needs_router_q:
            text_indices = layout.text_to_expanded
            if (
                no_anchor_common_fast_path
                and not trace_requested
                and not padded_static_decode
                and common_history_candidate_tensor is not None
                and not common_router_project_full_batch
            ):
                text_indices = layout.text_to_expanded.index_select(
                    0,
                    common_history_candidate_tensor,
                )
            if text_only_identity_layout and int(text_indices.numel()) == req_count:
                text_hidden = expanded_hidden
            else:
                text_hidden = expanded_hidden.index_select(0, text_indices)
        mark_decode_fine_timing("setup_text_hidden_select")
        fused_router_q_all = None
        fused_null_logits_all = None
        use_decode_projection_fusion = (
            total_anchor == 0
            and no_anchor_common_fast_path
            and text_only_identity_layout
            and not trace_requested
        )
        include_history_projection = bool(
            needs_router_q and text_hidden is expanded_hidden
        )
        fused_projection = (
            self._fused_decode_projections(
                expanded_hidden,
                include_history=include_history_projection,
            )
            if use_decode_projection_fusion
            else None
        )
        if fused_projection is None:
            mixed_qkvz, _ = self.in_proj_qkvz(expanded_hidden)
            mark_decode_fine_timing("proj_qkvz")
            ba, _ = self.in_proj_ba(expanded_hidden)
            mark_decode_fine_timing("proj_ba")
        else:
            mixed_qkvz, ba, fused_router_q_all, fused_null_logits_all = fused_projection
            mark_decode_fine_timing("proj_fused")
        mixed_qkv, z_all_flat = mixed_qkvz.split([qkv_size, z_size], dim=-1)
        b_all, a_all = ba.chunk(2, dim=-1)
        mark_decode_fine_timing("proj_unpack")
        mark_decode_timing("proj")
        maybe_sync_decode_stage("proj")

        if text_only_identity_layout:
            mixed_qkv_text_before_conv = mixed_qkv.contiguous()
            b_text = b_all.contiguous()
            a_text = a_all.contiguous()
        else:
            mixed_qkv_text_before_conv = mixed_qkv.index_select(
                0,
                layout.text_to_expanded,
            ).contiguous()
            b_text = b_all.index_select(0, layout.text_to_expanded).contiguous()
            a_text = a_all.index_select(0, layout.text_to_expanded).contiguous()
        record_common_parity_debug(
            "mixed_qkv_text_before_conv", mixed_qkv_text_before_conv
        )
        z_all = z_all_flat.view(
            layout.num_expanded_tokens,
            local_v_heads,
            self.head_v_dim,
        )
        mark_decode_fine_timing("conv_text_materialize")

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0),
            self.conv1d.weight.size(2),
        )
        trace_pre_conv_inputs = (
            mixed_qkv_text_before_conv.detach().clone() if trace_requested else None
        )
        trace_conv_states_before = (
            conv_cache.index_select(0, state_indices_tensor.long()).detach().clone()
            if trace_requested
            else None
        )
        use_request_loop_conv = self._use_request_loop_conv(req_count)
        if use_request_loop_conv:
            conv_chunks: list[torch.Tensor] = []
            for req_idx, state_idx in enumerate(state_indices):
                conv_chunks.append(
                    self._run_text_conv(
                        mixed_qkv_text_before_conv[req_idx : req_idx + 1]
                        .unsqueeze(0)
                        .contiguous(),
                        state_idx,
                        past_text_lens[req_idx],
                        decode_step=True,
                    ).squeeze(0)
                )
            mixed_qkv_text = torch.cat(conv_chunks, dim=0).contiguous()
        else:
            use_native_conv_state_update = _env_truthy(
                "VLLM_HSD_USE_NATIVE_CONV_STATE_UPDATE"
            )
            if _env_truthy("VLLM_HSD_DISABLE_NATIVE_CONV_STATE_UPDATE"):
                use_native_conv_state_update = False
            validate_native_conv_state_update = _env_truthy(
                "VLLM_HSD_VALIDATE_NATIVE_CONV_STATE_UPDATE"
            )
            need_manual_conv_state_store = not use_native_conv_state_update
            need_conv_state_reference = (
                validate_native_conv_state_update or need_manual_conv_state_store
            )
            conv_state_snapshot = (
                conv_cache.index_select(0, state_indices_tensor.long()).detach().clone()
                if need_conv_state_reference
                else None
            )
            # causal_conv1d_update may overwrite x in-place. Keep a copy only
            # when validation or the legacy manual conv-state store needs it.
            token_inputs_for_state = (
                mixed_qkv_text_before_conv.detach().clone()
                if need_conv_state_reference
                else None
            )
            mixed_qkv_text = causal_conv1d_update(
                mixed_qkv_text_before_conv,
                conv_cache,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=state_indices_tensor,
                validate_data=False,
            ).contiguous()
            if validate_native_conv_state_update:
                assert conv_state_snapshot is not None
                assert token_inputs_for_state is not None
                self._validate_native_decode_conv_state_update(
                    state_indices=state_indices_tensor,
                    conv_state_snapshot=conv_state_snapshot,
                    token_inputs=token_inputs_for_state,
                )
            if use_native_conv_state_update:
                self._record_hsd_varlen_event(
                    "native_conv_state_update_hit",
                    requests=req_count,
                    padded_static_decode=int(padded_static_decode),
                )
            else:
                assert conv_state_snapshot is not None
                assert token_inputs_for_state is not None
                if padded_static_decode:
                    self._store_batched_conv_states_decode_tensor(
                        state_indices=state_indices_tensor,
                        conv_state_snapshot=conv_state_snapshot,
                        token_inputs=token_inputs_for_state,
                    )
                else:
                    self._store_batched_conv_states(
                        state_indices=state_indices,
                        token_offsets=list(range(req_count + 1)),
                        conv_state_snapshot=conv_state_snapshot,
                        token_inputs=token_inputs_for_state,
                    )
            if req_count > 1:
                self._record_hsd_decode_prepare_counter("batched_conv_decode_calls")
                self._record_hsd_decode_prepare_counter(
                    "batched_conv_decode_requests",
                    req_count,
                )
                self._record_hsd_varlen_event(
                    "batched_conv_decode_hit",
                    requests=req_count,
                    text_tokens=req_count,
                )
        record_common_parity_debug("mixed_qkv_text_after_conv", mixed_qkv_text)
        mark_decode_fine_timing("conv_kernel_state")
        mark_decode_timing("conv")
        maybe_sync_decode_stage("conv")

        use_batched_recurrent_decode = (
            self.hsd_decode_backend == "packed"
            and not trace_requested
            and not _env_truthy("VLLM_HSD_DISABLE_BATCHED_RECURRENT_DECODE")
        )
        query_only_post_conv = (
            total_anchor == 0
            and no_anchor_common_fast_path
            and use_batched_recurrent_decode
            and _env_truthy("VLLM_HSD_ENABLE_QUERY_ONLY_POST_CONV")
            and not _env_truthy("VLLM_HSD_DISABLE_QUERY_ONLY_POST_CONV")
        )
        raw_q_paged_read = (
            total_anchor == 0
            and no_anchor_common_fast_path
            and use_batched_recurrent_decode
            and "shared_local" == "shared_local"
            and _env_truthy("VLLM_HSD_ENABLE_RAW_Q_PAGED_READ")
            and not _env_truthy("VLLM_HSD_DISABLE_RAW_Q_PAGED_READ")
        )
        key: torch.Tensor | None
        value: torch.Tensor | None
        g: torch.Tensor | None
        beta: torch.Tensor | None
        if raw_q_paged_read:
            query = None
            key = None
            value = None
            g = None
            beta = None
        elif query_only_post_conv:
            query = fused_post_conv_read_q(
                conv_output=mixed_qkv_text,
                num_k_heads=local_k_heads,
                num_v_heads=local_v_heads,
                head_k_dim=self.head_k_dim,
            )
            key = None
            value = None
            g = None
            beta = None
        else:
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
        mark_decode_fine_timing("recurrent_prep")

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
            anchor_hidden = expanded_hidden.new_zeros(0, expanded_hidden.shape[-1])
            anchor_q_all = expanded_hidden.new_zeros(
                0,
                local_v_heads,
                self.head_k_dim,
            )
        mark_decode_fine_timing("anchor_q_proj")

        anchor_k_all = None
        router_q_all = None
        null_logits_all = None
        if self.hsd_layer_uses_global_read and (needs_router_q or total_anchor):
            if self.anchor_k_proj is None:
                raise RuntimeError("anchor_k_proj is required for HSD global read.")
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
            if needs_router_q:
                mark_decode_fine_timing("anchor_k_proj")
                if self.router_q_proj is None:
                    raise RuntimeError("router_q_proj is required for HSD global read.")
                assert text_hidden is not None
                if fused_router_q_all is not None:
                    router_q_all = fused_router_q_all
                    null_logits_all = fused_null_logits_all
                else:
                    router_q_all, _ = self.router_q_proj(text_hidden)
                    null_logits_all = self._project_null_gate_logits(text_hidden)
                router_batch = int(text_hidden.shape[0])
                router_q_all = router_q_all.view(
                    router_batch,
                    self.hsd_local_route_query_heads,
                    self.hsd_route_head_dim,
                )
                if self.hsd_route_qk_l2norm:
                    router_q_all = _l2norm(router_q_all.contiguous())
                mark_decode_fine_timing("router_q_proj")
                mark_decode_fine_timing("null_gate_proj")
            elif total_anchor:
                mark_decode_fine_timing("anchor_k_proj")

        history_read_query = query

        text_o_all = expanded_hidden.new_empty(
            req_count,
            local_v_heads,
            self.head_v_dim,
        )
        anchor_o_all = expanded_hidden.new_zeros(
            total_anchor,
            local_v_heads,
            self.head_v_dim,
        )
        use_text_only_no_anchor_finalize = (
            no_anchor_common_fast_path
            and text_only_identity_layout
            and not trace_requested
            and _env_truthy("VLLM_HSD_ENABLE_TEXT_ONLY_NO_ANCHOR_FINALIZE")
            and not _env_truthy("VLLM_HSD_DISABLE_TEXT_ONLY_NO_ANCHOR_FINALIZE")
        )
        core_attn_out: torch.Tensor | None = None
        paged_history_pending: list[
            tuple[
                int,
                int,
                int,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
            ]
        ] = []
        skip_redundant_recurrent_state_store = (
            total_anchor == 0
            and use_batched_recurrent_decode
            and not trace_requested
            and no_anchor_common_fast_path
            and _skip_redundant_recurrent_state_store_enabled()
        )
        recurrent_state_all = None
        batched_anchor_boundary = False
        batched_anchor_states_all: torch.Tensor | None = None
        if use_batched_recurrent_decode:
            if common_parity_debug and self.layer_idx in common_parity_debug_layers:
                record_common_parity_debug(
                    "recurrent_state_before",
                    ssm_state.index_select(0, state_indices_tensor.long()),
                )
            recurrent_output_alias = not _env_truthy(
                "VLLM_HSD_DISABLE_RECURRENT_OUTPUT_ALIAS"
            )
            text_o_packed = (
                text_o_all.unsqueeze(1)
                if recurrent_output_alias
                else expanded_hidden.new_empty(
                    req_count,
                    1,
                    local_v_heads,
                    self.head_v_dim,
                )
            )
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=mixed_qkv_text.contiguous(),
                a=a_text.contiguous(),
                b=b_text.contiguous(),
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                scale=self.head_k_dim**-0.5,
                initial_state=ssm_state,
                out=text_o_packed,
                ssm_state_indices=state_indices_tensor,
                use_qk_l2norm_in_kernel=True,
            )
            if recurrent_output_alias:
                self._record_hsd_varlen_event(
                    "recurrent_output_alias_hit",
                    requests=req_count,
                )
            else:
                text_o_all.copy_(text_o_packed[:, 0])
            if not skip_redundant_recurrent_state_store or (
                common_parity_debug and self.layer_idx in common_parity_debug_layers
            ):
                recurrent_state_all = (
                    ssm_state.index_select(
                        0,
                        state_indices_tensor.long(),
                    )
                    .transpose(-1, -2)
                    .contiguous()
                )
        mark_decode_timing("recurrent")
        maybe_sync_decode_stage("recurrent")
        record_common_parity_debug("recurrent_text_o", text_o_all)
        record_common_parity_debug("recurrent_state", recurrent_state_all)
        self._capture_hsd_graph_v3_tensor_sidecar(
            "hsd_recurrent_out",
            text_o_all,
        )

        return DecodeCore(
            a_text=a_text,
            anchor_hidden=anchor_hidden,
            anchor_k_all=anchor_k_all,
            anchor_o_all=anchor_o_all,
            anchor_q_all=anchor_q_all,
            b_text=b_text,
            batched_anchor_boundary=batched_anchor_boundary,
            batched_anchor_states_all=batched_anchor_states_all,
            beta=beta,
            core_attn_out=core_attn_out,
            expanded_hidden=expanded_hidden,
            fused_projection=fused_projection,
            g=g,
            history_read_query=history_read_query,
            include_history_projection=include_history_projection,
            key=key,
            mixed_qkv_text=mixed_qkv_text,
            null_logits_all=null_logits_all,
            paged_history_pending=paged_history_pending,
            query=query,
            query_only_post_conv=query_only_post_conv,
            raw_q_paged_read=raw_q_paged_read,
            recurrent_state_all=recurrent_state_all,
            router_q_all=router_q_all,
            skip_redundant_recurrent_state_store=skip_redundant_recurrent_state_store,
            text_hidden=text_hidden,
            text_o_all=text_o_all,
            trace_backend=trace_backend,
            trace_conv_states_before=trace_conv_states_before,
            trace_pre_conv_inputs=trace_pre_conv_inputs,
            use_batched_recurrent_decode=use_batched_recurrent_decode,
            use_text_only_no_anchor_finalize=use_text_only_no_anchor_finalize,
            value=value,
            z_all=z_all,
            z_all_flat=z_all_flat,
        )

    def _append_hsd_batched_decode_boundaries(
        self, prepared: PreparedDecode, core: DecodeCore
    ) -> DecodeBoundary:
        anchor_counts_by_req = prepared.batch.anchor_counts_by_req
        anchor_k_all = core.anchor_k_all
        anchor_o_all = core.anchor_o_all
        anchor_q_all = core.anchor_q_all
        batched_anchor_boundary = core.batched_anchor_boundary
        batched_anchor_states_all = core.batched_anchor_states_all
        history_read_query = core.history_read_query
        layout = prepared.batch.layout
        local_v_heads = prepared.batch.local_v_heads
        null_logits_all = core.null_logits_all
        recurrent_state_all = core.recurrent_state_all
        req_count = prepared.batch.req_count
        router_q_all = core.router_q_all
        text_o_all = core.text_o_all
        total_anchor = prepared.batch.total_anchor
        trace_requested = prepared.policy.trace_requested
        use_batched_recurrent_decode = core.use_batched_recurrent_decode

        if (
            total_anchor == req_count
            and req_count > 1
            and all(anchor_count == 1 for anchor_count in anchor_counts_by_req)
            and use_batched_recurrent_decode
            and recurrent_state_all is not None
            and not trace_requested
            and self.hsd_layer_uses_global_read
            and anchor_k_all is not None
            and router_q_all is not None
            and history_read_query is not None
            and self.head_k_dim == self.head_v_dim
            and not _env_truthy("VLLM_HSD_DISABLE_BATCHED_ANCHOR_BOUNDARY")
        ):
            page_cache = getattr(self, "hsd_anchor_page_cache", None)
            history_state_indices = tuple(
                self._history_state_index_for_request(layout, req_idx)
                for req_idx in range(req_count)
            )
            first_null_logits = (
                null_logits_all[:1].view(1, 1, local_v_heads, 1)
                if null_logits_all is not None
                else None
            )
            can_use_paged_history = self._can_use_paged_history_read(
                read_q=history_read_query[:1].unsqueeze(0),
                router_q=router_q_all[:1].unsqueeze(0),
                null_logits=first_null_logits,
                trace_requested=False,
            )
            legacy_migrated = page_cache is not None and all(
                self._migrate_legacy_anchor_cache_to_pages(state_idx)
                for state_idx in history_state_indices
            )
            if can_use_paged_history and legacy_migrated:
                batched_anchor_states_all = recurrent_state_all.unsqueeze(1)
                batched_anchor_q = anchor_q_all.view(
                    req_count,
                    1,
                    local_v_heads,
                    self.head_k_dim,
                )
                batched_anchor_o = (
                    torch.einsum(
                        "bmhk,bmhkv->bmhv",
                        batched_anchor_q.to(batched_anchor_states_all.dtype),
                        batched_anchor_states_all,
                    )
                    * (self.head_k_dim**-0.5)
                ).to(text_o_all.dtype)
                assert page_cache is not None
                batched_anchor_boundary = page_cache.try_append_batch_one(
                    history_state_indices,
                    self._anchor_states_to_paged_cache(batched_anchor_states_all[:, 0]),
                    anchor_k_all.view(
                        req_count,
                        local_v_heads,
                        self.head_k_dim,
                    ),
                    layout.anchor_boundaries,
                )
                if batched_anchor_boundary:
                    anchor_o_all.copy_(
                        batched_anchor_o.view(
                            total_anchor,
                            local_v_heads,
                            self.head_v_dim,
                        )
                    )
                    self._record_hsd_varlen_event(
                        "batched_anchor_boundary_hit",
                        requests=req_count,
                        anchor_tokens=total_anchor,
                    )

        return DecodeBoundary(
            batched_anchor_boundary=batched_anchor_boundary,
            batched_anchor_states_all=batched_anchor_states_all,
        )
