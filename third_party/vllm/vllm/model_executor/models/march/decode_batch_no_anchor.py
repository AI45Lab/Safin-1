# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-anchor completion: True/False terminates; None continues with updated states.

Methods preserve arithmetic, checks and cache-write order. No model parameters
or request-state lifecycle are owned by this mixin.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

from . import backends
from .decode_batch_state import DecodeCore, PreparedDecode
from .runtime import _env_truthy

logger = init_logger("vllm.model_executor.models.safin1")


class BatchDecodeNoAnchorMixin:
    def _try_hsd_batched_decode_no_anchor(
        self, prepared: PreparedDecode, core: DecodeCore
    ) -> bool | None:
        common_history_candidate_count = prepared.history.common_history_candidate_count
        common_history_candidate_state_key = (
            prepared.history.common_history_candidate_state_key
        )
        common_history_candidate_tensor = (
            prepared.history.common_history_candidate_tensor
        )
        common_history_use_full_batch = prepared.history.common_history_use_full_batch
        common_history_valid_bools = prepared.history.common_history_valid_bools
        common_history_valid_count = prepared.history.common_history_valid_count
        common_history_valid_tensor = prepared.history.common_history_valid_tensor
        common_router_project_full_batch = (
            prepared.history.common_router_project_full_batch
        )
        core_attn_out = core.core_attn_out
        decode_timing_enabled = prepared.observers.decode_timing_enabled
        decode_timing_payload = prepared.observers.decode_timing_payload
        decode_valid_mask = prepared.batch.decode_valid_mask
        device = prepared.batch.device
        expanded_hidden = core.expanded_hidden
        fallback = prepared.observers.fallback
        fused_projection = core.fused_projection
        graphable_decode_requested = prepared.policy.graphable_decode_requested
        history_read_query = core.history_read_query
        history_state_indices_tensor = prepared.batch.history_state_indices_tensor
        include_history_projection = core.include_history_projection
        layout = prepared.batch.layout
        local_k_heads = prepared.batch.local_k_heads
        local_v_heads = prepared.batch.local_v_heads
        mark_decode_timing = prepared.observers.mark_decode_timing
        max_history_batch = prepared.history.max_history_batch
        maybe_sync_decode_stage = prepared.observers.maybe_sync_decode_stage
        mixed_qkv_text = core.mixed_qkv_text
        no_anchor_common_fast_path = prepared.policy.no_anchor_common_fast_path
        no_anchor_fast_path = prepared.policy.no_anchor_fast_path
        no_anchor_read_only_enabled = prepared.policy.no_anchor_read_only_enabled
        no_anchor_requests = prepared.policy.no_anchor_requests
        null_logits_all = core.null_logits_all
        output = prepared.batch.output
        padded_static_decode = prepared.batch.padded_static_decode
        page_cache_for_common = prepared.history.page_cache_for_common
        past_text_lens_tensor = prepared.batch.past_text_lens_tensor
        persistent_text_cache = prepared.policy.persistent_text_cache
        query_only_post_conv = core.query_only_post_conv
        raw_q_paged_read = core.raw_q_paged_read
        record_common_parity_debug = prepared.observers.record_common_parity_debug
        record_no_anchor_common_fallback = (
            prepared.observers.record_no_anchor_common_fallback
        )
        recurrent_state_all = core.recurrent_state_all
        req_count = prepared.batch.req_count
        router_q_all = core.router_q_all
        skip_redundant_recurrent_state_store = core.skip_redundant_recurrent_state_store
        state_indices = prepared.batch.state_indices
        state_indices_tensor = prepared.batch.state_indices_tensor
        text_o_all = core.text_o_all
        total_anchor = prepared.batch.total_anchor
        use_batched_recurrent_decode = core.use_batched_recurrent_decode
        use_static_sidecar = prepared.policy.use_static_sidecar
        use_text_only_no_anchor_finalize = core.use_text_only_no_anchor_finalize
        z_all = core.z_all

        if no_anchor_common_fast_path and use_batched_recurrent_decode:
            history_candidate_requests = common_history_candidate_count
            if graphable_decode_requested:
                self._record_hsd_varlen_event(
                    "hsd_graphable_decode_candidate",
                    requests=req_count,
                    history_candidate_requests=history_candidate_requests,
                    history_valid_requests=common_history_valid_count,
                )
            mark_decode_timing("request_loop")

            paged_history_out = None
            no_anchor_common_paged_fallback_reason = None
            if history_candidate_requests:
                assert (
                    page_cache_for_common is not None
                    and common_history_candidate_tensor is not None
                    and router_q_all is not None
                )
                loaded = backends._load_hsd_paged_history_read()
                if loaded is None:
                    no_anchor_common_paged_fallback_reason = (
                        "paged_history_loader_missing"
                    )
                    candidate_null_logits = None
                    candidate_valid_mask = None
                else:
                    _, _, paged_decode_read = loaded
                    history_lookup_state_indices = (
                        history_state_indices_tensor
                        if history_state_indices_tensor is not None
                        else state_indices_tensor
                    )
                    candidate_state_indices = history_lookup_state_indices.index_select(
                        0,
                        common_history_candidate_tensor,
                    )
                    candidate_start_indices = past_text_lens_tensor.index_select(
                        0,
                        common_history_candidate_tensor,
                    )
                    if raw_q_paged_read:
                        candidate_read_q = (
                            mixed_qkv_text
                            if common_history_use_full_batch
                            else mixed_qkv_text.index_select(
                                0,
                                common_history_candidate_tensor,
                            ).contiguous()
                        )
                    else:
                        assert history_read_query is not None
                        candidate_read_q = history_read_query.index_select(
                            0,
                            common_history_candidate_tensor,
                        ).contiguous()
                    if common_router_project_full_batch:
                        candidate_router_q = router_q_all.index_select(
                            0,
                            common_history_candidate_tensor,
                        ).contiguous()
                    else:
                        candidate_router_q = router_q_all.contiguous()
                    candidate_null_logits = (
                        null_logits_all.index_select(
                            0,
                            common_history_candidate_tensor,
                        ).contiguous()
                        if (
                            common_router_project_full_batch
                            and null_logits_all is not None
                        )
                        else null_logits_all.contiguous()
                        if null_logits_all is not None
                        else None
                    )
                    if padded_static_decode or common_history_use_full_batch:
                        if common_history_valid_tensor is not None:
                            candidate_valid_mask = common_history_valid_tensor
                        elif common_history_valid_bools is None:
                            candidate_valid_mask = layout.decode_valid_mask
                        else:
                            candidate_valid_mask = torch.tensor(
                                common_history_valid_bools,
                                dtype=torch.bool,
                                device=device,
                            )
                        if (
                            candidate_valid_mask is not None
                            and int(candidate_valid_mask.numel())
                            != history_candidate_requests
                        ):
                            candidate_valid_mask = candidate_valid_mask.index_select(
                                0,
                                common_history_candidate_tensor,
                            )
                        if candidate_valid_mask is not None:
                            candidate_valid_mask = candidate_valid_mask.contiguous()
                    else:
                        candidate_valid_mask = None
                    route_scale = (
                        self.hsd_route_head_dim**-0.5
                        if self.hsd_use_route_scale
                        else 1.0
                    )
                    route_scale *= self.hsd_route_logit_scale
                    record_common_parity_debug(
                        "common_history_state_indices", candidate_state_indices
                    )
                    record_common_parity_debug(
                        "common_history_start_indices", candidate_start_indices
                    )
                    if common_history_candidate_state_key is not None:
                        record_common_parity_debug(
                            "common_history_state_indices_key",
                            torch.tensor(
                                common_history_candidate_state_key,
                                dtype=torch.long,
                                device=device,
                            ),
                        )
                    if hasattr(page_cache_for_common, "lengths"):
                        record_common_parity_debug(
                            "common_history_cache_lengths",
                            page_cache_for_common.lengths(
                                candidate_state_indices,
                                device=device,
                            ),
                        )
                    record_common_parity_debug(
                        "common_history_read_q", candidate_read_q
                    )
                    record_common_parity_debug(
                        "common_history_router_q", candidate_router_q
                    )
                    record_common_parity_debug(
                        "common_history_null_logits", candidate_null_logits
                    )
                    read_batch_limit = max_history_batch or history_candidate_requests
                    fuse_paged_scale_add = (
                        raw_q_paged_read
                        and common_history_use_full_batch
                        and history_candidate_requests == req_count
                        and read_batch_limit >= history_candidate_requests
                        and self.hsd_history_out_rmsnorm_weight is None
                        and _env_truthy("VLLM_HSD_ENABLE_FUSED_PAGED_HISTORY_SCALE_ADD")
                        and not _env_truthy(
                            "VLLM_HSD_DISABLE_FUSED_PAGED_HISTORY_SCALE_ADD"
                        )
                        and not _env_truthy("VLLM_HSD_VALIDATE_FULL_BATCH_HISTORY_READ")
                    )
                    paged_chunks: list[torch.Tensor] = []
                    for chunk_start in range(
                        0,
                        history_candidate_requests,
                        read_batch_limit,
                    ):
                        chunk_end = min(
                            history_candidate_requests,
                            chunk_start + read_batch_limit,
                        )
                        chunk_out = paged_decode_read(
                            cache=page_cache_for_common,
                            read_topk=int(self.hsd_read_topk),
                            read_q=candidate_read_q[chunk_start:chunk_end],
                            router_q=candidate_router_q[chunk_start:chunk_end],
                            state_indices=candidate_state_indices[
                                chunk_start:chunk_end
                            ],
                            start_indices=candidate_start_indices[
                                chunk_start:chunk_end
                            ],
                            route_scale=float(route_scale),
                            read_scale=float(self.head_k_dim**-0.5),
                            null_logits=(
                                candidate_null_logits[chunk_start:chunk_end]
                                if candidate_null_logits is not None
                                else None
                            ),
                            valid_mask=(
                                candidate_valid_mask[chunk_start:chunk_end]
                                if candidate_valid_mask is not None
                                else None
                            ),
                            state_indices_key=(
                                None
                                if common_history_candidate_state_key is None
                                else common_history_candidate_state_key[
                                    chunk_start:chunk_end
                                ]
                            ),
                            read_q_is_raw=raw_q_paged_read,
                            read_q_num_heads=(
                                local_k_heads if raw_q_paged_read else None
                            ),
                            base_out=(text_o_all if fuse_paged_scale_add else None),
                            history_scale=(
                                self._history_scale_tensor(text_o_all.device)
                                if fuse_paged_scale_add
                                else None
                            ),
                            persistent_memory_enabled=(
                                self.hsd_persistent_memory_mode != "removed"
                            ),
                            memory_metrics=(
                                self._paged_persistent_memory_metric_sink(
                                    candidate_read_q,
                                    residual=(
                                        text_o_all
                                        if common_history_use_full_batch
                                        else text_o_all.index_select(
                                            0,
                                            common_history_candidate_tensor,
                                        )
                                    )[chunk_start:chunk_end],
                                )
                            ),
                        )
                        if chunk_out is None:
                            paged_chunks = []
                            break
                        paged_chunks.append(chunk_out)
                    paged_history_out = (
                        torch.cat(paged_chunks, dim=0)
                        if len(paged_chunks) > 1
                        else (paged_chunks[0] if paged_chunks else None)
                    )
                    if paged_history_out is None:
                        no_anchor_common_paged_fallback_reason = (
                            "paged_history_read_none"
                        )
                    else:
                        record_common_parity_debug(
                            "common_history_out_raw", paged_history_out
                        )
                        if (
                            common_history_use_full_batch
                            and common_history_valid_tensor is not None
                            and common_history_valid_count > 0
                            and _env_truthy("VLLM_HSD_VALIDATE_FULL_BATCH_HISTORY_READ")
                        ):
                            valid_subset = common_history_valid_tensor.nonzero(
                                as_tuple=False
                            ).flatten()
                            ref_paged_history_out = paged_decode_read(
                                cache=page_cache_for_common,
                                read_topk=int(self.hsd_read_topk),
                                read_q=candidate_read_q.index_select(0, valid_subset),
                                router_q=candidate_router_q.index_select(
                                    0, valid_subset
                                ),
                                state_indices=candidate_state_indices.index_select(
                                    0, valid_subset
                                ),
                                start_indices=candidate_start_indices.index_select(
                                    0, valid_subset
                                ),
                                route_scale=float(route_scale),
                                read_scale=float(self.head_k_dim**-0.5),
                                null_logits=(
                                    None
                                    if candidate_null_logits is None
                                    else candidate_null_logits.index_select(
                                        0, valid_subset
                                    )
                                ),
                                valid_mask=None,
                                state_indices_key=(
                                    tuple(
                                        int(common_history_candidate_state_key[int(i)])
                                        for i in valid_subset.detach().cpu().tolist()
                                    )
                                    if common_history_candidate_state_key is not None
                                    else None
                                ),
                                read_q_is_raw=raw_q_paged_read,
                                read_q_num_heads=(
                                    local_k_heads if raw_q_paged_read else None
                                ),
                                persistent_memory_enabled=(
                                    self.hsd_persistent_memory_mode != "removed"
                                ),
                                memory_metrics=(
                                    self._paged_persistent_memory_metric_sink(
                                        candidate_read_q
                                    )
                                ),
                            )
                            if ref_paged_history_out is not None:
                                got_paged_history_out = paged_history_out.index_select(
                                    0, valid_subset
                                )
                                diff = (
                                    got_paged_history_out.float()
                                    - ref_paged_history_out.float()
                                ).abs()
                                self._record_hsd_varlen_event(
                                    "full_batch_history_read_validate",
                                    requests=history_candidate_requests,
                                    valid_requests=common_history_valid_count,
                                    mean_abs=float(diff.mean().item()),
                                    max_abs=float(diff.max().item()),
                                )
                                if _env_truthy(
                                    "VLLM_HSD_USE_VALIDATED_SUBSET_HISTORY_READ"
                                ):
                                    corrected = torch.zeros_like(paged_history_out)
                                    corrected.index_copy_(
                                        0,
                                        valid_subset,
                                        ref_paged_history_out.to(dtype=corrected.dtype),
                                    )
                                    paged_history_out = corrected
                        if len(paged_chunks) > 1:
                            self._record_hsd_varlen_event(
                                "paged_history_read_split_hit",
                                requests=history_candidate_requests,
                                valid_requests=common_history_valid_count,
                                chunks=len(paged_chunks),
                                max_history_batch=read_batch_limit,
                            )
                        self._record_hsd_varlen_event(
                            "paged_history_read_hit",
                            requests=history_candidate_requests,
                            valid_requests=common_history_valid_count,
                            has_null_logits=candidate_null_logits is not None,
                            full_batch_history_read=int(common_history_use_full_batch),
                        )
                        paged_history_out = paged_history_out.view(
                            history_candidate_requests,
                            local_v_heads,
                            self.head_v_dim,
                        )
                        fused_history_scale_add = fuse_paged_scale_add
                        if fused_history_scale_add:
                            self._record_hsd_varlen_event(
                                "fused_paged_history_scale_add_hit",
                                requests=history_candidate_requests,
                                valid_requests=common_history_valid_count,
                            )
                        if (
                            not fused_history_scale_add
                            and common_history_use_full_batch
                            and self.hsd_history_out_rmsnorm_weight is None
                            and not _env_truthy(
                                "VLLM_HSD_DISABLE_FUSED_HISTORY_SCALE_ADD"
                            )
                        ):
                            fused_add = backends._load_hsd_history_scale_add()
                            if fused_add is not None:
                                fused_history_scale_add = fused_add(
                                    text_o_all,
                                    paged_history_out,
                                    self._history_scale_tensor(
                                        paged_history_out.device
                                    ),
                                )
                        if fused_history_scale_add:
                            self._record_hsd_varlen_event(
                                "fused_history_scale_add_hit",
                                requests=history_candidate_requests,
                                valid_requests=common_history_valid_count,
                            )
                        else:
                            paged_history_out = self._apply_history_out_rmsnorm(
                                paged_history_out.unsqueeze(1)
                            )
                            scaled = (
                                paged_history_out
                                * self._history_scale_tensor(
                                    paged_history_out.device
                                ).to(dtype=paged_history_out.dtype)
                            ).to(text_o_all.dtype)
                            text_o_all.index_add_(
                                0,
                                common_history_candidate_tensor,
                                scaled[:, 0],
                            )
            mark_decode_timing("paged_read")
            maybe_sync_decode_stage("paged_read")
            if no_anchor_common_paged_fallback_reason is None:
                record_common_parity_debug("common_text_o_after_history", text_o_all)

            if no_anchor_common_paged_fallback_reason is not None:
                record_no_anchor_common_fallback(
                    no_anchor_common_paged_fallback_reason,
                    history_candidate_requests=history_candidate_requests,
                )
                if graphable_decode_requested:
                    self._record_hsd_varlen_event(
                        "hsd_static_decode_fallback",
                        reason=no_anchor_common_paged_fallback_reason,
                        requests=req_count,
                        anchor_tokens=total_anchor,
                    )
                    self._record_hsd_varlen_event(
                        "hsd_graphable_decode_fallback",
                        reason=no_anchor_common_paged_fallback_reason,
                        requests=req_count,
                        anchor_tokens=total_anchor,
                    )
                    if padded_static_decode:
                        self._record_hsd_varlen_event(
                            "hsd_padded_static_decode_fallback",
                            reason=no_anchor_common_paged_fallback_reason,
                            requests=req_count,
                            anchor_tokens=total_anchor,
                        )
            else:
                if skip_redundant_recurrent_state_store:
                    self._record_hsd_varlen_event(
                        "redundant_recurrent_state_store_skip_hit",
                        requests=req_count,
                    )
                else:
                    assert recurrent_state_all is not None
                    if not self._store_recurrent_state_batched(
                        state_indices_tensor,
                        recurrent_state_all,
                    ):
                        for req_idx, state_idx in enumerate(state_indices):
                            self._store_recurrent_state(
                                state_idx,
                                recurrent_state_all[req_idx : req_idx + 1],
                            )
                if persistent_text_cache and use_static_sidecar:
                    if not self._update_hsd_static_text_state_sidecar_batched(
                        state_indices_tensor,
                        decode_valid_mask,
                        past_text_lens_tensor,
                    ):
                        return fallback("static_text_state_sidecar_update_failed")
                elif persistent_text_cache:
                    if not self._update_hsd_text_state_cache_batched(
                        state_indices_tensor,
                    ):
                        for state_idx in state_indices:
                            self._update_hsd_text_state_cache(state_idx)
                maybe_sync_decode_stage("state_store")
                mark_decode_timing("state_store")

                if use_text_only_no_anchor_finalize:
                    core_text_out = text_o_all.reshape(-1, self.head_v_dim)
                    z_text_flat = z_all.reshape(-1, self.head_v_dim)
                    core_text_out = self._apply_hsd_decode_gate_norm(
                        core_text_out,
                        z_text_flat,
                    )
                    core_text_out = core_text_out.reshape(req_count, -1)
                    projected_out, _ = self.out_proj(core_text_out)
                    self._record_hsd_varlen_event(
                        "no_anchor_text_only_finalize_hit",
                        requests=req_count,
                        padded_static_decode=int(padded_static_decode),
                    )
                else:
                    if core_attn_out is None:
                        core_attn_out = expanded_hidden.new_zeros(
                            layout.num_expanded_tokens,
                            local_v_heads,
                            self.head_v_dim,
                        )
                    core_attn_out.index_copy_(0, layout.text_to_expanded, text_o_all)
                    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
                    z_flat = z_all.reshape(-1, self.head_v_dim)
                    core_attn_out = self._apply_hsd_decode_gate_norm(
                        core_attn_out,
                        z_flat,
                    )
                    core_attn_out = core_attn_out.reshape(
                        layout.num_expanded_tokens, -1
                    )
                    projected_out, _ = self.out_proj(core_attn_out)
                if padded_static_decode and layout.decode_valid_mask is not None:
                    projected_out = projected_out.masked_fill(
                        ~layout.decode_valid_mask.view(-1, 1),
                        0,
                    )
                record_common_parity_debug("common_projected_out", projected_out)
                output[: layout.num_expanded_tokens] = projected_out
                mark_decode_timing("out_proj")
                maybe_sync_decode_stage("out_proj")

                logger.info_once("Using HSD no-anchor common batched decode fast path")
                decode_valid_requests = (
                    int(layout.decode_valid_count_value)
                    if padded_static_decode
                    and layout.decode_valid_count_value is not None
                    else req_count
                )
                history_valid_requests = (
                    common_history_valid_count
                    if common_history_valid_tensor is not None
                    or common_history_valid_bools is not None
                    else history_candidate_requests
                )
                no_history_skip_count = max(
                    0,
                    decode_valid_requests - history_valid_requests,
                )
                self._record_hsd_varlen_event(
                    "no_anchor_common_fast_path_hit",
                    requests=req_count,
                    history_candidate_requests=history_candidate_requests,
                    history_valid_requests=history_valid_requests,
                    no_history_skip_requests=no_history_skip_count,
                    full_batch_history_read=int(common_history_use_full_batch),
                )
                self._record_hsd_varlen_event(
                    "batched_decode_light_hit",
                    requests=req_count,
                    anchor_tokens=0,
                )
                self._record_hsd_varlen_event(
                    "batched_recurrent_decode_hit",
                    requests=req_count,
                    anchor_tokens=0,
                )
                if self._use_hsd_persistent_state_sidecar():
                    self._record_hsd_varlen_event(
                        "persistent_state_sidecar_hit",
                        requests=req_count,
                    )
                self._record_hsd_varlen_event(
                    "hsd_static_decode_hit",
                    requests=req_count,
                    history_candidate_requests=history_candidate_requests,
                    history_valid_requests=history_valid_requests,
                    padded_static_decode=int(padded_static_decode),
                    full_batch_history_read=int(common_history_use_full_batch),
                )
                if graphable_decode_requested:
                    self._record_hsd_varlen_event(
                        "hsd_graphable_decode_hit",
                        requests=req_count,
                        history_candidate_requests=history_candidate_requests,
                        history_valid_requests=history_valid_requests,
                    )
                if padded_static_decode:
                    self._record_hsd_varlen_event(
                        "hsd_padded_static_decode_hit",
                        requests=req_count,
                        valid_requests=decode_valid_requests,
                        padded_requests=req_count - decode_valid_requests,
                        history_candidate_requests=history_candidate_requests,
                        history_valid_requests=history_valid_requests,
                    )
                if no_anchor_fast_path and no_anchor_requests:
                    self._record_hsd_varlen_event(
                        "no_anchor_decode_fast_path_hit",
                        requests=req_count,
                        no_anchor_requests=no_anchor_requests,
                        read_only_requests=0,
                        no_history_skip_requests=no_history_skip_count,
                        read_only_enabled=int(no_anchor_read_only_enabled),
                    )
                if fused_projection is not None and self.layer_idx == 0:
                    self._record_hsd_varlen_event(
                        "decode_projection_fusion_hit",
                        requests=req_count,
                        includes_history=int(include_history_projection),
                    )
                if query_only_post_conv and self.layer_idx == 0:
                    self._record_hsd_varlen_event(
                        "query_only_post_conv_hit",
                        requests=req_count,
                    )
                if raw_q_paged_read and self.layer_idx == 0:
                    self._record_hsd_varlen_event(
                        "raw_q_paged_history_read_hit",
                        requests=req_count,
                    )
                if decode_timing_enabled:
                    self._record_hsd_varlen_event(
                        "decode_timing",
                        requests=req_count,
                        anchor_boundary_requests=0,
                        history_candidate_requests=history_candidate_requests,
                        history_valid_requests=history_valid_requests,
                        no_anchor_requests=no_anchor_requests,
                        **decode_timing_payload,
                    )
                return True

        return None
