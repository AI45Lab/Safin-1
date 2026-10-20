# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-wise fallback, history merge, output filtering and original state writes.

Methods preserve arithmetic, checks and cache-write order. No model parameters
or request-state lifecycle are owned by this mixin.
"""

from __future__ import annotations

import os
from typing import Literal

import torch

from vllm.logger import init_logger

from . import backends
from .decode_batch_state import (
    DecodeBoundary,
    DecodeCore,
    DecodeReference,
    PreparedDecode,
)
from .layout import _to_cpu_int_list
from .runtime import _env_truthy

logger = init_logger("vllm.model_executor.models.safin1")


class BatchDecodeReferenceMixin:
    def _run_hsd_batched_decode_reference(
        self, prepared: PreparedDecode, core: DecodeCore, boundary: DecodeBoundary
    ) -> DecodeReference | Literal[False]:
        a_text = core.a_text
        anchor_hidden = core.anchor_hidden
        anchor_k_all = core.anchor_k_all
        anchor_o_all = core.anchor_o_all
        anchor_offsets = prepared.batch.anchor_offsets
        anchor_q_all = core.anchor_q_all
        b_text = core.b_text
        batched_anchor_boundary = boundary.batched_anchor_boundary
        batched_anchor_states_all = boundary.batched_anchor_states_all
        beta = core.beta
        boundary_counts_by_req = prepared.batch.boundary_counts_by_req
        device = prepared.batch.device
        expanded_hidden = core.expanded_hidden
        fallback = prepared.observers.fallback
        g = core.g
        global_read_needed_by_req = prepared.policy.global_read_needed_by_req
        history_read_query = core.history_read_query
        history_state_indices_tensor = prepared.batch.history_state_indices_tensor
        key = core.key
        layout = prepared.batch.layout
        local_k_heads = prepared.batch.local_k_heads
        local_key_dim = prepared.batch.local_key_dim
        local_v_heads = prepared.batch.local_v_heads
        local_value_dim = prepared.batch.local_value_dim
        mark_decode_timing = prepared.observers.mark_decode_timing
        mixed_qkv_text = core.mixed_qkv_text
        no_anchor_fast_path = prepared.policy.no_anchor_fast_path
        no_anchor_read_only_enabled = prepared.policy.no_anchor_read_only_enabled
        null_logits_all = core.null_logits_all
        paged_history_pending = core.paged_history_pending
        past_text_lens = prepared.batch.past_text_lens
        persistent_text_cache = prepared.policy.persistent_text_cache
        qkv_size = prepared.batch.qkv_size
        query = core.query
        record_common_parity_debug = prepared.observers.record_common_parity_debug
        recurrent_state_all = core.recurrent_state_all
        req_count = prepared.batch.req_count
        router_q_all = core.router_q_all
        skip_redundant_recurrent_state_store = core.skip_redundant_recurrent_state_store
        state_indices = prepared.batch.state_indices
        state_indices_tensor = prepared.batch.state_indices_tensor
        text_hidden = core.text_hidden
        text_o_all = core.text_o_all
        total_anchor = prepared.batch.total_anchor
        trace_backend = core.trace_backend
        trace_conv_states_before = core.trace_conv_states_before
        trace_pre_conv_inputs = core.trace_pre_conv_inputs
        trace_requested = prepared.policy.trace_requested
        use_batched_recurrent_decode = core.use_batched_recurrent_decode
        value = core.value
        z_all = core.z_all
        z_all_flat = core.z_all_flat
        z_size = prepared.batch.z_size

        no_anchor_batched_history_tensor: torch.Tensor | None = None
        no_anchor_batched_history_count = 0
        use_no_anchor_batched_history_finalize = False
        if (
            total_anchor == 0
            and no_anchor_fast_path
            and not no_anchor_read_only_enabled
            and not trace_requested
            and self.hsd_layer_uses_global_read
            and any(global_read_needed_by_req)
            and _env_truthy("VLLM_HSD_ENABLE_NO_ANCHOR_BATCHED_HISTORY_FINALIZE")
        ):
            page_cache_for_finalize = getattr(self, "hsd_anchor_page_cache", None)
            loaded_finalize = backends._load_hsd_paged_history_read()
            if page_cache_for_finalize is not None and loaded_finalize is not None:
                needed_tensor = torch.tensor(
                    global_read_needed_by_req,
                    dtype=torch.bool,
                    device=device,
                )
                history_lookup_state_indices = (
                    history_state_indices_tensor
                    if history_state_indices_tensor is not None
                    else state_indices_tensor
                )
                if hasattr(page_cache_for_finalize, "lengths"):
                    history_lengths_tensor = page_cache_for_finalize.lengths(
                        history_lookup_state_indices,
                        device=device,
                    ).to(device=device, dtype=torch.long)
                else:
                    history_lookup_state_list = _to_cpu_int_list(
                        history_lookup_state_indices,
                        req_count,
                    )
                    history_lengths_tensor = torch.tensor(
                        [
                            page_cache_for_finalize.length(state_idx)
                            for state_idx in history_lookup_state_list
                        ],
                        device=device,
                        dtype=torch.long,
                    )
                valid_history_tensor = needed_tensor & (history_lengths_tensor > 0)
                needed_count = int(needed_tensor.sum().item())
                valid_count = int(valid_history_tensor.sum().item())
                if needed_count > 0 and valid_count == needed_count:
                    no_anchor_batched_history_tensor = valid_history_tensor.nonzero(
                        as_tuple=False
                    ).flatten()
                    no_anchor_batched_history_count = int(
                        no_anchor_batched_history_tensor.numel()
                    )
                    use_no_anchor_batched_history_finalize = (
                        no_anchor_batched_history_count > 0
                    )
                elif needed_count > 0:
                    self._record_hsd_varlen_event(
                        "no_anchor_batched_history_finalize_fallback",
                        reason="missing_paged_history",
                        requests=req_count,
                        needed_requests=needed_count,
                        valid_requests=valid_count,
                    )
            else:
                self._record_hsd_varlen_event(
                    "no_anchor_batched_history_finalize_fallback",
                    reason="missing_paged_cache",
                    requests=req_count,
                    has_page_cache=page_cache_for_finalize is not None,
                    has_kernel=loaded_finalize is not None,
                )

        batched_state_store_pending = (
            use_batched_recurrent_decode
            and recurrent_state_all is not None
            and not trace_requested
            and not persistent_text_cache
            and not _env_truthy("VLLM_HSD_DISABLE_BATCHED_STATE_STORE")
        )
        if key is None or value is None or g is None or beta is None:
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
        if boundary_counts_by_req is None:
            # Compact decode reaches this loop only after a deliberate common
            # path fallback. Materialize the reference metadata lazily so the
            # successful no-anchor operator island remains list-free.
            empty_boundary_counts = layout.anchor_boundaries[:0]
            boundary_counts_by_req = [empty_boundary_counts] * req_count
        for req_idx in range(req_count):
            state_idx = state_indices[req_idx]
            history_state_idx = self._history_state_index_for_request(layout, req_idx)
            past_text_len = past_text_lens[req_idx]
            anchor_begin = anchor_offsets[req_idx]
            anchor_end = anchor_offsets[req_idx + 1]
            anchor_count = anchor_end - anchor_begin
            boundary_counts = boundary_counts_by_req[req_idx]
            if anchor_count:
                record_common_parity_debug(
                    "anchor_boundary_state_idx",
                    torch.tensor([state_idx], dtype=torch.long, device=device),
                )
                record_common_parity_debug(
                    "anchor_boundary_history_state_idx",
                    torch.tensor([history_state_idx], dtype=torch.long, device=device),
                )

            req_query = query[req_idx : req_idx + 1].unsqueeze(0)
            req_history_read_query = (
                history_read_query[req_idx : req_idx + 1].unsqueeze(0)
                if history_read_query is not None
                else req_query
            )
            req_key = key[req_idx : req_idx + 1].unsqueeze(0)
            req_value = value[req_idx : req_idx + 1].unsqueeze(0)
            req_g = g[req_idx : req_idx + 1].unsqueeze(0)
            req_beta = beta[req_idx : req_idx + 1].unsqueeze(0)
            req_anchor_q = anchor_q_all[anchor_begin:anchor_end].unsqueeze(0)
            req_text_o_modified = False

            if trace_requested:
                assert trace_pre_conv_inputs is not None
                assert trace_conv_states_before is not None
                self._trace_internal(
                    "hsd_past_text_len",
                    expanded_hidden.new_tensor([past_text_len], dtype=torch.int64),
                )
                self._trace_internal(
                    "hsd_state_idx",
                    expanded_hidden.new_tensor([state_idx], dtype=torch.int64),
                )
                self._trace_internal(
                    "hidden_states_text_before_proj",
                    text_hidden[req_idx : req_idx + 1].view(
                        1,
                        1,
                        text_hidden.shape[-1],
                    ),
                )
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="hidden_states_text_before_proj",
                    tensor=text_hidden[req_idx : req_idx + 1].view(
                        1,
                        1,
                        text_hidden.shape[-1],
                    ),
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
                req_pre_conv = trace_pre_conv_inputs[req_idx : req_idx + 1].view(
                    1, 1, qkv_size
                )
                self._trace_internal("qkv_text_before_conv", req_pre_conv)
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="qkv_text_before_conv",
                    tensor=req_pre_conv,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
                req_conv_state_before = trace_conv_states_before[req_idx : req_idx + 1]
                self._trace_internal(
                    "conv_state_before_text_conv",
                    req_conv_state_before,
                )
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="conv_state_before_text_conv",
                    tensor=req_conv_state_before,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
                z_text = z_all_flat.index_select(
                    0,
                    layout.text_to_expanded[req_idx : req_idx + 1],
                ).view(1, 1, z_size)
                qkvzba_text = torch.cat(
                    [
                        req_pre_conv,
                        z_text,
                        b_text[req_idx : req_idx + 1].view(1, 1, -1),
                        a_text[req_idx : req_idx + 1].view(1, 1, -1),
                    ],
                    dim=-1,
                )
                if qkvzba_text.shape[-1] % max(1, local_k_heads) == 0:
                    qkvzba_text = qkvzba_text.view(1, 1, local_k_heads, -1)
                self._trace_internal("qkvzba_text_after_proj", qkvzba_text)
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="qkvzba_text_after_proj",
                    tensor=qkvzba_text,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
                self._trace_internal(
                    "gate_text",
                    z_all.index_select(
                        0, layout.text_to_expanded[req_idx : req_idx + 1]
                    ).view(1, 1, local_v_heads, self.head_v_dim),
                )
                self._trace_internal(
                    "conv_state_after_text_conv",
                    self._conv_state_for_request(state_idx),
                )
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="conv_state_after_text_conv",
                    tensor=self._conv_state_for_request(state_idx),
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
                self._trace_internal(
                    "qkvzba_text_after_conv",
                    mixed_qkv_text[req_idx : req_idx + 1].view(1, 1, qkv_size),
                )
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="qkvzba_text_after_conv",
                    tensor=mixed_qkv_text[req_idx : req_idx + 1].view(1, 1, qkv_size),
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
                self._trace_internal("q_text", req_query)
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="q_text",
                    tensor=req_query,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
                self._trace_internal("k_text", req_key)
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="k_text",
                    tensor=req_key,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
                self._trace_internal("v_text", req_value)
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="v_text",
                    tensor=req_value,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
                self._trace_internal("beta_text", req_beta)
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="beta_text",
                    tensor=req_beta,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
                self._trace_internal("g_text", req_g)
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="g_text",
                    tensor=req_g,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
                self._trace_internal(
                    "anchor_hidden_pre",
                    anchor_hidden[anchor_begin:anchor_end].unsqueeze(0),
                )
                self._trace_internal("anchor_q", req_anchor_q)
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="anchor_q",
                    tensor=req_anchor_q,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
                if self.hsd_layer_uses_global_read and anchor_k_all is not None:
                    req_anchor_k_trace = anchor_k_all[
                        anchor_begin:anchor_end
                    ].unsqueeze(0)
                    self._trace_internal("anchor_k", req_anchor_k_trace)
                    self._trace_decode_stage(
                        backend=trace_backend,
                        stage="anchor_k",
                        tensor=req_anchor_k_trace,
                        req_idx=req_idx,
                        state_idx=state_idx,
                        past_text_len=past_text_len,
                        anchor_count=anchor_count,
                        boundary_counts=boundary_counts,
                    )
                self._trace_internal(
                    "anchor_boundaries",
                    boundary_counts.view(1, -1).to(device=device),
                )
                recurrent_state_before = self._stored_recurrent_state_for_request(
                    state_idx
                )
                if recurrent_state_before is not None:
                    self._trace_decode_stage(
                        backend=trace_backend,
                        stage="recurrent_state_before_decode",
                        tensor=recurrent_state_before,
                        req_idx=req_idx,
                        state_idx=state_idx,
                        past_text_len=past_text_len,
                        anchor_count=anchor_count,
                        boundary_counts=boundary_counts,
                    )

            if use_batched_recurrent_decode:
                req_text_o = text_o_all[req_idx : req_idx + 1].view(
                    1,
                    1,
                    local_v_heads,
                    self.head_v_dim,
                )
                recurrent_state = (
                    None
                    if recurrent_state_all is None
                    else recurrent_state_all[req_idx : req_idx + 1]
                )
                if anchor_count:
                    if recurrent_state is None:
                        return fallback("missing_recurrent_state_for_anchor")
                    expected_boundary = int(past_text_len) + 1
                    expected = torch.full_like(boundary_counts, expected_boundary)
                    if not torch.equal(boundary_counts, expected):
                        return fallback("anchor_boundaries_not_current_token")
                    if batched_anchor_boundary:
                        assert batched_anchor_states_all is not None
                        new_anchor_states = batched_anchor_states_all[
                            req_idx : req_idx + 1
                        ]
                        req_anchor_o = anchor_o_all[anchor_begin:anchor_end].unsqueeze(
                            0
                        )
                    else:
                        new_anchor_states = (
                            recurrent_state.unsqueeze(1)
                            .expand(
                                1,
                                anchor_count,
                                local_v_heads,
                                self.head_k_dim,
                                self.head_v_dim,
                            )
                            .contiguous()
                        )
                        req_anchor_o = (
                            torch.einsum(
                                "bmhk,bmhkv->bmhv",
                                req_anchor_q.to(new_anchor_states.dtype),
                                new_anchor_states,
                            )
                            * (self.head_k_dim**-0.5)
                        ).to(req_text_o.dtype)
                else:
                    req_anchor_o = query.new_zeros(
                        1,
                        0,
                        local_v_heads,
                        self.head_v_dim,
                    )
                    new_anchor_states = query.new_zeros(
                        1,
                        0,
                        local_v_heads,
                        self.head_v_dim,
                        self.head_k_dim,
                    )
            else:
                req_text_o, req_anchor_o, new_anchor_states, recurrent_state = (
                    self._run_hsd_decode_vllm(
                        mixed_qkv_text=mixed_qkv_text[req_idx : req_idx + 1].view(
                            1,
                            1,
                            qkv_size,
                        ),
                        a=a_text[req_idx : req_idx + 1],
                        b=b_text[req_idx : req_idx + 1],
                        query=req_query,
                        key=req_key,
                        value=req_value,
                        g=req_g,
                        beta=req_beta,
                        anchor_q=req_anchor_q,
                        boundary_counts=boundary_counts,
                        boundary_start=past_text_len,
                        state_idx=state_idx,
                    )
                )
            if trace_requested:
                self._trace_internal("text_o_before_history", req_text_o)
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="text_o_before_history",
                    tensor=req_text_o,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
                self._trace_internal("anchor_o", req_anchor_o)
                self._trace_internal("anchor_states", new_anchor_states)
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="anchor_states",
                    tensor=new_anchor_states,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
            if anchor_count and not batched_anchor_boundary:
                anchor_o_all[anchor_begin:anchor_end].copy_(req_anchor_o.squeeze(0))

            if (
                self.hsd_layer_uses_global_read
                and global_read_needed_by_req[req_idx]
                and not use_no_anchor_batched_history_finalize
            ):
                assert router_q_all is not None
                router_q = router_q_all[req_idx : req_idx + 1].unsqueeze(0)
                null_logits = None
                if null_logits_all is not None:
                    null_logits = (
                        null_logits_all[req_idx : req_idx + 1]
                        .view(1, 1, local_v_heads, 1)
                        .contiguous()
                    )
                read_only_history = no_anchor_read_only_enabled and anchor_count == 0
                req_anchor_k = None
                if not read_only_history:
                    assert anchor_k_all is not None
                    req_anchor_k = anchor_k_all[anchor_begin:anchor_end].unsqueeze(0)
                use_paged_history = self._can_use_paged_history_read(
                    read_q=req_history_read_query,
                    router_q=router_q,
                    null_logits=null_logits,
                    trace_requested=trace_requested,
                )
                page_cache = None
                if anchor_count:
                    record_common_parity_debug(
                        "anchor_boundary_can_use_paged",
                        torch.tensor(
                            [int(use_paged_history)],
                            dtype=torch.long,
                            device=device,
                        ),
                    )
                if use_paged_history:
                    if read_only_history:
                        page_cache = getattr(self, "hsd_anchor_page_cache", None)
                        if page_cache is not None:
                            self._migrate_legacy_anchor_cache_to_pages(
                                history_state_idx
                            )
                        if (
                            page_cache is None
                            or page_cache.length(history_state_idx) <= 0
                        ):
                            use_paged_history = False
                    else:
                        if batched_anchor_boundary:
                            page_cache = getattr(self, "hsd_anchor_page_cache", None)
                        else:
                            use_paged_history, page_cache, _ = (
                                self._append_new_anchor_to_history_cache(
                                    state_idx=history_state_idx,
                                    past_text_len=past_text_len,
                                    new_anchor_states=new_anchor_states,
                                    new_anchor_keys=req_anchor_k,
                                    new_anchor_boundaries=boundary_counts,
                                    use_paged_history=True,
                                )
                            )
                            if anchor_count:
                                record_common_parity_debug(
                                    "anchor_boundary_append_succeeded",
                                    torch.tensor(
                                        [int(use_paged_history)],
                                        dtype=torch.long,
                                        device=device,
                                    ),
                                )
                                if page_cache is not None:
                                    record_common_parity_debug(
                                        "anchor_boundary_page_length",
                                        torch.tensor(
                                            [page_cache.length(history_state_idx)],
                                            dtype=torch.long,
                                            device=device,
                                        ),
                                    )
                    if (
                        use_paged_history
                        and page_cache is not None
                        and page_cache.length(history_state_idx) > 0
                    ):
                        null_logits_flat = (
                            null_logits.squeeze(0).squeeze(0).squeeze(-1)
                            if null_logits is not None
                            else None
                        )
                        paged_history_pending.append(
                            (
                                req_idx,
                                history_state_idx,
                                past_text_len,
                                req_history_read_query.squeeze(0).squeeze(0),
                                router_q.squeeze(0).squeeze(0),
                                null_logits_flat,
                            )
                        )
                if not use_paged_history:
                    materialized = None
                    if read_only_history:
                        page_cache = getattr(self, "hsd_anchor_page_cache", None)
                        if (
                            page_cache is not None
                            and page_cache.length(history_state_idx) > 0
                        ):
                            materialized = self._materialize_anchor_page_history(
                                page_cache,
                                history_state_idx,
                                device=device,
                            )
                        if materialized is None:
                            materialized = self._legacy_anchor_history(
                                history_state_idx,
                                device=device,
                            )
                    else:
                        materialized = self._merge_anchor_cache(
                            state_idx=history_state_idx,
                            past_text_len=past_text_len,
                            new_anchor_states=new_anchor_states,
                            new_anchor_keys=req_anchor_k,
                            new_anchor_boundaries=boundary_counts,
                        )
                    if materialized is not None:
                        anchor_states, anchor_keys, anchor_boundaries = materialized
                    else:
                        anchor_states = None
                        anchor_keys = None
                        anchor_boundaries = boundary_counts
                    if (
                        anchor_states is not None
                        and anchor_states.shape[1] > 0
                        and anchor_keys is not None
                    ):
                        if trace_requested:
                            self._trace_internal("router_q", router_q)
                            self._trace_internal(
                                "read_q",
                                req_history_read_query,
                            )
                            self._trace_internal("null_gate_logits", null_logits)
                            self._trace_decode_stage(
                                backend=trace_backend,
                                stage="router_q",
                                tensor=router_q,
                                req_idx=req_idx,
                                state_idx=state_idx,
                                past_text_len=past_text_len,
                                anchor_count=anchor_count,
                                boundary_counts=anchor_boundaries,
                            )
                            self._trace_decode_stage(
                                backend=trace_backend,
                                stage="read_q",
                                tensor=req_history_read_query,
                                req_idx=req_idx,
                                state_idx=state_idx,
                                past_text_len=past_text_len,
                                anchor_count=anchor_count,
                                boundary_counts=anchor_boundaries,
                            )
                            self._trace_decode_stage(
                                backend=trace_backend,
                                stage="null_gate_logits",
                                tensor=null_logits,
                                req_idx=req_idx,
                                state_idx=state_idx,
                                past_text_len=past_text_len,
                                anchor_count=anchor_count,
                                boundary_counts=anchor_boundaries,
                            )
                        history_out = self._history_read(
                            read_q=req_history_read_query,
                            router_q=router_q,
                            states=anchor_states,
                            anchor_k=anchor_keys,
                            boundary_counts=anchor_boundaries,
                            text_len=1,
                            start_index=past_text_len,
                            null_logits=null_logits,
                        )
                        if trace_requested:
                            self._trace_internal("history_out_text", history_out)
                            self._trace_decode_stage(
                                backend=trace_backend,
                                stage="history_out_text",
                                tensor=history_out,
                                req_idx=req_idx,
                                state_idx=state_idx,
                                past_text_len=past_text_len,
                                anchor_count=anchor_count,
                                boundary_counts=anchor_boundaries,
                            )
                        history_out = self._apply_history_out_rmsnorm(history_out)
                        if (
                            trace_requested
                            and self.hsd_history_out_rmsnorm_weight is not None
                        ):
                            self._trace_internal(
                                "history_out_rmsnorm_text", history_out
                            )
                            self._trace_decode_stage(
                                backend=trace_backend,
                                stage="history_out_rmsnorm_text",
                                tensor=history_out,
                                req_idx=req_idx,
                                state_idx=state_idx,
                                past_text_len=past_text_len,
                                anchor_count=anchor_count,
                                boundary_counts=anchor_boundaries,
                            )
                        req_text_o = req_text_o + (
                            history_out
                            * self._history_scale_tensor(history_out.device).to(
                                dtype=history_out.dtype,
                            )
                        ).to(req_text_o.dtype)
                        req_text_o_modified = True

            if trace_requested:
                self._trace_internal("delta_o_text", req_text_o)
                self._trace_internal("recurrent_state_to_store", recurrent_state)
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="delta_o_text",
                    tensor=req_text_o,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="recurrent_state_to_store",
                    tensor=recurrent_state,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
            if (
                not batched_state_store_pending
                and not skip_redundant_recurrent_state_store
            ):
                assert recurrent_state is not None
                self._store_recurrent_state(state_idx, recurrent_state)
            if trace_requested:
                recurrent_state_readback = self._stored_recurrent_state_for_request(
                    state_idx
                )
                self._trace_internal(
                    "recurrent_state_after_store_readback",
                    recurrent_state_readback,
                )
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="recurrent_state_after_store_readback",
                    tensor=recurrent_state_readback,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
            if persistent_text_cache:
                self._update_hsd_text_state_cache(state_idx)
            if not use_batched_recurrent_decode or req_text_o_modified:
                text_o_all[req_idx].copy_(req_text_o.squeeze(0).squeeze(0))

        if batched_state_store_pending:
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
        mark_decode_timing("request_loop")
        if skip_redundant_recurrent_state_store:
            self._record_hsd_varlen_event(
                "redundant_recurrent_state_store_skip_hit",
                requests=req_count,
            )

        return DecodeReference(
            boundary_counts_by_req=boundary_counts_by_req,
            no_anchor_batched_history_count=no_anchor_batched_history_count,
            no_anchor_batched_history_tensor=no_anchor_batched_history_tensor,
            query=query,
            use_no_anchor_batched_history_finalize=use_no_anchor_batched_history_finalize,
        )

    def _finalize_hsd_batched_decode_reference(
        self, prepared: PreparedDecode, core: DecodeCore, reference: DecodeReference
    ) -> bool:
        anchor_counts_by_req = prepared.batch.anchor_counts_by_req
        anchor_o_all = core.anchor_o_all
        anchor_offsets = prepared.batch.anchor_offsets
        anchor_positions = prepared.batch.anchor_positions
        boundary_counts_by_req = reference.boundary_counts_by_req
        core_attn_out = core.core_attn_out
        decode_timing_enabled = prepared.observers.decode_timing_enabled
        decode_timing_payload = prepared.observers.decode_timing_payload
        device = prepared.batch.device
        expanded_hidden = core.expanded_hidden
        history_state_indices_tensor = prepared.batch.history_state_indices_tensor
        layout = prepared.batch.layout
        local_v_heads = prepared.batch.local_v_heads
        mark_decode_timing = prepared.observers.mark_decode_timing
        no_anchor_batched_history_count = reference.no_anchor_batched_history_count
        no_anchor_batched_history_tensor = reference.no_anchor_batched_history_tensor
        no_anchor_fast_path = prepared.policy.no_anchor_fast_path
        no_anchor_read_only_enabled = prepared.policy.no_anchor_read_only_enabled
        no_anchor_requests = prepared.policy.no_anchor_requests
        no_history_skip_requests = prepared.policy.no_history_skip_requests
        null_logits_all = core.null_logits_all
        output = prepared.batch.output
        padded_static_decode = prepared.batch.padded_static_decode
        paged_history_pending = core.paged_history_pending
        past_text_lens = prepared.batch.past_text_lens
        past_text_lens_tensor = prepared.batch.past_text_lens_tensor
        query = reference.query
        read_only_requests = prepared.policy.read_only_requests
        record_common_parity_debug = prepared.observers.record_common_parity_debug
        req_count = prepared.batch.req_count
        router_q_all = core.router_q_all
        state_indices = prepared.batch.state_indices
        state_indices_tensor = prepared.batch.state_indices_tensor
        text_o_all = core.text_o_all
        total_anchor = prepared.batch.total_anchor
        trace_backend = core.trace_backend
        trace_requested = prepared.policy.trace_requested
        use_batched_recurrent_decode = core.use_batched_recurrent_decode
        use_no_anchor_batched_history_finalize = (
            reference.use_no_anchor_batched_history_finalize
        )
        z_all = core.z_all

        if use_no_anchor_batched_history_finalize:
            assert no_anchor_batched_history_tensor is not None
            assert router_q_all is not None
            loaded = backends._load_hsd_paged_history_read()
            page_cache = getattr(self, "hsd_anchor_page_cache", None)
            if loaded is not None and page_cache is not None:
                _, _, paged_decode_read = loaded
                history_lookup_state_indices = (
                    history_state_indices_tensor
                    if history_state_indices_tensor is not None
                    else state_indices_tensor
                )
                pending_req_indices = no_anchor_batched_history_tensor.to(
                    dtype=torch.long,
                    device=device,
                )
                pending_state_indices = history_lookup_state_indices.index_select(
                    0,
                    pending_req_indices,
                )
                pending_start_indices = past_text_lens_tensor.index_select(
                    0,
                    pending_req_indices,
                )
                pending_read_q = query.index_select(
                    0,
                    pending_req_indices,
                ).contiguous()
                pending_router_q = router_q_all.index_select(
                    0,
                    pending_req_indices,
                ).contiguous()
                pending_null_logits = (
                    null_logits_all.index_select(
                        0,
                        pending_req_indices,
                    ).contiguous()
                    if null_logits_all is not None
                    else None
                )
                route_scale = (
                    self.hsd_route_head_dim**-0.5 if self.hsd_use_route_scale else 1.0
                )
                route_scale *= self.hsd_route_logit_scale
                max_history_batch = int(
                    os.environ.get(
                        "HSD_PAGED_HISTORY_MAX_BATCH",
                        str(max(16, no_anchor_batched_history_count)),
                    )
                )
                if max_history_batch <= 0:
                    max_history_batch = max(16, no_anchor_batched_history_count)
                paged_chunks: list[torch.Tensor] = []
                pending_state_key = tuple(
                    int(v) for v in pending_state_indices.detach().cpu().tolist()
                )
                for chunk_start in range(
                    0,
                    no_anchor_batched_history_count,
                    max_history_batch,
                ):
                    chunk_end = min(
                        no_anchor_batched_history_count,
                        chunk_start + max_history_batch,
                    )
                    chunk_out = paged_decode_read(
                        cache=page_cache,
                        read_topk=int(self.hsd_read_topk),
                        read_q=pending_read_q[chunk_start:chunk_end],
                        router_q=pending_router_q[chunk_start:chunk_end],
                        state_indices=pending_state_indices[chunk_start:chunk_end],
                        start_indices=pending_start_indices[chunk_start:chunk_end],
                        route_scale=float(route_scale),
                        read_scale=float(self.head_k_dim**-0.5),
                        null_logits=(
                            pending_null_logits[chunk_start:chunk_end]
                            if pending_null_logits is not None
                            else None
                        ),
                        state_indices_key=pending_state_key[chunk_start:chunk_end],
                        persistent_memory_enabled=(
                            self.hsd_persistent_memory_mode != "removed"
                        ),
                        memory_metrics=self._paged_persistent_memory_metric_sink(
                            pending_read_q,
                            residual=text_o_all.index_select(0, pending_req_indices)[
                                chunk_start:chunk_end
                            ],
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
                if paged_history_out is not None:
                    self._record_hsd_varlen_event(
                        "paged_history_read_hit",
                        requests=no_anchor_batched_history_count,
                        has_null_logits=pending_null_logits is not None,
                    )
                    self._record_hsd_varlen_event(
                        "no_anchor_batched_history_finalize_hit",
                        requests=req_count,
                        history_candidate_requests=(no_anchor_batched_history_count),
                    )
                    paged_history_out = paged_history_out.view(
                        no_anchor_batched_history_count,
                        1,
                        local_v_heads,
                        self.head_v_dim,
                    )
                    self._capture_hsd_graph_v3_tensor_sidecar(
                        "hsd_history_raw",
                        paged_history_out[:, 0],
                    )
                    paged_history_out = self._apply_history_out_rmsnorm(
                        paged_history_out
                    )
                    scaled = (
                        paged_history_out
                        * self._history_scale_tensor(paged_history_out.device).to(
                            dtype=paged_history_out.dtype,
                        )
                    ).to(text_o_all.dtype)
                    text_o_all.index_add_(0, pending_req_indices, scaled[:, 0])
                else:
                    self._record_hsd_varlen_event(
                        "no_anchor_batched_history_finalize_fallback",
                        reason="paged_history_read_none",
                        requests=req_count,
                        history_candidate_requests=(no_anchor_batched_history_count),
                    )
                    fallback_req_indices = _to_cpu_int_list(
                        pending_req_indices,
                        no_anchor_batched_history_count,
                    )
                    fallback_state_indices = _to_cpu_int_list(
                        pending_state_indices,
                        no_anchor_batched_history_count,
                    )
                    fallback_start_indices = _to_cpu_int_list(
                        pending_start_indices,
                        no_anchor_batched_history_count,
                    )
                    for fallback_offset, req_idx in enumerate(fallback_req_indices):
                        state_idx = fallback_state_indices[fallback_offset]
                        materialized = self._materialize_anchor_page_history(
                            page_cache,
                            state_idx,
                            device=device,
                        )
                        if materialized is None:
                            continue
                        anchor_states, anchor_keys, anchor_boundaries = materialized
                        null_logits = (
                            pending_null_logits[
                                fallback_offset : fallback_offset + 1
                            ].view(1, 1, local_v_heads, 1)
                            if pending_null_logits is not None
                            else None
                        )
                        history_out = self._history_read(
                            read_q=pending_read_q[
                                fallback_offset : fallback_offset + 1
                            ].view(1, 1, local_v_heads, self.head_k_dim),
                            router_q=pending_router_q[
                                fallback_offset : fallback_offset + 1
                            ].view(1, 1, local_v_heads, self.head_k_dim),
                            states=anchor_states,
                            anchor_k=anchor_keys,
                            boundary_counts=anchor_boundaries,
                            text_len=1,
                            start_index=fallback_start_indices[fallback_offset],
                            null_logits=null_logits,
                        )
                        history_out = self._apply_history_out_rmsnorm(history_out)
                        text_o_all[req_idx].add_(
                            (
                                history_out.squeeze(0).squeeze(0)
                                * self._history_scale_tensor(history_out.device).to(
                                    dtype=history_out.dtype
                                )
                            ).to(text_o_all.dtype)
                        )

        if paged_history_pending:
            loaded = backends._load_hsd_paged_history_read()
            page_cache = getattr(self, "hsd_anchor_page_cache", None)
            paged_history_out = None
            if loaded is not None and page_cache is not None:
                _, _, paged_decode_read = loaded
                pending_req_indices = torch.tensor(
                    [item[0] for item in paged_history_pending],
                    dtype=torch.long,
                    device=device,
                )
                pending_state_indices = torch.tensor(
                    [item[1] for item in paged_history_pending],
                    dtype=torch.long,
                    device=device,
                )
                pending_start_indices = torch.tensor(
                    [item[2] for item in paged_history_pending],
                    dtype=torch.long,
                    device=device,
                )
                pending_read_q = torch.stack(
                    [item[3] for item in paged_history_pending], dim=0
                ).contiguous()
                pending_router_q = torch.stack(
                    [item[4] for item in paged_history_pending], dim=0
                ).contiguous()
                pending_null_items = [item[5] for item in paged_history_pending]
                pending_null_logits = (
                    torch.stack(
                        [item for item in pending_null_items if item is not None],
                        dim=0,
                    ).contiguous()
                    if all(item is not None for item in pending_null_items)
                    else None
                )
                route_scale = (
                    self.hsd_route_head_dim**-0.5 if self.hsd_use_route_scale else 1.0
                )
                route_scale *= self.hsd_route_logit_scale
                record_common_parity_debug(
                    "reference_history_state_indices", pending_state_indices
                )
                record_common_parity_debug(
                    "reference_history_start_indices", pending_start_indices
                )
                if hasattr(page_cache, "lengths"):
                    record_common_parity_debug(
                        "reference_history_cache_lengths",
                        page_cache.lengths(
                            pending_state_indices,
                            device=device,
                        ),
                    )
                record_common_parity_debug("reference_history_read_q", pending_read_q)
                record_common_parity_debug(
                    "reference_history_router_q", pending_router_q
                )
                record_common_parity_debug(
                    "reference_history_null_logits", pending_null_logits
                )
                paged_history_out = paged_decode_read(
                    cache=page_cache,
                    read_topk=int(self.hsd_read_topk),
                    read_q=pending_read_q,
                    router_q=pending_router_q,
                    state_indices=pending_state_indices,
                    start_indices=pending_start_indices,
                    route_scale=float(route_scale),
                    read_scale=float(self.head_k_dim**-0.5),
                    null_logits=pending_null_logits,
                    state_indices_key=tuple(
                        int(item[1]) for item in paged_history_pending
                    ),
                    persistent_memory_enabled=(
                        self.hsd_persistent_memory_mode != "removed"
                    ),
                    memory_metrics=self._paged_persistent_memory_metric_sink(
                        pending_read_q,
                        residual=text_o_all.index_select(0, pending_req_indices),
                    ),
                )
                if paged_history_out is not None:
                    record_common_parity_debug(
                        "reference_history_out_raw", paged_history_out
                    )
                    self._record_hsd_varlen_event(
                        "paged_history_read_hit",
                        requests=len(paged_history_pending),
                        has_null_logits=pending_null_logits is not None,
                    )
                    paged_history_out = paged_history_out.view(
                        len(paged_history_pending),
                        1,
                        local_v_heads,
                        self.head_v_dim,
                    )
                    paged_history_out = self._apply_history_out_rmsnorm(
                        paged_history_out
                    )
                    scaled = (
                        paged_history_out
                        * self._history_scale_tensor(paged_history_out.device).to(
                            dtype=paged_history_out.dtype,
                        )
                    ).to(text_o_all.dtype)
                    text_o_all.index_add_(0, pending_req_indices, scaled[:, 0])

            if paged_history_out is None:
                for (
                    req_idx,
                    state_idx,
                    past_text_len,
                    read_q_flat,
                    router_q_flat,
                    null_logits_flat,
                ) in paged_history_pending:
                    materialized = (
                        None
                        if page_cache is None
                        else self._materialize_anchor_page_history(
                            page_cache,
                            state_idx,
                            device=device,
                        )
                    )
                    if materialized is None:
                        continue
                    anchor_states, anchor_keys, anchor_boundaries = materialized
                    null_logits = (
                        null_logits_flat.view(1, 1, local_v_heads, 1)
                        if null_logits_flat is not None
                        else None
                    )
                    history_out = self._history_read(
                        read_q=read_q_flat.view(1, 1, local_v_heads, self.head_k_dim),
                        router_q=router_q_flat.view(
                            1,
                            1,
                            self.hsd_local_route_query_heads,
                            self.hsd_route_head_dim,
                        ),
                        states=anchor_states,
                        anchor_k=anchor_keys,
                        boundary_counts=anchor_boundaries,
                        text_len=1,
                        start_index=past_text_len,
                        null_logits=null_logits,
                    )
                    history_out = self._apply_history_out_rmsnorm(history_out)
                    text_o_all[req_idx].add_(
                        (
                            history_out.squeeze(0).squeeze(0)
                            * self._history_scale_tensor(history_out.device).to(
                                dtype=history_out.dtype,
                            )
                        ).to(text_o_all.dtype)
                    )

        record_common_parity_debug("reference_text_o_after_history", text_o_all)
        self._capture_hsd_graph_v3_tensor_sidecar(
            "hsd_history_out",
            text_o_all,
        )
        mark_decode_timing("paged_read")
        if core_attn_out is None:
            core_attn_out = expanded_hidden.new_zeros(
                layout.num_expanded_tokens,
                local_v_heads,
                self.head_v_dim,
            )
        core_attn_out.index_copy_(0, layout.text_to_expanded, text_o_all)
        if total_anchor:
            core_attn_out.index_copy_(0, anchor_positions, anchor_o_all)
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
        core_attn_out = self._apply_hsd_decode_gate_norm(
            core_attn_out,
            z_flat,
        )
        core_attn_out = core_attn_out.reshape(layout.num_expanded_tokens, -1)
        self._capture_hsd_graph_v3_tensor_sidecar(
            "hsd_gate_norm_out",
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
        self._capture_hsd_graph_v3_tensor_sidecar(
            "hsd_projected_out",
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
        if padded_static_decode and layout.decode_valid_mask is not None:
            projected_out = projected_out.masked_fill(
                ~layout.decode_valid_mask.view(-1, 1),
                0,
            )
        record_common_parity_debug("reference_projected_out", projected_out)
        output[: layout.num_expanded_tokens] = projected_out
        mark_decode_timing("out_proj")

        logger.info_once("Using HSD batched decode light prepare/finalize")
        self._record_hsd_decode_prepare_counter("batched_decode_light_calls")
        self._record_hsd_decode_prepare_counter(
            "batched_decode_light_requests",
            req_count,
        )
        self._record_hsd_decode_prepare_counter(
            "batched_decode_light_anchor_tokens",
            total_anchor,
        )
        self._record_hsd_varlen_event(
            "batched_decode_light_hit",
            requests=req_count,
            anchor_tokens=total_anchor,
        )
        if use_batched_recurrent_decode:
            self._record_hsd_varlen_event(
                "batched_recurrent_decode_hit",
                requests=req_count,
                anchor_tokens=total_anchor,
            )
        if no_anchor_fast_path and no_anchor_requests:
            self._record_hsd_varlen_event(
                "no_anchor_decode_fast_path_hit",
                requests=req_count,
                no_anchor_requests=no_anchor_requests,
                read_only_requests=read_only_requests,
                no_history_skip_requests=no_history_skip_requests,
                read_only_enabled=int(no_anchor_read_only_enabled),
            )
            if read_only_requests:
                self._record_hsd_varlen_event(
                    "no_anchor_history_read_only_hit",
                    requests=read_only_requests,
                )
        if decode_timing_enabled:
            self._record_hsd_varlen_event(
                "decode_timing",
                requests=req_count,
                anchor_boundary_requests=sum(
                    1 for count in anchor_counts_by_req if count > 0
                ),
                history_candidate_requests=len(paged_history_pending),
                no_anchor_requests=no_anchor_requests,
                **decode_timing_payload,
            )
        return True
