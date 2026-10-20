"""MARCH paged decode top4: path implementation, selected by paged_decode."""

from __future__ import annotations

import os
import time
import torch
try:
    import triton
except ImportError:
    triton = None
from .paged_reference import (
    _paged_decode_top4_history_read_torch,
)
if triton is not None:
    from .paged_triton import (
        _paged_decode_top4_fused_read_kernel,
        _paged_decode_top4_read_kernel,
        _paged_decode_top4_select_kernel,
    )
from .paged_utils import (
    _cuda_elapsed_us,
    _history_read_num_warps,
    _next_power_of_2,
    _paged_history_num_warps,
    _value_block_size,
)
from march.diagnostics.history_read import (
    compare_fused_history_read,
)


def _run_paged_decode_top4(
    *,
    _fallback,
    _selected,
    _apply_valid_mask,
    _record_paged_history_event,
    _sync_after_if_requested,
    anchor_lengths,
    base_out,
    batch_size,
    cache,
    fuse_scale_add,
    has_null,
    heads,
    history_scale,
    impl,
    max_anchor_len,
    max_len,
    max_pages,
    memory_metrics,
    null_logits,
    original_null_logits,
    page_table,
    paged_batched_decode_history_read,
    persistent_memory_enabled,
    policy,
    read_q,
    read_q_head_repeat,
    read_q_is_raw,
    read_q_num_heads,
    read_scale,
    read_topk,
    route_dim,
    route_scale,
    router_q,
    start_indices,
    state_indices,
    state_indices_key,
    state_key_dim,
    timing_enabled,
    timing_payload,
    valid_mask,
    value_dim,
):
    if cache.route_heads != heads or route_dim != state_key_dim:
        return _fallback("routing_shape_mismatch")
    key_dim = route_dim
    use_triton = impl in {"triton", "fused_triton"} or (
        impl == "auto"
        and triton is not None
        and key_dim <= 128
        and value_dim <= 128
        and max_len <= 2048
    )
    if (read_q_is_raw or fuse_scale_add) and not use_triton:
        return _fallback("triton_required")
    if not use_triton:
        _selected("torch")
        if timing_enabled:
            torch.cuda.synchronize(read_q.device)
            torch_start = time.perf_counter()
        out = _paged_decode_top4_history_read_torch(
            cache=cache,
            read_q=read_q,
            router_q=router_q,
            page_table=page_table,
            anchor_lengths=anchor_lengths,
            start_indices=start_indices,
            route_scale=route_scale,
            read_scale=read_scale,
            null_logits=null_logits,
        )
        if timing_enabled:
            torch.cuda.synchronize(read_q.device)
            timing_payload.update(
                impl=str(impl),
                use_triton=False,
                max_len=int(max_len),
                max_pages=int(max_pages),
                torch_read_us=int(max(0.0, time.perf_counter() - torch_start) * 1_000_000),
                select_kernel_us=0,
                read_kernel_us=0,
            )
            _record_paged_history_event("paged_history_read_timing", **timing_payload)
        out = _apply_valid_mask(out)
        _sync_after_if_requested()
        return out
    if triton is None:
        return _fallback("triton_unavailable")

    if null_logits is None:
        null_logits = read_q.new_empty((1, 1))
    null_bias = read_q.new_empty((1,))
    if fuse_scale_add:
        assert base_out is not None and history_scale is not None
        out = base_out
        history_scale = history_scale.to(
            device=read_q.device, dtype=torch.float32
        ).reshape(1)
    else:
        out = torch.empty(
            batch_size,
            heads,
            value_dim,
            device=read_q.device,
            dtype=read_q.dtype,
        )
        history_scale = read_q.new_empty((1,), dtype=torch.float32)
    block_k = _next_power_of_2(int(key_dim))
    block_v = _value_block_size(value_dim)
    block_c = _next_power_of_2(cache.page_m + 4)
    select_num_warps = _paged_history_num_warps(
        "HSD_PAGED_HISTORY_SELECT_NUM_WARPS"
    )
    read_num_warps = _history_read_num_warps()
    fused_num_warps = _paged_history_num_warps(
        "HSD_PAGED_HISTORY_FUSED_NUM_WARPS"
    )
    num_v_blocks = triton.cdiv(value_dim, block_v)
    fused_max_pages = int(os.environ.get("HSD_PAGED_HISTORY_FUSED_MAX_PAGES", "0"))
    use_fused_triton = impl == "fused_triton" or (
        impl == "auto" and fused_max_pages > 0 and max_pages <= fused_max_pages
    )
    _selected("fused_triton" if use_fused_triton else "triton")
    if use_fused_triton:
        assert cache.states_pages is not None
        read_q_stride_h = key_dim if read_q_is_raw else read_q.stride(1)
        read_q_stride_k = 1 if read_q_is_raw else read_q.stride(2)
        with torch.cuda.device(read_q.device):
            if timing_enabled:
                fused_start = torch.cuda.Event(enable_timing=True)
                fused_end = torch.cuda.Event(enable_timing=True)
                fused_start.record()
            _paged_decode_top4_fused_read_kernel[(batch_size, heads, num_v_blocks)](
                read_q,
                router_q,
                cache.states_pages,
                cache.keys_pages,
                cache.boundary_pages,
                page_table,
                anchor_lengths,
                start_indices,
                state_indices,
                valid_mask if valid_mask is not None else state_indices,
                null_logits,
                null_bias,
                history_scale,
                out,
                key_dim=int(key_dim),
                value_dim=int(value_dim),
                route_scale=float(route_scale),
                read_scale=float(read_scale),
                read_q_stride_b=read_q.stride(0),
                read_q_stride_h=read_q_stride_h,
                read_q_stride_k=read_q_stride_k,
                router_q_stride_b=router_q.stride(0),
                router_q_stride_h=router_q.stride(1),
                router_q_stride_k=router_q.stride(2),
                states_stride_p=cache.states_pages.stride(0),
                states_stride_s=cache.states_pages.stride(1),
                states_stride_h=cache.states_pages.stride(2),
                states_stride_v=cache.states_pages.stride(3),
                states_stride_k=cache.states_pages.stride(4),
                keys_stride_p=cache.keys_pages.stride(0),
                keys_stride_s=cache.keys_pages.stride(1),
                keys_stride_h=cache.keys_pages.stride(2),
                keys_stride_k=cache.keys_pages.stride(3),
                boundary_stride_p=cache.boundary_pages.stride(0),
                boundary_stride_s=cache.boundary_pages.stride(1),
                page_table_stride_b=page_table.stride(0),
                page_table_stride_p=page_table.stride(1),
                null_logits_stride_b=null_logits.stride(0),
                null_logits_stride_h=null_logits.stride(1),
                null_bias_stride_h=null_bias.stride(0),
                out_stride_b=out.stride(0),
                out_stride_h=out.stride(1),
                out_stride_v=out.stride(2),
                max_pages=max_pages,
                state_capacity=0,
                HAS_NULL=has_null,
                FUSE_SCALE_ADD=fuse_scale_add,
                STATIC_STATE_METADATA=False,
                IGNORE_BOUNDARY_START=False,
                NORMALIZE_ROUTER_Q=False,
                ADD_NULL_BIAS=False,
                RAW_READ_Q=read_q_is_raw,
                READ_Q_HEAD_REPEAT=read_q_head_repeat,
                PAGE_M=int(cache.page_m),
                BLOCK_K=block_k,
                BLOCK_V=block_v,
                BLOCK_C=block_c,
                num_warps=fused_num_warps,
            )
            if timing_enabled:
                fused_end.record()
                timing_payload.update(
                    impl=str(impl),
                    use_triton=True,
                    use_fused_triton=True,
                    max_len=int(max_len),
                    max_pages=int(max_pages),
                    torch_read_us=0,
                    select_kernel_us=0,
                    read_kernel_us=0,
                    fused_kernel_us=_cuda_elapsed_us(fused_start, fused_end),
                )
                _record_paged_history_event("paged_history_read_timing", **timing_payload)
        if not fuse_scale_add:
            out = _apply_valid_mask(out)
        _sync_after_if_requested()
        return out

    top_indices = torch.empty(
        batch_size,
        heads,
        4,
        device=read_q.device,
        dtype=torch.int32,
    )
    top_scores = torch.empty(
        batch_size,
        heads,
        4,
        device=read_q.device,
        dtype=torch.float32,
    )
    with torch.cuda.device(read_q.device):
        if timing_enabled:
            select_start = torch.cuda.Event(enable_timing=True)
            select_end = torch.cuda.Event(enable_timing=True)
            select_start.record()
        _paged_decode_top4_select_kernel[(batch_size, heads)](
            router_q,
            cache.keys_pages,
            cache.boundary_pages,
            page_table,
            anchor_lengths,
            start_indices,
            state_indices,
            state_indices,
            top_indices,
            top_scores,
            key_dim=int(key_dim),
            route_scale=float(route_scale),
            router_q_stride_b=router_q.stride(0),
            router_q_stride_h=router_q.stride(1),
            router_q_stride_k=router_q.stride(2),
            keys_stride_p=cache.keys_pages.stride(0),
            keys_stride_s=cache.keys_pages.stride(1),
            keys_stride_h=cache.keys_pages.stride(2),
            keys_stride_k=cache.keys_pages.stride(3),
            boundary_stride_p=cache.boundary_pages.stride(0),
            boundary_stride_s=cache.boundary_pages.stride(1),
            page_table_stride_b=page_table.stride(0),
            page_table_stride_p=page_table.stride(1),
            top_indices_stride_b=top_indices.stride(0),
            top_indices_stride_h=top_indices.stride(1),
            top_indices_stride_t=top_indices.stride(2),
            top_scores_stride_b=top_scores.stride(0),
            top_scores_stride_h=top_scores.stride(1),
            top_scores_stride_t=top_scores.stride(2),
            max_pages=max_pages,
            state_capacity=0,
            STATIC_STATE_METADATA=False,
            IGNORE_BOUNDARY_START=False,
            NORMALIZE_ROUTER_Q=False,
            PAGE_M=int(cache.page_m),
            BLOCK_K=block_k,
            BLOCK_C=block_c,
            num_warps=select_num_warps,
        )
        if timing_enabled:
            select_end.record()
            timing_payload["select_kernel_us"] = _cuda_elapsed_us(select_start, select_end)
            read_start = torch.cuda.Event(enable_timing=True)
            read_end = torch.cuda.Event(enable_timing=True)
            read_start.record()
        read_q_stride_h = key_dim if read_q_is_raw else read_q.stride(1)
        read_q_stride_k = 1 if read_q_is_raw else read_q.stride(2)
        common_read_kwargs = dict(
            key_dim=int(key_dim),
            value_dim=int(value_dim),
            read_scale=float(read_scale),
            read_q_stride_b=read_q.stride(0),
            read_q_stride_h=read_q_stride_h,
            read_q_stride_k=read_q_stride_k,
            page_table_stride_b=page_table.stride(0),
            page_table_stride_p=page_table.stride(1),
            null_logits_stride_b=null_logits.stride(0),
            null_logits_stride_h=null_logits.stride(1),
            top_indices_stride_b=top_indices.stride(0),
            top_indices_stride_h=top_indices.stride(1),
            top_indices_stride_t=top_indices.stride(2),
            top_scores_stride_b=top_scores.stride(0),
            top_scores_stride_h=top_scores.stride(1),
            top_scores_stride_t=top_scores.stride(2),
            out_stride_b=out.stride(0),
            out_stride_h=out.stride(1),
            out_stride_v=out.stride(2),
            max_pages=max_pages,
            state_capacity=0,
            HAS_NULL=has_null,
            ADD_NULL_BIAS=False,
            FUSE_SCALE_ADD=fuse_scale_add,
            STATIC_STATE_METADATA=False,
            RAW_READ_Q=read_q_is_raw,
            READ_Q_HEAD_REPEAT=read_q_head_repeat,
            PAGE_M=int(cache.page_m),
            BLOCK_K=block_k,
            BLOCK_V=block_v,
            num_warps=read_num_warps,
        )
        assert cache.states_pages is not None
        _paged_decode_top4_read_kernel[
            (batch_size, heads, num_v_blocks)
        ](
            read_q,
            cache.states_pages,
            page_table,
            state_indices,
            state_indices,
            null_logits,
            null_bias,
            top_indices,
            top_scores,
            history_scale,
            out,
            states_stride_p=cache.states_pages.stride(0),
            states_stride_s=cache.states_pages.stride(1),
            states_stride_h=cache.states_pages.stride(2),
            states_stride_v=cache.states_pages.stride(3),
            states_stride_k=cache.states_pages.stride(4),
            **common_read_kwargs,
        )
        if timing_enabled:
            read_end.record()
            timing_payload.update(
                impl=str(impl),
                use_triton=True,
                max_len=int(max_len),
                max_pages=int(max_pages),
                torch_read_us=0,
                read_kernel_us=_cuda_elapsed_us(read_start, read_end),
            )
            cache.history_read_time_us += int(timing_payload["read_kernel_us"])
            _record_paged_history_event("paged_history_read_timing", **timing_payload)
    if not fuse_scale_add:
        out = _apply_valid_mask(out)
    _sync_after_if_requested()
    if (
        not use_fused_triton
        and not read_q_is_raw
        and not fuse_scale_add
        and policy.validate_fused
    ):
        compare_fused_history_read(
            out,
            read=paged_batched_decode_history_read,
            read_kwargs=dict(
                cache=cache, read_q=read_q, router_q=router_q,
                state_indices=state_indices, start_indices=start_indices,
                route_scale=route_scale, read_scale=read_scale,
                null_logits=original_null_logits, valid_mask=valid_mask,
                max_anchor_len=max_anchor_len, state_indices_key=state_indices_key,
                read_q_is_raw=read_q_is_raw, read_q_num_heads=read_q_num_heads,
                # This comparison is only eligible without residual fusion.
                # history_scale now holds kernel scratch, not a caller input.
                base_out=None, history_scale=None, read_topk=read_topk,
                persistent_memory_enabled=persistent_memory_enabled,
                memory_metrics=memory_metrics, record_backend=False,
            ),
            record_event=_record_paged_history_event,
            batch=batch_size, max_pages=max_pages, max_len=max_len,
        )
    return out

