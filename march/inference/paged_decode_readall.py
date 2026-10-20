"""MARCH paged decode readall: path implementation, selected by paged_decode."""

from __future__ import annotations

import torch
try:
    import triton
except ImportError:
    triton = None
from .paged_reference import (
    _paged_decode_read_all_history_read_torch,
)
if triton is not None:
    from .paged_triton import (
        _paged_decode_read_all_fused_kernel,
    )
from .paged_utils import (
    _cuda_elapsed_us,
    _next_power_of_2,
    _paged_history_num_warps,
    _value_block_size,
)


def _run_paged_decode_readall(
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
    max_len,
    max_pages,
    null_logits,
    page_table,
    read_q,
    read_q_head_repeat,
    read_q_is_raw,
    read_scale,
    route_dim,
    route_scale,
    router_q,
    start_indices,
    state_indices,
    state_key_dim,
    timing_enabled,
    valid_mask,
    value_dim,
):
    if cache.states_pages is None:
        _record_paged_history_event(
            "paged_readall_fallback", reason="dense_state_required"
        )
        return _fallback("unsupported_inputs")
    if impl == "auto" and (
        int(cache.route_heads) != 1 or int(route_dim) != 64
    ):
        _record_paged_history_event(
            "paged_readall_fallback",
            reason="auto_requires_k1_d64",
            route_heads=int(cache.route_heads),
            route_dim=int(route_dim),
        )
        return _fallback("unsupported_inputs")
    use_readall_triton = impl in {"triton", "fused_triton"} or (
        impl == "auto" and triton is not None and max_len <= 2048
    )
    if not use_readall_triton:
        if read_q_is_raw:
            _record_paged_history_event(
                "paged_readall_fallback", reason="raw_read_q_requires_triton"
            )
            return _fallback("raw_read_query_requires_triton")
        _selected("readall_torch")
        history = _paged_decode_read_all_history_read_torch(
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
        history = _apply_valid_mask(history)
        if fuse_scale_add:
            assert base_out is not None and history_scale is not None
            base_out.add_(
                history.to(base_out.dtype)
                * history_scale.to(
                    device=base_out.device, dtype=base_out.dtype
                ).reshape(1)
            )
            return base_out
        return history
    if triton is None:
        _record_paged_history_event(
            "paged_readall_fallback", reason="triton_unavailable"
        )
        return _fallback("triton_unavailable")

    has_null = null_logits is not None
    if null_logits is None:
        null_logits = read_q.new_empty((1, 1))
    if fuse_scale_add:
        assert base_out is not None and history_scale is not None
        out = base_out
        history_scale = history_scale.to(
            device=read_q.device, dtype=torch.float32
        ).reshape(1)
    else:
        out = (
            torch.zeros(
                batch_size,
                heads,
                value_dim,
                device=read_q.device,
                dtype=read_q.dtype,
            )
            if valid_mask is not None
            else torch.empty(
                batch_size,
                heads,
                value_dim,
                device=read_q.device,
                dtype=read_q.dtype,
            )
        )
        history_scale = read_q.new_empty((1,), dtype=torch.float32)
    block_state_k = _next_power_of_2(state_key_dim)
    block_route_k = _next_power_of_2(route_dim)
    block_v = _value_block_size(value_dim)
    read_q_stride_h = state_key_dim if read_q_is_raw else read_q.stride(1)
    read_q_stride_k = 1 if read_q_is_raw else read_q.stride(2)
    readall_num_warps = _paged_history_num_warps(
        "HSD_PAGED_HISTORY_READALL_NUM_WARPS", default=4
    )
    event_start = None
    event_end = None
    _selected("readall_triton")
    with torch.cuda.device(read_q.device):
        if timing_enabled:
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            event_start.record()
        _paged_decode_read_all_fused_kernel[
            (batch_size, heads, triton.cdiv(value_dim, block_v))
        ](
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
            state_indices,
            history_scale,
            out,
            state_key_dim=state_key_dim,
            route_dim=route_dim,
            value_dim=value_dim,
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
            null_bias_stride_h=state_indices.stride(0),
            out_stride_b=out.stride(0),
            out_stride_h=out.stride(1),
            out_stride_v=out.stride(2),
            max_pages=max_pages,
            state_capacity=0,
            HAS_NULL=has_null,
            HAS_VALID_MASK=valid_mask is not None,
            FUSE_SCALE_ADD=fuse_scale_add,
            STATIC_STATE_METADATA=False,
            IGNORE_BOUNDARY_START=False,
            NORMALIZE_ROUTER_Q=False,
            ADD_NULL_BIAS=False,
            RAW_READ_Q=read_q_is_raw,
            READ_Q_HEAD_REPEAT=read_q_head_repeat,
            ROUTE_HEADS=int(cache.route_heads),
            PAGE_M=int(cache.page_m),
            BLOCK_STATE_K=block_state_k,
            BLOCK_ROUTE_K=block_route_k,
            BLOCK_V=block_v,
            num_warps=readall_num_warps,
        )
        if event_end is not None:
            event_end.record()
    _record_paged_history_event(
        "paged_readall_hit",
        batch=batch_size,
        heads=heads,
        route_heads=int(cache.route_heads),
        route_dim=route_dim,
        state_key_dim=state_key_dim,
        value_dim=value_dim,
        max_anchor_len=max_len,
        cache_pages_used=int(cache.used_pages),
        cache_pages_total=int(cache.total_pages),
        cache_active_states=int(len(cache.entries)),
        cache_allocated_bytes=int(
            cache.total_pages * cache.allocated_page_bytes
        ),
        kernel_us=(
            _cuda_elapsed_us(event_start, event_end)
            if event_start is not None and event_end is not None
            else 0
        ),
    )
    _sync_after_if_requested()
    return out

