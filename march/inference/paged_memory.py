"""MARCH paged memory: execution and capability checks."""

from __future__ import annotations

import torch

try:
    import triton
except ImportError:  # CPU reference remains available without Triton.
    triton = None

from .history_cache import (
    HSDAnchorPageCache,
)
if triton is not None:
    from .paged_triton import (
        _paged_decode_memory_top4_read_kernel,
        _paged_decode_memory_top4_select_kernel,
    )
from .paged_utils import (
    _history_read_num_warps,
    _next_power_of_2,
    _normalize_value_block_size,
    _paged_history_num_warps,
    _value_block_size,
)


def _launch_paged_decode_memory_top4_dense(
    *,
    cache: HSDAnchorPageCache,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    page_table: torch.Tensor,
    anchor_lengths: torch.Tensor,
    start_indices: torch.Tensor,
    state_indices: torch.Tensor,
    valid_mask: torch.Tensor,
    null_logits: torch.Tensor,
    null_bias: torch.Tensor,
    history_scale: torch.Tensor,
    out: torch.Tensor,
    route_scale: float,
    read_scale: float,
    state_capacity: int,
    has_null: bool,
    add_null_bias: bool,
    fuse_scale_add: bool,
    static_state_metadata: bool,
    ignore_boundary_start: bool,
    normalize_router_q: bool,
    read_q_is_raw: bool,
    read_q_head_repeat: int,
    value_block_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the dense fixed-memory split select/read path."""
    if triton is None:
        raise RuntimeError("Persistent-memory paged decode requires Triton.")
    if cache.states_pages is None:
        raise ValueError("Persistent-memory Triton read requires dense pages.")
    assert cache.keys_pages is not None and cache.boundary_pages is not None
    assert cache.key_dim is not None and cache.value_dim is not None
    batch_size, heads = int(router_q.shape[0]), int(router_q.shape[1])
    key_dim = int(cache.key_dim)
    value_dim = int(cache.value_dim)
    max_pages = int(page_table.shape[1])
    top_indices = torch.empty(
        batch_size,
        heads,
        5,
        device=read_q.device,
        dtype=torch.int32,
    )
    top_scores = torch.empty(
        batch_size,
        heads,
        5,
        device=read_q.device,
        dtype=torch.float32,
    )
    block_k = _next_power_of_2(key_dim)
    block_v = (
        _value_block_size(value_dim)
        if value_block_size is None
        else _normalize_value_block_size(
            value_dim, int(value_block_size), "value_block_size"
        )
    )
    block_c = _next_power_of_2(cache.page_m + 4)
    read_q_stride_h = key_dim if read_q_is_raw else read_q.stride(1)
    read_q_stride_k = 1 if read_q_is_raw else read_q.stride(2)
    select_num_warps = _paged_history_num_warps(
        "HSD_PAGED_HISTORY_SELECT_NUM_WARPS"
    )
    read_num_warps = _history_read_num_warps()
    _paged_decode_memory_top4_select_kernel[(batch_size, heads)](
        router_q,
        cache.keys_pages,
        cache.boundary_pages,
        page_table,
        anchor_lengths,
        start_indices,
        state_indices,
        valid_mask,
        top_indices,
        top_scores,
        key_dim=key_dim,
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
        state_capacity=int(state_capacity),
        STATIC_STATE_METADATA=bool(static_state_metadata),
        IGNORE_BOUNDARY_START=bool(ignore_boundary_start),
        NORMALIZE_ROUTER_Q=bool(normalize_router_q),
        PAGE_M=int(cache.page_m),
        BLOCK_K=block_k,
        BLOCK_C=block_c,
        num_warps=select_num_warps,
    )
    _paged_decode_memory_top4_read_kernel[
        (batch_size, heads, triton.cdiv(value_dim, block_v))
    ](
        read_q,
        cache.states_pages,
        page_table,
        state_indices,
        valid_mask,
        null_logits,
        null_bias,
        top_indices,
        top_scores,
        history_scale,
        out,
        key_dim=key_dim,
        value_dim=value_dim,
        read_scale=float(read_scale),
        read_q_stride_b=read_q.stride(0),
        read_q_stride_h=read_q_stride_h,
        read_q_stride_k=read_q_stride_k,
        states_stride_p=cache.states_pages.stride(0),
        states_stride_s=cache.states_pages.stride(1),
        states_stride_h=cache.states_pages.stride(2),
        states_stride_v=cache.states_pages.stride(3),
        states_stride_k=cache.states_pages.stride(4),
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
        state_capacity=int(state_capacity),
        HAS_NULL=bool(has_null),
        ADD_NULL_BIAS=bool(add_null_bias),
        FUSE_SCALE_ADD=bool(fuse_scale_add),
        STATIC_STATE_METADATA=bool(static_state_metadata),
        RAW_READ_Q=bool(read_q_is_raw),
        READ_Q_HEAD_REPEAT=int(read_q_head_repeat),
        PAGE_M=int(cache.page_m),
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        BLOCK_T=8,
        num_warps=read_num_warps,
    )
    return top_indices, top_scores

