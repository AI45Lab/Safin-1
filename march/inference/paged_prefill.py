"""MARCH paged prefill: execution and capability checks."""

from __future__ import annotations

import torch

try:
    import triton
except ImportError:  # CPU reference remains available without Triton.
    triton = None

from .history_cache import (
    HSDAnchorPageCache,
)
from .paged_reference import (
    _paged_dynamic_state_rms_median,
    _record_paged_memory_metrics_reference,
    _record_pre_history_residual_rms,
)
if triton is not None:
    from .paged_triton import (
        _paged_decode_top4_select_kernel,
        _paged_prefill_top4_blocked_read_kernel,
        _paged_prefill_top4_selected_blocked_read_kernel,
    )
from .paged_utils import (
    _next_power_of_2,
    _normalize_value_block_size,
    _paged_history_num_warps,
)
from .paged_memory import _launch_paged_decode_memory_top4_dense


def paged_blocked_prefill_history_read(
    *,
    cache: HSDAnchorPageCache,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    token_state_indices: torch.Tensor,
    request_state_indices: torch.Tensor,
    text_offsets: torch.Tensor,
    start_indices: torch.Tensor,
    valid_mask: torch.Tensor,
    max_text_len: int,
    route_scale: float,
    read_scale: float,
    base_out: torch.Tensor,
    history_scale: torch.Tensor,
    null_logits: torch.Tensor | None = None,
    block_t: int = 2,
    value_block_size: int = 32,
    arbitrary_top4: bool = False,
    reuse_uniform_top4: bool = False,
    persistent_memory_enabled: bool = False,
    memory_metrics: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor | None:
    """Blocked prefill read with an optional exact uniform-selection fast path.

    Selection remains token-wise and bit-for-bit identical to the split decode
    path. ``reuse_uniform_top4`` loads each rank's selected state once when all
    valid tokens in a block select the same index; non-uniform blocks execute
    the original per-token state-load implementation.
    """
    if triton is None or not read_q.is_cuda:
        return None
    if (
        not cache.initialized
        or cache.keys_pages is None
        or cache.boundary_pages is None
        or cache._lengths_by_state is None
        or cache._page_table_by_state is None
    ):
        return None
    if read_q.ndim != 3 or router_q.shape != read_q.shape:
        return None
    total_tokens, heads, key_dim = read_q.shape
    assert cache.value_dim is not None
    value_dim = cache.value_dim
    if tuple(base_out.shape) != (total_tokens, heads, value_dim):
        return None
    request_count = int(request_state_indices.numel())
    if (
        request_state_indices.ndim != 1
        or text_offsets.ndim != 1
        or int(text_offsets.numel()) != request_count + 1
        or token_state_indices.ndim != 1
        or int(token_state_indices.numel()) != total_tokens
        or start_indices.ndim != 1
        or int(start_indices.numel()) != total_tokens
        or valid_mask.ndim != 1
        or int(valid_mask.numel()) != total_tokens
        or request_count <= 0
        or int(max_text_len) <= 0
    ):
        return None
    tensors = (
        router_q,
        token_state_indices,
        request_state_indices,
        text_offsets,
        start_indices,
        valid_mask,
        base_out,
        cache._primary_state_pages,
        cache.keys_pages,
        cache.boundary_pages,
        cache._lengths_by_state,
        cache._page_table_by_state,
    )
    if any(tensor.device != read_q.device for tensor in tensors):
        return None
    _record_pre_history_residual_rms(
        memory_metrics=memory_metrics,
        base_out=base_out,
        valid_mask=valid_mask,
    )
    if null_logits is not None and (
        tuple(null_logits.shape) != (total_tokens, heads)
        or null_logits.device != read_q.device
    ):
        return None

    read_q = read_q.contiguous()
    router_q = router_q.contiguous()
    token_state_indices = token_state_indices.to(
        device=read_q.device, dtype=torch.long
    ).contiguous()
    request_state_indices = request_state_indices.to(
        device=read_q.device, dtype=torch.long
    ).contiguous()
    text_offsets = text_offsets.to(
        device=read_q.device, dtype=torch.long
    ).contiguous()
    start_indices = start_indices.to(
        device=read_q.device, dtype=torch.long
    ).contiguous()
    valid_mask = valid_mask.to(
        device=read_q.device, dtype=torch.bool
    ).contiguous()
    if persistent_memory_enabled:
        cache.assert_persistent_memory_ready(request_state_indices)
    has_null = null_logits is not None
    if null_logits is None:
        null_logits = read_q.new_empty((1, 1))
    else:
        null_logits = null_logits.contiguous()
    history_scale = history_scale.to(
        device=read_q.device, dtype=torch.float32
    ).reshape(1)

    block_t = _normalize_value_block_size(
        int(max_text_len), int(block_t), "block_t"
    )
    block_v = _normalize_value_block_size(
        value_dim, int(value_block_size), "value_block_size"
    )
    block_k = _next_power_of_2(int(key_dim))
    block_c = _next_power_of_2(cache.page_m + 4)
    max_pages = int(cache._page_table_by_state.shape[1])
    if max_pages <= 0:
        return None
    # The small-cache kernel materializes only physical slots 0..3. Use the
    # selected-state kernel whenever the metadata can address later slots.
    # A capacity bound is conservative and needs no host readback in a graph.
    arbitrary_top4 = bool(arbitrary_top4 or max_pages * cache.page_m > 4)
    capacity = int(cache._lengths_by_state.shape[0])
    if persistent_memory_enabled:
        in_range = (token_state_indices >= 0) & (
            token_state_indices < capacity
        )
        active = valid_mask & in_range
        safe_token_states = torch.where(
            in_range,
            token_state_indices,
            torch.zeros_like(token_state_indices),
        )
        token_lengths = cache._lengths_by_state.index_select(
            0, safe_token_states
        )
        token_lengths = torch.where(
            active, token_lengths, torch.zeros_like(token_lengths)
        )
        token_page_table = cache._page_table_by_state.index_select(
            0, safe_token_states
        )
        token_page_table = torch.where(
            active.view(-1, 1),
            token_page_table,
            torch.full_like(token_page_table, -1),
        )
        empty_null_bias = read_q.new_empty((1,))
        with torch.cuda.device(read_q.device):
            _launch_paged_decode_memory_top4_dense(
                cache=cache,
                read_q=read_q,
                router_q=router_q,
                page_table=cache._page_table_by_state,
                anchor_lengths=cache._lengths_by_state,
                start_indices=start_indices,
                state_indices=token_state_indices,
                valid_mask=valid_mask,
                null_logits=null_logits,
                null_bias=empty_null_bias,
                history_scale=history_scale,
                out=base_out,
                route_scale=route_scale,
                read_scale=read_scale,
                state_capacity=capacity,
                has_null=has_null,
                add_null_bias=False,
                fuse_scale_add=True,
                static_state_metadata=True,
                ignore_boundary_start=False,
                normalize_router_q=False,
                read_q_is_raw=False,
                read_q_head_repeat=1,
                value_block_size=value_block_size,
            )
        _record_paged_memory_metrics_reference(
            cache=cache,
            read_q=read_q,
            router_q=router_q,
            page_table=token_page_table,
            anchor_lengths=token_lengths,
            start_indices=start_indices,
            route_scale=route_scale,
            read_scale=read_scale,
            null_logits=null_logits if has_null else None,
            memory_metrics=memory_metrics,
            record_dynamic_state_rms_median=False,
        )
        if memory_metrics is not None:
            request_in_range = (request_state_indices >= 0) & (
                request_state_indices < capacity
            )
            safe_request_states = torch.where(
                request_in_range,
                request_state_indices,
                torch.zeros_like(request_state_indices),
            )
            request_lengths = cache._lengths_by_state.index_select(
                0, safe_request_states
            )
            request_lengths = torch.where(
                request_in_range,
                request_lengths,
                torch.zeros_like(request_lengths),
            )
            request_page_table = cache._page_table_by_state.index_select(
                0, safe_request_states
            )
            request_page_table = torch.where(
                request_in_range.view(-1, 1),
                request_page_table,
                torch.full_like(request_page_table, -1),
            )
            memory_metrics["dynamic_state_rms_median"] = (
                _paged_dynamic_state_rms_median(
                    cache=cache,
                    page_table=request_page_table,
                    anchor_lengths=request_lengths,
                    deduplicate_physical_slots=True,
                )
            )
        return base_out
    top_indices = torch.empty(
        total_tokens,
        heads,
        4,
        device=read_q.device,
        dtype=torch.int32,
    )
    top_scores = torch.empty(
        total_tokens,
        heads,
        4,
        device=read_q.device,
        dtype=torch.float32,
    )
    select_num_warps = _paged_history_num_warps(
        "HSD_PAGED_HISTORY_SELECT_NUM_WARPS"
    )
    read_num_warps = _paged_history_num_warps(
        "HSD_PAGED_PREFILL_HISTORY_READ_NUM_WARPS",
        default=4,
    )
    with torch.cuda.device(read_q.device):
        _paged_decode_top4_select_kernel[(total_tokens, heads)](
            router_q,
            cache.keys_pages,
            cache.boundary_pages,
            cache._page_table_by_state,
            cache._lengths_by_state,
            start_indices,
            token_state_indices,
            valid_mask,
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
            page_table_stride_b=cache._page_table_by_state.stride(0),
            page_table_stride_p=cache._page_table_by_state.stride(1),
            top_indices_stride_b=top_indices.stride(0),
            top_indices_stride_h=top_indices.stride(1),
            top_indices_stride_t=top_indices.stride(2),
            top_scores_stride_b=top_scores.stride(0),
            top_scores_stride_h=top_scores.stride(1),
            top_scores_stride_t=top_scores.stride(2),
            max_pages=max_pages,
            state_capacity=capacity,
            STATIC_STATE_METADATA=True,
            IGNORE_BOUNDARY_START=False,
            NORMALIZE_ROUTER_Q=False,
            PAGE_M=int(cache.page_m),
            BLOCK_K=block_k,
            BLOCK_C=block_c,
            num_warps=select_num_warps,
        )
        read_kernel = (
            _paged_prefill_top4_selected_blocked_read_kernel
            if arbitrary_top4
            else _paged_prefill_top4_blocked_read_kernel
        )
        uniform_modes = (1, 2) if arbitrary_top4 and reuse_uniform_top4 else (0,)
        for uniform_mode in uniform_modes:
            assert cache.states_pages is not None
            read_kernel[
                (
                    request_count * heads,
                    triton.cdiv(int(max_text_len), block_t),
                    triton.cdiv(value_dim, block_v),
                )
            ](
                read_q,
                cache.states_pages,
                cache._page_table_by_state,
                cache._lengths_by_state,
                request_state_indices,
                text_offsets,
                null_logits,
                top_indices,
                top_scores,
                history_scale,
                base_out,
                heads=int(heads),
                key_dim=int(key_dim),
                value_dim=value_dim,
                read_scale=float(read_scale),
                read_q_stride_b=read_q.stride(0),
                read_q_stride_h=read_q.stride(1),
                read_q_stride_k=read_q.stride(2),
                states_stride_p=cache.states_pages.stride(0),
                states_stride_s=cache.states_pages.stride(1),
                states_stride_h=cache.states_pages.stride(2),
                states_stride_v=cache.states_pages.stride(3),
                states_stride_k=cache.states_pages.stride(4),
                page_table_stride_b=cache._page_table_by_state.stride(0),
                page_table_stride_p=cache._page_table_by_state.stride(1),
                null_logits_stride_b=null_logits.stride(0),
                null_logits_stride_h=null_logits.stride(1),
                top_indices_stride_b=top_indices.stride(0),
                top_indices_stride_h=top_indices.stride(1),
                top_indices_stride_t=top_indices.stride(2),
                top_scores_stride_b=top_scores.stride(0),
                top_scores_stride_h=top_scores.stride(1),
                top_scores_stride_t=top_scores.stride(2),
                out_stride_b=base_out.stride(0),
                out_stride_h=base_out.stride(1),
                out_stride_v=base_out.stride(2),
                max_pages=max_pages,
                state_capacity=capacity,
                HAS_NULL=has_null,
                PAGE_M=int(cache.page_m),
                BLOCK_T=block_t,
                BLOCK_K=block_k,
                BLOCK_V=block_v,
                UNIFORM_MODE=uniform_mode,
                num_warps=read_num_warps,
            )
    return base_out

