"""MARCH paged static: execution and capability checks."""

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
        _paged_decode_read_all_fused_kernel,
        _paged_decode_top4_fused_read_kernel,
        _paged_decode_top4_read_kernel,
        _paged_decode_top4_select_kernel,
    )
from .paged_utils import (
    _env_truthy,
    _history_read_num_warps,
    _next_power_of_2,
    _normalize_value_block_size,
    _paged_history_num_warps,
    _record_prefill_top4_locality,
    _value_block_size,
)
from .paged_memory import _launch_paged_decode_memory_top4_dense


def paged_static_decode_history_read(
    *,
    cache: HSDAnchorPageCache,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    state_indices: torch.Tensor,
    start_indices: torch.Tensor,
    valid_mask: torch.Tensor,
    route_scale: float,
    read_scale: float,
    base_out: torch.Tensor,
    history_scale: torch.Tensor,
    null_logits: torch.Tensor | None = None,
    null_bias: torch.Tensor | None = None,
    read_q_is_raw: bool = False,
    read_q_num_heads: int | None = None,
    normalize_router_q: bool = False,
    value_block_size: int | None = None,
    locality_text_offsets: torch.Tensor | None = None,
    locality_tag: str | None = None,
    locality_max_history_len: int = 0,
    fuse_scale_add: bool = True,
    read_topk: int = 4,
    persistent_memory_enabled: bool = False,
    memory_metrics: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Fixed-shape, capture-safe paged read for Graph V3 decode."""
    if triton is None:
        raise RuntimeError("HSD Graph V3 static history read requires Triton.")
    if (
        not cache.initialized
        or cache.keys_pages is None
        or cache.boundary_pages is None
        or cache._lengths_by_state is None
        or cache._page_table_by_state is None
    ):
        raise RuntimeError("HSD Graph V3 static history metadata is not initialized.")
    if router_q.ndim != 3:
        raise ValueError(
            "Graph V3 router_q must have shape [batch, heads, key_dim], got "
            f"{tuple(router_q.shape)}."
        )
    batch_size, heads, route_dim = router_q.shape
    assert cache.value_dim is not None
    value_dim = cache.value_dim
    state_key_dim = int(cache.key_dim or 0)
    read_topk = int(read_topk)
    if read_topk not in {0, 4}:
        raise ValueError(
            f"Graph V3 static history read supports read_topk=0 or 4, got "
            f"{read_topk}."
        )
    if persistent_memory_enabled:
        if read_topk != 4:
            raise ValueError(
                "Persistent memory requires dynamic Top4 routing (read_topk=4)."
            )
        if cache.states_pages is None:
            raise ValueError(
                "Graph V3 persistent-memory routing currently requires dense "
                "anchor state pages."
            )
    if cache.heads != heads or cache.route_dim != route_dim:
        raise ValueError(
            "Graph V3 history cache shape mismatch: "
            f"cache_heads={cache.heads} query_heads={heads} "
            f"cache_route_dim={cache.route_dim} route_dim={route_dim}."
        )
    if read_topk == 0:
        if cache.states_pages is None:
            raise ValueError("Graph V3 read-all requires dense anchor states.")
        if cache.route_heads not in {1, heads}:
            raise ValueError(
                "Graph V3 read-all requires one shared route-key head or one "
                f"route-key head per query head, got {cache.route_heads}."
            )
    elif cache.route_heads != heads or route_dim != state_key_dim:
        raise ValueError(
            "Graph V3 Top4 requires route heads/dim to match state heads/dim: "
            f"route_heads={cache.route_heads} heads={heads} "
            f"route_dim={route_dim} state_key_dim={state_key_dim}."
        )
    if tuple(base_out.shape) != (batch_size, heads, value_dim):
        raise ValueError(
            "Graph V3 history base_out shape mismatch: "
            f"got={tuple(base_out.shape)} expected={(batch_size, heads, value_dim)}"
        )
    for name, tensor in (
        ("state_indices", state_indices),
        ("start_indices", start_indices),
        ("valid_mask", valid_mask),
    ):
        if tensor.ndim != 1 or int(tensor.shape[0]) != batch_size:
            raise ValueError(
                f"Graph V3 {name} must have shape [{batch_size}], got "
                f"{tuple(tensor.shape)}."
            )
    if any(
        tensor.device != read_q.device
        for tensor in (
            router_q,
            base_out,
            state_indices,
            start_indices,
            valid_mask,
            cache._lengths_by_state,
            cache._page_table_by_state,
        )
    ):
        raise ValueError("Graph V3 history-read tensors must share one device.")
    if fuse_scale_add:
        _record_pre_history_residual_rms(
            memory_metrics=memory_metrics,
            base_out=base_out,
            valid_mask=valid_mask,
        )
    if persistent_memory_enabled and not torch.cuda.is_current_stream_capturing():
        cache.assert_persistent_memory_ready(
            state_indices,
            valid_mask=valid_mask,
        )
    strided_history_inputs = _env_truthy(
        "VLLM_HSD_GRAPH_V3_STRIDED_HISTORY_INPUTS"
    )
    if not (strided_history_inputs and router_q.stride(-1) == 1):
        router_q = router_q.contiguous()
    if not (
        read_q_is_raw
        and _env_truthy("VLLM_HSD_GRAPH_V3_STRIDED_QKV")
        and read_q.stride(-1) == 1
    ):
        read_q = read_q.contiguous()
    capacity = int(cache._lengths_by_state.shape[0])
    static_metadata_fusion = not _env_truthy(
        "VLLM_HSD_DISABLE_STATIC_HISTORY_METADATA_FUSION"
    )
    trust_cache_history_length = _env_truthy(
        "VLLM_HSD_GRAPH_V3_TRUST_CACHE_HISTORY_LENGTH"
    )
    if static_metadata_fusion:
        anchor_lengths = cache._lengths_by_state
        page_table = cache._page_table_by_state
        kernel_start_indices = start_indices
    else:
        state_indices_long = state_indices.to(
            device=read_q.device,
            dtype=torch.long,
        )
        in_range = (state_indices_long >= 0) & (state_indices_long < capacity)
        active = valid_mask.to(device=read_q.device, dtype=torch.bool) & in_range
        safe_indices = torch.where(
            in_range,
            state_indices_long,
            torch.zeros_like(state_indices_long),
        )
        anchor_lengths = cache._lengths_by_state.index_select(0, safe_indices)
        anchor_lengths = torch.where(
            active,
            anchor_lengths,
            torch.zeros_like(anchor_lengths),
        )
        page_table = cache._page_table_by_state.index_select(0, safe_indices)
        page_table = torch.where(
            active.view(-1, 1),
            page_table,
            torch.full_like(page_table, -1),
        )
        kernel_start_indices = torch.where(
            active,
            start_indices.to(device=read_q.device, dtype=torch.long),
            torch.zeros_like(anchor_lengths),
        )

    has_null = null_logits is not None
    add_null_bias = null_bias is not None
    if add_null_bias and not has_null:
        raise ValueError("Graph V3 null_bias requires null_logits.")
    if null_logits is None:
        null_logits = read_q.new_empty((1, 1))
    else:
        if tuple(null_logits.shape) != (batch_size, heads):
            raise ValueError(
                "Graph V3 null_logits shape mismatch: "
                f"got={tuple(null_logits.shape)} expected={(batch_size, heads)}."
            )
        if not (strided_history_inputs and null_logits.stride(-1) == 1):
            null_logits = null_logits.contiguous()
    if null_bias is None:
        null_bias = read_q.new_empty((1,))
    else:
        if null_bias.ndim != 1 or int(null_bias.shape[0]) != heads:
            raise ValueError(
                f"Graph V3 null_bias must have shape [{heads}], got "
                f"{tuple(null_bias.shape)}."
            )
        if null_bias.device != read_q.device:
            raise ValueError("Graph V3 null_bias must share the read_q device.")
        null_bias = null_bias.contiguous()
    read_q_head_repeat = 1
    if read_q_is_raw:
        if read_q_num_heads is None or int(read_q_num_heads) <= 0:
            raise ValueError("read_q_num_heads is required for raw static read_q.")
        if read_q.ndim != 2 or int(read_q.shape[0]) != batch_size:
            raise ValueError(
                "Raw Graph V3 read_q must have shape [batch, packed_qkv], got "
                f"{tuple(read_q.shape)}."
            )
        if heads % int(read_q_num_heads) != 0:
            raise ValueError(
                f"history heads={heads} must be divisible by raw read_q heads="
                f"{int(read_q_num_heads)}."
            )
        read_q_head_repeat = heads // int(read_q_num_heads)
    elif tuple(read_q.shape) != (batch_size, heads, state_key_dim):
        raise ValueError(
            "Graph V3 read_q shape mismatch: "
            f"got={tuple(read_q.shape)} "
            f"expected={(batch_size, heads, state_key_dim)}"
        )
    read_q_stride_h = state_key_dim if read_q_is_raw else read_q.stride(1)
    read_q_stride_k = 1 if read_q_is_raw else read_q.stride(2)
    history_scale = history_scale.to(
        device=read_q.device,
        dtype=torch.float32,
    ).reshape(1)
    block_k = _next_power_of_2(state_key_dim)
    block_v = (
        _value_block_size(value_dim)
        if value_block_size is None
        else _normalize_value_block_size(
            value_dim,
            int(value_block_size),
            "value_block_size",
        )
    )
    block_c = _next_power_of_2(cache.page_m + 4)
    select_num_warps = _paged_history_num_warps(
        "HSD_PAGED_HISTORY_SELECT_NUM_WARPS"
    )
    read_num_warps = _history_read_num_warps()
    fused_num_warps = _paged_history_num_warps(
        "HSD_PAGED_HISTORY_FUSED_NUM_WARPS"
    )
    num_v_blocks = triton.cdiv(value_dim, block_v)
    max_pages = int(page_table.shape[1])
    if max_pages <= 0:
        raise RuntimeError("HSD Graph V3 static page table has no page columns.")
    if persistent_memory_enabled:
        with torch.cuda.device(read_q.device):
            _launch_paged_decode_memory_top4_dense(
                cache=cache,
                read_q=read_q,
                router_q=router_q,
                page_table=page_table,
                anchor_lengths=anchor_lengths,
                start_indices=kernel_start_indices,
                state_indices=state_indices,
                valid_mask=valid_mask,
                null_logits=null_logits,
                null_bias=null_bias,
                history_scale=history_scale,
                out=base_out,
                route_scale=route_scale,
                read_scale=read_scale,
                state_capacity=capacity if static_metadata_fusion else 0,
                has_null=has_null,
                add_null_bias=add_null_bias,
                fuse_scale_add=fuse_scale_add,
                static_state_metadata=static_metadata_fusion,
                ignore_boundary_start=trust_cache_history_length,
                normalize_router_q=normalize_router_q,
                read_q_is_raw=read_q_is_raw,
                read_q_head_repeat=read_q_head_repeat,
                value_block_size=value_block_size,
            )
        if memory_metrics is not None:
            state_indices_long = state_indices.to(
                device=read_q.device, dtype=torch.long
            )
            in_range = (state_indices_long >= 0) & (
                state_indices_long < capacity
            )
            active = valid_mask.to(
                device=read_q.device, dtype=torch.bool
            ) & in_range
            safe_indices = torch.where(
                in_range, state_indices_long, torch.zeros_like(state_indices_long)
            )
            metric_lengths = cache._lengths_by_state.index_select(
                0, safe_indices
            )
            metric_lengths = torch.where(
                active, metric_lengths, torch.zeros_like(metric_lengths)
            )
            metric_table = cache._page_table_by_state.index_select(
                0, safe_indices
            )
            metric_table = torch.where(
                active.view(-1, 1),
                metric_table,
                torch.full_like(metric_table, -1),
            )
            metric_starts = torch.where(
                active,
                start_indices.to(device=read_q.device, dtype=torch.long),
                torch.zeros_like(metric_lengths),
            )
            _record_paged_memory_metrics_reference(
                cache=cache,
                read_q=read_q,
                router_q=router_q,
                page_table=metric_table,
                anchor_lengths=metric_lengths,
                start_indices=metric_starts,
                route_scale=route_scale,
                read_scale=read_scale,
                null_logits=null_logits if has_null else None,
                null_bias=null_bias if add_null_bias else None,
                memory_metrics=memory_metrics,
                read_q_is_raw=read_q_is_raw,
                read_q_num_heads=read_q_num_heads,
                normalize_router_q=normalize_router_q,
                record_dynamic_state_rms_median=False,
            )
            memory_metrics["dynamic_state_rms_median"] = (
                _paged_dynamic_state_rms_median(
                    cache=cache,
                    page_table=metric_table,
                    anchor_lengths=metric_lengths,
                    deduplicate_physical_slots=True,
                )
            )
        return base_out
    if read_topk == 0:
        assert cache.states_pages is not None
        block_route_k = _next_power_of_2(route_dim)
        readall_num_warps = _paged_history_num_warps(
            "HSD_PAGED_HISTORY_READALL_NUM_WARPS", default=4
        )
        with torch.cuda.device(read_q.device):
            _paged_decode_read_all_fused_kernel[
                (batch_size, heads, num_v_blocks)
            ](
                read_q,
                router_q,
                cache.states_pages,
                cache.keys_pages,
                cache.boundary_pages,
                page_table,
                anchor_lengths,
                kernel_start_indices,
                state_indices,
                valid_mask,
                null_logits,
                null_bias,
                history_scale,
                base_out,
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
                null_bias_stride_h=null_bias.stride(0),
                out_stride_b=base_out.stride(0),
                out_stride_h=base_out.stride(1),
                out_stride_v=base_out.stride(2),
                max_pages=max_pages,
                state_capacity=capacity if static_metadata_fusion else 0,
                HAS_NULL=has_null,
                HAS_VALID_MASK=True,
                FUSE_SCALE_ADD=fuse_scale_add,
                STATIC_STATE_METADATA=static_metadata_fusion,
                IGNORE_BOUNDARY_START=trust_cache_history_length,
                NORMALIZE_ROUTER_Q=normalize_router_q,
                ADD_NULL_BIAS=add_null_bias,
                RAW_READ_Q=read_q_is_raw,
                READ_Q_HEAD_REPEAT=read_q_head_repeat,
                ROUTE_HEADS=int(cache.route_heads),
                PAGE_M=int(cache.page_m),
                BLOCK_STATE_K=block_k,
                BLOCK_ROUTE_K=block_route_k,
                BLOCK_V=block_v,
                num_warps=readall_num_warps,
            )
        return base_out
    if _env_truthy('VLLM_HSD_GRAPH_V3_SPLIT_HISTORY_READ'):
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
            _paged_decode_top4_select_kernel[(batch_size, heads)](
                router_q,
                cache.keys_pages,
                cache.boundary_pages,
                page_table,
                anchor_lengths,
                kernel_start_indices,
                state_indices,
                valid_mask,
                top_indices,
                top_scores,
                key_dim=state_key_dim,
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
                state_capacity=capacity if static_metadata_fusion else 0,
                STATIC_STATE_METADATA=static_metadata_fusion,
                IGNORE_BOUNDARY_START=trust_cache_history_length,
                NORMALIZE_ROUTER_Q=normalize_router_q,
                PAGE_M=int(cache.page_m),
                BLOCK_K=block_k,
                BLOCK_C=block_c,
                num_warps=select_num_warps,
            )
            _record_prefill_top4_locality(
                top_indices,
                locality_text_offsets,
                tag=locality_tag,
                max_history_len=int(locality_max_history_len),
            )
            common_read_kwargs = dict(
                key_dim=state_key_dim,
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
                out_stride_b=base_out.stride(0),
                out_stride_h=base_out.stride(1),
                out_stride_v=base_out.stride(2),
                max_pages=max_pages,
                state_capacity=capacity if static_metadata_fusion else 0,
                HAS_NULL=has_null,
                ADD_NULL_BIAS=add_null_bias,
                FUSE_SCALE_ADD=fuse_scale_add,
                STATIC_STATE_METADATA=static_metadata_fusion,
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
                valid_mask,
                null_logits,
                null_bias,
                top_indices,
                top_scores,
                history_scale,
                base_out,
                states_stride_p=cache.states_pages.stride(0),
                states_stride_s=cache.states_pages.stride(1),
                states_stride_h=cache.states_pages.stride(2),
                states_stride_v=cache.states_pages.stride(3),
                states_stride_k=cache.states_pages.stride(4),
                **common_read_kwargs,
            )
        return base_out
    assert cache.states_pages is not None
    with torch.cuda.device(read_q.device):
        _paged_decode_top4_fused_read_kernel[(batch_size, heads, num_v_blocks)](
            read_q,
            router_q,
            cache.states_pages,
            cache.keys_pages,
            cache.boundary_pages,
            page_table,
            anchor_lengths,
            kernel_start_indices,
            state_indices,
            valid_mask,
            null_logits,
            null_bias,
            history_scale,
            base_out,
            key_dim=state_key_dim,
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
            out_stride_b=base_out.stride(0),
            out_stride_h=base_out.stride(1),
            out_stride_v=base_out.stride(2),
            max_pages=max_pages,
            state_capacity=capacity if static_metadata_fusion else 0,
            HAS_NULL=has_null,
            FUSE_SCALE_ADD=fuse_scale_add,
            STATIC_STATE_METADATA=static_metadata_fusion,
            IGNORE_BOUNDARY_START=trust_cache_history_length,
            NORMALIZE_ROUTER_Q=normalize_router_q,
            ADD_NULL_BIAS=add_null_bias,
            RAW_READ_Q=read_q_is_raw,
            READ_Q_HEAD_REPEAT=read_q_head_repeat,
            PAGE_M=int(cache.page_m),
            BLOCK_K=block_k,
            BLOCK_V=block_v,
            BLOCK_C=block_c,
            num_warps=fused_num_warps,
        )
    return base_out

