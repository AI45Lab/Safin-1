"""MARCH paged decode memory: path implementation, selected by paged_decode."""

from __future__ import annotations

import torch
try:
    import triton
except ImportError:
    triton = None
from .paged_memory import (
    _launch_paged_decode_memory_top4_dense,
)
from .paged_reference import (
    _paged_decode_top4_memory_history_read_torch,
    _paged_dynamic_state_rms_median,
    _record_paged_memory_metrics_reference,
)


def _run_paged_decode_memory(
    *,
    _fallback,
    _selected,
    _apply_valid_mask,
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
    memory_metrics,
    null_logits,
    page_table,
    read_q,
    read_q_head_repeat,
    read_q_is_raw,
    read_q_num_heads,
    read_scale,
    route_dim,
    route_scale,
    router_q,
    start_indices,
    state_indices,
    state_key_dim,
    valid_mask,
    value_dim,
):
    if cache.route_heads != heads or route_dim != state_key_dim:
        return _fallback("routing_shape_mismatch")
    use_memory_triton = (
        impl != 'torch' and triton is not None and (cache.states_pages is not None)
    )
    if not use_memory_triton:
        if read_q_is_raw:
            return _fallback("raw_read_query_requires_triton")
        _selected("memory_torch")
        history = _paged_decode_top4_memory_history_read_torch(
            cache=cache,
            read_q=read_q,
            router_q=router_q,
            page_table=page_table,
            anchor_lengths=anchor_lengths,
            start_indices=start_indices,
            route_scale=route_scale,
            read_scale=read_scale,
            null_logits=null_logits,
            memory_metrics=memory_metrics,
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

    has_null = null_logits is not None
    metric_null_logits = null_logits
    if null_logits is None:
        null_logits = read_q.new_empty((1, 1))
    empty_null_bias = read_q.new_empty((1,))
    if fuse_scale_add:
        assert base_out is not None and history_scale is not None
        out = base_out
        kernel_history_scale = history_scale.to(
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
        kernel_history_scale = read_q.new_empty(
            (1,), dtype=torch.float32
        )
    if valid_mask is None:
        kernel_valid_mask = state_indices
    else:
        kernel_valid_mask = valid_mask
    _selected("memory_triton")
    with torch.cuda.device(read_q.device):
        _launch_paged_decode_memory_top4_dense(
            cache=cache,
            read_q=read_q,
            router_q=router_q,
            page_table=page_table,
            anchor_lengths=anchor_lengths,
            start_indices=start_indices,
            state_indices=state_indices,
            valid_mask=kernel_valid_mask,
            null_logits=null_logits,
            null_bias=empty_null_bias,
            history_scale=kernel_history_scale,
            out=out,
            route_scale=route_scale,
            read_scale=read_scale,
            state_capacity=0,
            has_null=has_null,
            add_null_bias=False,
            fuse_scale_add=fuse_scale_add,
            static_state_metadata=False,
            ignore_boundary_start=False,
            normalize_router_q=False,
            read_q_is_raw=read_q_is_raw,
            read_q_head_repeat=read_q_head_repeat,
        )
    _record_paged_memory_metrics_reference(
        cache=cache,
        read_q=read_q,
        router_q=router_q,
        page_table=page_table,
        anchor_lengths=anchor_lengths,
        start_indices=start_indices,
        route_scale=route_scale,
        read_scale=read_scale,
        null_logits=metric_null_logits,
        memory_metrics=memory_metrics,
        read_q_is_raw=read_q_is_raw,
        read_q_num_heads=read_q_num_heads,
        record_dynamic_state_rms_median=False,
    )
    if memory_metrics is not None:
        memory_metrics["dynamic_state_rms_median"] = (
            _paged_dynamic_state_rms_median(
                cache=cache,
                page_table=page_table,
                anchor_lengths=anchor_lengths,
                deduplicate_physical_slots=True,
            )
        )
    if not fuse_scale_add:
        out = _apply_valid_mask(out)
    _sync_after_if_requested()
    return out

