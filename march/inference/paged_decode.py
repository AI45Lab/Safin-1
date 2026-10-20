"""MARCH paged decode: execution and capability checks."""

from __future__ import annotations

import os
import time
import torch

try:
    import triton
except ImportError:  # CPU reference remains available without Triton.
    triton = None

from .history_cache import (
    HSDAnchorPageCache,
)
from .paged_reference import (
    _paged_decode_read_all_history_read_torch,
    _paged_decode_top4_history_read_torch,
    _paged_decode_top4_memory_history_read_torch,
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
    _cuda_elapsed_us,
    _env_truthy,
    _history_read_num_warps,
    _next_power_of_2,
    _paged_history_num_warps,
    _paged_history_timing_enabled,
    _record_paged_history_event,
    _value_block_size,
)
from .paged_memory import _launch_paged_decode_memory_top4_dense

from .backend_policy import PagedHistoryReadPolicy
from march.diagnostics.history_read import compare_fused_history_read

from .paged_decode_memory import _run_paged_decode_memory
from .paged_decode_readall import _run_paged_decode_readall
from .paged_decode_top4 import _run_paged_decode_top4


def paged_batched_decode_history_read(
    *,
    cache: HSDAnchorPageCache,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    state_indices: torch.Tensor,
    start_indices: torch.Tensor,
    route_scale: float,
    read_scale: float,
    null_logits: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    max_anchor_len: int | None = None,
    state_indices_key: tuple[int, ...] | None = None,
    read_q_is_raw: bool = False,
    read_q_num_heads: int | None = None,
    base_out: torch.Tensor | None = None,
    history_scale: torch.Tensor | None = None,
    read_topk: int = 4,
    persistent_memory_enabled: bool = False,
    memory_metrics: dict[str, torch.Tensor] | None = None,
    backend: str | None = None,
    validate_fused: bool | None = None,
    record_backend: bool | None = None,
) -> torch.Tensor | None:
    """Return decode history, or ``None`` when the requested path is unsupported.

    Explicit backend/validation arguments override compatibility environment
    defaults. Split batches inherit the same resolved policy. Static graph
    decode has its own fixed-shape entrypoint and does not use this selector.
    """
    policy = PagedHistoryReadPolicy.resolve(backend=backend, validate_fused=validate_fused,
                                            record_backend=record_backend)

    def _fallback(reason: str) -> None:
        if policy.record_backend:
            _record_paged_history_event("paged_history_backend", status="fallback",
                                        requested=policy.backend, reason=reason,
                                        read_topk=int(read_topk), persistent_memory=bool(persistent_memory_enabled))
        return None

    def _selected(path: str) -> None:
        if policy.record_backend:
            _record_paged_history_event("paged_history_backend", status="selected",
                                        requested=policy.backend, path=path,
                                        read_topk=int(read_topk), persistent_memory=bool(persistent_memory_enabled))

    def _apply_valid_mask(out: torch.Tensor) -> torch.Tensor:
        if valid_mask is None:
            return out
        return out * valid_mask.to(dtype=out.dtype).view(batch_size, 1, 1)

    def _sync_after_if_requested() -> None:
        if _env_truthy("HSD_PAGED_HISTORY_SYNC_AFTER"):
            torch.cuda.synchronize(read_q.device)

    if os.environ.get("HSD_DISABLE_PAGED_HISTORY_READ", "").lower() in {"1", "true", "yes", "on"}:
        return _fallback("disabled")
    if not read_q.is_cuda:
        return _fallback("cuda_required")
    if router_q.ndim != 3:
        return _fallback("invalid_router_query")
    batch_size, heads, route_dim = router_q.shape
    state_key_dim = int(cache.key_dim or 0)
    read_q_head_repeat = 1
    if read_q_is_raw:
        if (
            read_q.ndim != 2
            or int(read_q.shape[0]) != batch_size
            or read_q_num_heads is None
            or int(read_q_num_heads) <= 0
            or heads % int(read_q_num_heads) != 0
            or int(read_q.shape[1]) < int(read_q_num_heads) * state_key_dim
        ):
            return _fallback("invalid_batch_shape")
        read_q_head_repeat = heads // int(read_q_num_heads)
    elif (
        read_q.ndim != 3
        or tuple(read_q.shape[:2]) != (batch_size, heads)
        or int(read_q.shape[2]) != state_key_dim
    ):
        return _fallback("invalid_batch_shape")
    if not cache.initialized or cache.keys_pages is None or cache.boundary_pages is None:
        return _fallback("cache_uninitialized")
    if cache.device != read_q.device or cache.keys_pages.device != read_q.device:
        return _fallback("device_mismatch")
    if (
        cache.value_dim is None
        or cache.heads is None
        or cache.key_dim is None
        or cache.route_heads is None
        or cache.route_dim is None
    ):
        return _fallback("cache_shape_uninitialized")
    value_dim = cache.value_dim
    if (
        cache.heads != heads
        or cache.route_heads not in {1, heads}
        or cache.route_dim != route_dim
    ):
        return _fallback("routing_shape_mismatch")
    if state_key_dim > 128 or route_dim > 128 or value_dim > 128:
        return _fallback("unsupported_geometry")
    if int(read_topk) not in {0, 4}:
        return _fallback("unsupported_topk")
    if persistent_memory_enabled:
        if int(read_topk) != 4:
            raise ValueError(
                "Persistent memory requires dynamic Top4 routing (read_topk=4)."
            )
    fuse_scale_add = base_out is not None or history_scale is not None
    if fuse_scale_add:
        if base_out is None or history_scale is None:
            return _fallback("unsupported_inputs")
        if (
            tuple(base_out.shape) != (batch_size, heads, value_dim)
            or base_out.device != read_q.device
            or base_out.dtype != read_q.dtype
            or not base_out.is_contiguous()
            or history_scale.numel() != 1
            or history_scale.device != read_q.device
        ):
            return _fallback("invalid_batch_shape")
    if state_indices.ndim != 1 or start_indices.ndim != 1:
        return _fallback("invalid_state_indices")
    if int(state_indices.shape[0]) != batch_size or int(start_indices.shape[0]) != batch_size:
        return _fallback("invalid_state_count")
    if null_logits is not None:
        if null_logits.ndim != 2 or tuple(null_logits.shape) != (batch_size, heads):
            return _fallback("invalid_null_logits")
        if null_logits.device != read_q.device:
            return _fallback("invalid_null_logits")
    if valid_mask is not None:
        if valid_mask.ndim != 1 or int(valid_mask.shape[0]) != batch_size:
            return _fallback("invalid_valid_mask")
        if valid_mask.device != read_q.device:
            return _fallback("invalid_valid_mask")
    if fuse_scale_add:
        assert base_out is not None
        _record_pre_history_residual_rms(
            memory_metrics=memory_metrics,
            base_out=base_out,
            valid_mask=valid_mask,
        )

    timing_enabled = _paged_history_timing_enabled()
    timing_payload: dict[str, int | float | str | bool] = {}
    if timing_enabled:
        timing_payload.update(
            batch=int(batch_size),
            heads=int(heads),
            state_key_dim=int(state_key_dim),
            route_dim=int(route_dim),
            route_heads=int(cache.route_heads),
            value_dim=int(value_dim),
            read_topk=int(read_topk),
            has_null=bool(null_logits is not None),
            has_valid_mask=bool(valid_mask is not None),
            read_q_is_raw=bool(read_q_is_raw),
            fuse_scale_add=bool(fuse_scale_add),
        )
    timing_start = time.perf_counter() if timing_enabled else 0.0
    state_indices = state_indices.to(device=read_q.device, dtype=torch.long).contiguous()
    start_indices = start_indices.to(device=read_q.device, dtype=torch.long).contiguous()
    page_table, anchor_lengths, max_len = cache.export_page_table(
        state_indices,
        device=read_q.device,
        state_indices_key=state_indices_key,
    )
    if timing_enabled:
        timing_payload["page_table_us"] = int(
            max(0.0, time.perf_counter() - timing_start) * 1_000_000
        )
    if valid_mask is not None:
        valid_mask = valid_mask.to(device=read_q.device, dtype=torch.bool).contiguous()
        anchor_lengths = torch.where(
            valid_mask,
            anchor_lengths,
            torch.zeros_like(anchor_lengths),
        )
        start_indices = torch.where(
            valid_mask,
            start_indices,
            torch.zeros_like(start_indices),
        )
        max_len = int(anchor_lengths.max().item()) if int(anchor_lengths.numel()) else 0
    if persistent_memory_enabled:
        cache.assert_persistent_memory_ready(
            state_indices_key if state_indices_key is not None else state_indices,
            valid_mask=valid_mask,
        )
    if max_len <= 0:
        if fuse_scale_add:
            assert base_out is not None
            return base_out
        return torch.zeros(batch_size, heads, value_dim, device=read_q.device, dtype=read_q.dtype)
    if max_anchor_len is None:
        env_max_anchor = os.environ.get("HSD_PAGED_HISTORY_MAX_ANCHOR_LEN", "2048").strip()
        if env_max_anchor:
            parsed_max_anchor = int(env_max_anchor)
            max_anchor_len = None if parsed_max_anchor <= 0 else parsed_max_anchor
    if max_anchor_len is not None and max_len > max_anchor_len:
        return _fallback("history_length_limit")
    max_pages = int(page_table.shape[1])
    if max_pages <= 0:
        if fuse_scale_add:
            assert base_out is not None
            return base_out
        return torch.zeros(batch_size, heads, value_dim, device=read_q.device, dtype=read_q.dtype)
    default_max_batch = str(max(16, batch_size)) if valid_mask is not None else "16"
    max_batch = int(os.environ.get("HSD_PAGED_HISTORY_MAX_BATCH", default_max_batch))
    if max_batch <= 0:
        raise ValueError(f"HSD_PAGED_HISTORY_MAX_BATCH must be positive, got {max_batch}.")
    if batch_size > max_batch and not persistent_memory_enabled:
        chunks: list[torch.Tensor] | None = [] if not fuse_scale_add else None
        for batch_start in range(0, batch_size, max_batch):
            batch_end = min(batch_size, batch_start + max_batch)
            chunk = paged_batched_decode_history_read(
                cache=cache,
                read_q=read_q[batch_start:batch_end],
                router_q=router_q[batch_start:batch_end],
                state_indices=state_indices[batch_start:batch_end],
                start_indices=start_indices[batch_start:batch_end],
                route_scale=route_scale,
                read_scale=read_scale,
                null_logits=(
                    None
                    if null_logits is None
                    else null_logits[batch_start:batch_end]
                ),
                valid_mask=(
                    None
                    if valid_mask is None
                    else valid_mask[batch_start:batch_end]
                ),
                max_anchor_len=max_anchor_len,
                state_indices_key=(
                    None
                    if state_indices_key is None
                    else state_indices_key[batch_start:batch_end]
                ),
                read_q_is_raw=read_q_is_raw,
                read_q_num_heads=read_q_num_heads,
                base_out=(
                    None
                    if base_out is None
                    else base_out[batch_start:batch_end]
                ),
                history_scale=history_scale,
                read_topk=read_topk,
                persistent_memory_enabled=False,
                backend=policy.backend,
                validate_fused=policy.validate_fused,
                record_backend=policy.record_backend,
            )
            if chunk is None:
                return _fallback("unsupported_inputs")
            if chunks is not None:
                chunks.append(chunk)
        if timing_enabled:
            _record_paged_history_event(
                "paged_history_read_split_batch",
                batch=int(batch_size),
                max_batch=int(max_batch),
                chunks=int((batch_size + max_batch - 1) // max_batch),
            )
        if fuse_scale_add:
            assert base_out is not None
            return base_out
        assert chunks is not None
        return torch.cat(chunks, dim=0)

    read_q = read_q.contiguous()
    router_q = router_q.contiguous()
    has_null = null_logits is not None
    original_null_logits = null_logits
    if null_logits is not None:
        null_logits = null_logits.contiguous()

    impl = policy.backend
    if persistent_memory_enabled:
        return _run_paged_decode_memory(
            _fallback=_fallback, _selected=_selected,
            _apply_valid_mask=_apply_valid_mask,
            _sync_after_if_requested=_sync_after_if_requested,
            anchor_lengths=anchor_lengths,
            base_out=base_out,
            batch_size=batch_size,
            cache=cache,
            fuse_scale_add=fuse_scale_add,
            has_null=has_null,
            heads=heads,
            history_scale=history_scale,
            impl=impl,
            memory_metrics=memory_metrics,
            null_logits=null_logits,
            page_table=page_table,
            read_q=read_q,
            read_q_head_repeat=read_q_head_repeat,
            read_q_is_raw=read_q_is_raw,
            read_q_num_heads=read_q_num_heads,
            read_scale=read_scale,
            route_dim=route_dim,
            route_scale=route_scale,
            router_q=router_q,
            start_indices=start_indices,
            state_indices=state_indices,
            state_key_dim=state_key_dim,
            valid_mask=valid_mask,
            value_dim=value_dim,
        )
    if int(read_topk) == 0:
        return _run_paged_decode_readall(
            _fallback=_fallback, _selected=_selected,
            _apply_valid_mask=_apply_valid_mask,
            _record_paged_history_event=_record_paged_history_event,
            _sync_after_if_requested=_sync_after_if_requested,
            anchor_lengths=anchor_lengths,
            base_out=base_out,
            batch_size=batch_size,
            cache=cache,
            fuse_scale_add=fuse_scale_add,
            has_null=has_null,
            heads=heads,
            history_scale=history_scale,
            impl=impl,
            max_len=max_len,
            max_pages=max_pages,
            null_logits=null_logits,
            page_table=page_table,
            read_q=read_q,
            read_q_head_repeat=read_q_head_repeat,
            read_q_is_raw=read_q_is_raw,
            read_scale=read_scale,
            route_dim=route_dim,
            route_scale=route_scale,
            router_q=router_q,
            start_indices=start_indices,
            state_indices=state_indices,
            state_key_dim=state_key_dim,
            timing_enabled=timing_enabled,
            valid_mask=valid_mask,
            value_dim=value_dim,
        )
    return _run_paged_decode_top4(
        _fallback=_fallback, _selected=_selected,
        _apply_valid_mask=_apply_valid_mask,
        _record_paged_history_event=_record_paged_history_event,
        _sync_after_if_requested=_sync_after_if_requested,
        anchor_lengths=anchor_lengths,
        base_out=base_out,
        batch_size=batch_size,
        cache=cache,
        fuse_scale_add=fuse_scale_add,
        has_null=has_null,
        heads=heads,
        history_scale=history_scale,
        impl=impl,
        max_anchor_len=max_anchor_len,
        max_len=max_len,
        max_pages=max_pages,
        memory_metrics=memory_metrics,
        null_logits=null_logits,
        original_null_logits=original_null_logits,
        page_table=page_table,
        paged_batched_decode_history_read=paged_batched_decode_history_read,
        persistent_memory_enabled=persistent_memory_enabled,
        policy=policy,
        read_q=read_q,
        read_q_head_repeat=read_q_head_repeat,
        read_q_is_raw=read_q_is_raw,
        read_q_num_heads=read_q_num_heads,
        read_scale=read_scale,
        read_topk=read_topk,
        route_dim=route_dim,
        route_scale=route_scale,
        router_q=router_q,
        start_indices=start_indices,
        state_indices=state_indices,
        state_indices_key=state_indices_key,
        state_key_dim=state_key_dim,
        timing_enabled=timing_enabled,
        timing_payload=timing_payload,
        valid_mask=valid_mask,
        value_dim=value_dim,
    )
