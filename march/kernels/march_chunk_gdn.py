# Copyright (c) ModelScope Contributors. All rights reserved.
"""Training helpers for HSD chunk GDN selective scans.

This shared training implementation is owned by march.kernels; mcore-bridge
imports it through a thin adapter. It uses the FLA forward kernels, exposes
fp32 anchor-boundary states, and routes anchor-state gradients back into the
chunk recurrence through a local ``dh_external`` backward kernel.
"""

from __future__ import annotations

import contextlib
import os
from typing import Optional

import torch

from .chunk_dependencies import (
    RCP_LN2,
    chunk_bwd_dqkwg,
    chunk_bwd_dv_local,
    chunk_fwd_o,
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_fwd_h,
    chunk_gated_delta_rule_fwd_intra,
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    chunk_gated_delta_rule_fwd_kkt_solve_kernel,
    chunk_local_cumsum,
    custom_device_ctx,
    exp2,
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_wy_repr_bwd,
    recompute_w_u_fwd,
    tl,
    triton,
)

from .chunk_dependencies import FLA_CHUNK_SIZE
from .chunk_reference import (
    _is_packed_cu,
    _packed_chunk_index,
    _run_anchor_read_reference,
    _run_reference_backward,
    _run_segmented_packed_reference,
    _run_segmented_reference,
)

_PROFILE_STATS: dict[str, float] = {}
_FLASHQLA_PACKED_METADATA_CACHE: list[tuple] = []
_FLASHQLA_TILE_BACKWARD_ALIASES = {
    "reference": "reference",
    "ref": "reference",
    "optimized": "optimized",
    "fla": "optimized",
}
# Keep every supported Safin-1 training geometry explicit.
_FLASHQLA_TILE_GEOMETRY_BACKENDS = {
    (1, 16, 96, 96): frozenset({"reference", "optimized"}),
    (1, 16, 128, 128): frozenset({"reference", "optimized"}),
}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _normalize_flashqla_tile_bwd_backend(raw: str, *, source: str) -> str:
    backend = _FLASHQLA_TILE_BACKWARD_ALIASES.get(raw.strip().lower())
    if backend is None:
        raise RuntimeError(
            f"{source} must be reference or optimized; received {raw!r}."
        )
    return backend


def _resolve_flashqla_tile_bwd_backend(
    backward_backend: Optional[str] = None,
) -> str:
    """Resolve and freeze the supported FlashQLA backward implementation."""

    if backward_backend is not None:
        return _normalize_flashqla_tile_bwd_backend(
            str(backward_backend),
            source="backward_backend",
        )

    raw = os.environ.get("HSD_FLASHQLA_TILE_BWD_BACKEND", "").strip().lower()
    if raw:
        return _normalize_flashqla_tile_bwd_backend(
            raw,
            source="HSD_FLASHQLA_TILE_BWD_BACKEND",
        )
    if _env_bool("HSD_FLASHQLA_TILE_REFERENCE_BWD", False):
        return "reference"
    return "optimized"


def _validate_flashqla_tile_geometry_backend(
    *,
    batch: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    backward_backend: str,
) -> bool:
    """Return whether a geometry is admitted, and reject unsafe backends.

    Unsupported geometries retain the public producer's opt-in ``None``
    behavior.  A known geometry paired with an unsafe backward is different:
    returning ``None`` could silently route production training to another
    implementation, so that combination fails closed with a diagnostic.
    """

    geometry = (int(batch), int(heads), int(key_dim), int(value_dim))
    allowed_backends = _FLASHQLA_TILE_GEOMETRY_BACKENDS.get(geometry)
    if allowed_backends is None:
        return False
    if backward_backend not in allowed_backends:
        allowed = ", ".join(sorted(allowed_backends))
        raise RuntimeError(
            "FlashQLA TileLang backward backend "
            f"{backward_backend!r} is not supported for geometry "
            f"B={geometry[0]}, H={geometry[1]}, K={geometry[2]}, "
            f"V={geometry[3]}; allowed backends: {allowed}."
        )
    return True


def _resolve_selective_scan_bwd_launch_params(
    *,
    key_dim: int,
    value_dim: int,
) -> tuple[int, int]:
    if (int(key_dim), int(value_dim)) == (96, 96):
        # Closed H16/K96/V96 production LaunchSpec.  Three V32 programs per
        # head cover V=96 exactly and provide ample occupancy without a tail
        # tile.  Ignore legacy tuning environment variables for this admitted
        # geometry so forward/backward validation cannot be invalidated by a
        # process-wide override.
        return 32, 4
    if (int(key_dim), int(value_dim)) == (128, 128):
        # The validated 128x128 producer geometry is a closed production
        # LaunchSpec.  Match the validated FlashQLA path and do not let HSD_*
        # mutate its backward kernel after the model contract is prepared.
        return 16, 4
    block_v = _env_int("HSD_TRAIN_SELECTIVE_SCAN_BWD_BV", 0)
    if block_v <= 0:
        block_v = 4 if key_dim >= 192 and value_dim >= 256 else 32
    if block_v not in (4, 8, 16, 32):
        raise RuntimeError("HSD_TRAIN_SELECTIVE_SCAN_BWD_BV must be one of 4, 8, 16, or 32.")

    default_num_warps = 1 if key_dim >= 192 and value_dim >= 256 else 4
    raw_num_warps = os.environ.get("HSD_TRAIN_SELECTIVE_SCAN_BWD_WARPS", "").strip()
    if raw_num_warps:
        try:
            num_warps = int(raw_num_warps)
        except ValueError as exc:
            raise RuntimeError(
                "HSD_TRAIN_SELECTIVE_SCAN_BWD_WARPS must be one of 1, 2, 4, or 8."
            ) from exc
    else:
        num_warps = default_num_warps
    if num_warps not in (1, 2, 4, 8):
        raise RuntimeError("HSD_TRAIN_SELECTIVE_SCAN_BWD_WARPS must be one of 1, 2, 4, or 8.")
    return block_v, num_warps


def _profile_enabled() -> bool:
    return _env_bool("HSD_PROFILE_TIMING", False)


def _prepare_flashqla_packed_metadata(
    *,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    anchor_cu_seqlens: torch.Tensor,
    text_len: int,
    anchor_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Canonicalize and validate immutable packed sidecars once.

    HSD passes the same packed-layout tensors through every layer.  Copying
    them GPU->CPU for validation in every producer call introduced a stream
    synchronization into the measured forward path.  Keep a small
    identity-and-version cache, matching FlashQLA's own metadata helpers, so
    the synchronization and int32 conversion occur only once per layout.
    """

    def tensor_version(tensor: torch.Tensor) -> int | None:
        # vLLM executes model/profile forwards under ``inference_mode``;
        # tensors created there intentionally have no autograd version
        # counter.  These sidecars are immutable by this function's contract,
        # so identity remains a valid cache key in that mode.
        try:
            return int(tensor._version)
        except RuntimeError as exc:
            if "do not track version counter" not in str(exc):
                raise
            return None

    versions = tuple(
        tensor_version(tensor)
        for tensor in (
            boundary_counts,
            text_cu_seqlens,
            anchor_cu_seqlens,
        )
    )
    for entry in _FLASHQLA_PACKED_METADATA_CACHE:
        (
            cached_boundary,
            cached_text_cu,
            cached_anchor_cu,
            cached_versions,
            cached_text_len,
            cached_anchor_count,
            cached_device,
            result,
        ) = entry
        if (
            boundary_counts is cached_boundary
            and text_cu_seqlens is cached_text_cu
            and anchor_cu_seqlens is cached_anchor_cu
            and versions == cached_versions
            and int(text_len) == cached_text_len
            and int(anchor_count) == cached_anchor_count
            and device == cached_device
        ):
            return result

    text_cu = text_cu_seqlens.to(
        device=device,
        dtype=torch.int32,
    ).contiguous()
    anchor_cu = anchor_cu_seqlens.to(
        device=device,
        dtype=torch.int32,
    ).contiguous()
    boundary = boundary_counts.to(
        device=device,
        dtype=torch.long,
    ).contiguous()

    text_cu_cpu = text_cu.detach().cpu()
    anchor_cu_cpu = anchor_cu.detach().cpu()
    boundary_cpu = boundary.detach().cpu()
    valid = (
        int(text_cu_cpu[0]) == 0
        and int(text_cu_cpu[-1]) == int(text_len)
        and int(anchor_cu_cpu[0]) == 0
        and int(anchor_cu_cpu[-1]) == int(anchor_count)
    )
    if valid:
        for doc_idx in range(int(text_cu_cpu.numel()) - 1):
            text_start = int(text_cu_cpu[doc_idx])
            text_end = int(text_cu_cpu[doc_idx + 1])
            anchor_start = int(anchor_cu_cpu[doc_idx])
            anchor_end = int(anchor_cu_cpu[doc_idx + 1])
            expected_count = (text_end - text_start) // 512
            if anchor_end - anchor_start != expected_count:
                valid = False
                break
            local = boundary_cpu[anchor_start:anchor_end] - text_start
            expected = torch.arange(
                1,
                expected_count + 1,
                dtype=torch.long,
            ) * 512
            if not torch.equal(local, expected):
                valid = False
                break

    result = (text_cu, anchor_cu, boundary, valid)
    _FLASHQLA_PACKED_METADATA_CACHE.insert(
        0,
        (
            boundary_counts,
            text_cu_seqlens,
            anchor_cu_seqlens,
            versions,
            int(text_len),
            int(anchor_count),
            device,
            result,
        ),
    )
    del _FLASHQLA_PACKED_METADATA_CACHE[4:]
    return result


def clear_hsd_chunk_profile_stats() -> None:
    _PROFILE_STATS.clear()


def pop_hsd_chunk_profile_stats() -> dict[str, float]:
    stats = dict(_PROFILE_STATS)
    _PROFILE_STATS.clear()
    return stats


@contextlib.contextmanager
def _profile_cuda_ms(name: str):
    if not _profile_enabled() or not torch.cuda.is_available():
        yield
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        end.synchronize()
        _PROFILE_STATS[name] = _PROFILE_STATS.get(name, 0.0) + float(start.elapsed_time(end))


def _chunk_gated_delta_rule_fwd_h_fp32_states(
    *,
    key: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    if triton is None or chunk_gated_delta_rule_fwd_kernel_h_blockdim64 is None:
        raise RuntimeError("Triton/FLA chunk state kernels are unavailable.")

    batch, text_len, key_heads, key_dim = key.shape
    value_heads = int(u.shape[2])
    value_dim = int(u.shape[-1])
    if cu_seqlens is None:
        sequence_count = batch
        chunk_count = triton.cdiv(text_len, FLA_CHUNK_SIZE)
        chunk_offsets = None
    else:
        if prepare_chunk_indices is None or prepare_chunk_offsets is None:
            raise RuntimeError("FLA chunk index helpers are unavailable.")
        sequence_count = int(cu_seqlens.numel()) - 1
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, FLA_CHUNK_SIZE)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, FLA_CHUNK_SIZE)
        chunk_count = int(chunk_indices.shape[0])
    chunk_states = torch.empty(
        batch,
        chunk_count,
        value_heads,
        key_dim,
        value_dim,
        device=key.device,
        dtype=torch.float32,
    )

    def grid(meta):
        # FLA indexes value tiles with program_id(0), sequence/head with program_id(1).
        return (triton.cdiv(value_dim, meta["BV"]), sequence_count * value_heads)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=key,
        v=u,
        w=w,
        v_new=None,
        g=g,
        gk=None,
        h=chunk_states,
        h0=None,
        ht=None,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=text_len,
        H=key_heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BT=FLA_CHUNK_SIZE,
        USE_EXP2=True,
        TRANSPOSE_STATE=False,
    )
    return chunk_states


if triton is not None:
    from .chunk_anchor_triton import (
        _hsd_anchor_read_fwd_kernel,
        _hsd_anchor_read_bwd_q_kernel,
        _hsd_compact_anchor_read_fwd_kernel,
        _hsd_anchor_state_scatter_bwd_kernel,
    )
    from .chunk_scan_triton import _chunk_gated_delta_rule_bwd_kernel_dhu_external
else:
    _hsd_anchor_read_fwd_kernel = None
    _hsd_anchor_read_bwd_q_kernel = None
    _hsd_compact_anchor_read_fwd_kernel = None
    _hsd_anchor_state_scatter_bwd_kernel = None
    _chunk_gated_delta_rule_bwd_kernel_dhu_external = None




def _chunk_gated_delta_rule_bwd_dhu_external(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    dht: torch.Tensor | None,
    dht_extra: torch.Tensor | None,
    dh_direct: torch.Tensor | None,
    dh_extra: torch.Tensor | None,
    dtext_o: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    compact_dh: torch.Tensor | None = None,
    anchor_cu_seqlens: torch.Tensor | None = None,
    compact_chunk_stride: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None or _chunk_gated_delta_rule_bwd_kernel_dhu_external is None:
        raise RuntimeError("Triton backward kernel is unavailable.")
    batch, text_len, key_heads, key_dim = query.shape
    value_heads = int(dtext_o.shape[2])
    value_dim = int(dtext_o.shape[-1])
    if cu_seqlens is None:
        sequence_count = batch
        chunk_count = triton.cdiv(text_len, FLA_CHUNK_SIZE)
    else:
        if chunk_offsets is None:
            if prepare_chunk_offsets is None:
                raise RuntimeError("FLA chunk offset helper is unavailable.")
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, FLA_CHUNK_SIZE)
        sequence_count = int(cu_seqlens.numel()) - 1
        chunk_count = int(chunk_offsets[-1].item())
    if key_heads != value_heads:
        raise RuntimeError("The HSD training selective scan currently requires H == HV.")
    if (dht is None) != (dht_extra is None):
        raise RuntimeError("dht and dht_extra must either both be provided or both be omitted.")
    if (dh_direct is None) != (dh_extra is None):
        raise RuntimeError("dh_direct and dh_extra must either both be provided or both be omitted.")
    if compact_dh is not None and dh_direct is not None:
        raise RuntimeError("Dense and compact HSD boundary gradients are mutually exclusive.")
    if compact_dh is not None:
        if cu_seqlens is None or anchor_cu_seqlens is None:
            raise RuntimeError("Compact HSD boundary gradients require packed text and anchor cu_seqlens.")
        if batch != 1:
            raise RuntimeError("Compact HSD boundary-gradient injection currently requires batch=1.")
        if compact_chunk_stride <= 0:
            raise RuntimeError("compact_chunk_stride must be positive.")
        expected_compact_shape = (
            batch,
            int(anchor_cu_seqlens[-1].item()),
            value_heads,
            key_dim,
            value_dim,
        )
        if tuple(compact_dh.shape) != expected_compact_shape:
            raise RuntimeError(
                "Compact HSD boundary-gradient shape mismatch: "
                f"expected={expected_compact_shape}, actual={tuple(compact_dh.shape)}."
            )
        if compact_dh.dtype != torch.float32 or not compact_dh.is_contiguous():
            raise RuntimeError("Compact HSD boundary gradients must be contiguous FP32 tensors.")
    dh = query.new_empty(batch, chunk_count, value_heads, key_dim, value_dim)
    dv2 = torch.empty_like(dv)
    order = (
        0
        if (int(key_dim), int(value_dim)) == (128, 128)
        else _env_int("HSD_TRAIN_SELECTIVE_SCAN_BWD_ORDER", 0)
    )
    block_v, num_warps = _resolve_selective_scan_bwd_launch_params(
        key_dim=key_dim,
        value_dim=value_dim,
    )

    def grid(_meta):
        return (triton.cdiv(value_dim, block_v) * sequence_count * value_heads,)

    _chunk_gated_delta_rule_bwd_kernel_dhu_external[grid](
        q=query,
        k=key,
        w=w,
        g=g,
        dht=dht,
        dht_extra=dht_extra,
        dh_direct=dh_direct,
        dh_extra=dh_extra,
        compact_dh=compact_dh,
        do=dtext_o,
        dh=dh,
        dv=dv,
        dv2=dv2,
        cu_seqlens=cu_seqlens,
        anchor_cu_seqlens=anchor_cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=text_len,
        H=key_heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BT=FLA_CHUNK_SIZE,
        BV=block_v,
        COMPACT_CHUNK_STRIDE=int(compact_chunk_stride),
        EXTERNAL_ORDER=order,
        num_warps=num_warps,
        num_stages=1,
    )
    return dh, dv2


def _chunk_local_cumsum_packed_segmented(
    g: torch.Tensor,
    *,
    text_cu_seqlens: torch.Tensor,
    reverse: bool,
) -> torch.Tensor:
    text_cu_cpu = text_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
    pieces = []
    for doc_idx in range(int(text_cu_cpu.numel()) - 1):
        text_start = int(text_cu_cpu[doc_idx].item())
        text_end = int(text_cu_cpu[doc_idx + 1].item())
        pieces.append(
            chunk_local_cumsum(
                g[:, text_start:text_end],
                chunk_size=FLA_CHUNK_SIZE,
                reverse=reverse,
            )
        )
    return torch.cat(pieces, dim=1) if pieces else torch.empty_like(g)


def _run_anchor_read(
    *,
    query: torch.Tensor,
    text_o: torch.Tensor,
    anchor_q: torch.Tensor,
    chunk_states_fp32: torch.Tensor,
    final_state: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor | None,
    anchor_cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, text_len, heads, key_dim = query.shape
    anchor_count = int(anchor_q.shape[1])
    value_dim = text_o.shape[-1]
    if anchor_count == 0:
        return (
            query.new_zeros(batch, 0, heads, value_dim),
            query.new_zeros(batch, 0, heads, key_dim, value_dim),
        )
    packed = _is_packed_cu(text_cu_seqlens, anchor_cu_seqlens)
    use_fused = (
        _env_bool("HSD_FUSED_ANCHOR_READ", True)
        and triton is not None
        and _hsd_anchor_read_fwd_kernel is not None
        and query.is_cuda
    )
    if not use_fused:
        return _run_anchor_read_reference(
            query=query,
            text_o=text_o,
            anchor_q=anchor_q,
            chunk_states_fp32=chunk_states_fp32,
            final_state=final_state,
            boundary_counts=boundary_counts,
            text_cu_seqlens=text_cu_seqlens,
            anchor_cu_seqlens=anchor_cu_seqlens,
            chunk_offsets=chunk_offsets,
            scale=scale,
        )

    if packed:
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(text_cu_seqlens, FLA_CHUNK_SIZE)
        text_cu = text_cu_seqlens
        anchor_cu = anchor_cu_seqlens
        offsets = chunk_offsets
        n_docs = int(text_cu_seqlens.numel()) - 1
    else:
        text_cu = boundary_counts.new_empty((0,), dtype=torch.long)
        anchor_cu = boundary_counts.new_empty((0,), dtype=torch.long)
        offsets = boundary_counts.new_empty((0,), dtype=torch.long)
        n_docs = batch

    chunk_count = int(chunk_states_fp32.shape[1])
    anchor_o = torch.empty(
        batch,
        anchor_count,
        heads,
        value_dim,
        device=query.device,
        dtype=text_o.dtype,
    )
    anchor_states = torch.empty(
        batch,
        anchor_count,
        heads,
        key_dim,
        value_dim,
        device=query.device,
        dtype=torch.float32,
    )

    def grid(_meta):
        return (triton.cdiv(value_dim, 32), anchor_count, batch * heads)

    _hsd_anchor_read_fwd_kernel[grid](
        anchor_q=anchor_q,
        chunk_states=chunk_states_fp32,
        final_state=final_state,
        boundary_counts=boundary_counts,
        text_cu_seqlens=text_cu,
        anchor_cu_seqlens=anchor_cu,
        chunk_offsets=offsets,
        anchor_o=anchor_o,
        anchor_states=anchor_states,
        scale=float(scale),
        TEXT_LEN=text_len,
        ANCHOR_COUNT=anchor_count,
        CHUNK_COUNT=chunk_count,
        H=heads,
        K=key_dim,
        V=value_dim,
        BT=FLA_CHUNK_SIZE,
        BK=64,
        BV=32,
        N_DOCS=n_docs,
        IS_PACKED=packed,
    )
    return anchor_o, anchor_states


def _scatter_anchor_state_grads_fused(
    *,
    anchor_q: torch.Tensor,
    anchor_states: torch.Tensor,
    boundary_counts: torch.Tensor,
    danchor_o: torch.Tensor | None,
    danchor_states: torch.Tensor | None,
    text_len: int,
    chunk_count: int,
    scale: float,
    text_cu_seqlens: torch.Tensor | None = None,
    anchor_cu_seqlens: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not (
        _env_bool("HSD_FUSED_ANCHOR_READ", True)
        and triton is not None
        and _hsd_anchor_read_bwd_q_kernel is not None
        and _hsd_anchor_state_scatter_bwd_kernel is not None
        and anchor_q.is_cuda
    ):
        return None

    batch, anchor_count, heads, key_dim = anchor_q.shape
    if anchor_count == 0:
        return None
    value_dim = anchor_states.shape[-1]
    state_dtype = anchor_states.dtype
    packed = _is_packed_cu(text_cu_seqlens, anchor_cu_seqlens)
    if packed:
        if chunk_offsets is None:
            if prepare_chunk_offsets is None:
                return None
            chunk_offsets = prepare_chunk_offsets(text_cu_seqlens, FLA_CHUNK_SIZE)
        text_cu = text_cu_seqlens
        anchor_cu = anchor_cu_seqlens
        offsets = chunk_offsets
        sequence_count = int(text_cu_seqlens.numel()) - 1
    else:
        text_cu = boundary_counts.new_empty((0,), dtype=torch.long)
        anchor_cu = boundary_counts.new_empty((0,), dtype=torch.long)
        offsets = boundary_counts.new_empty((0,), dtype=torch.long)
        sequence_count = batch

    if danchor_o is None:
        danchor_o_f = anchor_states.new_zeros(batch, anchor_count, heads, value_dim)
    else:
        # Autograd may supply expanded zero-stride gradients (for example
        # through ``output.sum()``).  The fused kernels consume dense
        # row-major tensors, so normalize the layout before passing pointers.
        danchor_o_f = danchor_o.to(state_dtype).contiguous()
    if danchor_states is None:
        danchor_states_direct = torch.zeros_like(anchor_states)
    else:
        danchor_states_direct = danchor_states.to(state_dtype).contiguous()

    danchor_q = torch.empty_like(anchor_q)
    dh_direct = anchor_states.new_zeros(batch, chunk_count, heads, key_dim, value_dim)
    dh_extra = anchor_states.new_zeros(batch, chunk_count, heads, key_dim, value_dim)
    dht_direct = anchor_states.new_zeros(sequence_count, heads, key_dim, value_dim)
    dht_extra = anchor_states.new_zeros(sequence_count, heads, key_dim, value_dim)

    def q_grid(_meta):
        return (triton.cdiv(key_dim, 64), anchor_count, batch * heads)

    _hsd_anchor_read_bwd_q_kernel[q_grid](
        danchor_o=danchor_o_f,
        anchor_states=anchor_states,
        danchor_q=danchor_q,
        scale=float(scale),
        ANCHOR_COUNT=anchor_count,
        H=heads,
        K=key_dim,
        V=value_dim,
        BK=64,
        BV=32,
    )

    def scatter_grid(_meta):
        return (triton.cdiv(value_dim, 32), triton.cdiv(key_dim, 64), anchor_count * batch * heads)

    _hsd_anchor_state_scatter_bwd_kernel[scatter_grid](
        anchor_q=anchor_q,
        danchor_o=danchor_o_f,
        danchor_states=danchor_states_direct,
        boundary_counts=boundary_counts,
        text_cu_seqlens=text_cu,
        anchor_cu_seqlens=anchor_cu,
        chunk_offsets=offsets,
        dh_direct=dh_direct,
        dh_extra=dh_extra,
        dht_direct=dht_direct,
        dht_extra=dht_extra,
        scale=float(scale),
        TEXT_LEN=text_len,
        ANCHOR_COUNT=anchor_count,
        CHUNK_COUNT=chunk_count,
        H=heads,
        K=key_dim,
        V=value_dim,
        BT=FLA_CHUNK_SIZE,
        BK=64,
        BV=32,
        N_DOCS=sequence_count,
        IS_PACKED=packed,
    )
    return danchor_q, dh_direct, dht_direct, dh_extra, dht_extra


def _run_full_scan_forward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    anchor_q: torch.Tensor,
    boundary_counts: torch.Tensor,
    scale: float,
    text_cu_seqlens: torch.Tensor | None = None,
    anchor_cu_seqlens: torch.Tensor | None = None,
    return_intermediates: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch, text_len, heads, key_dim = query.shape
    packed = _is_packed_cu(text_cu_seqlens, anchor_cu_seqlens)
    chunk_indices = None
    chunk_offsets = None
    if packed:
        if prepare_chunk_indices is None or prepare_chunk_offsets is None:
            raise RuntimeError("FLA chunk index helpers are unavailable.")
        chunk_indices = prepare_chunk_indices(text_cu_seqlens, FLA_CHUNK_SIZE)
        chunk_offsets = prepare_chunk_offsets(text_cu_seqlens, FLA_CHUNK_SIZE)

    with _profile_cuda_ms("chunk/fwd_cumsum_ms"):
        g_cumsum = chunk_local_cumsum(
            g,
            chunk_size=FLA_CHUNK_SIZE,
            scale=RCP_LN2,
            cu_seqlens=text_cu_seqlens if packed else None,
            chunk_indices=chunk_indices,
        )
    with _profile_cuda_ms("chunk/fwd_intra_ms"):
        w, u, A = chunk_gated_delta_rule_fwd_intra(
            k=key,
            v=value,
            g=g_cumsum,
            beta=beta,
            cu_seqlens=text_cu_seqlens if packed else None,
            chunk_indices=chunk_indices,
            use_exp2=True,
        )
    with _profile_cuda_ms("chunk/fwd_h_for_o_ms"):
        chunk_states_for_o, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            k=key,
            w=w,
            u=u,
            g=g_cumsum,
            initial_state=None,
            output_final_state=True,
            cu_seqlens=text_cu_seqlens if packed else None,
            chunk_indices=chunk_indices,
            use_exp2=True,
            transpose_state_layout=False,
        )
    with _profile_cuda_ms("chunk/fwd_o_ms"):
        text_o = chunk_fwd_o(
            q=query,
            k=key,
            v=v_new,
            h=chunk_states_for_o,
            g=g_cumsum,
            scale=scale,
            cu_seqlens=text_cu_seqlens if packed else None,
            chunk_indices=chunk_indices,
            use_exp2=True,
            transpose_state_layout=False,
        )
    with _profile_cuda_ms("chunk/fwd_h_fp32_states_ms"):
        chunk_states_fp32 = _chunk_gated_delta_rule_fwd_h_fp32_states(
            key=key,
            w=w,
            u=u,
            g=g_cumsum,
            cu_seqlens=text_cu_seqlens if packed else None,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
        )

    with _profile_cuda_ms("chunk/fwd_anchor_ms"):
        anchor_o, anchor_states = _run_anchor_read(
            query=query,
            text_o=text_o,
            anchor_q=anchor_q,
            chunk_states_fp32=chunk_states_fp32,
            final_state=final_state,
            boundary_counts=boundary_counts,
            text_cu_seqlens=text_cu_seqlens if packed else None,
            anchor_cu_seqlens=anchor_cu_seqlens if packed else None,
            chunk_offsets=chunk_offsets if packed else None,
            scale=scale,
        )
    if return_intermediates:
        return text_o, anchor_o, anchor_states, g_cumsum, A
    return text_o, anchor_o, anchor_states


def _prepare_fla_backward_intermediates(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    text_cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build only the FLA intermediates required by the existing HSD reverse scan.

    FlashQLA and FLA use different intra-chunk KKT/WY layouts.  The HSD
    backward consumes FLA's cumulative gate and solved lower-triangular
    matrix, but can recompute ``w`` and ``u`` itself.  Calling the private KKT
    solve kernel directly avoids materializing a redundant full-sequence
    ``w``/``u`` pair in the forward.
    """

    del value  # Kept in the signature to make the producer call site explicit.
    if (
        triton is None
        or chunk_local_cumsum is None
        or chunk_gated_delta_rule_fwd_kkt_solve_kernel is None
    ):
        raise RuntimeError("FLA KKT solve kernels are unavailable for the FlashQLA backward.")

    batch, text_len, key_heads, key_dim = key.shape
    value_heads = int(beta.shape[2])
    packed = text_cu_seqlens is not None
    chunk_indices = None
    if packed:
        if prepare_chunk_indices is None:
            raise RuntimeError("FLA chunk index helpers are unavailable.")
        chunk_indices = prepare_chunk_indices(text_cu_seqlens, FLA_CHUNK_SIZE)
        chunk_count = int(chunk_indices.shape[0])
    else:
        chunk_count = triton.cdiv(text_len, FLA_CHUNK_SIZE)

    with _profile_cuda_ms("chunk/flashqla_tile_bwd_prep_cumsum_ms"):
        g_cumsum = chunk_local_cumsum(
            g,
            chunk_size=FLA_CHUNK_SIZE,
            scale=RCP_LN2,
            cu_seqlens=text_cu_seqlens if packed else None,
            chunk_indices=chunk_indices,
        )
    with _profile_cuda_ms("chunk/flashqla_tile_bwd_prep_kkt_ms"):
        A = torch.zeros(
            batch,
            text_len,
            value_heads,
            FLA_CHUNK_SIZE,
            device=key.device,
            dtype=key.dtype,
        )
        chunk_gated_delta_rule_fwd_kkt_solve_kernel[(chunk_count, batch * value_heads)](
            k=key,
            g=g_cumsum,
            beta=beta,
            A=A,
            cu_seqlens=text_cu_seqlens if packed else None,
            chunk_indices=chunk_indices,
            T=text_len,
            H=key_heads,
            HV=value_heads,
            K=key_dim,
            BT=FLA_CHUNK_SIZE,
            BC=16,
            USE_EXP2=True,
        )
    return g_cumsum, A


def _scatter_anchor_state_grads(
    *,
    anchor_q: torch.Tensor,
    anchor_states: torch.Tensor,
    boundary_counts: torch.Tensor,
    danchor_o: torch.Tensor | None,
    danchor_states: torch.Tensor | None,
    text_len: int,
    chunk_count: int,
    scale: float,
    text_cu_seqlens: torch.Tensor | None = None,
    anchor_cu_seqlens: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    batch, anchor_count, heads, key_dim = anchor_q.shape
    value_dim = anchor_states.shape[-1]
    fused = _scatter_anchor_state_grads_fused(
        anchor_q=anchor_q,
        anchor_states=anchor_states,
        boundary_counts=boundary_counts,
        danchor_o=danchor_o,
        danchor_states=danchor_states,
        text_len=text_len,
        chunk_count=chunk_count,
        scale=scale,
        text_cu_seqlens=text_cu_seqlens,
        anchor_cu_seqlens=anchor_cu_seqlens,
        chunk_offsets=chunk_offsets,
    )
    if fused is not None:
        return fused

    state_dtype = anchor_states.dtype
    if danchor_o is None:
        danchor_o_f = anchor_states.new_zeros(batch, anchor_count, heads, value_dim)
    else:
        danchor_o_f = danchor_o.to(state_dtype)
    danchor_o_scaled = danchor_o_f * scale
    danchor_q = torch.einsum("bmhv,bmhkv->bmhk", danchor_o_scaled, anchor_states)
    danchor_states_extra = torch.einsum("bmhk,bmhv->bmhkv", anchor_q.to(state_dtype), danchor_o_scaled)
    danchor_states_direct = torch.zeros_like(anchor_states) if danchor_states is None else danchor_states.to(
        state_dtype
    )
    dh_direct = anchor_states.new_zeros(batch, chunk_count, heads, key_dim, value_dim)
    dh_extra = anchor_states.new_zeros(batch, chunk_count, heads, key_dim, value_dim)
    dht_direct = None
    dht_extra = None
    if _is_packed_cu(text_cu_seqlens, anchor_cu_seqlens):
        text_cu_cpu = text_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        anchor_cu_cpu = anchor_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        if chunk_offsets is None:
            if prepare_chunk_offsets is None:
                raise RuntimeError("FLA chunk offset helper is unavailable.")
            chunk_offsets = prepare_chunk_offsets(text_cu_seqlens, FLA_CHUNK_SIZE)
        chunk_offsets_cpu = chunk_offsets.detach().to(device="cpu", dtype=torch.long)
        boundary_cpu = boundary_counts.detach().to(device="cpu", dtype=torch.long)
        sequence_count = int(text_cu_cpu.numel()) - 1
        dht_direct = anchor_states.new_zeros(sequence_count, heads, key_dim, value_dim)
        dht_extra = anchor_states.new_zeros(sequence_count, heads, key_dim, value_dim)
        for doc_idx in range(sequence_count):
            text_start = int(text_cu_cpu[doc_idx].item())
            text_end = int(text_cu_cpu[doc_idx + 1].item())
            doc_len = text_end - text_start
            anchor_start = int(anchor_cu_cpu[doc_idx].item())
            anchor_end = int(anchor_cu_cpu[doc_idx + 1].item())
            for anchor_idx in range(anchor_start, anchor_end):
                local_boundary = int(boundary_cpu[anchor_idx].item()) - text_start
                if local_boundary <= 0:
                    continue
                if local_boundary == doc_len:
                    dht_direct[doc_idx] = dht_direct[doc_idx] + danchor_states_direct[0, anchor_idx]
                    dht_extra[doc_idx] = dht_extra[doc_idx] + danchor_states_extra[0, anchor_idx]
                    continue
                chunk_idx = int(chunk_offsets_cpu[doc_idx].item()) + local_boundary // FLA_CHUNK_SIZE
                if chunk_idx <= int(chunk_offsets_cpu[doc_idx].item()):
                    continue
                dh_direct[:, chunk_idx - 1] = dh_direct[:, chunk_idx - 1] + danchor_states_direct[:, anchor_idx]
                dh_extra[:, chunk_idx - 1] = dh_extra[:, chunk_idx - 1] + danchor_states_extra[:, anchor_idx]
        if not bool(dht_direct.abs().any().item()) and not bool(dht_extra.abs().any().item()):
            dht_direct = None
            dht_extra = None
    else:
        for anchor_idx, boundary in enumerate(boundary_counts.tolist()):
            boundary = int(boundary)
            if boundary <= 0:
                continue
            if boundary == text_len:
                if dht_direct is None:
                    dht_direct = anchor_states.new_zeros(batch, heads, key_dim, value_dim)
                    dht_extra = anchor_states.new_zeros(batch, heads, key_dim, value_dim)
                dht_direct = dht_direct + danchor_states_direct[:, anchor_idx]
                dht_extra = dht_extra + danchor_states_extra[:, anchor_idx]
                continue
            chunk_idx = boundary // FLA_CHUNK_SIZE
            if chunk_idx <= 0:
                continue
            dh_direct[:, chunk_idx - 1] = dh_direct[:, chunk_idx - 1] + danchor_states_direct[:, anchor_idx]
            dh_extra[:, chunk_idx - 1] = dh_extra[:, chunk_idx - 1] + danchor_states_extra[:, anchor_idx]

    return danchor_q.to(anchor_q.dtype), dh_direct, dht_direct, dh_extra, dht_extra


def _run_optimized_backward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    anchor_q: torch.Tensor,
    anchor_states: torch.Tensor,
    boundary_counts: torch.Tensor,
    dtext_o: torch.Tensor | None,
    danchor_o: torch.Tensor | None,
    danchor_states: torch.Tensor | None,
    scale: float,
    needs_grad,
    text_cu_seqlens: torch.Tensor | None = None,
    anchor_cu_seqlens: torch.Tensor | None = None,
) -> list[torch.Tensor | None]:
    batch, text_len, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    packed = _is_packed_cu(text_cu_seqlens, anchor_cu_seqlens)
    chunk_indices = None
    chunk_offsets = None
    if packed:
        if prepare_chunk_indices is None or prepare_chunk_offsets is None:
            raise RuntimeError("FLA chunk index helpers are unavailable.")
        chunk_indices = prepare_chunk_indices(text_cu_seqlens, FLA_CHUNK_SIZE)
        chunk_offsets = prepare_chunk_offsets(text_cu_seqlens, FLA_CHUNK_SIZE)
        chunk_count = int(chunk_indices.shape[0])
    else:
        chunk_count = triton.cdiv(text_len, FLA_CHUNK_SIZE)
    if dtext_o is None:
        dtext_o = value.new_zeros(batch, text_len, heads, value_dim)
    else:
        dtext_o = dtext_o.contiguous()

    with torch.no_grad():
        with _profile_cuda_ms("chunk/bwd_scatter_anchor_ms"):
            danchor_q, dh_direct, dht_direct, dh_extra, dht_extra = _scatter_anchor_state_grads(
                anchor_q=anchor_q,
                anchor_states=anchor_states,
                boundary_counts=boundary_counts,
                danchor_o=danchor_o,
                danchor_states=danchor_states,
                text_len=text_len,
                chunk_count=chunk_count,
                scale=scale,
                text_cu_seqlens=text_cu_seqlens if packed else None,
                anchor_cu_seqlens=anchor_cu_seqlens if packed else None,
                chunk_offsets=chunk_offsets if packed else None,
            )
        with _profile_cuda_ms("chunk/bwd_recompute_w_u_ms"):
            w, u = recompute_w_u_fwd(
                k=key,
                v=value,
                beta=beta,
                A=A,
                g=g_cumsum,
                cu_seqlens=text_cu_seqlens if packed else None,
                chunk_indices=chunk_indices,
                use_exp2=True,
            )
        with _profile_cuda_ms("chunk/bwd_fwd_h_ms"):
            h, v_new, _ = chunk_gated_delta_rule_fwd_h(
                k=key,
                w=w,
                u=u,
                g=g_cumsum,
                initial_state=None,
                output_final_state=False,
                cu_seqlens=text_cu_seqlens if packed else None,
                chunk_indices=chunk_indices,
                use_exp2=True,
                transpose_state_layout=False,
            )
        with _profile_cuda_ms("chunk/bwd_dv_local_ms"):
            dv_local = chunk_bwd_dv_local(
                q=query,
                k=key,
                g=g_cumsum,
                do=dtext_o,
                scale=scale,
                cu_seqlens=text_cu_seqlens if packed else None,
                chunk_indices=chunk_indices,
                use_exp2=True,
            )
        with _profile_cuda_ms("chunk/bwd_dhu_external_ms"):
            dh, dv = _chunk_gated_delta_rule_bwd_dhu_external(
                query=query,
                key=key,
                w=w,
                g=g_cumsum,
                dht=dht_direct,
                dht_extra=dht_extra,
                dh_direct=dh_direct,
                dh_extra=dh_extra,
                dtext_o=dtext_o,
                dv=dv_local,
                scale=scale,
                cu_seqlens=text_cu_seqlens if packed else None,
                chunk_offsets=chunk_offsets if packed else None,
            )
        with _profile_cuda_ms("chunk/bwd_dqkwg_ms"):
            dq, dk, dw, dg = chunk_bwd_dqkwg(
                q=query,
                k=key,
                v=v_new,
                w=w,
                g=g_cumsum,
                h=h,
                dv=dv,
                do=dtext_o,
                dh=dh,
                scale=scale,
                cu_seqlens=text_cu_seqlens if packed else None,
                chunk_indices=chunk_indices,
                use_exp2=True,
                transpose_state_layout=False,
            )
        with _profile_cuda_ms("chunk/bwd_prepare_wy_ms"):
            dk2, dv, db, dg2 = prepare_wy_repr_bwd(
                k=key,
                v=value,
                beta=beta,
                g=g_cumsum,
                A=A,
                dw=dw,
                du=dv,
                cu_seqlens=text_cu_seqlens if packed else None,
                chunk_indices=chunk_indices,
                use_exp2=True,
            )
        dk.add_(dk2)
        dg.add_(dg2)
        with _profile_cuda_ms("chunk/bwd_cumsum_reverse_ms"):
            if packed:
                dg = _chunk_local_cumsum_packed_segmented(
                    dg,
                    text_cu_seqlens=text_cu_seqlens,
                    reverse=True,
                )
            else:
                dg = chunk_local_cumsum(
                    dg,
                    chunk_size=FLA_CHUNK_SIZE,
                    reverse=True,
                )

    grads = [
        dq.to(query.dtype) if needs_grad[0] else None,
        dk.to(key.dtype) if needs_grad[1] else None,
        dv.to(value.dtype) if needs_grad[2] else None,
        db.to(beta.dtype) if needs_grad[3] else None,
        dg.to(g.dtype) if needs_grad[4] else None,
        danchor_q.to(anchor_q.dtype) if needs_grad[5] else None,
    ]
    return grads


def _prepare_flashqla_anchor_q_grad(
    *,
    anchor_q: torch.Tensor,
    anchor_states: torch.Tensor,
    danchor_o: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """Compute only d(anchor_q), leaving boundary dH unmaterialized."""

    batch, anchor_count, heads, key_dim = anchor_q.shape
    value_dim = int(anchor_states.shape[-1])
    expected_state_shape = (batch, anchor_count, heads, key_dim, value_dim)
    if tuple(anchor_states.shape) != expected_state_shape:
        raise RuntimeError(
            "FlashQLA compact anchor-state shape mismatch in backward: "
            f"expected={expected_state_shape}, actual={tuple(anchor_states.shape)}."
        )
    if anchor_states.dtype != torch.float32 or not anchor_states.is_contiguous():
        raise RuntimeError("FlashQLA compact anchor states must be contiguous FP32 tensors.")
    if anchor_count == 0 or danchor_o is None:
        return torch.zeros_like(anchor_q)
    expected_output_shape = (batch, anchor_count, heads, value_dim)
    if tuple(danchor_o.shape) != expected_output_shape:
        raise RuntimeError(
            "FlashQLA compact anchor-output gradient shape mismatch: "
            f"expected={expected_output_shape}, actual={tuple(danchor_o.shape)}."
        )

    flattened_count = batch * anchor_count * heads
    if (
        triton is not None
        and _hsd_anchor_read_bwd_q_kernel is not None
        and anchor_q.is_cuda
    ):
        danchor_q = torch.empty_like(anchor_q)

        def q_grid(_meta):
            return (
                triton.cdiv(key_dim, 64),
                anchor_count,
                batch * heads,
            )

        _hsd_anchor_read_bwd_q_kernel[q_grid](
            danchor_o=danchor_o.contiguous(),
            anchor_states=anchor_states,
            danchor_q=danchor_q,
            scale=float(scale),
            ANCHOR_COUNT=anchor_count,
            H=heads,
            K=key_dim,
            V=value_dim,
            BK=64,
            BV=32,
        )
        return danchor_q

    danchor_q_fp32 = torch.empty(
        (flattened_count, key_dim, 1),
        device=anchor_states.device,
        dtype=torch.float32,
    )
    torch.bmm(
        anchor_states.view(flattened_count, key_dim, value_dim),
        danchor_o.to(torch.float32).contiguous().view(
            flattened_count,
            value_dim,
            1,
        ),
        out=danchor_q_fp32,
    )
    danchor_q_fp32.mul_(float(scale))
    return danchor_q_fp32.view(
        batch,
        anchor_count,
        heads,
        key_dim,
    ).to(anchor_q.dtype)


def _prepare_flashqla_compact_anchor_grads(
    *,
    anchor_q: torch.Tensor,
    anchor_states: torch.Tensor,
    danchor_o: torch.Tensor | None,
    danchor_states: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build one total FP32 dH per compact FlashQLA boundary.

    The rank-one contribution from the anchor read is accumulated directly
    into the compact output.  In particular, this never materializes the two
    dense ``[num_chunk64, H, K, V]`` scatter buffers used by the general HSD
    backward.
    """

    batch, anchor_count, heads, key_dim = anchor_q.shape
    value_dim = int(anchor_states.shape[-1])
    expected_state_shape = (batch, anchor_count, heads, key_dim, value_dim)
    if tuple(anchor_states.shape) != expected_state_shape:
        raise RuntimeError(
            "FlashQLA compact anchor-state shape mismatch in backward: "
            f"expected={expected_state_shape}, actual={tuple(anchor_states.shape)}."
        )
    if anchor_states.dtype != torch.float32 or not anchor_states.is_contiguous():
        raise RuntimeError("FlashQLA compact anchor states must be contiguous FP32 tensors.")
    if anchor_count == 0:
        return torch.zeros_like(anchor_q), None

    flattened_count = batch * anchor_count * heads
    states_flat = anchor_states.view(flattened_count, key_dim, value_dim)
    danchor_o_fp32 = None
    if danchor_o is None:
        danchor_q = torch.zeros_like(anchor_q)
    else:
        expected_output_shape = (batch, anchor_count, heads, value_dim)
        if tuple(danchor_o.shape) != expected_output_shape:
            raise RuntimeError(
                "FlashQLA compact anchor-output gradient shape mismatch: "
                f"expected={expected_output_shape}, actual={tuple(danchor_o.shape)}."
            )
        danchor_o_fp32 = danchor_o.to(torch.float32).contiguous().view(
            flattened_count,
            value_dim,
        )
        danchor_q_fp32 = torch.empty(
            (flattened_count, key_dim, 1),
            device=anchor_states.device,
            dtype=torch.float32,
        )
        torch.bmm(
            states_flat,
            danchor_o_fp32.unsqueeze(-1),
            out=danchor_q_fp32,
        )
        danchor_q_fp32.mul_(float(scale))
        danchor_q = danchor_q_fp32.view(
            batch,
            anchor_count,
            heads,
            key_dim,
        ).to(anchor_q.dtype)

    compact_dh = None
    if danchor_states is not None:
        if tuple(danchor_states.shape) != expected_state_shape:
            raise RuntimeError(
                "FlashQLA compact anchor-state gradient shape mismatch: "
                f"expected={expected_state_shape}, actual={tuple(danchor_states.shape)}."
            )
        compact_dh = torch.empty_like(anchor_states)
        compact_dh.copy_(danchor_states)

    if danchor_o_fp32 is not None:
        anchor_q_fp32 = anchor_q.to(torch.float32).contiguous().view(
            flattened_count,
            key_dim,
        )
        q_column = anchor_q_fp32.unsqueeze(-1)
        do_row = danchor_o_fp32.unsqueeze(1)
        if compact_dh is None:
            compact_dh = torch.empty_like(anchor_states)
            torch.bmm(
                q_column,
                do_row,
                out=compact_dh.view(flattened_count, key_dim, value_dim),
            )
            compact_dh.mul_(float(scale))
        else:
            compact_dh.view(flattened_count, key_dim, value_dim).baddbmm_(
                q_column,
                do_row,
                beta=1.0,
                alpha=float(scale),
            )
    return danchor_q, compact_dh


def _run_flashqla_compact_optimized_backward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    anchor_q: torch.Tensor,
    anchor_states: torch.Tensor,
    dtext_o: torch.Tensor | None,
    danchor_o: torch.Tensor | None,
    danchor_states: torch.Tensor | None,
    scale: float,
    needs_grad,
    text_cu_seqlens: torch.Tensor,
    anchor_cu_seqlens: torch.Tensor,
) -> list[torch.Tensor | None]:
    """Optimized FlashQLA backward with compact chunk-512 dH injection."""

    if not _is_packed_cu(text_cu_seqlens, anchor_cu_seqlens):
        raise RuntimeError("FlashQLA compact backward requires packed cu_seqlens.")
    if prepare_chunk_indices is None or prepare_chunk_offsets is None:
        raise RuntimeError("FLA chunk index helpers are unavailable.")

    batch, text_len, heads, _ = query.shape
    value_dim = int(value.shape[-1])
    chunk_indices = prepare_chunk_indices(text_cu_seqlens, FLA_CHUNK_SIZE)
    chunk_offsets = prepare_chunk_offsets(text_cu_seqlens, FLA_CHUNK_SIZE)
    if dtext_o is None:
        dtext_o = value.new_zeros(batch, text_len, heads, value_dim)
    else:
        dtext_o = dtext_o.contiguous()

    with torch.no_grad():
        with _profile_cuda_ms("chunk/flashqla_tile_bwd_compact_anchor_ms"):
            danchor_q, compact_dh = _prepare_flashqla_compact_anchor_grads(
                anchor_q=anchor_q,
                anchor_states=anchor_states,
                danchor_o=danchor_o,
                danchor_states=danchor_states,
                scale=scale,
            )
        with _profile_cuda_ms("chunk/flashqla_tile_bwd_recompute_w_u_ms"):
            w, u = recompute_w_u_fwd(
                k=key,
                v=value,
                beta=beta,
                A=A,
                g=g_cumsum,
                cu_seqlens=text_cu_seqlens,
                chunk_indices=chunk_indices,
                use_exp2=True,
            )
        with _profile_cuda_ms("chunk/flashqla_tile_bwd_fwd_h_ms"):
            h, v_new, _ = chunk_gated_delta_rule_fwd_h(
                k=key,
                w=w,
                u=u,
                g=g_cumsum,
                initial_state=None,
                output_final_state=False,
                cu_seqlens=text_cu_seqlens,
                chunk_indices=chunk_indices,
                use_exp2=True,
                transpose_state_layout=False,
            )
        with _profile_cuda_ms("chunk/flashqla_tile_bwd_dv_local_ms"):
            dv_local = chunk_bwd_dv_local(
                q=query,
                k=key,
                g=g_cumsum,
                do=dtext_o,
                scale=scale,
                cu_seqlens=text_cu_seqlens,
                chunk_indices=chunk_indices,
                use_exp2=True,
            )
        with _profile_cuda_ms("chunk/flashqla_tile_bwd_dhu_compact_ms"):
            dh, dv = _chunk_gated_delta_rule_bwd_dhu_external(
                query=query,
                key=key,
                w=w,
                g=g_cumsum,
                dht=None,
                dht_extra=None,
                dh_direct=None,
                dh_extra=None,
                compact_dh=compact_dh,
                dtext_o=dtext_o,
                dv=dv_local,
                scale=scale,
                cu_seqlens=text_cu_seqlens,
                anchor_cu_seqlens=anchor_cu_seqlens,
                chunk_offsets=chunk_offsets,
                compact_chunk_stride=8,
            )
        with _profile_cuda_ms("chunk/flashqla_tile_bwd_dqkwg_ms"):
            dq, dk, dw, dg = chunk_bwd_dqkwg(
                q=query,
                k=key,
                v=v_new,
                w=w,
                g=g_cumsum,
                h=h,
                dv=dv,
                do=dtext_o,
                dh=dh,
                scale=scale,
                cu_seqlens=text_cu_seqlens,
                chunk_indices=chunk_indices,
                use_exp2=True,
                transpose_state_layout=False,
            )
        with _profile_cuda_ms("chunk/flashqla_tile_bwd_prepare_wy_ms"):
            dk2, dv, db, dg2 = prepare_wy_repr_bwd(
                k=key,
                v=value,
                beta=beta,
                g=g_cumsum,
                A=A,
                dw=dw,
                du=dv,
                cu_seqlens=text_cu_seqlens,
                chunk_indices=chunk_indices,
                use_exp2=True,
            )
        dk.add_(dk2)
        dg.add_(dg2)
        with _profile_cuda_ms("chunk/flashqla_tile_bwd_cumsum_reverse_ms"):
            dg = _chunk_local_cumsum_packed_segmented(
                dg,
                text_cu_seqlens=text_cu_seqlens,
                reverse=True,
            )

    return [
        dq.to(query.dtype) if needs_grad[0] else None,
        dk.to(key.dtype) if needs_grad[1] else None,
        dv.to(value.dtype) if needs_grad[2] else None,
        db.to(beta.dtype) if needs_grad[3] else None,
        dg.to(g.dtype) if needs_grad[4] else None,
        danchor_q.to(anchor_q.dtype) if needs_grad[5] else None,
    ]


class HSDChunkGatedDeltaWithAnchorStatesFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        anchor_q: torch.Tensor,
        boundary_counts: torch.Tensor,
        text_cu_seqlens: torch.Tensor,
        anchor_cu_seqlens: torch.Tensor,
        scale: float,
    ):
        ctx.scale = float(scale)
        ctx.is_packed = _is_packed_cu(text_cu_seqlens, anchor_cu_seqlens)
        with torch.no_grad():
            text_o, anchor_o, anchor_states, g_cumsum, A = _run_full_scan_forward(
                query=query,
                key=key,
                value=value,
                beta=beta,
                g=g,
                anchor_q=anchor_q,
                boundary_counts=boundary_counts,
                scale=float(scale),
                text_cu_seqlens=text_cu_seqlens if ctx.is_packed else None,
                anchor_cu_seqlens=anchor_cu_seqlens if ctx.is_packed else None,
                return_intermediates=True,
            )
        ctx.save_for_backward(
            query,
            key,
            value,
            beta,
            g,
            anchor_q,
            boundary_counts,
            text_cu_seqlens,
            anchor_cu_seqlens,
            g_cumsum,
            A,
            anchor_states,
        )
        return text_o, anchor_o, anchor_states

    @staticmethod
    def backward(ctx, dtext_o: torch.Tensor, danchor_o: torch.Tensor, danchor_states: torch.Tensor):
        (
            query,
            key,
            value,
            beta,
            g,
            anchor_q,
            boundary_counts,
            text_cu_seqlens,
            anchor_cu_seqlens,
            g_cumsum,
            A,
            anchor_states,
        ) = ctx.saved_tensors
        needs_grad = ctx.needs_input_grad

        if (
            _env_bool("HSD_TRAIN_SELECTIVE_SCAN_REF_BWD", False)
            or _chunk_gated_delta_rule_bwd_kernel_dhu_external is None
        ):
            with _profile_cuda_ms("chunk/bwd_reference_recompute_ms"):
                grad_inputs = _run_reference_backward(
                    query=query,
                    key=key,
                    value=value,
                    beta=beta,
                    g=g,
                    anchor_q=anchor_q,
                    boundary_counts=boundary_counts,
                    text_cu_seqlens=text_cu_seqlens if ctx.is_packed else None,
                    anchor_cu_seqlens=anchor_cu_seqlens if ctx.is_packed else None,
                    dtext_o=dtext_o,
                    danchor_o=danchor_o,
                    danchor_states=danchor_states,
                    scale=ctx.scale,
                    needs_grad=needs_grad,
                )
        else:
            with _profile_cuda_ms("chunk/bwd_optimized_total_ms"):
                grad_inputs = _run_optimized_backward(
                    query=query,
                    key=key,
                    value=value,
                    beta=beta,
                    g=g,
                    g_cumsum=g_cumsum,
                    A=A,
                    anchor_q=anchor_q,
                    anchor_states=anchor_states,
                    boundary_counts=boundary_counts,
                    dtext_o=dtext_o,
                    danchor_o=danchor_o,
                    danchor_states=danchor_states,
                    scale=ctx.scale,
                    needs_grad=needs_grad,
                    text_cu_seqlens=text_cu_seqlens if ctx.is_packed else None,
                    anchor_cu_seqlens=anchor_cu_seqlens if ctx.is_packed else None,
                )

        return (*grad_inputs, None, None, None, None)


def _run_flashqla_compact_anchor_read(
    *,
    text_o: torch.Tensor,
    anchor_q: torch.Tensor,
    compact_states: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the already ordered FP32 chunk-512 boundary states."""

    batch, anchor_count, heads, key_dim = anchor_q.shape
    value_dim = int(text_o.shape[-1])
    if anchor_count == 0:
        return (
            text_o.new_zeros((batch, 0, heads, value_dim)),
            torch.zeros(
                (batch, 0, heads, key_dim, value_dim),
                device=text_o.device,
                dtype=torch.float32,
            ),
        )
    expected_shape = (batch, anchor_count, heads, key_dim, value_dim)
    if tuple(compact_states.shape) != expected_shape:
        raise RuntimeError(
            "FlashQLA compact anchor-state shape mismatch: "
            f"expected={expected_shape}, actual={tuple(compact_states.shape)}."
        )
    if compact_states.dtype != torch.float32:
        raise RuntimeError(
            "FlashQLA compact anchor states must preserve the FP32 recurrent state."
        )
    anchor_states = compact_states
    if (
        triton is not None
        and _hsd_compact_anchor_read_fwd_kernel is not None
        and anchor_q.is_cuda
    ):
        anchor_o = torch.empty(
            (batch, anchor_count, heads, value_dim),
            device=text_o.device,
            dtype=text_o.dtype,
        )

        def grid(_meta):
            return (
                triton.cdiv(value_dim, 32),
                anchor_count,
                batch * heads,
            )

        _hsd_compact_anchor_read_fwd_kernel[grid](
            anchor_q=anchor_q,
            anchor_states=anchor_states,
            anchor_o=anchor_o,
            scale=float(scale),
            ANCHOR_COUNT=anchor_count,
            H=heads,
            K=key_dim,
            V=value_dim,
            BK=64,
            BV=32,
        )
        return anchor_o, anchor_states

    anchor_o = (
        torch.einsum(
            "bmhk,bmhkv->bmhv",
            anchor_q.to(anchor_states.dtype),
            anchor_states,
        )
        * float(scale)
    ).to(text_o.dtype)
    return anchor_o, anchor_states


class HSDFlashQLATileWithAnchorStatesFunction(torch.autograd.Function):
    """FlashQLA producer with compact boundary states for Safin-1 training."""

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        anchor_q: torch.Tensor,
        boundary_counts: torch.Tensor,
        text_cu_seqlens: torch.Tensor,
        anchor_cu_seqlens: torch.Tensor,
        scale: float,
        backward_backend: str,
    ):
        from ._flashqla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd

        # Keep unused producer outputs as ``None`` in backward.
        ctx.set_materialize_grads(False)
        ctx.scale = float(scale)
        ctx.backward_backend = backward_backend
        anchor_count = int(anchor_q.shape[1])

        # Hopper WGMMA cannot lower the recurrent ``K^T @ V'`` state update
        # with an M=96 output: its four-warp group requires a 16-row warp
        # tile, whereas the default layout would assign 96 / 4 = 24 rows per
        # warp.  Zero-extend only this admitted H16/K96/V96 forward to the
        # already validated H16/K128/V128 FlashQLA geometry.  The top-left
        # 96x96 state block is mathematically identical: padded q/k/v entries
        # are zero, the original 1/sqrt(96) scale is passed explicitly, and
        # KKT dot products therefore do not change.  The optimized/reference
        # backward continues to consume the original 96-wide tensors.
        pad_h16_k96_v96 = (
            int(query.shape[2]) == 16
            and int(query.shape[-1]) == 96
            and int(value.shape[-1]) == 96
        )
        if pad_h16_k96_v96:
            flash_query = torch.nn.functional.pad(query, (0, 32))
            flash_key = torch.nn.functional.pad(key, (0, 32))
            flash_value = torch.nn.functional.pad(value, (0, 32))
        else:
            flash_query = query
            flash_key = key
            flash_value = value
        with _profile_cuda_ms("chunk/flashqla_tile_fwd_total_ms"):
            _flash_g_cumsum, _flash_A, text_o, anchor_states, _final_state, _ = chunk_gated_delta_rule_fwd(
                q=flash_query,
                k=flash_key,
                v=flash_value,
                g=g,
                beta=beta,
                scale=float(scale),
                initial_state=None,
                cu_seqlens=text_cu_seqlens,
                output_final_state=False,
                output_h=anchor_count > 0,
                output_h_stride=8,
                chunk_start_h=None,
                auto_cp=False,
                state_v_first=False,
                enable_fwd_cp_cache=False,
            )
        if pad_h16_k96_v96:
            # Drop the algebraically-zero padded rows/columns before exposing
            # producer outputs to the HSD anchor/global-read path.
            text_o = text_o[..., :96].contiguous()
            if anchor_states is not None:
                anchor_states = anchor_states[..., :96, :96].contiguous()
        if ctx.backward_backend == "optimized":
            g_cumsum, A = _prepare_fla_backward_intermediates(
                key=key,
                value=value,
                beta=beta,
                g=g,
                text_cu_seqlens=text_cu_seqlens,
            )
        else:
            g_cumsum = g.new_empty((0,))
            A = g.new_empty((0,))
        if anchor_states is None:
            if anchor_count != 0:
                raise RuntimeError(
                    "FlashQLA TileLang HSD forward did not return anchor states."
                )
            anchor_states = torch.empty(
                (
                    query.shape[0],
                    0,
                    query.shape[2],
                    query.shape[3],
                    value.shape[-1],
                ),
                device=query.device,
                dtype=torch.float32,
            )

        with _profile_cuda_ms("chunk/flashqla_tile_anchor_read_ms"):
            anchor_o, anchor_states = _run_flashqla_compact_anchor_read(
                text_o=text_o,
                anchor_q=anchor_q,
                compact_states=anchor_states,
                scale=float(scale),
            )
        ctx.save_for_backward(
            query,
            key,
            value,
            beta,
            g,
            g_cumsum,
            A,
            anchor_q,
            anchor_states,
            boundary_counts,
            text_cu_seqlens,
            anchor_cu_seqlens,
        )
        return text_o, anchor_o, anchor_states

    @staticmethod
    def backward(ctx, dtext_o: torch.Tensor, danchor_o: torch.Tensor, danchor_states: torch.Tensor):
        (
            query,
            key,
            value,
            beta,
            g,
            g_cumsum,
            A,
            anchor_q,
            anchor_states,
            boundary_counts,
            text_cu_seqlens,
            anchor_cu_seqlens,
        ) = ctx.saved_tensors
        if ctx.backward_backend == "reference":
            with _profile_cuda_ms("chunk/flashqla_tile_reference_bwd_ms"):
                grad_inputs = _run_reference_backward(
                    query=query,
                    key=key,
                    value=value,
                    beta=beta,
                    g=g,
                    anchor_q=anchor_q,
                    boundary_counts=boundary_counts,
                    text_cu_seqlens=text_cu_seqlens,
                    anchor_cu_seqlens=anchor_cu_seqlens,
                    dtext_o=dtext_o,
                    danchor_o=danchor_o,
                    danchor_states=danchor_states,
                    scale=ctx.scale,
                    needs_grad=ctx.needs_input_grad,
                )
        else:
            with _profile_cuda_ms("chunk/flashqla_tile_hsd_bwd_ms"):
                grad_inputs = _run_flashqla_compact_optimized_backward(
                    query=query,
                    key=key,
                    value=value,
                    beta=beta,
                    g=g,
                    g_cumsum=g_cumsum,
                    A=A,
                    anchor_q=anchor_q,
                    anchor_states=anchor_states,
                    dtext_o=dtext_o,
                    danchor_o=danchor_o,
                    danchor_states=danchor_states,
                    scale=ctx.scale,
                    needs_grad=ctx.needs_input_grad,
                    text_cu_seqlens=text_cu_seqlens,
                    anchor_cu_seqlens=anchor_cu_seqlens,
                )
        return (*grad_inputs, None, None, None, None, None)


def hsd_flashqla_tile_gated_delta_rule_with_anchor_states(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    anchor_q: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: Optional[torch.Tensor] = None,
    anchor_cu_seqlens: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    backward_backend: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Run the opt-in Hopper TileLang producer for an admitted HSD layout.

    ``backward_backend=None`` preserves the legacy environment-variable
    resolution.  Supplying it explicitly makes the model configuration the
    source of truth and freezes that choice on the autograd context.
    """

    if not query.is_cuda or text_cu_seqlens is None or anchor_cu_seqlens is None:
        return None
    if query.dtype not in {torch.float16, torch.bfloat16}:
        return None
    if query.ndim != 4 or key.shape != query.shape or value.ndim != 4:
        return None
    batch, text_len, heads, key_dim = query.shape
    value_dim = int(value.shape[-1])
    resolved_backward_backend = _resolve_flashqla_tile_bwd_backend(
        backward_backend
    )
    if not _validate_flashqla_tile_geometry_backend(
        batch=batch,
        heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
        backward_backend=resolved_backward_backend,
    ):
        return None
    if value.shape[:3] != (batch, text_len, heads):
        return None
    if beta.shape != (batch, text_len, heads) or g.shape != (batch, text_len, heads):
        return None
    if (
        anchor_q.ndim != 4
        or anchor_q.shape[0] != batch
        or anchor_q.shape[2:] != (heads, key_dim)
    ):
        return None
    if (
        text_len <= 0
        or boundary_counts.numel() != anchor_q.shape[1]
    ):
        return None
    if text_cu_seqlens.ndim != 1 or anchor_cu_seqlens.ndim != 1:
        return None
    if text_cu_seqlens.numel() != anchor_cu_seqlens.numel() or text_cu_seqlens.numel() < 2:
        return None
    if torch.cuda.get_device_capability(query.device)[0] != 9:
        return None

    # FlashQLA's packed Hopper kernels use int32 metadata. Sequence lengths in
    # this fixed training backend are far below the limit, so canonicalize only
    # the private producer metadata here.
    text_cu, anchor_cu, boundary, packed_layout_valid = (
        _prepare_flashqla_packed_metadata(
            boundary_counts=boundary_counts,
            text_cu_seqlens=text_cu_seqlens,
            anchor_cu_seqlens=anchor_cu_seqlens,
            text_len=text_len,
            anchor_count=int(anchor_q.shape[1]),
            device=query.device,
        )
    )
    if not packed_layout_valid:
        return None

    try:
        from ._flashqla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd  # noqa: F401
    except Exception:
        return None

    if scale is None:
        scale = key_dim**-0.5
    return HSDFlashQLATileWithAnchorStatesFunction.apply(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        beta.contiguous(),
        g.contiguous(),
        anchor_q.contiguous(),
        boundary,
        text_cu,
        anchor_cu,
        float(scale),
        resolved_backward_backend,
    )


def hsd_flashqla_tile_gated_delta_rule_inference(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    anchor_cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the training-identical FlashQLA producer for inference prefill.

    Unlike :func:`hsd_flashqla_tile_gated_delta_rule_with_anchor_states`,
    this entry point is inference-only and also returns the recurrent state
    after each packed sequence.  The state ABI is HSD-native ``[H, K, V]``:
    compact anchor states are ``[1, M, H, K, V]`` and final states are
    ``[N, H, K, V]`` for ``N = len(text_cu_seqlens) - 1``.  It intentionally
    does not compute the anchor query read; serving already owns that step.

    This is a fail-closed production entry point rather than an optional
    training fast path.  Invalid geometry, packed metadata, dtype, device, or
    initial-state contracts raise instead of silently selecting a different
    recurrence implementation.
    """

    prefix = "FlashQLA TileLang HSD inference"
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise RuntimeError(
            f"{prefix} requires rank-4 query/key/value tensors; "
            f"received query={tuple(query.shape)}, key={tuple(key.shape)}, "
            f"value={tuple(value.shape)}."
        )
    if key.shape != query.shape:
        raise RuntimeError(
            f"{prefix} requires key.shape == query.shape; "
            f"received query={tuple(query.shape)}, key={tuple(key.shape)}."
        )
    batch, text_len, heads, key_dim = query.shape
    value_dim = int(value.shape[-1])
    if batch != 1:
        raise RuntimeError(
            f"{prefix} requires the packed physical batch dimension to be 1; "
            f"received B={batch}."
        )
    if text_len <= 0:
        raise RuntimeError(f"{prefix} requires at least one text token.")
    if value.shape[:3] != (batch, text_len, heads):
        raise RuntimeError(
            f"{prefix} value prefix shape mismatch: expected "
            f"{(batch, text_len, heads)}, received {tuple(value.shape[:3])}."
        )
    if beta.shape != (batch, text_len, heads):
        raise RuntimeError(
            f"{prefix} beta shape mismatch: expected "
            f"{(batch, text_len, heads)}, received {tuple(beta.shape)}."
        )
    if g.shape != (batch, text_len, heads):
        raise RuntimeError(
            f"{prefix} g shape mismatch: expected "
            f"{(batch, text_len, heads)}, received {tuple(g.shape)}."
        )
    if query.dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError(
            f"{prefix} requires FP16 or BF16 activations; received {query.dtype}."
        )
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise RuntimeError(
            f"{prefix} requires query/key/value to share one activation dtype; "
            f"received query={query.dtype}, key={key.dtype}, value={value.dtype}."
        )
    # The MCore/HF training producer receives sigmoid(beta) in activation
    # precision and the log decay in FP32.  Keeping this exact prevents the
    # serving adapter from reintroducing a producer-input precision drift.
    if beta.dtype != query.dtype:
        raise RuntimeError(
            f"{prefix} requires beta in the activation dtype {query.dtype}; "
            f"received {beta.dtype}."
        )
    if g.dtype != torch.float32:
        raise RuntimeError(
            f"{prefix} requires the raw log decay g in FP32; received {g.dtype}."
        )
    for name, tensor in (("key", key), ("value", value), ("beta", beta), ("g", g)):
        if tensor.device != query.device:
            raise RuntimeError(
                f"{prefix} requires {name} on {query.device}; received {tensor.device}."
            )

    if not _validate_flashqla_tile_geometry_backend(
        batch=batch,
        heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
        # Geometry admission is shared with the validated training producer.
        # ``optimized`` is supported by every admitted forward geometry and
        # does not select a backward implementation in this inference path.
        backward_backend="optimized",
    ):
        raise RuntimeError(
            f"{prefix} does not admit geometry "
            f"B={batch}, H={heads}, K={key_dim}, V={value_dim}."
        )

    integer_dtypes = {torch.int32, torch.int64}
    for name, tensor in (
        ("boundary_counts", boundary_counts),
        ("text_cu_seqlens", text_cu_seqlens),
        ("anchor_cu_seqlens", anchor_cu_seqlens),
    ):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1:
            raise RuntimeError(f"{prefix} requires {name} to be a rank-1 tensor.")
        if tensor.dtype not in integer_dtypes:
            raise RuntimeError(
                f"{prefix} requires integer {name}; received {tensor.dtype}."
            )
    if (
        text_cu_seqlens.numel() < 2
        or text_cu_seqlens.numel() != anchor_cu_seqlens.numel()
    ):
        raise RuntimeError(
            f"{prefix} requires matching text/anchor cu-seqlens with at "
            "least one packed sequence."
        )
    anchor_count = int(boundary_counts.numel())
    text_cu, _anchor_cu, _boundary, packed_layout_valid = (
        _prepare_flashqla_packed_metadata(
            boundary_counts=boundary_counts,
            text_cu_seqlens=text_cu_seqlens,
            anchor_cu_seqlens=anchor_cu_seqlens,
            text_len=text_len,
            anchor_count=anchor_count,
            device=query.device,
        )
    )
    if not packed_layout_valid:
        raise RuntimeError(
            f"{prefix} requires exact per-sequence 512-token anchor "
            "boundaries and sidecars spanning all text and anchor states."
        )

    sequence_count = int(text_cu_seqlens.numel()) - 1
    if initial_state is not None:
        expected_initial_shape = (
            sequence_count,
            heads,
            key_dim,
            value_dim,
        )
        if tuple(initial_state.shape) != expected_initial_shape:
            raise RuntimeError(
                f"{prefix} initial_state shape mismatch: expected "
                f"{expected_initial_shape}, received {tuple(initial_state.shape)}."
            )
        if initial_state.dtype != torch.float32:
            raise RuntimeError(
                f"{prefix} requires FP32 initial_state; received "
                f"{initial_state.dtype}."
            )
        if initial_state.device != query.device:
            raise RuntimeError(
                f"{prefix} requires initial_state on {query.device}; "
                f"received {initial_state.device}."
            )

    if not query.is_cuda:
        raise RuntimeError(f"{prefix} requires CUDA tensors on Hopper SM90.")
    if torch.cuda.get_device_capability(query.device)[0] != 9:
        raise RuntimeError(f"{prefix} requires Hopper SM90.")
    try:
        from ._flashqla.ops.gated_delta_rule.chunk import (
            chunk_gated_delta_rule_fwd,
        )
    except Exception as exc:
        raise RuntimeError(
            f"{prefix} could not import the shared FlashQLA producer."
        ) from exc

    if scale is None:
        scale = key_dim**-0.5

    # Match the training wrapper's admitted H16/K96/V96 padding exactly.
    # The shared Hopper kernel runs this profile as K128/V128 with algebraic
    # zero padding, while the public HSD state ABI remains K96/V96.
    pad_h16_k96_v96 = (
        heads == 16 and key_dim == 96 and value_dim == 96
    )
    if pad_h16_k96_v96:
        flash_query = torch.nn.functional.pad(query, (0, 32))
        flash_key = torch.nn.functional.pad(key, (0, 32))
        flash_value = torch.nn.functional.pad(value, (0, 32))
        flash_initial_state = (
            None
            if initial_state is None
            else torch.nn.functional.pad(initial_state, (0, 32, 0, 32))
        )
    else:
        flash_query = query
        flash_key = key
        flash_value = value
        flash_initial_state = initial_state

    with torch.inference_mode(), _profile_cuda_ms(
        "chunk/flashqla_tile_inference_fwd_total_ms"
    ):
        _, _, text_o, anchor_states, final_state, _ = (
            chunk_gated_delta_rule_fwd(
                q=flash_query.contiguous(),
                k=flash_key.contiguous(),
                v=flash_value.contiguous(),
                g=g.contiguous(),
                beta=beta.contiguous(),
                scale=float(scale),
                initial_state=(
                    None
                    if flash_initial_state is None
                    else flash_initial_state.contiguous()
                ),
                cu_seqlens=text_cu,
                output_final_state=True,
                output_h=anchor_count > 0,
                output_h_stride=8,
                auto_cp=False,
                state_v_first=False,
                enable_fwd_cp_cache=False,
            )
        )

    if anchor_states is None:
        if anchor_count:
            raise RuntimeError(
                f"{prefix} did not return the requested compact anchor states."
            )
        anchor_states = torch.empty(
            (1, 0, heads, flash_query.shape[-1], flash_value.shape[-1]),
            dtype=torch.float32,
            device=query.device,
        )
    if final_state is None:
        raise RuntimeError(f"{prefix} did not return final recurrent states.")

    if pad_h16_k96_v96:
        text_o = text_o[..., :value_dim]
        anchor_states = anchor_states[..., :key_dim, :value_dim]
        final_state = final_state[..., :key_dim, :value_dim]

    expected_text_shape = (batch, text_len, heads, value_dim)
    expected_anchor_shape = (
        batch,
        anchor_count,
        heads,
        key_dim,
        value_dim,
    )
    expected_final_shape = (
        sequence_count,
        heads,
        key_dim,
        value_dim,
    )
    if tuple(text_o.shape) != expected_text_shape:
        raise RuntimeError(
            f"{prefix} text output shape mismatch: expected "
            f"{expected_text_shape}, received {tuple(text_o.shape)}."
        )
    if tuple(anchor_states.shape) != expected_anchor_shape:
        raise RuntimeError(
            f"{prefix} compact anchor-state shape mismatch: expected "
            f"{expected_anchor_shape}, received {tuple(anchor_states.shape)}."
        )
    if tuple(final_state.shape) != expected_final_shape:
        raise RuntimeError(
            f"{prefix} final-state shape mismatch: expected "
            f"{expected_final_shape}, received {tuple(final_state.shape)}."
        )
    if anchor_states.dtype != torch.float32 or final_state.dtype != torch.float32:
        raise RuntimeError(
            f"{prefix} requires FP32 compact/final recurrent states; received "
            f"anchor={anchor_states.dtype}, final={final_state.dtype}."
        )
    return (
        text_o.contiguous(),
        anchor_states.contiguous(),
        final_state.contiguous(),
    )


def hsd_chunk_gated_delta_rule_with_anchor_states(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    anchor_q: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: Optional[torch.Tensor] = None,
    anchor_cu_seqlens: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if triton is None or chunk_gated_delta_rule is None:
        return None
    if not query.is_cuda:
        return None
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return None
    if query.shape != key.shape:
        return None
    batch, text_len, heads, key_dim = query.shape
    if value.shape[:3] != (batch, text_len, heads):
        return None
    if beta.shape != (batch, text_len, heads) or g.shape != (batch, text_len, heads):
        return None
    if anchor_q.ndim != 4 or anchor_q.shape[:1] != (batch,) or anchor_q.shape[2:] != (heads, key_dim):
        return None
    anchor_count = int(anchor_q.shape[1])
    if boundary_counts.ndim != 1 or int(boundary_counts.shape[0]) != anchor_count:
        return None
    if key_dim > 256 or text_len <= 0:
        return None
    packed = text_cu_seqlens is not None or anchor_cu_seqlens is not None
    if packed:
        if prepare_chunk_indices is None or prepare_chunk_offsets is None:
            return None
        if text_cu_seqlens is None or anchor_cu_seqlens is None:
            return None
        if batch != 1:
            return None
        if text_cu_seqlens.ndim != 1 or anchor_cu_seqlens.ndim != 1:
            return None
        if text_cu_seqlens.numel() < 2 or int(text_cu_seqlens.numel()) != int(anchor_cu_seqlens.numel()):
            return None
        text_cu = text_cu_seqlens.to(device=query.device, dtype=torch.long).contiguous()
        anchor_cu = anchor_cu_seqlens.to(device=query.device, dtype=torch.long).contiguous()
        text_cu_cpu = text_cu.detach().to(device="cpu", dtype=torch.long)
        anchor_cu_cpu = anchor_cu.detach().to(device="cpu", dtype=torch.long)
        if int(text_cu_cpu[0].item()) != 0 or int(text_cu_cpu[-1].item()) != text_len:
            return None
        if int(anchor_cu_cpu[0].item()) != 0 or int(anchor_cu_cpu[-1].item()) != anchor_count:
            return None
        if bool((text_cu_cpu[1:] < text_cu_cpu[:-1]).any().item()):
            return None
        if bool((anchor_cu_cpu[1:] < anchor_cu_cpu[:-1]).any().item()):
            return None
    else:
        text_cu = boundary_counts.new_empty((0,), dtype=torch.long)
        anchor_cu = boundary_counts.new_empty((0,), dtype=torch.long)

    boundaries = boundary_counts.to(device=query.device, dtype=torch.long)
    boundary_cpu = boundaries.detach().to(device="cpu", dtype=torch.long)
    if packed:
        for doc_idx in range(int(text_cu_cpu.numel()) - 1):
            text_start = int(text_cu_cpu[doc_idx].item())
            text_end = int(text_cu_cpu[doc_idx + 1].item())
            doc_len = text_end - text_start
            anchor_start = int(anchor_cu_cpu[doc_idx].item())
            anchor_end = int(anchor_cu_cpu[doc_idx + 1].item())
            local_boundaries = boundary_cpu[anchor_start:anchor_end] - text_start
            if bool((local_boundaries < 0).any().item()) or bool((local_boundaries > doc_len).any().item()):
                return None
            if local_boundaries.numel() > 1 and bool((local_boundaries[1:] < local_boundaries[:-1]).any().item()):
                return None
            is_final = local_boundaries == doc_len
            if not bool((is_final | ((local_boundaries % FLA_CHUNK_SIZE) == 0)).all().item()):
                return None
    else:
        if bool((boundaries < 0).any().item()) or bool((boundaries > text_len).any().item()):
            return None
        if boundaries.numel() > 1 and bool((boundaries[1:] < boundaries[:-1]).any().item()):
            return None
        is_final = boundaries == text_len
        if not bool((is_final | ((boundaries % FLA_CHUNK_SIZE) == 0)).all().item()):
            return None

    if scale is None:
        scale = key_dim**-0.5
    device_ctx = custom_device_ctx(query.device.index) if query.device.index is not None else contextlib.nullcontext()
    with device_ctx:
        return HSDChunkGatedDeltaWithAnchorStatesFunction.apply(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            beta.contiguous(),
            g.contiguous(),
            anchor_q.contiguous(),
            boundary_counts.to(device=query.device, dtype=torch.long).contiguous(),
            text_cu,
            anchor_cu,
            float(scale),
        )


__all__ = [
    "clear_hsd_chunk_profile_stats",
    "hsd_flashqla_tile_gated_delta_rule_inference",
    "hsd_flashqla_tile_gated_delta_rule_with_anchor_states",
    "hsd_chunk_gated_delta_rule_with_anchor_states",
    "pop_hsd_chunk_profile_stats",
]
