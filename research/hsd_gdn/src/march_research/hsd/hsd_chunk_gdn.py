# Copyright (c) ModelScope Contributors. All rights reserved.
"""Training helpers for HSD chunk GDN selective scans.

This module is intentionally local to mcore-bridge instead of patching the
installed FLA package.  It uses the FLA full-sequence forward kernels, exposes
fp32 summary-boundary states, and routes summary-state gradients back into the
chunk recurrence through a local ``dh_external`` backward kernel.
"""

from __future__ import annotations

import contextlib
import os
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    from fla.ops.common.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h,
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    )
    from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
    from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices, prepare_chunk_offsets
    from fla.ops.utils.constant import RCP_LN2
    from fla.ops.utils.op import exp2
    from fla.utils import custom_device_ctx
except Exception:  # pragma: no cover - optional CUDA/Triton dependency.
    triton = None
    tl = None
    chunk_gated_delta_rule_fwd_h = None
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64 = None
    chunk_bwd_dqkwg = None
    chunk_bwd_dv_local = None
    chunk_fwd_o = None
    chunk_gated_delta_rule = None
    chunk_gated_delta_rule_fwd_intra = None
    chunk_gated_delta_rule_fwd_kkt_solve_kernel = None
    prepare_wy_repr_bwd = None
    recompute_w_u_fwd = None
    chunk_local_cumsum = None
    prepare_chunk_indices = None
    prepare_chunk_offsets = None
    RCP_LN2 = None
    exp2 = None
    custom_device_ctx = None

# This private FLA symbol is needed only by the optimized compact backward.
# Keep it out of the broad dependency import so an older FLA build does not
# disable the established reference/native LC50B paths at module import time.
if triton is not None:
    try:
        from fla.ops.gated_delta_rule.chunk_fwd import (
            chunk_gated_delta_rule_fwd_kkt_solve_kernel,
        )
    except (ImportError, AttributeError):  # pragma: no cover - FLA version dependent.
        chunk_gated_delta_rule_fwd_kkt_solve_kernel = None


FLA_CHUNK_SIZE = 64
_PROFILE_STATS: dict[str, float] = {}
_FLASHQLA_PACKED_METADATA_CACHE: list[tuple] = []
_FLASHQLA_TILE_BACKWARD_ALIASES = {
    "direct": "direct_a",
    "direct-a": "direct_a",
    "direct_a": "direct_a",
    "native": "native",
    "reference": "reference",
    "ref": "reference",
    "split": "split",
    "optimized": "optimized",
    "fla": "optimized",
}
# Keep every admitted training geometry explicit.  The LC50B profile has
# exercised all experimental backward implementations.  The H16 producer
# profiles use the same generic Hopper forward and the FLA-compatible optimized
# backward; the native, split, and direct-A implementations still contain
# K192/V384 specializations and must never be selected for their smaller
# states.
_FLASHQLA_TILE_GEOMETRY_BACKENDS = {
    (1, 6, 192, 384): frozenset(
        {"direct_a", "native", "reference", "split", "optimized"}
    ),
    # Transformer-matched state geometry used by the H16/K96/V96 ablation.
    # Keep this exact: nearby H16 shapes have not been compiled or validated.
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
            f"{source} must be one of direct_a, native, reference, split, "
            f"or optimized; received {raw!r}."
        )
    return backend


def _resolve_flashqla_tile_bwd_backend(
    backward_backend: Optional[str] = None,
) -> str:
    """Resolve and freeze the FlashQLA producer-backward implementation.

    The explicit selector is the source of truth for new experiments.  Legacy
    booleans remain supported while the direct-A path is being validated.
    Resolving this in forward (and saving the result on ``ctx``) prevents an
    environment change between forward and backward from mixing two graphs.
    """

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
    if _env_bool("HSD_FLASHQLA_TILE_SPLIT_BWD", False):
        return "split"
    legacy_native = os.environ.get(
        "HSD_FLASHQLA_TILE_NATIVE_BWD",
        "",
    ).strip()
    if legacy_native:
        return (
            "native"
            if _env_bool("HSD_FLASHQLA_TILE_NATIVE_BWD", False)
            else "optimized"
        )
    # Keep the multi-kernel compact backward as the safe default.  The fused
    # direct-A path remains explicitly selectable, but real packed training has
    # exposed a layout-dependent non-finite gradient that is not covered by the
    # small random parity tests yet.  Do not silently select it for training
    # until the long packed regression has passed.
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
        # The admitted OLMo3 producer geometry is a closed production
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
    summary_cu_seqlens: torch.Tensor,
    text_len: int,
    summary_count: int,
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
            summary_cu_seqlens,
        )
    )
    for entry in _FLASHQLA_PACKED_METADATA_CACHE:
        (
            cached_boundary,
            cached_text_cu,
            cached_summary_cu,
            cached_versions,
            cached_text_len,
            cached_summary_count,
            cached_device,
            result,
        ) = entry
        if (
            boundary_counts is cached_boundary
            and text_cu_seqlens is cached_text_cu
            and summary_cu_seqlens is cached_summary_cu
            and versions == cached_versions
            and int(text_len) == cached_text_len
            and int(summary_count) == cached_summary_count
            and device == cached_device
        ):
            return result

    text_cu = text_cu_seqlens.to(
        device=device,
        dtype=torch.int32,
    ).contiguous()
    summary_cu = summary_cu_seqlens.to(
        device=device,
        dtype=torch.int32,
    ).contiguous()
    boundary = boundary_counts.to(
        device=device,
        dtype=torch.long,
    ).contiguous()

    text_cu_cpu = text_cu.detach().cpu()
    summary_cu_cpu = summary_cu.detach().cpu()
    boundary_cpu = boundary.detach().cpu()
    valid = (
        int(text_cu_cpu[0]) == 0
        and int(text_cu_cpu[-1]) == int(text_len)
        and int(summary_cu_cpu[0]) == 0
        and int(summary_cu_cpu[-1]) == int(summary_count)
    )
    if valid:
        for doc_idx in range(int(text_cu_cpu.numel()) - 1):
            text_start = int(text_cu_cpu[doc_idx])
            text_end = int(text_cu_cpu[doc_idx + 1])
            summary_start = int(summary_cu_cpu[doc_idx])
            summary_end = int(summary_cu_cpu[doc_idx + 1])
            expected_count = (text_end - text_start) // 512
            if summary_end - summary_start != expected_count:
                valid = False
                break
            local = boundary_cpu[summary_start:summary_end] - text_start
            expected = torch.arange(
                1,
                expected_count + 1,
                dtype=torch.long,
            ) * 512
            if not torch.equal(local, expected):
                valid = False
                break

    result = (text_cu, summary_cu, boundary, valid)
    _FLASHQLA_PACKED_METADATA_CACHE.insert(
        0,
        (
            boundary_counts,
            text_cu_seqlens,
            summary_cu_seqlens,
            versions,
            int(text_len),
            int(summary_count),
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


def _run_segmented_reference(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    summary_q: torch.Tensor,
    boundary_counts: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, text_len, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    text_outputs = []
    summary_outputs = []
    summary_states = []
    state = None
    cursor = 0
    zero_state = query.new_zeros(batch, heads, key_dim, value_dim)

    for summary_idx, boundary in enumerate(boundary_counts.tolist()):
        boundary = int(boundary)
        if boundary < cursor:
            raise ValueError("summary positions must be ordered after non-decreasing text prefixes.")
        if boundary > text_len:
            raise ValueError("summary positions must not exceed text length.")
        if boundary > cursor:
            out, state = chunk_gated_delta_rule(
                query[:, cursor:boundary],
                key[:, cursor:boundary],
                value[:, cursor:boundary],
                g=g[:, cursor:boundary],
                beta=beta[:, cursor:boundary],
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=False,
                cu_seqlens=None,
            )
            text_outputs.append(out)
            cursor = boundary

        state_for_summary = zero_state if state is None else state
        summary_states.append(state_for_summary)
        summary_outputs.append(
            torch.einsum(
                "bhk,bhkv->bhv",
                summary_q[:, summary_idx].to(state_for_summary.dtype),
                state_for_summary,
            )
            * scale
        )

    if cursor < text_len:
        out, _ = chunk_gated_delta_rule(
            query[:, cursor:text_len],
            key[:, cursor:text_len],
            value[:, cursor:text_len],
            g=g[:, cursor:text_len],
            beta=beta[:, cursor:text_len],
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=None,
        )
        text_outputs.append(out)

    text_o = torch.cat(text_outputs, dim=1) if text_outputs else query.new_zeros(batch, 0, heads, value_dim)
    summary_o = torch.stack(summary_outputs, dim=1).to(text_o.dtype) if summary_outputs else query.new_zeros(
        batch, 0, heads, value_dim
    )
    states = torch.stack(summary_states, dim=1) if summary_states else query.new_zeros(
        batch, 0, heads, key_dim, value_dim
    )
    return text_o, summary_o, states


def _run_segmented_packed_reference(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    summary_q: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    summary_cu_seqlens: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if query.shape[0] != 1:
        raise NotImplementedError("HSD packed reference path requires flattened batch size 1.")
    text_cu_cpu = text_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
    summary_cu_cpu = summary_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
    if text_cu_cpu.numel() != summary_cu_cpu.numel():
        raise ValueError("packed text and summary cu_seqlens must describe the same documents.")

    text_outputs = []
    summary_outputs = []
    state_outputs = []
    for doc_idx in range(int(text_cu_cpu.numel()) - 1):
        text_start = int(text_cu_cpu[doc_idx].item())
        text_end = int(text_cu_cpu[doc_idx + 1].item())
        summary_start = int(summary_cu_cpu[doc_idx].item())
        summary_end = int(summary_cu_cpu[doc_idx + 1].item())
        local_boundaries = boundary_counts[summary_start:summary_end] - text_start
        text_o_i, summary_o_i, states_i = _run_segmented_reference(
            query=query[:, text_start:text_end],
            key=key[:, text_start:text_end],
            value=value[:, text_start:text_end],
            beta=beta[:, text_start:text_end],
            g=g[:, text_start:text_end],
            summary_q=summary_q[:, summary_start:summary_end],
            boundary_counts=local_boundaries,
            scale=scale,
        )
        text_outputs.append(text_o_i)
        if summary_end > summary_start:
            summary_outputs.append(summary_o_i)
            state_outputs.append(states_i)

    batch, _, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    text_o = torch.cat(text_outputs, dim=1) if text_outputs else query.new_zeros(batch, 0, heads, value_dim)
    summary_o = (
        torch.cat(summary_outputs, dim=1)
        if summary_outputs else query.new_zeros(batch, 0, heads, value_dim)
    )
    states = (
        torch.cat(state_outputs, dim=1)
        if state_outputs else query.new_zeros(batch, 0, heads, key_dim, value_dim)
    )
    return text_o, summary_o, states


def _is_packed_cu(text_cu_seqlens: torch.Tensor | None, summary_cu_seqlens: torch.Tensor | None) -> bool:
    return (
        text_cu_seqlens is not None
        and summary_cu_seqlens is not None
        and text_cu_seqlens.numel() > 0
        and summary_cu_seqlens.numel() > 0
    )


def _packed_chunk_index(
    *,
    doc_idx: int,
    local_boundary: int,
    doc_len: int,
    chunk_offsets_cpu: torch.Tensor,
) -> int | None:
    if local_boundary == doc_len:
        return None
    return int(chunk_offsets_cpu[doc_idx].item()) + local_boundary // FLA_CHUNK_SIZE


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
        return (triton.cdiv(value_dim, meta["BV"]) * sequence_count * value_heads,)

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

    @triton.jit
    def _hsd_recompute_v_new_from_chunk_states_kernel(
        w,
        u,
        chunk_start_states,
        chunk_indices,
        text_cu_seqlens,
        v_new,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
    ):
        i_v, i_chunk, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_doc = tl.load(chunk_indices + i_chunk * 2).to(tl.int64)
        i_local_chunk = tl.load(chunk_indices + i_chunk * 2 + 1).to(tl.int64)
        doc_start = tl.load(text_cu_seqlens + i_doc).to(tl.int64)
        doc_end = tl.load(text_cu_seqlens + i_doc + 1).to(tl.int64)
        chunk_start = doc_start + i_local_chunk * BT

        offs_t = tl.arange(0, BT)
        offs_v = i_v * BV + tl.arange(0, BV)
        token_idx = chunk_start + offs_t
        token_mask = token_idx < doc_end
        value_mask = offs_v < V
        b_v_prime = tl.zeros((BT, BV), dtype=tl.float32)

        for i_k in tl.static_range(0, (K + BK - 1) // BK):
            offs_k = i_k * BK + tl.arange(0, BK)
            key_mask = offs_k < K
            w_ptrs = w + (
                (token_idx[:, None] * H + i_h) * K
                + offs_k[None, :]
            )
            h_ptrs = chunk_start_states + (
                ((i_chunk * H + i_h) * K + offs_k[:, None]) * V
                + offs_v[None, :]
            )
            b_w = tl.load(
                w_ptrs,
                mask=token_mask[:, None] & key_mask[None, :],
                other=0.0,
            )
            b_h = tl.load(
                h_ptrs,
                mask=key_mask[:, None] & value_mask[None, :],
                other=0.0,
            )
            b_v_prime += tl.dot(b_w, b_h)

        u_ptrs = u + (
            (token_idx[:, None] * H + i_h) * V
            + offs_v[None, :]
        )
        out_ptrs = v_new + (
            (token_idx[:, None] * H + i_h) * V
            + offs_v[None, :]
        )
        mask = token_mask[:, None] & value_mask[None, :]
        b_u = tl.load(u_ptrs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptrs, b_u - b_v_prime, mask=mask)

    @triton.jit(do_not_specialize=["TEXT_LEN"])
    def _hsd_summary_read_fwd_kernel(
        summary_q,
        chunk_states,
        final_state,
        boundary_counts,
        text_cu_seqlens,
        summary_cu_seqlens,
        chunk_offsets,
        summary_o,
        summary_states,
        scale,
        TEXT_LEN,
        SUMMARY_COUNT: tl.constexpr,
        CHUNK_COUNT: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        N_DOCS: tl.constexpr,
        IS_PACKED: tl.constexpr,
    ):
        i_v, i_s, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_b, i_h = i_bh // H, i_bh % H
        offs_v = i_v * BV + tl.arange(0, BV)
        v_mask = offs_v < V

        doc_idx = i_b
        text_start = 0
        doc_len = TEXT_LEN
        chunk_base = 0
        if IS_PACKED:
            doc_idx = 0
            for i_doc in range(N_DOCS):
                summary_start = tl.load(summary_cu_seqlens + i_doc)
                summary_end = tl.load(summary_cu_seqlens + i_doc + 1)
                in_doc = (i_s >= summary_start) & (i_s < summary_end)
                doc_idx = tl.where(in_doc, i_doc, doc_idx)
            text_start = tl.load(text_cu_seqlens + doc_idx)
            text_end = tl.load(text_cu_seqlens + doc_idx + 1)
            doc_len = text_end - text_start
            chunk_base = tl.load(chunk_offsets + doc_idx)

        boundary = tl.load(boundary_counts + i_s)
        local_boundary = boundary - text_start
        is_zero_state = (local_boundary == 0) & (doc_len == 0)
        is_final_state = (local_boundary == doc_len) & (~is_zero_state)
        chunk_idx = chunk_base + local_boundary // BT

        b_o = tl.zeros([BV], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            offs_k = i_k * BK + tl.arange(0, BK)
            k_mask = offs_k < K
            q_ptr = summary_q + (((i_b * SUMMARY_COUNT + i_s) * H + i_h) * K + offs_k)
            b_q = tl.load(q_ptr, mask=k_mask, other=0.0).to(tl.float32)

            chunk_ptrs = (
                chunk_states
                + ((((i_b * CHUNK_COUNT + chunk_idx) * H + i_h) * K + offs_k[:, None]) * V + offs_v[None, :])
            )
            final_ptrs = final_state + (((doc_idx * H + i_h) * K + offs_k[:, None]) * V + offs_v[None, :])
            state_ptrs = tl.where(is_final_state, final_ptrs, chunk_ptrs)
            state_mask = (k_mask[:, None] & v_mask[None, :]) & (~is_zero_state)
            b_state = tl.load(state_ptrs, mask=state_mask, other=0.0).to(tl.float32)

            out_state_ptrs = (
                summary_states
                + ((((i_b * SUMMARY_COUNT + i_s) * H + i_h) * K + offs_k[:, None]) * V + offs_v[None, :])
            )
            tl.store(out_state_ptrs, b_state, mask=k_mask[:, None] & v_mask[None, :])
            b_o += tl.sum(b_q[:, None] * b_state, axis=0)

        out_ptr = summary_o + (((i_b * SUMMARY_COUNT + i_s) * H + i_h) * V + offs_v)
        tl.store(out_ptr, (b_o * scale).to(out_ptr.dtype.element_ty), mask=v_mask)

    @triton.jit
    def _hsd_summary_read_bwd_q_kernel(
        dsummary_o,
        summary_states,
        dsummary_q,
        scale,
        SUMMARY_COUNT: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
    ):
        i_k, i_s, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_b, i_h = i_bh // H, i_bh % H
        offs_k = i_k * BK + tl.arange(0, BK)
        k_mask = offs_k < K
        b_dq = tl.zeros([BK], dtype=tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            offs_v = i_v * BV + tl.arange(0, BV)
            v_mask = offs_v < V
            do_ptr = dsummary_o + (((i_b * SUMMARY_COUNT + i_s) * H + i_h) * V + offs_v)
            state_ptrs = (
                summary_states
                + ((((i_b * SUMMARY_COUNT + i_s) * H + i_h) * K + offs_k[:, None]) * V + offs_v[None, :])
            )
            b_do = tl.load(do_ptr, mask=v_mask, other=0.0).to(tl.float32) * scale
            b_state = tl.load(state_ptrs, mask=k_mask[:, None] & v_mask[None, :], other=0.0).to(tl.float32)
            b_dq += tl.sum(b_state * b_do[None, :], axis=1)

        dq_ptr = dsummary_q + (((i_b * SUMMARY_COUNT + i_s) * H + i_h) * K + offs_k)
        tl.store(dq_ptr, b_dq.to(dq_ptr.dtype.element_ty), mask=k_mask)

    @triton.jit
    def _hsd_compact_summary_read_fwd_kernel(
        summary_q,
        summary_states,
        summary_o,
        scale,
        SUMMARY_COUNT: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
    ):
        i_v, i_s, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_b, i_h = i_bh // H, i_bh % H
        offs_v = i_v * BV + tl.arange(0, BV)
        v_mask = offs_v < V
        b_o = tl.zeros([BV], dtype=tl.float32)

        for i_k in range(tl.cdiv(K, BK)):
            offs_k = i_k * BK + tl.arange(0, BK)
            k_mask = offs_k < K
            q_ptr = summary_q + (
                ((i_b * SUMMARY_COUNT + i_s) * H + i_h) * K
                + offs_k
            )
            state_ptrs = summary_states + (
                (
                    (
                        (i_b * SUMMARY_COUNT + i_s) * H
                        + i_h
                    )
                    * K
                    + offs_k[:, None]
                )
                * V
                + offs_v[None, :]
            )
            b_q = tl.load(q_ptr, mask=k_mask, other=0.0).to(tl.float32)
            b_state = tl.load(
                state_ptrs,
                mask=k_mask[:, None] & v_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            b_o += tl.sum(b_q[:, None] * b_state, axis=0)

        out_ptr = summary_o + (
            ((i_b * SUMMARY_COUNT + i_s) * H + i_h) * V
            + offs_v
        )
        tl.store(
            out_ptr,
            (b_o * scale).to(out_ptr.dtype.element_ty),
            mask=v_mask,
        )

    @triton.jit(do_not_specialize=["TEXT_LEN"])
    def _hsd_summary_state_scatter_bwd_kernel(
        summary_q,
        dsummary_o,
        dsummary_states,
        boundary_counts,
        text_cu_seqlens,
        summary_cu_seqlens,
        chunk_offsets,
        dh_direct,
        dh_extra,
        dht_direct,
        dht_extra,
        scale,
        TEXT_LEN,
        SUMMARY_COUNT: tl.constexpr,
        CHUNK_COUNT: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        N_DOCS: tl.constexpr,
        IS_PACKED: tl.constexpr,
    ):
        i_v, i_k, i_s_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_s = i_s_bh % SUMMARY_COUNT
        i_bh = i_s_bh // SUMMARY_COUNT
        i_b, i_h = i_bh // H, i_bh % H
        offs_k = i_k * BK + tl.arange(0, BK)
        offs_v = i_v * BV + tl.arange(0, BV)
        k_mask = offs_k < K
        v_mask = offs_v < V

        doc_idx = i_b
        text_start = 0
        doc_len = TEXT_LEN
        chunk_base = 0
        if IS_PACKED:
            doc_idx = 0
            for i_doc in range(N_DOCS):
                summary_start = tl.load(summary_cu_seqlens + i_doc)
                summary_end = tl.load(summary_cu_seqlens + i_doc + 1)
                in_doc = (i_s >= summary_start) & (i_s < summary_end)
                doc_idx = tl.where(in_doc, i_doc, doc_idx)
            text_start = tl.load(text_cu_seqlens + doc_idx)
            text_end = tl.load(text_cu_seqlens + doc_idx + 1)
            doc_len = text_end - text_start
            chunk_base = tl.load(chunk_offsets + doc_idx)

        boundary = tl.load(boundary_counts + i_s)
        local_boundary = boundary - text_start
        is_final_state = (local_boundary == doc_len) & (local_boundary > 0)
        chunk_idx = chunk_base + local_boundary // BT
        is_chunk_state = (local_boundary > 0) & (~is_final_state) & (chunk_idx > chunk_base)

        direct_ptrs = (
            dsummary_states
            + ((((i_b * SUMMARY_COUNT + i_s) * H + i_h) * K + offs_k[:, None]) * V + offs_v[None, :])
        )
        q_ptr = summary_q + (((i_b * SUMMARY_COUNT + i_s) * H + i_h) * K + offs_k)
        do_ptr = dsummary_o + (((i_b * SUMMARY_COUNT + i_s) * H + i_h) * V + offs_v)
        b_direct = tl.load(direct_ptrs, mask=k_mask[:, None] & v_mask[None, :], other=0.0).to(tl.float32)
        b_q = tl.load(q_ptr, mask=k_mask, other=0.0).to(tl.float32)
        b_do = tl.load(do_ptr, mask=v_mask, other=0.0).to(tl.float32) * scale
        b_extra = b_q[:, None] * b_do[None, :]

        dh_chunk_idx = chunk_idx - 1
        dh_direct_ptrs = (
            dh_direct
            + ((((i_b * CHUNK_COUNT + dh_chunk_idx) * H + i_h) * K + offs_k[:, None]) * V + offs_v[None, :])
        )
        dh_extra_ptrs = (
            dh_extra
            + ((((i_b * CHUNK_COUNT + dh_chunk_idx) * H + i_h) * K + offs_k[:, None]) * V + offs_v[None, :])
        )
        chunk_mask = (k_mask[:, None] & v_mask[None, :]) & is_chunk_state
        tl.atomic_add(dh_direct_ptrs, b_direct, sem="relaxed", mask=chunk_mask)
        tl.atomic_add(dh_extra_ptrs, b_extra, sem="relaxed", mask=chunk_mask)

        dht_direct_ptrs = dht_direct + (((doc_idx * H + i_h) * K + offs_k[:, None]) * V + offs_v[None, :])
        dht_extra_ptrs = dht_extra + (((doc_idx * H + i_h) * K + offs_k[:, None]) * V + offs_v[None, :])
        final_mask = (k_mask[:, None] & v_mask[None, :]) & is_final_state
        tl.atomic_add(dht_direct_ptrs, b_direct, sem="relaxed", mask=final_mask)
        tl.atomic_add(dht_extra_ptrs, b_extra, sem="relaxed", mask=final_mask)

    @triton.heuristics({
        "USE_FINAL_STATE_GRADIENT": lambda args: args["dht"] is not None,
        "USE_DENSE_EXTERNAL_GRADIENT": lambda args: args["dh_direct"] is not None,
        "USE_COMPACT_BOUNDARY_GRADIENT": lambda args: args["compact_dh"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    })
    @triton.jit(do_not_specialize=["T"])
    def _chunk_gated_delta_rule_bwd_kernel_dhu_external(
        q,
        k,
        w,
        g,
        dht,
        dht_extra,
        dh_direct,
        dh_extra,
        compact_dh,
        do,
        dh,
        dv,
        dv2,
        cu_seqlens,
        summary_cu_seqlens,
        chunk_offsets,
        scale,
        T,
        H: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BV: tl.constexpr,
        USE_FINAL_STATE_GRADIENT: tl.constexpr,
        USE_DENSE_EXTERNAL_GRADIENT: tl.constexpr,
        USE_COMPACT_BOUNDARY_GRADIENT: tl.constexpr,
        COMPACT_CHUNK_STRIDE: tl.constexpr,
        EXTERNAL_ORDER: tl.constexpr,
        IS_VARLEN: tl.constexpr,
    ):
        # Flatten the logical (value_block, document_head) grid into grid.x.
        # Long-context packs can contain thousands of short documents, which
        # would otherwise overflow CUDA's 65,535-block grid.y limit.
        i_flat = tl.program_id(0)
        n_v_blocks = tl.cdiv(V, BV)
        i_v, i_bh = i_flat % n_v_blocks, i_flat // n_v_blocks
        i_b, i_h = i_bh // HV, i_bh % HV
        if IS_VARLEN:
            bos = tl.load(cu_seqlens + i_b).to(tl.int32)
            eos = tl.load(cu_seqlens + i_b + 1).to(tl.int32)
            T = eos - bos
            NT = tl.cdiv(T, BT)
            chunk_start = tl.load(chunk_offsets + i_b).to(tl.int32)
        else:
            NT = tl.cdiv(T, BT)
            bos = i_b * T
            chunk_start = i_b * NT

        b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

        q += (bos * H + i_h // (HV // H)).to(tl.int64) * K
        k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
        w += (bos * HV + i_h).to(tl.int64) * K
        g += bos * HV + i_h
        do += (bos * HV + i_h).to(tl.int64) * V
        dv += (bos * HV + i_h).to(tl.int64) * V
        dv2 += (bos * HV + i_h).to(tl.int64) * V
        dh += (chunk_start * HV + i_h).to(tl.int64) * K * V
        if USE_DENSE_EXTERNAL_GRADIENT:
            dh_direct += (chunk_start * HV + i_h).to(tl.int64) * K * V
            dh_extra += (chunk_start * HV + i_h).to(tl.int64) * K * V
        if USE_COMPACT_BOUNDARY_GRADIENT:
            summary_start = tl.load(summary_cu_seqlens + i_b).to(tl.int32)
            compact_dh += (summary_start * HV + i_h).to(tl.int64) * K * V
        if USE_FINAL_STATE_GRADIENT:
            dht += i_bh * K * V
            dht_extra += i_bh * K * V

        if USE_FINAL_STATE_GRADIENT:
            p_dht1 = tl.make_block_ptr(dht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            p_dht_extra1 = tl.make_block_ptr(
                dht_extra, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
            )
            b_dh1 += tl.load(p_dht1, boundary_check=(0, 1))
            b_dh1 += tl.load(p_dht_extra1, boundary_check=(0, 1))
            if K > 64:
                p_dht2 = tl.make_block_ptr(dht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                p_dht_extra2 = tl.make_block_ptr(
                    dht_extra, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
                )
                b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
                b_dh2 += tl.load(p_dht_extra2, boundary_check=(0, 1))
            if K > 128:
                p_dht3 = tl.make_block_ptr(dht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                p_dht_extra3 = tl.make_block_ptr(
                    dht_extra, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
                )
                b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
                b_dh3 += tl.load(p_dht_extra3, boundary_check=(0, 1))
            if K > 192:
                p_dht4 = tl.make_block_ptr(dht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                p_dht_extra4 = tl.make_block_ptr(
                    dht_extra, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
                )
                b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))
                b_dh4 += tl.load(p_dht_extra4, boundary_check=(0, 1))

        for i_t in range(NT - 1, -1, -1):
            i_t_int64 = i_t.to(tl.int64)
            last_idx = min((i_t + 1) * BT, T) - 1

            if USE_DENSE_EXTERNAL_GRADIENT:
                p_direct1 = tl.make_block_ptr(
                    dh_direct + i_t_int64 * HV * K * V,
                    (K, V),
                    (V, 1),
                    (0, i_v * BV),
                    (64, BV),
                    (1, 0),
                )
                p_extra1 = tl.make_block_ptr(
                    dh_extra + i_t_int64 * HV * K * V,
                    (K, V),
                    (V, 1),
                    (0, i_v * BV),
                    (64, BV),
                    (1, 0),
                )
                b_direct1 = tl.load(p_direct1, boundary_check=(0, 1))
                b_extra1 = tl.load(p_extra1, boundary_check=(0, 1))
                if EXTERNAL_ORDER == 0:
                    b_dh1 = (b_dh1 + b_direct1) + b_extra1
                elif EXTERNAL_ORDER == 1:
                    b_dh1 = (b_dh1 + b_extra1) + b_direct1
                elif EXTERNAL_ORDER == 2:
                    b_dh1 = (b_direct1 + b_dh1) + b_extra1
                elif EXTERNAL_ORDER == 3:
                    b_dh1 = (b_extra1 + b_dh1) + b_direct1
                elif EXTERNAL_ORDER == 4:
                    b_dh1 = (b_direct1 + b_extra1) + b_dh1
                else:
                    b_dh1 = (b_extra1 + b_direct1) + b_dh1
                if K > 64:
                    p_direct2 = tl.make_block_ptr(
                        dh_direct + i_t_int64 * HV * K * V,
                        (K, V),
                        (V, 1),
                        (64, i_v * BV),
                        (64, BV),
                        (1, 0),
                    )
                    p_extra2 = tl.make_block_ptr(
                        dh_extra + i_t_int64 * HV * K * V,
                        (K, V),
                        (V, 1),
                        (64, i_v * BV),
                        (64, BV),
                        (1, 0),
                    )
                    b_direct2 = tl.load(p_direct2, boundary_check=(0, 1))
                    b_extra2 = tl.load(p_extra2, boundary_check=(0, 1))
                    if EXTERNAL_ORDER == 0:
                        b_dh2 = (b_dh2 + b_direct2) + b_extra2
                    elif EXTERNAL_ORDER == 1:
                        b_dh2 = (b_dh2 + b_extra2) + b_direct2
                    elif EXTERNAL_ORDER == 2:
                        b_dh2 = (b_direct2 + b_dh2) + b_extra2
                    elif EXTERNAL_ORDER == 3:
                        b_dh2 = (b_extra2 + b_dh2) + b_direct2
                    elif EXTERNAL_ORDER == 4:
                        b_dh2 = (b_direct2 + b_extra2) + b_dh2
                    else:
                        b_dh2 = (b_extra2 + b_direct2) + b_dh2
                if K > 128:
                    p_direct3 = tl.make_block_ptr(
                        dh_direct + i_t_int64 * HV * K * V,
                        (K, V),
                        (V, 1),
                        (128, i_v * BV),
                        (64, BV),
                        (1, 0),
                    )
                    p_extra3 = tl.make_block_ptr(
                        dh_extra + i_t_int64 * HV * K * V,
                        (K, V),
                        (V, 1),
                        (128, i_v * BV),
                        (64, BV),
                        (1, 0),
                    )
                    b_direct3 = tl.load(p_direct3, boundary_check=(0, 1))
                    b_extra3 = tl.load(p_extra3, boundary_check=(0, 1))
                    if EXTERNAL_ORDER == 0:
                        b_dh3 = (b_dh3 + b_direct3) + b_extra3
                    elif EXTERNAL_ORDER == 1:
                        b_dh3 = (b_dh3 + b_extra3) + b_direct3
                    elif EXTERNAL_ORDER == 2:
                        b_dh3 = (b_direct3 + b_dh3) + b_extra3
                    elif EXTERNAL_ORDER == 3:
                        b_dh3 = (b_extra3 + b_dh3) + b_direct3
                    elif EXTERNAL_ORDER == 4:
                        b_dh3 = (b_direct3 + b_extra3) + b_dh3
                    else:
                        b_dh3 = (b_extra3 + b_direct3) + b_dh3
                if K > 192:
                    p_direct4 = tl.make_block_ptr(
                        dh_direct + i_t_int64 * HV * K * V,
                        (K, V),
                        (V, 1),
                        (192, i_v * BV),
                        (64, BV),
                        (1, 0),
                    )
                    p_extra4 = tl.make_block_ptr(
                        dh_extra + i_t_int64 * HV * K * V,
                        (K, V),
                        (V, 1),
                        (192, i_v * BV),
                        (64, BV),
                        (1, 0),
                    )
                    b_direct4 = tl.load(p_direct4, boundary_check=(0, 1))
                    b_extra4 = tl.load(p_extra4, boundary_check=(0, 1))
                    if EXTERNAL_ORDER == 0:
                        b_dh4 = (b_dh4 + b_direct4) + b_extra4
                    elif EXTERNAL_ORDER == 1:
                        b_dh4 = (b_dh4 + b_extra4) + b_direct4
                    elif EXTERNAL_ORDER == 2:
                        b_dh4 = (b_direct4 + b_dh4) + b_extra4
                    elif EXTERNAL_ORDER == 3:
                        b_dh4 = (b_extra4 + b_dh4) + b_direct4
                    elif EXTERNAL_ORDER == 4:
                        b_dh4 = (b_direct4 + b_extra4) + b_dh4
                    else:
                        b_dh4 = (b_extra4 + b_direct4) + b_dh4

            if USE_COMPACT_BOUNDARY_GRADIENT:
                is_compact_boundary = (
                    ((i_t + 1) % COMPACT_CHUNK_STRIDE == 0)
                    & ((i_t + 1) * BT <= T)
                )
                if is_compact_boundary:
                    compact_idx = (i_t + 1) // COMPACT_CHUNK_STRIDE - 1
                    compact_offset = compact_idx.to(tl.int64) * HV * K * V
                    p_compact1 = tl.make_block_ptr(
                        compact_dh + compact_offset,
                        (K, V),
                        (V, 1),
                        (0, i_v * BV),
                        (64, BV),
                        (1, 0),
                    )
                    b_dh1 += tl.load(p_compact1, boundary_check=(0, 1))
                    if K > 64:
                        p_compact2 = tl.make_block_ptr(
                            compact_dh + compact_offset,
                            (K, V),
                            (V, 1),
                            (64, i_v * BV),
                            (64, BV),
                            (1, 0),
                        )
                        b_dh2 += tl.load(p_compact2, boundary_check=(0, 1))
                    if K > 128:
                        p_compact3 = tl.make_block_ptr(
                            compact_dh + compact_offset,
                            (K, V),
                            (V, 1),
                            (128, i_v * BV),
                            (64, BV),
                            (1, 0),
                        )
                        b_dh3 += tl.load(p_compact3, boundary_check=(0, 1))
                    if K > 192:
                        p_compact4 = tl.make_block_ptr(
                            compact_dh + compact_offset,
                            (K, V),
                            (V, 1),
                            (192, i_v * BV),
                            (64, BV),
                            (1, 0),
                        )
                        b_dh4 += tl.load(p_compact4, boundary_check=(0, 1))

            p_dh1 = tl.make_block_ptr(
                dh + i_t_int64 * HV * K * V,
                (K, V),
                (V, 1),
                (0, i_v * BV),
                (64, BV),
                (1, 0),
            )
            tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_dh2 = tl.make_block_ptr(
                    dh + i_t_int64 * HV * K * V,
                    (K, V),
                    (V, 1),
                    (64, i_v * BV),
                    (64, BV),
                    (1, 0),
                )
                tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_dh3 = tl.make_block_ptr(
                    dh + i_t_int64 * HV * K * V,
                    (K, V),
                    (V, 1),
                    (128, i_v * BV),
                    (64, BV),
                    (1, 0),
                )
                tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_dh4 = tl.make_block_ptr(
                    dh + i_t_int64 * HV * K * V,
                    (K, V),
                    (V, 1),
                    (192, i_v * BV),
                    (64, BV),
                    (1, 0),
                )
                tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

            bg_last = tl.load(g + last_idx * HV).to(tl.float32)
            p_g = tl.make_block_ptr(g, (T,), (HV,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            bg_last_exp = exp2(bg_last)
            b_g_exp = exp2(b_g)

            p_dv = tl.make_block_ptr(dv, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv2 = tl.make_block_ptr(dv2, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_do = tl.load(p_do, boundary_check=(0, 1))

            p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))
            if K > 64:
                p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))
            if K > 128:
                p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))
            if K > 192:
                p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
            b_dv += tl.load(p_dv, boundary_check=(0, 1))
            tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

            p_w = tl.make_block_ptr(w, (K, T), (1, HV * K), (0, i_t * BT), (64, BT), (0, 1))
            p_q = tl.make_block_ptr(q, (K, T), (1, H * K), (0, i_t * BT), (64, BT), (0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_dh1 *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]
            b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 64:
                p_q = tl.make_block_ptr(q, (K, T), (1, H * K), (64, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w, (K, T), (1, HV * K), (64, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_dh2 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
                b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(
                    b_w, b_dv.to(b_w.dtype)
                )
            if K > 128:
                p_q = tl.make_block_ptr(q, (K, T), (1, H * K), (128, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w, (K, T), (1, HV * K), (128, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_dh3 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
                b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(
                    b_w, b_dv.to(b_w.dtype)
                )
            if K > 192:
                p_q = tl.make_block_ptr(q, (K, T), (1, H * K), (192, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w, (K, T), (1, HV * K), (192, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_dh4 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
                b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(
                    b_w, b_dv.to(b_w.dtype)
                )


else:
    _hsd_recompute_v_new_from_chunk_states_kernel = None
    _hsd_summary_read_fwd_kernel = None
    _hsd_summary_read_bwd_q_kernel = None
    _hsd_compact_summary_read_fwd_kernel = None
    _hsd_summary_state_scatter_bwd_kernel = None
    _chunk_gated_delta_rule_bwd_kernel_dhu_external = None


def _recompute_v_new_from_saved_chunk_states(
    *,
    w: torch.Tensor,
    u: torch.Tensor,
    chunk_start_states: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
) -> torch.Tensor:
    """Rebuild the temporary WY value using FlashQLA's saved chunk starts.

    FLA's ``chunk_gated_delta_rule_fwd_h`` computes two logically separate
    results: the recurrent chunk-start states and
    ``v_new = u - w @ H_start``.  The split backward already has the first
    result from the FlashQLA producer, so only the chunk-local matrix product
    is needed here.
    """

    if triton is None or _hsd_recompute_v_new_from_chunk_states_kernel is None:
        raise RuntimeError("Triton v_new reconstruction kernel is unavailable.")
    if w.ndim != 4 or u.ndim != 4 or chunk_start_states.ndim != 5:
        raise RuntimeError("Saved-state v_new reconstruction received invalid ranks.")
    batch, text_len, heads, key_dim = w.shape
    value_dim = int(u.shape[-1])
    chunk_count = int(chunk_indices.shape[0])
    expected_u_shape = (batch, text_len, heads, value_dim)
    expected_h_shape = (batch, chunk_count, heads, key_dim, value_dim)
    if tuple(u.shape) != expected_u_shape:
        raise RuntimeError(
            f"Saved-state v_new u shape mismatch: expected={expected_u_shape}, actual={tuple(u.shape)}."
        )
    if tuple(chunk_start_states.shape) != expected_h_shape:
        raise RuntimeError(
            "Saved FlashQLA chunk-start shape mismatch: "
            f"expected={expected_h_shape}, actual={tuple(chunk_start_states.shape)}."
        )
    if batch != 1:
        raise RuntimeError("Saved-state v_new reconstruction currently requires packed batch=1.")
    if chunk_indices.ndim != 2 or int(chunk_indices.shape[1]) != 2:
        raise RuntimeError("Packed chunk indices must have shape [num_chunks, 2].")
    if chunk_start_states.dtype != w.dtype:
        raise RuntimeError(
            "Saved FlashQLA chunk starts must use the q/k dtype required by the local TileLang backward."
        )
    if not chunk_start_states.is_contiguous():
        raise RuntimeError("Saved FlashQLA chunk starts must be contiguous.")

    v_new = torch.empty_like(u)
    block_v = 32
    grid = (triton.cdiv(value_dim, block_v), chunk_count, heads)
    _hsd_recompute_v_new_from_chunk_states_kernel[grid](
        w=w,
        u=u,
        chunk_start_states=chunk_start_states,
        chunk_indices=chunk_indices,
        text_cu_seqlens=text_cu_seqlens,
        v_new=v_new,
        H=heads,
        K=key_dim,
        V=value_dim,
        BT=FLA_CHUNK_SIZE,
        BK=64,
        BV=block_v,
        num_warps=4,
        num_stages=2,
    )
    return v_new


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
    summary_cu_seqlens: torch.Tensor | None = None,
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
        if cu_seqlens is None or summary_cu_seqlens is None:
            raise RuntimeError("Compact HSD boundary gradients require packed text and summary cu_seqlens.")
        if batch != 1:
            raise RuntimeError("Compact HSD boundary-gradient injection currently requires batch=1.")
        if compact_chunk_stride <= 0:
            raise RuntimeError("compact_chunk_stride must be positive.")
        expected_compact_shape = (
            batch,
            int(summary_cu_seqlens[-1].item()),
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
        summary_cu_seqlens=summary_cu_seqlens,
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


def _run_summary_read_reference(
    *,
    query: torch.Tensor,
    text_o: torch.Tensor,
    summary_q: torch.Tensor,
    chunk_states_fp32: torch.Tensor,
    final_state: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor | None,
    summary_cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, text_len, heads, key_dim = query.shape
    value_dim = text_o.shape[-1]
    states = []
    summary_outputs = []
    packed = _is_packed_cu(text_cu_seqlens, summary_cu_seqlens)
    if packed:
        text_cu_cpu = text_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        summary_cu_cpu = summary_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        chunk_offsets_cpu = chunk_offsets.detach().to(device="cpu", dtype=torch.long)
        boundary_cpu = boundary_counts.detach().to(device="cpu", dtype=torch.long)
        zero_state = chunk_states_fp32.new_zeros(batch, heads, key_dim, value_dim)
        for doc_idx in range(int(text_cu_cpu.numel()) - 1):
            text_start = int(text_cu_cpu[doc_idx].item())
            text_end = int(text_cu_cpu[doc_idx + 1].item())
            doc_len = text_end - text_start
            summary_start = int(summary_cu_cpu[doc_idx].item())
            summary_end = int(summary_cu_cpu[doc_idx + 1].item())
            for summary_idx in range(summary_start, summary_end):
                local_boundary = int(boundary_cpu[summary_idx].item()) - text_start
                if local_boundary == 0 and doc_len == 0:
                    state = zero_state
                elif local_boundary == doc_len:
                    state = final_state[doc_idx:doc_idx + 1]
                else:
                    chunk_idx = _packed_chunk_index(
                        doc_idx=doc_idx,
                        local_boundary=local_boundary,
                        doc_len=doc_len,
                        chunk_offsets_cpu=chunk_offsets_cpu,
                    )
                    state = chunk_states_fp32[:, chunk_idx]
                states.append(state)
                summary_outputs.append(
                    torch.einsum(
                        "bhk,bhkv->bhv",
                        summary_q[:, summary_idx].to(state.dtype),
                        state,
                    )
                    * scale
                )
    else:
        for summary_idx, boundary in enumerate(boundary_counts.tolist()):
            boundary = int(boundary)
            if boundary == text_len:
                state = final_state
            else:
                chunk_idx = boundary // FLA_CHUNK_SIZE
                state = chunk_states_fp32[:, chunk_idx]
            states.append(state)
            summary_outputs.append(
                torch.einsum(
                    "bhk,bhkv->bhv",
                    summary_q[:, summary_idx].to(state.dtype),
                    state,
                )
                * scale
            )

    summary_o = torch.stack(summary_outputs, dim=1).to(text_o.dtype) if summary_outputs else query.new_zeros(
        batch, 0, heads, value_dim
    )
    summary_states = torch.stack(states, dim=1) if states else query.new_zeros(
        batch, 0, heads, key_dim, value_dim
    )
    return summary_o, summary_states


def _run_summary_read(
    *,
    query: torch.Tensor,
    text_o: torch.Tensor,
    summary_q: torch.Tensor,
    chunk_states_fp32: torch.Tensor,
    final_state: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor | None,
    summary_cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, text_len, heads, key_dim = query.shape
    summary_count = int(summary_q.shape[1])
    value_dim = text_o.shape[-1]
    if summary_count == 0:
        return (
            query.new_zeros(batch, 0, heads, value_dim),
            query.new_zeros(batch, 0, heads, key_dim, value_dim),
        )
    packed = _is_packed_cu(text_cu_seqlens, summary_cu_seqlens)
    use_fused = (
        _env_bool("HSD_FUSED_SUMMARY_READ", True)
        and triton is not None
        and _hsd_summary_read_fwd_kernel is not None
        and query.is_cuda
    )
    if not use_fused:
        return _run_summary_read_reference(
            query=query,
            text_o=text_o,
            summary_q=summary_q,
            chunk_states_fp32=chunk_states_fp32,
            final_state=final_state,
            boundary_counts=boundary_counts,
            text_cu_seqlens=text_cu_seqlens,
            summary_cu_seqlens=summary_cu_seqlens,
            chunk_offsets=chunk_offsets,
            scale=scale,
        )

    if packed:
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(text_cu_seqlens, FLA_CHUNK_SIZE)
        text_cu = text_cu_seqlens
        summary_cu = summary_cu_seqlens
        offsets = chunk_offsets
        n_docs = int(text_cu_seqlens.numel()) - 1
    else:
        text_cu = boundary_counts.new_empty((0,), dtype=torch.long)
        summary_cu = boundary_counts.new_empty((0,), dtype=torch.long)
        offsets = boundary_counts.new_empty((0,), dtype=torch.long)
        n_docs = batch

    chunk_count = int(chunk_states_fp32.shape[1])
    summary_o = torch.empty(
        batch,
        summary_count,
        heads,
        value_dim,
        device=query.device,
        dtype=text_o.dtype,
    )
    summary_states = torch.empty(
        batch,
        summary_count,
        heads,
        key_dim,
        value_dim,
        device=query.device,
        dtype=torch.float32,
    )

    def grid(_meta):
        return (triton.cdiv(value_dim, 32), summary_count, batch * heads)

    _hsd_summary_read_fwd_kernel[grid](
        summary_q=summary_q,
        chunk_states=chunk_states_fp32,
        final_state=final_state,
        boundary_counts=boundary_counts,
        text_cu_seqlens=text_cu,
        summary_cu_seqlens=summary_cu,
        chunk_offsets=offsets,
        summary_o=summary_o,
        summary_states=summary_states,
        scale=float(scale),
        TEXT_LEN=text_len,
        SUMMARY_COUNT=summary_count,
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
    return summary_o, summary_states


def _scatter_summary_state_grads_fused(
    *,
    summary_q: torch.Tensor,
    summary_states: torch.Tensor,
    boundary_counts: torch.Tensor,
    dsummary_o: torch.Tensor | None,
    dsummary_states: torch.Tensor | None,
    text_len: int,
    chunk_count: int,
    scale: float,
    text_cu_seqlens: torch.Tensor | None = None,
    summary_cu_seqlens: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not (
        _env_bool("HSD_FUSED_SUMMARY_READ", True)
        and triton is not None
        and _hsd_summary_read_bwd_q_kernel is not None
        and _hsd_summary_state_scatter_bwd_kernel is not None
        and summary_q.is_cuda
    ):
        return None

    batch, summary_count, heads, key_dim = summary_q.shape
    if summary_count == 0:
        return None
    value_dim = summary_states.shape[-1]
    state_dtype = summary_states.dtype
    packed = _is_packed_cu(text_cu_seqlens, summary_cu_seqlens)
    if packed:
        if chunk_offsets is None:
            if prepare_chunk_offsets is None:
                return None
            chunk_offsets = prepare_chunk_offsets(text_cu_seqlens, FLA_CHUNK_SIZE)
        text_cu = text_cu_seqlens
        summary_cu = summary_cu_seqlens
        offsets = chunk_offsets
        sequence_count = int(text_cu_seqlens.numel()) - 1
    else:
        text_cu = boundary_counts.new_empty((0,), dtype=torch.long)
        summary_cu = boundary_counts.new_empty((0,), dtype=torch.long)
        offsets = boundary_counts.new_empty((0,), dtype=torch.long)
        sequence_count = batch

    if dsummary_o is None:
        dsummary_o_f = summary_states.new_zeros(batch, summary_count, heads, value_dim)
    else:
        # Autograd may supply expanded zero-stride gradients (for example
        # through ``output.sum()``).  The fused kernels consume dense
        # row-major tensors, so normalize the layout before passing pointers.
        dsummary_o_f = dsummary_o.to(state_dtype).contiguous()
    if dsummary_states is None:
        dsummary_states_direct = torch.zeros_like(summary_states)
    else:
        dsummary_states_direct = dsummary_states.to(state_dtype).contiguous()

    dsummary_q = torch.empty_like(summary_q)
    dh_direct = summary_states.new_zeros(batch, chunk_count, heads, key_dim, value_dim)
    dh_extra = summary_states.new_zeros(batch, chunk_count, heads, key_dim, value_dim)
    dht_direct = summary_states.new_zeros(sequence_count, heads, key_dim, value_dim)
    dht_extra = summary_states.new_zeros(sequence_count, heads, key_dim, value_dim)

    def q_grid(_meta):
        return (triton.cdiv(key_dim, 64), summary_count, batch * heads)

    _hsd_summary_read_bwd_q_kernel[q_grid](
        dsummary_o=dsummary_o_f,
        summary_states=summary_states,
        dsummary_q=dsummary_q,
        scale=float(scale),
        SUMMARY_COUNT=summary_count,
        H=heads,
        K=key_dim,
        V=value_dim,
        BK=64,
        BV=32,
    )

    def scatter_grid(_meta):
        return (triton.cdiv(value_dim, 32), triton.cdiv(key_dim, 64), summary_count * batch * heads)

    _hsd_summary_state_scatter_bwd_kernel[scatter_grid](
        summary_q=summary_q,
        dsummary_o=dsummary_o_f,
        dsummary_states=dsummary_states_direct,
        boundary_counts=boundary_counts,
        text_cu_seqlens=text_cu,
        summary_cu_seqlens=summary_cu,
        chunk_offsets=offsets,
        dh_direct=dh_direct,
        dh_extra=dh_extra,
        dht_direct=dht_direct,
        dht_extra=dht_extra,
        scale=float(scale),
        TEXT_LEN=text_len,
        SUMMARY_COUNT=summary_count,
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
    return dsummary_q, dh_direct, dht_direct, dh_extra, dht_extra


def _run_full_scan_forward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    summary_q: torch.Tensor,
    boundary_counts: torch.Tensor,
    scale: float,
    text_cu_seqlens: torch.Tensor | None = None,
    summary_cu_seqlens: torch.Tensor | None = None,
    return_intermediates: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch, text_len, heads, key_dim = query.shape
    packed = _is_packed_cu(text_cu_seqlens, summary_cu_seqlens)
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

    with _profile_cuda_ms("chunk/fwd_summary_ms"):
        summary_o, summary_states = _run_summary_read(
            query=query,
            text_o=text_o,
            summary_q=summary_q,
            chunk_states_fp32=chunk_states_fp32,
            final_state=final_state,
            boundary_counts=boundary_counts,
            text_cu_seqlens=text_cu_seqlens if packed else None,
            summary_cu_seqlens=summary_cu_seqlens if packed else None,
            chunk_offsets=chunk_offsets if packed else None,
            scale=scale,
        )
    if return_intermediates:
        return text_o, summary_o, summary_states, g_cumsum, A
    return text_o, summary_o, summary_states


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


def _scatter_summary_state_grads(
    *,
    summary_q: torch.Tensor,
    summary_states: torch.Tensor,
    boundary_counts: torch.Tensor,
    dsummary_o: torch.Tensor | None,
    dsummary_states: torch.Tensor | None,
    text_len: int,
    chunk_count: int,
    scale: float,
    text_cu_seqlens: torch.Tensor | None = None,
    summary_cu_seqlens: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    batch, summary_count, heads, key_dim = summary_q.shape
    value_dim = summary_states.shape[-1]
    fused = _scatter_summary_state_grads_fused(
        summary_q=summary_q,
        summary_states=summary_states,
        boundary_counts=boundary_counts,
        dsummary_o=dsummary_o,
        dsummary_states=dsummary_states,
        text_len=text_len,
        chunk_count=chunk_count,
        scale=scale,
        text_cu_seqlens=text_cu_seqlens,
        summary_cu_seqlens=summary_cu_seqlens,
        chunk_offsets=chunk_offsets,
    )
    if fused is not None:
        return fused

    state_dtype = summary_states.dtype
    if dsummary_o is None:
        dsummary_o_f = summary_states.new_zeros(batch, summary_count, heads, value_dim)
    else:
        dsummary_o_f = dsummary_o.to(state_dtype)
    dsummary_o_scaled = dsummary_o_f * scale
    dsummary_q = torch.einsum("bmhv,bmhkv->bmhk", dsummary_o_scaled, summary_states)
    dsummary_states_extra = torch.einsum("bmhk,bmhv->bmhkv", summary_q.to(state_dtype), dsummary_o_scaled)
    dsummary_states_direct = torch.zeros_like(summary_states) if dsummary_states is None else dsummary_states.to(
        state_dtype
    )
    dh_direct = summary_states.new_zeros(batch, chunk_count, heads, key_dim, value_dim)
    dh_extra = summary_states.new_zeros(batch, chunk_count, heads, key_dim, value_dim)
    dht_direct = None
    dht_extra = None
    if _is_packed_cu(text_cu_seqlens, summary_cu_seqlens):
        text_cu_cpu = text_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        summary_cu_cpu = summary_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        if chunk_offsets is None:
            if prepare_chunk_offsets is None:
                raise RuntimeError("FLA chunk offset helper is unavailable.")
            chunk_offsets = prepare_chunk_offsets(text_cu_seqlens, FLA_CHUNK_SIZE)
        chunk_offsets_cpu = chunk_offsets.detach().to(device="cpu", dtype=torch.long)
        boundary_cpu = boundary_counts.detach().to(device="cpu", dtype=torch.long)
        sequence_count = int(text_cu_cpu.numel()) - 1
        dht_direct = summary_states.new_zeros(sequence_count, heads, key_dim, value_dim)
        dht_extra = summary_states.new_zeros(sequence_count, heads, key_dim, value_dim)
        for doc_idx in range(sequence_count):
            text_start = int(text_cu_cpu[doc_idx].item())
            text_end = int(text_cu_cpu[doc_idx + 1].item())
            doc_len = text_end - text_start
            summary_start = int(summary_cu_cpu[doc_idx].item())
            summary_end = int(summary_cu_cpu[doc_idx + 1].item())
            for summary_idx in range(summary_start, summary_end):
                local_boundary = int(boundary_cpu[summary_idx].item()) - text_start
                if local_boundary <= 0:
                    continue
                if local_boundary == doc_len:
                    dht_direct[doc_idx] = dht_direct[doc_idx] + dsummary_states_direct[0, summary_idx]
                    dht_extra[doc_idx] = dht_extra[doc_idx] + dsummary_states_extra[0, summary_idx]
                    continue
                chunk_idx = int(chunk_offsets_cpu[doc_idx].item()) + local_boundary // FLA_CHUNK_SIZE
                if chunk_idx <= int(chunk_offsets_cpu[doc_idx].item()):
                    continue
                dh_direct[:, chunk_idx - 1] = dh_direct[:, chunk_idx - 1] + dsummary_states_direct[:, summary_idx]
                dh_extra[:, chunk_idx - 1] = dh_extra[:, chunk_idx - 1] + dsummary_states_extra[:, summary_idx]
        if not bool(dht_direct.abs().any().item()) and not bool(dht_extra.abs().any().item()):
            dht_direct = None
            dht_extra = None
    else:
        for summary_idx, boundary in enumerate(boundary_counts.tolist()):
            boundary = int(boundary)
            if boundary <= 0:
                continue
            if boundary == text_len:
                if dht_direct is None:
                    dht_direct = summary_states.new_zeros(batch, heads, key_dim, value_dim)
                    dht_extra = summary_states.new_zeros(batch, heads, key_dim, value_dim)
                dht_direct = dht_direct + dsummary_states_direct[:, summary_idx]
                dht_extra = dht_extra + dsummary_states_extra[:, summary_idx]
                continue
            chunk_idx = boundary // FLA_CHUNK_SIZE
            if chunk_idx <= 0:
                continue
            dh_direct[:, chunk_idx - 1] = dh_direct[:, chunk_idx - 1] + dsummary_states_direct[:, summary_idx]
            dh_extra[:, chunk_idx - 1] = dh_extra[:, chunk_idx - 1] + dsummary_states_extra[:, summary_idx]

    return dsummary_q.to(summary_q.dtype), dh_direct, dht_direct, dh_extra, dht_extra


def _run_reference_backward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    summary_q: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor | None,
    summary_cu_seqlens: torch.Tensor | None,
    dtext_o: torch.Tensor | None,
    dsummary_o: torch.Tensor | None,
    dsummary_states: torch.Tensor | None,
    scale: float,
    needs_grad,
) -> list[torch.Tensor | None]:
    query_ref = query.detach().requires_grad_(needs_grad[0])
    key_ref = key.detach().requires_grad_(needs_grad[1])
    value_ref = value.detach().requires_grad_(needs_grad[2])
    beta_ref = beta.detach().requires_grad_(needs_grad[3])
    g_ref = g.detach().requires_grad_(needs_grad[4])
    summary_q_ref = summary_q.detach().requires_grad_(needs_grad[5])

    with torch.enable_grad():
        if _is_packed_cu(text_cu_seqlens, summary_cu_seqlens):
            text_o, summary_o, summary_states = _run_segmented_packed_reference(
                query=query_ref,
                key=key_ref,
                value=value_ref,
                beta=beta_ref,
                g=g_ref,
                summary_q=summary_q_ref,
                boundary_counts=boundary_counts,
                text_cu_seqlens=text_cu_seqlens,
                summary_cu_seqlens=summary_cu_seqlens,
                scale=scale,
            )
        else:
            text_o, summary_o, summary_states = _run_segmented_reference(
                query=query_ref,
                key=key_ref,
                value=value_ref,
                beta=beta_ref,
                g=g_ref,
                summary_q=summary_q_ref,
                boundary_counts=boundary_counts,
                scale=scale,
            )
        if dtext_o is None:
            dtext_o = torch.zeros_like(text_o)
        if dsummary_o is None:
            dsummary_o = torch.zeros_like(summary_o)
        if dsummary_states is None:
            dsummary_states = torch.zeros_like(summary_states)

        grad_inputs = []
        input_refs = [query_ref, key_ref, value_ref, beta_ref, g_ref, summary_q_ref]
        active_refs = [tensor for tensor, need in zip(input_refs, needs_grad[:6]) if need]
        if active_refs:
            active_grads = torch.autograd.grad(
                outputs=(text_o, summary_o, summary_states),
                inputs=active_refs,
                grad_outputs=(dtext_o, dsummary_o, dsummary_states),
                allow_unused=True,
                retain_graph=False,
                create_graph=False,
            )
            active_iter = iter(active_grads)
        else:
            active_iter = iter(())
        for need in needs_grad[:6]:
            grad_inputs.append(next(active_iter) if need else None)
    return grad_inputs


def _run_optimized_backward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    summary_q: torch.Tensor,
    summary_states: torch.Tensor,
    boundary_counts: torch.Tensor,
    dtext_o: torch.Tensor | None,
    dsummary_o: torch.Tensor | None,
    dsummary_states: torch.Tensor | None,
    scale: float,
    needs_grad,
    text_cu_seqlens: torch.Tensor | None = None,
    summary_cu_seqlens: torch.Tensor | None = None,
) -> list[torch.Tensor | None]:
    batch, text_len, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    packed = _is_packed_cu(text_cu_seqlens, summary_cu_seqlens)
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
        with _profile_cuda_ms("chunk/bwd_scatter_summary_ms"):
            dsummary_q, dh_direct, dht_direct, dh_extra, dht_extra = _scatter_summary_state_grads(
                summary_q=summary_q,
                summary_states=summary_states,
                boundary_counts=boundary_counts,
                dsummary_o=dsummary_o,
                dsummary_states=dsummary_states,
                text_len=text_len,
                chunk_count=chunk_count,
                scale=scale,
                text_cu_seqlens=text_cu_seqlens if packed else None,
                summary_cu_seqlens=summary_cu_seqlens if packed else None,
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
        dsummary_q.to(summary_q.dtype) if needs_grad[5] else None,
    ]
    return grads


def _prepare_flashqla_summary_q_grad(
    *,
    summary_q: torch.Tensor,
    summary_states: torch.Tensor,
    dsummary_o: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """Compute only d(summary_q), leaving boundary dH unmaterialized."""

    batch, summary_count, heads, key_dim = summary_q.shape
    value_dim = int(summary_states.shape[-1])
    expected_state_shape = (batch, summary_count, heads, key_dim, value_dim)
    if tuple(summary_states.shape) != expected_state_shape:
        raise RuntimeError(
            "FlashQLA compact summary-state shape mismatch in backward: "
            f"expected={expected_state_shape}, actual={tuple(summary_states.shape)}."
        )
    if summary_states.dtype != torch.float32 or not summary_states.is_contiguous():
        raise RuntimeError("FlashQLA compact summary states must be contiguous FP32 tensors.")
    if summary_count == 0 or dsummary_o is None:
        return torch.zeros_like(summary_q)
    expected_output_shape = (batch, summary_count, heads, value_dim)
    if tuple(dsummary_o.shape) != expected_output_shape:
        raise RuntimeError(
            "FlashQLA compact summary-output gradient shape mismatch: "
            f"expected={expected_output_shape}, actual={tuple(dsummary_o.shape)}."
        )

    flattened_count = batch * summary_count * heads
    if (
        triton is not None
        and _hsd_summary_read_bwd_q_kernel is not None
        and summary_q.is_cuda
    ):
        dsummary_q = torch.empty_like(summary_q)

        def q_grid(_meta):
            return (
                triton.cdiv(key_dim, 64),
                summary_count,
                batch * heads,
            )

        _hsd_summary_read_bwd_q_kernel[q_grid](
            dsummary_o=dsummary_o.contiguous(),
            summary_states=summary_states,
            dsummary_q=dsummary_q,
            scale=float(scale),
            SUMMARY_COUNT=summary_count,
            H=heads,
            K=key_dim,
            V=value_dim,
            BK=64,
            BV=32,
        )
        return dsummary_q

    dsummary_q_fp32 = torch.empty(
        (flattened_count, key_dim, 1),
        device=summary_states.device,
        dtype=torch.float32,
    )
    torch.bmm(
        summary_states.view(flattened_count, key_dim, value_dim),
        dsummary_o.to(torch.float32).contiguous().view(
            flattened_count,
            value_dim,
            1,
        ),
        out=dsummary_q_fp32,
    )
    dsummary_q_fp32.mul_(float(scale))
    return dsummary_q_fp32.view(
        batch,
        summary_count,
        heads,
        key_dim,
    ).to(summary_q.dtype)


def _prepare_flashqla_compact_summary_grads(
    *,
    summary_q: torch.Tensor,
    summary_states: torch.Tensor,
    dsummary_o: torch.Tensor | None,
    dsummary_states: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build one total FP32 dH per compact FlashQLA boundary.

    The rank-one contribution from the summary read is accumulated directly
    into the compact output.  In particular, this never materializes the two
    dense ``[num_chunk64, H, K, V]`` scatter buffers used by the general HSD
    backward.
    """

    batch, summary_count, heads, key_dim = summary_q.shape
    value_dim = int(summary_states.shape[-1])
    expected_state_shape = (batch, summary_count, heads, key_dim, value_dim)
    if tuple(summary_states.shape) != expected_state_shape:
        raise RuntimeError(
            "FlashQLA compact summary-state shape mismatch in backward: "
            f"expected={expected_state_shape}, actual={tuple(summary_states.shape)}."
        )
    if summary_states.dtype != torch.float32 or not summary_states.is_contiguous():
        raise RuntimeError("FlashQLA compact summary states must be contiguous FP32 tensors.")
    if summary_count == 0:
        return torch.zeros_like(summary_q), None

    flattened_count = batch * summary_count * heads
    states_flat = summary_states.view(flattened_count, key_dim, value_dim)
    dsummary_o_fp32 = None
    if dsummary_o is None:
        dsummary_q = torch.zeros_like(summary_q)
    else:
        expected_output_shape = (batch, summary_count, heads, value_dim)
        if tuple(dsummary_o.shape) != expected_output_shape:
            raise RuntimeError(
                "FlashQLA compact summary-output gradient shape mismatch: "
                f"expected={expected_output_shape}, actual={tuple(dsummary_o.shape)}."
            )
        dsummary_o_fp32 = dsummary_o.to(torch.float32).contiguous().view(
            flattened_count,
            value_dim,
        )
        dsummary_q_fp32 = torch.empty(
            (flattened_count, key_dim, 1),
            device=summary_states.device,
            dtype=torch.float32,
        )
        torch.bmm(
            states_flat,
            dsummary_o_fp32.unsqueeze(-1),
            out=dsummary_q_fp32,
        )
        dsummary_q_fp32.mul_(float(scale))
        dsummary_q = dsummary_q_fp32.view(
            batch,
            summary_count,
            heads,
            key_dim,
        ).to(summary_q.dtype)

    compact_dh = None
    if dsummary_states is not None:
        if tuple(dsummary_states.shape) != expected_state_shape:
            raise RuntimeError(
                "FlashQLA compact summary-state gradient shape mismatch: "
                f"expected={expected_state_shape}, actual={tuple(dsummary_states.shape)}."
            )
        compact_dh = torch.empty_like(summary_states)
        compact_dh.copy_(dsummary_states)

    if dsummary_o_fp32 is not None:
        summary_q_fp32 = summary_q.to(torch.float32).contiguous().view(
            flattened_count,
            key_dim,
        )
        q_column = summary_q_fp32.unsqueeze(-1)
        do_row = dsummary_o_fp32.unsqueeze(1)
        if compact_dh is None:
            compact_dh = torch.empty_like(summary_states)
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
    return dsummary_q, compact_dh


def _run_flashqla_compact_optimized_backward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    summary_q: torch.Tensor,
    summary_states: torch.Tensor,
    dtext_o: torch.Tensor | None,
    dsummary_o: torch.Tensor | None,
    dsummary_states: torch.Tensor | None,
    scale: float,
    needs_grad,
    text_cu_seqlens: torch.Tensor,
    summary_cu_seqlens: torch.Tensor,
) -> list[torch.Tensor | None]:
    """Optimized FlashQLA backward with compact chunk-512 dH injection."""

    if not _is_packed_cu(text_cu_seqlens, summary_cu_seqlens):
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
        with _profile_cuda_ms("chunk/flashqla_tile_bwd_compact_summary_ms"):
            dsummary_q, compact_dh = _prepare_flashqla_compact_summary_grads(
                summary_q=summary_q,
                summary_states=summary_states,
                dsummary_o=dsummary_o,
                dsummary_states=dsummary_states,
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
                summary_cu_seqlens=summary_cu_seqlens,
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
        dsummary_q.to(summary_q.dtype) if needs_grad[5] else None,
    ]


def _run_flashqla_compact_native_backward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    flash_g_cumsum: torch.Tensor,
    flash_A: torch.Tensor,
    summary_q: torch.Tensor,
    summary_states: torch.Tensor,
    dtext_o: torch.Tensor | None,
    dsummary_o: torch.Tensor | None,
    dsummary_states: torch.Tensor | None,
    scale: float,
    needs_grad,
    text_cu_seqlens: torch.Tensor,
    summary_cu_seqlens: torch.Tensor,
) -> list[torch.Tensor | None]:
    """FlashQLA-style reverse scan with compact chunk-512 dH injection.

    Unlike the compatibility path above, this consumes the exact cumulative
    gate and ungated KKT inverse emitted by the FlashQLA forward.  It therefore
    avoids rebuilding the FLA cumsum/KKT/WY representation and replaces the
    old multi-kernel FLA reverse path with the native Hopper producer backward.
    """

    if not _is_packed_cu(text_cu_seqlens, summary_cu_seqlens):
        raise RuntimeError("FlashQLA native compact backward requires packed cu_seqlens.")
    from ._flashqla.ops.gated_delta_rule.chunk.hopper import (
        fused_gdr_bwd_compact,
        fused_gdr_fwd,
    )
    from ._flashqla.ops.utils import chunk_local_cumsum as flashqla_chunk_local_cumsum

    batch, text_len, heads, _ = query.shape
    value_dim = int(value.shape[-1])
    if dtext_o is None:
        dtext_o = value.new_zeros(batch, text_len, heads, value_dim)
    else:
        dtext_o = dtext_o.contiguous()

    with torch.no_grad():
        with _profile_cuda_ms("chunk/flashqla_tile_native_bwd_compact_summary_ms"):
            dsummary_q, compact_dh = _prepare_flashqla_compact_summary_grads(
                summary_q=summary_q,
                summary_states=summary_states,
                dsummary_o=dsummary_o,
                dsummary_states=dsummary_states,
                scale=scale,
            )
        with _profile_cuda_ms("chunk/flashqla_tile_native_bwd_prepare_h_ms"):
            _, chunk_start_states, _ = fused_gdr_fwd(
                q=query,
                k=key,
                v=value,
                a=flash_A,
                g=flash_g_cumsum,
                b=beta,
                scale=scale,
                initial_state=None,
                output_final_state=False,
                output_h=True,
                output_h_stride=1,
                output_o=False,
                cu_seqlens=text_cu_seqlens,
                state_v_first=False,
            )
        if chunk_start_states is None:
            raise RuntimeError("FlashQLA native backward failed to reconstruct chunk states.")
        with _profile_cuda_ms("chunk/flashqla_tile_native_bwd_reverse_ms"):
            dq, dk, dv, dg, db, _ = fused_gdr_bwd_compact(
                q=query,
                k=key,
                v=value,
                a=flash_A,
                g=flash_g_cumsum,
                b=beta,
                do=dtext_o,
                compact_dh=compact_dh,
                h=chunk_start_states,
                summary_cu_seqlens=summary_cu_seqlens,
                scale=scale,
                cu_seqlens=text_cu_seqlens,
                chunk_size=FLA_CHUNK_SIZE,
                compact_chunk_stride=8,
                value_tile_size=64,
                state_v_first=False,
            )
        with _profile_cuda_ms("chunk/flashqla_tile_native_bwd_cumsum_reverse_ms"):
            dg = flashqla_chunk_local_cumsum(
                dg,
                chunk_size=FLA_CHUNK_SIZE,
                reverse=True,
                cu_seqlens=text_cu_seqlens,
            )

    return [
        dq.to(query.dtype) if needs_grad[0] else None,
        dk.to(key.dtype) if needs_grad[1] else None,
        dv.to(value.dtype) if needs_grad[2] else None,
        db.to(beta.dtype) if needs_grad[3] else None,
        dg.to(flash_g_cumsum.dtype) if needs_grad[4] else None,
        dsummary_q.to(summary_q.dtype) if needs_grad[5] else None,
    ]


def _run_flashqla_compact_direct_a_backward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    flash_g_cumsum: torch.Tensor,
    flash_A: torch.Tensor,
    summary_q: torch.Tensor,
    summary_states: torch.Tensor,
    chunk_start_states: torch.Tensor,
    chunk_indices: torch.Tensor,
    dtext_o: torch.Tensor | None,
    dsummary_o: torch.Tensor | None,
    dsummary_states: torch.Tensor | None,
    scale: float,
    needs_grad,
    text_cu_seqlens: torch.Tensor,
    summary_cu_seqlens: torch.Tensor,
) -> list[torch.Tensor | None]:
    """Run compact state-adjoint scan plus the direct-A local Kernel B.

    This is the target producer backward: the local kernel consumes the
    FlashQLA forward's KKT inverse and chunk-start states directly, forms
    ``dD`` on chip, loops over all six V64 tiles in one chunk/head CTA, and
    returns the raw gate gradient.  It therefore has no WY reconstruction,
    materialized transformed-value gradient, six-way partial reduction, or
    external reverse-cumsum stage.
    """

    if not _is_packed_cu(text_cu_seqlens, summary_cu_seqlens):
        raise RuntimeError("FlashQLA direct-A backward requires packed cu_seqlens.")
    if query.dtype != torch.bfloat16:
        raise RuntimeError(
            "The first FlashQLA direct-A backward is specialized for BF16."
        )
    from ._flashqla.ops.gated_delta_rule.chunk.hopper.cp_bwd import (
        fused_gdr_dh_ws_compact,
    )
    from ._flashqla.ops.gated_delta_rule.chunk.hopper.direct_a_chunk_bwd_inline_dd import (
        direct_a_chunk_bwd_inline_dd_prototype,
    )

    batch, text_len, heads, _ = query.shape
    value_dim = int(value.shape[-1])
    if dtext_o is None:
        dtext_o = value.new_zeros(batch, text_len, heads, value_dim)
    else:
        dtext_o = dtext_o.contiguous()

    trace_direct_a = os.environ.get(
        "BENCHMARK_HSD_TRACE_DIRECT_A",
        "",
    ).strip().lower() in {"1", "true", "yes", "on"}
    trace_pre_scan = trace_direct_a and os.environ.get(
        "BENCHMARK_HSD_TRACE_DIRECT_A_PRE_SCAN",
        "",
    ).strip().lower() in {"1", "true", "yes", "on"}
    compare_state_scan_buffer = trace_pre_scan and os.environ.get(
        "BENCHMARK_HSD_TRACE_DIRECT_A_BUFFER_DIFF",
        "",
    ).strip().lower() in {"1", "true", "yes", "on"}

    with torch.no_grad():
        with _profile_cuda_ms("chunk/flashqla_tile_direct_a_bwd_compact_summary_ms"):
            dsummary_q = _prepare_flashqla_summary_q_grad(
                summary_q=summary_q,
                summary_states=summary_states,
                dsummary_o=dsummary_o,
                scale=scale,
            )
        chunk_start_states_before_scan = None
        if trace_pre_scan:
            pre_scan_nonfinite = int(
                (~torch.isfinite(chunk_start_states)).sum().item()
            )
            if pre_scan_nonfinite:
                bad_indices = (
                    (~torch.isfinite(chunk_start_states))
                    .nonzero(as_tuple=False)[:8]
                    .detach()
                    .cpu()
                    .tolist()
                )
                raise RuntimeError(
                    "FlashQLA direct-A chunk-start buffer was already "
                    "non-finite before the state-adjoint scan: "
                    f"nonfinite={pre_scan_nonfinite}, "
                    f"first_nonfinite_indices={bad_indices}"
                )
            if compare_state_scan_buffer:
                chunk_start_states_before_scan = (
                    chunk_start_states.view(torch.int16).clone()
                )
        with _profile_cuda_ms("chunk/flashqla_tile_direct_a_bwd_state_scan_ms"):
            chunk_end_adjoints, _ = fused_gdr_dh_ws_compact(
                q=query,
                k=key,
                a=flash_A,
                g=flash_g_cumsum,
                b=beta,
                do=dtext_o,
                compact_dh=None,
                summary_cu_seqlens=summary_cu_seqlens,
                summary_q=summary_q,
                dsummary_o=dsummary_o,
                dsummary_states=dsummary_states,
                summary_state_placeholder=summary_states,
                scale=scale,
                cu_seqlens=text_cu_seqlens,
                compact_chunk_stride=8,
                value_tile_size=64,
                state_v_first=False,
            )
        if chunk_start_states_before_scan is not None:
            changed = (
                chunk_start_states.view(torch.int16)
                != chunk_start_states_before_scan
            )
            changed_count = int(changed.sum().item())
            if changed_count:
                changed_indices = (
                    changed.nonzero(as_tuple=False)[:8]
                    .detach()
                    .cpu()
                    .tolist()
                )
                raise RuntimeError(
                    "FlashQLA state-adjoint scan modified the producer-only "
                    "chunk-start buffer: "
                    f"changed={changed_count}, "
                    f"first_changed_indices={changed_indices}"
                )
        if trace_direct_a:
            traced_inputs = {
                "query": query,
                "key": key,
                "value": value,
                "flash_A": flash_A,
                "flash_g_cumsum": flash_g_cumsum,
                "beta": beta,
                "dtext_o": dtext_o,
                "summary_states": summary_states,
                "chunk_start_states": chunk_start_states,
                "chunk_end_adjoints": chunk_end_adjoints,
            }
            if dsummary_o is not None:
                traced_inputs["dsummary_o"] = dsummary_o
            if dsummary_states is not None:
                traced_inputs["dsummary_states"] = dsummary_states
            nonfinite = {
                name: int((~torch.isfinite(tensor)).sum().item())
                for name, tensor in traced_inputs.items()
                if not bool(torch.isfinite(tensor).all().item())
            }
            if nonfinite:
                finite_max_abs = {}
                first_nonfinite_indices = {}
                for name, tensor in traced_inputs.items():
                    finite_values = tensor.detach()[torch.isfinite(tensor)]
                    finite_max_abs[name] = (
                        float(finite_values.abs().max().item())
                        if finite_values.numel()
                        else float("nan")
                    )
                    bad_indices = (~torch.isfinite(tensor)).nonzero(
                        as_tuple=False
                    )
                    if bad_indices.numel():
                        first_nonfinite_indices[name] = (
                            bad_indices[:8].detach().cpu().tolist()
                        )
                raise RuntimeError(
                    "FlashQLA direct-A received non-finite intermediates: "
                    f"nonfinite={nonfinite}, "
                    f"first_nonfinite_indices={first_nonfinite_indices}, "
                    f"finite_max_abs={finite_max_abs}"
                )
        with _profile_cuda_ms("chunk/flashqla_tile_direct_a_bwd_local_ms"):
            dq, dk, dv, db, dg = direct_a_chunk_bwd_inline_dd_prototype(
                q=query,
                k=key,
                v=value,
                a=flash_A,
                g_cumsum=flash_g_cumsum,
                beta=beta,
                do=dtext_o,
                chunk_start_h=chunk_start_states,
                chunk_end_dh=chunk_end_adjoints,
                cu_seqlens=text_cu_seqlens,
                chunk_indices=chunk_indices,
                scale=scale,
            )
        if trace_direct_a:
            traced_outputs = {
                "dq": dq,
                "dk": dk,
                "dv": dv,
                "db": db,
                "dg": dg,
            }
            nonfinite = {
                name: int((~torch.isfinite(tensor)).sum().item())
                for name, tensor in traced_outputs.items()
                if not bool(torch.isfinite(tensor).all().item())
            }
            if nonfinite:
                first_nonfinite_indices = {
                    name: (
                        (~torch.isfinite(tensor))
                        .nonzero(as_tuple=False)[:8]
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    for name, tensor in traced_outputs.items()
                    if not bool(torch.isfinite(tensor).all().item())
                }
                raise RuntimeError(
                    "FlashQLA direct-A produced non-finite outputs: "
                    f"nonfinite={nonfinite}, "
                    f"first_nonfinite_indices={first_nonfinite_indices}"
                )

    return [
        dq.to(query.dtype) if needs_grad[0] else None,
        dk.to(key.dtype) if needs_grad[1] else None,
        dv.to(value.dtype) if needs_grad[2] else None,
        db.to(beta.dtype) if needs_grad[3] else None,
        dg.to(flash_g_cumsum.dtype) if needs_grad[4] else None,
        dsummary_q.to(summary_q.dtype) if needs_grad[5] else None,
    ]


def _run_flashqla_compact_split_backward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    flash_g_cumsum: torch.Tensor,
    flash_A: torch.Tensor,
    summary_q: torch.Tensor,
    summary_states: torch.Tensor,
    chunk_start_states: torch.Tensor,
    chunk_indices: torch.Tensor,
    dtext_o: torch.Tensor | None,
    dsummary_o: torch.Tensor | None,
    dsummary_states: torch.Tensor | None,
    scale: float,
    needs_grad,
    text_cu_seqlens: torch.Tensor,
    summary_cu_seqlens: torch.Tensor,
) -> list[torch.Tensor | None]:
    """Split native HSD backward used while replacing the monolithic scan.

    Cross-chunk state dependencies are handled by a compact FlashQLA
    boundary-adjoint scan.  The remaining token gradients are computed by a
    chunk-parallel TileLang kernel whose CTA loops over all value tiles,
    avoiding the six complete q/k/g/A backward replicas in the original
    K192/V384 specialization.

    The current bring-up version still uses the existing WY reconstruction
    and transform kernels around the two native TileLang stages.  It is kept
    behind an explicit environment switch until full parity and performance
    gates pass.
    """

    if not _is_packed_cu(text_cu_seqlens, summary_cu_seqlens):
        raise RuntimeError("FlashQLA split backward requires packed cu_seqlens.")
    from ._flashqla.ops.gated_delta_rule.chunk.hopper.cp_bwd import (
        fused_gdr_dh_ws_compact,
        fused_gdr_prepare_dv_from_chunk_dh,
    )
    from ._flashqla.ops.utils import (
        chunk_local_cumsum as flashqla_chunk_local_cumsum,
    )
    from fla.ops.common.backends.tilelang.chunk_bwd import (
        chunk_bwd_dqkwg_tilelang,
    )

    batch, text_len, heads, _ = query.shape
    value_dim = int(value.shape[-1])
    if dtext_o is None:
        dtext_o = value.new_zeros(batch, text_len, heads, value_dim)
    else:
        dtext_o = dtext_o.contiguous()
    with torch.no_grad():
        with _profile_cuda_ms("chunk/flashqla_tile_split_bwd_compact_summary_ms"):
            dsummary_q, compact_dh = _prepare_flashqla_compact_summary_grads(
                summary_q=summary_q,
                summary_states=summary_states,
                dsummary_o=dsummary_o,
                dsummary_states=dsummary_states,
                scale=scale,
            )
        with _profile_cuda_ms("chunk/flashqla_tile_split_bwd_recompute_w_u_ms"):
            w, u = recompute_w_u_fwd(
                k=key,
                v=value,
                beta=beta,
                A=flash_A,
                g=flash_g_cumsum,
                cu_seqlens=text_cu_seqlens,
                chunk_indices=chunk_indices,
                use_exp2=False,
            )
        with _profile_cuda_ms("chunk/flashqla_tile_split_bwd_recompute_v_new_ms"):
            v_new = _recompute_v_new_from_saved_chunk_states(
                w=w,
                u=u,
                chunk_start_states=chunk_start_states,
                text_cu_seqlens=text_cu_seqlens,
                chunk_indices=chunk_indices,
            )
        with _profile_cuda_ms("chunk/flashqla_tile_split_bwd_state_scan_ms"):
            chunk_end_adjoints, _ = fused_gdr_dh_ws_compact(
                q=query,
                k=key,
                a=flash_A,
                g=flash_g_cumsum,
                b=beta,
                do=dtext_o,
                compact_dh=compact_dh,
                summary_cu_seqlens=summary_cu_seqlens,
                scale=scale,
                cu_seqlens=text_cu_seqlens,
                compact_chunk_stride=8,
                value_tile_size=64,
                state_v_first=False,
            )
        with _profile_cuda_ms("chunk/flashqla_tile_split_bwd_du_ms"):
            du = fused_gdr_prepare_dv_from_chunk_dh(
                q=query,
                k=key,
                g=flash_g_cumsum,
                do=dtext_o,
                dh=chunk_end_adjoints,
                cu_seqlens=text_cu_seqlens,
                scale=scale,
                chunk_size=FLA_CHUNK_SIZE,
                value_tile_size=64,
            )
        with _profile_cuda_ms("chunk/flashqla_tile_split_bwd_local_ms"):
            dq, dk, dw, dg = chunk_bwd_dqkwg_tilelang(
                q=query,
                k=key,
                v=v_new,
                do=dtext_o,
                h=chunk_start_states,
                dh=chunk_end_adjoints,
                w=w,
                g=flash_g_cumsum,
                dv=du,
                scale=scale,
                cu_seqlens=text_cu_seqlens,
                chunk_indices=chunk_indices,
                use_exp2=False,
                transpose_state_layout=False,
            )
        with _profile_cuda_ms("chunk/flashqla_tile_split_bwd_prepare_wy_ms"):
            dk2, dv, db, dg2 = prepare_wy_repr_bwd(
                k=key,
                v=value,
                beta=beta,
                g=flash_g_cumsum,
                A=flash_A,
                dw=dw,
                du=du,
                cu_seqlens=text_cu_seqlens,
                chunk_indices=chunk_indices,
                use_exp2=False,
            )
        dk.add_(dk2)
        dg.add_(dg2)
        with _profile_cuda_ms("chunk/flashqla_tile_split_bwd_cumsum_reverse_ms"):
            dg = flashqla_chunk_local_cumsum(
                dg,
                chunk_size=FLA_CHUNK_SIZE,
                reverse=True,
                cu_seqlens=text_cu_seqlens,
            )

    return [
        dq.to(query.dtype) if needs_grad[0] else None,
        dk.to(key.dtype) if needs_grad[1] else None,
        dv.to(value.dtype) if needs_grad[2] else None,
        db.to(beta.dtype) if needs_grad[3] else None,
        dg.to(flash_g_cumsum.dtype) if needs_grad[4] else None,
        dsummary_q.to(summary_q.dtype) if needs_grad[5] else None,
    ]


class HSDChunkGatedDeltaWithSummaryStatesFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        summary_q: torch.Tensor,
        boundary_counts: torch.Tensor,
        text_cu_seqlens: torch.Tensor,
        summary_cu_seqlens: torch.Tensor,
        scale: float,
    ):
        ctx.scale = float(scale)
        ctx.is_packed = _is_packed_cu(text_cu_seqlens, summary_cu_seqlens)
        with torch.no_grad():
            text_o, summary_o, summary_states, g_cumsum, A = _run_full_scan_forward(
                query=query,
                key=key,
                value=value,
                beta=beta,
                g=g,
                summary_q=summary_q,
                boundary_counts=boundary_counts,
                scale=float(scale),
                text_cu_seqlens=text_cu_seqlens if ctx.is_packed else None,
                summary_cu_seqlens=summary_cu_seqlens if ctx.is_packed else None,
                return_intermediates=True,
            )
        ctx.save_for_backward(
            query,
            key,
            value,
            beta,
            g,
            summary_q,
            boundary_counts,
            text_cu_seqlens,
            summary_cu_seqlens,
            g_cumsum,
            A,
            summary_states,
        )
        return text_o, summary_o, summary_states

    @staticmethod
    def backward(ctx, dtext_o: torch.Tensor, dsummary_o: torch.Tensor, dsummary_states: torch.Tensor):
        (
            query,
            key,
            value,
            beta,
            g,
            summary_q,
            boundary_counts,
            text_cu_seqlens,
            summary_cu_seqlens,
            g_cumsum,
            A,
            summary_states,
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
                    summary_q=summary_q,
                    boundary_counts=boundary_counts,
                    text_cu_seqlens=text_cu_seqlens if ctx.is_packed else None,
                    summary_cu_seqlens=summary_cu_seqlens if ctx.is_packed else None,
                    dtext_o=dtext_o,
                    dsummary_o=dsummary_o,
                    dsummary_states=dsummary_states,
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
                    summary_q=summary_q,
                    summary_states=summary_states,
                    boundary_counts=boundary_counts,
                    dtext_o=dtext_o,
                    dsummary_o=dsummary_o,
                    dsummary_states=dsummary_states,
                    scale=ctx.scale,
                    needs_grad=needs_grad,
                    text_cu_seqlens=text_cu_seqlens if ctx.is_packed else None,
                    summary_cu_seqlens=summary_cu_seqlens if ctx.is_packed else None,
                )

        return (*grad_inputs, None, None, None, None)


def _run_flashqla_compact_summary_read(
    *,
    text_o: torch.Tensor,
    summary_q: torch.Tensor,
    compact_states: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the already ordered FP32 chunk-512 boundary states."""

    batch, summary_count, heads, key_dim = summary_q.shape
    value_dim = int(text_o.shape[-1])
    if summary_count == 0:
        return (
            text_o.new_zeros((batch, 0, heads, value_dim)),
            torch.zeros(
                (batch, 0, heads, key_dim, value_dim),
                device=text_o.device,
                dtype=torch.float32,
            ),
        )
    expected_shape = (batch, summary_count, heads, key_dim, value_dim)
    if tuple(compact_states.shape) != expected_shape:
        raise RuntimeError(
            "FlashQLA compact summary-state shape mismatch: "
            f"expected={expected_shape}, actual={tuple(compact_states.shape)}."
        )
    if compact_states.dtype != torch.float32:
        raise RuntimeError(
            "FlashQLA compact summary states must preserve the FP32 recurrent state."
        )
    summary_states = compact_states
    if (
        triton is not None
        and _hsd_compact_summary_read_fwd_kernel is not None
        and summary_q.is_cuda
    ):
        summary_o = torch.empty(
            (batch, summary_count, heads, value_dim),
            device=text_o.device,
            dtype=text_o.dtype,
        )

        def grid(_meta):
            return (
                triton.cdiv(value_dim, 32),
                summary_count,
                batch * heads,
            )

        _hsd_compact_summary_read_fwd_kernel[grid](
            summary_q=summary_q,
            summary_states=summary_states,
            summary_o=summary_o,
            scale=float(scale),
            SUMMARY_COUNT=summary_count,
            H=heads,
            K=key_dim,
            V=value_dim,
            BK=64,
            BV=32,
        )
        return summary_o, summary_states

    summary_o = (
        torch.einsum(
            "bmhk,bmhkv->bmhv",
            summary_q.to(summary_states.dtype),
            summary_states,
        )
        * float(scale)
    ).to(text_o.dtype)
    return summary_o, summary_states


class HSDFlashQLATileWithSummaryStatesFunction(torch.autograd.Function):
    """Opt-in FlashQLA TileLang producer with compact boundary states.

    FlashQLA's Hopper forward keeps the recurrent state resident while it
    computes the chunk output.  This fork writes the FP32 recurrent fragment
    only after complete 512-token groups, so the producer does not materialize
    every internal chunk64 state.  The direct-A backward builds only one total
    dH per chunk-512 boundary, scans those boundary adjoints once, and consumes
    the saved chunk64 starts in its chunk-local inline-dD kernel.
    """

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        summary_q: torch.Tensor,
        boundary_counts: torch.Tensor,
        text_cu_seqlens: torch.Tensor,
        summary_cu_seqlens: torch.Tensor,
        scale: float,
        backward_backend: str,
    ):
        from ._flashqla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
        from ._flashqla.utils import (
            prepare_chunk_indices as flashqla_prepare_chunk_indices,
            prepare_chunk_offsets as flashqla_prepare_chunk_offsets,
        )

        # Do not turn unused producer outputs into dense zero gradients.  The
        # native compact backward can then compile away inactive summary
        # injection paths instead of reading and adding all-zero KxV tensors.
        ctx.set_materialize_grads(False)
        ctx.scale = float(scale)
        ctx.backward_backend = backward_backend
        if ctx.backward_backend == "direct_a":
            direct_a_low_precision_inputs = {
                "query": query,
                "key": key,
                "value": value,
                "beta": beta,
            }
            invalid_dtypes = {
                name: str(tensor.dtype)
                for name, tensor in direct_a_low_precision_inputs.items()
                if tensor.dtype != torch.bfloat16
            }
            if invalid_dtypes:
                raise RuntimeError(
                    "The first FlashQLA direct-A backward requires BF16 "
                    f"query/key/value/beta inputs; received {invalid_dtypes}."
                )
            if g.dtype != torch.float32:
                raise RuntimeError(
                    "The first FlashQLA direct-A backward requires an FP32 "
                    f"raw log gate; received {g.dtype}."
                )
        summary_count = int(summary_q.shape[1])
        needs_chunk_start_states = ctx.backward_backend in {
            "direct_a",
            "split",
        }
        if needs_chunk_start_states:
            _, flash_chunk_count = (
                flashqla_prepare_chunk_offsets(
                    text_cu_seqlens,
                    FLA_CHUNK_SIZE,
                )
            )
            flash_chunk_count = int(flash_chunk_count)
            chunk_indices = flashqla_prepare_chunk_indices(
                text_cu_seqlens,
                FLA_CHUNK_SIZE,
            ).contiguous()
            chunk_start_shape = (
                key.shape[0],
                flash_chunk_count,
                value.shape[2],
                key.shape[-1],
                value.shape[-1],
            )
            if _env_bool(
                "BENCHMARK_HSD_CHUNK_START_SENTINEL",
                False,
            ):
                # Diagnostic-only allocation.  Any element that the fused
                # producer fails to overwrite remains NaN and is reported by
                # BENCHMARK_HSD_TRACE_DIRECT_A before the backward kernel.
                chunk_start_states = key.new_full(
                    chunk_start_shape,
                    float("nan"),
                )
            else:
                chunk_start_states = key.new_empty(chunk_start_shape)
        else:
            chunk_start_states = key.new_empty((0,))
            chunk_indices = text_cu_seqlens.new_empty((0, 2))

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
            flash_g_cumsum, flash_A, text_o, summary_states, _final_state, _ = chunk_gated_delta_rule_fwd(
                q=flash_query,
                k=flash_key,
                v=flash_value,
                g=g,
                beta=beta,
                scale=float(scale),
                initial_state=None,
                cu_seqlens=text_cu_seqlens,
                output_final_state=False,
                output_h=summary_count > 0,
                output_h_stride=8,
                chunk_start_h=(
                    chunk_start_states
                    if needs_chunk_start_states
                    else None
                ),
                auto_cp=False,
                state_v_first=False,
                enable_fwd_cp_cache=False,
            )
        if pad_h16_k96_v96:
            # Drop the algebraically-zero padded rows/columns before exposing
            # producer outputs to the HSD summary/global-read path.
            text_o = text_o[..., :96].contiguous()
            if summary_states is not None:
                summary_states = summary_states[..., :96, :96].contiguous()
        if ctx.backward_backend != "optimized":
            g_cumsum, A = flash_g_cumsum, flash_A
        else:
            # Compatibility oracle for the pre-native implementation.  The
            # default path deliberately does not build these duplicate FLA
            # intermediates.
            g_cumsum, A = _prepare_fla_backward_intermediates(
                key=key,
                value=value,
                beta=beta,
                g=g,
                text_cu_seqlens=text_cu_seqlens,
            )
        if summary_states is None:
            if summary_count != 0:
                raise RuntimeError(
                    "FlashQLA TileLang HSD forward did not return summary states."
                )
            summary_states = torch.empty(
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

        with _profile_cuda_ms("chunk/flashqla_tile_summary_read_ms"):
            summary_o, summary_states = _run_flashqla_compact_summary_read(
                text_o=text_o,
                summary_q=summary_q,
                compact_states=summary_states,
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
            summary_q,
            summary_states,
            chunk_start_states,
            chunk_indices,
            boundary_counts,
            text_cu_seqlens,
            summary_cu_seqlens,
        )
        return text_o, summary_o, summary_states

    @staticmethod
    def backward(ctx, dtext_o: torch.Tensor, dsummary_o: torch.Tensor, dsummary_states: torch.Tensor):
        (
            query,
            key,
            value,
            beta,
            g,
            g_cumsum,
            A,
            summary_q,
            summary_states,
            chunk_start_states,
            chunk_indices,
            boundary_counts,
            text_cu_seqlens,
            summary_cu_seqlens,
        ) = ctx.saved_tensors
        if ctx.backward_backend == "reference":
            with _profile_cuda_ms("chunk/flashqla_tile_reference_bwd_ms"):
                grad_inputs = _run_reference_backward(
                    query=query,
                    key=key,
                    value=value,
                    beta=beta,
                    g=g,
                    summary_q=summary_q,
                    boundary_counts=boundary_counts,
                    text_cu_seqlens=text_cu_seqlens,
                    summary_cu_seqlens=summary_cu_seqlens,
                    dtext_o=dtext_o,
                    dsummary_o=dsummary_o,
                    dsummary_states=dsummary_states,
                    scale=ctx.scale,
                    needs_grad=ctx.needs_input_grad,
                )
        else:
            with _profile_cuda_ms("chunk/flashqla_tile_hsd_bwd_ms"):
                if ctx.backward_backend == "direct_a":
                    grad_inputs = _run_flashqla_compact_direct_a_backward(
                        query=query,
                        key=key,
                        value=value,
                        beta=beta,
                        flash_g_cumsum=g_cumsum,
                        flash_A=A,
                        summary_q=summary_q,
                        summary_states=summary_states,
                        chunk_start_states=chunk_start_states,
                        chunk_indices=chunk_indices,
                        dtext_o=dtext_o,
                        dsummary_o=dsummary_o,
                        dsummary_states=dsummary_states,
                        scale=ctx.scale,
                        needs_grad=ctx.needs_input_grad,
                        text_cu_seqlens=text_cu_seqlens,
                        summary_cu_seqlens=summary_cu_seqlens,
                    )
                elif ctx.backward_backend == "split":
                    grad_inputs = _run_flashqla_compact_split_backward(
                        query=query,
                        key=key,
                        value=value,
                        beta=beta,
                        flash_g_cumsum=g_cumsum,
                        flash_A=A,
                        summary_q=summary_q,
                        summary_states=summary_states,
                        chunk_start_states=chunk_start_states,
                        chunk_indices=chunk_indices,
                        dtext_o=dtext_o,
                        dsummary_o=dsummary_o,
                        dsummary_states=dsummary_states,
                        scale=ctx.scale,
                        needs_grad=ctx.needs_input_grad,
                        text_cu_seqlens=text_cu_seqlens,
                        summary_cu_seqlens=summary_cu_seqlens,
                    )
                elif ctx.backward_backend == "native":
                    grad_inputs = _run_flashqla_compact_native_backward(
                        query=query,
                        key=key,
                        value=value,
                        beta=beta,
                        flash_g_cumsum=g_cumsum,
                        flash_A=A,
                        summary_q=summary_q,
                        summary_states=summary_states,
                        dtext_o=dtext_o,
                        dsummary_o=dsummary_o,
                        dsummary_states=dsummary_states,
                        scale=ctx.scale,
                        needs_grad=ctx.needs_input_grad,
                        text_cu_seqlens=text_cu_seqlens,
                        summary_cu_seqlens=summary_cu_seqlens,
                    )
                elif ctx.backward_backend == "optimized":
                    grad_inputs = _run_flashqla_compact_optimized_backward(
                        query=query,
                        key=key,
                        value=value,
                        beta=beta,
                        g=g,
                        g_cumsum=g_cumsum,
                        A=A,
                        summary_q=summary_q,
                        summary_states=summary_states,
                        dtext_o=dtext_o,
                        dsummary_o=dsummary_o,
                        dsummary_states=dsummary_states,
                        scale=ctx.scale,
                        needs_grad=ctx.needs_input_grad,
                        text_cu_seqlens=text_cu_seqlens,
                        summary_cu_seqlens=summary_cu_seqlens,
                    )
                else:  # pragma: no cover - resolved and validated in forward.
                    raise AssertionError(
                        f"Unexpected FlashQLA backward backend: "
                        f"{ctx.backward_backend}"
                    )
        return (*grad_inputs, None, None, None, None, None)


def hsd_flashqla_tile_gated_delta_rule_with_summary_states(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    summary_q: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: Optional[torch.Tensor] = None,
    summary_cu_seqlens: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    backward_backend: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Run the opt-in Hopper TileLang producer for an admitted HSD layout.

    ``backward_backend=None`` preserves the legacy environment-variable
    resolution.  Supplying it explicitly makes the model configuration the
    source of truth and freezes that choice on the autograd context.
    """

    if not query.is_cuda or text_cu_seqlens is None or summary_cu_seqlens is None:
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
        summary_q.ndim != 4
        or summary_q.shape[0] != batch
        or summary_q.shape[2:] != (heads, key_dim)
    ):
        return None
    if (
        text_len <= 0
        or boundary_counts.numel() != summary_q.shape[1]
    ):
        return None
    if text_cu_seqlens.ndim != 1 or summary_cu_seqlens.ndim != 1:
        return None
    if text_cu_seqlens.numel() != summary_cu_seqlens.numel() or text_cu_seqlens.numel() < 2:
        return None
    if torch.cuda.get_device_capability(query.device)[0] != 9:
        return None

    # FlashQLA's packed Hopper kernels are specialized around int32 metadata.
    # Keeping the public HSD sidecars as int64 is harmless, but lowering the
    # TileLang direct-A kernel with int64 indices currently produces a kernel
    # that compiles yet fails at launch on GPU.  Sequence lengths in this
    # fixed training backend are far below the int32 limit, so canonicalize
    # only the private producer metadata here.
    text_cu, summary_cu, boundary, packed_layout_valid = (
        _prepare_flashqla_packed_metadata(
            boundary_counts=boundary_counts,
            text_cu_seqlens=text_cu_seqlens,
            summary_cu_seqlens=summary_cu_seqlens,
            text_len=text_len,
            summary_count=int(summary_q.shape[1]),
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
    return HSDFlashQLATileWithSummaryStatesFunction.apply(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        beta.contiguous(),
        g.contiguous(),
        summary_q.contiguous(),
        boundary,
        text_cu,
        summary_cu,
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
    summary_cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the training-identical FlashQLA producer for inference prefill.

    Unlike :func:`hsd_flashqla_tile_gated_delta_rule_with_summary_states`,
    this entry point is inference-only and also returns the recurrent state
    after each packed sequence.  The state ABI is HSD-native ``[H, K, V]``:
    compact summary states are ``[1, M, H, K, V]`` and final states are
    ``[N, H, K, V]`` for ``N = len(text_cu_seqlens) - 1``.  It intentionally
    does not compute the summary query read; serving already owns that step.

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
        ("summary_cu_seqlens", summary_cu_seqlens),
    ):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1:
            raise RuntimeError(f"{prefix} requires {name} to be a rank-1 tensor.")
        if tensor.dtype not in integer_dtypes:
            raise RuntimeError(
                f"{prefix} requires integer {name}; received {tensor.dtype}."
            )
    if (
        text_cu_seqlens.numel() < 2
        or text_cu_seqlens.numel() != summary_cu_seqlens.numel()
    ):
        raise RuntimeError(
            f"{prefix} requires matching text/summary cu-seqlens with at "
            "least one packed sequence."
        )
    summary_count = int(boundary_counts.numel())
    text_cu, _summary_cu, _boundary, packed_layout_valid = (
        _prepare_flashqla_packed_metadata(
            boundary_counts=boundary_counts,
            text_cu_seqlens=text_cu_seqlens,
            summary_cu_seqlens=summary_cu_seqlens,
            text_len=text_len,
            summary_count=summary_count,
            device=query.device,
        )
    )
    if not packed_layout_valid:
        raise RuntimeError(
            f"{prefix} requires exact per-sequence 512-token summary "
            "boundaries and sidecars spanning all text and summary states."
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
        _, _, text_o, summary_states, final_state, _ = (
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
                output_h=summary_count > 0,
                output_h_stride=8,
                auto_cp=False,
                state_v_first=False,
                enable_fwd_cp_cache=False,
            )
        )

    if summary_states is None:
        if summary_count:
            raise RuntimeError(
                f"{prefix} did not return the requested compact summary states."
            )
        summary_states = torch.empty(
            (1, 0, heads, flash_query.shape[-1], flash_value.shape[-1]),
            dtype=torch.float32,
            device=query.device,
        )
    if final_state is None:
        raise RuntimeError(f"{prefix} did not return final recurrent states.")

    if pad_h16_k96_v96:
        text_o = text_o[..., :value_dim]
        summary_states = summary_states[..., :key_dim, :value_dim]
        final_state = final_state[..., :key_dim, :value_dim]

    expected_text_shape = (batch, text_len, heads, value_dim)
    expected_summary_shape = (
        batch,
        summary_count,
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
    if tuple(summary_states.shape) != expected_summary_shape:
        raise RuntimeError(
            f"{prefix} compact summary-state shape mismatch: expected "
            f"{expected_summary_shape}, received {tuple(summary_states.shape)}."
        )
    if tuple(final_state.shape) != expected_final_shape:
        raise RuntimeError(
            f"{prefix} final-state shape mismatch: expected "
            f"{expected_final_shape}, received {tuple(final_state.shape)}."
        )
    if summary_states.dtype != torch.float32 or final_state.dtype != torch.float32:
        raise RuntimeError(
            f"{prefix} requires FP32 compact/final recurrent states; received "
            f"summary={summary_states.dtype}, final={final_state.dtype}."
        )
    return (
        text_o.contiguous(),
        summary_states.contiguous(),
        final_state.contiguous(),
    )


def hsd_chunk_gated_delta_rule_with_summary_states(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    summary_q: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: Optional[torch.Tensor] = None,
    summary_cu_seqlens: Optional[torch.Tensor] = None,
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
    if summary_q.ndim != 4 or summary_q.shape[:1] != (batch,) or summary_q.shape[2:] != (heads, key_dim):
        return None
    summary_count = int(summary_q.shape[1])
    if boundary_counts.ndim != 1 or int(boundary_counts.shape[0]) != summary_count:
        return None
    if key_dim > 256 or text_len <= 0:
        return None
    packed = text_cu_seqlens is not None or summary_cu_seqlens is not None
    if packed:
        if prepare_chunk_indices is None or prepare_chunk_offsets is None:
            return None
        if text_cu_seqlens is None or summary_cu_seqlens is None:
            return None
        if batch != 1:
            return None
        if text_cu_seqlens.ndim != 1 or summary_cu_seqlens.ndim != 1:
            return None
        if text_cu_seqlens.numel() < 2 or int(text_cu_seqlens.numel()) != int(summary_cu_seqlens.numel()):
            return None
        text_cu = text_cu_seqlens.to(device=query.device, dtype=torch.long).contiguous()
        summary_cu = summary_cu_seqlens.to(device=query.device, dtype=torch.long).contiguous()
        text_cu_cpu = text_cu.detach().to(device="cpu", dtype=torch.long)
        summary_cu_cpu = summary_cu.detach().to(device="cpu", dtype=torch.long)
        if int(text_cu_cpu[0].item()) != 0 or int(text_cu_cpu[-1].item()) != text_len:
            return None
        if int(summary_cu_cpu[0].item()) != 0 or int(summary_cu_cpu[-1].item()) != summary_count:
            return None
        if bool((text_cu_cpu[1:] < text_cu_cpu[:-1]).any().item()):
            return None
        if bool((summary_cu_cpu[1:] < summary_cu_cpu[:-1]).any().item()):
            return None
    else:
        text_cu = boundary_counts.new_empty((0,), dtype=torch.long)
        summary_cu = boundary_counts.new_empty((0,), dtype=torch.long)

    boundaries = boundary_counts.to(device=query.device, dtype=torch.long)
    boundary_cpu = boundaries.detach().to(device="cpu", dtype=torch.long)
    if packed:
        for doc_idx in range(int(text_cu_cpu.numel()) - 1):
            text_start = int(text_cu_cpu[doc_idx].item())
            text_end = int(text_cu_cpu[doc_idx + 1].item())
            doc_len = text_end - text_start
            summary_start = int(summary_cu_cpu[doc_idx].item())
            summary_end = int(summary_cu_cpu[doc_idx + 1].item())
            local_boundaries = boundary_cpu[summary_start:summary_end] - text_start
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
        return HSDChunkGatedDeltaWithSummaryStatesFunction.apply(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            beta.contiguous(),
            g.contiguous(),
            summary_q.contiguous(),
            boundary_counts.to(device=query.device, dtype=torch.long).contiguous(),
            text_cu,
            summary_cu,
            float(scale),
        )


__all__ = [
    "clear_hsd_chunk_profile_stats",
    "hsd_flashqla_tile_gated_delta_rule_inference",
    "hsd_flashqla_tile_gated_delta_rule_with_summary_states",
    "hsd_chunk_gated_delta_rule_with_summary_states",
    "pop_hsd_chunk_profile_stats",
]
