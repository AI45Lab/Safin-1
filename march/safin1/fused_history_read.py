"""Optional Triton kernels for HSD decode history read."""

from __future__ import annotations

import math
import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is optional at import time.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _hsd_decode_top4_history_read_kernel(
        read_q,
        router_q,
        states,
        anchor_k,
        boundaries,
        out,
        start_index: tl.constexpr,
        anchor_count: tl.constexpr,
        key_dim: tl.constexpr,
        value_dim: tl.constexpr,
        route_scale: tl.constexpr,
        read_scale: tl.constexpr,
        read_q_stride_b: tl.constexpr,
        read_q_stride_t: tl.constexpr,
        read_q_stride_h: tl.constexpr,
        read_q_stride_k: tl.constexpr,
        router_q_stride_b: tl.constexpr,
        router_q_stride_t: tl.constexpr,
        router_q_stride_h: tl.constexpr,
        router_q_stride_k: tl.constexpr,
        states_stride_b: tl.constexpr,
        states_stride_m: tl.constexpr,
        states_stride_h: tl.constexpr,
        states_stride_k: tl.constexpr,
        states_stride_v: tl.constexpr,
        anchor_k_stride_b: tl.constexpr,
        anchor_k_stride_m: tl.constexpr,
        anchor_k_stride_h: tl.constexpr,
        anchor_k_stride_k: tl.constexpr,
        boundaries_stride_m: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_t: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_v: tl.constexpr,
        num_v_blocks: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        v_block_idx = tl.program_id(2)
        offs_m = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = v_block_idx * BLOCK_V + tl.arange(0, BLOCK_V)
        offs_m_safe = tl.minimum(offs_m, anchor_count - 1)

        router = tl.load(
            router_q
            + batch_idx * router_q_stride_b
            + head_idx * router_q_stride_h
            + offs_k * router_q_stride_k,
            mask=offs_k < key_dim,
            other=0.0,
        ).to(tl.float32)
        keys = tl.load(
            anchor_k
            + batch_idx * anchor_k_stride_b
            + offs_m_safe[:, None] * anchor_k_stride_m
            + head_idx * anchor_k_stride_h
            + offs_k[None, :] * anchor_k_stride_k,
            mask=(offs_m[:, None] < anchor_count) & (offs_k[None, :] < key_dim),
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(keys * router[None, :], axis=1) * route_scale

        boundary = tl.load(
            boundaries + offs_m_safe * boundaries_stride_m,
            mask=offs_m < anchor_count,
            other=start_index + 1,
        )
        valid = (offs_m < anchor_count) & (boundary <= start_index)
        neg_inf = -float("inf")
        scores0 = tl.where(valid, logits, neg_inf)

        idx0 = tl.argmax(scores0, axis=0, tie_break_left=True)
        val0 = tl.max(scores0, axis=0)
        scores1 = tl.where(offs_m == idx0, neg_inf, scores0)
        idx1 = tl.argmax(scores1, axis=0, tie_break_left=True)
        val1 = tl.max(scores1, axis=0)
        scores2 = tl.where(offs_m == idx1, neg_inf, scores1)
        idx2 = tl.argmax(scores2, axis=0, tie_break_left=True)
        val2 = tl.max(scores2, axis=0)
        scores3 = tl.where(offs_m == idx2, neg_inf, scores2)
        idx3 = tl.argmax(scores3, axis=0, tie_break_left=True)
        val3 = tl.max(scores3, axis=0)

        has0 = val0 != neg_inf
        has1 = val1 != neg_inf
        has2 = val2 != neg_inf
        has3 = val3 != neg_inf
        idx0_safe = tl.minimum(idx0, anchor_count - 1)
        idx1_safe = tl.minimum(idx1, anchor_count - 1)
        idx2_safe = tl.minimum(idx2, anchor_count - 1)
        idx3_safe = tl.minimum(idx3, anchor_count - 1)
        max_val = tl.maximum(tl.maximum(val0, val1), tl.maximum(val2, val3))
        max_val = tl.where(has0 | has1 | has2 | has3, max_val, 0.0)
        exp0 = tl.where(has0, tl.exp(val0 - max_val), 0.0)
        exp1 = tl.where(has1, tl.exp(val1 - max_val), 0.0)
        exp2 = tl.where(has2, tl.exp(val2 - max_val), 0.0)
        exp3 = tl.where(has3, tl.exp(val3 - max_val), 0.0)
        denom = exp0 + exp1 + exp2 + exp3
        inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
        w0 = exp0 * inv_denom
        w1 = exp1 * inv_denom
        w2 = exp2 * inv_denom
        w3 = exp3 * inv_denom

        read = tl.load(
            read_q
            + batch_idx * read_q_stride_b
            + head_idx * read_q_stride_h
            + offs_k * read_q_stride_k,
            mask=offs_k < key_dim,
            other=0.0,
        ).to(tl.float32)

        state0 = tl.load(
            states
            + batch_idx * states_stride_b
            + idx0_safe * states_stride_m
            + head_idx * states_stride_h
            + offs_k[:, None] * states_stride_k
            + offs_v[None, :] * states_stride_v,
            mask=has0 & (idx0 < anchor_count) & (offs_k[:, None] < key_dim) & (offs_v[None, :] < value_dim),
            other=0.0,
        ).to(tl.float32)
        state1 = tl.load(
            states
            + batch_idx * states_stride_b
            + idx1_safe * states_stride_m
            + head_idx * states_stride_h
            + offs_k[:, None] * states_stride_k
            + offs_v[None, :] * states_stride_v,
            mask=has1 & (idx1 < anchor_count) & (offs_k[:, None] < key_dim) & (offs_v[None, :] < value_dim),
            other=0.0,
        ).to(tl.float32)
        state2 = tl.load(
            states
            + batch_idx * states_stride_b
            + idx2_safe * states_stride_m
            + head_idx * states_stride_h
            + offs_k[:, None] * states_stride_k
            + offs_v[None, :] * states_stride_v,
            mask=has2 & (idx2 < anchor_count) & (offs_k[:, None] < key_dim) & (offs_v[None, :] < value_dim),
            other=0.0,
        ).to(tl.float32)
        state3 = tl.load(
            states
            + batch_idx * states_stride_b
            + idx3_safe * states_stride_m
            + head_idx * states_stride_h
            + offs_k[:, None] * states_stride_k
            + offs_v[None, :] * states_stride_v,
            mask=has3 & (idx3 < anchor_count) & (offs_k[:, None] < key_dim) & (offs_v[None, :] < value_dim),
            other=0.0,
        ).to(tl.float32)

        read0 = tl.sum(read[:, None] * state0, axis=0) * read_scale
        read1 = tl.sum(read[:, None] * state1, axis=0) * read_scale
        read2 = tl.sum(read[:, None] * state2, axis=0) * read_scale
        read3 = tl.sum(read[:, None] * state3, axis=0) * read_scale
        result = read0 * w0 + read1 * w1 + read2 * w2 + read3 * w3

        tl.store(
            out
            + batch_idx * out_stride_b
            + head_idx * out_stride_h
            + offs_v * out_stride_v,
            result,
            mask=offs_v < value_dim,
        )


    @triton.jit
    def _hsd_prefill_top4_history_read_kernel(
        read_q,
        router_q,
        states,
        anchor_k,
        boundaries,
        out,
        start_index: tl.constexpr,
        text_len: tl.constexpr,
        anchor_count: tl.constexpr,
        key_dim: tl.constexpr,
        value_dim: tl.constexpr,
        route_scale: tl.constexpr,
        read_scale: tl.constexpr,
        read_q_stride_b: tl.constexpr,
        read_q_stride_t: tl.constexpr,
        read_q_stride_h: tl.constexpr,
        read_q_stride_k: tl.constexpr,
        router_q_stride_b: tl.constexpr,
        router_q_stride_t: tl.constexpr,
        router_q_stride_h: tl.constexpr,
        router_q_stride_k: tl.constexpr,
        states_stride_b: tl.constexpr,
        states_stride_m: tl.constexpr,
        states_stride_h: tl.constexpr,
        states_stride_k: tl.constexpr,
        states_stride_v: tl.constexpr,
        anchor_k_stride_b: tl.constexpr,
        anchor_k_stride_m: tl.constexpr,
        anchor_k_stride_h: tl.constexpr,
        anchor_k_stride_k: tl.constexpr,
        boundaries_stride_m: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_t: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_v: tl.constexpr,
        num_v_blocks: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        batch_token_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        v_block_idx = tl.program_id(2)
        batch_idx = batch_token_idx // text_len
        token_idx = batch_token_idx - batch_idx * text_len
        offs_m = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = v_block_idx * BLOCK_V + tl.arange(0, BLOCK_V)
        offs_m_safe = tl.minimum(offs_m, anchor_count - 1)

        router = tl.load(
            router_q
            + batch_idx * router_q_stride_b
            + token_idx * router_q_stride_t
            + head_idx * router_q_stride_h
            + offs_k * router_q_stride_k,
            mask=(token_idx < text_len) & (offs_k < key_dim),
            other=0.0,
        ).to(tl.float32)
        keys = tl.load(
            anchor_k
            + batch_idx * anchor_k_stride_b
            + offs_m_safe[:, None] * anchor_k_stride_m
            + head_idx * anchor_k_stride_h
            + offs_k[None, :] * anchor_k_stride_k,
            mask=(offs_m[:, None] < anchor_count) & (offs_k[None, :] < key_dim),
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(keys * router[None, :], axis=1) * route_scale

        boundary = tl.load(
            boundaries + offs_m_safe * boundaries_stride_m,
            mask=offs_m < anchor_count,
            other=start_index + token_idx + 1,
        )
        valid = (offs_m < anchor_count) & (boundary <= start_index + token_idx)
        neg_inf = -float("inf")
        scores0 = tl.where(valid, logits, neg_inf)

        idx0 = tl.argmax(scores0, axis=0, tie_break_left=True)
        val0 = tl.max(scores0, axis=0)
        scores1 = tl.where(offs_m == idx0, neg_inf, scores0)
        idx1 = tl.argmax(scores1, axis=0, tie_break_left=True)
        val1 = tl.max(scores1, axis=0)
        scores2 = tl.where(offs_m == idx1, neg_inf, scores1)
        idx2 = tl.argmax(scores2, axis=0, tie_break_left=True)
        val2 = tl.max(scores2, axis=0)
        scores3 = tl.where(offs_m == idx2, neg_inf, scores2)
        idx3 = tl.argmax(scores3, axis=0, tie_break_left=True)
        val3 = tl.max(scores3, axis=0)

        has0 = val0 != neg_inf
        has1 = val1 != neg_inf
        has2 = val2 != neg_inf
        has3 = val3 != neg_inf
        idx0_safe = tl.minimum(idx0, anchor_count - 1)
        idx1_safe = tl.minimum(idx1, anchor_count - 1)
        idx2_safe = tl.minimum(idx2, anchor_count - 1)
        idx3_safe = tl.minimum(idx3, anchor_count - 1)
        max_val = tl.maximum(tl.maximum(val0, val1), tl.maximum(val2, val3))
        max_val = tl.where(has0 | has1 | has2 | has3, max_val, 0.0)
        exp0 = tl.where(has0, tl.exp(val0 - max_val), 0.0)
        exp1 = tl.where(has1, tl.exp(val1 - max_val), 0.0)
        exp2 = tl.where(has2, tl.exp(val2 - max_val), 0.0)
        exp3 = tl.where(has3, tl.exp(val3 - max_val), 0.0)
        denom = exp0 + exp1 + exp2 + exp3
        inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
        w0 = exp0 * inv_denom
        w1 = exp1 * inv_denom
        w2 = exp2 * inv_denom
        w3 = exp3 * inv_denom

        read = tl.load(
            read_q
            + batch_idx * read_q_stride_b
            + token_idx * read_q_stride_t
            + head_idx * read_q_stride_h
            + offs_k * read_q_stride_k,
            mask=(token_idx < text_len) & (offs_k < key_dim),
            other=0.0,
        ).to(tl.float32)

        state0 = tl.load(
            states
            + batch_idx * states_stride_b
            + idx0_safe * states_stride_m
            + head_idx * states_stride_h
            + offs_k[:, None] * states_stride_k
            + offs_v[None, :] * states_stride_v,
            mask=has0 & (idx0 < anchor_count) & (offs_k[:, None] < key_dim) & (offs_v[None, :] < value_dim),
            other=0.0,
        ).to(tl.float32)
        state1 = tl.load(
            states
            + batch_idx * states_stride_b
            + idx1_safe * states_stride_m
            + head_idx * states_stride_h
            + offs_k[:, None] * states_stride_k
            + offs_v[None, :] * states_stride_v,
            mask=has1 & (idx1 < anchor_count) & (offs_k[:, None] < key_dim) & (offs_v[None, :] < value_dim),
            other=0.0,
        ).to(tl.float32)
        state2 = tl.load(
            states
            + batch_idx * states_stride_b
            + idx2_safe * states_stride_m
            + head_idx * states_stride_h
            + offs_k[:, None] * states_stride_k
            + offs_v[None, :] * states_stride_v,
            mask=has2 & (idx2 < anchor_count) & (offs_k[:, None] < key_dim) & (offs_v[None, :] < value_dim),
            other=0.0,
        ).to(tl.float32)
        state3 = tl.load(
            states
            + batch_idx * states_stride_b
            + idx3_safe * states_stride_m
            + head_idx * states_stride_h
            + offs_k[:, None] * states_stride_k
            + offs_v[None, :] * states_stride_v,
            mask=has3 & (idx3 < anchor_count) & (offs_k[:, None] < key_dim) & (offs_v[None, :] < value_dim),
            other=0.0,
        ).to(tl.float32)

        read0 = tl.sum(read[:, None] * state0, axis=0) * read_scale
        read1 = tl.sum(read[:, None] * state1, axis=0) * read_scale
        read2 = tl.sum(read[:, None] * state2, axis=0) * read_scale
        read3 = tl.sum(read[:, None] * state3, axis=0) * read_scale
        result = read0 * w0 + read1 * w1 + read2 * w2 + read3 * w3

        tl.store(
            out
            + batch_idx * out_stride_b
            + token_idx * out_stride_t
            + head_idx * out_stride_h
            + offs_v * out_stride_v,
            result,
            mask=(token_idx < text_len) & (offs_v < value_dim),
        )


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _value_block_size(value_dim: int) -> int:
    env_value = os.environ.get("HSD_READ_VALUE_BLOCK_SIZE", "").strip()
    if env_value:
        block = int(env_value)
        if block <= 0:
            raise ValueError(f"HSD_READ_VALUE_BLOCK_SIZE must be positive, got {block}.")
    else:
        block = 16
    block = min(block, _next_power_of_2(value_dim))
    return _next_power_of_2(block)


def fused_decode_top4_history_read(
    *,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    states: torch.Tensor,
    anchor_k: torch.Tensor,
    boundary_counts: torch.Tensor,
    start_index: int,
    route_scale: float,
    read_scale: float,
) -> torch.Tensor | None:
    """Return fused decode history read output, or ``None`` when unsupported."""

    if os.environ.get("HSD_DISABLE_FUSED_HISTORY_READ", "").lower() in {"1", "true", "yes", "on"}:
        return None
    if os.environ.get("HSD_DISABLE_FUSED_HISTORY_READ_DECODE", "").lower() in {"1", "true", "yes", "on"}:
        return None
    if triton is None or not read_q.is_cuda:
        return None
    if read_q.ndim != 4 or router_q.ndim != 4 or states.ndim != 5 or anchor_k.ndim != 4:
        return None
    if router_q.device != read_q.device or states.device != read_q.device or anchor_k.device != read_q.device:
        return None
    batch_size, text_len, heads, key_dim = read_q.shape
    if text_len != 1:
        return None
    if router_q.shape != read_q.shape:
        return None
    if states.shape[0] != batch_size or states.shape[2] != heads or states.shape[3] != key_dim:
        return None
    if anchor_k.shape[:3] != states.shape[:3] or anchor_k.shape[3] != key_dim:
        return None
    anchor_count = int(states.shape[1])
    value_dim = int(states.shape[4])
    if anchor_count <= 4 or anchor_count > 1024:
        return None
    if key_dim > 128 or value_dim > 128:
        return None
    if boundary_counts.ndim != 1 or int(boundary_counts.shape[0]) != anchor_count:
        return None

    read_q = read_q.contiguous()
    router_q = router_q.contiguous()
    states = states.contiguous()
    anchor_k = anchor_k.contiguous()
    boundary_counts = boundary_counts.to(device=read_q.device, dtype=torch.long).contiguous()
    out = torch.empty((batch_size, 1, heads, value_dim), device=read_q.device, dtype=read_q.dtype)

    block_m = _next_power_of_2(anchor_count)
    block_k = _next_power_of_2(key_dim)
    block_v = _value_block_size(value_dim)
    num_v_blocks = triton.cdiv(value_dim, block_v)
    num_warps = 8 if block_m >= 512 else 4
    with torch.cuda.device(read_q.device):
        _hsd_decode_top4_history_read_kernel[(batch_size, heads, num_v_blocks)](
            read_q,
            router_q,
            states,
            anchor_k,
            boundary_counts,
            out,
            start_index=int(start_index),
            anchor_count=anchor_count,
            key_dim=int(key_dim),
            value_dim=value_dim,
            route_scale=float(route_scale),
            read_scale=float(read_scale),
            read_q_stride_b=read_q.stride(0),
            read_q_stride_t=read_q.stride(1),
            read_q_stride_h=read_q.stride(2),
            read_q_stride_k=read_q.stride(3),
            router_q_stride_b=router_q.stride(0),
            router_q_stride_t=router_q.stride(1),
            router_q_stride_h=router_q.stride(2),
            router_q_stride_k=router_q.stride(3),
            states_stride_b=states.stride(0),
            states_stride_m=states.stride(1),
            states_stride_h=states.stride(2),
            states_stride_k=states.stride(3),
            states_stride_v=states.stride(4),
            anchor_k_stride_b=anchor_k.stride(0),
            anchor_k_stride_m=anchor_k.stride(1),
            anchor_k_stride_h=anchor_k.stride(2),
            anchor_k_stride_k=anchor_k.stride(3),
            boundaries_stride_m=boundary_counts.stride(0),
            out_stride_b=out.stride(0),
            out_stride_t=out.stride(1),
            out_stride_h=out.stride(2),
            out_stride_v=out.stride(3),
            num_v_blocks=num_v_blocks,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            BLOCK_V=block_v,
            num_warps=num_warps,
        )
    return out


def fused_prefill_top4_history_read(
    *,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    states: torch.Tensor,
    anchor_k: torch.Tensor,
    boundary_counts: torch.Tensor,
    start_index: int,
    route_scale: float,
    read_scale: float,
) -> torch.Tensor | None:
    """Return fused block/prefill history read output, or ``None`` when unsupported.

    This forward-only kernel intentionally falls back when autograd is active for
    any input. Training should keep using the PyTorch reference path until a
    matching backward implementation exists.
    """

    if os.environ.get("HSD_DISABLE_FUSED_HISTORY_READ", "").lower() in {"1", "true", "yes", "on"}:
        return None
    if os.environ.get("HSD_DISABLE_FUSED_HISTORY_READ_PREFILL", "").lower() in {"1", "true", "yes", "on"}:
        return None
    if triton is None or not read_q.is_cuda:
        return None
    if torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (read_q, router_q, states, anchor_k) if torch.is_tensor(tensor)
    ):
        return None
    if read_q.ndim != 4 or router_q.ndim != 4 or states.ndim != 5 or anchor_k.ndim != 4:
        return None
    if router_q.device != read_q.device or states.device != read_q.device or anchor_k.device != read_q.device:
        return None
    batch_size, text_len, heads, key_dim = read_q.shape
    if text_len <= 1:
        return None
    if router_q.shape != read_q.shape:
        return None
    if states.shape[0] != batch_size or states.shape[2] != heads or states.shape[3] != key_dim:
        return None
    if anchor_k.shape[:3] != states.shape[:3] or anchor_k.shape[3] != key_dim:
        return None
    anchor_count = int(states.shape[1])
    value_dim = int(states.shape[4])
    if anchor_count <= 4 or anchor_count > 1024:
        return None
    if key_dim > 128 or value_dim > 128:
        return None
    if boundary_counts.ndim != 1 or int(boundary_counts.shape[0]) != anchor_count:
        return None

    read_q = read_q.contiguous()
    router_q = router_q.contiguous()
    states = states.contiguous()
    anchor_k = anchor_k.contiguous()
    boundary_counts = boundary_counts.to(device=read_q.device, dtype=torch.long).contiguous()
    out = torch.empty((batch_size, text_len, heads, value_dim), device=read_q.device, dtype=read_q.dtype)

    block_m = _next_power_of_2(anchor_count)
    block_k = _next_power_of_2(key_dim)
    block_v = _value_block_size(value_dim)
    num_v_blocks = triton.cdiv(value_dim, block_v)
    num_warps = 8 if block_m >= 512 else 4
    with torch.cuda.device(read_q.device):
        _hsd_prefill_top4_history_read_kernel[(batch_size * text_len, heads, num_v_blocks)](
            read_q,
            router_q,
            states,
            anchor_k,
            boundary_counts,
            out,
            start_index=int(start_index),
            text_len=int(text_len),
            anchor_count=anchor_count,
            key_dim=int(key_dim),
            value_dim=value_dim,
            route_scale=float(route_scale),
            read_scale=float(read_scale),
            read_q_stride_b=read_q.stride(0),
            read_q_stride_t=read_q.stride(1),
            read_q_stride_h=read_q.stride(2),
            read_q_stride_k=read_q.stride(3),
            router_q_stride_b=router_q.stride(0),
            router_q_stride_t=router_q.stride(1),
            router_q_stride_h=router_q.stride(2),
            router_q_stride_k=router_q.stride(3),
            states_stride_b=states.stride(0),
            states_stride_m=states.stride(1),
            states_stride_h=states.stride(2),
            states_stride_k=states.stride(3),
            states_stride_v=states.stride(4),
            anchor_k_stride_b=anchor_k.stride(0),
            anchor_k_stride_m=anchor_k.stride(1),
            anchor_k_stride_h=anchor_k.stride(2),
            anchor_k_stride_k=anchor_k.stride(3),
            boundaries_stride_m=boundary_counts.stride(0),
            out_stride_b=out.stride(0),
            out_stride_t=out.stride(1),
            out_stride_h=out.stride(2),
            out_stride_v=out.stride(3),
            num_v_blocks=num_v_blocks,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            BLOCK_V=block_v,
            num_warps=num_warps,
        )
    return out


__all__ = ["fused_decode_top4_history_read", "fused_prefill_top4_history_read"]
