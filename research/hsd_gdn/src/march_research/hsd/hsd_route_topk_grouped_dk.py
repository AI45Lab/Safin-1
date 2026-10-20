r"""Experimental small-M summary-grouped dRouteK kernel.

This module is isolated from the production fused-route implementation.  It
specializes the HSD production geometry

``B=1, QH=H=6, KH=1, D=64, K=4, M<=32``

and computes only the gradient of the shared route key.  There is no CSR
construction: each ``(summary, split)`` CTA scans a disjoint strided partition
of the flat selected-route array, retains entries targeting its summary, and
atomically publishes one accumulated ``D`` vector.  This changes dRouteK from
one vector atomic per selected token/head/slot to one vector atomic per
``(summary, split)``.

For selected summary probability ``p_j`` and upstream gradient ``dp_j``, with
an optional null candidate, the kernel implements

.. math::

    c = \sum_j p_j dp_j + p_\\varnothing dp_\\varnothing,

    dz_j = p_j (dp_j - c)\\,s,

    dK_m = \sum_{t,h,j:\\,i_{thj}=m} dz_{thj} Q_{th}.

The public entry point requires ``force=True`` and never falls back.
"""

from __future__ import annotations

import torch

from .hsd_route_topk import (
    _route_backward_launch_config,
    _route_topk_bwd_kernel,
)

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional CUDA dependency.
    triton = None
    tl = None


FIXED_BATCH = 1
FIXED_HEADS = 6
FIXED_ROUTE_DIM = 64
FIXED_TOPK = 4
MAX_SUMMARIES = 32
DEFAULT_SPLITS = 32
DEFAULT_BLOCK_N = 64


if triton is not None:

    @triton.jit
    def _small_m_scan_all_grouped_dk_kernel(
        route_q,
        indices,
        weights,
        valid,
        null_prob,
        dweights,
        dnull_prob,
        dk_acc,
        T: tl.constexpr,
        M: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        TOPK: tl.constexpr,
        ROUTE_SCALE: tl.constexpr,
        HAS_NULL: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        GROUP_SPLITS: tl.constexpr,
    ):
        i_d = tl.program_id(0)
        group = tl.program_id(1)
        split = tl.program_id(2)
        summary = group % M
        batch = group // M

        offs_n = tl.arange(0, BLOCK_N)
        offs_d = i_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        offs_j = tl.arange(0, TOPK)
        selections_per_batch = T * H * TOPK
        pos = split * BLOCK_N
        dk = tl.zeros((BLOCK_D,), tl.float32)

        while pos < selections_per_batch:
            local_selection = pos + offs_n
            selection_mask = local_selection < selections_per_batch
            selection = batch * selections_per_batch + local_selection
            flat_th = selection // TOPK
            slot = selection % TOPK
            selected_summary = tl.load(
                indices + selection,
                mask=selection_mask,
                other=-1,
            ).to(tl.int32)
            selected_valid = tl.load(
                valid + selection,
                mask=selection_mask,
                other=0,
            ).to(tl.int1)
            matched = (
                selection_mask
                & selected_valid
                & (selected_summary == summary)
            )

            route_base = flat_th[:, None] * TOPK + offs_j[None, :]
            p = tl.load(
                weights + route_base,
                mask=matched[:, None],
                other=0.0,
            ).to(tl.float32)
            dp = tl.load(
                dweights + route_base,
                mask=matched[:, None],
                other=0.0,
            ).to(tl.float32)
            if HAS_NULL:
                p_null = tl.load(
                    null_prob + flat_th,
                    mask=matched,
                    other=0.0,
                ).to(tl.float32)
                dp_null = tl.load(
                    dnull_prob + flat_th,
                    mask=matched,
                    other=0.0,
                ).to(tl.float32)
            else:
                p_null = tl.zeros((BLOCK_N,), tl.float32)
                dp_null = tl.zeros((BLOCK_N,), tl.float32)
            correction = tl.sum(p * dp, axis=1) + p_null * dp_null

            p_slot = tl.load(
                weights + selection,
                mask=matched,
                other=0.0,
            ).to(tl.float32)
            dp_slot = tl.load(
                dweights + selection,
                mask=matched,
                other=0.0,
            ).to(tl.float32)
            dz = tl.where(
                matched,
                p_slot * (dp_slot - correction) * ROUTE_SCALE,
                0.0,
            )
            q = tl.load(
                route_q + flat_th[:, None] * D + offs_d[None, :],
                mask=matched[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            dk += tl.sum(dz[:, None] * q, axis=0)
            pos += GROUP_SPLITS * BLOCK_N

        tl.atomic_add(
            dk_acc + ((batch * M + summary) * D + offs_d),
            dk,
            mask=d_mask,
        )

else:
    _small_m_scan_all_grouped_dk_kernel = None


def _validate_inputs(
    *,
    route_q: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_valid: torch.Tensor,
    dtopk_weights: torch.Tensor,
    summary_count: int,
    null_prob: torch.Tensor | None,
    dnull_prob: torch.Tensor | None,
) -> tuple[int, bool]:
    if triton is None or _small_m_scan_all_grouped_dk_kernel is None:
        raise RuntimeError("summary-grouped dRouteK requires Triton.")
    if not route_q.is_cuda:
        raise ValueError("summary-grouped dRouteK requires CUDA tensors.")
    if route_q.dtype != torch.bfloat16:
        raise ValueError("summary-grouped dRouteK requires BF16 route_q.")
    if route_q.ndim != 4:
        raise ValueError("route_q must have shape [1,T,6,64].")
    batch, text_len, heads, route_dim = map(int, route_q.shape)
    if (batch, heads, route_dim) != (
        FIXED_BATCH,
        FIXED_HEADS,
        FIXED_ROUTE_DIM,
    ):
        raise ValueError("route_q must have exact production shape [1,T,6,64].")
    route_shape = (batch, text_len, heads, FIXED_TOPK)
    if tuple(topk_indices.shape) != route_shape:
        raise ValueError("topk_indices must have shape [1,T,6,4].")
    if tuple(topk_weights.shape) != route_shape:
        raise ValueError("topk_weights must have shape [1,T,6,4].")
    if tuple(topk_valid.shape) != route_shape:
        raise ValueError("topk_valid must have shape [1,T,6,4].")
    if tuple(dtopk_weights.shape) != route_shape:
        raise ValueError("dtopk_weights must have shape [1,T,6,4].")
    if topk_indices.dtype != torch.int32:
        raise ValueError("topk_indices must be int32.")
    if topk_weights.dtype != torch.float32:
        raise ValueError("topk_weights must be float32.")
    if dtopk_weights.dtype != torch.float32:
        raise ValueError("dtopk_weights must be float32.")
    if topk_valid.dtype != torch.bool:
        raise ValueError("topk_valid must be bool.")
    tensors = (
        topk_indices,
        topk_weights,
        topk_valid,
        dtopk_weights,
    )
    if any(tensor.device != route_q.device for tensor in tensors):
        raise ValueError("all summary-grouped dRouteK tensors must share one device.")
    if any(not tensor.is_contiguous() for tensor in (route_q,) + tensors):
        raise ValueError("all summary-grouped dRouteK tensors must be contiguous.")
    if not 1 <= int(summary_count) <= MAX_SUMMARIES:
        raise ValueError(
            f"summary_count must be in [1, {MAX_SUMMARIES}], "
            f"got {summary_count}."
        )

    has_null = null_prob is not None
    if has_null != (dnull_prob is not None):
        raise ValueError("null_prob and dnull_prob must either both be set or both be None.")
    if has_null:
        null_shape = (batch, text_len, heads, 1)
        if tuple(null_prob.shape) != null_shape or tuple(dnull_prob.shape) != null_shape:
            raise ValueError("null_prob and dnull_prob must have shape [1,T,6,1].")
        if null_prob.dtype != torch.float32 or dnull_prob.dtype != torch.float32:
            raise ValueError("null_prob and dnull_prob must be float32.")
        if null_prob.device != route_q.device or dnull_prob.device != route_q.device:
            raise ValueError("null tensors must share route_q's device.")
        if not null_prob.is_contiguous() or not dnull_prob.is_contiguous():
            raise ValueError("null tensors must be contiguous.")
    return text_len, has_null


def hsd_route_topk_grouped_dk(
    *,
    route_q: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_valid: torch.Tensor,
    dtopk_weights: torch.Tensor,
    summary_count: int,
    route_scale: float,
    null_prob: torch.Tensor | None,
    dnull_prob: torch.Tensor | None,
    force: bool,
    group_splits: int = DEFAULT_SPLITS,
    block_n: int = DEFAULT_BLOCK_N,
    num_warps: int = 4,
    return_fp32: bool = False,
) -> torch.Tensor:
    """Compute shared-key dRouteK with a no-CSR small-M scan-all schedule.

    This operator computes only ``dRouteK``.  Existing dRouteQ/dNull paths stay
    unchanged.  ``force=True`` is mandatory; unsupported geometry raises and
    never dispatches another implementation.
    """

    if force is not True:
        raise ValueError(
            "summary-grouped dRouteK is experimental and requires force=True."
        )
    text_len, has_null = _validate_inputs(
        route_q=route_q,
        topk_indices=topk_indices,
        topk_weights=topk_weights,
        topk_valid=topk_valid,
        dtopk_weights=dtopk_weights,
        summary_count=int(summary_count),
        null_prob=null_prob,
        dnull_prob=dnull_prob,
    )
    if group_splits not in (4, 8, 16, 32, 64):
        raise ValueError("group_splits must be one of 4, 8, 16, 32, 64.")
    if block_n not in (32, 64, 128):
        raise ValueError("block_n must be one of 32, 64, 128.")
    if num_warps not in (4, 8):
        raise ValueError("num_warps must be 4 or 8.")

    empty_null = topk_weights.new_empty(0)
    dk_acc = torch.zeros(
        FIXED_BATCH,
        int(summary_count),
        1,
        FIXED_ROUTE_DIM,
        device=route_q.device,
        dtype=torch.float32,
    )
    grid = (
        triton.cdiv(FIXED_ROUTE_DIM, FIXED_ROUTE_DIM),
        FIXED_BATCH * int(summary_count),
        int(group_splits),
    )
    _small_m_scan_all_grouped_dk_kernel[grid](
        route_q,
        topk_indices,
        topk_weights,
        topk_valid,
        null_prob if has_null else empty_null,
        dtopk_weights,
        dnull_prob if has_null else empty_null,
        dk_acc,
        T=text_len,
        M=int(summary_count),
        H=FIXED_HEADS,
        D=FIXED_ROUTE_DIM,
        TOPK=FIXED_TOPK,
        ROUTE_SCALE=float(route_scale),
        HAS_NULL=bool(has_null),
        BLOCK_N=int(block_n),
        BLOCK_D=FIXED_ROUTE_DIM,
        GROUP_SPLITS=int(group_splits),
        num_warps=int(num_warps),
        num_stages=1,
    )
    return dk_acc if return_fp32 else dk_acc.to(dtype=route_q.dtype)


def hsd_route_topk_grouped_dk_reference(
    *,
    route_q: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_valid: torch.Tensor,
    dtopk_weights: torch.Tensor,
    summary_count: int,
    route_scale: float,
    null_prob: torch.Tensor | None,
    dnull_prob: torch.Tensor | None,
    return_fp32: bool = False,
) -> torch.Tensor:
    """Small explicit reference for tests and diagnostics."""

    p_null = 0.0 if null_prob is None else null_prob.float()
    dp_null = 0.0 if dnull_prob is None else dnull_prob.float()
    correction = (
        (topk_weights.float() * dtopk_weights.float()).sum(dim=-1, keepdim=True)
        + p_null * dp_null
    )
    dz = (
        topk_weights.float()
        * (dtopk_weights.float() - correction)
        * topk_valid
        * float(route_scale)
    )
    q = route_q.float().unsqueeze(-2)
    output = torch.zeros(
        FIXED_BATCH,
        int(summary_count),
        1,
        FIXED_ROUTE_DIM,
        device=route_q.device,
        dtype=torch.float32,
    )
    safe_indices = topk_indices.clamp(0, int(summary_count) - 1)
    for summary in range(int(summary_count)):
        selected = topk_valid & (safe_indices == summary)
        output[:, summary, 0] = (
            dz.unsqueeze(-1) * q * selected.unsqueeze(-1)
        ).sum(dim=(1, 2, 3))
    return output if return_fp32 else output.to(dtype=route_q.dtype)


def _validate_skip_dk_inputs(
    *,
    route_q: torch.Tensor,
    route_k: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_valid: torch.Tensor,
    dtopk_weights: torch.Tensor,
    null_prob: torch.Tensor | None,
    dnull_prob: torch.Tensor | None,
) -> tuple[int, int, bool]:
    if triton is None or _route_topk_bwd_kernel is None:
        raise RuntimeError("skip-dK route backward requires Triton.")
    if not route_q.is_cuda:
        raise ValueError("skip-dK route backward requires CUDA tensors.")
    if route_q.dtype != torch.bfloat16 or route_k.dtype != route_q.dtype:
        raise ValueError("skip-dK route backward requires BF16 route_q and route_k.")
    if route_q.ndim != 4 or route_k.ndim != 4:
        raise ValueError("route_q and route_k must be rank-four tensors.")
    batch, text_len, query_heads, route_dim = map(int, route_q.shape)
    if (batch, query_heads, route_dim) != (
        FIXED_BATCH,
        FIXED_HEADS,
        FIXED_ROUTE_DIM,
    ):
        raise ValueError("route_q must have exact production shape [1,T,6,64].")
    if (
        int(route_k.shape[0]) != FIXED_BATCH
        or int(route_k.shape[2]) != 1
        or int(route_k.shape[3]) != FIXED_ROUTE_DIM
    ):
        raise ValueError("route_k must have exact production shape [1,M,1,64].")
    summary_count = int(route_k.shape[1])
    if not 1 <= summary_count <= 256:
        raise ValueError("route_k summary count must be in [1, 256].")

    route_shape = (batch, text_len, FIXED_HEADS, FIXED_TOPK)
    routed = (
        ("topk_indices", topk_indices, torch.int32),
        ("topk_weights", topk_weights, torch.float32),
        ("topk_valid", topk_valid, torch.bool),
        ("dtopk_weights", dtopk_weights, torch.float32),
    )
    for name, tensor, dtype in routed:
        if tuple(tensor.shape) != route_shape:
            raise ValueError(f"{name} must have shape [1,T,6,4].")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}.")

    has_null = null_prob is not None
    if has_null != (dnull_prob is not None):
        raise ValueError("null_prob and dnull_prob must either both be set or both be None.")
    null_tensors: tuple[torch.Tensor, ...] = ()
    if has_null:
        null_shape = (batch, text_len, FIXED_HEADS, 1)
        if tuple(null_prob.shape) != null_shape or tuple(dnull_prob.shape) != null_shape:
            raise ValueError("null_prob and dnull_prob must have shape [1,T,6,1].")
        if null_prob.dtype != torch.float32 or dnull_prob.dtype != torch.float32:
            raise ValueError("null_prob and dnull_prob must be float32.")
        null_tensors = (null_prob, dnull_prob)

    tensors = (
        route_q,
        route_k,
        topk_indices,
        topk_weights,
        topk_valid,
        dtopk_weights,
    ) + null_tensors
    if any(tensor.device != route_q.device for tensor in tensors):
        raise ValueError("all skip-dK route backward tensors must share one device.")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all skip-dK route backward tensors must be contiguous.")
    return text_len, summary_count, has_null


def hsd_route_topk_backward_skip_dk(
    *,
    route_q: torch.Tensor,
    route_k: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_valid: torch.Tensor,
    dtopk_weights: torch.Tensor,
    route_scale: float,
    null_prob: torch.Tensor | None,
    dnull_prob: torch.Tensor | None,
    null_dtype: torch.dtype,
    force: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the production route-backward kernel for dQ/dNull only.

    This experimental entry point is deliberately separate from autograd.
    ``force=True`` is mandatory, unsupported inputs raise, and the launch uses
    ``COMPUTE_DK=False`` so no dK accumulator is allocated or written.
    """

    if force is not True:
        raise ValueError("skip-dK route backward is experimental and requires force=True.")
    text_len, summary_count, has_null = _validate_skip_dk_inputs(
        route_q=route_q,
        route_k=route_k,
        topk_indices=topk_indices,
        topk_weights=topk_weights,
        topk_valid=topk_valid,
        dtopk_weights=dtopk_weights,
        null_prob=null_prob,
        dnull_prob=dnull_prob,
    )
    if has_null and null_dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ):
        raise ValueError("null_dtype must be bfloat16, float16, or float32.")

    direct_input_grads, block_t, num_warps = _route_backward_launch_config()
    direct_dq = bool(direct_input_grads)
    dq_acc = (
        torch.empty_like(route_q)
        if direct_dq
        else torch.zeros_like(route_q, dtype=torch.float32)
    )
    direct_dnull = bool(direct_input_grads and has_null)
    dnull_acc = torch.empty(
        FIXED_BATCH,
        text_len,
        FIXED_HEADS,
        1 if has_null else 0,
        device=route_q.device,
        dtype=null_dtype if direct_dnull else torch.float32,
    )
    empty_null = topk_weights.new_empty(0)
    block_d = FIXED_ROUTE_DIM
    _route_topk_bwd_kernel[
        (
            triton.cdiv(text_len, block_t),
            FIXED_BATCH * FIXED_HEADS,
            triton.cdiv(FIXED_ROUTE_DIM, block_d),
        )
    ](
        route_q,
        route_k,
        topk_indices,
        topk_weights,
        topk_valid,
        null_prob if has_null else empty_null,
        dtopk_weights,
        dnull_prob if has_null else empty_null,
        dq_acc,
        # COMPUTE_DK=False makes this pointer dead at Triton specialization
        # time.  Reusing an existing input avoids allocating a dK placeholder.
        route_k,
        dnull_acc,
        FIXED_BATCH * text_len * FIXED_HEADS,
        text_len,
        summary_count,
        FIXED_HEADS,
        FIXED_HEADS,
        1,
        FIXED_ROUTE_DIM,
        FIXED_TOPK,
        float(route_scale),
        bool(has_null),
        COMPUTE_DK=False,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    dq = dq_acc if direct_dq else dq_acc.to(dtype=route_q.dtype)
    if direct_dnull:
        dnull = dnull_acc
    elif has_null:
        dnull = dnull_acc.to(dtype=null_dtype)
    else:
        dnull = None
    return dq, dnull


def build_route_bwd_grouped_dk_benchmark_calls(
    *,
    route_q: torch.Tensor,
    route_k: torch.Tensor,
    null_logits: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    summary_cu_seqlens: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    null_prob: torch.Tensor,
    dweights: torch.Tensor,
    dnull: torch.Tensor,
    current_gradients: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    value_heads: int,
    topk: int,
    start_index: int,
    route_scale: float,
    group_splits: int = DEFAULT_SPLITS,
    block_n: int = DEFAULT_BLOCK_N,
    num_warps: int = 4,
) -> dict[str, object]:
    """Build opt-in benchmark calls for skip-dK plus summary-grouped dK."""

    del boundary_counts, text_cu_seqlens, summary_cu_seqlens
    if (int(value_heads), int(topk), int(start_index)) != (
        FIXED_HEADS,
        FIXED_TOPK,
        0,
    ):
        raise ValueError("prototype requires value_heads=6, topk=4, start_index=0.")
    if tuple(null_logits.shape) != tuple(null_prob.shape):
        raise ValueError("null_logits and null_prob must have the same shape.")
    if len(current_gradients) != 3 or any(
        gradient is None for gradient in current_gradients
    ):
        raise ValueError("current_gradients must contain dQ, dK, and dNull.")
    summary_count = int(route_k.shape[1])
    if summary_count > MAX_SUMMARIES:
        raise ValueError(
            f"grouped-dK prototype supports at most {MAX_SUMMARIES} summaries."
        )

    def skip_dk_floor() -> tuple[torch.Tensor, torch.Tensor | None]:
        return hsd_route_topk_backward_skip_dk(
            route_q=route_q,
            route_k=route_k,
            topk_indices=indices,
            topk_weights=weights,
            topk_valid=valid,
            dtopk_weights=dweights,
            route_scale=route_scale,
            null_prob=null_prob,
            dnull_prob=dnull,
            null_dtype=null_logits.dtype,
            force=True,
        )

    def grouped_dk() -> torch.Tensor:
        return hsd_route_topk_grouped_dk(
            route_q=route_q,
            topk_indices=indices,
            topk_weights=weights,
            topk_valid=valid,
            dtopk_weights=dweights,
            summary_count=summary_count,
            route_scale=route_scale,
            null_prob=null_prob,
            dnull_prob=dnull,
            group_splits=group_splits,
            block_n=block_n,
            num_warps=num_warps,
            force=True,
        )

    def validate() -> dict[str, float]:
        actual_dq, actual_dnull = skip_dk_floor()
        actual_dk = grouped_dk()
        expected_dq, expected_dk, expected_dnull = current_gradients
        torch.testing.assert_close(actual_dq, expected_dq, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(actual_dk, expected_dk, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(
            actual_dnull,
            expected_dnull,
            rtol=2e-2,
            atol=2e-2,
        )
        return {
            "dq_max_abs_error": float(
                (actual_dq.float() - expected_dq.float()).abs().max().item()
            ),
            "dk_max_abs_error": float(
                (actual_dk.float() - expected_dk.float()).abs().max().item()
            ),
            "dnull_max_abs_error": float(
                (actual_dnull.float() - expected_dnull.float()).abs().max().item()
            ),
        }

    return {
        "skip_dk_floor": skip_dk_floor,
        "grouped_dk": grouped_dk,
        "validate": validate,
    }


__all__ = [
    "build_route_bwd_grouped_dk_benchmark_calls",
    "hsd_route_topk_backward_skip_dk",
    "hsd_route_topk_grouped_dk",
    "hsd_route_topk_grouped_dk_reference",
]
