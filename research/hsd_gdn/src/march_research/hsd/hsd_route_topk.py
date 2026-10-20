from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional CUDA/Triton dependency.
    triton = None
    tl = None


_SUPPORTED_TOPK = {1, 2, 4, 8, 16}

# H200 production defaults selected by the paired 10-warmup/50-repetition
# Q6/K1/R64 route sweep.  All remain explicit environment-controlled launch
# choices so the pre-optimization schedule is a one-line rollback:
#
#   HSD_TOPK_ROUTE_DIRECT_INPUT_GRADS=0
#   HSD_TOPK_ROUTE_BLOCK_T=16
#   HSD_TOPK_ROUTE_BWD_BLOCK_T=1
#
# Both forward and backward use four warps in the selected profile.
_ROUTE_DIRECT_INPUT_GRADS_DEFAULT = True
_ROUTE_FWD_BLOCK_T_DEFAULT = 32
_ROUTE_FWD_NUM_WARPS_DEFAULT = 4
_ROUTE_BWD_BLOCK_T_DEFAULT = 8
_ROUTE_BWD_NUM_WARPS_DEFAULT = 4
_ROUTE_GROUPED_DK_MAX_M_DEFAULT = 16
_ROUTE_GROUPED_DK_GROUP_SPLITS_DEFAULT = 16
_ROUTE_GROUPED_DK_BLOCK_N_DEFAULT = 128
_ROUTE_GROUPED_DK_NUM_WARPS_DEFAULT = 4


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, allowed: set[int]) -> int:
    raw = os.environ.get(name, "").strip()
    value = default if not raw else int(raw)
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value}.")
    return value


def _route_forward_launch_config() -> tuple[int, int]:
    """Return the environment-resolved forward token tile and warp count."""
    return (
        _env_int(
            "HSD_TOPK_ROUTE_BLOCK_T",
            _ROUTE_FWD_BLOCK_T_DEFAULT,
            allowed={16, 32},
        ),
        _env_int(
            "HSD_TOPK_ROUTE_NUM_WARPS",
            _ROUTE_FWD_NUM_WARPS_DEFAULT,
            allowed={4, 8},
        ),
    )


def _route_backward_launch_config() -> tuple[bool, int, int]:
    """Return direct-gradient, token-tile, and warp launch choices."""
    return (
        _env_bool(
            "HSD_TOPK_ROUTE_DIRECT_INPUT_GRADS",
            _ROUTE_DIRECT_INPUT_GRADS_DEFAULT,
        ),
        _env_int(
            "HSD_TOPK_ROUTE_BWD_BLOCK_T",
            _ROUTE_BWD_BLOCK_T_DEFAULT,
            allowed={1, 2, 4, 8},
        ),
        _env_int(
            "HSD_TOPK_ROUTE_BWD_NUM_WARPS",
            _ROUTE_BWD_NUM_WARPS_DEFAULT,
            allowed={4, 8},
        ),
    )


def _route_grouped_dk_launch_config() -> tuple[bool, int, int, int, int]:
    """Return strictness, maximum M, and grouped-dK launch choices."""
    return (
        _env_bool("HSD_TOPK_ROUTE_GROUPED_DK_STRICT", False),
        _env_int(
            "HSD_TOPK_ROUTE_GROUPED_DK_MAX_M",
            _ROUTE_GROUPED_DK_MAX_M_DEFAULT,
            allowed={8, 16, 32},
        ),
        _env_int(
            "HSD_TOPK_ROUTE_GROUPED_DK_GROUP_SPLITS",
            _ROUTE_GROUPED_DK_GROUP_SPLITS_DEFAULT,
            allowed={4, 8, 16, 32, 64},
        ),
        _env_int(
            "HSD_TOPK_ROUTE_GROUPED_DK_BLOCK_N",
            _ROUTE_GROUPED_DK_BLOCK_N_DEFAULT,
            allowed={32, 64, 128},
        ),
        _env_int(
            "HSD_TOPK_ROUTE_GROUPED_DK_NUM_WARPS",
            _ROUTE_GROUPED_DK_NUM_WARPS_DEFAULT,
            allowed={4, 8},
        ),
    )


def _supports_route_grouped_dk(
    route_q: torch.Tensor,
    route_k: torch.Tensor,
    *,
    value_heads: int,
    topk: int,
    max_summaries: int,
) -> bool:
    """Match only the validated B1 BF16 Q6/K1/R64/K4 small-M geometry."""
    return bool(
        route_q.dtype == torch.bfloat16
        and route_k.dtype == torch.bfloat16
        and route_q.ndim == 4
        and route_k.ndim == 4
        and tuple(route_q.shape[:1]) == (1,)
        and tuple(route_q.shape[2:]) == (6, 64)
        and tuple(route_k.shape[:1]) == (1,)
        and tuple(route_k.shape[2:]) == (1, 64)
        and int(value_heads) == 6
        and int(topk) == 4
        and 1 <= int(route_k.shape[1]) <= int(max_summaries)
    )


if triton is not None:

    @triton.jit
    def _route_topk_fwd_kernel(
        route_q,
        route_k,
        boundaries,
        token_doc,
        summary_doc,
        null_logits,
        out_indices,
        out_weights,
        out_valid,
        out_null_prob,
        T: tl.constexpr,
        M,
        H: tl.constexpr,
        QH: tl.constexpr,
        KH: tl.constexpr,
        D: tl.constexpr,
        TOPK: tl.constexpr,
        TOPK_BLOCK: tl.constexpr,
        START_INDEX: tl.constexpr,
        ROUTE_SCALE: tl.constexpr,
        HAS_NULL: tl.constexpr,
        PACKED: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        tile_t = tl.program_id(0)
        flat_bh = tl.program_id(1)
        batch = flat_bh // H
        head = flat_bh % H
        q_head = 0 if QH == 1 else head
        k_head = 0 if KH == 1 else head

        offs_t = tile_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        t_mask = offs_t < T
        m_mask = offs_m < M
        d_mask = offs_d < D

        q = tl.load(
            route_q + (((batch * T + offs_t[:, None]) * QH + q_head) * D + offs_d[None, :]),
            mask=t_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        k = tl.load(
            route_k + (((batch * M + offs_m[:, None]) * KH + k_head) * D + offs_d[None, :]),
            mask=m_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        logits = tl.dot(q, tl.trans(k)).to(tl.float32) * ROUTE_SCALE
        boundary = tl.load(boundaries + offs_m, mask=m_mask, other=T + START_INDEX + 1).to(tl.int32)
        causal = boundary[None, :] <= (START_INDEX + offs_t[:, None])
        route_valid = t_mask[:, None] & m_mask[None, :] & causal
        if PACKED:
            t_doc = tl.load(token_doc + offs_t, mask=t_mask, other=-1).to(tl.int32)
            m_doc = tl.load(summary_doc + offs_m, mask=m_mask, other=-2).to(tl.int32)
            route_valid &= t_doc[:, None] == m_doc[None, :]
        logits = tl.where(route_valid, logits, -float("inf"))

        offs_j = tl.arange(0, TOPK_BLOCK)
        selected_logits = tl.full((BLOCK_T, TOPK_BLOCK), -float("inf"), tl.float32)
        selected_indices = tl.zeros((BLOCK_T, TOPK_BLOCK), tl.int32)
        selected_valid = tl.zeros((BLOCK_T, TOPK_BLOCK), tl.int1)
        for slot in range(TOPK):
            value = tl.max(logits, axis=1)
            index = tl.argmax(logits, axis=1).to(tl.int32)
            is_valid = t_mask & (value != -float("inf"))
            slot_mask = offs_j[None, :] == slot
            selected_logits = tl.where(slot_mask, value[:, None], selected_logits)
            selected_indices = tl.where(slot_mask, index[:, None], selected_indices)
            selected_valid = tl.where(slot_mask, is_valid[:, None], selected_valid)
            logits = tl.where(offs_m[None, :] == index[:, None], -float("inf"), logits)

        has_summary = tl.max(selected_valid.to(tl.int32), axis=1) != 0
        max_summary = tl.max(selected_logits, axis=1)
        if HAS_NULL:
            null_value = tl.load(
                null_logits + ((batch * T + offs_t) * H + head),
                mask=t_mask,
                other=-float("inf"),
            ).to(tl.float32)
            max_value = tl.maximum(max_summary, null_value)
            exp_null = tl.exp(null_value - max_value)
        else:
            null_value = tl.zeros((BLOCK_T,), tl.float32)
            max_value = tl.where(has_summary, max_summary, 0.0)
            exp_null = tl.zeros((BLOCK_T,), tl.float32)
        exp_selected = tl.where(
            selected_valid,
            tl.exp(selected_logits - max_value[:, None]),
            0.0,
        )
        denominator = tl.sum(exp_selected, axis=1) + exp_null
        denominator = tl.where(denominator > 0.0, denominator, 1.0)
        selected_weights = exp_selected / denominator[:, None]
        null_prob = exp_null / denominator

        out_base = ((batch * T + offs_t[:, None]) * H + head) * TOPK + offs_j[None, :]
        output_mask = t_mask[:, None] & (offs_j[None, :] < TOPK)
        tl.store(out_indices + out_base, selected_indices, mask=output_mask)
        tl.store(out_weights + out_base, selected_weights, mask=output_mask)
        tl.store(out_valid + out_base, selected_valid, mask=output_mask)
        if HAS_NULL:
            tl.store(
                out_null_prob + ((batch * T + offs_t) * H + head),
                null_prob,
                mask=t_mask,
            )

    @triton.jit
    def _route_topk_bwd_kernel(
        route_q,
        route_k,
        indices,
        weights,
        valid,
        null_prob,
        dweights,
        dnull_prob,
        dq_acc,
        dk_acc,
        dnull_acc,
        BTH: tl.constexpr,
        T: tl.constexpr,
        M,
        H: tl.constexpr,
        QH: tl.constexpr,
        KH: tl.constexpr,
        D: tl.constexpr,
        TOPK: tl.constexpr,
        ROUTE_SCALE: tl.constexpr,
        HAS_NULL: tl.constexpr,
        COMPUTE_DK: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        tile_t = tl.program_id(0)
        flat_bh = tl.program_id(1)
        i_d = tl.program_id(2)
        batch = flat_bh // H
        head = flat_bh % H
        q_head = 0 if QH == 1 else head
        k_head = 0 if KH == 1 else head
        token = tile_t * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = token < T
        flat_th = (batch * T + token) * H + head
        offs_d = i_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        offs_j = tl.arange(0, TOPK)
        selection = flat_th[:, None] * TOPK + offs_j[None, :]
        p = tl.load(
            weights + selection,
            mask=t_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        dp = tl.load(
            dweights + selection,
            mask=t_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        if HAS_NULL:
            p_null = tl.load(null_prob + flat_th, mask=t_mask, other=0.0).to(tl.float32)
            dp_null = tl.load(dnull_prob + flat_th, mask=t_mask, other=0.0).to(tl.float32)
        else:
            p_null = tl.zeros((BLOCK_T,), tl.float32)
            dp_null = tl.zeros((BLOCK_T,), tl.float32)
        correction = tl.sum(p * dp, axis=1) + p_null * dp_null
        dnull = p_null * (dp_null - correction)

        if COMPUTE_DK:
            q = tl.load(
                route_q
                + (((batch * T + token[:, None]) * QH + q_head) * D + offs_d[None, :]),
                mask=t_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
        dq = tl.zeros((BLOCK_T, BLOCK_D), tl.float32)
        for slot in range(TOPK):
            selected = flat_th * TOPK + slot
            summary = tl.load(indices + selected, mask=t_mask, other=0).to(tl.int32)
            slot_valid = tl.load(valid + selected, mask=t_mask, other=0).to(tl.int1)
            summary = tl.where(slot_valid, summary, 0)
            p_slot = tl.load(weights + selected, mask=t_mask, other=0.0).to(tl.float32)
            dp_slot = tl.load(dweights + selected, mask=t_mask, other=0.0).to(tl.float32)
            contribution = tl.where(
                t_mask & slot_valid,
                p_slot * (dp_slot - correction) * ROUTE_SCALE,
                0.0,
            )
            key = tl.load(
                route_k
                + (((batch * M + summary[:, None]) * KH + k_head) * D + offs_d[None, :]),
                mask=t_mask[:, None] & slot_valid[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            dq += contribution[:, None] * key
            if COMPUTE_DK:
                tl.atomic_add(
                    dk_acc
                    + (((batch * M + summary[:, None]) * KH + k_head) * D + offs_d[None, :]),
                    contribution[:, None] * q,
                    mask=t_mask[:, None] & slot_valid[:, None] & d_mask[None, :],
                )
        if QH == 1 and H != 1:
            tl.atomic_add(
                dq_acc
                + (((batch * T + token[:, None]) * QH + q_head) * D + offs_d[None, :]),
                dq,
                mask=t_mask[:, None] & d_mask[None, :],
            )
        else:
            tl.store(
                dq_acc
                + (((batch * T + token[:, None]) * QH + q_head) * D + offs_d[None, :]),
                dq,
                mask=t_mask[:, None] & d_mask[None, :],
            )
        if HAS_NULL and i_d == 0:
            tl.store(dnull_acc + flat_th, dnull, mask=t_mask)

else:
    _route_topk_fwd_kernel = None
    _route_topk_bwd_kernel = None


def _supports_fused_route_topk(
    route_q: torch.Tensor,
    route_k: torch.Tensor,
    null_logits: torch.Tensor | None,
    *,
    value_heads: int,
    topk: int,
) -> bool:
    if triton is None or _route_topk_fwd_kernel is None or not route_q.is_cuda:
        return False
    if route_q.dtype not in (torch.bfloat16, torch.float16) or route_k.dtype != route_q.dtype:
        return False
    if route_q.ndim != 4 or route_k.ndim != 4 or route_q.shape[0] != route_k.shape[0]:
        return False
    if route_q.shape[-1] != route_k.shape[-1] or not (16 <= int(route_q.shape[-1]) <= 256):
        return False
    if int(route_k.shape[1]) <= 0 or int(route_k.shape[1]) > 256 or topk not in _SUPPORTED_TOPK:
        return False
    if int(route_q.shape[2]) not in (1, value_heads) or int(route_k.shape[2]) not in (1, value_heads):
        return False
    if null_logits is not None and tuple(null_logits.shape) != (
        int(route_q.shape[0]), int(route_q.shape[1]), value_heads, 1
    ):
        return False
    return True


class _HSDFusedRouteTopKFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        route_q: torch.Tensor,
        route_k: torch.Tensor,
        boundaries: torch.Tensor,
        token_doc: torch.Tensor,
        summary_doc: torch.Tensor,
        null_logits: torch.Tensor,
        value_heads: int,
        topk: int,
        start_index: int,
        route_scale: float,
        packed: bool,
        has_null: bool,
    ):
        q = route_q.contiguous()
        k = route_k.contiguous()
        batch, text_len, query_heads, route_dim = q.shape
        summary_count = int(k.shape[1])
        key_heads = int(k.shape[2])
        indices = torch.empty(batch, text_len, value_heads, topk, device=q.device, dtype=torch.int32)
        weights = torch.empty(batch, text_len, value_heads, topk, device=q.device, dtype=torch.float32)
        valid = torch.empty(batch, text_len, value_heads, topk, device=q.device, dtype=torch.bool)
        null_prob = torch.empty(
            batch, text_len, value_heads, 1 if has_null else 0, device=q.device, dtype=torch.float32
        )
        # The selected H200 production profile amortizes each route-key load
        # across 32 query tokens.  The prior BLOCK_T=16 schedule remains an
        # explicit environment rollback.
        block_t, route_num_warps = _route_forward_launch_config()
        block_m = triton.next_power_of_2(summary_count)
        block_d = max(16, triton.next_power_of_2(route_dim))
        topk_block = triton.next_power_of_2(topk)
        _route_topk_fwd_kernel[(triton.cdiv(text_len, block_t), batch * value_heads)](
            q,
            k,
            boundaries,
            token_doc,
            summary_doc,
            null_logits,
            indices,
            weights,
            valid,
            null_prob,
            text_len,
            summary_count,
            value_heads,
            query_heads,
            key_heads,
            route_dim,
            topk,
            topk_block,
            int(start_index),
            float(route_scale),
            bool(has_null),
            bool(packed),
            BLOCK_T=block_t,
            BLOCK_M=block_m,
            BLOCK_D=block_d,
            num_warps=route_num_warps,
            num_stages=1,
        )
        ctx.value_heads = int(value_heads)
        ctx.topk = int(topk)
        ctx.route_scale = float(route_scale)
        ctx.has_null = bool(has_null)
        ctx.null_dtype = null_logits.dtype
        ctx.save_for_backward(q, k, indices, weights, valid, null_prob)
        ctx.mark_non_differentiable(indices, valid)
        return indices, weights, valid, null_prob

    @staticmethod
    def backward(ctx, dindices, dweights, dvalid, dnull_prob):
        route_q, route_k, indices, weights, valid, null_prob = ctx.saved_tensors
        batch, text_len, query_heads, route_dim = route_q.shape
        summary_count = int(route_k.shape[1])
        key_heads = int(route_k.shape[2])
        value_heads = int(ctx.value_heads)
        topk = int(ctx.topk)
        dweights = torch.zeros_like(weights) if dweights is None else dweights.contiguous().float()
        if ctx.has_null:
            dnull_prob = (
                torch.zeros_like(null_prob) if dnull_prob is None else dnull_prob.contiguous().float()
            )
        else:
            dnull_prob = weights.new_empty(0)
        # The summary-grouped dK path is a deliberately narrow production
        # opt-in.  With the flag absent/off, this branch resolves no tuning
        # variables, imports no prototype module, and leaves the established
        # fused backward launch and allocations below unchanged.
        if _env_bool("HSD_TOPK_ROUTE_GROUPED_DK", False):
            (
                grouped_dk_strict,
                grouped_dk_max_m,
                grouped_dk_splits,
                grouped_dk_block_n,
                grouped_dk_num_warps,
            ) = _route_grouped_dk_launch_config()
            if _supports_route_grouped_dk(
                route_q,
                route_k,
                value_heads=value_heads,
                topk=topk,
                max_summaries=grouped_dk_max_m,
            ):
                try:
                    # Local import is required: the prototype reuses the
                    # production backward kernel and launch configuration.
                    from .hsd_route_topk_grouped_dk import (
                        hsd_route_topk_backward_skip_dk,
                        hsd_route_topk_grouped_dk,
                    )

                    grouped_null_prob = null_prob if ctx.has_null else None
                    grouped_dnull_prob = dnull_prob if ctx.has_null else None
                    dq, dnull = hsd_route_topk_backward_skip_dk(
                        route_q=route_q,
                        route_k=route_k,
                        topk_indices=indices,
                        topk_weights=weights,
                        topk_valid=valid,
                        dtopk_weights=dweights,
                        route_scale=ctx.route_scale,
                        null_prob=grouped_null_prob,
                        dnull_prob=grouped_dnull_prob,
                        null_dtype=ctx.null_dtype,
                        force=True,
                    )
                    dk = hsd_route_topk_grouped_dk(
                        route_q=route_q,
                        topk_indices=indices,
                        topk_weights=weights,
                        topk_valid=valid,
                        dtopk_weights=dweights,
                        summary_count=summary_count,
                        route_scale=ctx.route_scale,
                        null_prob=grouped_null_prob,
                        dnull_prob=grouped_dnull_prob,
                        force=True,
                        group_splits=grouped_dk_splits,
                        block_n=grouped_dk_block_n,
                        num_warps=grouped_dk_num_warps,
                    )
                    return (
                        dq,
                        dk,
                        None,
                        None,
                        None,
                        dnull,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                except Exception:
                    if grouped_dk_strict:
                        raise
        # QH is restricted to 1 or H.  For the production QH=H layout every
        # dQ element has exactly one writer, so the kernel takes its tl.store
        # branch and can perform the sole FP32->input-dtype rounding directly.
        # QH=1 still needs an FP32 atomic reduction across value heads.
        (
            direct_input_grads,
            bwd_block_t,
            bwd_num_warps,
        ) = _route_backward_launch_config()
        direct_dq = bool(direct_input_grads and query_heads == value_heads)
        dq_acc = (
            torch.empty_like(route_q)
            if direct_dq
            else torch.zeros_like(route_q, dtype=torch.float32)
        )
        dk_acc = torch.zeros_like(route_k, dtype=torch.float32)
        # dNull has one writer (the i_d==0 program) per token/head, independent
        # of query/key head sharing, and therefore follows the same direct-store
        # rule without requiring an atomic accumulator.
        direct_dnull = bool(direct_input_grads and ctx.has_null)
        dnull_acc = torch.empty(
            batch, text_len, value_heads, 1 if ctx.has_null else 0,
            device=route_q.device,
            dtype=ctx.null_dtype if direct_dnull else torch.float32,
        )
        block_d = 64
        # BLOCK_T=8 is the selected production schedule.  BLOCK_T=1 exactly
        # preserves the prior one-token-per-CTA rollback, while all variants
        # retain identical per-token softmax and atomic dK semantics.
        bth = batch * text_len * value_heads
        _route_topk_bwd_kernel[
            (
                triton.cdiv(text_len, bwd_block_t),
                batch * value_heads,
                triton.cdiv(route_dim, block_d),
            )
        ](
            route_q,
            route_k,
            indices,
            weights,
            valid,
            null_prob,
            dweights,
            dnull_prob,
            dq_acc,
            dk_acc,
            dnull_acc,
            bth,
            text_len,
            summary_count,
            value_heads,
            query_heads,
            key_heads,
            route_dim,
            topk,
            ctx.route_scale,
            ctx.has_null,
            COMPUTE_DK=True,
            BLOCK_T=bwd_block_t,
            BLOCK_D=block_d,
            num_warps=bwd_num_warps,
        )
        dq = dq_acc if direct_dq else dq_acc.to(dtype=route_q.dtype)
        if direct_dnull:
            dnull = dnull_acc
        elif ctx.has_null:
            dnull = dnull_acc.to(dtype=ctx.null_dtype)
        else:
            dnull = None
        return (
            dq,
            dk_acc.to(dtype=route_k.dtype),
            None,
            None,
            None,
            dnull,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def hsd_fused_route_topk(
    *,
    route_q: torch.Tensor,
    route_k: torch.Tensor,
    boundary_counts: torch.Tensor,
    value_heads: int,
    topk: int,
    start_index: int,
    route_scale: float,
    null_logits: torch.Tensor | None,
    text_cu_seqlens: torch.Tensor | None = None,
    summary_cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
    """Fused dense-route logits, causal/document mask, hard top-k, and K+null softmax."""
    if not _supports_fused_route_topk(
        route_q, route_k, null_logits, value_heads=int(value_heads), topk=int(topk)
    ):
        return None
    if (text_cu_seqlens is None) != (summary_cu_seqlens is None):
        raise ValueError("Fused HSD top-k routing requires both packed cu_seqlens or neither.")
    packed = text_cu_seqlens is not None
    # A canonical two-entry cu_seqlens describes exactly one packed document.
    # In that case the document-equality mask is identically true, so building
    # token_doc/summary_doc with two arange + searchsorted + cast chains and
    # loading those arrays in every route CTA is redundant.  Treat the layout
    # as the mathematically equivalent non-packed single-document case.  The
    # real multi-document packed path remains unchanged.
    single_document_packed = bool(
        packed
        and _env_bool("HSD_TOPK_SINGLE_DOC_PACKED_FASTPATH", True)
        and int(route_q.shape[0]) == 1
        and int(text_cu_seqlens.numel()) == 2
        and int(summary_cu_seqlens.numel()) == 2
    )
    kernel_packed = packed and not single_document_packed
    if kernel_packed:
        text_cu = text_cu_seqlens.to(device=route_q.device, dtype=torch.long)
        summary_cu = summary_cu_seqlens.to(device=route_q.device, dtype=torch.long)
        token_index = torch.arange(route_q.shape[1], device=route_q.device, dtype=torch.long)
        summary_index = torch.arange(route_k.shape[1], device=route_q.device, dtype=torch.long)
        token_doc = torch.searchsorted(text_cu[1:], token_index, right=True).to(torch.int32)
        summary_doc = torch.searchsorted(summary_cu[1:], summary_index, right=True).to(torch.int32)
    else:
        token_doc = boundary_counts.new_empty(0, dtype=torch.int32)
        summary_doc = boundary_counts.new_empty(0, dtype=torch.int32)
    empty_null = route_q.new_empty(0)
    indices, weights, valid, null_prob = _HSDFusedRouteTopKFunction.apply(
        route_q,
        route_k,
        boundary_counts.to(device=route_q.device, dtype=torch.int32).contiguous(),
        token_doc,
        summary_doc,
        null_logits if null_logits is not None else empty_null,
        int(value_heads),
        int(topk),
        int(start_index),
        float(route_scale),
        bool(kernel_packed),
        bool(null_logits is not None),
    )
    return indices, weights, valid, null_prob if null_logits is not None else None


def fused_route_topk_enabled() -> bool:
    return _env_bool("HSD_TOPK_FUSED_ROUTE", True)


__all__ = ["fused_route_topk_enabled", "hsd_fused_route_topk"]
