from __future__ import annotations

import os
from typing import Callable, Optional

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional CUDA/Triton dependency.
    triton = None
    tl = None


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _route_momentum_coefficients(
    raw_weights: torch.Tensor,
    has_history: torch.Tensor,
    reset_mask: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if raw_weights.ndim != 4:
        raise ValueError(f"raw route weights must be [B,T,H,M], got {tuple(raw_weights.shape)}.")
    batch, text_len = raw_weights.shape[:2]
    if tuple(has_history.shape) != (batch, text_len):
        raise ValueError(
            f"has_history must have shape {(batch, text_len)}, got {tuple(has_history.shape)}."
        )
    if tuple(reset_mask.shape) != (batch, text_len):
        raise ValueError(f"reset_mask must have shape {(batch, text_len)}, got {tuple(reset_mask.shape)}.")
    active = has_history.to(device=raw_weights.device, dtype=torch.bool)
    reset = reset_mask.to(device=raw_weights.device, dtype=torch.bool)
    continue_mask = active & ~reset
    a = torch.where(
        continue_mask,
        torch.full_like(active, float(gamma), dtype=torch.float32),
        torch.zeros_like(active, dtype=torch.float32),
    ).view(batch, text_len, 1, 1)
    innovation_scale = torch.where(
        continue_mask,
        torch.full_like(active, 1.0 - float(gamma), dtype=torch.float32),
        torch.ones_like(active, dtype=torch.float32),
    ).view(batch, text_len, 1, 1)
    return a, innovation_scale


def route_weight_momentum_reference(
    raw_weights: torch.Tensor,
    has_history: torch.Tensor,
    reset_mask: torch.Tensor,
    gamma: float,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference token recurrence for dense route-weight momentum."""
    if not 0.0 < float(gamma) < 1.0:
        raise ValueError("route-weight momentum gamma must be in (0, 1).")
    raw = raw_weights.float()
    batch, text_len, heads, candidates = raw.shape
    if initial_state is None:
        state = raw.new_zeros(batch, heads, candidates)
    else:
        if tuple(initial_state.shape) != (batch, heads, candidates):
            raise ValueError(
                "initial route momentum state shape mismatch: "
                f"got {tuple(initial_state.shape)}, expected {(batch, heads, candidates)}."
            )
        state = initial_state.float()
    a, innovation_scale = _route_momentum_coefficients(raw, has_history, reset_mask, gamma)
    outputs = []
    for token_idx in range(text_len):
        state = a[:, token_idx] * state + innovation_scale[:, token_idx] * raw[:, token_idx]
        outputs.append(state)
    if outputs:
        smoothed = torch.stack(outputs, dim=1)
    else:
        smoothed = raw.new_empty(batch, 0, heads, candidates)
    return smoothed, state


def _route_weight_momentum_scan_cuda(
    innovation: torch.Tensor,
    a: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """Three-stage 256-token affine prefix scan used by forward and backward."""
    if triton is None or _route_momentum_local_scan_kernel is None:
        raise RuntimeError("CUDA route-weight momentum requires the Triton segmented-scan kernels.")
    batch, text_len, heads, candidates = innovation.shape
    if text_len == 0:
        return innovation.new_empty(batch, 0, heads, candidates, dtype=torch.float32)
    flat_candidates = heads * candidates
    block_t = 256
    block_c = 4
    num_blocks = triton.cdiv(text_len, block_t)
    block_n = _next_power_of_2(num_blocks)
    local_a = torch.empty(batch, text_len, device=innovation.device, dtype=torch.float32)
    block_a = torch.empty(batch, num_blocks, device=innovation.device, dtype=torch.float32)
    local_b = torch.empty_like(innovation, dtype=torch.float32)
    block_b = torch.empty(
        batch,
        num_blocks,
        flat_candidates,
        device=innovation.device,
        dtype=torch.float32,
    )
    candidate_blocks = triton.cdiv(flat_candidates, block_c)
    _route_momentum_local_scan_kernel[(candidate_blocks, num_blocks, batch)](
        innovation,
        a,
        local_a,
        local_b,
        block_a,
        block_b,
        text_len,
        flat_candidates,
        NUM_BLOCKS=num_blocks,
        BLOCK_T=block_t,
        BLOCK_C=block_c,
        num_warps=4,
    )
    block_initial = torch.empty_like(block_b)
    _route_momentum_block_scan_kernel[(candidate_blocks, batch)](
        block_a,
        block_b,
        initial_state,
        block_initial,
        flat_candidates,
        NUM_BLOCKS=num_blocks,
        BLOCK_N=block_n,
        BLOCK_C=block_c,
        num_warps=4,
    )
    _route_momentum_apply_prefix_kernel[(candidate_blocks, num_blocks, batch)](
        local_a,
        local_b,
        block_initial,
        text_len,
        flat_candidates,
        NUM_BLOCKS=num_blocks,
        BLOCK_T=block_t,
        BLOCK_C=block_c,
        num_warps=4,
    )
    return local_b


class _RouteWeightMomentumFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        raw_weights: torch.Tensor,
        has_history: torch.Tensor,
        reset_mask: torch.Tensor,
        initial_state: torch.Tensor,
        gamma: float,
        has_initial_state: bool,
    ):
        raw = raw_weights.float().contiguous()
        a, innovation_scale = _route_momentum_coefficients(raw, has_history, reset_mask, gamma)
        innovation = innovation_scale * raw
        smoothed = _route_weight_momentum_scan_cuda(
            innovation,
            a.squeeze(-1).squeeze(-1).contiguous(),
            initial_state.float().contiguous(),
        )
        final_state = smoothed[:, -1] if smoothed.shape[1] else initial_state.float()
        ctx.has_initial_state = bool(has_initial_state)
        ctx.save_for_backward(a, innovation_scale)
        return smoothed, final_state

    @staticmethod
    def backward(ctx, dsmoothed: torch.Tensor, dfinal_state: torch.Tensor | None):
        a, innovation_scale = ctx.saved_tensors
        grad = dsmoothed.float().contiguous()
        if dfinal_state is not None and grad.shape[1] > 0:
            grad = grad.clone()
            grad[:, -1] += dfinal_state.float()
        if grad.shape[1] == 0:
            dinitial = torch.zeros_like(dfinal_state) if ctx.has_initial_state and dfinal_state is not None else None
            return grad, None, None, dinitial, None, None
        next_a = torch.cat([a[:, 1:], torch.zeros_like(a[:, :1])], dim=1)
        total_grad = _route_weight_momentum_scan_cuda(
            grad.flip(1).contiguous(),
            next_a.squeeze(-1).squeeze(-1).flip(1).contiguous(),
            grad.new_zeros(grad.shape[0], grad.shape[2], grad.shape[3]),
        )
        total_grad = total_grad.flip(1)
        draw = total_grad * innovation_scale
        dinitial = a[:, 0] * total_grad[:, 0] if ctx.has_initial_state else None
        return draw, None, None, dinitial, None, None


def route_weight_momentum_scan(
    raw_weights: torch.Tensor,
    has_history: torch.Tensor,
    reset_mask: torch.Tensor,
    gamma: float,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parallel dense route-weight EMA with document/first-history resets."""
    gamma = float(gamma)
    if not 0.0 < gamma < 1.0:
        raise ValueError("route-weight momentum gamma must be in (0, 1).")
    expected = (raw_weights.shape[0], raw_weights.shape[2], raw_weights.shape[3])
    if initial_state is not None and tuple(initial_state.shape) != expected:
        raise ValueError(
            f"initial route momentum state shape mismatch: got {tuple(initial_state.shape)}, expected {expected}."
        )
    if raw_weights.shape[1] == 1:
        raw = raw_weights.float()
        a, innovation_scale = _route_momentum_coefficients(raw, has_history, reset_mask, gamma)
        state = raw.new_zeros(expected) if initial_state is None else initial_state.float()
        state = a[:, 0] * state + innovation_scale[:, 0] * raw[:, 0]
        return state.unsqueeze(1), state
    if not raw_weights.is_cuda:
        return route_weight_momentum_reference(raw_weights, has_history, reset_mask, gamma, initial_state)
    if initial_state is None:
        initial_arg = raw_weights.new_zeros(expected, dtype=torch.float32)
        has_initial = False
    else:
        initial_arg = initial_state.float().contiguous()
        has_initial = True
    return _RouteWeightMomentumFunction.apply(
        raw_weights,
        has_history,
        reset_mask,
        initial_arg,
        gamma,
        has_initial,
    )


def _history_read_packed_reference(
    *,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    states: torch.Tensor,
    summary_k: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    summary_cu_seqlens: torch.Tensor,
    null_logits: torch.Tensor | None,
    route_scale: float,
    read_scale: float,
    output_dtype: torch.dtype | None = None,
    debug_stats_callback: Optional[Callable[..., None]] = None,
    return_local_prob: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    text_cu_cpu = text_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
    summary_cu_cpu = summary_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
    outputs = []
    local_probs = []
    value_dim = states.shape[-1]
    for doc_idx in range(int(text_cu_cpu.numel()) - 1):
        text_start = int(text_cu_cpu[doc_idx].item())
        text_end = int(text_cu_cpu[doc_idx + 1].item())
        summary_start = int(summary_cu_cpu[doc_idx].item())
        summary_end = int(summary_cu_cpu[doc_idx + 1].item())
        doc_text_len = text_end - text_start
        if doc_text_len <= 0:
            continue
        if summary_end <= summary_start:
            outputs.append(read_q.new_zeros(1, doc_text_len, read_q.shape[2], value_dim))
            if return_local_prob:
                local_probs.append(read_q.new_ones(1, doc_text_len, read_q.shape[2], 1))
            continue

        read_q_i = read_q[:, text_start:text_end]
        router_q_i = router_q[:, text_start:text_end]
        states_i = states[:, summary_start:summary_end]
        summary_k_i = summary_k[:, summary_start:summary_end]
        boundaries_i = boundary_counts[summary_start:summary_end] - text_start
        null_i = None if null_logits is None else null_logits[:, text_start:text_end]

        summary_count = int(states_i.shape[1])
        token_index = torch.arange(doc_text_len, device=read_q.device, dtype=torch.long)
        valid_prefix = boundaries_i.to(device=read_q.device, dtype=torch.long).view(1, summary_count) <= (
            token_index.view(doc_text_len, 1)
        )
        route_query_heads = int(router_q_i.shape[2])
        route_key_heads = int(summary_k_i.shape[2])
        if route_query_heads != int(read_q_i.shape[2]):
            raise ValueError(
                "HSD route query heads must match read/value heads: "
                f"query_heads={route_query_heads}, value_heads={read_q_i.shape[2]}."
            )
        if route_key_heads == route_query_heads:
            logits = torch.einsum(
                "bthk,bmhk->bthm",
                router_q_i.float(),
                summary_k_i.float(),
            )
        elif route_key_heads == 1:
            logits = torch.einsum(
                "bthk,bmk->bthm",
                router_q_i.float(),
                summary_k_i[:, :, 0].float(),
            )
        else:
            raise ValueError(
                "HSD route key heads must be one or match query heads: "
                f"key_heads={route_key_heads}, query_heads={route_query_heads}."
            )
        logits = logits * float(route_scale)
        valid = valid_prefix.view(1, doc_text_len, 1, summary_count).expand(
            read_q_i.shape[0], doc_text_len, read_q_i.shape[2], summary_count
        )
        router_debug_logits = logits.masked_fill(~valid, -torch.inf)
        has_valid = valid_prefix.any(dim=-1).view(1, doc_text_len, 1, 1)

        null_prob = None
        summary_mass = None
        if null_i is not None:
            null_valid = torch.ones(
                read_q_i.shape[0], doc_text_len, read_q_i.shape[2], 1,
                device=valid.device, dtype=torch.bool,
            )
            logits_with_null = torch.cat([logits, null_i.float()], dim=-1)
            valid_with_null = torch.cat([valid, null_valid], dim=-1)
            weights_with_null = torch.softmax(logits_with_null.masked_fill(~valid_with_null, -torch.inf), dim=-1)
            weights = weights_with_null[..., :summary_count].to(states.dtype)
            null_prob = weights_with_null[..., summary_count:summary_count + 1].to(states.dtype)
            summary_mass = weights.sum(dim=-1, keepdim=True)
            debug_weights = torch.where(
                summary_mass > 0,
                weights / summary_mass.clamp_min(torch.finfo(weights.dtype).tiny),
                torch.zeros_like(weights),
            )
        else:
            masked_logits = logits.masked_fill(~valid, -torch.inf)
            masked_logits = torch.where(has_valid, masked_logits, torch.zeros_like(masked_logits))
            weights = torch.softmax(masked_logits, dim=-1)
            weights = torch.where(has_valid, weights, torch.zeros_like(weights)).to(states.dtype)
            debug_weights = weights

        if debug_stats_callback is not None:
            debug_stats_callback(
                weights=debug_weights,
                logits=router_debug_logits,
                summary_k=summary_k_i,
                valid_prefix=valid_prefix,
                has_valid=has_valid,
                boundaries=boundaries_i.to(device=read_q.device, dtype=torch.long),
                token_index=token_index,
                null_prob=null_prob,
                summary_mass=summary_mass,
                record_debug_stats=True,
            )

        reads = torch.einsum("bthk,bmhkv->bthmv", read_q_i.to(states.dtype), states_i) * float(read_scale)
        outputs.append(torch.einsum("bthm,bthmv->bthv", weights.to(reads.dtype), reads))
        if return_local_prob:
            if null_prob is None:
                raise RuntimeError("HSD local probability was requested without local/null logits.")
            local_probs.append(null_prob)

    if not outputs:
        out = read_q.new_zeros(read_q.shape[0], 0, read_q.shape[2], states.shape[-1])
    else:
        out = torch.cat(outputs, dim=1)
    out = out if output_dtype is None else out.to(output_dtype)
    if return_local_prob:
        local_prob = torch.cat(local_probs, dim=1) if local_probs else read_q.new_zeros(
            read_q.shape[0], 0, read_q.shape[2], 1
        )
        return out, local_prob
    return out


if triton is not None:

    @triton.jit
    def _route_momentum_affine_combine(a_left, b_left, a_right, b_right):
        return a_right * a_left, a_right * b_left + b_right

    @triton.jit
    def _route_momentum_mul_combine(left, right):
        return right * left

    @triton.jit
    def _route_momentum_local_scan_kernel(
        innovation,
        a,
        local_a,
        local_b,
        block_a,
        block_b,
        T,
        C,
        NUM_BLOCKS: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        i_c_block = tl.program_id(0)
        i_t_block = tl.program_id(1)
        i_b = tl.program_id(2)
        offs_t = i_t_block * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_c = i_c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        t_mask = offs_t < T
        c_mask = offs_c < C
        a_scalar = tl.load(
            a + i_b * T + offs_t,
            mask=t_mask,
            other=1.0,
        ).to(tl.float32)
        a_values = a_scalar[:, None] + tl.zeros([BLOCK_T, BLOCK_C], dtype=tl.float32)
        b_values = tl.load(
            innovation + (i_b * T + offs_t[:, None]) * C + offs_c[None, :],
            mask=t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        prefix_a, prefix_b = tl.associative_scan(
            (a_values, b_values),
            axis=0,
            combine_fn=_route_momentum_affine_combine,
        )
        prefix_a_scalar = tl.associative_scan(
            a_scalar,
            axis=0,
            combine_fn=_route_momentum_mul_combine,
        )
        tl.store(
            local_a + i_b * T + offs_t,
            prefix_a_scalar,
            mask=t_mask & (i_c_block == 0),
        )
        tl.store(
            local_b + (i_b * T + offs_t[:, None]) * C + offs_c[None, :],
            prefix_b,
            mask=t_mask[:, None] & c_mask[None, :],
        )
        tl.store(
            block_a + i_b * NUM_BLOCKS + i_t_block,
            tl.sum(
                tl.where(tl.arange(0, BLOCK_T) == BLOCK_T - 1, prefix_a_scalar, 0.0),
                axis=0,
            ),
            mask=i_c_block == 0,
        )
        last_b = tl.sum(
            tl.where((tl.arange(0, BLOCK_T) == BLOCK_T - 1)[:, None], prefix_b, 0.0),
            axis=0,
        )
        tl.store(
            block_b + (i_b * NUM_BLOCKS + i_t_block) * C + offs_c,
            last_b,
            mask=c_mask,
        )

    @triton.jit
    def _route_momentum_block_scan_kernel(
        block_a,
        block_b,
        initial_state,
        block_initial,
        C,
        NUM_BLOCKS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        i_c_block = tl.program_id(0)
        i_b = tl.program_id(1)
        offs_n = tl.arange(0, BLOCK_N)
        offs_c = i_c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        source_n = offs_n - 1
        source_valid = (offs_n > 0) & (source_n < NUM_BLOCKS)
        c_mask = offs_c < C
        shifted_a = tl.load(
            block_a + i_b * NUM_BLOCKS + source_n,
            mask=source_valid,
            other=1.0,
        ).to(tl.float32)
        shifted_a = shifted_a[:, None] + tl.zeros([BLOCK_N, BLOCK_C], dtype=tl.float32)
        shifted_b = tl.load(
            block_b + (i_b * NUM_BLOCKS + source_n[:, None]) * C + offs_c[None, :],
            mask=source_valid[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        prefix_a, prefix_b = tl.associative_scan(
            (shifted_a, shifted_b),
            axis=0,
            combine_fn=_route_momentum_affine_combine,
        )
        initial = tl.load(
            initial_state + i_b * C + offs_c,
            mask=c_mask,
            other=0.0,
        ).to(tl.float32)
        incoming = prefix_a * initial[None, :] + prefix_b
        tl.store(
            block_initial + (i_b * NUM_BLOCKS + offs_n[:, None]) * C + offs_c[None, :],
            incoming,
            mask=(offs_n[:, None] < NUM_BLOCKS) & c_mask[None, :],
        )

    @triton.jit
    def _route_momentum_apply_prefix_kernel(
        local_a,
        local_b,
        block_initial,
        T,
        C,
        NUM_BLOCKS: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        i_c_block = tl.program_id(0)
        i_t_block = tl.program_id(1)
        i_b = tl.program_id(2)
        offs_t = i_t_block * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_c = i_c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        t_mask = offs_t < T
        c_mask = offs_c < C
        prefix_a = tl.load(
            local_a + i_b * T + offs_t,
            mask=t_mask,
            other=0.0,
        ).to(tl.float32)
        prefix_b = tl.load(
            local_b + (i_b * T + offs_t[:, None]) * C + offs_c[None, :],
            mask=t_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        incoming = tl.load(
            block_initial + (i_b * NUM_BLOCKS + i_t_block) * C + offs_c,
            mask=c_mask,
            other=0.0,
        ).to(tl.float32)
        result = prefix_a[:, None] * incoming[None, :] + prefix_b
        tl.store(
            local_b + (i_b * T + offs_t[:, None]) * C + offs_c[None, :],
            result,
            mask=t_mask[:, None] & c_mask[None, :],
        )

    @triton.jit
    def _hsd_global_read_decode_fwd_kernel(
        read_q,
        router_q,
        states,
        summary_k,
        boundary_counts,
        null_logits,
        out,
        local_prob,
        M: tl.constexpr,
        H: tl.constexpr,
        STATE_K: tl.constexpr,
        ROUTE_K: tl.constexpr,
        V: tl.constexpr,
        START_INDEX,
        ROUTE_SCALE: tl.constexpr,
        READ_SCALE: tl.constexpr,
        HAS_NULL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_ROUTE_K: tl.constexpr,
        BLOCK_STATE_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_M_BLOCKS: tl.constexpr,
        NUM_ROUTE_K_BLOCKS: tl.constexpr,
        NUM_STATE_K_BLOCKS: tl.constexpr,
    ):
        i_v = tl.program_id(0)
        i_bh = tl.program_id(1)
        i_b = i_bh // H
        i_h = i_bh - i_b * H

        offs_m = tl.arange(0, BLOCK_M)
        offs_route_k = tl.arange(0, BLOCK_ROUTE_K)
        offs_state_k = tl.arange(0, BLOCK_STATE_K)
        offs_v = i_v * BLOCK_V + tl.arange(0, BLOCK_V)
        v_mask = offs_v < V
        neg_inf = -3.4028234663852886e38

        max_logit = neg_inf
        for i_m_block in range(NUM_M_BLOCKS):
            m_idx = i_m_block * BLOCK_M + offs_m
            logits = tl.zeros([BLOCK_M], dtype=tl.float32)
            for i_k_block in range(NUM_ROUTE_K_BLOCKS):
                k_idx = i_k_block * BLOCK_ROUTE_K + offs_route_k
                router = tl.load(
                    router_q + (i_b * H + i_h) * ROUTE_K + k_idx,
                    mask=k_idx < ROUTE_K,
                    other=0.0,
                ).to(tl.float32)
                keys = tl.load(
                    summary_k + ((i_b * M + m_idx[:, None]) * H + i_h) * ROUTE_K + k_idx[None, :],
                    mask=(m_idx[:, None] < M) & (k_idx[None, :] < ROUTE_K),
                    other=0.0,
                ).to(tl.float32)
                logits += tl.sum(keys * router[None, :], axis=1)

            boundaries = tl.load(boundary_counts + m_idx, mask=m_idx < M, other=2147483647)
            valid = (m_idx < M) & (boundaries <= START_INDEX)
            logits = tl.where(valid, logits * ROUTE_SCALE, neg_inf)
            max_logit = tl.maximum(max_logit, tl.max(logits, axis=0))

        if HAS_NULL:
            null_value = tl.load(null_logits + i_b * H + i_h).to(tl.float32)
            max_logit = tl.maximum(max_logit, null_value)

        max_safe = tl.where(max_logit == neg_inf, 0.0, max_logit)
        denom = 0.0
        out_acc = tl.zeros([BLOCK_V], dtype=tl.float32)
        for i_m_block in range(NUM_M_BLOCKS):
            m_idx = i_m_block * BLOCK_M + offs_m
            logits = tl.zeros([BLOCK_M], dtype=tl.float32)
            for i_k_block in range(NUM_ROUTE_K_BLOCKS):
                k_idx = i_k_block * BLOCK_ROUTE_K + offs_route_k
                router = tl.load(
                    router_q + (i_b * H + i_h) * ROUTE_K + k_idx,
                    mask=k_idx < ROUTE_K,
                    other=0.0,
                ).to(tl.float32)
                keys = tl.load(
                    summary_k + ((i_b * M + m_idx[:, None]) * H + i_h) * ROUTE_K + k_idx[None, :],
                    mask=(m_idx[:, None] < M) & (k_idx[None, :] < ROUTE_K),
                    other=0.0,
                ).to(tl.float32)
                logits += tl.sum(keys * router[None, :], axis=1)

            boundaries = tl.load(boundary_counts + m_idx, mask=m_idx < M, other=2147483647)
            valid = (m_idx < M) & (boundaries <= START_INDEX)
            logits = tl.where(valid, logits * ROUTE_SCALE, neg_inf)
            weights_num = tl.where(valid, tl.exp(logits - max_safe), 0.0)
            denom += tl.sum(weights_num, axis=0)

            for i_m_local in tl.static_range(0, BLOCK_M):
                cur_m = i_m_block * BLOCK_M + i_m_local
                cur_weight = tl.sum(tl.where(offs_m == i_m_local, weights_num, 0.0), axis=0)
                read_value = tl.zeros([BLOCK_V], dtype=tl.float32)
                for i_k_block in range(NUM_STATE_K_BLOCKS):
                    k_idx = i_k_block * BLOCK_STATE_K + offs_state_k
                    read_block = tl.load(
                        read_q + (i_b * H + i_h) * STATE_K + k_idx,
                        mask=k_idx < STATE_K,
                        other=0.0,
                    ).to(tl.float32)
                    state_block = tl.load(
                        states + (((i_b * M + cur_m) * H + i_h) * STATE_K + k_idx[:, None]) * V + offs_v[None, :],
                        mask=(cur_m < M) & (k_idx[:, None] < STATE_K) & v_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    read_value += tl.sum(read_block[:, None] * state_block, axis=0)
                out_acc += cur_weight * (read_value * READ_SCALE)

        if HAS_NULL:
            null_value = tl.load(null_logits + i_b * H + i_h).to(tl.float32)
            null_num = tl.exp(null_value - max_safe)
            denom += null_num
            if i_v == 0:
                tl.store(local_prob + i_b * H + i_h, null_num / denom)

        result = tl.where(denom > 0.0, out_acc / denom, 0.0)
        tl.store(
            out + (i_b * H + i_h) * V + offs_v,
            result,
            mask=v_mask,
        )

    @triton.jit
    def _hsd_global_read_decode_route_prob_kernel(
        router_q,
        summary_k,
        boundary_counts,
        null_logits,
        prob,
        local_prob,
        M: tl.constexpr,
        H: tl.constexpr,
        ROUTE_K: tl.constexpr,
        START_INDEX,
        ROUTE_SCALE: tl.constexpr,
        HAS_NULL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_ROUTE_K: tl.constexpr,
        NUM_M_BLOCKS: tl.constexpr,
        NUM_ROUTE_K_BLOCKS: tl.constexpr,
    ):
        i_bh = tl.program_id(0)
        i_b = i_bh // H
        i_h = i_bh - i_b * H

        offs_m = tl.arange(0, BLOCK_M)
        offs_route_k = tl.arange(0, BLOCK_ROUTE_K)
        neg_inf = -3.4028234663852886e38

        max_logit = neg_inf
        for i_m_block in range(NUM_M_BLOCKS):
            m_idx = i_m_block * BLOCK_M + offs_m
            logits = tl.zeros([BLOCK_M], dtype=tl.float32)
            for i_k_block in range(NUM_ROUTE_K_BLOCKS):
                k_idx = i_k_block * BLOCK_ROUTE_K + offs_route_k
                router = tl.load(
                    router_q + (i_b * H + i_h) * ROUTE_K + k_idx,
                    mask=k_idx < ROUTE_K,
                    other=0.0,
                ).to(tl.float32)
                keys = tl.load(
                    summary_k + ((i_b * M + m_idx[:, None]) * H + i_h) * ROUTE_K + k_idx[None, :],
                    mask=(m_idx[:, None] < M) & (k_idx[None, :] < ROUTE_K),
                    other=0.0,
                ).to(tl.float32)
                logits += tl.sum(keys * router[None, :], axis=1)

            boundaries = tl.load(boundary_counts + m_idx, mask=m_idx < M, other=2147483647)
            valid = (m_idx < M) & (boundaries <= START_INDEX)
            logits = tl.where(valid, logits * ROUTE_SCALE, neg_inf)
            max_logit = tl.maximum(max_logit, tl.max(logits, axis=0))

        if HAS_NULL:
            null_value = tl.load(null_logits + i_b * H + i_h).to(tl.float32)
            max_logit = tl.maximum(max_logit, null_value)

        max_safe = tl.where(max_logit == neg_inf, 0.0, max_logit)
        denom = 0.0
        for i_m_block in range(NUM_M_BLOCKS):
            m_idx = i_m_block * BLOCK_M + offs_m
            logits = tl.zeros([BLOCK_M], dtype=tl.float32)
            for i_k_block in range(NUM_ROUTE_K_BLOCKS):
                k_idx = i_k_block * BLOCK_ROUTE_K + offs_route_k
                router = tl.load(
                    router_q + (i_b * H + i_h) * ROUTE_K + k_idx,
                    mask=k_idx < ROUTE_K,
                    other=0.0,
                ).to(tl.float32)
                keys = tl.load(
                    summary_k + ((i_b * M + m_idx[:, None]) * H + i_h) * ROUTE_K + k_idx[None, :],
                    mask=(m_idx[:, None] < M) & (k_idx[None, :] < ROUTE_K),
                    other=0.0,
                ).to(tl.float32)
                logits += tl.sum(keys * router[None, :], axis=1)

            boundaries = tl.load(boundary_counts + m_idx, mask=m_idx < M, other=2147483647)
            valid = (m_idx < M) & (boundaries <= START_INDEX)
            logits = tl.where(valid, logits * ROUTE_SCALE, neg_inf)
            weights_num = tl.where(valid, tl.exp(logits - max_safe), 0.0)
            denom += tl.sum(weights_num, axis=0)

        if HAS_NULL:
            null_value = tl.load(null_logits + i_b * H + i_h).to(tl.float32)
            null_num = tl.exp(null_value - max_safe)
            denom += null_num
            tl.store(local_prob + i_b * H + i_h, null_num / denom)

        for i_m_block in range(NUM_M_BLOCKS):
            m_idx = i_m_block * BLOCK_M + offs_m
            logits = tl.zeros([BLOCK_M], dtype=tl.float32)
            for i_k_block in range(NUM_ROUTE_K_BLOCKS):
                k_idx = i_k_block * BLOCK_ROUTE_K + offs_route_k
                router = tl.load(
                    router_q + (i_b * H + i_h) * ROUTE_K + k_idx,
                    mask=k_idx < ROUTE_K,
                    other=0.0,
                ).to(tl.float32)
                keys = tl.load(
                    summary_k + ((i_b * M + m_idx[:, None]) * H + i_h) * ROUTE_K + k_idx[None, :],
                    mask=(m_idx[:, None] < M) & (k_idx[None, :] < ROUTE_K),
                    other=0.0,
                ).to(tl.float32)
                logits += tl.sum(keys * router[None, :], axis=1)

            boundaries = tl.load(boundary_counts + m_idx, mask=m_idx < M, other=2147483647)
            valid = (m_idx < M) & (boundaries <= START_INDEX)
            logits = tl.where(valid, logits * ROUTE_SCALE, neg_inf)
            weights_num = tl.where(valid, tl.exp(logits - max_safe), 0.0)
            p = tl.where(denom > 0.0, weights_num / denom, 0.0)
            tl.store(
                prob + (i_b * H + i_h) * M + m_idx,
                p,
                mask=m_idx < M,
            )

    @triton.jit
    def _hsd_global_read_decode_prob_read_kernel(
        read_q,
        states,
        prob,
        out,
        M: tl.constexpr,
        H: tl.constexpr,
        STATE_K: tl.constexpr,
        V: tl.constexpr,
        READ_SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_STATE_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_M_BLOCKS: tl.constexpr,
        NUM_STATE_K_BLOCKS: tl.constexpr,
    ):
        i_v = tl.program_id(0)
        i_bh = tl.program_id(1)
        i_b = i_bh // H
        i_h = i_bh - i_b * H

        offs_m = tl.arange(0, BLOCK_M)
        offs_state_k = tl.arange(0, BLOCK_STATE_K)
        offs_v = i_v * BLOCK_V + tl.arange(0, BLOCK_V)
        v_mask = offs_v < V
        out_acc = tl.zeros([BLOCK_V], dtype=tl.float32)

        for i_k_block in range(NUM_STATE_K_BLOCKS):
            k_idx = i_k_block * BLOCK_STATE_K + offs_state_k
            k_mask = k_idx < STATE_K
            read_block = tl.load(
                read_q + (i_b * H + i_h) * STATE_K + k_idx,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            weighted_state = tl.zeros([BLOCK_STATE_K, BLOCK_V], dtype=tl.float32)
            for i_m_block in range(NUM_M_BLOCKS):
                m_idx = i_m_block * BLOCK_M + offs_m
                prob_block = tl.load(
                    prob + (i_b * H + i_h) * M + m_idx,
                    mask=m_idx < M,
                    other=0.0,
                ).to(tl.float32)
                for i_m_local in tl.static_range(0, BLOCK_M):
                    cur_m = i_m_block * BLOCK_M + i_m_local
                    cur_prob = tl.sum(tl.where(offs_m == i_m_local, prob_block, 0.0), axis=0)
                    state_block = tl.load(
                        states + (((i_b * M + cur_m) * H + i_h) * STATE_K + k_idx[:, None]) * V + offs_v[None, :],
                        mask=(cur_m < M) & k_mask[:, None] & v_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    weighted_state += cur_prob * state_block
            out_acc += tl.sum(read_block[:, None] * weighted_state, axis=0)

        tl.store(
            out + (i_b * H + i_h) * V + offs_v,
            out_acc * READ_SCALE,
            mask=v_mask,
        )

    @triton.jit
    def _hsd_packed_global_read_fwd_kernel(
        read_q,
        router_q,
        states,
        summary_k,
        boundary_counts,
        text_cu_seqlens,
        summary_cu_seqlens,
        null_logits,
        out,
        softmax_lse,
        T: tl.constexpr,
        M: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        N_DOCS: tl.constexpr,
        ROUTE_SCALE: tl.constexpr,
        READ_SCALE: tl.constexpr,
        HAS_NULL: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_M_BLOCKS: tl.constexpr,
        NUM_K_BLOCKS: tl.constexpr,
    ):
        i_v = tl.program_id(0)
        i_t_block = tl.program_id(1)
        i_h = tl.program_id(2)
        offs_t = i_t_block * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_m = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = i_v * BLOCK_V + tl.arange(0, BLOCK_V)
        t_mask = offs_t < T
        v_mask = offs_v < V

        text_start = tl.zeros([BLOCK_T], dtype=tl.int64)
        text_end = tl.full([BLOCK_T], T, dtype=tl.int64)
        summary_start = tl.zeros([BLOCK_T], dtype=tl.int64)
        summary_end = tl.full([BLOCK_T], M, dtype=tl.int64)
        for i_doc in range(N_DOCS):
            doc_text_start = tl.load(text_cu_seqlens + i_doc)
            doc_text_end = tl.load(text_cu_seqlens + i_doc + 1)
            in_doc = (offs_t >= doc_text_start) & (offs_t < doc_text_end)
            text_start = tl.where(in_doc, doc_text_start, text_start)
            text_end = tl.where(in_doc, doc_text_end, text_end)
            summary_start = tl.where(in_doc, tl.load(summary_cu_seqlens + i_doc), summary_start)
            summary_end = tl.where(in_doc, tl.load(summary_cu_seqlens + i_doc + 1), summary_end)

        neg_inf = -3.4028234663852886e38
        max_logit = tl.full([BLOCK_T], neg_inf, dtype=tl.float32)

        for i_m_block in range(NUM_M_BLOCKS):
            m_idx = i_m_block * BLOCK_M + offs_m
            logits = tl.zeros([BLOCK_T, BLOCK_M], dtype=tl.float32)
            for i_k_block in range(NUM_K_BLOCKS):
                k_idx = i_k_block * BLOCK_K + offs_k
                router = tl.load(
                    router_q + ((offs_t[:, None] * H + i_h) * K + k_idx[None, :]),
                    mask=t_mask[:, None] & (k_idx[None, :] < K),
                    other=0.0,
                ).to(tl.float32)
                keys = tl.load(
                    summary_k + ((m_idx[:, None] * H + i_h) * K + k_idx[None, :]),
                    mask=(m_idx[:, None] < M) & (k_idx[None, :] < K),
                    other=0.0,
                ).to(tl.float32)
                logits += tl.dot(router, tl.trans(keys), input_precision="ieee")

            boundaries = tl.load(boundary_counts + m_idx, mask=m_idx < M, other=T + 1)
            valid = (
                t_mask[:, None]
                & (m_idx[None, :] >= summary_start[:, None])
                & (m_idx[None, :] < summary_end[:, None])
                & (boundaries[None, :] <= offs_t[:, None])
            )
            logits = tl.where(valid, logits * ROUTE_SCALE, neg_inf)
            max_logit = tl.maximum(max_logit, tl.max(logits, axis=1))

        if HAS_NULL:
            null_value = tl.load(
                null_logits + (offs_t * H + i_h),
                mask=t_mask,
                other=neg_inf,
            ).to(tl.float32)
            max_logit = tl.maximum(max_logit, null_value)

        max_safe = tl.where(max_logit == neg_inf, 0.0, max_logit)
        denom = tl.zeros([BLOCK_T], dtype=tl.float32)
        out_acc = tl.zeros([BLOCK_T, BLOCK_V], dtype=tl.float32)
        for i_m_block in range(NUM_M_BLOCKS):
            m_idx = i_m_block * BLOCK_M + offs_m
            logits = tl.zeros([BLOCK_T, BLOCK_M], dtype=tl.float32)
            for i_k_block in range(NUM_K_BLOCKS):
                k_idx = i_k_block * BLOCK_K + offs_k
                router = tl.load(
                    router_q + ((offs_t[:, None] * H + i_h) * K + k_idx[None, :]),
                    mask=t_mask[:, None] & (k_idx[None, :] < K),
                    other=0.0,
                ).to(tl.float32)
                keys = tl.load(
                    summary_k + ((m_idx[:, None] * H + i_h) * K + k_idx[None, :]),
                    mask=(m_idx[:, None] < M) & (k_idx[None, :] < K),
                    other=0.0,
                ).to(tl.float32)
                logits += tl.dot(router, tl.trans(keys), input_precision="ieee")

            boundaries = tl.load(boundary_counts + m_idx, mask=m_idx < M, other=T + 1)
            valid = (
                t_mask[:, None]
                & (m_idx[None, :] >= summary_start[:, None])
                & (m_idx[None, :] < summary_end[:, None])
                & (boundaries[None, :] <= offs_t[:, None])
            )
            logits = tl.where(valid, logits * ROUTE_SCALE, neg_inf)
            weights_num = tl.where(valid, tl.exp(logits - max_safe[:, None]), 0.0)
            denom += tl.sum(weights_num, axis=1)

            for i_m_local in tl.static_range(0, BLOCK_M):
                cur_m = i_m_block * BLOCK_M + i_m_local
                cur_weight = tl.sum(
                    tl.where(offs_m[None, :] == i_m_local, weights_num, 0.0),
                    axis=1,
                )
                read_value = tl.zeros([BLOCK_T, BLOCK_V], dtype=tl.float32)
                for i_k_block in range(NUM_K_BLOCKS):
                    k_idx = i_k_block * BLOCK_K + offs_k
                    read_block = tl.load(
                        read_q + ((offs_t[:, None] * H + i_h) * K + k_idx[None, :]),
                        mask=t_mask[:, None] & (k_idx[None, :] < K),
                        other=0.0,
                    ).to(tl.float32)
                    state_block = tl.load(
                        states + (((cur_m * H + i_h) * K + k_idx[:, None]) * V + offs_v[None, :]),
                        mask=(cur_m < M) & (k_idx[:, None] < K) & v_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    read_value += tl.dot(read_block, state_block, input_precision="ieee")
                out_acc += cur_weight[:, None] * (read_value * READ_SCALE)

        if HAS_NULL:
            null_value = tl.load(
                null_logits + (offs_t * H + i_h),
                mask=t_mask,
                other=neg_inf,
            ).to(tl.float32)
            denom += tl.exp(null_value - max_safe)

        lse = tl.where(denom > 0.0, max_safe + tl.log(denom), 0.0)
        if i_v == 0:
            tl.store(softmax_lse + offs_t * H + i_h, lse, mask=t_mask)

        result = tl.where(denom[:, None] > 0.0, out_acc / denom[:, None], 0.0)
        tl.store(
            out + ((offs_t[:, None] * H + i_h) * V + offs_v[None, :]),
            result,
            mask=t_mask[:, None] & v_mask[None, :],
        )

    @triton.jit
    def _hsd_packed_global_read_bwd_preprocess_kernel(
        dout,
        out,
        softmax_lse,
        null_logits,
        dlocal_prob,
        delta,
        dnull_logits,
        T: tl.constexpr,
        H: tl.constexpr,
        V: tl.constexpr,
        HAS_NULL: tl.constexpr,
        HAS_DLOCAL: tl.constexpr,
        NEED_DNULL: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_V_BLOCKS: tl.constexpr,
    ):
        i_t_block = tl.program_id(0)
        i_h = tl.program_id(1)
        offs_t = i_t_block * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_v = tl.arange(0, BLOCK_V)
        t_mask = offs_t < T
        acc = tl.zeros([BLOCK_T], dtype=tl.float32)
        for i_v_block in range(NUM_V_BLOCKS):
            v_idx = i_v_block * BLOCK_V + offs_v
            mask = t_mask[:, None] & (v_idx[None, :] < V)
            do = tl.load(
                dout + ((offs_t[:, None] * H + i_h) * V + v_idx[None, :]),
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            o = tl.load(
                out + ((offs_t[:, None] * H + i_h) * V + v_idx[None, :]),
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(do * o, axis=1)

        p_null = tl.zeros([BLOCK_T], dtype=tl.float32)
        if HAS_NULL:
            lse = tl.load(softmax_lse + offs_t * H + i_h, mask=t_mask, other=0.0).to(tl.float32)
            null_value = tl.load(null_logits + offs_t * H + i_h, mask=t_mask, other=-3.4028234663852886e38)
            p_null = tl.exp(null_value.to(tl.float32) - lse)
        gp = tl.load(dlocal_prob + offs_t * H + i_h, mask=t_mask, other=0.0).to(tl.float32) if HAS_DLOCAL else 0.0
        acc += gp * p_null
        tl.store(delta + offs_t * H + i_h, acc, mask=t_mask)
        if HAS_NULL and NEED_DNULL:
            dn = p_null * (gp - acc)
            tl.store(dnull_logits + offs_t * H + i_h, dn, mask=t_mask)

    @triton.jit
    def _hsd_packed_global_read_token_bwd_kernel(
        read_q,
        router_q,
        states,
        summary_k,
        boundary_counts,
        text_cu_seqlens,
        summary_cu_seqlens,
        dout,
        softmax_lse,
        delta,
        dread_q,
        drouter_q,
        T: tl.constexpr,
        M: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        N_DOCS: tl.constexpr,
        ROUTE_SCALE: tl.constexpr,
        READ_SCALE: tl.constexpr,
        NEED_DREAD: tl.constexpr,
        NEED_DROUTER: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_M_BLOCKS: tl.constexpr,
        NUM_K_BLOCKS: tl.constexpr,
        NUM_V_BLOCKS: tl.constexpr,
    ):
        i_k_out = tl.program_id(0)
        i_t_block = tl.program_id(1)
        i_h = tl.program_id(2)
        offs_t = i_t_block * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_m = tl.arange(0, BLOCK_M)
        offs_k_out = i_k_out * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = tl.arange(0, BLOCK_V)
        t_mask = offs_t < T
        k_out_mask = offs_k_out < K

        summary_start = tl.zeros([BLOCK_T], dtype=tl.int64)
        summary_end = tl.full([BLOCK_T], M, dtype=tl.int64)
        for i_doc in range(N_DOCS):
            doc_text_start = tl.load(text_cu_seqlens + i_doc)
            doc_text_end = tl.load(text_cu_seqlens + i_doc + 1)
            in_doc = (offs_t >= doc_text_start) & (offs_t < doc_text_end)
            summary_start = tl.where(in_doc, tl.load(summary_cu_seqlens + i_doc), summary_start)
            summary_end = tl.where(in_doc, tl.load(summary_cu_seqlens + i_doc + 1), summary_end)

        lse = tl.load(softmax_lse + offs_t * H + i_h, mask=t_mask, other=0.0).to(tl.float32)
        delta_t = tl.load(delta + offs_t * H + i_h, mask=t_mask, other=0.0).to(tl.float32)
        dread_acc = tl.zeros([BLOCK_T, BLOCK_K], dtype=tl.float32)
        drouter_acc = tl.zeros([BLOCK_T, BLOCK_K], dtype=tl.float32)

        for i_m_block in range(NUM_M_BLOCKS):
            m_idx = i_m_block * BLOCK_M + offs_m
            logits = tl.zeros([BLOCK_T, BLOCK_M], dtype=tl.float32)
            for i_k_block in range(NUM_K_BLOCKS):
                k_idx = i_k_block * BLOCK_K + offs_k
                router = tl.load(
                    router_q + ((offs_t[:, None] * H + i_h) * K + k_idx[None, :]),
                    mask=t_mask[:, None] & (k_idx[None, :] < K),
                    other=0.0,
                ).to(tl.float32)
                keys = tl.load(
                    summary_k + ((m_idx[:, None] * H + i_h) * K + k_idx[None, :]),
                    mask=(m_idx[:, None] < M) & (k_idx[None, :] < K),
                    other=0.0,
                ).to(tl.float32)
                logits += tl.dot(router, tl.trans(keys), input_precision="ieee")

            boundaries = tl.load(boundary_counts + m_idx, mask=m_idx < M, other=T + 1)
            valid = (
                t_mask[:, None]
                & (m_idx[None, :] >= summary_start[:, None])
                & (m_idx[None, :] < summary_end[:, None])
                & (boundaries[None, :] <= offs_t[:, None])
            )
            probs = tl.where(valid, tl.exp(logits * ROUTE_SCALE - lse[:, None]), 0.0)

            for i_m_local in tl.static_range(0, BLOCK_M):
                cur_m = i_m_block * BLOCK_M + i_m_local
                p = tl.sum(tl.where(offs_m[None, :] == i_m_local, probs, 0.0), axis=1)

                if NEED_DREAD:
                    for i_v_block in range(NUM_V_BLOCKS):
                        v_idx = i_v_block * BLOCK_V + offs_v
                        do = tl.load(
                            dout + ((offs_t[:, None] * H + i_h) * V + v_idx[None, :]),
                            mask=t_mask[:, None] & (v_idx[None, :] < V),
                            other=0.0,
                        ).to(tl.float32)
                        state_cur = tl.load(
                            states + (((cur_m * H + i_h) * K + offs_k_out[:, None]) * V + v_idx[None, :]),
                            mask=(cur_m < M) & k_out_mask[:, None] & (v_idx[None, :] < V),
                            other=0.0,
                        ).to(tl.float32)
                        dread_acc += tl.dot(do, tl.trans(state_cur), input_precision="ieee") * (
                            p[:, None] * READ_SCALE
                        )

                if NEED_DROUTER:
                    dot_read = tl.zeros([BLOCK_T], dtype=tl.float32)
                    for i_v_block in range(NUM_V_BLOCKS):
                        v_idx = i_v_block * BLOCK_V + offs_v
                        read_value = tl.zeros([BLOCK_T, BLOCK_V], dtype=tl.float32)
                        for i_k_block in range(NUM_K_BLOCKS):
                            k_idx = i_k_block * BLOCK_K + offs_k
                            read_block = tl.load(
                                read_q + ((offs_t[:, None] * H + i_h) * K + k_idx[None, :]),
                                mask=t_mask[:, None] & (k_idx[None, :] < K),
                                other=0.0,
                            ).to(tl.float32)
                            state_block = tl.load(
                                states + (((cur_m * H + i_h) * K + k_idx[:, None]) * V + v_idx[None, :]),
                                mask=(cur_m < M) & (k_idx[:, None] < K) & (v_idx[None, :] < V),
                                other=0.0,
                            ).to(tl.float32)
                            read_value += tl.dot(read_block, state_block, input_precision="ieee")
                        do = tl.load(
                            dout + ((offs_t[:, None] * H + i_h) * V + v_idx[None, :]),
                            mask=t_mask[:, None] & (v_idx[None, :] < V),
                            other=0.0,
                        ).to(tl.float32)
                        dot_read += tl.sum(do * (read_value * READ_SCALE), axis=1)
                    dz = p * (dot_read - delta_t)
                    key_cur = tl.load(
                        summary_k + ((cur_m * H + i_h) * K + offs_k_out),
                        mask=(cur_m < M) & k_out_mask,
                        other=0.0,
                    ).to(tl.float32)
                    drouter_acc += dz[:, None] * key_cur[None, :] * ROUTE_SCALE

        if NEED_DREAD:
            tl.store(
                dread_q + ((offs_t[:, None] * H + i_h) * K + offs_k_out[None, :]),
                dread_acc,
                mask=t_mask[:, None] & k_out_mask[None, :],
            )
        if NEED_DROUTER:
            tl.store(
                drouter_q + ((offs_t[:, None] * H + i_h) * K + offs_k_out[None, :]),
                drouter_acc,
                mask=t_mask[:, None] & k_out_mask[None, :],
            )

    @triton.jit
    def _hsd_packed_global_read_state_bwd_kernel(
        read_q,
        router_q,
        states,
        summary_k,
        boundary_counts,
        text_cu_seqlens,
        summary_cu_seqlens,
        dout,
        softmax_lse,
        dstates,
        T: tl.constexpr,
        M: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        N_DOCS: tl.constexpr,
        ROUTE_SCALE: tl.constexpr,
        READ_SCALE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_T_BLOCKS: tl.constexpr,
        NUM_K_BLOCKS: tl.constexpr,
    ):
        i_v = tl.program_id(0)
        i_k = tl.program_id(1)
        i_mh = tl.program_id(2)
        i_m = i_mh % M
        i_h = i_mh // M
        offs_v = i_v * BLOCK_V + tl.arange(0, BLOCK_V)
        offs_k_out = i_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_k = tl.arange(0, BLOCK_K)
        offs_t_base = tl.arange(0, BLOCK_T)
        v_mask = offs_v < V
        k_out_mask = offs_k_out < K

        text_start = tl.zeros((), dtype=tl.int64)
        text_end = tl.zeros((), dtype=tl.int64)
        for i_doc in range(N_DOCS):
            s_start = tl.load(summary_cu_seqlens + i_doc)
            s_end = tl.load(summary_cu_seqlens + i_doc + 1)
            in_doc = (i_m >= s_start) & (i_m < s_end)
            text_start = tl.where(in_doc, tl.load(text_cu_seqlens + i_doc), text_start)
            text_end = tl.where(in_doc, tl.load(text_cu_seqlens + i_doc + 1), text_end)
        boundary = tl.load(boundary_counts + i_m, mask=i_m < M, other=T + 1)
        acc = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)

        i_t_block = 0
        while i_t_block < NUM_T_BLOCKS:
            offs_t = i_t_block * BLOCK_T + offs_t_base
            t_mask = (offs_t < T) & (offs_t >= text_start) & (offs_t < text_end) & (offs_t >= boundary)
            logit = tl.zeros([BLOCK_T], dtype=tl.float32)
            for i_k_block in range(NUM_K_BLOCKS):
                k_idx = i_k_block * BLOCK_K + offs_k
                router = tl.load(
                    router_q + ((offs_t[:, None] * H + i_h) * K + k_idx[None, :]),
                    mask=t_mask[:, None] & (k_idx[None, :] < K),
                    other=0.0,
                ).to(tl.float32)
                key = tl.load(
                    summary_k + ((i_m * H + i_h) * K + k_idx),
                    mask=(i_m < M) & (k_idx < K),
                    other=0.0,
                ).to(tl.float32)
                logit += tl.sum(router * key[None, :], axis=1)
            lse = tl.load(softmax_lse + offs_t * H + i_h, mask=t_mask, other=0.0).to(tl.float32)
            p = tl.where(t_mask, tl.exp(logit * ROUTE_SCALE - lse), 0.0)
            read_cur = tl.load(
                read_q + ((offs_t[:, None] * H + i_h) * K + offs_k_out[None, :]),
                mask=t_mask[:, None] & k_out_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            do = tl.load(
                dout + ((offs_t[:, None] * H + i_h) * V + offs_v[None, :]),
                mask=t_mask[:, None] & v_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(tl.trans(read_cur), do * (p[:, None] * READ_SCALE), input_precision="ieee")
            i_t_block += 1

        tl.store(
            dstates + (((i_m * H + i_h) * K + offs_k_out[:, None]) * V + offs_v[None, :]),
            acc,
            mask=(i_m < M) & k_out_mask[:, None] & v_mask[None, :],
        )

    @triton.jit
    def _hsd_packed_global_read_key_bwd_kernel(
        read_q,
        router_q,
        states,
        summary_k,
        boundary_counts,
        text_cu_seqlens,
        summary_cu_seqlens,
        dout,
        softmax_lse,
        delta,
        dsummary_k,
        T: tl.constexpr,
        M: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        N_DOCS: tl.constexpr,
        ROUTE_SCALE: tl.constexpr,
        READ_SCALE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_T_BLOCKS: tl.constexpr,
        NUM_K_BLOCKS: tl.constexpr,
        NUM_V_BLOCKS: tl.constexpr,
    ):
        i_k_out = tl.program_id(0)
        i_m = tl.program_id(1)
        i_h = tl.program_id(2)
        offs_k_out = i_k_out * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_k = tl.arange(0, BLOCK_K)
        offs_t_base = tl.arange(0, BLOCK_T)
        offs_v = tl.arange(0, BLOCK_V)
        k_out_mask = offs_k_out < K

        text_start = tl.zeros((), dtype=tl.int64)
        text_end = tl.zeros((), dtype=tl.int64)
        for i_doc in range(N_DOCS):
            s_start = tl.load(summary_cu_seqlens + i_doc)
            s_end = tl.load(summary_cu_seqlens + i_doc + 1)
            in_doc = (i_m >= s_start) & (i_m < s_end)
            text_start = tl.where(in_doc, tl.load(text_cu_seqlens + i_doc), text_start)
            text_end = tl.where(in_doc, tl.load(text_cu_seqlens + i_doc + 1), text_end)
        boundary = tl.load(boundary_counts + i_m, mask=i_m < M, other=T + 1)
        acc = tl.zeros([BLOCK_K], dtype=tl.float32)

        i_t_block = 0
        while i_t_block < NUM_T_BLOCKS:
            offs_t = i_t_block * BLOCK_T + offs_t_base
            t_mask = (offs_t < T) & (offs_t >= text_start) & (offs_t < text_end) & (offs_t >= boundary)
            logit = tl.zeros([BLOCK_T], dtype=tl.float32)
            for i_k_block in range(NUM_K_BLOCKS):
                k_idx = i_k_block * BLOCK_K + offs_k
                router = tl.load(
                    router_q + ((offs_t[:, None] * H + i_h) * K + k_idx[None, :]),
                    mask=t_mask[:, None] & (k_idx[None, :] < K),
                    other=0.0,
                ).to(tl.float32)
                key = tl.load(
                    summary_k + ((i_m * H + i_h) * K + k_idx),
                    mask=(i_m < M) & (k_idx < K),
                    other=0.0,
                ).to(tl.float32)
                logit += tl.sum(router * key[None, :], axis=1)
            lse = tl.load(softmax_lse + offs_t * H + i_h, mask=t_mask, other=0.0).to(tl.float32)
            p = tl.where(t_mask, tl.exp(logit * ROUTE_SCALE - lse), 0.0)
            dot_read = tl.zeros([BLOCK_T], dtype=tl.float32)
            for i_v_block in range(NUM_V_BLOCKS):
                v_idx = i_v_block * BLOCK_V + offs_v
                read_value = tl.zeros([BLOCK_T, BLOCK_V], dtype=tl.float32)
                for i_k_block in range(NUM_K_BLOCKS):
                    k_idx = i_k_block * BLOCK_K + offs_k
                    read_block = tl.load(
                        read_q + ((offs_t[:, None] * H + i_h) * K + k_idx[None, :]),
                        mask=t_mask[:, None] & (k_idx[None, :] < K),
                        other=0.0,
                    ).to(tl.float32)
                    state_block = tl.load(
                        states + (((i_m * H + i_h) * K + k_idx[:, None]) * V + v_idx[None, :]),
                        mask=(i_m < M) & (k_idx[:, None] < K) & (v_idx[None, :] < V),
                        other=0.0,
                    ).to(tl.float32)
                    read_value += tl.dot(read_block, state_block, input_precision="ieee")
                do = tl.load(
                    dout + ((offs_t[:, None] * H + i_h) * V + v_idx[None, :]),
                    mask=t_mask[:, None] & (v_idx[None, :] < V),
                    other=0.0,
                ).to(tl.float32)
                dot_read += tl.sum(do * (read_value * READ_SCALE), axis=1)
            delta_t = tl.load(delta + offs_t * H + i_h, mask=t_mask, other=0.0).to(tl.float32)
            dz = p * (dot_read - delta_t)
            router_cur = tl.load(
                router_q + ((offs_t[:, None] * H + i_h) * K + offs_k_out[None, :]),
                mask=t_mask[:, None] & k_out_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(router_cur * (dz[:, None] * ROUTE_SCALE), axis=0)
            i_t_block += 1

        tl.store(
            dsummary_k + ((i_m * H + i_h) * K + offs_k_out),
            acc,
            mask=(i_m < M) & k_out_mask,
        )

    @triton.jit
    def _hsd_weighted_read_fwd_kernel(
        read_q,
        states,
        weights,
        out,
        T: tl.constexpr,
        M: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        READ_SCALE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_M_BLOCKS: tl.constexpr,
        NUM_K_BLOCKS: tl.constexpr,
    ):
        i_v = tl.program_id(0)
        i_t_block = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_b = i_bh // H
        i_h = i_bh % H
        offs_t = i_t_block * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_m = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = i_v * BLOCK_V + tl.arange(0, BLOCK_V)
        t_mask = offs_t < T
        v_mask = offs_v < V
        acc = tl.zeros([BLOCK_T, BLOCK_V], dtype=tl.float32)

        for i_m_block in range(NUM_M_BLOCKS):
            m_idx = i_m_block * BLOCK_M + offs_m
            weight_block = tl.load(
                weights + ((i_b * T + offs_t[:, None]) * H + i_h) * M + m_idx[None, :],
                mask=t_mask[:, None] & (m_idx[None, :] < M),
                other=0.0,
            ).to(tl.float32)
            for i_m_local in tl.static_range(0, BLOCK_M):
                cur_m = i_m_block * BLOCK_M + i_m_local
                weight = tl.sum(tl.where(offs_m[None, :] == i_m_local, weight_block, 0.0), axis=1)
                read_value = tl.zeros([BLOCK_T, BLOCK_V], dtype=tl.float32)
                for i_k_block in range(NUM_K_BLOCKS):
                    k_idx = i_k_block * BLOCK_K + offs_k
                    read_block = tl.load(
                        read_q + ((i_b * T + offs_t[:, None]) * H + i_h) * K + k_idx[None, :],
                        mask=t_mask[:, None] & (k_idx[None, :] < K),
                        other=0.0,
                    ).to(tl.float32)
                    state_block = tl.load(
                        states
                        + ((((i_b * M + cur_m) * H + i_h) * K + k_idx[:, None]) * V + offs_v[None, :]),
                        mask=(cur_m < M) & (k_idx[:, None] < K) & v_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    read_value += tl.dot(read_block, state_block, input_precision="tf32x3")
                acc += weight[:, None] * read_value

        tl.store(
            out + ((i_b * T + offs_t[:, None]) * H + i_h) * V + offs_v[None, :],
            acc * READ_SCALE,
            mask=t_mask[:, None] & v_mask[None, :],
        )

    @triton.jit
    def _hsd_weighted_read_dq_kernel(
        states,
        weights,
        dout,
        dread_q,
        T: tl.constexpr,
        M: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        READ_SCALE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_M_BLOCKS: tl.constexpr,
        NUM_V_BLOCKS: tl.constexpr,
    ):
        i_k = tl.program_id(0)
        i_t_block = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_b = i_bh // H
        i_h = i_bh % H
        offs_t = i_t_block * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_m = tl.arange(0, BLOCK_M)
        offs_k = i_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_v = tl.arange(0, BLOCK_V)
        t_mask = offs_t < T
        k_mask = offs_k < K
        acc = tl.zeros([BLOCK_T, BLOCK_K], dtype=tl.float32)

        for i_m_block in range(NUM_M_BLOCKS):
            m_idx = i_m_block * BLOCK_M + offs_m
            weight_block = tl.load(
                weights + ((i_b * T + offs_t[:, None]) * H + i_h) * M + m_idx[None, :],
                mask=t_mask[:, None] & (m_idx[None, :] < M),
                other=0.0,
            ).to(tl.float32)
            for i_m_local in tl.static_range(0, BLOCK_M):
                cur_m = i_m_block * BLOCK_M + i_m_local
                weight = tl.sum(tl.where(offs_m[None, :] == i_m_local, weight_block, 0.0), axis=1)
                for i_v_block in range(NUM_V_BLOCKS):
                    v_idx = i_v_block * BLOCK_V + offs_v
                    do = tl.load(
                        dout + ((i_b * T + offs_t[:, None]) * H + i_h) * V + v_idx[None, :],
                        mask=t_mask[:, None] & (v_idx[None, :] < V),
                        other=0.0,
                    ).to(tl.float32)
                    state_block = tl.load(
                        states
                        + ((((i_b * M + cur_m) * H + i_h) * K + offs_k[:, None]) * V + v_idx[None, :]),
                        mask=(cur_m < M) & k_mask[:, None] & (v_idx[None, :] < V),
                        other=0.0,
                    ).to(tl.float32)
                    acc += tl.dot(do, tl.trans(state_block), input_precision="tf32x3") * weight[:, None]

        tl.store(
            dread_q + ((i_b * T + offs_t[:, None]) * H + i_h) * K + offs_k[None, :],
            acc * READ_SCALE,
            mask=t_mask[:, None] & k_mask[None, :],
        )

    @triton.jit
    def _hsd_weighted_read_dw_kernel(
        read_q,
        states,
        dout,
        dweights,
        T: tl.constexpr,
        M: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        READ_SCALE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_K_BLOCKS: tl.constexpr,
        NUM_V_BLOCKS: tl.constexpr,
    ):
        i_m_block = tl.program_id(0)
        i_t_block = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_b = i_bh // H
        i_h = i_bh % H
        offs_t = i_t_block * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = tl.arange(0, BLOCK_V)
        t_mask = offs_t < T

        for i_m_local in tl.static_range(0, BLOCK_M):
            cur_m = i_m_block * BLOCK_M + i_m_local
            acc = tl.zeros([BLOCK_T], dtype=tl.float32)
            for i_v_block in range(NUM_V_BLOCKS):
                v_idx = i_v_block * BLOCK_V + offs_v
                read_value = tl.zeros([BLOCK_T, BLOCK_V], dtype=tl.float32)
                for i_k_block in range(NUM_K_BLOCKS):
                    k_idx = i_k_block * BLOCK_K + offs_k
                    read_block = tl.load(
                        read_q + ((i_b * T + offs_t[:, None]) * H + i_h) * K + k_idx[None, :],
                        mask=t_mask[:, None] & (k_idx[None, :] < K),
                        other=0.0,
                    ).to(tl.float32)
                    state_block = tl.load(
                        states
                        + ((((i_b * M + cur_m) * H + i_h) * K + k_idx[:, None]) * V + v_idx[None, :]),
                        mask=(cur_m < M) & (k_idx[:, None] < K) & (v_idx[None, :] < V),
                        other=0.0,
                    ).to(tl.float32)
                    read_value += tl.dot(read_block, state_block, input_precision="tf32x3")
                do = tl.load(
                    dout + ((i_b * T + offs_t[:, None]) * H + i_h) * V + v_idx[None, :],
                    mask=t_mask[:, None] & (v_idx[None, :] < V),
                    other=0.0,
                ).to(tl.float32)
                acc += tl.sum(do * read_value, axis=1)
            tl.store(
                dweights + ((i_b * T + offs_t) * H + i_h) * M + cur_m,
                acc * READ_SCALE,
                mask=t_mask & (cur_m < M),
            )

    @triton.jit
    def _hsd_weighted_read_ds_kernel(
        read_q,
        weights,
        dout,
        dstates,
        T: tl.constexpr,
        M: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        READ_SCALE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_T_BLOCKS: tl.constexpr,
    ):
        i_v = tl.program_id(0)
        i_k = tl.program_id(1)
        i_bmh = tl.program_id(2)
        i_h = i_bmh % H
        i_bm = i_bmh // H
        i_m = i_bm % M
        i_b = i_bm // M
        offs_v = i_v * BLOCK_V + tl.arange(0, BLOCK_V)
        offs_k = i_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_t_base = tl.arange(0, BLOCK_T)
        v_mask = offs_v < V
        k_mask = offs_k < K
        acc = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)

        for i_t_block in range(NUM_T_BLOCKS):
            offs_t = i_t_block * BLOCK_T + offs_t_base
            t_mask = offs_t < T
            read = tl.load(
                read_q + ((i_b * T + offs_t[:, None]) * H + i_h) * K + offs_k[None, :],
                mask=t_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(
                weights + ((i_b * T + offs_t) * H + i_h) * M + i_m,
                mask=t_mask,
                other=0.0,
            ).to(tl.float32)
            do = tl.load(
                dout + ((i_b * T + offs_t[:, None]) * H + i_h) * V + offs_v[None, :],
                mask=t_mask[:, None] & v_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(tl.trans(read), do * weight[:, None], input_precision="tf32x3")

        tl.store(
            dstates + ((((i_b * M + i_m) * H + i_h) * K + offs_k[:, None]) * V + offs_v[None, :]),
            acc * READ_SCALE,
            mask=k_mask[:, None] & v_mask[None, :],
        )

else:
    _route_momentum_local_scan_kernel = None
    _route_momentum_block_scan_kernel = None
    _route_momentum_apply_prefix_kernel = None
    _hsd_global_read_decode_fwd_kernel = None
    _hsd_global_read_decode_route_prob_kernel = None
    _hsd_global_read_decode_prob_read_kernel = None
    _hsd_packed_global_read_fwd_kernel = None
    _hsd_packed_global_read_bwd_preprocess_kernel = None
    _hsd_packed_global_read_token_bwd_kernel = None
    _hsd_packed_global_read_state_bwd_kernel = None
    _hsd_packed_global_read_key_bwd_kernel = None
    _hsd_weighted_read_fwd_kernel = None
    _hsd_weighted_read_dq_kernel = None
    _hsd_weighted_read_dw_kernel = None
    _hsd_weighted_read_ds_kernel = None


def _weighted_global_read_reference(
    read_q: torch.Tensor,
    states: torch.Tensor,
    summary_weights: torch.Tensor,
    read_scale: float,
) -> torch.Tensor:
    batch, text_len, heads, _ = read_q.shape
    summary_count = int(states.shape[1])
    value_dim = int(states.shape[-1])
    out = read_q.new_zeros(batch, text_len, heads, value_dim)
    for summary_idx in range(summary_count):
        read_i = torch.einsum(
            "bthk,bhkv->bthv",
            read_q.to(states.dtype),
            states[:, summary_idx],
        )
        out = out + summary_weights[..., summary_idx:summary_idx + 1].to(read_i.dtype) * read_i
    return out * float(read_scale)


def _weighted_global_read_decode(
    read_q: torch.Tensor,
    states: torch.Tensor,
    summary_weights: torch.Tensor,
    read_scale: float,
) -> torch.Tensor:
    """T=1 inference read using the existing decode probability-read kernel."""
    batch, _, heads, state_dim = read_q.shape
    summary_count = int(states.shape[1])
    value_dim = int(states.shape[-1])
    block_m = 16 if summary_count <= 128 else 32
    block_state_k = min(64, _next_power_of_2(state_dim))
    block_v = 64 if value_dim >= 64 else _next_power_of_2(value_dim)
    out = torch.empty(batch, 1, heads, value_dim, device=read_q.device, dtype=read_q.dtype)
    prob = summary_weights[:, 0].float().contiguous()
    _hsd_global_read_decode_prob_read_kernel[(triton.cdiv(value_dim, block_v), batch * heads)](
        read_q.contiguous(),
        states.contiguous(),
        prob,
        out,
        summary_count,
        heads,
        state_dim,
        value_dim,
        float(read_scale),
        BLOCK_M=block_m,
        BLOCK_STATE_K=block_state_k,
        BLOCK_V=block_v,
        NUM_M_BLOCKS=triton.cdiv(summary_count, block_m),
        NUM_STATE_K_BLOCKS=triton.cdiv(state_dim, block_state_k),
        num_warps=4,
        num_stages=3,
    )
    return out


def _validate_weighted_global_read(
    read_q: torch.Tensor,
    states: torch.Tensor,
    summary_weights: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    if read_q.ndim != 4 or states.ndim != 5 or summary_weights.ndim != 4:
        raise ValueError(
            "weighted global read expects read_q=[B,T,H,K], states=[B,M,H,K,V], "
            f"weights=[B,T,H,M]; got {tuple(read_q.shape)}, {tuple(states.shape)}, "
            f"{tuple(summary_weights.shape)}."
        )
    batch, text_len, heads, key_dim = read_q.shape
    summary_count = int(states.shape[1])
    value_dim = int(states.shape[-1])
    if tuple(states.shape[:4]) != (batch, summary_count, heads, key_dim):
        raise ValueError(f"weighted global-read state shape mismatch: {tuple(states.shape)} vs {tuple(read_q.shape)}.")
    if tuple(summary_weights.shape) != (batch, text_len, heads, summary_count):
        raise ValueError(
            "weighted global-read weights shape mismatch: "
            f"got {tuple(summary_weights.shape)}, expected {(batch, text_len, heads, summary_count)}."
        )
    if read_q.device != states.device or read_q.device != summary_weights.device:
        raise ValueError("weighted global-read tensors must be on the same device.")
    return batch, text_len, heads, summary_count, key_dim, value_dim


class _HSDWeightedGlobalReadFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        read_q: torch.Tensor,
        states: torch.Tensor,
        summary_weights: torch.Tensor,
        read_scale: float,
    ):
        batch, text_len, heads, summary_count, key_dim, value_dim = _validate_weighted_global_read(
            read_q, states, summary_weights
        )
        read_q_c = read_q.contiguous()
        states_c = states.contiguous()
        weights_c = summary_weights.float().contiguous()
        out = torch.empty(batch, text_len, heads, value_dim, device=read_q.device, dtype=read_q.dtype)
        block_t = 16
        block_m = 4
        block_k = max(16, min(32, _next_power_of_2(key_dim)))
        block_v = max(16, min(32, _next_power_of_2(value_dim)))
        num_m_blocks = triton.cdiv(summary_count, block_m)
        num_k_blocks = triton.cdiv(key_dim, block_k)
        _hsd_weighted_read_fwd_kernel[
            (triton.cdiv(value_dim, block_v), triton.cdiv(text_len, block_t), batch * heads)
        ](
            read_q_c,
            states_c,
            weights_c,
            out,
            text_len,
            summary_count,
            heads,
            key_dim,
            value_dim,
            float(read_scale),
            BLOCK_T=block_t,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            BLOCK_V=block_v,
            NUM_M_BLOCKS=num_m_blocks,
            NUM_K_BLOCKS=num_k_blocks,
            num_warps=4,
            num_stages=2,
        )
        ctx.read_scale = float(read_scale)
        ctx.save_for_backward(read_q_c, states_c, weights_c)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        read_q, states, summary_weights = ctx.saved_tensors
        batch, text_len, heads, summary_count, key_dim, value_dim = _validate_weighted_global_read(
            read_q, states, summary_weights
        )
        needs = ctx.needs_input_grad
        dout_c = dout.contiguous()
        block_t = 16
        block_m = 4
        block_k = max(16, min(32, _next_power_of_2(key_dim)))
        block_v = max(16, min(32, _next_power_of_2(value_dim)))
        num_t_blocks = triton.cdiv(text_len, block_t)
        num_m_blocks = triton.cdiv(summary_count, block_m)
        num_v_blocks = triton.cdiv(value_dim, block_v)
        num_k_blocks = triton.cdiv(key_dim, block_k)

        dread = torch.empty_like(read_q) if needs[0] else None
        dstates = torch.empty_like(states) if needs[1] else None
        dweights = torch.empty_like(summary_weights) if needs[2] else None
        if dread is not None:
            _hsd_weighted_read_dq_kernel[
                (num_k_blocks, num_t_blocks, batch * heads)
            ](
                states,
                summary_weights,
                dout_c,
                dread,
                text_len,
                summary_count,
                heads,
                key_dim,
                value_dim,
                ctx.read_scale,
                BLOCK_T=block_t,
                BLOCK_M=block_m,
                BLOCK_K=block_k,
                BLOCK_V=block_v,
                NUM_M_BLOCKS=num_m_blocks,
                NUM_V_BLOCKS=num_v_blocks,
                num_warps=4,
                num_stages=2,
            )
        if dweights is not None:
            _hsd_weighted_read_dw_kernel[
                (num_m_blocks, num_t_blocks, batch * heads)
            ](
                read_q,
                states,
                dout_c,
                dweights,
                text_len,
                summary_count,
                heads,
                key_dim,
                value_dim,
                ctx.read_scale,
                BLOCK_T=block_t,
                BLOCK_M=block_m,
                BLOCK_K=block_k,
                BLOCK_V=block_v,
                NUM_K_BLOCKS=num_k_blocks,
                NUM_V_BLOCKS=num_v_blocks,
                num_warps=4,
                num_stages=2,
            )
        if dstates is not None:
            _hsd_weighted_read_ds_kernel[
                (triton.cdiv(value_dim, block_v), num_k_blocks, batch * summary_count * heads)
            ](
                read_q,
                summary_weights,
                dout_c,
                dstates,
                text_len,
                summary_count,
                heads,
                key_dim,
                value_dim,
                ctx.read_scale,
                BLOCK_T=block_t,
                BLOCK_K=block_k,
                BLOCK_V=block_v,
                NUM_T_BLOCKS=num_t_blocks,
                num_warps=4,
                num_stages=2,
            )
        return dread, dstates, dweights, None


def hsd_weighted_global_read(
    *,
    read_q: torch.Tensor,
    states: torch.Tensor,
    summary_weights: torch.Tensor,
    read_scale: float,
) -> torch.Tensor:
    """Read HSD states using externally supplied dense route weights."""
    _validate_weighted_global_read(read_q, states, summary_weights)
    if summary_weights.shape[-1] == 0:
        return read_q.new_zeros(*read_q.shape[:-1], states.shape[-1])
    if read_q.is_cuda:
        weighted_backend = os.environ.get("HSD_MOMENTUM_WEIGHTED_READ_BACKEND", "").strip().lower()
        if not weighted_backend:
            global_backend = os.environ.get("HSD_FUSED_GLOBAL_READ_BACKEND", "").strip().lower()
            weighted_backend = "tile_speed" if global_backend in {"tile_speed", "tilespeed"} else "triton"
        if weighted_backend in {"tile_speed", "tilespeed"}:
            tile_candidate = (
                read_q.shape[0] == 1
                and read_q.shape[1] > 1
                and read_q.dtype in (torch.float16, torch.bfloat16)
                and states.dtype == torch.float32
                and summary_weights.dtype == torch.float32
                and 16 <= read_q.shape[-1] <= 256
                and states.shape[-1] >= 32
            )
            try:
                from march_research.hsd.hsd_global_read_tile import hsd_tile_weighted_global_read
            except Exception:
                tile_out = None
            else:
                tile_out = hsd_tile_weighted_global_read(
                    read_q=read_q,
                    states=states,
                    summary_weights=summary_weights,
                    read_scale=float(read_scale),
                )
            if tile_out is not None:
                return tile_out
            if tile_candidate and _env_bool("HSD_MOMENTUM_WEIGHTED_READ_STRICT", True):
                raise RuntimeError("HSD momentum TileSpeed weighted-read strict mode requested a fastpath hit, but it missed.")
        elif weighted_backend not in {"triton", "legacy", "reference"}:
            raise ValueError(
                "HSD_MOMENTUM_WEIGHTED_READ_BACKEND must be tile_speed or triton, "
                f"got {weighted_backend!r}."
            )
        if triton is None or _hsd_weighted_read_fwd_kernel is None:
            if torch.is_grad_enabled() and any(x.requires_grad for x in (read_q, states, summary_weights)):
                raise RuntimeError("CUDA route-weight momentum training requires the Triton weighted-read kernel.")
        else:
            decode_inference = read_q.shape[1] == 1 and (
                not torch.is_grad_enabled()
                or not any(x.requires_grad for x in (read_q, states, summary_weights))
            )
            if decode_inference and _hsd_global_read_decode_prob_read_kernel is not None:
                return _weighted_global_read_decode(read_q, states, summary_weights, float(read_scale))
            return _HSDWeightedGlobalReadFunction.apply(read_q, states, summary_weights, float(read_scale))
    return _weighted_global_read_reference(read_q, states, summary_weights, float(read_scale)).to(read_q.dtype)


def _supports_fused_global_read(
    *,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    states: torch.Tensor,
    summary_k: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    summary_cu_seqlens: torch.Tensor,
    null_logits: torch.Tensor | None,
    route_scale: float | None = None,
    read_scale: float | None = None,
) -> bool:
    if triton is None or _hsd_packed_global_read_fwd_kernel is None or not read_q.is_cuda:
        return False
    if read_q.ndim != 4 or router_q.shape != read_q.shape or states.ndim != 5 or summary_k.ndim != 4:
        return False
    if read_q.shape[0] != 1:
        return False
    if states.device != read_q.device or summary_k.device != read_q.device or router_q.device != read_q.device:
        return False
    if null_logits is not None and null_logits.device != read_q.device:
        return False

    batch, text_len, heads, key_dim = read_q.shape
    summary_count = int(states.shape[1])
    if summary_count <= 0 or text_len <= 0:
        return True
    if states.shape[:4] != (batch, summary_count, heads, key_dim):
        return False
    if summary_k.shape != (batch, summary_count, heads, key_dim):
        return False
    if null_logits is not None and tuple(null_logits.shape) != (batch, text_len, heads, 1):
        return False
    if text_cu_seqlens.ndim != 1 or summary_cu_seqlens.ndim != 1:
        return False
    if int(text_cu_seqlens.numel()) != int(summary_cu_seqlens.numel()) or int(text_cu_seqlens.numel()) < 2:
        return False
    if boundary_counts.ndim != 1 or int(boundary_counts.shape[0]) != summary_count:
        return False
    if key_dim > 256:
        return False
    return True


def _run_fused_global_read_forward(
    *,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    states: torch.Tensor,
    summary_k: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    summary_cu_seqlens: torch.Tensor,
    null_logits: torch.Tensor | None,
    route_scale: float,
    read_scale: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not _supports_fused_global_read(
        read_q=read_q,
        router_q=router_q,
        states=states,
        summary_k=summary_k,
        boundary_counts=boundary_counts,
        text_cu_seqlens=text_cu_seqlens,
        summary_cu_seqlens=summary_cu_seqlens,
        null_logits=null_logits,
    ):
        return None

    batch, text_len, heads, key_dim = read_q.shape
    summary_count = int(states.shape[1])
    value_dim = int(states.shape[-1])
    if summary_count <= 0 or text_len <= 0:
        return (
            read_q.new_zeros(batch, text_len, heads, value_dim),
            torch.zeros(text_len, heads, device=read_q.device, dtype=torch.float32),
        )

    read_q = read_q.contiguous()
    router_q = router_q.contiguous()
    states = states.contiguous()
    summary_k = summary_k.contiguous()
    boundary_counts = boundary_counts.to(device=read_q.device, dtype=torch.long).contiguous()
    text_cu_seqlens = text_cu_seqlens.to(device=read_q.device, dtype=torch.long).contiguous()
    summary_cu_seqlens = summary_cu_seqlens.to(device=read_q.device, dtype=torch.long).contiguous()
    if null_logits is not None:
        null_logits = null_logits.contiguous()
    else:
        null_logits = read_q.new_empty((0,), dtype=torch.float32)

    block_t = 4
    block_m = min(8, _next_power_of_2(max(1, summary_count)))
    block_k = max(16, min(64, _next_power_of_2(max(1, key_dim))))
    block_v = 16
    out = torch.empty(batch, text_len, heads, value_dim, device=read_q.device, dtype=read_q.dtype)
    softmax_lse = torch.empty(text_len, heads, device=read_q.device, dtype=torch.float32)
    grid = (triton.cdiv(value_dim, block_v), triton.cdiv(text_len, block_t), heads)

    with torch.cuda.device(read_q.device):
        _hsd_packed_global_read_fwd_kernel[grid](
            read_q,
            router_q,
            states,
            summary_k,
            boundary_counts,
            text_cu_seqlens,
            summary_cu_seqlens,
            null_logits,
            out,
            softmax_lse,
            T=int(text_len),
            M=int(summary_count),
            H=int(heads),
            K=int(key_dim),
            V=int(value_dim),
            N_DOCS=int(text_cu_seqlens.numel()) - 1,
            ROUTE_SCALE=float(route_scale),
            READ_SCALE=float(read_scale),
            HAS_NULL=bool(null_logits.numel() > 0),
            BLOCK_T=block_t,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            BLOCK_V=block_v,
            NUM_M_BLOCKS=triton.cdiv(summary_count, block_m),
            NUM_K_BLOCKS=triton.cdiv(key_dim, block_k),
            num_warps=4,
            num_stages=1,
        )
    return out, softmax_lse


def _supports_fused_global_read_backward(
    *,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    states: torch.Tensor,
    summary_k: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    summary_cu_seqlens: torch.Tensor,
    null_logits: torch.Tensor | None,
) -> bool:
    if not _supports_fused_global_read(
        read_q=read_q,
        router_q=router_q,
        states=states,
        summary_k=summary_k,
        boundary_counts=boundary_counts,
        text_cu_seqlens=text_cu_seqlens,
        summary_cu_seqlens=summary_cu_seqlens,
        null_logits=null_logits,
    ):
        return False
    return (
        triton is not None
        and _hsd_packed_global_read_bwd_preprocess_kernel is not None
        and _hsd_packed_global_read_token_bwd_kernel is not None
        and _hsd_packed_global_read_state_bwd_kernel is not None
        and _hsd_packed_global_read_key_bwd_kernel is not None
    )


def _run_fused_global_read_backward(
    *,
    dout: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    states: torch.Tensor,
    summary_k: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    summary_cu_seqlens: torch.Tensor,
    null_logits: torch.Tensor | None,
    dlocal_prob: torch.Tensor | None,
    route_scale: float,
    read_scale: float,
    needs: tuple[bool, ...],
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None:
    if not _env_bool("HSD_FUSED_GLOBAL_READ_BWD", True):
        return None
    if not _supports_fused_global_read_backward(
        read_q=read_q,
        router_q=router_q,
        states=states,
        summary_k=summary_k,
        boundary_counts=boundary_counts,
        text_cu_seqlens=text_cu_seqlens,
        summary_cu_seqlens=summary_cu_seqlens,
        null_logits=null_logits,
    ):
        return None

    batch, text_len, heads, key_dim = read_q.shape
    summary_count = int(states.shape[1])
    value_dim = int(states.shape[-1])
    if batch != 1:
        return None

    need_dread = bool(needs[0])
    need_drouter = bool(needs[1])
    need_dstates = bool(needs[2])
    need_dsummary_k = bool(needs[3])
    need_dnull = bool(null_logits is not None and needs[7])
    if not (need_dread or need_drouter or need_dstates or need_dsummary_k or need_dnull):
        return None

    read_q = read_q.contiguous()
    router_q = router_q.contiguous()
    states = states.contiguous()
    summary_k = summary_k.contiguous()
    dout = dout.contiguous()
    out = out.contiguous()
    softmax_lse = softmax_lse.contiguous()
    boundary_counts = boundary_counts.to(device=read_q.device, dtype=torch.long).contiguous()
    text_cu_seqlens = text_cu_seqlens.to(device=read_q.device, dtype=torch.long).contiguous()
    summary_cu_seqlens = summary_cu_seqlens.to(device=read_q.device, dtype=torch.long).contiguous()
    if null_logits is None:
        null_arg = read_q.new_empty((0,), dtype=torch.float32)
    else:
        null_arg = null_logits.contiguous()
    has_dlocal = dlocal_prob is not None and dlocal_prob.numel() > 0
    dlocal_arg = (
        dlocal_prob.contiguous().view(text_len, heads)
        if has_dlocal else read_q.new_empty((0,), dtype=torch.float32)
    )

    block_t = 16
    block_m = min(8, _next_power_of_2(max(1, summary_count)))
    block_k = max(16, min(64, _next_power_of_2(max(1, key_dim))))
    block_v = 16
    num_t_blocks = triton.cdiv(text_len, block_t)
    num_m_blocks = triton.cdiv(max(1, summary_count), block_m)
    num_k_blocks = triton.cdiv(key_dim, block_k)
    num_v_blocks = triton.cdiv(value_dim, block_v)
    n_docs = int(text_cu_seqlens.numel()) - 1

    delta = torch.empty(text_len, heads, device=read_q.device, dtype=torch.float32)
    dnull = torch.empty_like(null_logits) if need_dnull else None
    dread = torch.empty_like(read_q) if need_dread else None
    drouter = torch.empty_like(router_q) if need_drouter else None
    dstates = torch.empty_like(states) if need_dstates else None
    dsummary_k = torch.empty_like(summary_k) if need_dsummary_k else None

    with torch.cuda.device(read_q.device):
        _hsd_packed_global_read_bwd_preprocess_kernel[(num_t_blocks, heads)](
            dout,
            out,
            softmax_lse,
            null_arg,
            dlocal_arg,
            delta,
            dnull if dnull is not None else null_arg,
            T=int(text_len),
            H=int(heads),
            V=int(value_dim),
            HAS_NULL=bool(null_logits is not None),
            HAS_DLOCAL=bool(has_dlocal),
            NEED_DNULL=bool(need_dnull),
            BLOCK_T=block_t,
            BLOCK_V=block_v,
            NUM_V_BLOCKS=num_v_blocks,
            num_warps=4,
            num_stages=1,
        )

        if need_dread or need_drouter:
            _hsd_packed_global_read_token_bwd_kernel[(num_k_blocks, num_t_blocks, heads)](
                read_q,
                router_q,
                states,
                summary_k,
                boundary_counts,
                text_cu_seqlens,
                summary_cu_seqlens,
                dout,
                softmax_lse,
                delta,
                dread if dread is not None else read_q,
                drouter if drouter is not None else router_q,
                T=int(text_len),
                M=int(summary_count),
                H=int(heads),
                K=int(key_dim),
                V=int(value_dim),
                N_DOCS=n_docs,
                ROUTE_SCALE=float(route_scale),
                READ_SCALE=float(read_scale),
                NEED_DREAD=bool(need_dread),
                NEED_DROUTER=bool(need_drouter),
                BLOCK_T=block_t,
                BLOCK_M=block_m,
                BLOCK_K=block_k,
                BLOCK_V=block_v,
                NUM_M_BLOCKS=num_m_blocks,
                NUM_K_BLOCKS=num_k_blocks,
                NUM_V_BLOCKS=num_v_blocks,
                num_warps=4,
                num_stages=1,
            )

        if need_dstates and summary_count > 0:
            _hsd_packed_global_read_state_bwd_kernel[(num_v_blocks, num_k_blocks, summary_count * heads)](
                read_q,
                router_q,
                states,
                summary_k,
                boundary_counts,
                text_cu_seqlens,
                summary_cu_seqlens,
                dout,
                softmax_lse,
                dstates,
                T=int(text_len),
                M=int(summary_count),
                H=int(heads),
                K=int(key_dim),
                V=int(value_dim),
                N_DOCS=n_docs,
                ROUTE_SCALE=float(route_scale),
                READ_SCALE=float(read_scale),
                BLOCK_T=block_t,
                BLOCK_K=block_k,
                BLOCK_V=block_v,
                NUM_T_BLOCKS=num_t_blocks,
                NUM_K_BLOCKS=num_k_blocks,
                num_warps=4,
                num_stages=1,
            )

        if need_dsummary_k and summary_count > 0:
            _hsd_packed_global_read_key_bwd_kernel[(num_k_blocks, summary_count, heads)](
                read_q,
                router_q,
                states,
                summary_k,
                boundary_counts,
                text_cu_seqlens,
                summary_cu_seqlens,
                dout,
                softmax_lse,
                delta,
                dsummary_k,
                T=int(text_len),
                M=int(summary_count),
                H=int(heads),
                K=int(key_dim),
                V=int(value_dim),
                N_DOCS=n_docs,
                ROUTE_SCALE=float(route_scale),
                READ_SCALE=float(read_scale),
                BLOCK_T=block_t,
                BLOCK_K=block_k,
                BLOCK_V=block_v,
                NUM_T_BLOCKS=num_t_blocks,
                NUM_K_BLOCKS=num_k_blocks,
                NUM_V_BLOCKS=num_v_blocks,
                num_warps=4,
                num_stages=1,
            )

    return dread, drouter, dstates, dsummary_k, dnull


class _HSDFusedGlobalReadFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        read_q: torch.Tensor,
        router_q: torch.Tensor,
        states: torch.Tensor,
        summary_k: torch.Tensor,
        boundary_counts: torch.Tensor,
        text_cu_seqlens: torch.Tensor,
        summary_cu_seqlens: torch.Tensor,
        null_logits: torch.Tensor,
        route_scale: float,
        read_scale: float,
        has_null: bool,
    ):
        null_arg = null_logits if bool(has_null) else None
        outputs = _run_fused_global_read_forward(
            read_q=read_q,
            router_q=router_q,
            states=states,
            summary_k=summary_k,
            boundary_counts=boundary_counts,
            text_cu_seqlens=text_cu_seqlens,
            summary_cu_seqlens=summary_cu_seqlens,
            null_logits=null_arg,
            route_scale=float(route_scale),
            read_scale=float(read_scale),
        )
        if outputs is None:
            raise RuntimeError("HSD fused global read forward was called for an unsupported input.")
        out, softmax_lse = outputs
        ctx.route_scale = float(route_scale)
        ctx.read_scale = float(read_scale)
        ctx.has_null = bool(has_null)
        ctx.save_for_backward(
            read_q,
            router_q,
            states,
            summary_k,
            boundary_counts,
            text_cu_seqlens,
            summary_cu_seqlens,
            null_logits,
            out,
            softmax_lse,
        )
        local_prob = (
            (null_logits[0, ..., 0].float() - softmax_lse).exp().unsqueeze(0).unsqueeze(-1)
            if bool(has_null) else out.new_empty((0,), dtype=torch.float32)
        )
        return out, local_prob

    @staticmethod
    def backward(ctx, dout: torch.Tensor, dlocal_prob: torch.Tensor | None):
        (
            read_q,
            router_q,
            states,
            summary_k,
            boundary_counts,
            text_cu_seqlens,
            summary_cu_seqlens,
            null_logits,
            out,
            softmax_lse,
        ) = ctx.saved_tensors
        needs = ctx.needs_input_grad
        if not (needs[0] or needs[1] or needs[2] or needs[3] or (ctx.has_null and needs[7])):
            return (None, None, None, None, None, None, None, None, None, None, None)
        null_arg = null_logits if ctx.has_null else None
        fused_grads = _run_fused_global_read_backward(
            dout=dout,
            out=out,
            softmax_lse=softmax_lse,
            read_q=read_q,
            router_q=router_q,
            states=states,
            summary_k=summary_k,
            boundary_counts=boundary_counts,
            text_cu_seqlens=text_cu_seqlens,
            summary_cu_seqlens=summary_cu_seqlens,
            null_logits=null_arg,
            dlocal_prob=dlocal_prob,
            route_scale=ctx.route_scale,
            read_scale=ctx.read_scale,
            needs=needs,
        )
        if fused_grads is not None:
            dread, drouter, dstates, dsummary_k, dnull = fused_grads
            return (
                dread,
                drouter,
                dstates,
                dsummary_k,
                None,
                None,
                None,
                dnull if ctx.has_null else None,
                None,
                None,
                None,
            )
        if _env_bool("HSD_FUSED_GLOBAL_READ_BWD_STRICT", False):
            raise RuntimeError("HSD fused global read backward strict mode requested a fastpath hit, but it missed.")

        read_q_ref = read_q.detach().requires_grad_(needs[0])
        router_q_ref = router_q.detach().requires_grad_(needs[1])
        states_ref = states.detach().requires_grad_(needs[2])
        summary_k_ref = summary_k.detach().requires_grad_(needs[3])
        if ctx.has_null:
            null_ref = null_logits.detach().requires_grad_(needs[7])
        else:
            null_ref = None

        named_inputs = [
            ("read_q", read_q_ref, needs[0]),
            ("router_q", router_q_ref, needs[1]),
            ("states", states_ref, needs[2]),
            ("summary_k", summary_k_ref, needs[3]),
        ]
        if null_ref is not None:
            named_inputs.append(("null_logits", null_ref, needs[7]))
        grad_inputs = [tensor for _, tensor, need in named_inputs if need]
        with torch.enable_grad():
            out_ref = _history_read_packed_reference(
                read_q=read_q_ref,
                router_q=router_q_ref,
                states=states_ref,
                summary_k=summary_k_ref,
                boundary_counts=boundary_counts,
                text_cu_seqlens=text_cu_seqlens,
                summary_cu_seqlens=summary_cu_seqlens,
                null_logits=null_ref,
                route_scale=ctx.route_scale,
                read_scale=ctx.read_scale,
                output_dtype=read_q.dtype,
                return_local_prob=ctx.has_null,
            )
            if ctx.has_null:
                history_ref, local_ref = out_ref
                local_grad = torch.zeros_like(local_ref) if dlocal_prob is None else dlocal_prob
                grads = torch.autograd.grad(
                    (history_ref, local_ref),
                    grad_inputs,
                    grad_outputs=(dout, local_grad),
                    allow_unused=True,
                ) if grad_inputs else ()
            else:
                grads = torch.autograd.grad(out_ref, grad_inputs, grad_outputs=dout, allow_unused=True) if grad_inputs else ()

        grad_by_name = {name: grad for (name, _, need), grad in zip((item for item in named_inputs if item[2]), grads)}
        return (
            grad_by_name.get("read_q"),
            grad_by_name.get("router_q"),
            grad_by_name.get("states"),
            grad_by_name.get("summary_k"),
            None,
            None,
            None,
            grad_by_name.get("null_logits") if ctx.has_null else None,
            None,
            None,
            None,
        )


def hsd_fused_global_read_decode(
    *,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    states: torch.Tensor,
    summary_k: torch.Tensor,
    boundary_counts: torch.Tensor,
    start_index: int,
    null_logits: Optional[torch.Tensor] = None,
    route_scale: float,
    read_scale: float,
    debug_stats_callback: Optional[Callable[..., None]] = None,
    return_local_prob: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    """Forward-only HSD global-read fastpath for HF cache decode.

    The decode path intentionally does not support debug stats or top-k routing.
    Callers should fall back to the torch reference path for those cases.
    """
    if not _env_bool("HSD_FUSED_GLOBAL_READ_DECODE", True):
        return None
    backend = os.environ.get("HSD_FUSED_GLOBAL_READ_DECODE_BACKEND", "auto").strip().lower()
    if backend in {"", "default"}:
        backend = "auto"
    if backend in {"0", "false", "off", "none", "disable", "disabled", "torch", "reference", "gemm"}:
        return None
    if triton is None or not read_q.is_cuda:
        return None
    if debug_stats_callback is not None:
        return None
    if read_q.ndim != 4 or router_q.ndim != 4 or states.ndim != 5 or summary_k.ndim != 4:
        return None
    batch_size, text_len, heads, state_dim = read_q.shape
    if text_len != 1:
        return None
    if router_q.shape[:3] != (batch_size, text_len, heads):
        return None
    summary_count = int(states.shape[1])
    if summary_count <= 0:
        history = read_q.new_zeros(batch_size, 1, heads, states.shape[-1])
        if return_local_prob:
            return history, torch.ones(batch_size, 1, heads, 1, device=read_q.device, dtype=torch.float32)
        return history
    value_dim = int(states.shape[-1])
    route_dim = int(router_q.shape[-1])
    if states.shape[:4] != (batch_size, summary_count, heads, state_dim):
        return None
    if summary_k.shape != (batch_size, summary_count, heads, route_dim):
        return None
    if boundary_counts.ndim != 1 or int(boundary_counts.shape[0]) != summary_count:
        return None
    has_null = null_logits is not None
    if return_local_prob and not has_null:
        raise ValueError("HSD decode local probability requires local/null logits.")
    if has_null and tuple(null_logits.shape) != (batch_size, 1, heads, 1):
        return None
    if torch.is_grad_enabled() and (
        read_q.requires_grad
        or router_q.requires_grad
        or states.requires_grad
        or summary_k.requires_grad
        or (null_logits is not None and null_logits.requires_grad)
    ):
        return None

    streaming_max_m = _env_int("HSD_FUSED_GLOBAL_READ_DECODE_MAX_M", 8)
    prob_max_m = _env_int("HSD_FUSED_GLOBAL_READ_DECODE_PROB_MAX_M", 256)
    use_streaming = backend in {"streaming", "triton"}
    use_prob = backend in {"prob", "prob_v2", "prob-v2"}
    if backend == "auto":
        use_streaming = streaming_max_m <= 0 or summary_count <= streaming_max_m
        use_prob = (not use_streaming) and (prob_max_m <= 0 or summary_count <= prob_max_m)
    elif not (use_streaming or use_prob):
        return None

    read_q_c = read_q.contiguous()
    router_q_c = router_q.contiguous()
    states_c = states.contiguous()
    summary_k_c = summary_k.contiguous()
    boundaries_c = boundary_counts.to(device=read_q.device, dtype=torch.long).contiguous()
    null_c = null_logits.contiguous() if null_logits is not None else read_q
    out = torch.empty(batch_size, 1, heads, value_dim, device=read_q.device, dtype=read_q.dtype)
    local_prob = torch.empty(batch_size, 1, heads, 1, device=read_q.device, dtype=torch.float32)

    if use_prob:
        if prob_max_m > 0 and summary_count > prob_max_m:
            return None
        block_m = 16 if summary_count <= 128 else 32
        block_route_k = min(64, _next_power_of_2(route_dim))
        block_state_k = min(64, _next_power_of_2(state_dim))
        block_v = 64 if value_dim >= 64 else _next_power_of_2(value_dim)
        num_m_blocks = triton.cdiv(summary_count, block_m)
        num_route_k_blocks = triton.cdiv(route_dim, block_route_k)
        num_state_k_blocks = triton.cdiv(state_dim, block_state_k)
        prob = torch.empty(batch_size, heads, summary_count, device=read_q.device, dtype=torch.float32)
        _hsd_global_read_decode_route_prob_kernel[(batch_size * heads,)](
            router_q_c,
            summary_k_c,
            boundaries_c,
            null_c,
            prob,
            local_prob,
            summary_count,
            heads,
            route_dim,
            int(start_index),
            float(route_scale),
            has_null,
            BLOCK_M=block_m,
            BLOCK_ROUTE_K=block_route_k,
            NUM_M_BLOCKS=num_m_blocks,
            NUM_ROUTE_K_BLOCKS=num_route_k_blocks,
            num_warps=4,
            num_stages=3,
        )
        _hsd_global_read_decode_prob_read_kernel[(triton.cdiv(value_dim, block_v), batch_size * heads)](
            read_q_c,
            states_c,
            prob,
            out,
            summary_count,
            heads,
            state_dim,
            value_dim,
            float(read_scale),
            BLOCK_M=block_m,
            BLOCK_STATE_K=block_state_k,
            BLOCK_V=block_v,
            NUM_M_BLOCKS=num_m_blocks,
            NUM_STATE_K_BLOCKS=num_state_k_blocks,
            num_warps=4,
            num_stages=3,
        )
        return (out, local_prob) if return_local_prob else out

    if streaming_max_m > 0 and summary_count > streaming_max_m:
        return None
    block_m = 16 if summary_count <= 64 else 32
    block_route_k = min(64, _next_power_of_2(route_dim))
    block_state_k = min(64, _next_power_of_2(state_dim))
    block_v = 64 if value_dim >= 64 else _next_power_of_2(value_dim)
    num_m_blocks = triton.cdiv(summary_count, block_m)
    num_route_k_blocks = triton.cdiv(route_dim, block_route_k)
    num_state_k_blocks = triton.cdiv(state_dim, block_state_k)
    grid = (triton.cdiv(value_dim, block_v), batch_size * heads)
    _hsd_global_read_decode_fwd_kernel[grid](
        read_q_c,
        router_q_c,
        states_c,
        summary_k_c,
        boundaries_c,
        null_c,
        out,
        local_prob,
        summary_count,
        heads,
        state_dim,
        route_dim,
        value_dim,
        int(start_index),
        float(route_scale),
        float(read_scale),
        has_null,
        BLOCK_M=block_m,
        BLOCK_ROUTE_K=block_route_k,
        BLOCK_STATE_K=block_state_k,
        BLOCK_V=block_v,
        NUM_M_BLOCKS=num_m_blocks,
        NUM_ROUTE_K_BLOCKS=num_route_k_blocks,
        NUM_STATE_K_BLOCKS=num_state_k_blocks,
        num_warps=4,
        num_stages=3,
    )
    return (out, local_prob) if return_local_prob else out


def hsd_fused_global_read(
    *,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    states: torch.Tensor,
    summary_k: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    summary_cu_seqlens: torch.Tensor,
    null_logits: Optional[torch.Tensor] = None,
    route_scale: float,
    read_scale: float,
    debug_stats_callback: Optional[Callable[..., None]] = None,
    return_local_prob: bool = False,
    backend: str | None = None,
    profile: str | None = None,
    strict: bool | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    if return_local_prob and null_logits is None:
        raise ValueError("HSD local probability requires local/null logits.")
    # Explicit model/runtime arguments take precedence over the legacy HF
    # environment contract.  Omitting them preserves the old behavior.
    if backend is None and not _env_bool("HSD_FUSED_GLOBAL_READ", True):
        return None
    backend_name = (
        os.environ.get("HSD_FUSED_GLOBAL_READ_BACKEND", "")
        if backend is None
        else str(backend)
    ).strip().lower()
    if backend_name in {"", "auto", "reference", "gemm", "torch"}:
        return _history_read_packed_reference(
            read_q=read_q,
            router_q=router_q,
            states=states,
            summary_k=summary_k,
            boundary_counts=boundary_counts,
            text_cu_seqlens=text_cu_seqlens,
            summary_cu_seqlens=summary_cu_seqlens,
            null_logits=null_logits,
            route_scale=route_scale,
            read_scale=read_scale,
            output_dtype=read_q.dtype,
            debug_stats_callback=debug_stats_callback,
            return_local_prob=return_local_prob,
        )
    tile_backend = backend_name in {
        "tile",
        "tile_lowmem",
        "tile-lowmem",
        "tile_low_memory",
        "tile_speed",
        "tilespeed",
        "tile_online",
        "tile-online",
        "tile_grouped_online",
        "flashqla_tile",
    }
    tile_strict = bool(strict) if strict is not None else (
        _env_bool("HSD_FUSED_GLOBAL_READ_TILE_STRICT", False)
        or backend_name in {
            "tile_speed",
            "tilespeed",
            "tile_online",
            "tile-online",
            "tile_grouped_online",
            "flashqla_tile",
        }
        or _env_bool("HSD_FUSED_GLOBAL_READ_TILE_SPEED_STRICT", False)
    )
    if tile_backend or (backend_name == "tile" and _env_bool("HSD_FUSED_GLOBAL_READ_TILE", True)):
        import_error = None
        try:
            from march_research.hsd.hsd_global_read_tile import hsd_tile_global_read
        except Exception as exc:
            import_error = exc
            tile = None
        else:
            tile = hsd_tile_global_read(
                read_q=read_q,
                router_q=router_q,
                states=states,
                summary_k=summary_k,
                boundary_counts=boundary_counts,
                text_cu_seqlens=text_cu_seqlens,
                summary_cu_seqlens=summary_cu_seqlens,
                null_logits=null_logits,
                route_scale=route_scale,
                read_scale=read_scale,
                debug_stats_callback=debug_stats_callback,
                return_local_prob=return_local_prob,
                backend=backend_name,
                profile=profile,
                strict=tile_strict,
            )
        if tile is not None:
            return tile
        if tile_strict:
            message = "HSD fused packed global read tile strict mode requested a fastpath hit, but it missed."
            if import_error is not None:
                raise RuntimeError(message) from import_error
            raise RuntimeError(message)
    if backend_name and backend_name not in {"streaming", "triton"}:
        return None
    if not _supports_fused_global_read(
        read_q=read_q,
        router_q=router_q,
        states=states,
        summary_k=summary_k,
        boundary_counts=boundary_counts,
        text_cu_seqlens=text_cu_seqlens,
        summary_cu_seqlens=summary_cu_seqlens,
        null_logits=null_logits,
    ):
        return None
    null_arg = null_logits if null_logits is not None else read_q.new_empty((0,), dtype=read_q.dtype)
    out, local_prob = _HSDFusedGlobalReadFunction.apply(
        read_q,
        router_q,
        states,
        summary_k,
        boundary_counts.to(device=read_q.device, dtype=torch.long).contiguous(),
        text_cu_seqlens.to(device=read_q.device, dtype=torch.long).contiguous(),
        summary_cu_seqlens.to(device=read_q.device, dtype=torch.long).contiguous(),
        null_arg,
        float(route_scale),
        float(read_scale),
        null_logits is not None,
    )
    return (out, local_prob) if return_local_prob else out


__all__ = [
    "hsd_fused_global_read",
    "hsd_fused_global_read_decode",
    "hsd_weighted_global_read",
    "route_weight_momentum_reference",
    "route_weight_momentum_scan",
]
