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

# Grouped Top-K production defaults. Each knob remains independently
# overridable so a run can roll back to the earlier launch geometry without
# changing model semantics.
DEFAULT_TOPK_FWD_GROUP_SPLITS = 8
DEFAULT_TOPK_DQ_GROUP_SPLITS = 8
DEFAULT_TOPK_DQDW_BLOCK_K = 32


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


def _env_choice(name: str, default: str, *, allowed: set[str]) -> str:
    raw = os.environ.get(name, "").strip().lower()
    value = default if not raw else raw
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}.")
    return value


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _validate_sparse_topk_global_read(
    read_q: torch.Tensor,
    states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_valid: torch.Tensor,
) -> tuple[int, int, int, int, int, int, int]:
    if read_q.ndim != 4 or states.ndim != 5:
        raise ValueError(
            "sparse top-k global read expects read_q=[B,T,H,Dk] and states=[B,M,H,Dk,Dv], "
            f"got {tuple(read_q.shape)} and {tuple(states.shape)}."
        )
    if topk_indices.ndim != 4 or topk_weights.ndim != 4 or topk_valid.ndim != 4:
        raise ValueError("top-k indices, weights, and valid mask must all be [B,T,H,K].")
    batch, text_len, heads, key_dim = read_q.shape
    summary_count = int(states.shape[1])
    value_dim = int(states.shape[-1])
    topk = int(topk_indices.shape[-1])
    expected_topk_shape = (batch, text_len, heads, topk)
    if tuple(states.shape[:4]) != (batch, summary_count, heads, key_dim):
        raise ValueError(
            "sparse top-k state shape mismatch: "
            f"got {tuple(states.shape)}, expected prefix {(batch, summary_count, heads, key_dim)}."
        )
    if tuple(topk_weights.shape) != expected_topk_shape or tuple(topk_valid.shape) != expected_topk_shape:
        raise ValueError(
            "sparse top-k route tensor shape mismatch: "
            f"indices={tuple(topk_indices.shape)}, weights={tuple(topk_weights.shape)}, "
            f"valid={tuple(topk_valid.shape)}."
        )
    if topk not in _SUPPORTED_TOPK:
        raise ValueError(f"sparse top-k global read supports K in {sorted(_SUPPORTED_TOPK)}, got {topk}.")
    if summary_count <= 0:
        raise ValueError("sparse top-k global read requires at least one summary state.")
    devices = {read_q.device, states.device, topk_indices.device, topk_weights.device, topk_valid.device}
    if len(devices) != 1:
        raise ValueError("sparse top-k global-read tensors must be on the same device.")
    if topk_valid.dtype != torch.bool:
        raise ValueError(f"topk_valid must be bool, got {topk_valid.dtype}.")
    if topk_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"topk_indices must be int32 or int64, got {topk_indices.dtype}.")
    if topk_weights.dtype != torch.float32:
        raise ValueError(f"topk_weights must be float32, got {topk_weights.dtype}.")
    # Avoid a device-wide synchronization on the CUDA hot path. Routing builds
    # safe indices before entering this operator; retain the explicit range
    # check for CPU/reference diagnostics.
    if not topk_valid.is_cuda and bool(topk_valid.any().item()):
        valid_indices = topk_indices.masked_select(topk_valid)
        if int(valid_indices.min().item()) < 0 or int(valid_indices.max().item()) >= summary_count:
            raise ValueError("valid sparse top-k indices must be in [0, summary_count).")
    return batch, text_len, heads, summary_count, key_dim, value_dim, topk


def sparse_topk_global_read_reference(
    *,
    read_q: torch.Tensor,
    states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_valid: torch.Tensor,
    read_scale: float,
) -> torch.Tensor:
    """Autograd reference that only materializes one selected slot at a time."""
    batch, text_len, heads, summary_count, _, value_dim, topk = _validate_sparse_topk_global_read(
        read_q, states, topk_indices, topk_weights, topk_valid
    )
    head_outputs: list[torch.Tensor] = []
    batch_index = torch.arange(batch, device=read_q.device).view(batch, 1).expand(batch, text_len)
    for head_idx in range(heads):
        q_h = read_q[:, :, head_idx].to(states.dtype)
        out_h = torch.zeros(batch, text_len, value_dim, device=read_q.device, dtype=states.dtype)
        for slot_idx in range(topk):
            valid = topk_valid[:, :, head_idx, slot_idx]
            safe_idx = topk_indices[:, :, head_idx, slot_idx].clamp(0, summary_count - 1)
            selected = states[batch_index, safe_idx, head_idx]
            read = torch.einsum("btk,btkv->btv", q_h, selected)
            weight = topk_weights[:, :, head_idx, slot_idx] * valid.to(topk_weights.dtype)
            out_h = out_h + weight.unsqueeze(-1).to(read.dtype) * read
        head_outputs.append(out_h)
    return (torch.stack(head_outputs, dim=2) * float(read_scale)).to(read_q.dtype)


if triton is not None:

    @triton.jit
    def _topk_csr_count_kernel(
        indices,
        valid,
        counts,
        N: tl.constexpr,
        T: tl.constexpr,
        H: tl.constexpr,
        M,
        TOPK: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        is_valid = tl.load(valid + offs, mask=mask, other=0).to(tl.int1)
        summary = tl.load(indices + offs, mask=mask, other=0).to(tl.int32)
        summary = tl.where(is_valid, summary, 0)
        flat_th = offs // TOPK
        head = flat_th % H
        flat_bt = flat_th // H
        batch = flat_bt // T
        group = (batch * H + head) * M + summary
        tl.atomic_add(counts + group, 1, mask=mask & is_valid)

    @triton.jit
    def _topk_csr_offsets_kernel(
        counts,
        offsets,
        cursors,
        GROUPS,
        BLOCK: tl.constexpr,
    ):
        """Build the small group prefix sum and clear fill cursors in one CTA."""
        offs = tl.arange(0, BLOCK)
        mask = offs < GROUPS
        values = tl.load(counts + offs, mask=mask, other=0).to(tl.int32)
        inclusive = tl.cumsum(values, axis=0)
        tl.store(offsets + offs + 1, inclusive, mask=mask)
        tl.store(cursors + offs, 0, mask=mask)
        tl.store(offsets, 0)

    @triton.jit
    def _topk_csr_fill_kernel(
        indices,
        valid,
        offsets,
        cursors,
        selection_ids,
        N: tl.constexpr,
        T: tl.constexpr,
        H: tl.constexpr,
        M,
        TOPK: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        is_valid = tl.load(valid + offs, mask=mask, other=0).to(tl.int1)
        summary = tl.load(indices + offs, mask=mask, other=0).to(tl.int32)
        summary = tl.where(is_valid, summary, 0)
        flat_th = offs // TOPK
        head = flat_th % H
        flat_bt = flat_th // H
        batch = flat_bt // T
        group = (batch * H + head) * M + summary
        local = tl.atomic_add(cursors + group, 1, mask=mask & is_valid)
        start = tl.load(offsets + group, mask=mask & is_valid, other=0)
        tl.store(selection_ids + start + local, offs, mask=mask & is_valid)

    @triton.jit
    def _split_fp32_high_residual_kernel(
        source,
        high,
        residual,
        N: tl.constexpr,
        USE_BF16: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        value = tl.load(source + offs, mask=mask, other=0.0).to(tl.float32)
        if USE_BF16:
            high_value = value.to(tl.bfloat16)
            residual_value = (value - high_value.to(tl.float32)).to(tl.bfloat16)
        else:
            high_value = value.to(tl.float16)
            residual_value = (value - high_value.to(tl.float32)).to(tl.float16)
        tl.store(high + offs, high_value, mask=mask)
        tl.store(residual + offs, residual_value, mask=mask)

    @triton.jit
    def _sparse_topk_fwd_grouped_kernel(
        read_q,
        states,
        states_lo,
        weights,
        selection_ids,
        counts,
        offsets,
        out_acc,
        staged_contrib,
        T: tl.constexpr,
        M,
        H: tl.constexpr,
        DK: tl.constexpr,
        DV: tl.constexpr,
        TOPK: tl.constexpr,
        READ_SCALE: tl.constexpr,
        Q_IS_BF16: tl.constexpr,
        STATE_PREPACKED: tl.constexpr,
        USE_TF32_STATE: tl.constexpr,
        SINGLE_BF16_DOT: tl.constexpr,
        V_START,
        STAGED_OUTPUT: tl.constexpr,
        ATOMIC_OUTPUT_IS_INPUT: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_K_BLOCKS: tl.constexpr,
        GROUP_SPLITS: tl.constexpr,
    ):
        i_v = tl.program_id(0)
        group = tl.program_id(1)
        split = tl.program_id(2)
        head = (group // M) % H
        batch = group // (M * H)
        summary = group % M
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        local_v = tl.arange(0, BLOCK_V)
        offs_v = V_START + i_v * BLOCK_V + local_v
        v_mask = offs_v < DV
        start = tl.load(offsets + group).to(tl.int32)
        count = tl.load(counts + group).to(tl.int32)
        # Split a long summary group across independent CTAs.  Selection ids
        # are unique within a group, so the split does not introduce any new
        # same-group output collision; the existing cross-summary accumulation
        # remains atomic exactly as in the unsplit implementation.  Its storage
        # dtype is FP32 by default and optionally the input dtype.
        pos = split * BLOCK_N
        while pos < count:
            n_mask = offs_n < count - pos
            selection = tl.load(selection_ids + start + pos + offs_n, mask=n_mask, other=0).to(tl.int32)
            flat_th = selection // TOPK
            flat_bt = flat_th // H
            token = flat_bt % T
            acc = tl.zeros((BLOCK_N, BLOCK_V), dtype=tl.float32)
            for i_k in range(NUM_K_BLOCKS):
                k_idx = i_k * BLOCK_K + offs_k
                q_value = tl.load(
                    read_q + ((batch * T + token[:, None]) * H + head) * DK + k_idx[None, :],
                    mask=n_mask[:, None] & (k_idx[None, :] < DK),
                    other=0.0,
                )
                state_offset = ((((batch * M + summary) * H + head) * DK + k_idx[:, None]) * DV + offs_v[None, :])
                if USE_TF32_STATE:
                    state_f = tl.load(
                        states + state_offset,
                        mask=(k_idx[:, None] < DK) & v_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    acc += tl.dot(q_value.to(tl.float32), state_f, input_precision="tf32")
                elif SINGLE_BF16_DOT:
                    state_high = tl.load(
                        states + state_offset,
                        mask=(k_idx[:, None] < DK) & v_mask[None, :],
                        other=0.0,
                    ).to(tl.bfloat16)
                    acc += tl.dot(q_value.to(tl.bfloat16), state_high)
                elif STATE_PREPACKED:
                    q_dot = q_value
                    state_hi_value = tl.load(
                        states + state_offset,
                        mask=(k_idx[:, None] < DK) & v_mask[None, :],
                        other=0.0,
                    )
                    state_lo_value = tl.load(
                        states_lo + state_offset,
                        mask=(k_idx[:, None] < DK) & v_mask[None, :],
                        other=0.0,
                    )
                    acc += tl.dot(q_dot, state_hi_value) + tl.dot(q_dot, state_lo_value)
                else:
                    q_dot = q_value
                    state_f = tl.load(
                        states + state_offset,
                        mask=(k_idx[:, None] < DK) & v_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    if Q_IS_BF16:
                        state_hi_value = state_f.to(tl.bfloat16)
                        state_lo_value = (state_f - state_hi_value.to(tl.float32)).to(tl.bfloat16)
                    else:
                        state_hi_value = state_f.to(tl.float16)
                        state_lo_value = (state_f - state_hi_value.to(tl.float32)).to(tl.float16)
                    acc += tl.dot(q_dot, state_hi_value) + tl.dot(q_dot, state_lo_value)
            weight = tl.load(weights + selection, mask=n_mask, other=0.0).to(tl.float32)
            contribution = acc * weight[:, None] * READ_SCALE
            if STAGED_OUTPUT:
                tl.store(
                    staged_contrib + selection[:, None] * BLOCK_V + local_v[None, :],
                    contribution,
                    mask=n_mask[:, None] & v_mask[None, :],
                )
            else:
                # The default path accumulates into an FP32 tensor.  Hopper
                # also supports atomic add directly into BF16/FP16 storage;
                # keeping this compile-time branch here lets the opt-in path
                # avoid both the FP32 output accumulator and its final cast.
                # Staged output never enters this branch.
                if ATOMIC_OUTPUT_IS_INPUT:
                    if Q_IS_BF16:
                        contribution = contribution.to(tl.bfloat16)
                    else:
                        contribution = contribution.to(tl.float16)
                tl.atomic_add(
                    out_acc + ((batch * T + token[:, None]) * H + head) * DV + offs_v[None, :],
                    contribution,
                    mask=n_mask[:, None] & v_mask[None, :],
                )
            pos += GROUP_SPLITS * BLOCK_N

    @triton.jit
    def _sparse_topk_fwd_reduce_kernel(
        staged_contrib,
        valid,
        out,
        BTH,
        DV: tl.constexpr,
        TOPK: tl.constexpr,
        V_START,
        BLOCK_T: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        offs_t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_v = tl.arange(0, BLOCK_V)
        t_mask = offs_t < BTH
        v_mask = V_START + offs_v < DV
        acc = tl.zeros((BLOCK_T, BLOCK_V), dtype=tl.float32)
        for slot in range(TOPK):
            selection = offs_t * TOPK + slot
            slot_valid = tl.load(valid + selection, mask=t_mask, other=0).to(tl.int1)
            value = tl.load(
                staged_contrib + selection[:, None] * BLOCK_V + offs_v[None, :],
                mask=t_mask[:, None] & slot_valid[:, None] & v_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += value
        tl.store(
            out + offs_t[:, None] * DV + V_START + offs_v[None, :],
            acc,
            mask=t_mask[:, None] & v_mask[None, :],
        )

    @triton.jit
    def _sparse_topk_dq_grouped_kernel(
        states,
        states_lo,
        weights,
        dout,
        selection_ids,
        counts,
        offsets,
        dq_acc,
        T: tl.constexpr,
        M,
        H: tl.constexpr,
        DK: tl.constexpr,
        DV: tl.constexpr,
        TOPK: tl.constexpr,
        READ_SCALE: tl.constexpr,
        Q_IS_BF16: tl.constexpr,
        STATE_PREPACKED: tl.constexpr,
        USE_TF32_STATE: tl.constexpr,
        SINGLE_BF16_DOT: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_V_BLOCKS: tl.constexpr,
        GROUP_SPLITS: tl.constexpr,
    ):
        i_k = tl.program_id(0)
        group = tl.program_id(1)
        split = tl.program_id(2)
        head = (group // M) % H
        batch = group // (M * H)
        summary = group % M
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = i_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_v = tl.arange(0, BLOCK_V)
        k_mask = offs_k < DK
        start = tl.load(offsets + group).to(tl.int32)
        count = tl.load(counts + group).to(tl.int32)
        pos = split * BLOCK_N
        while pos < count:
            n_mask = offs_n < count - pos
            selection = tl.load(selection_ids + start + pos + offs_n, mask=n_mask, other=0).to(tl.int32)
            flat_th = selection // TOPK
            flat_bt = flat_th // H
            token = flat_bt % T
            weight = tl.load(weights + selection, mask=n_mask, other=0.0).to(tl.float32)
            acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
            for i_v in range(NUM_V_BLOCKS):
                v_idx = i_v * BLOCK_V + offs_v
                do_value = tl.load(
                    dout + ((batch * T + token[:, None]) * H + head) * DV + v_idx[None, :],
                    mask=n_mask[:, None] & (v_idx[None, :] < DV),
                    other=0.0,
                )
                state_offset = ((((batch * M + summary) * H + head) * DK + offs_k[:, None]) * DV + v_idx[None, :])
                if USE_TF32_STATE:
                    state_f = tl.load(
                        states + state_offset,
                        mask=k_mask[:, None] & (v_idx[None, :] < DV),
                        other=0.0,
                    ).to(tl.float32)
                    acc += tl.dot(do_value.to(tl.float32), tl.trans(state_f), input_precision="tf32")
                elif SINGLE_BF16_DOT:
                    state_high = tl.load(
                        states + state_offset,
                        mask=k_mask[:, None] & (v_idx[None, :] < DV),
                        other=0.0,
                    ).to(tl.bfloat16)
                    acc += tl.dot(
                        do_value.to(tl.bfloat16),
                        tl.trans(state_high),
                    )
                elif STATE_PREPACKED:
                    do_dot = do_value
                    state_hi_value = tl.load(
                        states + state_offset,
                        mask=k_mask[:, None] & (v_idx[None, :] < DV),
                        other=0.0,
                    )
                    state_lo_value = tl.load(
                        states_lo + state_offset,
                        mask=k_mask[:, None] & (v_idx[None, :] < DV),
                        other=0.0,
                    )
                    acc += tl.dot(do_dot, tl.trans(state_hi_value)) + tl.dot(do_dot, tl.trans(state_lo_value))
                else:
                    do_dot = do_value
                    state_f = tl.load(
                        states + state_offset,
                        mask=k_mask[:, None] & (v_idx[None, :] < DV),
                        other=0.0,
                    ).to(tl.float32)
                    if Q_IS_BF16:
                        state_hi_value = state_f.to(tl.bfloat16)
                        state_lo_value = (state_f - state_hi_value.to(tl.float32)).to(tl.bfloat16)
                    else:
                        state_hi_value = state_f.to(tl.float16)
                        state_lo_value = (state_f - state_hi_value.to(tl.float32)).to(tl.float16)
                    acc += tl.dot(do_dot, tl.trans(state_hi_value)) + tl.dot(do_dot, tl.trans(state_lo_value))
            tl.atomic_add(
                dq_acc + ((batch * T + token[:, None]) * H + head) * DK + offs_k[None, :],
                acc * weight[:, None] * READ_SCALE,
                mask=n_mask[:, None] & k_mask[None, :],
            )
            pos += GROUP_SPLITS * BLOCK_N

    @triton.jit
    def _sparse_topk_dq_dw_grouped_kernel(
        read_q,
        states,
        states_lo,
        weights,
        dout,
        selection_ids,
        counts,
        offsets,
        dq_acc,
        dweights,
        T: tl.constexpr,
        M,
        H: tl.constexpr,
        DK: tl.constexpr,
        DV: tl.constexpr,
        TOPK: tl.constexpr,
        READ_SCALE: tl.constexpr,
        Q_IS_BF16: tl.constexpr,
        STATE_PREPACKED: tl.constexpr,
        USE_TF32_STATE: tl.constexpr,
        SINGLE_BF16_DOT: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_V_BLOCKS: tl.constexpr,
        GROUP_SPLITS: tl.constexpr,
    ):
        """Compute dQ and dWeight from the shared state^T @ dO product.

        One program owns a (summary, head, K-block, split) tile.  dQ keeps the
        same atomic accumulation as the standalone kernel; dWeight receives
        one partial per K-block, so it also uses FP32 atomics.  This avoids a
        second state read and tensor-core contraction in the common training
        case where both gradients are requested.
        """
        i_k = tl.program_id(0)
        group = tl.program_id(1)
        split = tl.program_id(2)
        head = (group // M) % H
        batch = group // (M * H)
        summary = group % M
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = i_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_v = tl.arange(0, BLOCK_V)
        k_mask = offs_k < DK
        start = tl.load(offsets + group).to(tl.int32)
        count = tl.load(counts + group).to(tl.int32)
        pos = split * BLOCK_N
        while pos < count:
            n_mask = offs_n < count - pos
            selection = tl.load(selection_ids + start + pos + offs_n, mask=n_mask, other=0).to(tl.int32)
            flat_th = selection // TOPK
            flat_bt = flat_th // H
            token = flat_bt % T
            weight = tl.load(weights + selection, mask=n_mask, other=0.0).to(tl.float32)
            acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
            for i_v in range(NUM_V_BLOCKS):
                v_idx = i_v * BLOCK_V + offs_v
                do_value = tl.load(
                    dout + ((batch * T + token[:, None]) * H + head) * DV + v_idx[None, :],
                    mask=n_mask[:, None] & (v_idx[None, :] < DV),
                    other=0.0,
                )
                state_offset = ((((batch * M + summary) * H + head) * DK + offs_k[:, None]) * DV + v_idx[None, :])
                if USE_TF32_STATE:
                    state_f = tl.load(
                        states + state_offset,
                        mask=k_mask[:, None] & (v_idx[None, :] < DV),
                        other=0.0,
                    ).to(tl.float32)
                    acc += tl.dot(do_value.to(tl.float32), tl.trans(state_f), input_precision="tf32")
                elif SINGLE_BF16_DOT:
                    state_high = tl.load(
                        states + state_offset,
                        mask=k_mask[:, None] & (v_idx[None, :] < DV),
                        other=0.0,
                    ).to(tl.bfloat16)
                    acc += tl.dot(
                        do_value.to(tl.bfloat16),
                        tl.trans(state_high),
                    )
                elif STATE_PREPACKED:
                    do_dot = do_value
                    state_hi_value = tl.load(
                        states + state_offset,
                        mask=k_mask[:, None] & (v_idx[None, :] < DV),
                        other=0.0,
                    )
                    state_lo_value = tl.load(
                        states_lo + state_offset,
                        mask=k_mask[:, None] & (v_idx[None, :] < DV),
                        other=0.0,
                    )
                    acc += tl.dot(do_dot, tl.trans(state_hi_value)) + tl.dot(do_dot, tl.trans(state_lo_value))
                else:
                    do_dot = do_value
                    state_f = tl.load(
                        states + state_offset,
                        mask=k_mask[:, None] & (v_idx[None, :] < DV),
                        other=0.0,
                    ).to(tl.float32)
                    if Q_IS_BF16:
                        state_hi_value = state_f.to(tl.bfloat16)
                        state_lo_value = (state_f - state_hi_value.to(tl.float32)).to(tl.bfloat16)
                    else:
                        state_hi_value = state_f.to(tl.float16)
                        state_lo_value = (state_f - state_hi_value.to(tl.float32)).to(tl.float16)
                    acc += tl.dot(do_dot, tl.trans(state_hi_value)) + tl.dot(do_dot, tl.trans(state_lo_value))

            tl.atomic_add(
                dq_acc + ((batch * T + token[:, None]) * H + head) * DK + offs_k[None, :],
                acc * weight[:, None] * READ_SCALE,
                mask=n_mask[:, None] & k_mask[None, :],
            )
            q_value = tl.load(
                read_q + ((batch * T + token[:, None]) * H + head) * DK + offs_k[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            dw_partial = tl.sum(acc * q_value.to(tl.float32), axis=1) * READ_SCALE
            tl.atomic_add(dweights + selection, dw_partial, mask=n_mask)
            pos += GROUP_SPLITS * BLOCK_N

    @triton.jit
    def _sparse_topk_dw_grouped_kernel(
        read_q,
        states,
        states_lo,
        dout,
        selection_ids,
        counts,
        offsets,
        dweights,
        T: tl.constexpr,
        M,
        H: tl.constexpr,
        DK: tl.constexpr,
        DV: tl.constexpr,
        TOPK: tl.constexpr,
        READ_SCALE: tl.constexpr,
        Q_IS_BF16: tl.constexpr,
        STATE_PREPACKED: tl.constexpr,
        USE_TF32_STATE: tl.constexpr,
        SINGLE_BF16_DOT: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_K_BLOCKS: tl.constexpr,
        NUM_V_BLOCKS: tl.constexpr,
        GROUP_SPLITS: tl.constexpr,
    ):
        group = tl.program_id(0)
        split = tl.program_id(1)
        head = (group // M) % H
        batch = group // (M * H)
        summary = group % M
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = tl.arange(0, BLOCK_V)
        start = tl.load(offsets + group).to(tl.int32)
        count = tl.load(counts + group).to(tl.int32)
        # dWeight has one unique destination per selection id.  Splitting the
        # group therefore increases parallelism without atomics or a reduction.
        pos = split * BLOCK_N
        while pos < count:
            n_mask = offs_n < count - pos
            selection = tl.load(selection_ids + start + pos + offs_n, mask=n_mask, other=0).to(tl.int32)
            flat_th = selection // TOPK
            flat_bt = flat_th // H
            token = flat_bt % T
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for i_v in range(NUM_V_BLOCKS):
                v_idx = i_v * BLOCK_V + offs_v
                read_value = tl.zeros((BLOCK_N, BLOCK_V), dtype=tl.float32)
                for i_k in range(NUM_K_BLOCKS):
                    k_idx = i_k * BLOCK_K + offs_k
                    q_f = tl.load(
                        read_q + ((batch * T + token[:, None]) * H + head) * DK + k_idx[None, :],
                        mask=n_mask[:, None] & (k_idx[None, :] < DK),
                        other=0.0,
                    ).to(tl.float32)
                    state_offset = ((((batch * M + summary) * H + head) * DK + k_idx[:, None]) * DV + v_idx[None, :])
                    if USE_TF32_STATE:
                        state_f = tl.load(
                            states + state_offset,
                            mask=(k_idx[:, None] < DK) & (v_idx[None, :] < DV),
                            other=0.0,
                        ).to(tl.float32)
                        read_value += tl.dot(q_f, state_f, input_precision="tf32")
                    elif SINGLE_BF16_DOT:
                        state_high = tl.load(
                            states + state_offset,
                            mask=(k_idx[:, None] < DK) & (v_idx[None, :] < DV),
                            other=0.0,
                        ).to(tl.bfloat16)
                        read_value += tl.dot(q_f.to(tl.bfloat16), state_high)
                    elif STATE_PREPACKED:
                        if Q_IS_BF16:
                            q_dot = q_f.to(tl.bfloat16)
                        else:
                            q_dot = q_f.to(tl.float16)
                        state_hi_value = tl.load(
                            states + state_offset,
                            mask=(k_idx[:, None] < DK) & (v_idx[None, :] < DV),
                            other=0.0,
                        )
                        state_lo_value = tl.load(
                            states_lo + state_offset,
                            mask=(k_idx[:, None] < DK) & (v_idx[None, :] < DV),
                            other=0.0,
                        )
                        read_value += tl.dot(q_dot, state_hi_value) + tl.dot(q_dot, state_lo_value)
                    else:
                        if Q_IS_BF16:
                            q_dot = q_f.to(tl.bfloat16)
                        else:
                            q_dot = q_f.to(tl.float16)
                        state_f = tl.load(
                            states + state_offset,
                            mask=(k_idx[:, None] < DK) & (v_idx[None, :] < DV),
                            other=0.0,
                        ).to(tl.float32)
                        if Q_IS_BF16:
                            state_hi_value = state_f.to(tl.bfloat16)
                            state_lo_value = (state_f - state_hi_value.to(tl.float32)).to(tl.bfloat16)
                        else:
                            state_hi_value = state_f.to(tl.float16)
                            state_lo_value = (state_f - state_hi_value.to(tl.float32)).to(tl.float16)
                        read_value += tl.dot(q_dot, state_hi_value) + tl.dot(q_dot, state_lo_value)
                do = tl.load(
                    dout + ((batch * T + token[:, None]) * H + head) * DV + v_idx[None, :],
                    mask=n_mask[:, None] & (v_idx[None, :] < DV),
                    other=0.0,
                ).to(tl.float32)
                acc += tl.sum(read_value * do, axis=1)
            tl.store(dweights + selection, acc * READ_SCALE, mask=n_mask)
            pos += GROUP_SPLITS * BLOCK_N

    @triton.jit
    def _sparse_topk_ds_grouped_kernel(
        read_q,
        weights,
        dout,
        selection_ids,
        counts,
        offsets,
        dstates,
        T: tl.constexpr,
        M,
        H: tl.constexpr,
        DK: tl.constexpr,
        DV: tl.constexpr,
        TOPK: tl.constexpr,
        READ_SCALE: tl.constexpr,
        Q_IS_BF16: tl.constexpr,
        USE_TF32_STATE: tl.constexpr,
        SINGLE_BF16_DOT: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        i_v = tl.program_id(0)
        i_k = tl.program_id(1)
        group = tl.program_id(2)
        head = (group // M) % H
        batch = group // (M * H)
        summary = group % M
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = i_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_v = i_v * BLOCK_V + tl.arange(0, BLOCK_V)
        k_mask = offs_k < DK
        v_mask = offs_v < DV
        start = tl.load(offsets + group).to(tl.int32)
        count = tl.load(counts + group).to(tl.int32)
        acc = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
        pos = 0
        while pos < count:
            n_mask = offs_n < count - pos
            selection = tl.load(selection_ids + start + pos + offs_n, mask=n_mask, other=0).to(tl.int32)
            flat_th = selection // TOPK
            flat_bt = flat_th // H
            token = flat_bt % T
            q_value = tl.load(
                read_q + ((batch * T + token[:, None]) * H + head) * DK + offs_k[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            do_f = tl.load(
                dout + ((batch * T + token[:, None]) * H + head) * DV + offs_v[None, :],
                mask=n_mask[:, None] & v_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(weights + selection, mask=n_mask, other=0.0).to(tl.float32)
            weighted_do = do_f * weight[:, None] * READ_SCALE
            if USE_TF32_STATE:
                acc += tl.dot(tl.trans(q_value.to(tl.float32)), weighted_do, input_precision="tf32")
            elif SINGLE_BF16_DOT:
                acc += tl.dot(
                    tl.trans(q_value.to(tl.bfloat16)),
                    weighted_do.to(tl.bfloat16),
                )
            elif Q_IS_BF16:
                q_dot = q_value
                do_hi = weighted_do.to(tl.bfloat16)
                do_lo = (weighted_do - do_hi.to(tl.float32)).to(tl.bfloat16)
                acc += tl.dot(tl.trans(q_dot), do_hi) + tl.dot(tl.trans(q_dot), do_lo)
            else:
                q_dot = q_value
                do_hi = weighted_do.to(tl.float16)
                do_lo = (weighted_do - do_hi.to(tl.float32)).to(tl.float16)
                acc += tl.dot(tl.trans(q_dot), do_hi) + tl.dot(tl.trans(q_dot), do_lo)
            pos += BLOCK_N
        tl.store(
            dstates
            + ((((batch * M + summary) * H + head) * DK + offs_k[:, None]) * DV + offs_v[None, :]),
            acc,
            mask=k_mask[:, None] & v_mask[None, :],
        )

    @triton.jit
    def _sparse_topk_fwd_token_kernel(
        read_q,
        states,
        indices,
        weights,
        valid,
        out,
        BTH: tl.constexpr,
        T: tl.constexpr,
        M,
        H: tl.constexpr,
        DK: tl.constexpr,
        DV: tl.constexpr,
        TOPK: tl.constexpr,
        READ_SCALE: tl.constexpr,
        Q_IS_BF16: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_K_BLOCKS: tl.constexpr,
    ):
        flat_th = tl.program_id(0)
        i_v = tl.program_id(1)
        batch = flat_th // (T * H)
        head = flat_th % H
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = i_v * BLOCK_V + tl.arange(0, BLOCK_V)
        v_mask = offs_v < DV
        out_acc = tl.zeros((BLOCK_V,), dtype=tl.float32)

        # TOPK is a compile-time bucket.  Each program owns one token/head and
        # output-value tile, so it can combine all selected states in registers
        # and write the result exactly once without CSR or atomics.
        for slot in range(TOPK):
            selection = flat_th * TOPK + slot
            is_valid = tl.load(valid + selection).to(tl.int1)
            summary = tl.load(indices + selection).to(tl.int32)
            summary = tl.where(is_valid, summary, 0)
            weight = tl.load(weights + selection).to(tl.float32)
            read_acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for i_k in range(NUM_K_BLOCKS):
                k_idx = i_k * BLOCK_K + offs_k
                k_mask = k_idx < DK
                q_f = tl.load(
                    read_q + flat_th * DK + k_idx,
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)
                state_f = tl.load(
                    states
                    + ((((batch * M + summary) * H + head) * DK + k_idx[:, None]) * DV + offs_v[None, :]),
                    mask=is_valid & k_mask[:, None] & v_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                if Q_IS_BF16:
                    q_dot = q_f.to(tl.bfloat16)
                    state_hi = state_f.to(tl.bfloat16)
                    state_lo = (state_f - state_hi.to(tl.float32)).to(tl.bfloat16)
                else:
                    q_dot = q_f.to(tl.float16)
                    state_hi = state_f.to(tl.float16)
                    state_lo = (state_f - state_hi.to(tl.float32)).to(tl.float16)
                q_fp32 = q_dot.to(tl.float32)
                read_acc += tl.sum(
                    q_fp32[:, None] * state_hi.to(tl.float32)
                    + q_fp32[:, None] * state_lo.to(tl.float32),
                    axis=0,
                )
            out_acc += tl.where(is_valid, weight, 0.0) * read_acc

        tl.store(out + flat_th * DV + offs_v, out_acc * READ_SCALE, mask=(flat_th < BTH) & v_mask)

    @triton.jit
    def _sparse_topk_dq_token_kernel(
        read_q,
        states,
        indices,
        weights,
        valid,
        dout,
        dread,
        dweights,
        BTH: tl.constexpr,
        T: tl.constexpr,
        M,
        H: tl.constexpr,
        DK: tl.constexpr,
        DV: tl.constexpr,
        TOPK: tl.constexpr,
        READ_SCALE: tl.constexpr,
        Q_IS_BF16: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_V_BLOCKS: tl.constexpr,
        FUSE_DW: tl.constexpr,
    ):
        flat_th = tl.program_id(0)
        i_k = tl.program_id(1)
        batch = flat_th // (T * H)
        head = flat_th % H
        offs_k = i_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_v = tl.arange(0, BLOCK_V)
        k_mask = offs_k < DK
        dq_acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

        for slot in range(TOPK):
            selection = flat_th * TOPK + slot
            is_valid = tl.load(valid + selection).to(tl.int1)
            summary = tl.load(indices + selection).to(tl.int32)
            summary = tl.where(is_valid, summary, 0)
            weight = tl.load(weights + selection).to(tl.float32)
            slot_acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for i_v in range(NUM_V_BLOCKS):
                v_idx = i_v * BLOCK_V + offs_v
                v_mask = v_idx < DV
                do_f = tl.load(
                    dout + flat_th * DV + v_idx,
                    mask=v_mask,
                    other=0.0,
                ).to(tl.float32)
                state_f = tl.load(
                    states
                    + ((((batch * M + summary) * H + head) * DK + offs_k[:, None]) * DV + v_idx[None, :]),
                    mask=is_valid & k_mask[:, None] & v_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                if Q_IS_BF16:
                    do_dot = do_f.to(tl.bfloat16)
                    state_hi = state_f.to(tl.bfloat16)
                    state_lo = (state_f - state_hi.to(tl.float32)).to(tl.bfloat16)
                else:
                    do_dot = do_f.to(tl.float16)
                    state_hi = state_f.to(tl.float16)
                    state_lo = (state_f - state_hi.to(tl.float32)).to(tl.float16)
                do_fp32 = do_dot.to(tl.float32)
                slot_acc += tl.sum(
                    state_hi.to(tl.float32) * do_fp32[None, :]
                    + state_lo.to(tl.float32) * do_fp32[None, :],
                    axis=1,
                )
            if FUSE_DW:
                # slot_acc is exactly H_m @ dO for this K tile.  Reuse it for
                # dWeight instead of launching the token dWeight kernel, which
                # would reread and recontract the complete selected state.
                q_f = tl.load(
                    read_q + flat_th * DK + offs_k,
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)
                dw_partial = tl.sum(q_f * slot_acc, axis=0) * READ_SCALE
                tl.atomic_add(dweights + selection, dw_partial, mask=is_valid)
            dq_acc += tl.where(is_valid, weight, 0.0) * slot_acc

        tl.store(dread + flat_th * DK + offs_k, dq_acc * READ_SCALE, mask=(flat_th < BTH) & k_mask)

    @triton.jit
    def _sparse_topk_dw_token_kernel(
        read_q,
        states,
        indices,
        valid,
        dout,
        dweights,
        BTH: tl.constexpr,
        T: tl.constexpr,
        M,
        H: tl.constexpr,
        DK: tl.constexpr,
        DV: tl.constexpr,
        TOPK: tl.constexpr,
        READ_SCALE: tl.constexpr,
        Q_IS_BF16: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_K_BLOCKS: tl.constexpr,
        NUM_V_BLOCKS: tl.constexpr,
    ):
        flat_th = tl.program_id(0)
        slot = tl.program_id(1)
        selection = flat_th * TOPK + slot
        lane = tl.arange(0, 1)
        is_valid = tl.load(valid + selection).to(tl.int1)
        summary = tl.load(indices + selection).to(tl.int32)
        summary = tl.where(is_valid, summary, 0)
        batch = flat_th // (T * H)
        head = flat_th % H
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = tl.arange(0, BLOCK_V)
        acc = tl.zeros((1,), dtype=tl.float32)

        for i_v in range(NUM_V_BLOCKS):
            v_idx = i_v * BLOCK_V + offs_v
            v_mask = v_idx < DV
            read_value = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for i_k in range(NUM_K_BLOCKS):
                k_idx = i_k * BLOCK_K + offs_k
                k_mask = k_idx < DK
                q_f = tl.load(
                    read_q + flat_th * DK + k_idx,
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)
                state_f = tl.load(
                    states
                    + ((((batch * M + summary) * H + head) * DK + k_idx[:, None]) * DV + v_idx[None, :]),
                    mask=is_valid & k_mask[:, None] & v_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                if Q_IS_BF16:
                    q_dot = q_f.to(tl.bfloat16)
                    state_hi = state_f.to(tl.bfloat16)
                    state_lo = (state_f - state_hi.to(tl.float32)).to(tl.bfloat16)
                else:
                    q_dot = q_f.to(tl.float16)
                    state_hi = state_f.to(tl.float16)
                    state_lo = (state_f - state_hi.to(tl.float32)).to(tl.float16)
                q_fp32 = q_dot.to(tl.float32)
                read_value += tl.sum(
                    q_fp32[:, None] * state_hi.to(tl.float32)
                    + q_fp32[:, None] * state_lo.to(tl.float32),
                    axis=0,
                )
            do_f = tl.load(dout + flat_th * DV + v_idx, mask=v_mask, other=0.0).to(tl.float32)
            acc += tl.sum(read_value * do_f, axis=0)

        tl.store(dweights + selection + lane, tl.where(is_valid, acc * READ_SCALE, 0.0))

else:
    _topk_csr_count_kernel = None
    _topk_csr_offsets_kernel = None
    _topk_csr_fill_kernel = None
    _split_fp32_high_residual_kernel = None
    _sparse_topk_fwd_grouped_kernel = None
    _sparse_topk_fwd_reduce_kernel = None
    _sparse_topk_dq_grouped_kernel = None
    _sparse_topk_dq_dw_grouped_kernel = None
    _sparse_topk_dw_grouped_kernel = None
    _sparse_topk_ds_grouped_kernel = None
    _sparse_topk_fwd_token_kernel = None
    _sparse_topk_dq_token_kernel = None
    _sparse_topk_dw_token_kernel = None


def _supports_sparse_topk_cuda(
    read_q: torch.Tensor,
    states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk: int,
) -> bool:
    return bool(
        triton is not None
        and _sparse_topk_fwd_grouped_kernel is not None
        and read_q.is_cuda
        and read_q.dtype in (torch.bfloat16, torch.float16)
        and states.dtype == torch.float32
        and topk_weights.dtype == torch.float32
        and topk in _SUPPORTED_TOPK
        and 16 <= int(read_q.shape[-1]) <= 256
        and int(states.shape[-1]) >= 16
    )


def _supports_sparse_topk_token_cuda(
    read_q: torch.Tensor,
    states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk: int,
) -> bool:
    return bool(
        _supports_sparse_topk_cuda(read_q, states, topk_weights, topk)
        and _sparse_topk_fwd_token_kernel is not None
        and _sparse_topk_dq_token_kernel is not None
        and _sparse_topk_dw_token_kernel is not None
    )


def _maybe_prepack_sparse_state(
    states: torch.Tensor,
    read_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if not _env_bool("HSD_TOPK_PREPACK_STATE", False):
        return states, states, False
    if _split_fp32_high_residual_kernel is None or not states.is_cuda:
        raise RuntimeError("HSD_TOPK_PREPACK_STATE requires the Triton CUDA state-split kernel.")
    high = torch.empty(states.shape, device=states.device, dtype=read_dtype)
    residual = torch.empty_like(high)
    total = int(states.numel())
    block = 256
    _split_fp32_high_residual_kernel[(triton.cdiv(total, block),)](
        states,
        high,
        residual,
        total,
        read_dtype == torch.bfloat16,
        BLOCK=block,
        num_warps=4,
    )
    return high, residual, True


def _build_topk_csr(
    indices: torch.Tensor,
    valid: torch.Tensor,
    *,
    batch: int,
    text_len: int,
    heads: int,
    summary_count: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total = int(indices.numel())
    groups = batch * heads * summary_count
    counts = torch.zeros(groups, device=indices.device, dtype=torch.int32)
    block = _env_int("HSD_TOPK_CSR_BLOCK", 256, allowed={128, 256, 512, 1024})
    _topk_csr_count_kernel[(triton.cdiv(total, block),)](
        indices,
        valid,
        counts,
        total,
        text_len,
        heads,
        summary_count,
        topk,
        BLOCK=block,
        num_warps=4,
    )
    offsets = torch.empty(groups + 1, device=indices.device, dtype=torch.int32)
    cursors = torch.empty_like(counts)
    # Production packed HSD has B=1 and only a few hundred (head, summary)
    # groups.  A single-CTA scan avoids the separate PyTorch cumsum, slice
    # copy, scalar store, and cursor memset launches.  Retain the general
    # fallback for unusually large ordinary batches.
    if groups <= 1024 and _topk_csr_offsets_kernel is not None:
        prefix_block = _next_power_of_2(groups)
        _topk_csr_offsets_kernel[(1,)](
            counts,
            offsets,
            cursors,
            groups,
            BLOCK=prefix_block,
            num_warps=4,
        )
    else:
        offsets[0] = 0
        offsets[1:] = counts.cumsum(dim=0, dtype=torch.int32)
        cursors.zero_()
    selection_ids = torch.empty(total, device=indices.device, dtype=torch.int32)
    _topk_csr_fill_kernel[(triton.cdiv(total, block),)](
        indices,
        valid,
        offsets,
        cursors,
        selection_ids,
        total,
        text_len,
        heads,
        summary_count,
        topk,
        BLOCK=block,
        num_warps=4,
    )
    return selection_ids, counts, offsets


class _HSDSparseTopKGlobalReadFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        read_q: torch.Tensor,
        states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_valid: torch.Tensor,
        read_scale: float,
    ) -> torch.Tensor:
        batch, text_len, heads, summary_count, key_dim, value_dim, topk = _validate_sparse_topk_global_read(
            read_q, states, topk_indices, topk_weights, topk_valid
        )
        q = read_q.contiguous()
        state = states.contiguous()
        precision_mode = _env_choice(
            "HSD_TOPK_STATE_PRECISION_MODE",
            "high_residual",
            allowed={"bf16_high", "high_residual", "tf32"},
        )
        if precision_mode == "bf16_high" and q.dtype != torch.bfloat16:
            raise RuntimeError(
                "HSD_TOPK_STATE_PRECISION_MODE=bf16_high requires BF16 read_q."
            )
        state_read, state_read_lo, state_prepacked = _maybe_prepack_sparse_state(state, q.dtype)
        use_tf32_state = precision_mode == "tf32" and not state_prepacked
        single_bf16_dot = precision_mode == "bf16_high"
        indices = topk_indices.to(dtype=torch.int32).contiguous()
        weights = topk_weights.contiguous()
        valid = topk_valid.contiguous()
        selection_ids, counts, offsets = _build_topk_csr(
            indices,
            valid,
            batch=batch,
            text_len=text_len,
            heads=heads,
            summary_count=summary_count,
            topk=topk,
        )
        block_n = _env_int("HSD_TOPK_BLOCK_N", 64, allowed={16, 32, 64, 128})
        block_k = _env_int("HSD_TOPK_BLOCK_K", 64, allowed={16, 32, 64})
        # GPU production shape (Dv=384) is substantially faster at 64 than
        # 128 because the extra V tile occupancy outweighs the additional
        # loop iterations.  Keep small dimensions on their natural power of 2.
        default_block_v = 64 if value_dim >= 64 else _next_power_of_2(value_dim)
        block_v = _env_int("HSD_TOPK_BLOCK_V", default_block_v, allowed={16, 32, 64, 128})
        num_warps = _env_int("HSD_TOPK_NUM_WARPS", 4, allowed={4, 8})
        num_stages = _env_int("HSD_TOPK_NUM_STAGES", 1, allowed={1, 2, 3, 4})
        group_splits = _env_int("HSD_TOPK_GROUP_SPLITS", 1, allowed={1, 2, 4, 8, 16})
        fwd_group_splits = _env_int(
            "HSD_TOPK_FWD_GROUP_SPLITS",
            DEFAULT_TOPK_FWD_GROUP_SPLITS,
            allowed={1, 2, 4, 8, 16},
        )
        groups = batch * heads * summary_count
        # GPU production measurements show that the one-launch atomic path is
        # faster for K4/K8/K16 across 4K--128K.  It also removes the
        # selection-by-BV staged buffer and the six compute/reduce launch
        # pairs.  Keep the staged implementation as an explicit low-workspace
        # rollback for K4 via HSD_TOPK_STAGED_FORWARD=1.
        staged_forward = _env_bool("HSD_TOPK_STAGED_FORWARD", False)
        if staged_forward:
            total_selections = int(indices.numel())
            staged_contrib = torch.empty(
                total_selections, block_v, device=q.device, dtype=torch.float32
            )
            out = torch.empty(batch, text_len, heads, value_dim, device=q.device, dtype=q.dtype)
            reduce_block_t = _env_int("HSD_TOPK_REDUCE_BLOCK_T", 16, allowed={16, 32, 64, 128})
            bth = batch * text_len * heads
            for v_start in range(0, value_dim, block_v):
                _sparse_topk_fwd_grouped_kernel[(1, groups, fwd_group_splits)](
                    q,
                    state_read,
                    state_read_lo,
                    weights,
                    selection_ids,
                    counts,
                    offsets,
                    out,
                    staged_contrib,
                    text_len,
                    summary_count,
                    heads,
                    key_dim,
                    value_dim,
                    topk,
                    float(read_scale),
                    q.dtype == torch.bfloat16,
                    state_prepacked,
                    use_tf32_state,
                    single_bf16_dot,
                    v_start,
                    True,
                    False,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    BLOCK_V=block_v,
                    NUM_K_BLOCKS=triton.cdiv(key_dim, block_k),
                    GROUP_SPLITS=fwd_group_splits,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
                _sparse_topk_fwd_reduce_kernel[(triton.cdiv(bth, reduce_block_t),)](
                    staged_contrib,
                    valid,
                    out,
                    bth,
                    value_dim,
                    topk,
                    v_start,
                    BLOCK_T=reduce_block_t,
                    BLOCK_V=block_v,
                    num_warps=4,
                )
        else:
            # Low-precision atomics are opt-in because every selected summary
            # rounds the partial sum to the output dtype.  They remove the
            # FP32 accumulator and the trailing cast, but can trade numerical
            # accuracy (and architecture-dependent atomic throughput) for
            # lower memory traffic.  The explicit grouped/atomic-scoped name
            # prevents this switch from changing staged or token execution.
            atomic_output_dtype = _env_choice(
                "HSD_TOPK_GROUPED_ATOMIC_OUTPUT_DTYPE",
                "fp32",
                allowed={"fp32", "input"},
            )
            atomic_output_is_input = atomic_output_dtype == "input"
            out_acc = torch.zeros(
                batch,
                text_len,
                heads,
                value_dim,
                device=q.device,
                dtype=q.dtype if atomic_output_is_input else torch.float32,
            )
            _sparse_topk_fwd_grouped_kernel[(triton.cdiv(value_dim, block_v), groups, fwd_group_splits)](
                q,
                state_read,
                state_read_lo,
                weights,
                selection_ids,
                counts,
                offsets,
                out_acc,
                out_acc,
                text_len,
                summary_count,
                heads,
                key_dim,
                value_dim,
                topk,
                float(read_scale),
                q.dtype == torch.bfloat16,
                state_prepacked,
                use_tf32_state,
                single_bf16_dot,
                0,
                False,
                atomic_output_is_input,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                BLOCK_V=block_v,
                NUM_K_BLOCKS=triton.cdiv(key_dim, block_k),
                GROUP_SPLITS=fwd_group_splits,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            out = out_acc if atomic_output_is_input else out_acc.to(dtype=q.dtype)
        ctx.read_scale = float(read_scale)
        ctx.group_splits = group_splits
        ctx.state_shape = tuple(state.shape)
        ctx.state_prepacked = state_prepacked
        ctx.use_tf32_state = use_tf32_state
        ctx.single_bf16_dot = single_bf16_dot
        ctx.save_for_backward(q, state_read, state_read_lo, weights, selection_ids, counts, offsets)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        read_q, states, states_lo, weights, selection_ids, counts, offsets = ctx.saved_tensors
        batch, text_len, heads, key_dim = read_q.shape
        summary_count = int(states.shape[1])
        value_dim = int(states.shape[-1])
        topk = int(weights.shape[-1])
        groups = batch * heads * summary_count
        needs = ctx.needs_input_grad
        dout_c = dout.contiguous()
        block_n = _env_int("HSD_TOPK_BLOCK_N", 64, allowed={16, 32, 64, 128})
        block_k = _env_int("HSD_TOPK_BLOCK_K", 64, allowed={16, 32, 64})
        default_block_v = 64 if value_dim >= 64 else _next_power_of_2(value_dim)
        block_v = _env_int("HSD_TOPK_BLOCK_V", default_block_v, allowed={16, 32, 64, 128})
        num_warps = _env_int("HSD_TOPK_NUM_WARPS", 4, allowed={4, 8})
        num_stages = _env_int("HSD_TOPK_NUM_STAGES", 1, allowed={1, 2, 3, 4})
        group_splits = int(ctx.group_splits)
        dq_group_splits = _env_int(
            "HSD_TOPK_DQ_GROUP_SPLITS",
            DEFAULT_TOPK_DQ_GROUP_SPLITS,
            allowed={1, 2, 4, 8, 16},
        )
        dw_group_splits = _env_int(
            "HSD_TOPK_DW_GROUP_SPLITS", group_splits, allowed={1, 2, 4, 8, 16}
        )
        q_is_bf16 = read_q.dtype == torch.bfloat16
        dqdw_block_n = _env_int("HSD_TOPK_DQDW_BLOCK_N", block_n, allowed={16, 32, 64, 128})
        dqdw_block_k = _env_int(
            "HSD_TOPK_DQDW_BLOCK_K",
            DEFAULT_TOPK_DQDW_BLOCK_K,
            allowed={16, 32, 64},
        )
        dqdw_block_v = _env_int("HSD_TOPK_DQDW_BLOCK_V", block_v, allowed={16, 32, 64, 128})
        dqdw_num_warps = _env_int("HSD_TOPK_DQDW_NUM_WARPS", num_warps, allowed={4, 8})
        dqdw_num_stages = _env_int("HSD_TOPK_DQDW_NUM_STAGES", num_stages, allowed={1, 2, 3, 4})
        num_k_blocks = triton.cdiv(key_dim, dqdw_block_k)
        num_v_blocks = triton.cdiv(value_dim, dqdw_block_v)

        fuse_dq_dw = bool(
            needs[0]
            and needs[3]
            and _sparse_topk_dq_dw_grouped_kernel is not None
            and _env_bool("HSD_TOPK_FUSE_DQ_DW", True)
        )
        dread = None
        dweights = None
        if fuse_dq_dw:
            dq_acc = torch.zeros(batch, text_len, heads, key_dim, device=read_q.device, dtype=torch.float32)
            dweights = torch.zeros_like(weights)
            _sparse_topk_dq_dw_grouped_kernel[(num_k_blocks, groups, dq_group_splits)](
                read_q,
                states,
                states_lo,
                weights,
                dout_c,
                selection_ids,
                counts,
                offsets,
                dq_acc,
                dweights,
                text_len,
                summary_count,
                heads,
                key_dim,
                value_dim,
                topk,
                ctx.read_scale,
                q_is_bf16,
                ctx.state_prepacked,
                ctx.use_tf32_state,
                ctx.single_bf16_dot,
                BLOCK_N=dqdw_block_n,
                BLOCK_K=dqdw_block_k,
                BLOCK_V=dqdw_block_v,
                NUM_V_BLOCKS=num_v_blocks,
                GROUP_SPLITS=dq_group_splits,
                num_warps=dqdw_num_warps,
                num_stages=dqdw_num_stages,
            )
            dread = dq_acc.to(dtype=read_q.dtype)
        elif needs[0]:
            dq_acc = torch.zeros(batch, text_len, heads, key_dim, device=read_q.device, dtype=torch.float32)
            _sparse_topk_dq_grouped_kernel[(num_k_blocks, groups, dq_group_splits)](
                states,
                states_lo,
                weights,
                dout_c,
                selection_ids,
                counts,
                offsets,
                dq_acc,
                text_len,
                summary_count,
                heads,
                key_dim,
                value_dim,
                topk,
                ctx.read_scale,
                q_is_bf16,
                ctx.state_prepacked,
                ctx.use_tf32_state,
                ctx.single_bf16_dot,
                BLOCK_N=dqdw_block_n,
                BLOCK_K=dqdw_block_k,
                BLOCK_V=dqdw_block_v,
                NUM_V_BLOCKS=num_v_blocks,
                GROUP_SPLITS=dq_group_splits,
                num_warps=dqdw_num_warps,
                num_stages=dqdw_num_stages,
            )
            dread = dq_acc.to(dtype=read_q.dtype)

        dstates = None
        if needs[1]:
            ds_block_n = _env_int("HSD_TOPK_DS_BLOCK_N", block_n, allowed={16, 32, 64, 128})
            ds_block_k = _env_int("HSD_TOPK_DS_BLOCK_K", block_k, allowed={16, 32, 64})
            ds_block_v = _env_int("HSD_TOPK_DS_BLOCK_V", block_v, allowed={16, 32, 64, 128})
            ds_num_warps = _env_int("HSD_TOPK_DS_NUM_WARPS", num_warps, allowed={4, 8})
            ds_num_stages = _env_int("HSD_TOPK_DS_NUM_STAGES", num_stages, allowed={1, 2, 3, 4})
            dstates = torch.empty(ctx.state_shape, device=read_q.device, dtype=torch.float32)
            _sparse_topk_ds_grouped_kernel[
                (triton.cdiv(value_dim, ds_block_v), triton.cdiv(key_dim, ds_block_k), groups)
            ](
                read_q,
                weights,
                dout_c,
                selection_ids,
                counts,
                offsets,
                dstates,
                text_len,
                summary_count,
                heads,
                key_dim,
                value_dim,
                topk,
                ctx.read_scale,
                q_is_bf16,
                ctx.use_tf32_state,
                ctx.single_bf16_dot,
                BLOCK_N=ds_block_n,
                BLOCK_K=ds_block_k,
                BLOCK_V=ds_block_v,
                num_warps=ds_num_warps,
                num_stages=ds_num_stages,
            )

        if needs[3] and not fuse_dq_dw:
            dweights = torch.zeros_like(weights)
            _sparse_topk_dw_grouped_kernel[(groups, dw_group_splits)](
                read_q,
                states,
                states_lo,
                dout_c,
                selection_ids,
                counts,
                offsets,
                dweights,
                text_len,
                summary_count,
                heads,
                key_dim,
                value_dim,
                topk,
                ctx.read_scale,
                q_is_bf16,
                ctx.state_prepacked,
                ctx.use_tf32_state,
                ctx.single_bf16_dot,
                BLOCK_N=dqdw_block_n,
                BLOCK_K=dqdw_block_k,
                BLOCK_V=dqdw_block_v,
                NUM_K_BLOCKS=num_k_blocks,
                NUM_V_BLOCKS=num_v_blocks,
                GROUP_SPLITS=dw_group_splits,
                num_warps=dqdw_num_warps,
                num_stages=dqdw_num_stages,
            )
        return dread, dstates, None, dweights, None, None


class _HSDSparseTopKTokenReadFunction(torch.autograd.Function):
    """CSR-free token-owned forward/dQ/dWeight; grouped CSR is only used by dState."""

    @staticmethod
    def forward(
        ctx,
        read_q: torch.Tensor,
        states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_valid: torch.Tensor,
        read_scale: float,
    ) -> torch.Tensor:
        batch, text_len, heads, summary_count, key_dim, value_dim, topk = _validate_sparse_topk_global_read(
            read_q, states, topk_indices, topk_weights, topk_valid
        )
        q = read_q.contiguous()
        state = states.contiguous()
        indices = topk_indices.to(dtype=torch.int32).contiguous()
        weights = topk_weights.contiguous()
        valid = topk_valid.contiguous()
        block_k = _env_int("HSD_TOPK_TOKEN_BLOCK_K", 32, allowed={16, 32, 64})
        block_v = _env_int("HSD_TOPK_TOKEN_BLOCK_V", 64, allowed={16, 32, 64, 128})
        num_warps = _env_int("HSD_TOPK_TOKEN_NUM_WARPS", 4, allowed={4, 8})
        num_stages = _env_int("HSD_TOPK_TOKEN_NUM_STAGES", 1, allowed={1, 2, 3, 4})
        bth = batch * text_len * heads
        out = torch.empty(batch, text_len, heads, value_dim, device=q.device, dtype=q.dtype)
        _sparse_topk_fwd_token_kernel[(bth, triton.cdiv(value_dim, block_v))](
            q,
            state,
            indices,
            weights,
            valid,
            out,
            bth,
            text_len,
            summary_count,
            heads,
            key_dim,
            value_dim,
            topk,
            float(read_scale),
            q.dtype == torch.bfloat16,
            BLOCK_K=block_k,
            BLOCK_V=block_v,
            NUM_K_BLOCKS=triton.cdiv(key_dim, block_k),
            num_warps=num_warps,
            num_stages=num_stages,
        )
        ctx.read_scale = float(read_scale)
        ctx.token_block_k = block_k
        ctx.token_block_v = block_v
        ctx.token_num_warps = num_warps
        ctx.token_num_stages = num_stages
        ctx.save_for_backward(q, state, indices, weights, valid)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        read_q, states, indices, weights, valid = ctx.saved_tensors
        batch, text_len, heads, key_dim = read_q.shape
        summary_count = int(states.shape[1])
        value_dim = int(states.shape[-1])
        topk = int(weights.shape[-1])
        bth = batch * text_len * heads
        needs = ctx.needs_input_grad
        dout_c = dout.contiguous()
        block_k = int(ctx.token_block_k)
        block_v = int(ctx.token_block_v)
        num_warps = int(ctx.token_num_warps)
        num_stages = int(ctx.token_num_stages)
        q_is_bf16 = read_q.dtype == torch.bfloat16
        fuse_dq_dw = bool(
            needs[0]
            and needs[3]
            and _env_bool("HSD_TOPK_FUSE_DQ_DW", True)
        )
        dweights = torch.zeros_like(weights) if fuse_dq_dw else None

        dread = None
        if needs[0]:
            dread = torch.empty_like(read_q)
            _sparse_topk_dq_token_kernel[(bth, triton.cdiv(key_dim, block_k))](
                read_q,
                states,
                indices,
                weights,
                valid,
                dout_c,
                dread,
                dweights if dweights is not None else weights,
                bth,
                text_len,
                summary_count,
                heads,
                key_dim,
                value_dim,
                topk,
                ctx.read_scale,
                q_is_bf16,
                BLOCK_K=block_k,
                BLOCK_V=block_v,
                NUM_V_BLOCKS=triton.cdiv(value_dim, block_v),
                FUSE_DW=fuse_dq_dw,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        dstates = None
        if needs[1]:
            # Only dState is a many-token-to-one-summary reduction.  Build the
            # grouped selection list lazily here; forward, dQ, and dWeight never
            # pay the CSR construction or storage cost in token mode.
            selection_ids, counts, offsets = _build_topk_csr(
                indices,
                valid,
                batch=batch,
                text_len=text_len,
                heads=heads,
                summary_count=summary_count,
                topk=topk,
            )
            block_n = _env_int("HSD_TOPK_BLOCK_N", 64, allowed={16, 32, 64, 128})
            state_block_k = _env_int("HSD_TOPK_BLOCK_K", 64, allowed={16, 32, 64})
            default_block_v = 64 if value_dim >= 64 else _next_power_of_2(value_dim)
            state_block_v = _env_int("HSD_TOPK_BLOCK_V", default_block_v, allowed={16, 32, 64, 128})
            state_num_warps = _env_int("HSD_TOPK_NUM_WARPS", 4, allowed={4, 8})
            state_num_stages = _env_int("HSD_TOPK_NUM_STAGES", 1, allowed={1, 2, 3, 4})
            groups = batch * heads * summary_count
            dstates = torch.empty_like(states)
            _sparse_topk_ds_grouped_kernel[
                (triton.cdiv(value_dim, state_block_v), triton.cdiv(key_dim, state_block_k), groups)
            ](
                read_q,
                weights,
                dout_c,
                selection_ids,
                counts,
                offsets,
                dstates,
                text_len,
                summary_count,
                heads,
                key_dim,
                value_dim,
                topk,
                ctx.read_scale,
                q_is_bf16,
                False,
                False,
                BLOCK_N=block_n,
                BLOCK_K=state_block_k,
                BLOCK_V=state_block_v,
                num_warps=state_num_warps,
                num_stages=state_num_stages,
            )

        if needs[3] and not fuse_dq_dw:
            dweights = torch.empty_like(weights)
            _sparse_topk_dw_token_kernel[(bth, topk)](
                read_q,
                states,
                indices,
                valid,
                dout_c,
                dweights,
                bth,
                text_len,
                summary_count,
                heads,
                key_dim,
                value_dim,
                topk,
                ctx.read_scale,
                q_is_bf16,
                BLOCK_K=block_k,
                BLOCK_V=block_v,
                NUM_K_BLOCKS=triton.cdiv(key_dim, block_k),
                NUM_V_BLOCKS=triton.cdiv(value_dim, block_v),
                num_warps=num_warps,
                num_stages=num_stages,
            )
        return dread, dstates, None, dweights, None, None


def hsd_sparse_topk_global_read(
    *,
    read_q: torch.Tensor,
    states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_valid: torch.Tensor,
    read_scale: float,
) -> torch.Tensor:
    """Exact sparse state read for hard top-k routes without a dense candidate-read tensor."""
    _, _, _, _, _, _, topk = _validate_sparse_topk_global_read(
        read_q, states, topk_indices, topk_weights, topk_valid
    )
    execution_mode = _env_choice(
        "HSD_TOPK_EXECUTION_MODE", "grouped", allowed={"grouped", "token"}
    )
    precision_mode = _env_choice(
        "HSD_TOPK_STATE_PRECISION_MODE",
        "high_residual",
        allowed={"bf16_high", "high_residual", "tf32"},
    )
    if precision_mode == "bf16_high":
        if execution_mode != "grouped":
            raise RuntimeError(
                "HSD_TOPK_STATE_PRECISION_MODE=bf16_high requires "
                "HSD_TOPK_EXECUTION_MODE=grouped; token mode has no fallback."
            )
        if read_q.dtype != torch.bfloat16:
            raise RuntimeError(
                "HSD_TOPK_STATE_PRECISION_MODE=bf16_high requires BF16 read_q; "
                f"got {read_q.dtype} with no fallback."
            )
        if not _supports_sparse_topk_cuda(
            read_q, states, topk_weights, topk
        ):
            raise RuntimeError(
                "HSD_TOPK_STATE_PRECISION_MODE=bf16_high requires the supported "
                "grouped CUDA path with FP32 states and FP32 weights; no fallback is allowed."
            )
    if execution_mode == "token" and _supports_sparse_topk_token_cuda(
        read_q, states, topk_weights, topk
    ):
        return _HSDSparseTopKTokenReadFunction.apply(
            read_q,
            states,
            topk_indices,
            topk_weights,
            topk_valid,
            float(read_scale),
        )
    if execution_mode == "grouped" and _supports_sparse_topk_cuda(
        read_q, states, topk_weights, topk
    ):
        return _HSDSparseTopKGlobalReadFunction.apply(
            read_q,
            states,
            topk_indices,
            topk_weights,
            topk_valid,
            float(read_scale),
        )
    if read_q.is_cuda and _env_bool("HSD_TOPK_GLOBAL_READ_STRICT", True):
        raise RuntimeError(
            f"HSD sparse top-k strict mode requested CUDA execution_mode={execution_mode!r}, but it missed."
        )
    return sparse_topk_global_read_reference(
        read_q=read_q,
        states=states,
        topk_indices=topk_indices,
        topk_weights=topk_weights,
        topk_valid=topk_valid,
        read_scale=float(read_scale),
    )


__all__ = [
    "hsd_sparse_topk_global_read",
    "sparse_topk_global_read_reference",
]
