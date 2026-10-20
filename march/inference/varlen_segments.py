"""Optional FLA varlen packed helpers for HSD prefill experiments."""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is optional at import time.
    triton = None
    tl = None


FLA_CHUNK_SIZE = 64
_FLA_OPS: SimpleNamespace | None = None
_FLA_IMPORT_ERROR: Exception | None = None


if triton is not None:

    @triton.heuristics(
        {
            "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        }
    )
    @triton.jit(do_not_specialize=["T"])
    def _hsd_varlen_chunk_gdn_fwd_h_anchor_kernel(
        k,
        v,
        w,
        v_new,
        g,
        h,
        h0,
        ht,
        anchor_states,
        cu_seqlens,
        chunk_offsets,
        anchor_offsets,
        local_anchor_positions,
        T,
        H: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BV: tl.constexpr,
        USE_INITIAL_STATE: tl.constexpr,
    ):
        i_v, i_nh = tl.program_id(0), tl.program_id(1)
        i_n, i_h = i_nh // HV, i_nh % HV
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)

        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

        h += (boh * HV + i_h).to(tl.int64) * K * V
        k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
        w += (bos * HV + i_h).to(tl.int64) * K
        v += (bos * HV + i_h).to(tl.int64) * V
        v_new += (bos * HV + i_h).to(tl.int64) * V
        if USE_INITIAL_STATE:
            h0 += i_nh * K * V
        ht += i_nh * K * V

        if USE_INITIAL_STATE:
            p_h0_1 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
            )
            b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
            if K > 64:
                p_h0_2 = tl.make_block_ptr(
                    h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
                )
                b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
            if K > 128:
                p_h0_3 = tl.make_block_ptr(
                    h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
                )
                b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
            if K > 192:
                p_h0_4 = tl.make_block_ptr(
                    h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
                )
                b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

        o_k = tl.arange(0, 64)
        o_v = tl.arange(0, BV)
        v_offsets = i_v * BV + o_v
        anchor_begin = tl.load(anchor_offsets + i_n).to(tl.int32)
        anchor_end = tl.load(anchor_offsets + i_n + 1).to(tl.int32)
        anchor_idx = anchor_begin
        next_anchor_pos = tl.load(
            local_anchor_positions + anchor_idx,
            mask=anchor_idx < anchor_end,
            other=-1,
        ).to(tl.int32)

        for i_t in range(NT):
            i_t_int64 = i_t.to(tl.int64)
            p_h1 = tl.make_block_ptr(
                h + i_t_int64 * HV * K * V,
                (K, V),
                (V, 1),
                (0, i_v * BV),
                (64, BV),
                (1, 0),
            )
            tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_h2 = tl.make_block_ptr(
                    h + i_t_int64 * HV * K * V,
                    (K, V),
                    (V, 1),
                    (64, i_v * BV),
                    (64, BV),
                    (1, 0),
                )
                tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_h3 = tl.make_block_ptr(
                    h + i_t_int64 * HV * K * V,
                    (K, V),
                    (V, 1),
                    (128, i_v * BV),
                    (64, BV),
                    (1, 0),
                )
                tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_h4 = tl.make_block_ptr(
                    h + i_t_int64 * HV * K * V,
                    (K, V),
                    (V, 1),
                    (192, i_v * BV),
                    (64, BV),
                    (1, 0),
                )
                tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

            chunk_start = i_t * BT
            chunk_end = tl.minimum((i_t + 1) * BT, T)
            o_t = chunk_start + tl.arange(0, BT)
            m_t = o_t < T

            p_w = tl.make_block_ptr(
                w, (T, K), (HV * K, 1), (chunk_start, 0), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
            if K > 64:
                p_w = tl.make_block_ptr(
                    w,
                    (T, K),
                    (HV * K, 1),
                    (chunk_start, 64),
                    (BT, 64),
                    (1, 0),
                )
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
            if K > 128:
                p_w = tl.make_block_ptr(
                    w,
                    (T, K),
                    (HV * K, 1),
                    (chunk_start, 128),
                    (BT, 64),
                    (1, 0),
                )
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
            if K > 192:
                p_w = tl.make_block_ptr(
                    w,
                    (T, K),
                    (HV * K, 1),
                    (chunk_start, 192),
                    (BT, 64),
                    (1, 0),
                )
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
            p_v = tl.make_block_ptr(
                v, (T, V), (HV * V, 1), (chunk_start, i_v * BV), (BT, BV), (1, 0)
            )
            b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

            p_v_new = tl.make_block_ptr(
                v_new,
                (T, V),
                (HV * V, 1),
                (chunk_start, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

            p_g = tl.make_block_ptr(
                g + (bos * HV + i_h).to(tl.int64),
                (T,),
                (HV,),
                (chunk_start,),
                (BT,),
                (0,),
            )
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)

            anchor_in_chunk = (
                (anchor_idx < anchor_end)
                & (next_anchor_pos > chunk_start)
                & (next_anchor_pos <= chunk_end)
            )
            if anchor_in_chunk:
                anchor_last_idx = next_anchor_pos - 1
                b_g_anchor_last = tl.load(
                    g + (bos * HV + anchor_last_idx * HV + i_h).to(tl.int64)
                ).to(tl.float32)
                b_g_anchor_last_exp = tl.exp2(b_g_anchor_last)
                b_v_anchor = b_v * tl.where(
                    o_t < next_anchor_pos, tl.exp2(b_g_anchor_last - b_g), 0
                )[:, None]
                b_v_anchor = b_v_anchor.to(k.dtype.element_ty)

                b_s1 = b_h1 * b_g_anchor_last_exp
                p_k = tl.make_block_ptr(
                    k, (K, T), (1, H * K), (0, chunk_start), (64, BT), (0, 1)
                )
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_s1 += tl.dot(b_k, b_v_anchor)
                if K > 64:
                    b_s2 = b_h2 * b_g_anchor_last_exp
                    p_k = tl.make_block_ptr(
                        k,
                        (K, T),
                        (1, H * K),
                        (64, chunk_start),
                        (64, BT),
                        (0, 1),
                    )
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    b_s2 += tl.dot(b_k, b_v_anchor)
                if K > 128:
                    b_s3 = b_h3 * b_g_anchor_last_exp
                    p_k = tl.make_block_ptr(
                        k,
                        (K, T),
                        (1, H * K),
                        (128, chunk_start),
                        (64, BT),
                        (0, 1),
                    )
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    b_s3 += tl.dot(b_k, b_v_anchor)
                if K > 192:
                    b_s4 = b_h4 * b_g_anchor_last_exp
                    p_k = tl.make_block_ptr(
                        k,
                        (K, T),
                        (1, H * K),
                        (192, chunk_start),
                        (64, BT),
                        (0, 1),
                    )
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    b_s4 += tl.dot(b_k, b_v_anchor)

                anchor_base = anchor_states + (
                    (anchor_idx * HV + i_h).to(tl.int64) * K * V
                )
                p_s1 = anchor_base + o_k[:, None] * V + v_offsets[None, :]
                tl.store(
                    p_s1,
                    b_s1,
                    mask=(o_k[:, None] < K) & (v_offsets[None, :] < V),
                )
                if K > 64:
                    p_s2 = anchor_base + (64 + o_k[:, None]) * V + v_offsets[None, :]
                    tl.store(
                        p_s2,
                        b_s2,
                        mask=((64 + o_k[:, None]) < K) & (v_offsets[None, :] < V),
                    )
                if K > 128:
                    p_s3 = anchor_base + (128 + o_k[:, None]) * V + v_offsets[None, :]
                    tl.store(
                        p_s3,
                        b_s3,
                        mask=((128 + o_k[:, None]) < K) & (v_offsets[None, :] < V),
                    )
                if K > 192:
                    p_s4 = anchor_base + (192 + o_k[:, None]) * V + v_offsets[None, :]
                    tl.store(
                        p_s4,
                        b_s4,
                        mask=((192 + o_k[:, None]) < K) & (v_offsets[None, :] < V),
                    )
                anchor_idx += 1
                next_anchor_pos = tl.load(
                    local_anchor_positions + anchor_idx,
                    mask=anchor_idx < anchor_end,
                    other=-1,
                ).to(tl.int32)

            last_idx = chunk_end - 1
            b_g_last = tl.load(
                g + (bos * HV + last_idx * HV + i_h).to(tl.int64)
            ).to(tl.float32)
            b_v = b_v * tl.where(m_t, tl.exp2(b_g_last - b_g), 0)[:, None]
            b_g_last = tl.exp2(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

            b_v = b_v.to(k.dtype.element_ty)
            p_k = tl.make_block_ptr(
                k, (K, T), (1, H * K), (0, chunk_start), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h1 += tl.dot(b_k, b_v)
            if K > 64:
                p_k = tl.make_block_ptr(
                    k, (K, T), (1, H * K), (64, chunk_start), (64, BT), (0, 1)
                )
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h2 += tl.dot(b_k, b_v)
            if K > 128:
                p_k = tl.make_block_ptr(
                    k, (K, T), (1, H * K), (128, chunk_start), (64, BT), (0, 1)
                )
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h3 += tl.dot(b_k, b_v)
            if K > 192:
                p_k = tl.make_block_ptr(
                    k, (K, T), (1, H * K), (192, chunk_start), (64, BT), (0, 1)
                )
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h4 += tl.dot(b_k, b_v)

        p_ht1 = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht1, b_h1.to(p_ht1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht2 = tl.make_block_ptr(
                ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht2, b_h2.to(p_ht2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht3 = tl.make_block_ptr(
                ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht3, b_h3.to(p_ht3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht4 = tl.make_block_ptr(
                ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht4, b_h4.to(p_ht4.dtype.element_ty), boundary_check=(0, 1))


    @triton.jit
    def _hsd_deterministic_anchor_state_read_kernel(
        anchor_q,
        anchor_states,
        anchor_o,
        scale,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
    ):
        i_mh = tl.program_id(0)
        i_v = tl.program_id(1)
        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)

        q = tl.load(
            anchor_q + i_mh.to(tl.int64) * K + o_k,
            mask=o_k < K,
            other=0.0,
        ).to(tl.float32)
        state = tl.load(
            anchor_states
            + i_mh.to(tl.int64) * K * V
            + o_k[:, None] * V
            + o_v[None, :],
            mask=(o_k[:, None] < K) & (o_v[None, :] < V),
            other=0.0,
        ).to(tl.float32)
        value = tl.sum(q[:, None] * state, axis=0) * scale
        tl.store(
            anchor_o + i_mh.to(tl.int64) * V + o_v,
            value,
            mask=o_v < V,
        )


    @triton.jit(do_not_specialize=["T"])
    def _hsd_deterministic_scaled_dot_kkt_fwd_kernel(
        k,
        beta,
        g,
        A,
        cu_seqlens,
        chunk_indices,
        T,
        H: tl.constexpr,
        HG: tl.constexpr,
        K: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
    ):
        i_chunk_global, i_h = tl.program_id(0), tl.program_id(1)
        i_n = tl.load(chunk_indices + i_chunk_global * 2).to(tl.int32)
        i_chunk = tl.load(chunk_indices + i_chunk_global * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        seq_len = eos - bos
        o_t = i_chunk * BT + tl.arange(0, BT)
        m_t = o_t < seq_len

        p_beta = tl.make_block_ptr(
            beta + bos * H + i_h,
            (seq_len,),
            (H,),
            (i_chunk * BT,),
            (BT,),
            (0,),
        )
        b_beta = tl.load(
            p_beta,
            boundary_check=(0,),
            padding_option="zero",
        )

        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_k = tl.make_block_ptr(
                k + (bos * HG + i_h // (H // HG)) * K,
                (seq_len, K),
                (HG * K, 1),
                (i_chunk * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            b_k = tl.load(
                p_k,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            b_A += tl.dot((b_k * b_beta[:, None]).to(b_k.dtype), tl.trans(b_k))

        p_g = tl.make_block_ptr(
            g + bos * H + i_h,
            (seq_len,),
            (H,),
            (i_chunk * BT,),
            (BT,),
            (0,),
        )
        b_g = tl.load(
            p_g,
            boundary_check=(0,),
            padding_option="zero",
        )
        b_A *= tl.exp(b_g[:, None] - b_g[None, :])

        m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t[None, :])
        b_A = tl.where(m_A, b_A, 0.0)
        p_A = tl.make_block_ptr(
            A + (bos * H + i_h) * BT,
            (seq_len, BT),
            (BT * H, 1),
            (i_chunk * BT, 0),
            (BT, BT),
            (1, 0),
        )
        tl.store(p_A, b_A, boundary_check=(0, 1))


else:
    _hsd_varlen_chunk_gdn_fwd_h_anchor_kernel = None
    _hsd_deterministic_anchor_state_read_kernel = None
    _hsd_deterministic_scaled_dot_kkt_fwd_kernel = None


def deterministic_anchor_state_read(
    anchor_q: torch.Tensor,
    anchor_states: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor | None:
    """Read packed HSD anchor states with a fixed FP32 reduction order."""

    if triton is None or not anchor_q.is_cuda or not anchor_states.is_cuda:
        return None
    if torch.is_grad_enabled():
        return None
    if anchor_q.ndim != 3 or anchor_states.ndim != 4:
        return None
    anchor_count, heads, key_dim = anchor_q.shape
    if anchor_states.shape[:3] != (anchor_count, heads, key_dim):
        return None
    value_dim = int(anchor_states.shape[-1])
    if key_dim <= 0 or key_dim > 256 or value_dim <= 0:
        return None
    if anchor_q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    if anchor_states.dtype != torch.float32:
        return None
    if anchor_q.device != anchor_states.device:
        return None
    if int(anchor_count) == 0:
        return anchor_q.new_empty(0, heads, value_dim)
    if scale is None:
        scale = key_dim**-0.5

    anchor_q = anchor_q.contiguous()
    anchor_states = anchor_states.contiguous()
    output = anchor_q.new_empty(anchor_count, heads, value_dim)
    block_k = triton.next_power_of_2(int(key_dim))
    block_v = 32
    _hsd_deterministic_anchor_state_read_kernel[
        (int(anchor_count) * int(heads), triton.cdiv(value_dim, block_v))
    ](
        anchor_q,
        anchor_states,
        output,
        float(scale),
        H=int(heads),
        K=int(key_dim),
        V=value_dim,
        BK=block_k,
        BV=block_v,
        num_warps=4,
    )
    return output


def deterministic_scaled_dot_kkt_fwd(
    *,
    key: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_size: int = FLA_CHUNK_SIZE,
) -> torch.Tensor | None:
    """Compute packed beta*K*K^T with one fixed, repeat-stable config."""

    if triton is None or torch.is_grad_enabled():
        return None
    if not all(
        tensor.is_cuda
        for tensor in (key, beta, g, cu_seqlens, chunk_indices)
    ):
        return None
    if key.ndim != 4 or key.shape[0] != 1:
        return None
    if beta.ndim != 3 or g.shape != beta.shape:
        return None
    if key.shape[:2] != beta.shape[:2]:
        return None
    if int(chunk_size) != FLA_CHUNK_SIZE:
        return None
    _, text_len, key_heads, key_dim = key.shape
    heads = int(beta.shape[-1])
    if heads % int(key_heads) != 0 or int(key_dim) <= 0:
        return None
    if cu_seqlens.ndim != 1 or chunk_indices.ndim != 2:
        return None

    cu_seqlens = cu_seqlens.to(device=key.device, dtype=torch.int32).contiguous()
    chunk_indices = chunk_indices.to(
        device=key.device,
        dtype=torch.int32,
    ).contiguous()
    output = torch.empty(
        1,
        text_len,
        heads,
        int(chunk_size),
        device=key.device,
        dtype=torch.float32,
    )
    _hsd_deterministic_scaled_dot_kkt_fwd_kernel[
        (int(chunk_indices.shape[0]), heads)
    ](
        key,
        beta,
        g,
        output,
        cu_seqlens,
        chunk_indices,
        text_len,
        H=heads,
        HG=int(key_heads),
        K=int(key_dim),
        BT=int(chunk_size),
        BK=64,
        num_warps=4,
        num_stages=1,
    )
    return output


def _load_fla_ops() -> SimpleNamespace | None:
    global _FLA_OPS, _FLA_IMPORT_ERROR
    if _FLA_OPS is not None:
        return _FLA_OPS
    if _FLA_IMPORT_ERROR is not None:
        return None
    try:
        import triton

        from fla.ops.common.chunk_delta_h import (
            chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
        )
        from fla.ops.common.chunk_o import chunk_fwd_o
        from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
        from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices, prepare_chunk_offsets
        from fla.ops.utils.constant import RCP_LN2
        from fla.utils import custom_device_ctx
        from vllm.model_executor.layers.fla.ops.chunk_delta_h import (
            chunk_gated_delta_rule_fwd_h as vllm_chunk_gated_delta_rule_fwd_h,
        )
        from vllm.model_executor.layers.fla.ops.chunk_o import (
            chunk_fwd_o as vllm_chunk_fwd_o,
        )
        from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (
            chunk_scaled_dot_kkt_fwd as vllm_chunk_scaled_dot_kkt_fwd,
        )
        from vllm.model_executor.layers.fla.ops.cumsum import (
            chunk_local_cumsum as vllm_chunk_local_cumsum,
        )
        from vllm.model_executor.layers.fla.ops.index import (
            prepare_chunk_indices as vllm_prepare_chunk_indices,
        )
        from vllm.model_executor.layers.fla.ops.solve_tril import (
            solve_tril as vllm_solve_tril,
        )
        from vllm.model_executor.layers.fla.ops.wy_fast import (
            recompute_w_u_fwd as vllm_recompute_w_u_fwd,
        )
    except Exception as exc:  # pragma: no cover - optional CUDA dependency.
        _FLA_IMPORT_ERROR = exc
        return None
    _FLA_OPS = SimpleNamespace(
        triton=triton,
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64=chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
        chunk_fwd_o=chunk_fwd_o,
        chunk_gated_delta_rule_fwd_intra=chunk_gated_delta_rule_fwd_intra,
        chunk_local_cumsum=chunk_local_cumsum,
        custom_device_ctx=custom_device_ctx,
        prepare_chunk_indices=prepare_chunk_indices,
        prepare_chunk_offsets=prepare_chunk_offsets,
        RCP_LN2=RCP_LN2,
        vllm_chunk_gated_delta_rule_fwd_h=vllm_chunk_gated_delta_rule_fwd_h,
        vllm_chunk_fwd_o=vllm_chunk_fwd_o,
        vllm_chunk_scaled_dot_kkt_fwd=vllm_chunk_scaled_dot_kkt_fwd,
        vllm_chunk_local_cumsum=vllm_chunk_local_cumsum,
        vllm_prepare_chunk_indices=vllm_prepare_chunk_indices,
        vllm_recompute_w_u_fwd=vllm_recompute_w_u_fwd,
        vllm_solve_tril=vllm_solve_tril,
    )
    return _FLA_OPS


def _normalize_packed_input(tensor: torch.Tensor, *, name: str) -> tuple[torch.Tensor, bool] | None:
    if tensor.ndim == 3:
        return tensor.unsqueeze(0), True
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        return tensor, False
    raise ValueError(f"{name} must be [N, H, D] or [1, N, H, D], got {tuple(tensor.shape)}")


def _normalize_packed_gate(tensor: torch.Tensor, *, name: str) -> tuple[torch.Tensor, bool] | None:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0), True
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor, False
    raise ValueError(f"{name} must be [N, H] or [1, N, H], got {tuple(tensor.shape)}")


def _fallback() -> None:
    return None


def _validate_inputs(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    anchor_offsets: torch.Tensor,
    anchor_chunk_size: int,
    initial_state: torch.Tensor | None,
) -> bool:
    if torch.is_grad_enabled():
        return False
    if not (query.is_cuda and key.is_cuda and value.is_cuda and g.is_cuda and beta.is_cuda):
        return False
    if query.dtype not in (torch.float16, torch.bfloat16) or key.dtype != query.dtype:
        return False
    if value.dtype != query.dtype:
        return False
    if beta.dtype not in (torch.float32, query.dtype):
        return False
    if g.dtype not in (torch.float32, query.dtype):
        return False
    if anchor_chunk_size <= 0 or anchor_chunk_size % FLA_CHUNK_SIZE != 0:
        return False
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or g.ndim != 3 or beta.ndim != 3:
        return False
    if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1 or g.shape[0] != 1 or beta.shape[0] != 1:
        return False
    if key.shape != query.shape or value.shape[:3] != query.shape[:3]:
        return False
    if g.shape != query.shape[:3] or beta.shape != query.shape[:3]:
        return False
    if query.shape[-1] > 256:
        return False
    if cu_seqlens.ndim != 1 or anchor_offsets.ndim != 1:
        return False
    if cu_seqlens.numel() < 2 or anchor_offsets.numel() != cu_seqlens.numel():
        return False
    if int(cu_seqlens[0].item()) != 0 or int(anchor_offsets[0].item()) != 0:
        return False
    if int(cu_seqlens[-1].item()) != query.shape[1]:
        return False
    if initial_state is not None:
        batch_size = cu_seqlens.numel() - 1
        expected = (batch_size, query.shape[2], query.shape[3], value.shape[3])
        if tuple(initial_state.shape) != expected or initial_state.dtype != torch.float32:
            return False
    return True


def _validate_local_anchor_positions(
    *,
    local_anchor_positions: torch.Tensor,
    cu_seqlens: torch.Tensor,
    anchor_offsets: torch.Tensor,
) -> bool:
    if local_anchor_positions.ndim != 1:
        return False
    total_anchor = int(anchor_offsets[-1].item())
    if int(local_anchor_positions.numel()) != total_anchor:
        return False
    if total_anchor == 0:
        return True

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).detach().cpu().tolist()
    anchor_offsets_cpu = anchor_offsets.detach().cpu().tolist()
    positions_cpu = local_anchor_positions.detach().cpu().tolist()
    for seq_idx, seq_len_raw in enumerate(lengths):
        seq_len = int(seq_len_raw)
        begin = int(anchor_offsets_cpu[seq_idx])
        end = int(anchor_offsets_cpu[seq_idx + 1])
        previous = 0
        for out_idx in range(begin, end):
            pos = int(positions_cpu[out_idx])
            if pos <= 0 or pos > seq_len:
                return False
            if pos <= previous:
                return False
            previous = pos
    return True


def _chunk_gated_delta_rule_fwd_h_varlen_fp32(
    ops: SimpleNamespace,
    *,
    key: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, text_len, heads, key_dim = key.shape
    if batch_size != 1:
        raise ValueError("varlen packed helper expects B=1 packed tensors")
    value_heads = int(u.shape[2])
    value_dim = int(u.shape[-1])
    sequence_count = int(cu_seqlens.numel() - 1)
    chunk_count = int(chunk_indices.shape[0])
    chunk_offsets = ops.prepare_chunk_offsets(cu_seqlens, FLA_CHUNK_SIZE)

    chunk_states = torch.empty(
        batch_size,
        chunk_count,
        value_heads,
        key_dim,
        value_dim,
        device=key.device,
        dtype=torch.float32,
    )
    final_state = torch.zeros(
        sequence_count,
        value_heads,
        key_dim,
        value_dim,
        device=key.device,
        dtype=torch.float32,
    )
    v_new = torch.empty_like(u)

    def grid(meta):
        return (
            ops.triton.cdiv(value_dim, meta["BV"]),
            sequence_count * value_heads,
        )

    ops.chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=key,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=None,
        h=chunk_states,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=text_len,
        H=heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BT=FLA_CHUNK_SIZE,
        USE_EXP2=True,
        TRANSPOSE_STATE=False,
    )
    return chunk_states, v_new, final_state


def _chunk_gated_delta_rule_fwd_h_varlen_anchor_fp32(
    ops: SimpleNamespace,
    *,
    key: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    anchor_offsets: torch.Tensor,
    local_anchor_positions: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if triton is None or _hsd_varlen_chunk_gdn_fwd_h_anchor_kernel is None:
        return None
    batch_size, text_len, heads, key_dim = key.shape
    if batch_size != 1:
        raise ValueError("varlen packed helper expects B=1 packed tensors")
    value_heads = int(u.shape[2])
    value_dim = int(u.shape[-1])
    sequence_count = int(cu_seqlens.numel() - 1)
    chunk_count = int(chunk_indices.shape[0])
    anchor_count = int(local_anchor_positions.numel())
    chunk_offsets = ops.prepare_chunk_offsets(cu_seqlens, FLA_CHUNK_SIZE)

    chunk_states = torch.empty(
        batch_size,
        chunk_count,
        value_heads,
        key_dim,
        value_dim,
        device=key.device,
        dtype=torch.float32,
    )
    final_state = torch.zeros(
        sequence_count,
        value_heads,
        key_dim,
        value_dim,
        device=key.device,
        dtype=torch.float32,
    )
    anchor_states = torch.empty(
        anchor_count,
        value_heads,
        key_dim,
        value_dim,
        device=key.device,
        dtype=torch.float32,
    )
    v_new = torch.empty_like(u)

    def grid(meta):
        return (ops.triton.cdiv(value_dim, meta["BV"]), sequence_count * value_heads)

    _hsd_varlen_chunk_gdn_fwd_h_anchor_kernel[grid](
        k=key,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        h=chunk_states,
        h0=initial_state,
        ht=final_state,
        anchor_states=anchor_states,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        anchor_offsets=anchor_offsets,
        local_anchor_positions=local_anchor_positions,
        T=text_len,
        H=heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BT=FLA_CHUNK_SIZE,
        BV=32,
    )
    return chunk_states, v_new, final_state, anchor_states


def _gather_anchor_states(
    *,
    chunk_states: torch.Tensor,
    final_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    anchor_offsets: torch.Tensor,
    anchor_chunk_size: int,
) -> torch.Tensor | None:
    batch_size = cu_seqlens.numel() - 1
    total_anchor = int(anchor_offsets[-1].item())
    heads, key_dim, value_dim = final_state.shape[1:]
    anchor_states = final_state.new_empty(total_anchor, heads, key_dim, value_dim)
    if total_anchor == 0:
        return anchor_states

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    chunks_per_seq = torch.div(lengths + (FLA_CHUNK_SIZE - 1), FLA_CHUNK_SIZE, rounding_mode="floor")
    chunk_offsets = torch.cat([chunks_per_seq.new_zeros(1), chunks_per_seq.cumsum(0)])

    for seq_idx in range(batch_size):
        seq_len = int(lengths[seq_idx].item())
        seq_anchor_begin = int(anchor_offsets[seq_idx].item())
        seq_anchor_end = int(anchor_offsets[seq_idx + 1].item())
        seq_anchor_count = seq_anchor_end - seq_anchor_begin
        max_regular_count = seq_len // anchor_chunk_size
        if seq_anchor_count > max_regular_count:
            return None
        seq_chunk_begin = int(chunk_offsets[seq_idx].item())
        for local_anchor_idx in range(seq_anchor_count):
            boundary = (local_anchor_idx + 1) * anchor_chunk_size
            if boundary > seq_len:
                return None
            out_idx = seq_anchor_begin + local_anchor_idx
            if boundary == seq_len:
                anchor_states[out_idx].copy_(final_state[seq_idx])
            else:
                local_chunk_idx = boundary // FLA_CHUNK_SIZE
                anchor_states[out_idx].copy_(chunk_states[0, seq_chunk_begin + local_chunk_idx])
    return anchor_states


def _vllm_reference_varlen_forward(
    ops: SimpleNamespace,
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    anchor_offsets: torch.Tensor,
    anchor_chunk_size: int,
    initial_state: torch.Tensor | None,
    scale: float,
    use_deterministic_kkt: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Run packed varlen with the exact primitives used by vLLM GDN."""

    # vLLM's GDN segment reference builds cu_seqlens as int32. The Triton
    # varlen kernels accept int64 syntactically, but that specialization is not
    # numerically equivalent for long real HSD inputs and can be unstable
    # across repeated MoE prefills. Keep the packed path on the exact index
    # dtype used by the segment reference.
    cu_seqlens = cu_seqlens.to(device=query.device, dtype=torch.int32)

    chunk_indices = ops.vllm_prepare_chunk_indices(cu_seqlens, FLA_CHUNK_SIZE)
    g_cumsum = ops.vllm_chunk_local_cumsum(
        g,
        chunk_size=FLA_CHUNK_SIZE,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    if use_deterministic_kkt:
        a = deterministic_scaled_dot_kkt_fwd(
            key=key,
            beta=beta,
            g=g_cumsum,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        if a is None:
            return None
    else:
        a = ops.vllm_chunk_scaled_dot_kkt_fwd(
            k=key,
            beta=beta,
            g=g_cumsum,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            output_dtype=torch.float32,
        )
    a = ops.vllm_solve_tril(
        a,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=key.dtype,
    )
    w, u = ops.vllm_recompute_w_u_fwd(
        k=key,
        v=value,
        beta=beta,
        g_cumsum=g_cumsum,
        A=a,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    initial_state_vlk = (
        initial_state.transpose(-1, -2).contiguous()
        if initial_state is not None
        else None
    )
    chunk_states_vlk, v_new, final_state_vlk = (
        ops.vllm_chunk_gated_delta_rule_fwd_h(
            k=key,
            w=w,
            u=u,
            g=g_cumsum,
            initial_state=initial_state_vlk,
            output_final_state=True,
            chunk_size=FLA_CHUNK_SIZE,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            state_dtype=torch.float32,
        )
    )
    text_o = ops.vllm_chunk_fwd_o(
        q=query,
        k=key,
        v=v_new,
        h=chunk_states_vlk.to(dtype=key.dtype),
        g=g_cumsum,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=FLA_CHUNK_SIZE,
    )
    chunk_states = chunk_states_vlk.transpose(-1, -2).contiguous()
    final_state = final_state_vlk.transpose(-1, -2).contiguous()
    anchor_states = _gather_anchor_states(
        chunk_states=chunk_states,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        anchor_offsets=anchor_offsets,
        anchor_chunk_size=anchor_chunk_size,
    )
    if anchor_states is None:
        return None
    return text_o, anchor_states, final_state


def chunk_gated_delta_rule_with_anchor_states_varlen_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    anchor_offsets: torch.Tensor,
    anchor_chunk_size: int,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    local_anchor_positions: torch.Tensor | None = None,
    use_vllm_reference_kernel: bool | None = None,
    use_deterministic_kkt: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Run packed text GDN and extract per-sequence boundary anchor states.

    This helper is intentionally opt-in. It is not wired into the HF model path;
    callers can compare it against the existing segment reference before any
    future integration work.
    """

    ops = _load_fla_ops()
    if ops is None:
        return _fallback()

    query, query_squeezed = _normalize_packed_input(query, name="query")
    key, key_squeezed = _normalize_packed_input(key, name="key")
    value, value_squeezed = _normalize_packed_input(value, name="value")
    g, g_squeezed = _normalize_packed_gate(g, name="g")
    beta, beta_squeezed = _normalize_packed_gate(beta, name="beta")
    squeezed = query_squeezed and key_squeezed and value_squeezed and g_squeezed and beta_squeezed

    if cu_seqlens.device != query.device:
        cu_seqlens = cu_seqlens.to(device=query.device)
    if anchor_offsets.device != query.device:
        anchor_offsets = anchor_offsets.to(device=query.device)
    if local_anchor_positions is not None and local_anchor_positions.device != query.device:
        local_anchor_positions = local_anchor_positions.to(device=query.device)
    anchor_offsets = anchor_offsets.to(torch.long)
    if local_anchor_positions is not None:
        local_anchor_positions = local_anchor_positions.to(torch.long)

    if not _validate_inputs(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        anchor_offsets=anchor_offsets,
        anchor_chunk_size=anchor_chunk_size,
        initial_state=initial_state,
    ):
        return _fallback()
    if local_anchor_positions is not None and not _validate_local_anchor_positions(
        local_anchor_positions=local_anchor_positions,
        cu_seqlens=cu_seqlens,
        anchor_offsets=anchor_offsets,
    ):
        return _fallback()

    scale = key.shape[-1] ** -0.5 if scale is None else scale
    if use_vllm_reference_kernel is None:
        use_vllm_reference_kernel = (
            os.environ.get(
                "VLLM_HSD_VARLEN_VLLM_REFERENCE_KERNEL",
                "1",
            ).strip().lower()
            not in {"0", "false", "no", "n", "off"}
        )
    use_vllm_reference_kernel = bool(
        local_anchor_positions is None and use_vllm_reference_kernel
    )
    cu_seqlens = cu_seqlens.to(
        torch.int32 if use_vllm_reference_kernel else torch.long
    )
    if use_vllm_reference_kernel:
        result = _vllm_reference_varlen_forward(
            ops,
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            anchor_offsets=anchor_offsets,
            anchor_chunk_size=anchor_chunk_size,
            initial_state=initial_state,
            scale=scale,
            use_deterministic_kkt=use_deterministic_kkt,
        )
        if result is None:
            return _fallback()
        text_o, anchor_states, final_state = result
        if squeezed:
            text_o = text_o.squeeze(0)
        return text_o, anchor_states, final_state

    chunk_indices = ops.prepare_chunk_indices(cu_seqlens, FLA_CHUNK_SIZE)
    with ops.custom_device_ctx(query.device.index):
        g_cumsum = ops.chunk_local_cumsum(
            g,
            chunk_size=FLA_CHUNK_SIZE,
            scale=ops.RCP_LN2,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        w, u, _ = ops.chunk_gated_delta_rule_fwd_intra(
            key,
            value,
            g_cumsum,
            beta,
            cu_seqlens=cu_seqlens,
            chunk_size=FLA_CHUNK_SIZE,
            chunk_indices=chunk_indices,
            use_exp2=True,
        )
        if local_anchor_positions is None:
            chunk_states, v_new, final_state = (
                _chunk_gated_delta_rule_fwd_h_varlen_fp32(
                    ops,
                    key=key,
                    w=w,
                    u=u,
                    g=g_cumsum,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices,
                    initial_state=initial_state,
                )
            )
            anchor_states = None
        else:
            anchor_result = _chunk_gated_delta_rule_fwd_h_varlen_anchor_fp32(
                ops,
                key=key,
                w=w,
                u=u,
                g=g_cumsum,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                anchor_offsets=anchor_offsets,
                local_anchor_positions=local_anchor_positions,
                initial_state=initial_state,
            )
            if anchor_result is None:
                return _fallback()
            chunk_states, v_new, final_state, anchor_states = anchor_result
        text_o = ops.chunk_fwd_o(
            query,
            key,
            v_new,
            chunk_states.to(query.dtype),
            g_cumsum,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=FLA_CHUNK_SIZE,
            chunk_indices=chunk_indices,
            use_exp2=True,
            transpose_state_layout=False,
        )

    if anchor_states is None:
        anchor_states = _gather_anchor_states(
            chunk_states=chunk_states,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
            anchor_offsets=anchor_offsets,
            anchor_chunk_size=anchor_chunk_size,
        )
    if anchor_states is None:
        return _fallback()
    if squeezed:
        text_o = text_o.squeeze(0)
    return text_o, anchor_states, final_state


__all__ = [
    "FLA_CHUNK_SIZE",
    "chunk_gated_delta_rule_with_anchor_states_varlen_forward",
]
