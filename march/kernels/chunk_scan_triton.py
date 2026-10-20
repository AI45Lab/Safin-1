# Copyright (c) ModelScope Contributors. All rights reserved.
"""MARCH external-gradient chunk recurrence Triton kernels."""

from __future__ import annotations

import triton
import triton.language as tl
from fla.ops.utils.op import exp2


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
    anchor_cu_seqlens,
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
        anchor_start = tl.load(anchor_cu_seqlens + i_b).to(tl.int32)
        compact_dh += (anchor_start * HV + i_h).to(tl.int64) * K * V
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

