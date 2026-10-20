# Copyright (c) ModelScope Contributors. All rights reserved.
"""MARCH anchor reads and boundary-gradient scatter Triton kernels."""

from __future__ import annotations

import triton
import triton.language as tl




@triton.jit(do_not_specialize=["TEXT_LEN"])
def _hsd_anchor_read_fwd_kernel(
    anchor_q,
    chunk_states,
    final_state,
    boundary_counts,
    text_cu_seqlens,
    anchor_cu_seqlens,
    chunk_offsets,
    anchor_o,
    anchor_states,
    scale,
    TEXT_LEN,
    ANCHOR_COUNT: tl.constexpr,
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
            anchor_start = tl.load(anchor_cu_seqlens + i_doc)
            anchor_end = tl.load(anchor_cu_seqlens + i_doc + 1)
            in_doc = (i_s >= anchor_start) & (i_s < anchor_end)
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
        q_ptr = anchor_q + (((i_b * ANCHOR_COUNT + i_s) * H + i_h) * K + offs_k)
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
            anchor_states
            + ((((i_b * ANCHOR_COUNT + i_s) * H + i_h) * K + offs_k[:, None]) * V + offs_v[None, :])
        )
        tl.store(out_state_ptrs, b_state, mask=k_mask[:, None] & v_mask[None, :])
        b_o += tl.sum(b_q[:, None] * b_state, axis=0)

    out_ptr = anchor_o + (((i_b * ANCHOR_COUNT + i_s) * H + i_h) * V + offs_v)
    tl.store(out_ptr, (b_o * scale).to(out_ptr.dtype.element_ty), mask=v_mask)


@triton.jit
def _hsd_anchor_read_bwd_q_kernel(
    danchor_o,
    anchor_states,
    danchor_q,
    scale,
    ANCHOR_COUNT: tl.constexpr,
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
        do_ptr = danchor_o + (((i_b * ANCHOR_COUNT + i_s) * H + i_h) * V + offs_v)
        state_ptrs = (
            anchor_states
            + ((((i_b * ANCHOR_COUNT + i_s) * H + i_h) * K + offs_k[:, None]) * V + offs_v[None, :])
        )
        b_do = tl.load(do_ptr, mask=v_mask, other=0.0).to(tl.float32) * scale
        b_state = tl.load(state_ptrs, mask=k_mask[:, None] & v_mask[None, :], other=0.0).to(tl.float32)
        b_dq += tl.sum(b_state * b_do[None, :], axis=1)

    dq_ptr = danchor_q + (((i_b * ANCHOR_COUNT + i_s) * H + i_h) * K + offs_k)
    tl.store(dq_ptr, b_dq.to(dq_ptr.dtype.element_ty), mask=k_mask)


@triton.jit
def _hsd_compact_anchor_read_fwd_kernel(
    anchor_q,
    anchor_states,
    anchor_o,
    scale,
    ANCHOR_COUNT: tl.constexpr,
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
        q_ptr = anchor_q + (
            ((i_b * ANCHOR_COUNT + i_s) * H + i_h) * K
            + offs_k
        )
        state_ptrs = anchor_states + (
            (
                (
                    (i_b * ANCHOR_COUNT + i_s) * H
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

    out_ptr = anchor_o + (
        ((i_b * ANCHOR_COUNT + i_s) * H + i_h) * V
        + offs_v
    )
    tl.store(
        out_ptr,
        (b_o * scale).to(out_ptr.dtype.element_ty),
        mask=v_mask,
    )


@triton.jit(do_not_specialize=["TEXT_LEN"])
def _hsd_anchor_state_scatter_bwd_kernel(
    anchor_q,
    danchor_o,
    danchor_states,
    boundary_counts,
    text_cu_seqlens,
    anchor_cu_seqlens,
    chunk_offsets,
    dh_direct,
    dh_extra,
    dht_direct,
    dht_extra,
    scale,
    TEXT_LEN,
    ANCHOR_COUNT: tl.constexpr,
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
    i_s = i_s_bh % ANCHOR_COUNT
    i_bh = i_s_bh // ANCHOR_COUNT
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
            anchor_start = tl.load(anchor_cu_seqlens + i_doc)
            anchor_end = tl.load(anchor_cu_seqlens + i_doc + 1)
            in_doc = (i_s >= anchor_start) & (i_s < anchor_end)
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
        danchor_states
        + ((((i_b * ANCHOR_COUNT + i_s) * H + i_h) * K + offs_k[:, None]) * V + offs_v[None, :])
    )
    q_ptr = anchor_q + (((i_b * ANCHOR_COUNT + i_s) * H + i_h) * K + offs_k)
    do_ptr = danchor_o + (((i_b * ANCHOR_COUNT + i_s) * H + i_h) * V + offs_v)
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

