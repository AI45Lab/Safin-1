"""Optional FLA forward helpers for HSD prefill segment fast paths."""

from __future__ import annotations

import contextlib
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
    def _hsd_chunk_gdn_fwd_h_anchor_kernel(
        q,
        k,
        v,
        w,
        g,
        o,
        anchor_q,
        anchor_o,
        anchor_states,
        h0,
        ht,
        scale,
        T,
        H: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BV: tl.constexpr,
        ANCHOR_STRIDE_CHUNKS: tl.constexpr,
        ANCHOR_COUNT: tl.constexpr,
        FUSE_ANCHOR_O: tl.constexpr,
        USE_INITIAL_STATE: tl.constexpr,
    ):
        i_v, i_nh = tl.program_id(0), tl.program_id(1)
        i_n, i_h = i_nh // HV, i_nh % HV
        NT = tl.cdiv(T, BT)
        bos = i_n * T

        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

        anchor_states += ((i_n * ANCHOR_COUNT) * HV + i_h).to(tl.int64) * K * V
        anchor_q += ((i_n * ANCHOR_COUNT) * HV + i_h).to(tl.int64) * K
        anchor_o += ((i_n * ANCHOR_COUNT) * HV + i_h).to(tl.int64) * V
        q += (bos * H + i_h // (HV // H)).to(tl.int64) * K
        k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
        v += (bos * HV + i_h).to(tl.int64) * V
        w += (bos * HV + i_h).to(tl.int64) * K
        o += (bos * HV + i_h).to(tl.int64) * V
        if USE_INITIAL_STATE:
            h0 += i_nh * K * V
        ht += i_nh * K * V

        if USE_INITIAL_STATE:
            p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
            if K > 64:
                p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
            if K > 128:
                p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
            if K > 192:
                p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

        o_k = tl.arange(0, 64)
        o_v = tl.arange(0, BV)
        v_offsets = i_v * BV + o_v

        for i_t in range(NT):
            anchor_idx = i_t // ANCHOR_STRIDE_CHUNKS - 1
            safe_anchor_idx = tl.maximum(anchor_idx, 0)
            store_anchor = (i_t > 0) & ((i_t % ANCHOR_STRIDE_CHUNKS) == 0) & (anchor_idx < ANCHOR_COUNT)
            anchor_base = anchor_states + safe_anchor_idx.to(tl.int64) * HV * K * V
            p_s1 = anchor_base + o_k[:, None] * V + v_offsets[None, :]
            tl.store(
                p_s1,
                b_h1,
                mask=store_anchor & (o_k[:, None] < K) & (v_offsets[None, :] < V),
            )
            if K > 64:
                p_s2 = anchor_base + (64 + o_k[:, None]) * V + v_offsets[None, :]
                tl.store(
                    p_s2,
                    b_h2,
                    mask=store_anchor & ((64 + o_k[:, None]) < K) & (v_offsets[None, :] < V),
                )
            if K > 128:
                p_s3 = anchor_base + (128 + o_k[:, None]) * V + v_offsets[None, :]
                tl.store(
                    p_s3,
                    b_h3,
                    mask=store_anchor & ((128 + o_k[:, None]) < K) & (v_offsets[None, :] < V),
                )
            if K > 192:
                p_s4 = anchor_base + (192 + o_k[:, None]) * V + v_offsets[None, :]
                tl.store(
                    p_s4,
                    b_h4,
                    mask=store_anchor & ((192 + o_k[:, None]) < K) & (v_offsets[None, :] < V),
                )
            if FUSE_ANCHOR_O:
                p_sq1 = anchor_q + safe_anchor_idx.to(tl.int64) * HV * K + o_k
                b_sq1 = tl.load(p_sq1, mask=store_anchor & (o_k < K), other=0.0).to(tl.float32)
                b_so = tl.sum(tl.dot(b_sq1[None, :], b_h1, input_precision="ieee"), axis=0)
                if K > 64:
                    p_sq2 = anchor_q + safe_anchor_idx.to(tl.int64) * HV * K + 64 + o_k
                    b_sq2 = tl.load(p_sq2, mask=store_anchor & ((64 + o_k) < K), other=0.0).to(tl.float32)
                    b_so += tl.sum(tl.dot(b_sq2[None, :], b_h2, input_precision="ieee"), axis=0)
                if K > 128:
                    p_sq3 = anchor_q + safe_anchor_idx.to(tl.int64) * HV * K + 128 + o_k
                    b_sq3 = tl.load(p_sq3, mask=store_anchor & ((128 + o_k) < K), other=0.0).to(tl.float32)
                    b_so += tl.sum(tl.dot(b_sq3[None, :], b_h3, input_precision="ieee"), axis=0)
                if K > 192:
                    p_sq4 = anchor_q + safe_anchor_idx.to(tl.int64) * HV * K + 192 + o_k
                    b_sq4 = tl.load(p_sq4, mask=store_anchor & ((192 + o_k) < K), other=0.0).to(tl.float32)
                    b_so += tl.sum(tl.dot(b_sq4[None, :], b_h4, input_precision="ieee"), axis=0)
                p_so = anchor_o + safe_anchor_idx.to(tl.int64) * HV * V + v_offsets
                tl.store(
                    p_so,
                    (b_so * scale).to(p_so.dtype.element_ty),
                    mask=store_anchor & (v_offsets < V),
                )

            p_w = tl.make_block_ptr(w, (T, K), (HV * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
            if K > 64:
                p_w = tl.make_block_ptr(w, (T, K), (HV * K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
            if K > 128:
                p_w = tl.make_block_ptr(w, (T, K), (HV * K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
            if K > 192:
                p_w = tl.make_block_ptr(w, (T, K), (HV * K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
            p_v = tl.make_block_ptr(v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v
            b_v_for_o = b_v.to(k.dtype.element_ty)

            b_o = tl.zeros([BT, BV], dtype=tl.float32)
            b_A = tl.zeros([BT, BT], dtype=tl.float32)
            p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            p_k_t = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_k_t = tl.load(p_k_t, boundary_check=(0, 1))
            b_o += tl.dot(b_q, b_h1.to(b_q.dtype))
            b_A += tl.dot(b_q, b_k_t)
            if K > 64:
                p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                p_k_t = tl.make_block_ptr(k, (K, T), (1, H * K), (64, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_k_t = tl.load(p_k_t, boundary_check=(0, 1))
                b_o += tl.dot(b_q, b_h2.to(b_q.dtype))
                b_A += tl.dot(b_q, b_k_t)
            if K > 128:
                p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                p_k_t = tl.make_block_ptr(k, (K, T), (1, H * K), (128, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_k_t = tl.load(p_k_t, boundary_check=(0, 1))
                b_o += tl.dot(b_q, b_h3.to(b_q.dtype))
                b_A += tl.dot(b_q, b_k_t)
            if K > 192:
                p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                p_k_t = tl.make_block_ptr(k, (K, T), (1, H * K), (192, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_k_t = tl.load(p_k_t, boundary_check=(0, 1))
                b_o += tl.dot(b_q, b_h4.to(b_q.dtype))
                b_A += tl.dot(b_q, b_k_t)

            last_idx = tl.minimum((i_t + 1) * BT, T) - 1
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(tl.float32)
            p_g = tl.make_block_ptr(g + (bos * HV + i_h).to(tl.int64), (T,), (HV,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)

            b_o = b_o * tl.exp2(b_g)[:, None]
            b_A = b_A * tl.exp2(b_g[:, None] - b_g[None, :])
            o_t = i_t * BT + tl.arange(0, BT)
            m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
            b_A = tl.where(m_A, b_A, 0)
            p_o = tl.make_block_ptr(o, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_o = b_o * scale + tl.dot(b_A.to(b_v_for_o.dtype), b_v_for_o) * scale
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

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

            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h1 += tl.dot(b_k, b_v)
            if K > 64:
                p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (64, i_t * BT), (64, BT), (0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h2 += tl.dot(b_k, b_v)
            if K > 128:
                p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (128, i_t * BT), (64, BT), (0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h3 += tl.dot(b_k, b_v)
            if K > 192:
                p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (192, i_t * BT), (64, BT), (0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h4 += tl.dot(b_k, b_v)

        final_anchor_idx = T // (ANCHOR_STRIDE_CHUNKS * BT) - 1
        store_final_anchor = (
            (ANCHOR_COUNT > 0)
            & ((T % (ANCHOR_STRIDE_CHUNKS * BT)) == 0)
            & (final_anchor_idx >= 0)
            & (final_anchor_idx < ANCHOR_COUNT)
        )
        final_anchor_base = anchor_states + final_anchor_idx.to(tl.int64) * HV * K * V
        p_fs1 = final_anchor_base + o_k[:, None] * V + v_offsets[None, :]
        tl.store(
            p_fs1,
            b_h1,
            mask=store_final_anchor & (o_k[:, None] < K) & (v_offsets[None, :] < V),
        )
        if K > 64:
            p_fs2 = final_anchor_base + (64 + o_k[:, None]) * V + v_offsets[None, :]
            tl.store(
                p_fs2,
                b_h2,
                mask=store_final_anchor & ((64 + o_k[:, None]) < K) & (v_offsets[None, :] < V),
            )
        if K > 128:
            p_fs3 = final_anchor_base + (128 + o_k[:, None]) * V + v_offsets[None, :]
            tl.store(
                p_fs3,
                b_h3,
                mask=store_final_anchor & ((128 + o_k[:, None]) < K) & (v_offsets[None, :] < V),
            )
        if K > 192:
            p_fs4 = final_anchor_base + (192 + o_k[:, None]) * V + v_offsets[None, :]
            tl.store(
                p_fs4,
                b_h4,
                mask=store_final_anchor & ((192 + o_k[:, None]) < K) & (v_offsets[None, :] < V),
            )
        if FUSE_ANCHOR_O:
            p_fsq1 = anchor_q + final_anchor_idx.to(tl.int64) * HV * K + o_k
            b_fsq1 = tl.load(p_fsq1, mask=store_final_anchor & (o_k < K), other=0.0).to(tl.float32)
            b_fso = tl.sum(tl.dot(b_fsq1[None, :], b_h1, input_precision="ieee"), axis=0)
            if K > 64:
                p_fsq2 = anchor_q + final_anchor_idx.to(tl.int64) * HV * K + 64 + o_k
                b_fsq2 = tl.load(p_fsq2, mask=store_final_anchor & ((64 + o_k) < K), other=0.0).to(tl.float32)
                b_fso += tl.sum(tl.dot(b_fsq2[None, :], b_h2, input_precision="ieee"), axis=0)
            if K > 128:
                p_fsq3 = anchor_q + final_anchor_idx.to(tl.int64) * HV * K + 128 + o_k
                b_fsq3 = tl.load(p_fsq3, mask=store_final_anchor & ((128 + o_k) < K), other=0.0).to(tl.float32)
                b_fso += tl.sum(tl.dot(b_fsq3[None, :], b_h3, input_precision="ieee"), axis=0)
            if K > 192:
                p_fsq4 = anchor_q + final_anchor_idx.to(tl.int64) * HV * K + 192 + o_k
                b_fsq4 = tl.load(p_fsq4, mask=store_final_anchor & ((192 + o_k) < K), other=0.0).to(tl.float32)
                b_fso += tl.sum(tl.dot(b_fsq4[None, :], b_h4, input_precision="ieee"), axis=0)
            p_fso = anchor_o + final_anchor_idx.to(tl.int64) * HV * V + v_offsets
            tl.store(
                p_fso,
                (b_fso * scale).to(p_fso.dtype.element_ty),
                mask=store_final_anchor & (v_offsets < V),
            )

        p_ht1 = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht1, b_h1.to(p_ht1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht2 = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht2, b_h2.to(p_ht2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht3 = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht3, b_h3.to(p_ht3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht4 = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht4, b_h4.to(p_ht4.dtype.element_ty), boundary_check=(0, 1))


def _disabled() -> bool:
    return os.environ.get("HSD_DISABLE_HSD_SEGMENT_FASTPATH", "").lower() in {"1", "true", "yes", "on"}


def _disable_fused_anchor_o() -> bool:
    return os.environ.get("HSD_DISABLE_FUSED_ANCHOR_O", "").lower() in {"1", "true", "yes", "on"}


def _enable_fused_anchor_o() -> bool:
    if _disable_fused_anchor_o():
        return False
    return os.environ.get("HSD_ENABLE_FUSED_ANCHOR_O", "").lower() in {"1", "true", "yes", "on"}


def _load_fla_ops() -> SimpleNamespace | None:
    global _FLA_OPS, _FLA_IMPORT_ERROR
    if _FLA_OPS is not None:
        return _FLA_OPS
    if _FLA_IMPORT_ERROR is not None:
        return None
    try:
        import triton

        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_kernel_h_blockdim64
        from fla.ops.common.chunk_o import chunk_fwd_o
        from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
        from fla.ops.utils import chunk_local_cumsum
        from fla.ops.utils.constant import RCP_LN2
        from fla.utils import custom_device_ctx
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
        RCP_LN2=RCP_LN2,
    )
    return _FLA_OPS


def _chunk_gated_delta_rule_fwd_h_fp32(
    ops: SimpleNamespace,
    *,
    key: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, text_len, heads, key_dim = key.shape
    value_dim = int(u.shape[-1])
    value_heads = int(u.shape[2])
    chunk_count = ops.triton.cdiv(text_len, FLA_CHUNK_SIZE)

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
        batch_size,
        value_heads,
        key_dim,
        value_dim,
        device=key.device,
        dtype=torch.float32,
    )
    v_new = torch.empty_like(u)

    def grid(meta):
        # FLA indexes the value tile on axis 0 and batch/head on axis 1.
        # Flattening them leaves every head except the first unwritten.
        return (ops.triton.cdiv(value_dim, meta["BV"]), batch_size * value_heads)

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
        cu_seqlens=None,
        chunk_offsets=None,
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


def _chunk_gated_delta_rule_fwd_h_anchor(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
    anchor_count: int,
    anchor_stride_chunks: int,
    scale: float,
    anchor_q: torch.Tensor | None = None,
    fuse_anchor_o: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor] | None:
    if triton is None:
        return None
    batch_size, text_len, heads, key_dim = key.shape
    value_dim = int(u.shape[-1])
    value_heads = int(u.shape[2])

    text_o = u.new_empty(batch_size, text_len, value_heads, value_dim)
    anchor_states = torch.empty(
        batch_size,
        int(anchor_count),
        value_heads,
        key_dim,
        value_dim,
        device=key.device,
        dtype=torch.float32,
    )
    final_state = torch.zeros(
        batch_size,
        value_heads,
        key_dim,
        value_dim,
        device=key.device,
        dtype=torch.float32,
    )
    anchor_o = u.new_empty(batch_size, int(anchor_count), value_heads, value_dim)
    if anchor_q is None:
        anchor_q = key.new_empty(batch_size, int(anchor_count), heads, key_dim)
    block_v = 32

    def grid(meta):
        return (triton.cdiv(value_dim, meta["BV"]), batch_size * value_heads)

    _hsd_chunk_gdn_fwd_h_anchor_kernel[grid](
        query,
        key,
        u,
        w,
        g,
        text_o,
        anchor_q,
        anchor_o,
        anchor_states,
        initial_state,
        final_state,
        float(scale),
        text_len,
        H=heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BT=FLA_CHUNK_SIZE,
        BV=block_v,
        ANCHOR_STRIDE_CHUNKS=int(anchor_stride_chunks),
        ANCHOR_COUNT=int(anchor_count),
        FUSE_ANCHOR_O=bool(fuse_anchor_o),
    )
    return text_o, anchor_o if fuse_anchor_o else None, anchor_states, final_state


def chunk_gated_delta_rule_with_anchor_states_forward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    anchor_count: int,
    anchor_chunk_size: int,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    anchor_q: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor] | None:
    """Run FLA chunk GDN forward and store only anchor-boundary fp32 states."""

    if _disabled():
        return None
    if triton is None or not query.is_cuda:
        return None
    if torch.is_grad_enabled() and any(
        tensor.requires_grad
        for tensor in (query, key, value, g, beta, initial_state, anchor_q)
        if torch.is_tensor(tensor)
    ):
        return None
    if int(anchor_count) <= 0:
        return None
    if int(anchor_chunk_size) <= 0 or int(anchor_chunk_size) % FLA_CHUNK_SIZE != 0:
        return None
    anchor_stride_chunks = int(anchor_chunk_size) // FLA_CHUNK_SIZE
    if query.dtype == torch.float32 or key.dtype == torch.float32 or value.dtype == torch.float32:
        return None
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or g.ndim != 3 or beta.ndim != 3:
        return None
    if query.shape != key.shape:
        return None
    batch_size, text_len, heads, key_dim = query.shape
    if value.shape[:3] != (batch_size, text_len, heads):
        return None
    if g.shape != (batch_size, text_len, heads) or beta.shape != (batch_size, text_len, heads):
        return None
    if key_dim > 256 or text_len <= 0:
        return None
    if initial_state is not None and initial_state.shape[:3] != (batch_size, heads, key_dim):
        return None
    fuse_anchor_o = anchor_q is not None and _enable_fused_anchor_o()
    if fuse_anchor_o and anchor_q.shape != (batch_size, int(anchor_count), heads, key_dim):
        return None

    ops = _load_fla_ops()
    if ops is None:
        return None
    if scale is None:
        scale = key_dim**-0.5

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    if fuse_anchor_o:
        anchor_q = anchor_q.contiguous()

    device_ctx = ops.custom_device_ctx(query.device.index) if query.device.index is not None else contextlib.nullcontext()
    with device_ctx:
        g_cumsum = ops.chunk_local_cumsum(
            g,
            chunk_size=FLA_CHUNK_SIZE,
            scale=ops.RCP_LN2,
            cu_seqlens=None,
            chunk_indices=None,
        )
        w, u, _ = ops.chunk_gated_delta_rule_fwd_intra(
            k=key,
            v=value,
            g=g_cumsum,
            beta=beta,
            cu_seqlens=None,
            chunk_indices=None,
            use_exp2=True,
        )
        selective = _chunk_gated_delta_rule_fwd_h_anchor(
            query=query,
            key=key,
            w=w,
            u=u,
            g=g_cumsum,
            initial_state=initial_state,
            anchor_count=int(anchor_count),
            anchor_stride_chunks=anchor_stride_chunks,
            scale=float(scale),
            anchor_q=anchor_q,
            fuse_anchor_o=fuse_anchor_o,
        )
        if selective is None:
            return None
        text_o, anchor_o, anchor_states, final_state = selective
    return text_o.to(query.dtype), None if anchor_o is None else anchor_o.to(query.dtype), anchor_states, final_state


def chunk_gated_delta_rule_with_chunk_states_forward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Run FLA chunk GDN forward and expose internal chunk states.

    Returns ``(text_o, chunk_states, final_state)`` or ``None`` when unsupported.
    This is an inference-only helper: training/autograd paths must keep using the
    reference segmented implementation.
    """

    if _disabled():
        return None
    if not query.is_cuda:
        return None
    if torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, key, value, g, beta, initial_state) if torch.is_tensor(tensor)
    ):
        return None
    if query.dtype == torch.float32 or key.dtype == torch.float32 or value.dtype == torch.float32:
        return None
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or g.ndim != 3 or beta.ndim != 3:
        return None
    if query.shape != key.shape:
        return None
    batch_size, text_len, heads, key_dim = query.shape
    if value.shape[:3] != (batch_size, text_len, heads):
        return None
    if g.shape != (batch_size, text_len, heads) or beta.shape != (batch_size, text_len, heads):
        return None
    if key_dim > 256:
        return None
    if text_len <= 0:
        return None
    if initial_state is not None and initial_state.shape[:3] != (batch_size, heads, key_dim):
        return None

    ops = _load_fla_ops()
    if ops is None:
        return None
    if scale is None:
        scale = key_dim**-0.5

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    if initial_state is not None:
        initial_state = initial_state.contiguous()

    device_ctx = ops.custom_device_ctx(query.device.index) if query.device.index is not None else contextlib.nullcontext()
    with device_ctx:
        g_cumsum = ops.chunk_local_cumsum(
            g,
            chunk_size=FLA_CHUNK_SIZE,
            scale=ops.RCP_LN2,
            cu_seqlens=None,
            chunk_indices=None,
        )
        w, u, _ = ops.chunk_gated_delta_rule_fwd_intra(
            k=key,
            v=value,
            g=g_cumsum,
            beta=beta,
            cu_seqlens=None,
            chunk_indices=None,
            use_exp2=True,
        )
        chunk_states, v_new, final_state = _chunk_gated_delta_rule_fwd_h_fp32(
            ops,
            key=key,
            w=w,
            u=u,
            g=g_cumsum,
            initial_state=initial_state,
        )
        text_o = ops.chunk_fwd_o(
            q=query,
            k=key,
            v=v_new,
            h=chunk_states.to(key.dtype),
            g=g_cumsum,
            scale=float(scale),
            cu_seqlens=None,
            chunk_indices=None,
            use_exp2=True,
            transpose_state_layout=False,
        )
    return text_o.to(query.dtype), chunk_states, final_state


__all__ = [
    "FLA_CHUNK_SIZE",
    "chunk_gated_delta_rule_with_chunk_states_forward",
    "chunk_gated_delta_rule_with_anchor_states_forward",
]
