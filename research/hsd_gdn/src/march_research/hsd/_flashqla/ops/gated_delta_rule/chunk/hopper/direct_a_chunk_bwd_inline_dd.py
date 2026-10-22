# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

r"""Single-CTA direct-A chunk backward with an inline ``dD`` path.

This is the optimized direct-A backend for the final HSD producer Kernel B.
The model-level FlashQLA producer remains explicit opt-in; once selected this
is its default backward implementation.

The fixed specialization is the current producer shape:

* packed ``B=1``;
* ``H=6, DK=192, DV=384``;
* internal chunk size ``S=64``;
* grid ``(num_chunks * H,)``;
* one CTA loops over all six V64 tiles.

Unlike :mod:`direct_a_chunk_bwd`, this kernel neither accepts a materialized
``dd`` tensor nor emits six V-tile partial tensors.  For every V tile it forms

.. math::

    dD = r\,(K\,dH_{\mathrm{end}}) + C^\top dO,

where ``r = exp(z_last - z)`` and
``C = scale * G * (Q K.T)``.  The CTA accumulates the V-reduced ``dB`` and
``dC`` matrices, then differentiates the route-independent KKT/direct-A graph
once.

There is exactly one owning CTA for every ``(token, head)`` element.  The
kernel therefore keeps the six V-tile contributions in persistent FP32
accumulators and writes the final ``dq/dk/dv/dbeta`` values directly in their
input BF16 dtypes.  Only ``dg`` remains FP32.  No split-gradient partial output
or atomic reduction is required.
"""

import os

import torch
import tilelang
import tilelang.language as T


_H = 6
_DK = 192
_DV = 384
_S = 64
_BV = 64
_NUM_VALUE_TILES = _DV // _BV
_MATRIX_VJP_DTYPES = {
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float32": "float32",
    "fp32": "float32",
    "tf32": "float32",
}


def _resolve_matrix_vjp_dtype() -> str:
    """Return the communication dtype for the direct-A matrix VJP.

    The legacy implementation rounded the V-reduced ``dB``/``dC`` matrices
    to BF16 and then rounded both matrix products in the inverse/KKT VJP to
    BF16 again.  Those are backward-only intermediates, so the repeated
    rounding is not required to match the BF16 forward graph and can
    accumulate into a materially different optimizer trajectory.

    Keep FP32 as the safe default.  ``bfloat16`` remains available as an
    explicit performance diagnostic, rather than silently being selected by
    the activation dtype.
    """

    requested = os.environ.get(
        "HSD_FLASHQLA_DIRECT_A_MATRIX_VJP_DTYPE",
        "float32",
    ).strip().lower()
    resolved = _MATRIX_VJP_DTYPES.get(requested)
    if resolved is None:
        raise ValueError(
            "HSD_FLASHQLA_DIRECT_A_MATRIX_VJP_DTYPE must be one of "
            "bfloat16, bf16, float32, fp32, or tf32; got "
            f"{requested!r}."
        )
    return resolved


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_direct_a_chunk_bwd_inline_dd(
    scale: float,
    qk_dtype: str,
    value_dtype: str,
    a_dtype: str,
    g_dtype: str,
    beta_dtype: str,
    state_dtype: str,
    seqlen_dtype: str,
    accum_dtype: str = "float32",
    matrix_vjp_dtype: str = "float32",
):
    """Build the fixed H6/K192/V384/S64 single-CTA prototype."""

    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    num_docs = T.dynamic("num_docs")

    qk_shape = (1, num_tokens, _H, _DK)
    value_shape = (1, num_tokens, _H, _DV)
    matrix_shape = (1, num_tokens, _H, _S)
    scalar_shape = (1, num_tokens, _H)
    state_shape = (1, num_chunks, _H, _DK, _DV)

    @T.prim_func
    def kernel(
        q: T.Tensor(qk_shape, dtype=qk_dtype),
        k: T.Tensor(qk_shape, dtype=qk_dtype),
        v: T.Tensor(value_shape, dtype=value_dtype),
        a: T.Tensor(matrix_shape, dtype=a_dtype),
        g: T.Tensor(scalar_shape, dtype=g_dtype),
        beta: T.Tensor(scalar_shape, dtype=beta_dtype),
        do: T.Tensor(value_shape, dtype=value_dtype),
        chunk_start_h: T.Tensor(state_shape, dtype=state_dtype),
        chunk_end_dh: T.Tensor(state_shape, dtype=state_dtype),
        cu_seqlens: T.Tensor([num_docs + 1], dtype=seqlen_dtype),
        chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
        dq: T.Tensor(qk_shape, dtype=qk_dtype),
        dk: T.Tensor(qk_shape, dtype=qk_dtype),
        dv: T.Tensor(value_shape, dtype=value_dtype),
        dbeta: T.Tensor(scalar_shape, dtype=beta_dtype),
        dg: T.Tensor(scalar_shape, dtype=accum_dtype),
    ):
        with T.Kernel(num_chunks * _H, threads=256) as (bch,):
            bc = bch // _H
            bh = bch % _H

            doc = chunk_indices[bc, 0]
            local_chunk = chunk_indices[bc, 1]
            seq_start = cu_seqlens[doc]
            seq_end = cu_seqlens[doc + 1]
            left = seq_start + local_chunk * _S
            valid = T.min(_S, seq_end - left)
            # ``chunk_indices`` only contains real chunks, so ``valid`` is
            # positive at runtime.  Clamp the derived index nevertheless so
            # the lowering-time bounds analyzer can prove every shared/global
            # access is non-negative.
            last = T.max(0, valid - 1)

            # Chunk/head-persistent operands.
            q_shared = T.alloc_shared((_S, _DK), dtype=qk_dtype)
            k_shared = T.alloc_shared((_S, _DK), dtype=qk_dtype)
            # A and the backward-only matrix VJPs intentionally communicate
            # in FP32 by default.  The forward still stores A in BF16; this
            # merely avoids repeatedly quantizing dB/dC and the inverse VJP.
            a_shared = T.alloc_shared((_S, _S), dtype=matrix_vjp_dtype)
            z_shared = T.alloc_shared(
                (_S,),
                dtype=accum_dtype,
                scope="shared",
            )
            beta_shared = T.alloc_shared(
                (_S,),
                dtype=accum_dtype,
                scope="shared",
            )
            exp_z_shared = T.alloc_shared(
                (_S,),
                dtype=accum_dtype,
                scope="shared",
            )
            reverse_decay_shared = T.alloc_shared(
                (_S,),
                dtype=accum_dtype,
                scope="shared",
            )
            dz_shared = T.alloc_shared(
                (_S,),
                dtype=accum_dtype,
                scope="shared",
            )

            # G, B and C are shared by all six V tiles and are built once.
            causal_shared = T.alloc_shared((_S, _S), dtype=qk_dtype)
            write_shared = T.alloc_shared((_S, _S), dtype=qk_dtype)
            read_shared = T.alloc_shared((_S, _S), dtype=qk_dtype)
            # dB and dC are never needed simultaneously after the V-tile
            # reduction.  Reuse one tile so FP32 communication costs no more
            # shared memory than the previous pair of BF16 tiles.
            matrix_vjp_shared = T.alloc_shared(
                (_S, _S),
                dtype=matrix_vjp_dtype,
            )
            matrix_scratch_shared = T.alloc_shared(
                (_S, _S),
                dtype=qk_dtype,
            )

            # Reused V64 working set.
            h_shared = T.alloc_shared((_DK, _BV), dtype=state_dtype)
            dh_shared = T.alloc_shared((_DK, _BV), dtype=state_dtype)
            v_shared = T.alloc_shared((_S, _BV), dtype=value_dtype)
            do_shared = T.alloc_shared((_S, _BV), dtype=value_dtype)
            u_shared = T.alloc_shared((_S, _BV), dtype=value_dtype)
            w_shared = T.alloc_shared((_S, _BV), dtype=value_dtype)
            d_shared = T.alloc_shared((_S, _BV), dtype=value_dtype)
            dd_shared = T.alloc_shared((_S, _BV), dtype=value_dtype)

            # V-reduced direct-A matrix VJPs.  Keeping these in FP32
            # fragments avoids any global V-tile partial tensor.
            db_matrix = T.alloc_fragment((_S, _S), dtype=accum_dtype)
            dc_matrix = T.alloc_fragment((_S, _S), dtype=accum_dtype)
            mat0 = T.alloc_fragment((_S, _S), dtype=accum_dtype)
            mat1 = T.alloc_fragment((_S, _S), dtype=accum_dtype)
            mat2 = T.alloc_fragment((_S, _S), dtype=accum_dtype)
            # Reduction-only scratch.  Keeping it distinct from ``mat1``
            # lets the inverse-VJP WGMMA and the later KKT Gram WGMMA retain
            # their different legal accumulator layouts.
            mat_reduce = T.alloc_fragment((_S, _S), dtype=accum_dtype)

            # Reused V-space and K-space MMA accumulators.
            val0 = T.alloc_fragment((_S, _BV), dtype=accum_dtype)
            val1 = T.alloc_fragment((_S, _BV), dtype=accum_dtype)
            val2 = T.alloc_fragment((_S, _BV), dtype=accum_dtype)
            qk_grad = T.alloc_fragment((_S, _DK), dtype=accum_dtype)
            state_grad = T.alloc_fragment((_S, _DK), dtype=accum_dtype)
            product_value = T.alloc_fragment((_S, _BV), dtype=accum_dtype)
            product_state = T.alloc_fragment((_DK, _BV), dtype=accum_dtype)
            row0 = T.alloc_fragment((_S,), dtype=accum_dtype)
            row1 = T.alloc_fragment((_S,), dtype=accum_dtype)
            key_row = T.alloc_fragment((_DK,), dtype=accum_dtype)
            dbeta_acc = T.alloc_fragment((_S,), dtype=accum_dtype)
            dz_acc = T.alloc_fragment((_S,), dtype=accum_dtype)
            dg_acc = T.alloc_fragment((_S,), dtype=accum_dtype)
            scalar0 = T.alloc_fragment((1,), dtype=accum_dtype)
            scalar1 = T.alloc_fragment((1,), dtype=accum_dtype)

            T.annotate_layout(
                {
                    q_shared: tilelang.layout.make_swizzled_layout(q_shared),
                    k_shared: tilelang.layout.make_swizzled_layout(k_shared),
                    a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                    causal_shared: tilelang.layout.make_swizzled_layout(
                        causal_shared
                    ),
                    write_shared: tilelang.layout.make_swizzled_layout(
                        write_shared
                    ),
                    read_shared: tilelang.layout.make_swizzled_layout(
                        read_shared
                    ),
                    matrix_vjp_shared: tilelang.layout.make_swizzled_layout(
                        matrix_vjp_shared
                    ),
                    matrix_scratch_shared: (
                        tilelang.layout.make_swizzled_layout(
                            matrix_scratch_shared
                        )
                    ),
                    h_shared: tilelang.layout.make_swizzled_layout(h_shared),
                    dh_shared: tilelang.layout.make_swizzled_layout(
                        dh_shared
                    ),
                    v_shared: tilelang.layout.make_swizzled_layout(v_shared),
                    do_shared: tilelang.layout.make_swizzled_layout(
                        do_shared
                    ),
                    u_shared: tilelang.layout.make_swizzled_layout(u_shared),
                    w_shared: tilelang.layout.make_swizzled_layout(w_shared),
                    d_shared: tilelang.layout.make_swizzled_layout(d_shared),
                    dd_shared: tilelang.layout.make_swizzled_layout(
                        dd_shared
                    ),
                }
            )
            T.use_swizzle(10)
            T.set_max_nreg(240, 1)

            # Every chunk_indices row denotes a real non-empty packed chunk.
            # Tail rows and columns are explicitly zeroed.  In particular, A
            # is masked in both dimensions rather than only by its row.
            for i_s, i_k in T.Parallel(_S, _DK):
                if i_s < valid:
                    q_shared[i_s, i_k] = q[0, left + i_s, bh, i_k]
                    k_shared[i_s, i_k] = k[0, left + i_s, bh, i_k]
                else:
                    q_shared[i_s, i_k] = 0
                    k_shared[i_s, i_k] = 0
            for i_s, i_t in T.Parallel(_S, _S):
                if i_s < valid and i_t < valid:
                    a_shared[i_s, i_t] = a[0, left + i_s, bh, i_t]
                else:
                    a_shared[i_s, i_t] = 0
            for i_s in T.Parallel(_S):
                if i_s < valid:
                    z_shared[i_s] = g[0, left + i_s, bh]
                    beta_shared[i_s] = beta[0, left + i_s, bh]
                else:
                    z_shared[i_s] = g[0, left + last, bh]
                    beta_shared[i_s] = 0
            T.sync_threads()

            for i_s in T.Parallel(_S):
                exp_z_shared[i_s] = T.exp2(
                    z_shared[i_s] * 1.4426950408889634
                )
                reverse_decay_shared[i_s] = T.exp2(
                    (
                        z_shared[last]
                        - z_shared[i_s]
                    )
                    * 1.4426950408889634
                )

            # P=QK.T, G=causal exp(z_i-z_j),
            # B=G*A*beta_col and C=scale*G*P.
            T.gemm(
                q_shared,
                k_shared,
                mat0,
                transpose_B=True,
                clear_accum=True,
            )
            for i_s, i_t in T.Parallel(_S, _S):
                if i_s < valid and i_t < valid and i_s >= i_t:
                    causal_shared[i_s, i_t] = T.exp2(
                        (
                            z_shared[i_s]
                            - z_shared[i_t]
                        )
                        * 1.4426950408889634
                    )
                else:
                    causal_shared[i_s, i_t] = 0
            # Keep the causal gate in shared memory.  ``mat1`` is reserved for
            # the inverse-VJP WGMMA accumulator whose legal fragment layout is
            # different from this elementwise causal layout.
            T.sync_threads()
            for i_s, i_t in T.Parallel(_S, _S):
                mat2[i_s, i_t] = (
                    T.cast(causal_shared[i_s, i_t], accum_dtype)
                    * T.cast(a_shared[i_s, i_t], accum_dtype)
                    * beta_shared[i_t]
                )
            T.copy(mat2, write_shared)
            for i_s, i_t in T.Parallel(_S, _S):
                mat0[i_s, i_t] = (
                    float(scale)
                    * T.cast(causal_shared[i_s, i_t], accum_dtype)
                    * mat0[i_s, i_t]
                )
            T.copy(mat0, read_shared)
            T.clear(db_matrix)
            T.clear(dc_matrix)
            T.clear(dbeta_acc)
            T.clear(dz_acc)
            T.clear(qk_grad)
            # One CTA owns all six V tiles.  Keep dK in one persistent FP32
            # fragment across the recurrent, output-read and KKT paths, then
            # publish it once.  Besides removing 14,400 atomics per CTA, this
            # fixes the accumulator layout to the first recurrent GEMM, which
            # is the same scheme used by the proven per-V direct-A kernel.
            T.clear(state_grad)
            T.sync_threads()

            # The only V loop in the CTA.  It owns all V=384, so all
            # route-independent VJPs can be reduced before the KKT VJP.
            for bv in T.serial(_NUM_VALUE_TILES):
                value_start = bv * _BV

                for i_s, i_v in T.Parallel(_S, _BV):
                    if i_s < valid:
                        v_shared[i_s, i_v] = v[
                            0,
                            left + i_s,
                            bh,
                            value_start + i_v,
                        ]
                        do_shared[i_s, i_v] = do[
                            0,
                            left + i_s,
                            bh,
                            value_start + i_v,
                        ]
                    else:
                        v_shared[i_s, i_v] = 0
                        do_shared[i_s, i_v] = 0
                for i_k, i_v in T.Parallel(_DK, _BV):
                    h_shared[i_k, i_v] = chunk_start_h[
                        0,
                        bc,
                        bh,
                        i_k,
                        value_start + i_v,
                    ]
                    dh_shared[i_k, i_v] = chunk_end_dh[
                        0,
                        bc,
                        bh,
                        i_k,
                        value_start + i_v,
                    ]
                T.sync_threads()

                # U=K H0, W=V-E U, D=B W.
                T.gemm(
                    k_shared,
                    h_shared,
                    val0,
                    clear_accum=True,
                )
                T.copy(val0, u_shared)
                for i_s, i_v in T.Parallel(_S, _BV):
                    val0[i_s, i_v] = (
                        v_shared[i_s, i_v]
                        - exp_z_shared[i_s] * val0[i_s, i_v]
                    )
                T.copy(val0, w_shared)
                T.sync_threads()
                T.gemm(
                    write_shared,
                    w_shared,
                    val1,
                    clear_accum=True,
                )
                T.copy(val1, d_shared)

                # N=rD is the direct state-update contribution to dK.
                for i_s, i_v in T.Parallel(_S, _BV):
                    val2[i_s, i_v] = (
                        reverse_decay_shared[i_s]
                        * val1[i_s, i_v]
                    )
                T.copy(val2, v_shared)
                T.sync_threads()
                T.gemm(
                    v_shared,
                    dh_shared,
                    state_grad,
                    transpose_B=True,
                    clear_accum=False,
                )

                # dN=K dH_end.  Form dD inline:
                #
                #   dD = r*dN + C.T*dO.
                #
                # No [T,H,V] dd tensor exists at the API or in HBM.
                T.gemm(
                    k_shared,
                    dh_shared,
                    val0,
                    clear_accum=True,
                )
                T.gemm(
                    read_shared,
                    do_shared,
                    val2,
                    transpose_A=True,
                    clear_accum=True,
                )
                for i_s, i_v in T.Parallel(_S, _BV):
                    val2[i_s, i_v] += (
                        reverse_decay_shared[i_s]
                        * val0[i_s, i_v]
                    )
                T.copy(val2, dd_shared)
                T.sync_threads()

                # Ratio-gate VJP: r * <dN,D>, including the z_last term.
                for i_s, i_v in T.Parallel(_S, _BV):
                    product_value[i_s, i_v] = (
                        val0[i_s, i_v]
                        * d_shared[i_s, i_v]
                    )
                T.reduce_sum(product_value, row0, dim=1, clear=True)
                for i_s in T.Parallel(_S):
                    row0[i_s] *= reverse_decay_shared[i_s]
                    dz_acc[i_s] -= row0[i_s]
                T.reduce_sum(row0, scalar0, dim=0, clear=True)
                for i_s in T.Parallel(_S):
                    if i_s == last:
                        dz_acc[i_s] += scalar0[0]

                # decay * <dH_end,H0> contributes only to z_last.
                for i_k, i_v in T.Parallel(_DK, _BV):
                    product_state[i_k, i_v] = (
                        T.cast(dh_shared[i_k, i_v], accum_dtype)
                        * T.cast(h_shared[i_k, i_v], accum_dtype)
                    )
                T.reduce_sum(
                    product_state,
                    key_row,
                    dim=1,
                    clear=True,
                )
                T.reduce_sum(key_row, scalar1, dim=0, clear=True)
                for i_s in T.Parallel(_S):
                    if i_s == last:
                        dz_acc[i_s] += (
                            exp_z_shared[last] * scalar1[0]
                        )

                # Accumulate dB=dD W.T and dC=dO D.T over all V tiles.
                T.gemm(
                    dd_shared,
                    w_shared,
                    db_matrix,
                    transpose_B=True,
                    clear_accum=False,
                )
                T.gemm(
                    do_shared,
                    d_shared,
                    dc_matrix,
                    transpose_B=True,
                    clear_accum=False,
                )

                # dW=B.T dD; dV=dW; dU=-E dW.
                T.gemm(
                    write_shared,
                    dd_shared,
                    val0,
                    transpose_A=True,
                    clear_accum=True,
                )
                for i_s, i_v in T.Parallel(_S, _BV):
                    if i_s < valid:
                        dv[
                            0,
                            left + i_s,
                            bh,
                            value_start + i_v,
                        ] = val0[i_s, i_v]
                    val1[i_s, i_v] = (
                        -exp_z_shared[i_s] * val0[i_s, i_v]
                    )

                # dz from W=V-EU.
                for i_s, i_v in T.Parallel(_S, _BV):
                    product_value[i_s, i_v] = (
                        val0[i_s, i_v] * u_shared[i_s, i_v]
                    )
                T.reduce_sum(product_value, row1, dim=1, clear=True)
                for i_s in T.Parallel(_S):
                    dz_acc[i_s] -= exp_z_shared[i_s] * row1[i_s]

                # dK += dU H0.T.  Keep both contributions in the persistent
                # CTA-owned accumulator; publishing happens after the KKT VJP.
                T.copy(val1, v_shared)
                T.sync_threads()
                T.gemm(
                    v_shared,
                    h_shared,
                    state_grad,
                    transpose_B=True,
                    clear_accum=False,
                )

                # Direct output-state path:
                # dQ += scale * (E*dO) H0.T.
                for i_s, i_v in T.Parallel(_S, _BV):
                    val0[i_s, i_v] = (
                        float(scale)
                        * exp_z_shared[i_s]
                        * do_shared[i_s, i_v]
                    )
                T.copy(val0, v_shared)
                T.sync_threads()
                T.gemm(
                    v_shared,
                    h_shared,
                    qk_grad,
                    transpose_B=True,
                    clear_accum=False,
                )

                # Direct output exp(z) path:
                # scale * E * <dO,QH0>.
                T.gemm(
                    q_shared,
                    h_shared,
                    val0,
                    clear_accum=True,
                )
                for i_s, i_v in T.Parallel(_S, _BV):
                    product_value[i_s, i_v] = (
                        do_shared[i_s, i_v] * val0[i_s, i_v]
                    )
                T.reduce_sum(product_value, row1, dim=1, clear=True)
                for i_s in T.Parallel(_S):
                    dz_acc[i_s] += (
                        float(scale)
                        * exp_z_shared[i_s]
                        * row1[i_s]
                    )
                T.sync_threads()

            # Publish dC first.  It stays FP32 while forming the route-gate
            # VJP; only dP is rounded to the q/k communication dtype at the
            # final tensor-core handoff.
            T.copy(dc_matrix, matrix_vjp_shared)
            T.sync_threads()

            # C=scale*G*P contributes row-sum minus column-sum to dz.  Do
            # this before reusing read_shared as the explicit dP transpose.
            for i_s, i_t in T.Parallel(_S, _S):
                mat0[i_s, i_t] = (
                    T.cast(read_shared[i_s, i_t], accum_dtype)
                    * T.cast(
                        matrix_vjp_shared[i_s, i_t],
                        accum_dtype,
                    )
                )
            T.reduce_sum(mat0, row0, dim=1, clear=True)
            for i_s, i_t in T.Parallel(_S, _S):
                mat0[i_s, i_t] = (
                    T.cast(read_shared[i_t, i_s], accum_dtype)
                    * T.cast(
                        matrix_vjp_shared[i_t, i_s],
                        accum_dtype,
                    )
                )
            T.reduce_sum(mat0, row1, dim=1, clear=True)
            for i_s in T.Parallel(_S):
                dz_acc[i_s] += row0[i_s] - row1[i_s]

            # dP=scale*G*dC, then dQ=dP K and dK=dP.T Q.
            for i_s, i_t in T.Parallel(_S, _S):
                mat0[i_s, i_t] = (
                    float(scale)
                    * T.cast(causal_shared[i_s, i_t], accum_dtype)
                    * T.cast(
                        matrix_vjp_shared[i_s, i_t],
                        accum_dtype,
                    )
                )
            T.copy(mat0, matrix_scratch_shared)
            T.sync_threads()
            T.gemm(
                matrix_scratch_shared,
                k_shared,
                qk_grad,
                clear_accum=False,
            )
            for i_s, i_k in T.Parallel(_S, _DK):
                if i_s < valid:
                    dq[0, left + i_s, bh, i_k] = qk_grad[i_s, i_k]
            # With two warp groups, a transpose-A WGMMA maps its output
            # accumulator differently from the non-transposed recurrent/KKT
            # GEMMs that own ``state_grad``.  C is dead now, so reuse its
            # shared tile to materialize dP.T without growing shared memory.
            for i_s, i_t in T.Parallel(_S, _S):
                read_shared[i_s, i_t] = (
                    matrix_scratch_shared[i_t, i_s]
                )
            T.sync_threads()
            T.gemm(
                read_shared,
                q_shared,
                state_grad,
                clear_accum=False,
            )

            # Publish dB into the same FP32 communication tile.  Account for
            # its G VJP before the tile is reused by the inverse/KKT path.
            T.copy(db_matrix, matrix_vjp_shared)
            T.sync_threads()
            for i_s, i_t in T.Parallel(_S, _S):
                mat0[i_s, i_t] = (
                    T.cast(write_shared[i_s, i_t], accum_dtype)
                    * T.cast(
                        matrix_vjp_shared[i_s, i_t],
                        accum_dtype,
                    )
                )
            T.reduce_sum(mat0, row0, dim=1, clear=True)
            for i_s, i_t in T.Parallel(_S, _S):
                mat0[i_s, i_t] = (
                    T.cast(write_shared[i_t, i_s], accum_dtype)
                    * T.cast(
                        matrix_vjp_shared[i_t, i_s],
                        accum_dtype,
                    )
                )
            T.reduce_sum(mat0, row1, dim=1, clear=True)
            for i_s in T.Parallel(_S):
                dz_acc[i_s] += row0[i_s] - row1[i_s]

            # Direct beta path and dA=G*beta_col*dB.
            for i_s, i_t in T.Parallel(_S, _S):
                # Explicit transpose before row reduction gives a column sum.
                mat0[i_s, i_t] = (
                    T.cast(causal_shared[i_t, i_s], accum_dtype)
                    * T.cast(a_shared[i_t, i_s], accum_dtype)
                    * T.cast(
                        matrix_vjp_shared[i_t, i_s],
                        accum_dtype,
                    )
                )
            T.reduce_sum(mat0, dbeta_acc, dim=1, clear=True)
            for i_s, i_t in T.Parallel(_S, _S):
                mat0[i_s, i_t] = (
                    T.cast(causal_shared[i_s, i_t], accum_dtype)
                    * beta_shared[i_t]
                    * T.cast(
                        matrix_vjp_shared[i_s, i_t],
                        accum_dtype,
                    )
                )
            T.copy(mat0, matrix_vjp_shared)
            T.sync_threads()

            # A=M^-1:
            # dM=-A.T*dA*A.T, restricted to the strict lower triangle.
            T.gemm(
                a_shared,
                matrix_vjp_shared,
                mat1,
                transpose_A=True,
                clear_accum=True,
            )
            T.copy(mat1, matrix_vjp_shared)
            T.sync_threads()
            T.gemm(
                matrix_vjp_shared,
                a_shared,
                mat2,
                transpose_B=True,
                clear_accum=True,
            )
            for i_s, i_t in T.Parallel(_S, _S):
                if i_s < valid and i_t < valid and i_s > i_t:
                    mat2[i_s, i_t] = -mat2[i_s, i_t]
                else:
                    mat2[i_s, i_t] = 0
            T.copy(mat2, matrix_vjp_shared)
            T.sync_threads()

            # M=I+strict_lower(diag(beta) K K.T).
            T.gemm(
                k_shared,
                k_shared,
                mat0,
                transpose_B=True,
                clear_accum=True,
            )
            for i_s, i_t in T.Parallel(_S, _S):
                mat_reduce[i_s, i_t] = (
                    T.cast(
                        matrix_vjp_shared[i_s, i_t],
                        accum_dtype,
                    )
                    * mat0[i_s, i_t]
                )
            T.reduce_sum(mat_reduce, dbeta_acc, dim=1, clear=False)
            for i_s, i_t in T.Parallel(_S, _S):
                mat0[i_s, i_t] = (
                    beta_shared[i_s]
                    * T.cast(
                        matrix_vjp_shared[i_s, i_t],
                        accum_dtype,
                    )
                    + beta_shared[i_t]
                    * T.cast(
                        matrix_vjp_shared[i_t, i_s],
                        accum_dtype,
                    )
                )
            T.copy(mat0, matrix_scratch_shared)
            T.sync_threads()
            T.gemm(
                matrix_scratch_shared,
                k_shared,
                state_grad,
                clear_accum=False,
            )
            for i_s, i_k in T.Parallel(_S, _DK):
                if i_s < valid:
                    dk[0, left + i_s, bh, i_k] = state_grad[i_s, i_k]

            for i_s in T.Parallel(_S):
                if i_s < valid:
                    dbeta[0, left + i_s, bh] = dbeta_acc[i_s]
                    dz_shared[i_s] = dz_acc[i_s]
                else:
                    dz_shared[i_s] = 0
            T.sync_threads()

            # z_i=sum_{j<=i} g_j, hence
            # dg_i=sum_{j>=i} dz_j.  Return the raw gate VJP directly so the
            # production path needs no extra reverse-cumsum kernel.
            T.clear(dg_acc)
            for j_s in T.serial(_S):
                for i_s in T.Parallel(_S):
                    if i_s <= j_s and j_s < valid:
                        dg_acc[i_s] += dz_shared[j_s]
            for i_s in T.Parallel(_S):
                if i_s < valid:
                    dg[0, left + i_s, bh] = dg_acc[i_s]

    return kernel


def direct_a_chunk_bwd_inline_dd_prototype(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    chunk_start_h: torch.Tensor,
    chunk_end_dh: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    scale: float | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run the isolated inline-dD prototype.

    Returns ``dq, dk, dv, dbeta, dg``.  ``dg`` is already the VJP of the raw
    per-token log gate; the chunk-local suffix sum is fused into this kernel.
    """

    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("The prototype requires packed q with B=1.")
    if q.shape != k.shape:
        raise ValueError("q and k must have identical shapes.")
    if q.shape[2:] != (_H, _DK):
        raise ValueError(
            f"The prototype requires H={_H}, DK={_DK}; got {q.shape[2:]}."
        )
    expected_value_shape = (1, q.shape[1], _H, _DV)
    if v.shape != expected_value_shape or do.shape != expected_value_shape:
        raise ValueError(
            "v/do must both have shape "
            f"{expected_value_shape}; got {v.shape}/{do.shape}."
        )
    if a.shape != (1, q.shape[1], _H, _S):
        raise ValueError(f"a must have shape [1,T,{_H},{_S}].")
    if g_cumsum.shape != beta.shape or beta.shape != (
        1,
        q.shape[1],
        _H,
    ):
        raise ValueError("g_cumsum/beta must have shape [1,T,H].")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must contain at least one document.")
    if chunk_indices.ndim != 2 or chunk_indices.shape[1] != 2:
        raise ValueError("chunk_indices must have shape [num_chunks,2].")
    if cu_seqlens.dtype != chunk_indices.dtype:
        raise TypeError("cu_seqlens and chunk_indices must share a dtype.")
    num_chunks = int(chunk_indices.shape[0])
    expected_state_shape = (1, num_chunks, _H, _DK, _DV)
    if (
        chunk_start_h.shape != expected_state_shape
        or chunk_end_dh.shape != expected_state_shape
    ):
        raise ValueError(
            "chunk_start_h/chunk_end_dh must both have shape "
            f"{expected_state_shape}."
        )
    low_precision_inputs = (
        q,
        k,
        v,
        a,
        beta,
        do,
        chunk_start_h,
        chunk_end_dh,
    )
    if any(tensor.dtype != torch.bfloat16 for tensor in low_precision_inputs):
        raise TypeError(
            "The fixed GPU prototype requires BF16 "
            "q/k/v/a/beta/do/chunk states."
        )
    if g_cumsum.dtype != torch.float32:
        raise TypeError("g_cumsum must use FP32.")
    if cu_seqlens.dtype != torch.int32:
        raise TypeError(
            "The GPU direct-A kernel requires int32 "
            "cu_seqlens/chunk_indices."
        )
    if not all(
        tensor.is_cuda
        for tensor in (
            q,
            k,
            v,
            a,
            g_cumsum,
            beta,
            do,
            chunk_start_h,
            chunk_end_dh,
            cu_seqlens,
            chunk_indices,
        )
    ):
        raise ValueError("All prototype inputs must be CUDA tensors.")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    a = a.contiguous()
    g_cumsum = g_cumsum.contiguous()
    beta = beta.contiguous()
    do = do.contiguous()
    chunk_start_h = chunk_start_h.contiguous()
    chunk_end_dh = chunk_end_dh.contiguous()
    cu_seqlens = cu_seqlens.contiguous()
    chunk_indices = chunk_indices.contiguous()
    scale = float(scale if scale is not None else _DK ** -0.5)

    # Each chunk/head CTA owns and directly stores every valid token element.
    # Emit the activation-gradient dtype in the producer kernel itself instead
    # of writing FP32 tensors and launching four post-kernel cast operations.
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dbeta = torch.empty_like(beta)
    dg = torch.empty_like(g_cumsum, dtype=torch.float32)

    compiled = tilelang_direct_a_chunk_bwd_inline_dd(
        scale=scale,
        qk_dtype=q.dtype,
        value_dtype=v.dtype,
        a_dtype=a.dtype,
        g_dtype=g_cumsum.dtype,
        beta_dtype=beta.dtype,
        state_dtype=chunk_start_h.dtype,
        seqlen_dtype=cu_seqlens.dtype,
        accum_dtype="float32",
        matrix_vjp_dtype=_resolve_matrix_vjp_dtype(),
    )
    compiled(
        q,
        k,
        v,
        a,
        g_cumsum,
        beta,
        do,
        chunk_start_h,
        chunk_end_dh,
        cu_seqlens,
        chunk_indices,
        dq,
        dk,
        dv,
        dbeta,
        dg,
    )
    return dq, dk, dv, dbeta, dg


__all__ = [
    "_resolve_matrix_vjp_dtype",
    "direct_a_chunk_bwd_inline_dd_prototype",
    "tilelang_direct_a_chunk_bwd_inline_dd",
]
