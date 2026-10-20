# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Correctness-first direct-A chunk-local TileLang backward prototype.

This file is intentionally not imported by the production FlashQLA/HSD
dispatch.  It is the first implementation of local Kernel B after the compact
reverse state-adjoint scan (Kernel A).

The specialization is fixed to the current HSD producer:

* packed ``B=1``;
* ``H=6, DK=192, DV=384``;
* internal chunk size ``S=64``;
* one CTA per ``(chunk, head, V64)``.

The V64 CTAs emit FP32 q/k/beta/cumulative-gate partials.  The Python wrapper
reduces the six partials.  This deliberately duplicates the KKT VJP six times;
the next optimization is one CTA per ``(chunk, head)`` looping over all V
tiles.  That final form must also inline
``dD = r * (K @ dH_end) + C.T @ dO`` so the standalone ``dd`` tensor and its
HBM round trip disappear.  Keeping the first prototype simple makes it a
useful parity oracle for that optimized kernel.
"""

import torch
import tilelang
import tilelang.language as T


_H = 6
_DK = 192
_DV = 384
_S = 64
_BV = 64


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_direct_a_chunk_bwd_partial(
    scale: float,
    qk_dtype: str,
    value_dtype: str,
    a_dtype: str,
    g_dtype: str,
    beta_dtype: str,
    state_dtype: str,
    seqlen_dtype: str,
    accum_dtype: str = "float32",
):
    """Return the fixed-shape V64-partial Kernel B."""

    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    num_docs = T.dynamic("num_docs")
    num_value_tiles = _DV // _BV

    qk_shape = (1, num_tokens, _H, _DK)
    value_shape = (1, num_tokens, _H, _DV)
    matrix_shape = (1, num_tokens, _H, _S)
    scalar_shape = (1, num_tokens, _H)
    state_shape = (1, num_chunks, _H, _DK, _DV)
    qk_partial_shape = (
        num_value_tiles,
        1,
        num_tokens,
        _H,
        _DK,
    )
    scalar_partial_shape = (
        num_value_tiles,
        1,
        num_tokens,
        _H,
    )

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
        dd: T.Tensor(value_shape, dtype=value_dtype),
        cu_seqlens: T.Tensor([num_docs + 1], dtype=seqlen_dtype),
        chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
        dq_partial: T.Tensor(qk_partial_shape, dtype=accum_dtype),
        dk_partial: T.Tensor(qk_partial_shape, dtype=accum_dtype),
        dv: T.Tensor(value_shape, dtype=accum_dtype),
        dbeta_partial: T.Tensor(scalar_partial_shape, dtype=accum_dtype),
        dz_partial: T.Tensor(scalar_partial_shape, dtype=accum_dtype),
    ):
        with T.Kernel(
            num_chunks * _H * num_value_tiles,
            threads=128,
        ) as (bchv,):
            bc = bchv // (_H * num_value_tiles)
            hv = bchv % (_H * num_value_tiles)
            bh = hv // num_value_tiles
            bv = hv % num_value_tiles
            value_start = bv * _BV

            doc = chunk_indices[bc, 0]
            local_chunk = chunk_indices[bc, 1]
            seq_start = cu_seqlens[doc]
            seq_end = cu_seqlens[doc + 1]
            left = seq_start + local_chunk * _S
            valid = T.min(_S, seq_end - left)

            # Persistent inputs.
            q_shared = T.alloc_shared((_S, _DK), dtype=qk_dtype)
            k_shared = T.alloc_shared((_S, _DK), dtype=qk_dtype)
            a_shared = T.alloc_shared((_S, _S), dtype=a_dtype)
            h_shared = T.alloc_shared((_DK, _BV), dtype=state_dtype)
            dh_shared = T.alloc_shared((_DK, _BV), dtype=state_dtype)
            v_shared = T.alloc_shared((_S, _BV), dtype=value_dtype)
            do_shared = T.alloc_shared((_S, _BV), dtype=value_dtype)
            dd_shared = T.alloc_shared((_S, _BV), dtype=value_dtype)
            z_shared = T.alloc_shared((_S,), dtype=accum_dtype, scope="shared")
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

            # Matrix communication buffers.  These are BF16/FP16 on purpose,
            # matching the current FlashQLA forward/backward communication
            # precision while all MMA accumulators remain FP32.
            causal_shared = T.alloc_shared((_S, _S), dtype=qk_dtype)
            write_shared = T.alloc_shared((_S, _S), dtype=qk_dtype)
            read_shared = T.alloc_shared((_S, _S), dtype=qk_dtype)
            dbmat_shared = T.alloc_shared((_S, _S), dtype=qk_dtype)
            dcmat_shared = T.alloc_shared((_S, _S), dtype=qk_dtype)
            scratch_matrix_shared = T.alloc_shared((_S, _S), dtype=qk_dtype)

            # Value communication buffers.
            state_read_shared = T.alloc_shared((_S, _BV), dtype=value_dtype)
            residual_shared = T.alloc_shared((_S, _BV), dtype=value_dtype)
            delta_shared = T.alloc_shared((_S, _BV), dtype=value_dtype)

            # FP32 MMA/reduction fragments.
            mat0 = T.alloc_fragment((_S, _S), dtype=accum_dtype)
            mat1 = T.alloc_fragment((_S, _S), dtype=accum_dtype)
            mat2 = T.alloc_fragment((_S, _S), dtype=accum_dtype)
            val0 = T.alloc_fragment((_S, _BV), dtype=accum_dtype)
            val1 = T.alloc_fragment((_S, _BV), dtype=accum_dtype)
            val2 = T.alloc_fragment((_S, _BV), dtype=accum_dtype)
            dq_fragment = T.alloc_fragment((_S, _DK), dtype=accum_dtype)
            dk_fragment = T.alloc_fragment((_S, _DK), dtype=accum_dtype)
            product_value = T.alloc_fragment((_S, _BV), dtype=accum_dtype)
            product_state = T.alloc_fragment((_DK, _BV), dtype=accum_dtype)
            row0 = T.alloc_fragment((_S,), dtype=accum_dtype)
            row1 = T.alloc_fragment((_S,), dtype=accum_dtype)
            key_row = T.alloc_fragment((_DK,), dtype=accum_dtype)
            dz_fragment = T.alloc_fragment((_S,), dtype=accum_dtype)
            dbeta_fragment = T.alloc_fragment((_S,), dtype=accum_dtype)
            scalar0 = T.alloc_fragment((1,), dtype=accum_dtype)
            scalar1 = T.alloc_fragment((1,), dtype=accum_dtype)

            T.annotate_layout(
                {
                    q_shared: tilelang.layout.make_swizzled_layout(q_shared),
                    k_shared: tilelang.layout.make_swizzled_layout(k_shared),
                    a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                    h_shared: tilelang.layout.make_swizzled_layout(h_shared),
                    dh_shared: tilelang.layout.make_swizzled_layout(dh_shared),
                    v_shared: tilelang.layout.make_swizzled_layout(v_shared),
                    do_shared: tilelang.layout.make_swizzled_layout(do_shared),
                    dd_shared: tilelang.layout.make_swizzled_layout(dd_shared),
                    causal_shared: tilelang.layout.make_swizzled_layout(
                        causal_shared
                    ),
                    write_shared: tilelang.layout.make_swizzled_layout(
                        write_shared
                    ),
                    read_shared: tilelang.layout.make_swizzled_layout(
                        read_shared
                    ),
                    dbmat_shared: tilelang.layout.make_swizzled_layout(
                        dbmat_shared
                    ),
                    dcmat_shared: tilelang.layout.make_swizzled_layout(
                        dcmat_shared
                    ),
                    scratch_matrix_shared: tilelang.layout.make_swizzled_layout(
                        scratch_matrix_shared
                    ),
                    state_read_shared: tilelang.layout.make_swizzled_layout(
                        state_read_shared
                    ),
                    residual_shared: tilelang.layout.make_swizzled_layout(
                        residual_shared
                    ),
                    delta_shared: tilelang.layout.make_swizzled_layout(
                        delta_shared
                    ),
                }
            )
            T.use_swizzle(10)
            T.set_max_nreg(240, 1)

            # Masked packed loads.  A tail chunk is padded with zero data and
            # a constant cumulative gate; all matrix paths also apply an
            # explicit valid-row/valid-column mask.
            for i_s, i_k in T.Parallel(_S, _DK):
                if i_s < valid:
                    q_shared[i_s, i_k] = q[0, left + i_s, bh, i_k]
                    k_shared[i_s, i_k] = k[0, left + i_s, bh, i_k]
                else:
                    q_shared[i_s, i_k] = 0
                    k_shared[i_s, i_k] = 0
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
                    dd_shared[i_s, i_v] = dd[
                        0,
                        left + i_s,
                        bh,
                        value_start + i_v,
                    ]
                else:
                    v_shared[i_s, i_v] = 0
                    do_shared[i_s, i_v] = 0
                    dd_shared[i_s, i_v] = 0
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
                    z_shared[i_s] = g[0, left + valid - 1, bh]
                    beta_shared[i_s] = 0
            T.sync_threads()

            for i_s in T.Parallel(_S):
                exp_z_shared[i_s] = T.exp2(z_shared[i_s] * 1.442695)
                reverse_decay_shared[i_s] = T.exp2(
                    (
                        z_shared[valid - 1]
                        - z_shared[i_s]
                    )
                    * 1.442695
                )

            # G, B=G*A*beta_col, C=scale*G*(QK.T).
            T.gemm(
                q_shared,
                k_shared,
                mat0,
                transpose_B=True,
                clear_accum=True,
            )
            for i_s, i_t in T.Parallel(_S, _S):
                if (
                    i_s < valid
                    and i_t < valid
                    and i_s >= i_t
                ):
                    mat1[i_s, i_t] = T.exp2(
                        (z_shared[i_s] - z_shared[i_t]) * 1.442695
                    )
                else:
                    mat1[i_s, i_t] = 0
            T.copy(mat1, causal_shared)
            for i_s, i_t in T.Parallel(_S, _S):
                mat2[i_s, i_t] = (
                    mat1[i_s, i_t]
                    * a_shared[i_s, i_t]
                    * beta_shared[i_t]
                )
                mat0[i_s, i_t] *= float(scale) * mat1[i_s, i_t]
            T.copy(mat2, write_shared)
            T.copy(mat0, read_shared)
            T.sync_threads()

            # Rebuild the direct-A chunk values.
            T.gemm(
                k_shared,
                h_shared,
                val0,
                clear_accum=True,
            )
            T.copy(val0, state_read_shared)
            for i_s, i_v in T.Parallel(_S, _BV):
                val0[i_s, i_v] = (
                    v_shared[i_s, i_v]
                    - exp_z_shared[i_s] * val0[i_s, i_v]
                )
            T.copy(val0, residual_shared)
            T.sync_threads()
            T.gemm(
                write_shared,
                residual_shared,
                val1,
                clear_accum=True,
            )
            T.copy(val1, delta_shared)
            T.sync_threads()
            for i_s, i_v in T.Parallel(_S, _BV):
                val2[i_s, i_v] = (
                    reverse_decay_shared[i_s] * val1[i_s, i_v]
                )
            # dK from H1 = decay*H0 + K.T@N.  Initialize the persistent
            # dK accumulator before reusing val2 for dN.  MMA inputs must
            # have a common low-precision dtype, so reuse the now-dead
            # v_shared buffer as the N communication tile.
            T.copy(val2, v_shared)
            T.sync_threads()
            T.gemm(
                v_shared,
                dh_shared,
                dk_fragment,
                transpose_B=True,
                clear_accum=True,
            )

            # dN is needed only for the ratio-gate derivative.  `dd` already
            # contains r*dN + C.T*dO from Kernel A.
            T.gemm(
                k_shared,
                dh_shared,
                val2,
                clear_accum=True,
            )
            for i_s, i_v in T.Parallel(_S, _BV):
                product_value[i_s, i_v] = (
                    val2[i_s, i_v] * delta_shared[i_s, i_v]
                )
            T.reduce_sum(product_value, row0, dim=1, clear=True)
            for i_s in T.Parallel(_S):
                row0[i_s] *= reverse_decay_shared[i_s]
                dz_fragment[i_s] = -row0[i_s]
            T.reduce_sum(row0, scalar0, dim=0, clear=True)

            # decay * <dH_end, H_start>.
            for i_k, i_v in T.Parallel(_DK, _BV):
                product_state[i_k, i_v] = (
                    dh_shared[i_k, i_v] * h_shared[i_k, i_v]
                )
            T.reduce_sum(product_state, key_row, dim=1, clear=True)
            T.reduce_sum(key_row, scalar1, dim=0, clear=True)
            dz_fragment[valid - 1] += (
                scalar0[0]
                + exp_z_shared[valid - 1] * scalar1[0]
            )

            # dW = B.T @ dD.  dV owns a disjoint V64 tile and writes directly.
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
                val1[i_s, i_v] = -exp_z_shared[i_s] * val0[i_s, i_v]
            for i_s, i_v in T.Parallel(_S, _BV):
                product_value[i_s, i_v] = (
                    val1[i_s, i_v] * state_read_shared[i_s, i_v]
                )
            T.reduce_sum(product_value, row1, dim=1, clear=True)
            for i_s in T.Parallel(_S):
                dz_fragment[i_s] += row1[i_s]
            T.copy(val1, v_shared)
            T.sync_threads()
            T.gemm(
                v_shared,
                h_shared,
                dk_fragment,
                transpose_B=True,
                clear_accum=False,
            )

            # dB=dD W.T and dC=dO D.T.
            T.gemm(
                dd_shared,
                residual_shared,
                mat0,
                transpose_B=True,
                clear_accum=True,
            )
            T.copy(mat0, dbmat_shared)
            T.gemm(
                do_shared,
                delta_shared,
                mat1,
                transpose_B=True,
                clear_accum=True,
            )
            T.copy(mat1, dcmat_shared)
            T.sync_threads()

            # Base q/k contributions from the recurrent/output state paths.
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
                dq_fragment,
                transpose_B=True,
                clear_accum=True,
            )
            # dP=scale*G*dC and its Q/K gradients.
            for i_s, i_t in T.Parallel(_S, _S):
                mat2[i_s, i_t] = (
                    float(scale)
                    * causal_shared[i_s, i_t]
                    * dcmat_shared[i_s, i_t]
                )
            T.copy(mat2, scratch_matrix_shared)
            T.sync_threads()
            T.gemm(
                scratch_matrix_shared,
                k_shared,
                dq_fragment,
                clear_accum=False,
            )
            T.gemm(
                scratch_matrix_shared,
                q_shared,
                dk_fragment,
                transpose_A=True,
                clear_accum=False,
            )

            # Direct beta path and dA.
            for i_s, i_t in T.Parallel(_S, _S):
                mat0[i_s, i_t] = (
                    causal_shared[i_s, i_t]
                    * beta_shared[i_t]
                    * dbmat_shared[i_s, i_t]
                )
                # Transpose the column-beta contribution so reduce(dim=1)
                # produces one value per beta column.
                mat1[i_s, i_t] = (
                    causal_shared[i_t, i_s]
                    * a_shared[i_t, i_s]
                    * dbmat_shared[i_t, i_s]
                )
            T.copy(mat0, scratch_matrix_shared)
            T.sync_threads()
            T.reduce_sum(
                mat1,
                dbeta_fragment,
                dim=1,
                clear=True,
            )

            # A=M^{-1}: dM=-A.T*dA*A.T, strict-lower only.
            T.gemm(
                a_shared,
                scratch_matrix_shared,
                mat1,
                transpose_A=True,
                clear_accum=True,
            )
            T.copy(mat1, scratch_matrix_shared)
            T.sync_threads()
            T.gemm(
                scratch_matrix_shared,
                a_shared,
                mat2,
                transpose_B=True,
                clear_accum=True,
            )
            for i_s, i_t in T.Parallel(_S, _S):
                if (
                    i_s < valid
                    and i_t < valid
                    and i_s > i_t
                ):
                    mat2[i_s, i_t] = -mat2[i_s, i_t]
                else:
                    mat2[i_s, i_t] = 0
            T.copy(mat2, scratch_matrix_shared)
            T.sync_threads()

            # KKT beta path and Gram=K K.T path.
            T.gemm(
                k_shared,
                k_shared,
                mat0,
                transpose_B=True,
                clear_accum=True,
            )
            for i_s, i_t in T.Parallel(_S, _S):
                mat1[i_s, i_t] = mat2[i_s, i_t] * mat0[i_s, i_t]
            T.reduce_sum(
                mat1,
                dbeta_fragment,
                dim=1,
                clear=False,
            )
            for i_s, i_t in T.Parallel(_S, _S):
                mat0[i_s, i_t] = (
                    beta_shared[i_s] * mat2[i_s, i_t]
                    + beta_shared[i_t]
                    * scratch_matrix_shared[i_t, i_s]
                )
            T.copy(mat0, scratch_matrix_shared)
            T.sync_threads()
            T.gemm(
                scratch_matrix_shared,
                k_shared,
                dk_fragment,
                clear_accum=False,
            )

            # Direct exp(z) output path: scale*E*<dO,QH>.
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
                dz_fragment[i_s] += (
                    float(scale) * exp_z_shared[i_s] * row1[i_s]
                )

            # G paths: X=C*dC+B*dB, dz += rowsum(X)-colsum(X).
            for i_s, i_t in T.Parallel(_S, _S):
                mat0[i_s, i_t] = (
                    read_shared[i_s, i_t] * dcmat_shared[i_s, i_t]
                    + write_shared[i_s, i_t] * dbmat_shared[i_s, i_t]
                )
                mat1[i_s, i_t] = (
                    read_shared[i_t, i_s] * dcmat_shared[i_t, i_s]
                    + write_shared[i_t, i_s] * dbmat_shared[i_t, i_s]
                )
            T.reduce_sum(mat0, row0, dim=1, clear=True)
            T.reduce_sum(mat1, row1, dim=1, clear=True)
            for i_s in T.Parallel(_S):
                dz_fragment[i_s] += row0[i_s] - row1[i_s]

            # One CTA owns every token in its chunk for this V tile.
            for i_s, i_k in T.Parallel(_S, _DK):
                if i_s < valid:
                    dq_partial[
                        bv,
                        0,
                        left + i_s,
                        bh,
                        i_k,
                    ] = dq_fragment[i_s, i_k]
                    dk_partial[
                        bv,
                        0,
                        left + i_s,
                        bh,
                        i_k,
                    ] = dk_fragment[i_s, i_k]
            for i_s in T.Parallel(_S):
                if i_s < valid:
                    dbeta_partial[
                        bv,
                        0,
                        left + i_s,
                        bh,
                    ] = dbeta_fragment[i_s]
                    dz_partial[
                        bv,
                        0,
                        left + i_s,
                        bh,
                    ] = dz_fragment[i_s]

    return kernel


def direct_a_chunk_bwd_prototype(
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
    dd: torch.Tensor,
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
    """Run the standalone partial prototype.

    Returns ``dq, dk, dv, dbeta, dz``.  ``dz`` is the gradient with respect
    to the chunk-local cumulative log gate.  The production wrapper must
    apply a packed chunk-local reverse cumsum to obtain the raw gate gradient.
    """

    if q.ndim != 4 or tuple(q.shape[:1]) != (1,):
        raise ValueError("The prototype requires packed q with B=1.")
    if tuple(q.shape) != tuple(k.shape):
        raise ValueError("q and k must have identical shapes.")
    if q.shape[2:] != (_H, _DK):
        raise ValueError(
            f"The prototype requires H={_H}, DK={_DK}; got {q.shape[2:]}."
        )
    if v.shape != (1, q.shape[1], _H, _DV):
        raise ValueError(
            f"The prototype requires v=[1,T,{_H},{_DV}], got {tuple(v.shape)}."
        )
    if do.shape != v.shape or dd.shape != v.shape:
        raise ValueError("v/do/dd must have identical shapes.")
    if a.shape != (1, q.shape[1], _H, _S):
        raise ValueError(f"a must have shape [1,T,{_H},{_S}].")
    if g_cumsum.shape != beta.shape or beta.shape != (
        1,
        q.shape[1],
        _H,
    ):
        raise ValueError("g_cumsum/beta must have shape [1,T,H].")
    if chunk_indices.ndim != 2 or chunk_indices.shape[1] != 2:
        raise ValueError("chunk_indices must have shape [num_chunks, 2].")
    num_chunks = int(chunk_indices.shape[0])
    expected_state_shape = (1, num_chunks, _H, _DK, _DV)
    if (
        chunk_start_h.shape != expected_state_shape
        or chunk_end_dh.shape != expected_state_shape
    ):
        raise ValueError(
            "chunk_start_h/chunk_end_dh must have shape "
            f"{expected_state_shape}."
        )
    if cu_seqlens.dtype != chunk_indices.dtype:
        raise TypeError("cu_seqlens and chunk_indices must have the same dtype.")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("q/k must use FP16 or BF16.")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    a = a.contiguous()
    g_cumsum = g_cumsum.contiguous()
    beta = beta.contiguous()
    do = do.contiguous()
    chunk_start_h = chunk_start_h.contiguous()
    chunk_end_dh = chunk_end_dh.contiguous()
    dd = dd.contiguous()
    cu_seqlens = cu_seqlens.contiguous()
    chunk_indices = chunk_indices.contiguous()
    scale = float(scale if scale is not None else _DK ** -0.5)

    num_value_tiles = _DV // _BV
    dq_partial = torch.empty(
        (num_value_tiles,) + tuple(q.shape),
        dtype=torch.float32,
        device=q.device,
    )
    dk_partial = torch.empty_like(dq_partial)
    dv = torch.empty_like(v, dtype=torch.float32)
    dbeta_partial = torch.empty(
        (num_value_tiles,) + tuple(beta.shape),
        dtype=torch.float32,
        device=beta.device,
    )
    dz_partial = torch.empty_like(dbeta_partial)

    compiled = tilelang_direct_a_chunk_bwd_partial(
        scale=scale,
        qk_dtype=q.dtype,
        value_dtype=v.dtype,
        a_dtype=a.dtype,
        g_dtype=g_cumsum.dtype,
        beta_dtype=beta.dtype,
        state_dtype=chunk_start_h.dtype,
        seqlen_dtype=cu_seqlens.dtype,
        accum_dtype="float32",
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
        dd,
        cu_seqlens,
        chunk_indices,
        dq_partial,
        dk_partial,
        dv,
        dbeta_partial,
        dz_partial,
    )
    return (
        dq_partial.sum(dim=0),
        dk_partial.sum(dim=0),
        dv,
        dbeta_partial.sum(dim=0),
        dz_partial.sum(dim=0),
    )


__all__ = [
    "direct_a_chunk_bwd_prototype",
    "tilelang_direct_a_chunk_bwd_partial",
]
