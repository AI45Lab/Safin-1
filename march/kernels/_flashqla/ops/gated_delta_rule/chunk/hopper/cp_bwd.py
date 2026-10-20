# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch
import tilelang
import tilelang.language as T

from march.kernels._flashqla.utils import prepare_chunk_offsets


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_prepare_dh_ws(
    H,
    Hg,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    b_dtype,
    dht_dtype,
    dh_dtype,
    o_dtype,
    seqlen_dtype,
    use_dht,
    store_dh0,
    store_dh,
    is_varlen,
    is_cp,
    state_v_first,
    value_tile_size=0,
    use_compact_dh=False,
    use_direct_anchor_output_grad=False,
    use_direct_anchor_state_grad=False,
    compact_chunk_stride=1,
    num_stages=2,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    num_anchors = T.dynamic("num_anchors")
    block_S = chunk_size
    block_V = value_tile_size if value_tile_size > 0 else DV
    num_value_tiles = (DV + block_V - 1) // block_V

    if is_varlen:
        q_shape = (1, num_tokens, Hg, DK)
        k_shape = (1, num_tokens, Hg, DK)
        a_shape = (1, num_tokens, H, chunk_size)
        g_shape = (1, num_tokens, H)
        b_shape = (1, num_tokens, H)
        do_shape = (1, num_tokens, H, DV)
        dh_shape = (
            (1, num_chunks, H, DV, DK)
            if state_v_first
            else (1, num_chunks, H, DK, DV)
        )
    else:
        q_shape = (batch_size, num_tokens, Hg, DK)
        k_shape = (batch_size, num_tokens, Hg, DK)
        a_shape = (batch_size, num_tokens, H, chunk_size)
        g_shape = (batch_size, num_tokens, H)
        b_shape = (batch_size, num_tokens, H)
        do_shape = (batch_size, num_tokens, H, DV)
        dh_shape = (
            (batch_size, num_chunks, H, DV, DK)
            if state_v_first
            else (batch_size, num_chunks, H, DK, DV)
        )
    dht_shape = (
        (batch_size, H, DV, DK)
        if state_v_first
        else (batch_size, H, DK, DV)
    )
    dh0_shape = (
        (batch_size, H, DV, DK)
        if state_v_first
        else (batch_size, H, DK, DV)
    )
    compact_dh_shape = (1, num_anchors, H, DK, DV)
    anchor_q_shape = (1, num_anchors, H, DK)
    danchor_o_shape = (1, num_anchors, H, DV)

    @T.prim_func
    def tilelang_prepare_dh_kernel(
        q: T.Tensor(q_shape, dtype=qkva_dtype),
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        do: T.Tensor(do_shape, dtype=o_dtype),
        dht: T.Tensor(dht_shape, dtype=dht_dtype),
        compact_dh: T.Tensor(compact_dh_shape, dtype=accum_dtype),
        anchor_q: T.Tensor(anchor_q_shape, dtype=qkva_dtype),
        danchor_o: T.Tensor(danchor_o_shape, dtype=o_dtype),
        danchor_states: T.Tensor(compact_dh_shape, dtype=accum_dtype),
        cu_seqlens: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        chunk_offsets: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        anchor_cu_seqlens: T.Tensor(
            [batch_size + 1],
            dtype=seqlen_dtype,
        ),
        num_warmup_chunks: T.Tensor([batch_size, H], dtype=seqlen_dtype),
        dh: T.Tensor(dh_shape, dtype=dh_dtype),
        dh0: T.Tensor(dh0_shape, dtype=accum_dtype),
    ):
        with T.Kernel(batch_size * H * num_value_tiles, threads=512) as (bbhv,):
            bb = bbhv // (H * num_value_tiles)
            bhv = bbhv % (H * num_value_tiles)
            bh = bhv // num_value_tiles
            bv = bhv % num_value_tiles
            bhg = bh // (H // Hg)
            value_start = bv * block_V

            batch_idx = T.alloc_var("int32")
            seq_start_idx = T.alloc_var("int32")
            seq_end_idx = T.alloc_var("int32")
            chunk_start_idx = T.alloc_var("int32")

            batch_idx = 0 if is_varlen else bb
            seq_start_idx = cu_seqlens[bb] if is_varlen else 0
            seq_end_idx = cu_seqlens[bb + 1] if is_varlen else num_tokens
            chunk_start_idx = chunk_offsets[bb] if is_varlen else 0

            num_iters = T.alloc_var("int32")
            num_iters = (
                num_warmup_chunks[bb, bh]
                if is_cp
                else T.ceildiv(seq_end_idx - seq_start_idx, block_S)
            )

            # ===== Staged shared memory (pipeline) =====
            k_shared = T.alloc_shared(
                (num_stages, block_S, DK), dtype=qkva_dtype
            )
            q_shared = T.alloc_shared(
                (num_stages, block_S, DK), dtype=qkva_dtype
            )
            a_shared = T.alloc_shared(
                (num_stages, block_S, block_S), dtype=qkva_dtype
            )
            do_shared = T.alloc_shared(
                (num_stages, block_S, block_V), dtype=o_dtype
            )
            g_shared = T.alloc_shared(
                (num_stages, block_S), dtype=accum_dtype, scope="shared"
            )
            b_shared = T.alloc_shared(
                (num_stages, block_S), dtype=accum_dtype, scope="shared"
            )

            # ===== Communication buffers =====
            dh_shared = T.alloc_shared(
                (block_V, DK) if state_v_first else (DK, block_V),
                dtype=qkva_dtype,
            )
            x_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            y_shared = T.alloc_shared((block_S, block_V), dtype=qkva_dtype)
            p_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)

            # ===== Fragments =====
            dh_fragment = T.alloc_fragment(
                (block_V, DK) if state_v_first else (DK, block_V),
                dtype=accum_dtype,
            )
            x_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            y_fragment = T.alloc_fragment((block_S, block_V), dtype=accum_dtype)
            p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            r_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            g_last_local_dh = T.alloc_local((1), dtype=accum_dtype)
            g_last_local_y = T.alloc_local((1), dtype=accum_dtype)

            # ===== Pipeline barriers =====
            data_is_ready = T.alloc_barrier(arrive_count=[96] * num_stages)
            data_is_free = T.alloc_barrier(arrive_count=[384] * num_stages)

            # ===== Compute barriers =====
            # bar_0: full sync (3 consumers 128 each + P3 32)
            bar_0 = T.alloc_barrier(arrive_count=416)
            # bar_1: stage 0 done (Consumer_PR 128 + Consumer_XY 128)
            bar_1 = T.alloc_barrier(arrive_count=256)
            # bar_2: stage 1 done (Consumer_PR 128 + Consumer_XY 128)
            bar_2 = T.alloc_barrier(arrive_count=256)
            # bar_3: stage 2 done (Consumer_dh 128 + Consumer_PR 128)
            bar_3 = T.alloc_barrier(arrive_count=256)
            # bar_4: stage 3 done (Consumer_dh 128 + Consumer_PR 128 + Consumer_XY 128)
            bar_4 = T.alloc_barrier(arrive_count=384)
            # bar_5: stage 4 done (Consumer_dh 128 + Consumer_XY 128)
            bar_5 = T.alloc_barrier(arrive_count=256)

            T.use_swizzle(10)

            tx = T.get_thread_binding()

            PRODUCER_NREG = 24
            CONSUMER_DH_NREG = 160
            CONSUMER_XY_NREG = 128
            CONSUMER_PR_NREG = 128

            if tx < 128:
                # ============ Consumer_dh ============
                # Owns dh_fragment accumulator.
                # Phase 1: dh = g_last * dh + X^T @ Y
                # Phase 2: dh += R^T @ dOg
                T.set_max_nreg(CONSUMER_DH_NREG, 1)

                if use_dht:
                    if state_v_first:
                        T.copy(
                            dht[
                                bb,
                                bh,
                                value_start : value_start + block_V,
                                0:DK,
                            ],
                            dh_fragment,
                        )
                    else:
                        T.copy(
                            dht[
                                bb,
                                bh,
                                0:DK,
                                value_start : value_start + block_V,
                            ],
                            dh_fragment,
                        )
                else:
                    T.clear(dh_fragment)

                for i_s in T.serial(num_iters):
                    # The compact HSD gradient belongs to the end state of
                    # this reverse iteration. Inject it before publishing
                    # that boundary adjoint and before applying the local
                    # gate/state Jacobian.
                    if (
                        use_compact_dh
                        or use_direct_anchor_output_grad
                        or use_direct_anchor_state_grad
                    ):
                        chunk_idx = num_iters - i_s - 1
                        if (chunk_idx + 1) % compact_chunk_stride == 0:
                            anchor_ordinal = (
                                chunk_idx + 1
                            ) // compact_chunk_stride
                            doc_anchor_count = (
                                anchor_cu_seqlens[bb + 1]
                                - anchor_cu_seqlens[bb]
                            )
                            if anchor_ordinal <= doc_anchor_count:
                                anchor_idx = (
                                    anchor_cu_seqlens[bb]
                                    + anchor_ordinal
                                    - 1
                                )
                                if state_v_first:
                                    for j_v, j_k in T.Parallel(block_V, DK):
                                        if use_compact_dh:
                                            dh_fragment[j_v, j_k] += compact_dh[
                                                0,
                                                anchor_idx,
                                                bh,
                                                j_k,
                                                value_start + j_v,
                                            ]
                                        if use_direct_anchor_state_grad:
                                            dh_fragment[j_v, j_k] += danchor_states[
                                                0,
                                                anchor_idx,
                                                bh,
                                                j_k,
                                                value_start + j_v,
                                            ]
                                        if use_direct_anchor_output_grad:
                                            dh_fragment[j_v, j_k] += (
                                                scale
                                                * anchor_q[
                                                    0,
                                                    anchor_idx,
                                                    bh,
                                                    j_k,
                                                ]
                                                * danchor_o[
                                                    0,
                                                    anchor_idx,
                                                    bh,
                                                    value_start + j_v,
                                                ]
                                            )
                                else:
                                    for j_k, j_v in T.Parallel(DK, block_V):
                                        if use_compact_dh:
                                            dh_fragment[j_k, j_v] += compact_dh[
                                                0,
                                                anchor_idx,
                                                bh,
                                                j_k,
                                                value_start + j_v,
                                            ]
                                        if use_direct_anchor_state_grad:
                                            dh_fragment[j_k, j_v] += danchor_states[
                                                0,
                                                anchor_idx,
                                                bh,
                                                j_k,
                                                value_start + j_v,
                                            ]
                                        if use_direct_anchor_output_grad:
                                            dh_fragment[j_k, j_v] += (
                                                scale
                                                * anchor_q[
                                                    0,
                                                    anchor_idx,
                                                    bh,
                                                    j_k,
                                                ]
                                                * danchor_o[
                                                    0,
                                                    anchor_idx,
                                                    bh,
                                                    value_start + j_v,
                                                ]
                                            )
                    T.barrier_wait(
                        data_is_ready[i_s % num_stages],
                        (i_s // num_stages + 0) % 2,
                    )
                    T.barrier_arrive(bar_0)

                    # [stage 0-2] exp(g), copy dh to shared
                    T.barrier_wait(bar_0, i_s % 2)

                    # exp(g)
                    for j_s in T.Parallel(block_S):
                        g_shared[i_s % num_stages, j_s] = T.exp2(
                            g_shared[i_s % num_stages, j_s] * 1.442695
                        )

                    # copy dh to shared
                    T.copy(dh_fragment, dh_shared)

                    T.barrier_arrive(bar_3)

                    # [stage 3] dh decay

                    T.barrier_wait(bar_3, i_s % 2)

                    # dh = g_last * dh  (g_shared already holds exp(g))
                    g_last_local_dh[0] = g_shared[i_s % num_stages, block_S - 1]
                    if state_v_first:
                        for j_v, j_k in T.Parallel(block_V, DK):
                            dh_fragment[j_v, j_k] *= g_last_local_dh[0]
                    else:
                        for j_k, j_v in T.Parallel(DK, block_V):
                            dh_fragment[j_k, j_v] *= g_last_local_dh[0]

                    T.barrier_arrive(bar_4)

                    # [stage 4] wait Rg and dO ready
                    T.barrier_wait(bar_4, i_s % 2)
                    # dh += Rg^T @ dO
                    if state_v_first:
                        T.gemm(
                            do_shared[i_s % num_stages, :, :],
                            q_shared[i_s % num_stages, :, :],
                            dh_fragment,
                            transpose_A=True,
                            clear_accum=False,
                        )
                    else:
                        T.gemm(
                            q_shared[i_s % num_stages, :, :],
                            do_shared[i_s % num_stages, :, :],
                            dh_fragment,
                            transpose_A=True,
                            clear_accum=False,
                        )
                    T.barrier_arrive(bar_5)

                    # [stage 5] wait X and Y ready
                    T.barrier_wait(bar_5, i_s % 2)

                    if state_v_first:
                        T.gemm(
                            y_shared,
                            x_shared,
                            dh_fragment,
                            transpose_A=True,
                            clear_accum=False,
                        )
                    else:
                        T.gemm(
                            x_shared,
                            y_shared,
                            dh_fragment,
                            transpose_A=True,
                            clear_accum=False,
                        )

                    T.barrier_arrive(data_is_free[i_s % num_stages])

                if store_dh0:
                    if state_v_first:
                        T.copy(
                            dh_fragment,
                            dh0[
                                bb,
                                bh,
                                value_start : value_start + block_V,
                                0:DK,
                            ],
                        )
                    else:
                        T.copy(
                            dh_fragment,
                            dh0[
                                bb,
                                bh,
                                0:DK,
                                value_start : value_start + block_V,
                            ],
                        )

            elif tx < 256:
                # ============ Consumer_PR ============
                # Step 1: P = QK^T
                # Step 2: PL = -Lower(P)
                # Step 3: R = Q + PL @ X
                # Step 4: Rg = diag(g) @ R
                T.set_max_nreg(CONSUMER_PR_NREG, 1)

                for i_s in T.serial(num_iters):
                    T.barrier_wait(
                        data_is_ready[i_s % num_stages],
                        (i_s // num_stages + 0) % 2,
                    )
                    T.barrier_arrive(bar_0)

                    # [stage 0] full sync
                    T.barrier_wait(bar_0, i_s % 2)

                    # Step 1: P = Q @ K^T
                    T.gemm(
                        q_shared[i_s % num_stages, :, :],
                        k_shared[i_s % num_stages, :, :],
                        p_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )

                    T.barrier_arrive(bar_1)
                    # [stage 1]
                    T.barrier_wait(bar_1, i_s % 2)

                    # Step 2: PL = -Lower(P)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        if j_s < j_t:
                            p_fragment[j_s, j_t] = 0
                        else:
                            p_fragment[j_s, j_t] *= -1
                    T.copy(p_fragment, p_shared)

                    T.barrier_arrive(bar_2)
                    # [stage 2]
                    T.barrier_wait(bar_2, i_s % 2)

                    # Step 3: R = Q + P @ X
                    T.copy(q_shared[i_s % num_stages, :, :], r_fragment)
                    T.gemm(
                        p_shared,
                        x_shared,
                        r_fragment,
                        clear_accum=False,
                    )

                    T.barrier_arrive(bar_3)
                    # [stage 3]
                    T.barrier_wait(bar_3, i_s % 2)

                    # Step 4: Rg = scale * diag(g) @ R
                    for j_s, j_k in T.Parallel(block_S, DK):
                        r_fragment[j_s, j_k] *= scale * g_shared[i_s % num_stages, j_s]
                    T.copy(r_fragment, q_shared[i_s % num_stages, :, :])

                    T.barrier_arrive(bar_4)
                    T.barrier_arrive(data_is_free[i_s % num_stages])

            elif tx < 384:
                # ============ Consumer_XY ============
                # Step 1: Ab = A * diag(b)
                # Step 2: X = Ab @ K
                # Step 3: Y = K @ dh
                # Step 4: Yg = -g_last * Y
                T.set_max_nreg(CONSUMER_XY_NREG, 1)

                for i_s in T.serial(num_iters):
                    T.barrier_wait(
                        data_is_ready[i_s % num_stages],
                        (i_s // num_stages + 0) % 2,
                    )
                    T.barrier_arrive(bar_0)

                    # [stage 0] full sync
                    T.barrier_wait(bar_0, i_s % 2)

                    # Step 1: Ab = A * diag(b)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_shared[i_s % num_stages, j_s, j_t] *= b_shared[i_s % num_stages, j_t]

                    T.barrier_arrive(bar_1)
                    # [stage 1]
                    T.barrier_wait(bar_1, i_s % 2)
                    # Step 2: X = Ab @ K
                    T.gemm(
                        a_shared[i_s % num_stages, :, :],
                        k_shared[i_s % num_stages, :, :],
                        x_fragment,
                        clear_accum=True,
                    )
                    T.copy(x_fragment, x_shared)

                    T.barrier_arrive(bar_2)


                    # [stage 3]
                    T.barrier_wait(bar_3, i_s % 2)

                    # Step 3: Y = K @ dh
                    if state_v_first:
                        T.gemm(
                            k_shared[i_s % num_stages, :, :],
                            dh_shared,
                            y_fragment,
                            transpose_B=True,
                            clear_accum=True,
                        )
                    else:
                        T.gemm(
                            k_shared[i_s % num_stages, :, :],
                            dh_shared,
                            y_fragment,
                            clear_accum=True,
                        )

                    T.barrier_arrive(bar_4)
                    # [stage 4]
                    T.barrier_wait(bar_4, i_s % 2)

                    # Step 4: Yg = -g_last * Y  (g_shared already holds exp(g))
                    g_last_local_y[0] = g_shared[i_s % num_stages, block_S - 1]

                    for j_s, j_v in T.Parallel(block_S, block_V):
                        y_fragment[j_s, j_v] *= -g_last_local_y[0]

                    T.copy(y_fragment, y_shared)

                    T.barrier_arrive(bar_5)

                    T.barrier_arrive(data_is_free[i_s % num_stages])

            else:
                # ============ Producer ============
                T.set_max_nreg(PRODUCER_NREG, 0)

                if tx < 384 + 32:
                    # P0: TMA load K and Q
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(
                            data_is_free[i_s % num_stages],
                            (i_s // num_stages + 1) % 2,
                        )
                        chunk_idx = num_iters - 1 - i_s
                        left = seq_start_idx + chunk_idx * block_S
                        right = left + block_S

                        if right <= seq_end_idx:
                            T.tma_copy(
                                k[batch_idx, left:right, bhg, 0:DK],
                                k_shared[i_s % num_stages, :, :],
                                barrier=data_is_ready[i_s % num_stages],
                            )
                        else:
                            for j_s, j_k in T.Parallel(block_S, DK):
                                if left + j_s < seq_end_idx:
                                    k_shared[i_s % num_stages, j_s, j_k] = k[batch_idx, left + j_s, bhg, j_k]
                                else:
                                    k_shared[i_s % num_stages, j_s, j_k] = 0
                        if right <= seq_end_idx:
                            T.tma_copy(
                                q[batch_idx, left:right, bhg, 0:DK],
                                q_shared[i_s % num_stages, :, :],
                                barrier=data_is_ready[i_s % num_stages],
                            )
                        else:
                            for j_s, j_k in T.Parallel(block_S, DK):
                                if left + j_s < seq_end_idx:
                                    q_shared[i_s % num_stages, j_s, j_k] = q[batch_idx, left + j_s, bhg, j_k]
                                else:
                                    q_shared[i_s % num_stages, j_s, j_k] = 0

                        T.barrier_arrive(
                            data_is_ready[i_s % num_stages]
                        )

                elif tx < 384 + 64:
                    # P1: TMA load dO and A
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(
                            data_is_free[i_s % num_stages],
                            (i_s // num_stages + 1) % 2,
                        )
                        chunk_idx = num_iters - 1 - i_s
                        left = seq_start_idx + chunk_idx * block_S
                        right = left + block_S

                        if right <= seq_end_idx:
                            T.tma_copy(
                                do[
                                    batch_idx,
                                    left:right,
                                    bh,
                                    value_start : value_start + block_V,
                                ],
                                do_shared[i_s % num_stages, :, :],
                                barrier=data_is_ready[i_s % num_stages],
                            )
                        else:
                            for j_s, j_v in T.Parallel(block_S, block_V):
                                if left + j_s < seq_end_idx:
                                    do_shared[i_s % num_stages, j_s, j_v] = do[
                                        batch_idx,
                                        left + j_s,
                                        bh,
                                        value_start + j_v,
                                    ]
                                else:
                                    do_shared[i_s % num_stages, j_s, j_v] = 0
                        if right <= seq_end_idx:
                            T.tma_copy(
                                a[batch_idx, left:right, bh, 0:block_S],
                                a_shared[i_s % num_stages, :, :],
                                barrier=data_is_ready[i_s % num_stages],
                            )
                        else:
                            for j_s, j_t in T.Parallel(block_S, block_S):
                                if left + j_s < seq_end_idx:
                                    a_shared[i_s % num_stages, j_s, j_t] = a[batch_idx, left + j_s, bh, j_t]
                                else:
                                    a_shared[i_s % num_stages, j_s, j_t] = 0

                        T.barrier_arrive(
                            data_is_ready[i_s % num_stages]
                        )

                elif tx < 384 + 96:
                    # P2: scalar load g and b (with boundary handling)
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(
                            data_is_free[i_s % num_stages],
                            (i_s // num_stages + 1) % 2,
                        )
                        chunk_idx = num_iters - 1 - i_s
                        left = seq_start_idx + chunk_idx * block_S
                        right = left + block_S

                        if right <= seq_end_idx:
                            for j_s in T.Parallel(block_S):
                                g_shared[i_s % num_stages, j_s] = g[
                                    batch_idx, left + j_s, bh
                                ]
                        else:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < seq_end_idx:
                                    g_shared[i_s % num_stages, j_s] = g[
                                        batch_idx, left + j_s, bh
                                    ]
                                else:
                                    g_shared[i_s % num_stages, j_s] = g[
                                        batch_idx, seq_end_idx - 1, bh
                                    ]

                        if right <= seq_end_idx:
                            for j_s in T.Parallel(block_S):
                                b_shared[i_s % num_stages, j_s] = b[
                                    batch_idx, left + j_s, bh
                                ]
                        else:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < seq_end_idx:
                                    b_shared[i_s % num_stages, j_s] = b[
                                        batch_idx, left + j_s, bh
                                    ]
                                else:
                                    b_shared[i_s % num_stages, j_s] = 0

                        T.barrier_arrive(
                            data_is_ready[i_s % num_stages]
                        )

                else:
                    # P3: store dh to global memory
                    for i_s in T.serial(num_iters):
                        T.barrier_arrive(bar_0)

                        T.barrier_wait(bar_0, i_s % 2)
                        T.barrier_wait(bar_3, i_s % 2)

                        if store_dh:
                            chunk_idx = num_iters - 1 - i_s
                            if state_v_first:
                                T.copy(
                                    dh_shared,
                                    dh[
                                        batch_idx,
                                        chunk_start_idx + chunk_idx,
                                        bh,
                                        value_start : value_start + block_V,
                                        0:DK,
                                    ],
                                )
                            else:
                                T.copy(
                                    dh_shared,
                                    dh[
                                        batch_idx,
                                        chunk_start_idx + chunk_idx,
                                        bh,
                                        0:DK,
                                        value_start : value_start + block_V,
                                    ],
                                )

    return tilelang_prepare_dh_kernel


def fused_gdr_dh_ws(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None = None,
    output_dh0: bool = True,
    output_dh: bool = True,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    num_warmup_chunks: torch.LongTensor | None = None,
    state_v_first: bool = False,
):
    batch_size, num_tokens, Hg, K = k.shape
    _, _, H, V = do.shape
    chunk_size = a.shape[-1]
    scale = scale or K ** (-0.5)
    assert K == V == 128
    assert chunk_size == 64

    if cu_seqlens is None:
        assert num_warmup_chunks is None
        real_batch_size = batch_size
        num_chunks = tilelang.cdiv(num_tokens, chunk_size) if output_dh else 0
        cu_seqlens = torch.empty(
            (batch_size + 1), dtype=torch.int32, device=k.device
        )
        chunk_offsets = torch.empty(
            (batch_size + 1), dtype=torch.int32, device=k.device
        )
        is_varlen = False
        is_cp = False
    else:
        real_batch_size = len(cu_seqlens) - 1
        chunk_offsets, num_chunks = prepare_chunk_offsets(
            cu_seqlens, chunk_size
        )
        chunk_offsets = chunk_offsets.to(cu_seqlens.dtype)
        num_chunks = num_chunks if output_dh else 0
        is_varlen = True
        if num_warmup_chunks is None:
            num_warmup_chunks = torch.empty(
                (real_batch_size, H),
                dtype=cu_seqlens.dtype,
                device=k.device,
            )
            is_cp = False
        else:
            is_cp = True

    use_dht = dht is not None
    if dht is None:
        dht = torch.empty(
            (real_batch_size, H, V, K)
            if state_v_first
            else (real_batch_size, H, K, V),
            dtype=torch.float32,
            device=k.device,
        )
    compact_dh = torch.empty(
        (1, 1, H, K, V),
        dtype=torch.float32,
        device=k.device,
    )
    anchor_q = torch.empty(
        (1, 1, H, K),
        dtype=q.dtype,
        device=q.device,
    )
    danchor_o = torch.empty(
        (1, 1, H, V),
        dtype=do.dtype,
        device=do.device,
    )
    anchor_cu_seqlens = torch.zeros_like(cu_seqlens)

    dh = torch.empty(
        (batch_size, num_chunks, H, V, K)
        if state_v_first
        else (batch_size, num_chunks, H, K, V),
        dtype=k.dtype,
        device=k.device,
    )
    dh0 = torch.empty(
        (real_batch_size, H, V, K)
        if state_v_first
        else (real_batch_size, H, K, V),
        dtype=torch.float32,
        device=k.device,
    )

    tilelang_prepare_dh_ws_kernel = tilelang_prepare_dh_ws(
        H,
        Hg,
        K,
        V,
        chunk_size,
        scale,
        qkva_dtype=k.dtype,
        g_dtype=g.dtype,
        b_dtype=b.dtype,
        dht_dtype=dht.dtype,
        dh_dtype=dh.dtype,
        o_dtype=do.dtype,
        seqlen_dtype=cu_seqlens.dtype,
        accum_dtype="float32",
        use_dht=use_dht,
        store_dh0=output_dh0,
        store_dh=output_dh,
        is_varlen=is_varlen,
        is_cp=is_cp,
        state_v_first=state_v_first,
    )
    tilelang_prepare_dh_ws_kernel(
        q,
        k,
        a,
        g,
        b,
        do,
        dht,
        compact_dh,
        anchor_q,
        danchor_o,
        compact_dh,
        cu_seqlens,
        chunk_offsets,
        anchor_cu_seqlens,
        num_warmup_chunks,
        dh,
        dh0,
    )

    if not output_dh0:
        dh0 = None
    if not output_dh:
        dh = None

    return dh, dh0


def fused_gdr_dh_ws_compact(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    do: torch.Tensor,
    compact_dh: torch.Tensor | None,
    anchor_cu_seqlens: torch.LongTensor,
    anchor_q: torch.Tensor | None = None,
    danchor_o: torch.Tensor | None = None,
    danchor_states: torch.Tensor | None = None,
    anchor_state_placeholder: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    compact_chunk_stride: int = 8,
    value_tile_size: int = 64,
    state_v_first: bool = False,
):
    """Prepare per-chunk state adjoints for the HSD split backward.

    The reverse dependency across chunks remains serial per packed
    document/head/value tile.  Unlike the monolithic FlashQLA backward this
    kernel only carries the boundary-state adjoint.  It injects one compact
    chunk-512 ``dH`` at the corresponding end boundary and emits the BF16
    end-state adjoint of every internal chunk64 for a later chunk-parallel
    local backward.
    """

    if cu_seqlens is None:
        raise ValueError("HSD compact state-adjoint scan requires packed cu_seqlens.")
    if state_v_first:
        raise ValueError("The first HSD split backward supports K-first states only.")
    batch_size, num_tokens, Hg, K = k.shape
    _, _, H, V = do.shape
    if batch_size != 1:
        raise ValueError("The first HSD split backward expects flattened packed B=1.")
    if H != Hg or K != 192 or V != 384:
        raise ValueError(
            "The HSD split state-adjoint scan is specialized for "
            f"H=Hg, K=192, V=384; received H={H}, Hg={Hg}, K={K}, V={V}."
        )
    if a.shape[-1] != 64:
        raise ValueError("The HSD split state-adjoint scan requires chunk_size=64.")
    if value_tile_size <= 0 or V % value_tile_size != 0:
        raise ValueError("value_tile_size must divide V exactly.")
    if compact_chunk_stride <= 0:
        raise ValueError("compact_chunk_stride must be positive.")
    if anchor_cu_seqlens.shape != cu_seqlens.shape:
        raise ValueError("anchor_cu_seqlens must match cu_seqlens.")
    if anchor_cu_seqlens.dtype != cu_seqlens.dtype:
        raise TypeError("anchor/text cu_seqlens must have the same dtype.")

    real_batch_size = len(cu_seqlens) - 1
    chunk_offsets, num_chunks = prepare_chunk_offsets(cu_seqlens, 64)
    chunk_offsets = chunk_offsets.to(cu_seqlens.dtype).contiguous()
    cu_seqlens = cu_seqlens.contiguous()
    anchor_cu_seqlens = anchor_cu_seqlens.contiguous()
    do = do.contiguous()
    scale = scale or K ** (-0.5)

    if anchor_q is not None:
        anchor_count = int(anchor_q.shape[1])
    elif compact_dh is not None:
        anchor_count = int(compact_dh.shape[1])
    else:
        # Legacy no-gradient callers do not expose a compact tensor from
        # which the dynamic extent can be inferred.
        anchor_count = int(anchor_cu_seqlens[-1].item())
    use_direct_anchor_output_grad = danchor_o is not None
    use_direct_anchor_state_grad = danchor_states is not None
    use_direct_anchor_grads = (
        use_direct_anchor_output_grad or use_direct_anchor_state_grad
    )
    if use_direct_anchor_grads and compact_dh is not None:
        raise ValueError(
            "Pass either materialized compact_dh or direct anchor gradients, not both."
        )
    if use_direct_anchor_grads and anchor_q is None:
        raise ValueError("Direct compact anchor-gradient injection requires anchor_q.")

    expected_compact_shape = (1, anchor_count, H, K, V)
    expected_anchor_q_shape = (1, anchor_count, H, K)
    expected_anchor_o_shape = (1, anchor_count, H, V)
    if use_direct_anchor_grads:
        if tuple(anchor_q.shape) != expected_anchor_q_shape:
            raise ValueError(
                "anchor_q shape mismatch: "
                f"expected {expected_anchor_q_shape}, got {tuple(anchor_q.shape)}."
            )
        if anchor_q.dtype != q.dtype:
            raise TypeError("Direct anchor_q must use the query dtype.")
        anchor_q = anchor_q.contiguous()
        if use_direct_anchor_output_grad:
            if tuple(danchor_o.shape) != expected_anchor_o_shape:
                raise ValueError(
                    "danchor_o shape mismatch: "
                    f"expected {expected_anchor_o_shape}, got {tuple(danchor_o.shape)}."
                )
            if danchor_o.dtype != do.dtype:
                raise TypeError("Direct danchor_o must use the text-output dtype.")
            danchor_o = danchor_o.contiguous()
        else:
            danchor_o = do[:, :anchor_count].contiguous()
        direct_state_tensor = (
            danchor_states
            if danchor_states is not None
            else anchor_state_placeholder
        )
        if direct_state_tensor is None:
            raise ValueError(
                "Direct compact injection requires an FP32 state-shaped placeholder "
                "when danchor_states is absent."
            )
        if tuple(direct_state_tensor.shape) != expected_compact_shape:
            raise ValueError(
                "direct anchor-state tensor shape mismatch: "
                f"expected {expected_compact_shape}, got {tuple(direct_state_tensor.shape)}."
            )
        if direct_state_tensor.dtype != torch.float32:
            raise TypeError("Direct anchor-state gradients must be FP32.")
        danchor_states = direct_state_tensor.contiguous()
        # The fixed TileLang signature still carries compact_dh.  Alias the
        # already-saved state-shaped tensor while its compile-time load path is
        # disabled, avoiding a second KxV buffer.
        compact_dh = danchor_states
    elif compact_dh is None:
        compact_dh = torch.zeros(
            expected_compact_shape,
            dtype=torch.float32,
            device=k.device,
        )
        anchor_q = q[:, :anchor_count].contiguous()
        danchor_o = do[:, :anchor_count].contiguous()
        danchor_states = compact_dh
    else:
        anchor_q = q[:, :anchor_count].contiguous()
        danchor_o = do[:, :anchor_count].contiguous()
        danchor_states = compact_dh
    if tuple(compact_dh.shape) != expected_compact_shape:
        raise ValueError(
            "compact_dh shape mismatch: "
            f"expected {expected_compact_shape}, got {tuple(compact_dh.shape)}."
        )
    compact_dh = compact_dh.to(dtype=torch.float32).contiguous()

    dht = torch.empty(
        (real_batch_size, H, K, V),
        dtype=torch.float32,
        device=k.device,
    )
    dh = torch.empty(
        (1, num_chunks, H, K, V),
        dtype=q.dtype,
        device=k.device,
    )
    # Packed HSD training always starts each document from the zero state and
    # does not expose an initial-state gradient.  Keep a correctly shaped
    # placeholder for the fixed TileLang signature, but do not make the
    # reverse scan publish its final ``dh0`` tile to HBM.
    dh0 = torch.empty(
        (real_batch_size, H, K, V),
        dtype=torch.float32,
        device=k.device,
    )
    num_warmup_chunks = torch.empty(
        (real_batch_size, H),
        dtype=cu_seqlens.dtype,
        device=k.device,
    )

    kernel = tilelang_prepare_dh_ws(
        H,
        Hg,
        K,
        V,
        64,
        scale,
        qkva_dtype=k.dtype,
        g_dtype=g.dtype,
        b_dtype=b.dtype,
        dht_dtype=dht.dtype,
        dh_dtype=dh.dtype,
        o_dtype=do.dtype,
        seqlen_dtype=cu_seqlens.dtype,
        accum_dtype="float32",
        use_dht=False,
        store_dh0=False,
        store_dh=True,
        is_varlen=True,
        is_cp=False,
        state_v_first=False,
        value_tile_size=value_tile_size,
        use_compact_dh=anchor_count > 0 and not use_direct_anchor_grads,
        use_direct_anchor_output_grad=use_direct_anchor_output_grad,
        use_direct_anchor_state_grad=use_direct_anchor_state_grad,
        compact_chunk_stride=compact_chunk_stride,
    )
    kernel(
        q,
        k,
        a,
        g,
        b,
        do,
        dht,
        compact_dh,
        anchor_q,
        danchor_o,
        danchor_states,
        cu_seqlens,
        chunk_offsets,
        anchor_cu_seqlens,
        num_warmup_chunks,
        dh,
        dh0,
    )
    return dh, None


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_prepare_dv_from_chunk_dh(
    H,
    DK,
    DV,
    chunk_size,
    value_tile_size,
    scale,
    accum_dtype,
    qk_dtype,
    g_dtype,
    dh_dtype,
    do_dtype,
    seqlen_dtype,
):
    """Build the transformed-value gradient from per-chunk end adjoints.

    Each CTA owns one ``chunk64 × head × V64`` tile and evaluates

    ``dV' = exp(g_last-g) * (K @ dH_end) + C.T @ dO``

    where ``C = scale * causal(exp(g_i-g_j) * QK.T)``.
    """

    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    num_docs = T.dynamic("num_docs")
    block_S = chunk_size
    block_V = value_tile_size
    num_value_tiles = (DV + block_V - 1) // block_V

    qk_shape = (1, num_tokens, H, DK)
    value_shape = (1, num_tokens, H, DV)
    g_shape = (1, num_tokens, H)
    dh_shape = (1, num_chunks, H, DK, DV)

    @T.prim_func
    def kernel(
        q: T.Tensor(qk_shape, dtype=qk_dtype),
        k: T.Tensor(qk_shape, dtype=qk_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        do: T.Tensor(value_shape, dtype=do_dtype),
        dh: T.Tensor(dh_shape, dtype=dh_dtype),
        cu_seqlens: T.Tensor([num_docs + 1], dtype=seqlen_dtype),
        chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
        dv: T.Tensor(value_shape, dtype=do_dtype),
    ):
        with T.Kernel(
            num_chunks * H * num_value_tiles,
            threads=128,
        ) as (bchv,):
            bc = bchv // (H * num_value_tiles)
            hv = bchv % (H * num_value_tiles)
            bh = hv // num_value_tiles
            bv = hv % num_value_tiles
            value_start = bv * block_V

            doc = chunk_indices[bc, 0]
            local_chunk = chunk_indices[bc, 1]
            seq_start = cu_seqlens[doc]
            seq_end = cu_seqlens[doc + 1]
            left = seq_start + local_chunk * block_S
            valid = T.min(block_S, seq_end - left)

            q_shared = T.alloc_shared((block_S, DK), dtype=qk_dtype)
            k_shared = T.alloc_shared((block_S, DK), dtype=qk_dtype)
            do_shared = T.alloc_shared((block_S, block_V), dtype=do_dtype)
            dh_shared = T.alloc_shared((DK, block_V), dtype=dh_dtype)
            p_shared = T.alloc_shared((block_S, block_S), dtype=qk_dtype)
            g_shared = T.alloc_shared(
                (block_S,),
                dtype=accum_dtype,
                scope="shared",
            )

            p_fragment = T.alloc_fragment(
                (block_S, block_S),
                dtype=accum_dtype,
            )
            dv_fragment = T.alloc_fragment(
                (block_S, block_V),
                dtype=accum_dtype,
            )

            T.annotate_layout(
                {
                    q_shared: tilelang.layout.make_swizzled_layout(q_shared),
                    k_shared: tilelang.layout.make_swizzled_layout(k_shared),
                    do_shared: tilelang.layout.make_swizzled_layout(do_shared),
                    dh_shared: tilelang.layout.make_swizzled_layout(dh_shared),
                    p_shared: tilelang.layout.make_swizzled_layout(p_shared),
                }
            )

            for i_s, i_k in T.Parallel(block_S, DK):
                if i_s < valid:
                    q_shared[i_s, i_k] = q[0, left + i_s, bh, i_k]
                    k_shared[i_s, i_k] = k[0, left + i_s, bh, i_k]
                else:
                    q_shared[i_s, i_k] = 0
                    k_shared[i_s, i_k] = 0
            for i_s, i_v in T.Parallel(block_S, block_V):
                if i_s < valid:
                    do_shared[i_s, i_v] = do[
                        0,
                        left + i_s,
                        bh,
                        value_start + i_v,
                    ]
                else:
                    do_shared[i_s, i_v] = 0
            for i_k, i_v in T.Parallel(DK, block_V):
                dh_shared[i_k, i_v] = dh[
                    0,
                    bc,
                    bh,
                    i_k,
                    value_start + i_v,
                ]
            for i_s in T.Parallel(block_S):
                if i_s < valid:
                    g_shared[i_s] = g[0, left + i_s, bh]
                else:
                    g_shared[i_s] = g[0, left + valid - 1, bh]

            # State-gradient contribution.
            T.gemm(
                k_shared,
                dh_shared,
                dv_fragment,
                clear_accum=True,
            )
            for i_s, i_v in T.Parallel(block_S, block_V):
                if i_s < valid:
                    dv_fragment[i_s, i_v] *= T.exp2(
                        (
                            g_shared[valid - 1]
                            - g_shared[i_s]
                        )
                        * 1.442695
                    )
                else:
                    dv_fragment[i_s, i_v] = 0

            # Local-output contribution C^T @ dO.
            T.gemm(
                q_shared,
                k_shared,
                p_fragment,
                transpose_B=True,
                clear_accum=True,
            )
            for i_s, i_t in T.Parallel(block_S, block_S):
                if i_s >= i_t and i_s < valid and i_t < valid:
                    p_fragment[i_s, i_t] *= (
                        scale
                        * T.exp2(
                            (
                                g_shared[i_s]
                                - g_shared[i_t]
                            )
                            * 1.442695
                        )
                    )
                else:
                    p_fragment[i_s, i_t] = 0
            T.copy(p_fragment, p_shared)
            T.gemm(
                p_shared,
                do_shared,
                dv_fragment,
                transpose_A=True,
                clear_accum=False,
            )

            for i_s, i_v in T.Parallel(block_S, block_V):
                if i_s < valid:
                    dv[
                        0,
                        left + i_s,
                        bh,
                        value_start + i_v,
                    ] = dv_fragment[i_s, i_v]

    return kernel


def fused_gdr_prepare_dv_from_chunk_dh(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    cu_seqlens: torch.LongTensor,
    scale: float | None = None,
    chunk_size: int = 64,
    value_tile_size: int = 64,
):
    """Python wrapper for the chunk-parallel transformed-value gradient."""

    batch_size, num_tokens, H, K = k.shape
    _, _, value_heads, V = do.shape
    if batch_size != 1 or H != value_heads:
        raise ValueError("HSD split dV requires packed B=1 and equal q/value heads.")
    if K != 192 or V != 384:
        raise ValueError(
            f"HSD split dV is specialized for K=192/V=384, got K={K}/V={V}."
        )
    if chunk_size != 64 or value_tile_size != 64:
        raise ValueError("HSD split dV currently requires chunk64 and Vtile64.")
    # ``prepare_chunk_indices`` supplies the packed document/local-chunk map
    # while preserving a single flattened token tensor.
    from march.kernels._flashqla.utils import prepare_chunk_indices

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size).contiguous()
    expected_dh_shape = (
        1,
        int(chunk_indices.shape[0]),
        H,
        K,
        V,
    )
    if tuple(dh.shape) != expected_dh_shape:
        raise ValueError(
            f"chunk dH shape mismatch: expected {expected_dh_shape}, "
            f"got {tuple(dh.shape)}."
        )
    dv = torch.empty_like(do)
    scale = scale or K ** (-0.5)
    kernel = tilelang_prepare_dv_from_chunk_dh(
        H,
        K,
        V,
        chunk_size,
        value_tile_size,
        scale,
        accum_dtype="float32",
        qk_dtype=q.dtype,
        g_dtype=g.dtype,
        dh_dtype=dh.dtype,
        do_dtype=do.dtype,
        seqlen_dtype=cu_seqlens.dtype,
    )
    kernel(
        q,
        k,
        g,
        do.contiguous(),
        dh.contiguous(),
        cu_seqlens.contiguous(),
        chunk_indices,
        dv,
    )
    return dv
